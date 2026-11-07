"""Self-built baseline RAG (the control group).

Deliberately simple and inspectable: fixed-size chunking, cosine retrieval,
one prompt with a hard abstention instruction. Nothing clever. Its only job is
to be the reference point that makes differences in the other system
attributable.

Three retrieval levers are exposed as *separate ablation variables*, because the
measured behaviour differs per language and per corpus:

    translate_query : translate the Chinese question into English before embedding
    hybrid          : mix dense cosine with pure-python BM25
    rerank          : lexical re-order of the top candidates

Interface shared by every target:
    .retrieve(question, k) -> list[page_ref]
    .answer(question) -> {"text", "refused", "contexts", "retrieved"}
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from evals import llm as llm_mod

SYSTEM_PROMPT = (
    "You are a document QA assistant. Answer ONLY from the provided CONTEXT. "
    "If the context does not contain the answer, reply with exactly I_DO_NOT_KNOW. "
    "Never use outside knowledge. Keep answers under 60 words."
)

REFUSAL = "I_DO_NOT_KNOW"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """English words/numbers plus single CJK characters: cheap, language-agnostic."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class Chunk:
    page: str
    text: str


def load_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split page-marked text into overlapping chunks, keeping the page ref."""
    chunks: list[Chunk] = []
    page = "0"
    buf: list[str] = []
    for raw in text.splitlines():
        m = re.match(r"\[\[PAGE (\S+)\]\]", raw.strip())
        if m:
            page = m.group(1)
            continue
        buf.append(raw)
        while len(" ".join(buf)) >= chunk_size:
            joined = " ".join(buf)
            chunks.append(Chunk(page=page, text=joined[:chunk_size]))
            keep = joined[chunk_size - overlap:]
            buf = [keep]
    tail = " ".join(buf).strip()
    if tail:
        chunks.append(Chunk(page=page, text=tail))
    return chunks


class BaselineRAG:
    name = "baseline-rag"

    def __init__(self, corpus_path: str | Path, provider: llm_mod.BaseProvider | None = None,
                 chunk_size: int = 512, overlap: int = 64, top_k: int = 5,
                 rerank: bool = False, translate_query: bool = False,
                 hybrid: bool = False, hybrid_alpha: float = 0.5,
                 page_level: bool = True):
        self.provider = provider or llm_mod.get_provider()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.rerank = rerank
        self.translate_query = translate_query
        self.hybrid = hybrid
        self.hybrid_alpha = hybrid_alpha
        self.page_level = page_level
        self.chunks = split_chunks(load_corpus(corpus_path), chunk_size, overlap)
        self._matrix: np.ndarray | None = None
        self._bm25 = None  # (docs_tokens, df, n)

    # ---------- dense ----------

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self.provider.embed([c.text for c in self.chunks])
        return self._matrix

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.provider.embed([query])[0]
        return self.matrix @ q

    # ---------- levers ----------

    def _query_for_embedding(self, question: str) -> str:
        """Query translation: the corpus is English, the questions are Chinese.

        Translating the *query* (not the corpus) is the cheap fix for the
        cross-lingual retrieval gap; whether it works is measured, not assumed.
        """
        if not self.translate_query:
            return question
        out = self.provider.chat(
            [{"role": "system", "content": "Translate the user's question into one English sentence. "
                                           "Reply with the translation only, no quotes, no explanation."},
             {"role": "user", "content": question}],
            max_tokens=96,
        )
        return out["text"].strip().strip('"') or question

    def _bm25_scores(self, query: str) -> np.ndarray:
        """Pure-python BM25. No dependency, inspectable, and a legitimate baseline."""
        if self._bm25 is None:
            docs = [_tokens(c.text) for c in self.chunks]
            df: dict[str, int] = {}
            for d in docs:
                for w in set(d):
                    df[w] = df.get(w, 0) + 1
            self._bm25 = (docs, df, len(docs))
        docs, df, n = self._bm25
        k1, b = 1.5, 0.75
        avg_len = sum(len(d) for d in docs) / max(1, len(docs))
        q_terms = set(_tokens(query))
        scores = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            tf: dict[str, int] = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q_terms:
                if w not in tf:
                    continue
                idf = math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg_len))
            scores[i] = s
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    # ---------- retrieval ----------

    def retrieve_pairs(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.top_k
        query = self._query_for_embedding(question)
        dense = self._dense_scores(query)
        if self.hybrid:
            lex = self._bm25_scores(query)
            scores = self.hybrid_alpha * dense + (1 - self.hybrid_alpha) * lex
        else:
            scores = dense
        order = np.argsort(-scores)[: max(k * 6, 30)]
        pairs = [(self.chunks[i], float(scores[i])) for i in order]
        if self.page_level:
            # The metric is page-level recall, so retrieval must be page-level too.
            # Otherwise 5 chunks can collapse onto one page and look like 1 hit out of 5.
            best: dict[str, tuple[Chunk, float]] = {}
            for chunk, score in pairs:
                if chunk.page not in best or score > best[chunk.page][1]:
                    best[chunk.page] = (chunk, score)
            pairs = sorted(best.values(), key=lambda p: -p[1])[:k]
            return pairs
        if self.rerank:
            q_tokens = set(_tokens(query))
            pairs.sort(key=lambda p: (len(q_tokens & set(_tokens(p[0].text))), p[1]), reverse=True)
        return pairs[:k]

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        return [p[0].page for p in self.retrieve_pairs(question, k)]

    def answer(self, question: str) -> dict:
        pairs = self.retrieve_pairs(question)
        context = "\n\n".join(f"[page {c.page}] {c.text}" for c, _ in pairs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ]
        result = self.provider.chat(messages)
        text = result["text"].strip()
        refused = result["refused"] or text.upper().startswith(REFUSAL)
        return {
            "text": text,
            "refused": refused,
            "contexts": [c.text for c, _ in pairs],
            "retrieved": [c.page for c, _ in pairs],
        }
