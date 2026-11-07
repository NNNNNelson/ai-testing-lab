"""Trace one golden case end to end, printing every intermediate state.

This exists so the pipeline stops being abstract: you can see the chunk that
matched, its score, the context the model actually received, the answer it
returned, and how that answer was scored.

    python scripts/trace_one_case.py --id rag-01
    python scripts/trace_one_case.py --id agt-17 --agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.corpus import load_pages, supporting_pages  # noqa: E402
from evals.llm import get_provider  # noqa: E402
from evals.metrics import context_precision, is_refusal, recall_at_k  # noqa: E402
from targets.baseline_rag import BaselineRAG  # noqa: E402

CORPUS = ROOT / "data" / "corpus.txt"
GOLDEN_RAG = ROOT / "evals" / "golden_rag.yaml"
GOLDEN_AGENT = ROOT / "evals" / "golden_agent.yaml"


def _load(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{kind} golden set missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_case(case_id: str, kind: str) -> dict:
    """Find the case, or explain the mistake instead of raising StopIteration."""
    own = _load(GOLDEN_RAG, "RAG") if kind == "rag" else _load(GOLDEN_AGENT, "agent")
    hit = next((c for c in own if c.get("id") == case_id), None)
    if hit:
        return hit

    other_kind = "agent" if kind == "rag" else "rag"
    other = _load(GOLDEN_AGENT, "agent") if other_kind == "agent" else _load(GOLDEN_RAG, "RAG")
    if any(c.get("id") == case_id for c in other):
        flag = "adding `--agent`" if other_kind == "agent" else "dropping `--agent`"
        raise SystemExit(
            f"[FAIL] '{case_id}' is a {other_kind} case, not a {kind} case.\n"
            f"       Re-run {flag}:  python scripts/trace_one_case.py --id {case_id}"
            + (" --agent" if other_kind == "agent" else "")
        )

    by_cat: dict[str, list[str]] = {}
    for c in own:
        by_cat.setdefault(str(c.get("category", "?")), []).append(str(c.get("id")))
    ids = "; ".join(f"{cat}: {', '.join(v)}" for cat, v in by_cat.items())
    raise SystemExit(f"[FAIL] no {kind} case with id '{case_id}'.\n       available -> {ids}")


def trace_rag(case_id: str) -> None:
    case = resolve_case(case_id, "rag")
    provider = get_provider()
    system = BaselineRAG(CORPUS, provider=provider, top_k=5)
    pages = load_pages(CORPUS)

    print(f"provider      : {provider.name} (real={provider.is_real})")
    print(f"corpus        : {len(pages)} pages, split into {len(system.chunks)} chunks "
          f"(chunk_size={system.chunk_size}, overlap={system.overlap})")
    print(f"\nquestion      : {case['question']}")
    print(f"hand gold     : pages {case.get('gold_sources')}")
    print(f"computed gold : pages {supporting_pages(case, pages)}  (regex {case.get('answer_pattern')!r})")

    query = system._query_for_embedding(case["question"])
    dense = system._dense_scores(query)
    print(f"\nquery vector  : {provider.embed([query])[0].shape} dimensions, L2-normalised")
    print(f"similarity    : cosine = dot product over {len(dense)} chunks "
          f"(range {dense.min():.3f} .. {dense.max():.3f})")

    pairs = system.retrieve_pairs(case["question"], k=5)
    print("\ntop-5 pages (page-level aggregation: best chunk on each page):")
    for rank, (chunk, score) in enumerate(pairs, 1):
        mark = " <== gold" if chunk.page in supporting_pages(case, pages) else ""
        print(f"  {rank}. page {chunk.page:>3}  score {score:.3f}{mark}")
        print(f"      {chunk.text[:90].replace(chr(10), ' ')} ...")

    result = system.answer(case["question"])
    print(f"\nanswer        : {result['text'][:200]}")
    print(f"refused       : {result['refused']}  (scorer also reads prose: {is_refusal({'answer': result['text']})})")

    retrieved = result["retrieved"]
    support = supporting_pages(case, pages)
    from evals.corpus import hits
    print("\nscoring this case:")
    print(f"  evidence hit@5      = {1.0 if hits(retrieved, support) else 0.0}")
    print(f"  recall@5 (coverage) = {recall_at_k(retrieved, support):.2%}  "
          f"(gold pages {len(support)}, got {len(set(retrieved) & set(support))})")
    print(f"  context precision   = {context_precision(retrieved, support):.2%}")


def trace_agent(case_id: str) -> None:
    from agent.baseline_router import IntentRouter, RuleRouter

    case = resolve_case(case_id, "agent")
    print(f"input    : {case['input']}")
    print(f"expected : {case['expect_intent']}   category={case['category']}")
    for cls in (RuleRouter, IntentRouter):
        router = cls()
        out = router.route(case["input"])
        print(f"\n{cls.__name__} ({router.name})")
        print(f"  intent : {out['intent']}")
        print(f"  target : {out.get('target')}")
        print(f"  reason : {out.get('reason')}")
        safe = out["intent"] in ("clarify", "confirm", "out_of_scope")
        print(f"  executed? {not safe}  (dangerous/ambiguous must NOT auto-execute)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--agent", action="store_true")
    args = parser.parse_args()
    if args.agent:
        trace_agent(args.id)
    else:
        trace_rag(args.id)
