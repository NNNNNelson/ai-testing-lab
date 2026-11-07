"""Model provider abstraction.

Three providers, one interface. The harness must never depend on one vendor,
otherwise the evaluation becomes unfalsifiable and unreproducible.

    ARK   : Volcengine Ark, OpenAI-compatible HTTP API (real scores)
    MOCK  : deterministic fake model (pipeline verification only, NOT real scores)
    NONE  : no model available -> callers must skip, never fabricate

Verified against this account (2026-11-07):
    chat : doubao-seed-2-1-pro-260628      (OpenAI-compatible /chat/completions)
    embed: doubao-embedding-vision-251215  (multimodal /embeddings/multimodal, 2048-dim,
                                            ONE text per request -> batched in threads)

Env vars:
    ARK_API_KEY       (required)
    ARK_BASE_URL      default https://ark.cn-beijing.volces.com/api/v3
    ARK_CHAT_MODEL    default doubao-seed-2-1-pro-260628
    ARK_EMBED_MODEL   default doubao-embedding-vision-251215
    EMBED_CONCURRENCY default 6
    EVAL_CACHE        default data/.cache   (embedding cache; makes reruns cheap)

Probe:
    python -m evals.llm --probe
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_CHAT = "doubao-seed-2-1-pro-260628"
DEFAULT_EMBED = "doubao-embedding-vision-251215"
EMBED_DIM = 256  # mock only

ROOT = Path(__file__).resolve().parent.parent


class ProviderUnavailable(RuntimeError):
    """No real model configured. Never swallow this."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class BaseProvider:
    name = "base"
    is_real = False

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 512) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class MockProvider(BaseProvider):
    """Deterministic offline stand-in. Any score from it is meaningless as evidence."""

    name = "mock"
    is_real = False

    def chat(self, messages, temperature=0.0, max_tokens=512):
        user = messages[-1]["content"] if messages else ""
        digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:8]
        has_context = "CONTEXT" in user.upper()
        if "I_DO_NOT_KNOW" in user and not has_context:
            text, refused = "I_DO_NOT_KNOW", True
        else:
            text = f"MOCK[{digest}] " + ("context-grounded answer" if has_context else "answer")
            refused = False
        return {"text": text, "refused": refused,
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
                "raw": {"mock": True}}

    def embed(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for j in range(EMBED_DIM):
                h = hashlib.sha256(f"{j}:{t}".encode("utf-8")).digest()
                out[i, j] = (h[0] / 255.0) - 0.5
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class ArkProvider(BaseProvider):
    name = "ark"
    is_real = True

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 chat_model: str | None = None, embed_model: str | None = None,
                 timeout: int = 180, max_retries: int = 3, cache_dir: str | Path | None = None):
        self.api_key = api_key or os.environ.get("ARK_API_KEY", "")
        if not self.api_key:
            raise ProviderUnavailable("ARK_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("ARK_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.chat_model = chat_model or os.environ.get("ARK_CHAT_MODEL") or DEFAULT_CHAT
        self.embed_model = embed_model or os.environ.get("ARK_EMBED_MODEL") or DEFAULT_EMBED
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self.concurrency = int(os.environ.get("EMBED_CONCURRENCY", "6"))
        cache = cache_dir or os.environ.get("EVAL_CACHE") or (ROOT / "data" / ".cache")
        self.cache_dir = Path(cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- transport ----------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"retryable {resp.status_code}: {resp.text[:150]}")
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as exc:  # noqa: BLE001 - retried below
                last_err = exc
                time.sleep(1.5 * (2 ** attempt))
        raise RuntimeError(f"ARK request failed after {self.max_retries} attempts: {last_err}")

    # ---------- chat ----------

    def chat(self, messages, temperature=0.0, max_tokens=512):
        payload = {"model": self.chat_model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        # Thinking chains on the pro tier are slow and non-deterministic; for
        # evaluation we disable them unless explicitly asked. This is a stated
        # experimental condition, not a hidden trick.
        thinking = os.environ.get("ARK_THINKING", "disabled")
        if thinking in ("enabled", "disabled"):
            payload["thinking"] = {"type": thinking}
        data = self._post("chat/completions", payload)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.usage.add(Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)))
        return {"text": text,
                "refused": text.strip().upper().startswith("I_DO_NOT_KNOW"),
                "usage": usage, "raw": data}

    # ---------- embeddings ----------

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.embed_model}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _embed_one(self, text: str) -> np.ndarray:
        path = self._cache_path(text)
        if path.exists():
            return np.load(path)
        if "vision" in self.embed_model:
            # multimodal endpoint: exactly one input item per request
            data = self._post("embeddings/multimodal",
                              {"model": self.embed_model,
                               "input": [{"type": "text", "text": text}]})
            vec = np.asarray(data["data"]["embedding"], dtype=np.float32)
        else:
            data = self._post("embeddings", {"model": self.embed_model, "input": [text]})
            vec = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
        np.save(path, vec)
        return vec

    def embed(self, texts):
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            vectors = list(pool.map(self._embed_one, list(texts)))
        arr = np.vstack(vectors)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_provider(prefer: str | None = None, allow_mock: bool = True) -> BaseProvider:
    prefer = (prefer or os.environ.get("EVAL_PROVIDER") or "").lower()
    if prefer == "mock":
        return MockProvider()
    try:
        return ArkProvider()
    except ProviderUnavailable:
        if allow_mock:
            return MockProvider()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe model availability")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        try:
            p = ArkProvider()
        except ProviderUnavailable as exc:
            print(f"[FAIL] {exc}")
            raise SystemExit(1)
        print(f"[cfg] base={p.base_url}")
        print(f"[cfg] chat={p.chat_model} embed={p.embed_model}")
        r = p.chat([{"role": "user", "content": "reply with the single word: pong"}], max_tokens=8)
        print("[chat] ->", r["text"].strip()[:60], "| usage", r["usage"])
        v = p.embed(["hello world", "你好，世界"])
        print("[embed] ->", v.shape, "normalised:", bool(np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)))
        print("[ok] both endpoints are reachable")
