"""Probe which model ids this ARK account can actually call.

ARK does not document a model-list API, so we do two things:
1. try the OpenAI-compatible /models endpoint (may or may not exist)
2. try a list of known model ids with a 1-token request and report the status

Never prints the API key.
"""

from __future__ import annotations

import json
import os
import sys

import requests

BASE = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
KEY = os.environ.get("ARK_API_KEY", "")

CHAT_CANDIDATES = [
    "doubao-seed-1-6-250615",
    "doubao-seed-1-6-251015",
    "doubao-seed-1-6-flash-250615",
    "doubao-1-5-pro-32k-250115",
    "doubao-pro-32k",
    "doubao-lite-32k",
]
EMBED_CANDIDATES = [
    "doubao-embedding-large-text-240915",
    "doubao-embedding-text-240715",
    "doubao-embedding-large",
    "doubao-embedding",
]


def head(msg: str) -> None:
    print(f"\n== {msg} ==")


def try_models() -> None:
    head("GET /models")
    try:
        r = requests.get(f"{BASE}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=20)
        print("status:", r.status_code)
        print(r.text[:600])
    except Exception as exc:  # noqa: BLE001
        print("failed:", exc)


def try_chat(model: str) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"network: {exc}"
    if r.status_code == 200:
        return True, "ok"
    try:
        code = r.json()["error"]["code"]
    except Exception:  # noqa: BLE001
        code = r.text[:80]
    return False, f"{r.status_code} {code}"


def try_embed(model: str) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{BASE}/embeddings",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": model, "input": ["hello"]},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"network: {exc}"
    if r.status_code == 200:
        dim = len(r.json()["data"][0]["embedding"])
        return True, f"ok dim={dim}"
    try:
        code = r.json()["error"]["code"]
    except Exception:  # noqa: BLE001
        code = r.text[:80]
    return False, f"{r.status_code} {code}"


def main() -> int:
    if not KEY:
        print("ARK_API_KEY missing")
        return 1
    print(f"base={BASE} key_len={len(KEY)}")
    try_models()

    head("chat candidates")
    good_chat = []
    for m in CHAT_CANDIDATES:
        ok, detail = try_chat(m)
        print(f"  {'OK  ' if ok else 'FAIL'} {m}  {detail}")
        if ok:
            good_chat.append(m)

    head("embedding candidates")
    good_embed = []
    for m in EMBED_CANDIDATES:
        ok, detail = try_embed(m)
        print(f"  {'OK  ' if ok else 'FAIL'} {m}  {detail}")
        if ok:
            good_embed.append(m)

    head("summary")
    print("usable chat  :", good_chat or "none")
    print("usable embed :", good_embed or "none")
    if not good_chat:
        print("\nNo chat model reachable. Create an endpoint in the Ark console and use its ep-xxxx id.")
    return 0 if good_chat else 2


if __name__ == "__main__":
    sys.exit(main())
