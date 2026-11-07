"""Look inside the chunker: what a chunk actually is, and where it breaks.

    python scripts/inspect_chunks.py --stats
    python scripts/inspect_chunks.py --page 27
    python scripts/inspect_chunks.py --sizes
    python scripts/inspect_chunks.py --phrase "$32.3bn"

Teaching tool, not part of the harness. It exposes the three consequences of
fixed-size chunking: hard cuts inside words, overlap windows, and page labels
that belong to the last page marker seen rather than to the chunk's content.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from targets.baseline_rag import load_corpus, split_chunks  # noqa: E402

CORPUS = ROOT / "data" / "corpus.txt"


def page_map(text: str) -> dict[str, str]:
    pages: dict[str, list[str]] = {}
    current = "0"
    for line in text.splitlines():
        m = re.match(r"\[\[PAGE (\S+)\]\]", line.strip())
        if m:
            current = m.group(1)
            pages.setdefault(current, [])
            continue
        pages.setdefault(current, []).append(line)
    return {k: " ".join(v) for k, v in pages.items()}


def starts_midword(chunk_text: str) -> bool:
    head = chunk_text.lstrip()
    return bool(head) and head[0].isascii() and head[0].isalpha() and head[0].islower()


def stats(chunk_size: int, overlap: int) -> None:
    text = load_corpus(CORPUS)
    chunks = split_chunks(text, chunk_size, overlap)
    pages = {c.page for c in chunks}
    lens = sorted(len(c.text) for c in chunks)
    mid = sum(1 for c in chunks if starts_midword(c.text))
    pmap = page_map(text)
    crossed = 0
    for c in chunks:
        probe = c.text.lstrip()[:40]
        if c.page in pmap and probe and probe not in pmap[c.page]:
            crossed += 1
    print(f"chunk_size={chunk_size} overlap={overlap}")
    print(f"  pages={len(pages)}  chunks={len(chunks)}  avg per page={len(chunks)/max(1,len(pages)):.1f}")
    print(f"  chunk length min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    print(f"  chunks starting mid-word (hard cut through a word): {mid} "
          f"({mid/max(1,len(chunks)):.1%})")
    print(f"  chunks whose opening text is NOT on their labelled page "
          f"(carried over from the previous page): {crossed} ({crossed/max(1,len(chunks)):.1%})")
    print(f"  vector memory: {len(chunks)} x 2048 float32 = {len(chunks)*2048*4/1e6:.1f} MB")


def show_page(number: str, chunk_size: int, overlap: int) -> None:
    chunks = split_chunks(load_corpus(CORPUS), chunk_size, overlap)
    hits = [c for c in chunks if c.page == number]
    print(f"page {number}: {len(hits)} chunk(s) carry this page label\n")
    for i, c in enumerate(hits):
        flag = "  <== starts mid-word" if starts_midword(c.text) else ""
        print(f"[{i}] len={len(c.text)}{flag}")
        print(f"    head: {c.text[:70].strip()}")
        print(f"    tail: {c.text[-70:].strip()}")
        print()


def show_phrase(phrase: str, chunk_size: int, overlap: int) -> None:
    chunks = split_chunks(load_corpus(CORPUS), chunk_size, overlap)
    found = [(c, c.text.find(phrase)) for c in chunks if phrase in c.text]
    print(f"phrase {phrase!r} appears in {len(found)} chunk(s)\n")
    for c, pos in found[:6]:
        where = "start" if pos < 64 else ("end" if pos > len(c.text) - 64 - len(phrase) else "middle")
        print(f"  page {c.page:>3}  offset {pos:>3}/{len(c.text)}  ({where} of the chunk)")
        print(f"     ...{c.text[max(0,pos-60):pos+len(phrase)+60].strip()}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="store_true")
    p.add_argument("--page", default=None)
    p.add_argument("--sizes", action="store_true")
    p.add_argument("--phrase", default=None)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    a = p.parse_args()
    if a.stats:
        stats(a.chunk_size, a.overlap)
    elif a.sizes:
        for cs in (256, 512, 1024):
            stats(cs, max(16, cs // 8))
            print()
    elif a.page:
        show_page(a.page, a.chunk_size, a.overlap)
    elif a.phrase:
        show_phrase(a.phrase, a.chunk_size, a.overlap)
    else:
        p.print_help()
