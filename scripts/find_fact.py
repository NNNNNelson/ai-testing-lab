"""Small helper: locate facts in the corpus and report the page they live on.

Used when authoring goldenset: every gold_source must be a real page, not a guess.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.txt"


def pages(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\[\[PAGE (\S+)\]\]", line.strip())
        if m:
            if current is not None:
                out[current] = "\n".join(buf)
            current, buf = m.group(1), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf)
    return out


def main(kw: str) -> None:
    data = pages(CORPUS.read_text(encoding="utf-8"))
    pat = re.compile(kw, re.I)
    hits = 0
    for page, body in data.items():
        flat = re.sub(r"\s+", " ", body)
        for m in pat.finditer(flat):
            start = max(0, m.start() - 60)
            print(f"[p{page}] ...{flat[start:m.end() + 120]}...")
            hits += 1
            if hits > 12:
                return
            break


if __name__ == "__main__":
    main(sys.argv[1])
