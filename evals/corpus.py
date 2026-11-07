"""Corpus page parsing + gold-set expansion.

A hand-written gold page is a guess about where a fact lives. Annual reports
repeat the same figure on several pages, so a single-page label punishes correct
retrieval of a different page that carries the same evidence.

We therefore expand each case to "all pages that literally contain the answer":
the author supplies an `answer_pattern` (a regex), and the supporting page set
is computed from the corpus, not from memory. Hand labels are kept as the
minimum and the union is used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


def parse_pages(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current: str | None = None
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


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def supporting_pages(case: dict[str, Any], pages: dict[str, str]) -> list[str]:
    """Pages that contain the declared answer pattern, plus the hand-labelled ones."""
    gold = {str(g) for g in case.get("gold_sources", [])}
    pattern = case.get("answer_pattern")
    if pattern:
        rx = re.compile(pattern, re.I)
        for page, body in pages.items():
            if rx.search(_flat(body)):
                gold.add(page)
    return sorted(gold, key=lambda x: (len(x), x))


def hits(retrieved: Iterable[str], supporting: Iterable[str]) -> bool:
    return bool({str(r) for r in retrieved} & {str(s) for s in supporting})


def load_pages(corpus_path: str | Path) -> dict[str, str]:
    return parse_pages(Path(corpus_path).read_text(encoding="utf-8"))
