"""Diagnose retrieval: where does the gold page actually rank?

Tells us whether the problem is the embedding (cross-lingual gap), the chunking,
or the gold page labelling. Run before changing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.llm import get_provider  # noqa: E402
from targets.baseline_rag import split_chunks, load_corpus  # noqa: E402

CORPUS = ROOT / "data" / "corpus.txt"
GOLDEN = ROOT / "evals" / "golden_rag.yaml"


def main() -> None:
    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    provider = get_provider()
    chunks = split_chunks(load_corpus(CORPUS), 512, 64)
    matrix = provider.embed([c.text for c in chunks])
    pages = [c.page for c in chunks]

    rows = []
    for case in cases:
        if case["category"] == "unanswerable":
            continue
        q = provider.embed([case["question"]])[0]
        scores = matrix @ q
        order = np.argsort(-scores)
        gold = {str(g) for g in case["gold_sources"]}
        rank = next((i + 1 for i, idx in enumerate(order) if pages[idx] in gold), None)
        best_page = pages[order[0]]
        rows.append((case["id"], rank, best_page, sorted(gold), round(float(scores[order[0]]), 3)))

    print(f"{'id':8} {'gold_rank':>9} {'top1_page':>10} {'gold':>10} {'top1_score':>10}")
    for r in rows:
        print(f"{r[0]:8} {str(r[1]):>9} {r[2]:>10} {str(r[3]):>10} {r[4]:>10}")

    ranks = [r[1] for r in rows if r[1] is not None]
    print(f"\nanswerable cases: {len(rows)}, gold found: {len(ranks)}")
    if ranks:
        print(f"median rank of gold = {sorted(ranks)[len(ranks)//2]}, ranks={ranks}")
    miss = [r[0] for r in rows if r[1] is None]
    print("gold page never retrieved inside k=10:", miss)


if __name__ == "__main__":
    main()
