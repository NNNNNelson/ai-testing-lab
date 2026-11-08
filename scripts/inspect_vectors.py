"""Look inside the embedding space: what a cosine score does and does not mean.

    python scripts/inspect_vectors.py --id rag-01
    python scripts/inspect_vectors.py --threshold-sweep

Teaching tool. Two claims are demonstrated with measurements:

1. Cosine scores on one corpus occupy a narrow band, so an absolute threshold
   ("similarity > 0.5 is relevant") is an arbitrary cut, not a property of the data.
2. Cutting on that threshold removes correct pages; ranking is what carries the
   signal, which is why retrieval metrics are rank-based (hit@k), not threshold-based.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.corpus import hits, load_pages, supporting_pages  # noqa: E402
from evals.llm import get_provider  # noqa: E402
from targets.baseline_rag import BaselineRAG  # noqa: E402

CORPUS = ROOT / "data" / "corpus.txt"
GOLDEN = ROOT / "evals" / "golden_rag.yaml"


def _cos(provider, a: str, b: str) -> float:
    va, vb = provider.embed([a])[0], provider.embed([b])[0]
    return float(np.dot(va, vb))


def show_case(case_id: str) -> None:
    provider = get_provider()
    system = BaselineRAG(CORPUS, provider=provider, top_k=5)
    case = next(c for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8")) if c["id"] == case_id)
    pages = load_pages(CORPUS)
    support = supporting_pages(case, pages)

    q = provider.embed([case["question"]])[0]
    print(f"question vector : shape {q.shape}, ||q|| = {np.linalg.norm(q):.4f}")
    print(f"self dot product: q·q = {float(np.dot(q, q)):.4f}  (unit length -> dot == cosine)")

    matrix = system.matrix
    print(f"chunk matrix    : {matrix.shape}, every row unit length: "
          f"{bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4))}")

    scores = matrix @ q
    qs = np.percentile(scores, [0, 25, 50, 75, 90, 100])
    print(f"\nscore band over all {len(scores)} chunks:")
    print(f"  min {qs[0]:.3f} | p25 {qs[1]:.3f} | median {qs[2]:.3f} | p75 {qs[3]:.3f} "
          f"| p90 {qs[4]:.3f} | max {qs[5]:.3f}")

    order = np.argsort(-scores)[:10]
    print("\ntop-10 chunks (page, score, gold?):")
    for rank, i in enumerate(order, 1):
        ch = system.chunks[i]
        print(f"  {rank:>2}. page {ch.page:>3}  {scores[i]:.3f}  "
              f"{'gold' if ch.page in support else '-'}   {ch.text[:48].strip()}")

    top1, top5 = scores[order[0]], scores[order[4]]
    print(f"\ngap top1 - top5 = {top1 - top5:.3f}  (a 0.05 band separates 'best' from '5th')")

    print("\nsame question, three reference texts:")
    gold_sent = "profit before tax of $32.3bn, up $2.0bn or 6% compared with 2023"
    far_page = pages[[p for p in pages if p != "27"][40]]
    cn_paraphrase = "2024年报告的税前利润为323亿美元，较2023年增加20亿美元，增幅6%。"
    print(f"  vs English source sentence : {_cos(provider, case['question'], gold_sent):.3f}")
    print(f"  vs Chinese paraphrase      : {_cos(provider, case['question'], cn_paraphrase):.3f}")
    print(f"  vs a chunk from another page: {_cos(provider, case['question'], far_page[:400]):.3f}")


def threshold_sweep() -> None:
    provider = get_provider()
    system = BaselineRAG(CORPUS, provider=provider, top_k=5)
    pages = load_pages(CORPUS)
    cases = [c for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
             if c.get("category") in ("answerable", "multi_hop")]
    print(f"{len(cases)} answerable cases, page-level k=5\n")
    print(f"{'threshold':>9} {'hit@5':>7} {'avg pages kept':>15}  note")
    for t in (0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.58):
        hit_n, kept = 0, []
        for case in cases:
            dense = system._dense_scores(case["question"])
            best: dict[str, float] = {}
            for i in np.where(dense >= t)[0] if t > 0 else np.arange(len(dense)):
                ch = system.chunks[i]
                if ch.page not in best or dense[i] > best[ch.page]:
                    best[ch.page] = float(dense[i])
            top = sorted(best, key=lambda p: -best[p])[:5]
            kept.append(len(best))
            if hits(top, supporting_pages(case, pages)):
                hit_n += 1
        note = "no threshold (published config)" if t == 0 else ""
        print(f"{t:>9.2f} {hit_n/len(cases):>6.2%} {np.mean(kept):>15.1f}  {note}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", default=None)
    p.add_argument("--threshold-sweep", action="store_true")
    a = p.parse_args()
    if a.threshold_sweep:
        threshold_sweep()
    elif a.id:
        show_case(a.id)
    else:
        p.print_help()
