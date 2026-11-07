"""LLM-as-judge with a human-reviewed sample.

Two ideas that most evaluation repos skip:

1. The judge gets a rubric, not a vibe. Binary labels on explicitly defined
   criteria beat a 1-10 score nobody can audit.
2. The judge is itself evaluated. A sample is blind-labelled by a human (or by a
   second pass) and agreement is reported (Cohen's kappa). If agreement is
   low, the judge's verdicts do not go into the report.

Export the review sheet:
    python -m evals.judge --export reports/human_review.csv --sample 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from evals import llm as llm_mod
from evals.metrics import cohen_kappa

GROUNDING_RUBRIC = """You are grading an answer against retrieved context.

Label each of these independently:
- supported: every factual claim in the answer appears in the context. "yes" or "no".
- correct: the answer matches the reference answer. "yes" or "no".
- refused_ok: if the question is unanswerable, did it refuse? "yes", "no" or "n/a".

Reply with JSON only: {"supported": "...", "correct": "...", "refused_ok": "..."}
"""


def _parse_flags(text: str) -> dict[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"supported": "unknown", "correct": "unknown", "refused_ok": "unknown"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"supported": "unknown", "correct": "unknown", "refused_ok": "unknown"}
    return {k: str(v).lower() for k, v in data.items()}


def judge_record(provider: llm_mod.BaseProvider, record: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"QUESTION:\n{record['question']}\n\n"
        f"REFERENCE_ANSWER:\n{record.get('gold_answer') or '(unanswerable - the correct behaviour is to refuse)'}\n\n"
        f"CONTEXT:\n{chr(10).join(record.get('contexts', [])[:5])[:3000]}\n\n"
        f"ANSWER:\n{record.get('answer', '')}"
    )
    out = provider.chat(
        [{"role": "system", "content": GROUNDING_RUBRIC},
         {"role": "user", "content": prompt}],
        max_tokens=80,
    )
    return _parse_flags(out["text"])


def sample_for_human(records: Sequence[dict[str, Any]], ratio: float = 0.2,
                     seed: int = 42) -> list[dict[str, Any]]:
    k = max(1, int(len(records) * ratio))
    return random.Random(seed).sample(list(records), min(k, len(records)))


def export_review_sheet(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "question", "answer", "judge_supported",
                                                "judge_correct", "human_supported", "human_correct", "note"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r.get("id"),
                "question": r.get("question"),
                "answer": (r.get("answer") or "")[:300],
                "judge_supported": r.get("judge", {}).get("supported"),
                "judge_correct": r.get("judge", {}).get("correct"),
                "human_supported": "",
                "human_correct": "",
                "note": "",
            })


def judge_reliability(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Cohen's kappa between judge labels and human labels, on overlapping rows."""
    pairs = [(r["judge"]["supported"], r["human"]["supported"])
             for r in records
             if r.get("judge", {}).get("supported") not in (None, "unknown")
             and r.get("human", {}).get("supported") not in (None, "", "unknown")]
    if not pairs:
        return {"n": 0, "kappa": float("nan")}
    return {"n": len(pairs), "kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="reports/raw_rag.json")
    parser.add_argument("--export", default="reports/human_review.csv")
    parser.add_argument("--sample", type=float, default=0.2)
    args = parser.parse_args()
    data = json.loads(Path(args.records).read_text(encoding="utf-8"))
    subset = sample_for_human(data, args.sample)
    export_review_sheet(subset, args.export)
    print(f"exported {len(subset)} rows -> {args.export}")
