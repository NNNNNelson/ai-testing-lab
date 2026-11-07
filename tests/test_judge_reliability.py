"""The judge must itself be judged."""

import pytest

from evals.judge import judge_reliability, sample_for_human
from evals.metrics import cohen_kappa


def test_kappa_floor_is_enforced():
    # Perfect agreement
    assert cohen_kappa(["yes", "no"], ["yes", "no"]) == 1.0
    # Systematic disagreement -> judge verdicts would be rejected
    assert cohen_kappa(["yes", "yes"], ["no", "no"]) < 0.4


def test_sampling_is_reproducible():
    records = [{"id": str(i)} for i in range(50)]
    a = [r["id"] for r in sample_for_human(records, 0.2, seed=7)]
    b = [r["id"] for r in sample_for_human(records, 0.2, seed=7)]
    assert a == b
    assert len(a) == 10


@pytest.mark.requires_real_model
def test_judge_agrees_with_human_sample():
    """Fill reports/human_review.csv, then this asserts judge-vs-human kappa >= 0.6."""
    import json
    from pathlib import Path

    path = Path("reports/human_labelled.json")
    if not path.exists():
        pytest.skip("no human labels yet: export the sheet and label 20%")
    records = json.loads(path.read_text(encoding="utf-8"))
    stats = judge_reliability(records)
    print(f"judge-human kappa={stats['kappa']:.2f} on {stats['n']} rows")
    assert stats["kappa"] >= 0.60, "judge reliability too low - the report must not quote these verdicts"
