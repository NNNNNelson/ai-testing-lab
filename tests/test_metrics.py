"""Metric unit tests: the scorer must be checkable without a model."""

import math

from evals.metrics import (
    abstention_rate, cohen_kappa, consistency, context_precision, determinism,
    hallucination_rate, paraphrase_pass_all, recall_at_k, routing_accuracy, safety_block_rate,
)


def test_recall_full_and_partial():
    assert recall_at_k(["12", "13"], ["12", "13"]) == 1.0
    assert recall_at_k(["12", "99"], ["12", "13"]) == 0.5
    assert math.isnan(recall_at_k(["1"], []))


def test_context_precision_penalises_noise():
    assert context_precision(["12", "13"], ["12", "13"]) == 1.0
    assert context_precision(["12", "99", "98"], ["12"]) == 1 / 3


def test_abstention_and_hallucination_are_complements():
    recs = [{"category": "unanswerable", "refused": True},
            {"category": "unanswerable", "refused": False},
            {"category": "answerable", "refused": False}]
    assert abstention_rate(recs) == 0.5
    assert hallucination_rate(recs) == 0.5


def test_routing_and_safety():
    recs = [{"expect_intent": "smoke", "intent": "smoke", "category": "clear"},
            {"expect_intent": "clarify", "intent": "smoke", "category": "ambiguous"}]
    assert routing_accuracy(recs) == 0.5
    assert safety_block_rate(recs) == 0.0


def test_paraphrase_requires_all_variants():
    recs = [
        {"group": "pay-smoke", "expect_intent": "smoke", "intent": "smoke"},
        {"group": "pay-smoke", "expect_intent": "smoke", "intent": "regression"},
    ]
    out = paraphrase_pass_all(recs)
    assert out["pay-smoke"]["passed"] is False
    assert out["pay-smoke"]["correct"] == 1


def test_consistency_and_determinism():
    assert consistency(["same answer", "same answer", "different"]) > 0.6
    assert determinism(["smoke", "smoke", "smoke"]) == 1.0
    assert determinism(["smoke", "regression"]) == 0.5


def test_kappa_behaves():
    assert cohen_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0
    # Perfect disagreement with zero chance agreement: kappa is 0 by convention
    # (the denominator is 1 - pe = 1, the numerator is 0 - pe = 0), the same value
    # scikit-learn returns. Asserting "< 0" would be asserting a bug.
    assert cohen_kappa(["yes", "yes"], ["no", "no"]) == 0.0
    # A rater that agrees with chance only: no skill, kappa <= 0.
    assert cohen_kappa(["yes", "no", "yes", "no"], ["yes", "yes", "no", "no"]) <= 0
