"""AI-specific quality: does it refuse when the document cannot answer?"""

import pytest

from evals.metrics import abstention_rate, hallucination_rate, numeric_leak_rate
from targets.baseline_rag import BaselineRAG


@pytest.mark.requires_real_model
def test_abstention_on_unanswerable(corpus_path, provider, golden_rag):
    """The single most important AI quality gate: no confident fabrication."""
    system = BaselineRAG(corpus_path, provider=provider, top_k=5)
    records = []
    for case in golden_rag:
        if case["category"] != "unanswerable":
            continue
        result = system.answer(case["question"])
        records.append({"category": "unanswerable", "refused": result["refused"],
                        "answer": result["text"], "id": case["id"]})
    rate = abstention_rate(records)
    print(f"abstention = {rate:.2%} on {len(records)} unanswerable cases")
    assert rate >= 0.80, f"hallucination risk: abstention only {rate:.2%}"
    assert hallucination_rate(records) <= 0.20


@pytest.mark.requires_real_model
def test_refusal_is_clean(corpus_path, provider, golden_rag):
    """A refusal that still volunteers a figure is a partial leak, not a clean refusal."""
    system = BaselineRAG(corpus_path, provider=provider, top_k=5)
    records = []
    for case in golden_rag:
        if case["category"] != "unanswerable":
            continue
        result = system.answer(case["question"])
        records.append({"category": "unanswerable", "refused": result["refused"],
                        "answer": result["text"], "id": case["id"]})
    leak = numeric_leak_rate(records)
    print(f"numeric leak in refusals = {leak:.2%}")
    assert leak <= 0.20, f"refusals carry figures: leak rate {leak:.2%}"


@pytest.mark.requires_real_model
def test_answers_are_grounded(corpus_path, provider, golden_rag):
    """Every answerable response must cite context; empty context means no grounding."""
    system = BaselineRAG(corpus_path, provider=provider, top_k=5)
    checked = 0
    for case in golden_rag:
        if case["category"] == "unanswerable":
            continue
        result = system.answer(case["question"])
        if result["refused"]:
            continue
        assert result["contexts"], f"{case['id']}: answered without any retrieved context"
        checked += 1
    assert checked > 0, "no answerable case produced an answer - check the corpus and model"


def test_consistency_smoke(corpus_path, provider, golden_rag):
    """Same question, three runs. With mock this only proves the plumbing."""
    from evals.metrics import consistency
    system = BaselineRAG(corpus_path, provider=provider, top_k=3)
    case = next(c for c in golden_rag if c["category"] == "answerable")
    answers = [system.answer(case["question"])["text"] for _ in range(3)]
    score = consistency(answers)
    print(f"consistency = {score:.2%}")
    assert score > 0
