"""Retrieval quality: does the evidence reach the context window at all?

Metric choice matters: page-level recall against every page that mentions the
figure can look poor while the reader still gets a correct answer, because the
answer only needs ONE supporting page. So the primary assertion is evidence
hit@k; strict Recall@k is reported, not asserted.
"""

import pytest

from evals.corpus import hits, load_pages, supporting_pages


@pytest.mark.requires_real_model
def test_baseline_evidence_hit_at_5(corpus_path, provider, golden_rag):
    from targets.baseline_rag import BaselineRAG

    pages = load_pages(corpus_path)
    system = BaselineRAG(corpus_path, provider=provider, top_k=5)
    got_hits = []
    for case in golden_rag:
        if case["category"] == "unanswerable":
            continue
        retrieved = system.retrieve(case["question"], k=5)
        got_hits.append(1.0 if hits(retrieved, supporting_pages(case, pages)) else 0.0)
    avg = sum(got_hits) / len(got_hits)
    print(f"baseline evidence hit@5 = {avg:.2%} over {len(got_hits)} answerable cases")
    assert avg >= 0.80, f"evidence hit@5 too low: {avg:.2%}"


@pytest.mark.requires_real_model
def test_oss_evidence_hit_at_5(corpus_path, provider, golden_rag):
    from targets.baseline_rag import load_corpus, split_chunks
    from targets.oss_rag import OssRAG

    pages = load_pages(corpus_path)
    chunks = split_chunks(load_corpus(corpus_path), 512, 64)
    system = OssRAG([(c.page, c.text) for c in chunks], provider=provider, top_k=5)
    got_hits = []
    for case in golden_rag:
        if case["category"] == "unanswerable":
            continue
        retrieved = system.retrieve(case["question"], k=5)
        got_hits.append(1.0 if hits(retrieved, supporting_pages(case, pages)) else 0.0)
    avg = sum(got_hits) / len(got_hits)
    print(f"oss-rag evidence hit@5 = {avg:.2%}")
    assert avg >= 0.80, f"evidence hit@5 too low: {avg:.2%}"
