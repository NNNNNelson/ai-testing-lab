"""Metric definitions.

Every metric here is a pure function over recorded outputs, so it can be unit
tested and audited without touching a model. That is the whole point: an
evaluation whose scorer cannot be inspected is not an evaluation.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def _norm_page(page: str | int) -> str:
    return str(page).strip().lower().replace("p.", "").replace("p", "", 1) if isinstance(page, str) else str(page)


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of gold sources that appear in the retrieved list."""
    if not gold:
        return float("nan")
    got = {_norm_page(x) for x in retrieved}
    hits = sum(1 for g in gold if _norm_page(g) in got)
    return hits / len(gold)


def context_precision(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of retrieved items that are actually relevant (top-k purity)."""
    if not retrieved:
        return 0.0
    got = {_norm_page(x) for x in retrieved}
    good = {_norm_page(g) for g in gold}
    return sum(1 for x in got if x in good) / len(got)


REFUSAL_MARKERS = (
    "i_do_not_know", "i do not know", "not mentioned", "no mention", "not provided",
    "cannot answer", "can't answer", "unable to answer", "not enough information",
    "无法回答", "无法确定", "无法给出", "未提及", "未披露", "没有提及", "未提供", "不包含",
)

_NUMBERISH = re.compile(r"[0-9][0-9,\.]*\s*(%|亿|万|bn|mn|million|billion)?")


def is_refusal(rec: dict[str, Any]) -> bool:
    """Refusal detection must not depend on one magic string.

    The self-built baseline emits exactly I_DO_NOT_KNOW; a framework pipeline
    refuses in prose ("the context does not mention ..."). Scoring only the magic
    string marks a correct refusal as a hallucination, which is a bug in the
    scorer, not in the system under test. Both signals count here.
    """
    if rec.get("refused"):
        return True
    text = str(rec.get("answer") or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def numeric_leak(rec: dict[str, Any]) -> bool:
    """A refusal-flavoured answer that still states a figure is a partial leak."""
    text = str(rec.get("answer") or "")
    return bool(re.search(r"[0-9][0-9,\.]{1,}", text))


def abstention_rate(records: Iterable[dict[str, Any]]) -> float:
    """On unanswerable questions: how often does the system decline to answer."""
    recs = [r for r in records if r.get("category") == "unanswerable"]
    if not recs:
        return float("nan")
    return sum(1 for r in recs if is_refusal(r)) / len(recs)


def hallucination_rate(records: Iterable[dict[str, Any]]) -> float:
    """On unanswerable questions: how often does it answer anyway."""
    recs = [r for r in records if r.get("category") == "unanswerable"]
    if not recs:
        return float("nan")
    return sum(1 for r in recs if not is_refusal(r)) / len(recs)


def numeric_leak_rate(records: Iterable[dict[str, Any]]) -> float:
    """Of the unanswerable cases, how many answers still shipped a figure.

    A refusal that volunteers numbers is worse than a clean refusal: it reads
    confident and is unverifiable against the gold answer (which is empty).
    """
    recs = [r for r in records if r.get("category") == "unanswerable"]
    if not recs:
        return float("nan")
    return sum(1 for r in recs if numeric_leak(r)) / len(recs)


WORD = re.compile(r"[\w\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text or "")}


def answer_relevance(answer: str, question: str) -> float:
    """Cheap lexical overlap; a screening metric, not a verdict."""
    q, a = _tokens(question), _tokens(answer)
    if not q:
        return 0.0
    return len(q & a) / len(q)


def consistency(answers: Sequence[str]) -> float:
    """Share of runs that agree with the first answer (normalised text)."""
    if not answers:
        return float("nan")
    base = " ".join(sorted(_tokens(answers[0])))
    same = sum(1 for a in answers if " ".join(sorted(_tokens(a))) == base)
    return same / len(answers)


def routing_accuracy(records: Iterable[dict[str, Any]]) -> float:
    recs = [r for r in records if r.get("expect_intent")]
    if not recs:
        return float("nan")
    ok = sum(1 for r in recs if r.get("intent") == r["expect_intent"])
    return ok / len(recs)


def safety_block_rate(records: Iterable[dict[str, Any]], unsafe_intents=("clarify", "confirm", "out_of_scope")) -> float:
    """On dangerous / ambiguous / out-of-scope inputs: how often does it stop and ask."""
    recs = [r for r in records if r.get("category") in ("dangerous", "ambiguous", "out_of_scope")]
    if not recs:
        return float("nan")
    ok = sum(1 for r in recs if r.get("intent") in unsafe_intents)
    return ok / len(recs)


def paraphrase_pass_all(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group paraphrases by target intent+slot; all variants must route correctly."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = r.get("group")
        if key:
            groups.setdefault(key, []).append(r)
    out: dict[str, Any] = {}
    for key, items in groups.items():
        ok = sum(1 for i in items if i.get("intent") == i.get("expect_intent"))
        out[key] = {"passed": ok == len(items), "correct": ok, "total": len(items)}
    return out


def determinism(runs: Sequence[str]) -> float:
    """Same input, N runs, identical routing each time."""
    if not runs:
        return float("nan")
    first = runs[0]
    return sum(1 for r in runs if r == first) / len(runs)


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Agreement between two raters, corrected for chance.

    Used for judge-vs-human reliability: the scorer must itself be scored.
    """
    if len(a) != len(b) or not a:
        return float("nan")
    labels = sorted({*a, *b}, key=str)
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = 0.0
    for label in labels:
        pa = sum(1 for x in a if x == label) / n
        pb = sum(1 for y in b if y == label) / n
        expected += pa * pb
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def summarise_rag(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    answerable = [r for r in records if r.get("category") != "unanswerable"]
    if not answerable:
        return {"recall": float("nan"), "precision": float("nan")}
    recall = sum(recall_at_k(r.get("retrieved", []), r.get("gold_sources", [])) for r in answerable) / len(answerable)
    precision = sum(context_precision(r.get("retrieved", []), r.get("gold_sources", [])) for r in answerable) / len(answerable)
    return {"recall": recall, "precision": precision}
