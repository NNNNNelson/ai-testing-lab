"""Run the full evaluation and write inspectable reports.

Usage:
    python -m evals.run_eval --provider mock            # pipeline smoke run
    python -m evals.run_eval --provider ark --systems all
    python -m evals.run_eval --provider ark --ablation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from evals import judge as judge_mod
from evals import llm as llm_mod
from evals import metrics
from evals import corpus as corpus_mod
from targets.baseline_rag import BaselineRAG, split_chunks, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "corpus.txt"
GOLDEN_RAG = ROOT / "evals" / "golden_rag.yaml"
GOLDEN_AGENT = ROOT / "evals" / "golden_agent.yaml"
REPORTS = ROOT / "reports"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_systems(provider, k: int, systems: str):
    out = {}
    chunks = split_chunks(load_corpus(DATA), 512, 64)
    if systems in ("all", "baseline"):
        out["baseline-rag"] = BaselineRAG(DATA, provider=provider, top_k=k)
    if systems in ("all", "oss"):
        try:
            from targets.oss_rag import OssRAG
            out["oss-rag"] = OssRAG([(c.page, c.text) for c in chunks], provider=provider, top_k=k)
        except Exception as exc:  # noqa: BLE001 - report, never fake
            print(f"[warn] oss-rag unavailable: {exc}")
    return out


def run_rag(system, cases, provider, do_judge: bool) -> list[dict]:
    pages = corpus_mod.load_pages(DATA)
    records = []
    for case in cases:
        result = system.answer(case["question"])
        supporting = corpus_mod.supporting_pages(case, pages)
        rec = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result["text"],
            "refused": result["refused"],
            "retrieved": result.get("retrieved", []),
            "gold_sources": case.get("gold_sources", []),
            "supporting": supporting,
            "hit": corpus_mod.hits(result.get("retrieved", []), supporting),
            "contexts": result.get("contexts", []),
        }
        if do_judge:
            rec["judge"] = judge_mod.judge_record(provider, rec)
        records.append(rec)
    return records


def evaluate_rag(provider, k: int, systems: str, do_judge: bool, tag: str):
    cases = _load_yaml(GOLDEN_RAG)
    built = build_systems(provider, k, systems)
    out = {}
    for name, system in built.items():
        print(f"[run] {name} on {len(cases)} cases")
        records = run_rag(system, cases, provider, do_judge)
        answerable = [r for r in records if r["category"] != "unanswerable"]
        summary = {
            "hit_at_k": sum(1 for r in answerable if r["hit"]) / max(1, len(answerable)),
            "recall": sum(metrics.recall_at_k(r["retrieved"], r["supporting"]) for r in answerable) / max(1, len(answerable)),
            "precision": sum(metrics.context_precision(r["retrieved"], r["supporting"]) for r in answerable) / max(1, len(answerable)),
            "abstention": metrics.abstention_rate(records),
            "hallucination": metrics.hallucination_rate(records),
            "numeric_leak": metrics.numeric_leak_rate(records),
            "cases": len(records),
        }
        out[name] = {"summary": summary, "records": records}
        print(f"  hit@{k}={summary['hit_at_k']:.2%} recall={summary['recall']:.2%} "
              f"abstain={summary['abstention']:.2%}")
    (REPORTS / f"raw_rag_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def evaluate_agent(provider, tag: str):
    cases = _load_yaml(GOLDEN_AGENT)
    from agent.baseline_router import IntentRouter, RuleRouter

    out = {}
    for label, router in {"rule-router": RuleRouter(provider), "intent-router": IntentRouter(provider)}.items():
        res = []
        for case in cases:
            r = router.route(case["input"])
            res.append({"id": case["id"], "category": case["category"], "group": case.get("group"),
                        "input": case["input"], "expect_intent": case["expect_intent"],
                        "intent": r["intent"], "target": r.get("target"), "reason": r.get("reason")})
        summary = {
            "routing_accuracy": metrics.routing_accuracy(res),
            "safety_block_rate": metrics.safety_block_rate(res),
            "paraphrase": metrics.paraphrase_pass_all(res),
            "cases": len(res),
        }
        out[label] = {"summary": summary, "records": res}
        print(f"[agent] {label}: acc={summary['routing_accuracy']:.2%} safety={summary['safety_block_rate']:.2%}")
    (REPORTS / f"raw_agent_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def write_results(rag: dict, agent: dict, tag: str, provider_name: str):
    REPORTS.mkdir(exist_ok=True)
    lines = [f"# Results ({tag})", "", f"provider: `{provider_name}`", ""]
    if rag:
        lines += ["## RAG", "",
                  "| metric | " + " | ".join(rag.keys()) + " | threshold |",
                  "|---|" + "---|" * len(rag) + "---|"]
        rows = [("hit_at_k", "≥0.90"), ("recall", "observe"), ("precision", "observe"),
                ("abstention", "≥0.80"), ("hallucination", "≤0.20"), ("numeric_leak", "≤0.20")]
        for key, threshold in rows:
            cells = [f"{v['summary'].get(key, float('nan')):.2%}" if v['summary'].get(key) == v['summary'].get(key)
                     else "n/a" for v in rag.values()]
            lines.append(f"| {key} | " + " | ".join(cells) + f" | {threshold} |")
    if agent:
        lines += ["", "## Agent", "",
                  "| metric | " + " | ".join(agent.keys()) + " |",
                  "|---|" + "---|" * len(agent),
                  "| routing_accuracy | " + " | ".join(
                      f"{v['summary']['routing_accuracy']:.2%}" for v in agent.values()) + " |",
                  "| safety_block_rate | " + " | ".join(
                      f"{v['summary']['safety_block_rate']:.2%}" for v in agent.values()) + " |"]
    (REPORTS / f"results_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] reports/results_{tag}.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None, help="ark | mock")
    parser.add_argument("--systems", default="all", help="all | baseline | oss")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"corpus missing: {DATA}\nrun: python scripts/build_corpus.py")
    REPORTS.mkdir(parents=True, exist_ok=True)
    provider = llm_mod.get_provider(args.provider)
    tag = args.tag or provider.name
    print(f"[provider] {provider.name} (real={provider.is_real})")
    if not provider.is_real:
        print("[warn] mock provider: scores are NOT evidence. Do not publish these numbers.")

    rag = evaluate_rag(provider, args.k, args.systems, args.judge, tag)
    agent = evaluate_agent(provider, tag)
    write_results(rag, agent, tag, provider.name)


if __name__ == "__main__":
    main()
