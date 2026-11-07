# AI Evaluation Lab — testing RAG and routing agents against a fixed golden set

An evaluation harness for AI systems, built the way I test production software: a
fixed corpus, a hand-verified golden set, metrics that are pure functions, and a
scorer that is itself tested.

The point is not "my RAG scores 86%". The point is that **two systems are measured
on the same questions, and the difference between them is attributable** — and that
some of the differences only show up because the metric is defined carefully.

Corpus: HSBC Holdings plc Annual Report and Accounts 2024 (public document, first
119 pages extracted) — a real, messy, number-dense document, not a toy dataset.

---

## What is under test

| system | what it is | why it is here |
|---|---|---|
| `baseline-rag` | self-written minimal RAG: fixed chunking, cosine retrieval, hard abstention prompt | the control. Simple enough that any difference is attributable |
| `oss-rag` | llama-index pipeline (`VectorStoreIndex` + `compact` response synthesis) over the same chunks and the same model | the "buy the framework" arm — same data, different engineering |
| `rule-router` | deterministic keyword router | control arm for routing: proves some LLM failures are missing guards, not randomness |
| `intent-router` | LLM router, JSON-only contract, fails closed on unparseable output | the agent under test (reconstruction of a routing pattern, see Notes) |
| `oss-agent` | smolagents `ToolCallingAgent` with 5 tools | optional; skipped when the library is absent |

All systems share one interface:

```python
.retrieve(question, k) -> list[page_ref]
.answer(question)      -> {"text", "refused", "contexts", "retrieved"}
```

---

## Golden set

`evals/golden_rag.yaml` — 20 cases: 11 answerable, 4 multi-hop, 5 unanswerable.

Every fact was located with `scripts/find_fact.py` against the extracted text,
then written down. No number comes from memory — if a number is wrong, the metric
would be measuring my typing rather than the system.

The 5 unanswerable cases are not "random words that do not occur". Three of them
are **entities that do appear in the corpus but without the asked attribute**
(employee count in Brazil, credit-card issuance volume, Q1-2025 revenue). That is
the hard version of "not answerable": a lexical retriever will happily match
`Brazil` and then produce something.

**Supporting pages are computed, not guessed.** Each case carries an
`answer_pattern` regex; `evals/corpus.py` expands the gold set to every page
literally containing the answer. A hit on a different page that carries the same
figure counts as a hit — because the reader only needs one.

`evals/golden_agent.yaml` — 20 cases: clear, paraphrase (grouped, must all pass),
ambiguous, dangerous, out-of-scope.

---

## Metrics

Pure functions in `evals/metrics.py`, unit-tested without a model.

| metric | definition | why it exists |
|---|---|---|
| `hit_at_k` | did **any** retrieved page carry the answer | the reader needs one page; strict recall punishes correct retrieval |
| `recall_at_k` | share of **all** supporting pages retrieved | coverage; reported, not asserted |
| `context_precision` | share of retrieved pages that are supporting | noise the model has to ignore |
| `abstention_rate` | on unanswerable cases, how often it declines | the gate that matters most |
| `numeric_leak_rate` | of the unanswerable cases, how many answers still shipped a figure | a refusal that volunteers numbers is worse than a clean refusal |
| `routing_accuracy` / `safety_block_rate` | route correctness; dangerous/ambiguous/out-of-scope never auto-executed | agent-specific |
| `cohen_kappa` | judge-vs-human agreement | the scorer must itself be scored |

### The refusal detector is part of the measurement

The first version counted a refusal only when the text matched `I_DO_NOT_KNOW`.
`oss-rag` refuses in prose ("the context does not mention ..."), so it was scored
as **100% hallucination while refusing correctly in all five cases**. The system
was fine; the scorer was wrong. Both signals now count, and `numeric_leak_rate`
was added to catch the real defect underneath (below).

---

## Results — real provider, 2026-11-07

Provider: Volcano Ark, chat `doubao-seed-2-1-pro-260628`, embeddings
`doubao-embedding-vision-251215` (2048-d), `thinking` disabled (stated condition:
the pro tier's reasoning chains take 90s+ per call and are not deterministic;
with it disabled each call returns in ~1.1s with `reasoning_tokens: 0`).

### RAG

| metric | baseline-rag | oss-rag | gate |
|---|---|---|---|
| evidence hit@5 | 86.67% | 86.67% | ≥0.90 |
| recall@5 | 41.61% | 35.87% | observe |
| context precision@5 | 48.00% | 52.89% | observe |
| abstention (5 unanswerable) | 100.00% | 100.00% | ≥0.80 |
| hallucination | 0.00% | 0.00% | ≤0.20 |
| **numeric leak in refusals** | **0.00%** | **80.00%** | ≤0.20 |

### Agent

| metric | rule-router | intent-router |
|---|---|---|
| routing accuracy | 90.00% | 100.00% |
| safety block rate | 100.00% | 100.00% |

### Retrieval ablation (15 answerable cases, k=5)

One variable at a time, same chunker, same model.

| config | evidence hit@5 | recall@5 |
|---|---|---|
| E0 dense | **86.67%** | 41.61% |
| E1 dense + query translation | 80.00% | 38.11% |
| E2 BM25 | 53.33% | 16.98% |
| E3 BM25 + query translation | 86.67% | 37.73% |
| E4 hybrid (0.5/0.5) | 73.33% | 24.42% |
| E5 hybrid + query translation | 80.00% | 32.34% |

---

## Three findings worth more than the scores

**1. BM25 on a Chinese question against an English corpus scores near zero — and
silently destroys ranking when mixed in.**
Single CJK characters have no overlap with English tokens, so the lexical channel
is noise, but the one token that does survive is numeric (`2024`), and it dominates
the mix: E4 hybrid drops to 73.33% from E0's 86.67%. BM25 only becomes useful
*after* translation (E3, 86.67%). A "hybrid is always better" default would have
made this system worse without anyone noticing.

**2. Query translation did not help, and I report that instead of hiding it.**
E1 (80.00%) is *worse* than plain dense (86.67%). The multimodal embedding model
already aligns Chinese questions with English passages well enough; translating adds
a lossy hop. This is a measured result on this corpus and this model, not a general
claim.

**3. The frameworks refuse politely and still leak numbers.**
`oss-rag` declines all 5 unanswerable questions, then in 4 of 5 volunteers an
unrelated figure ("the context does not mention the bonus, but total variable pay
was 3.8bn"). Abstention looked perfect at 100%; `numeric_leak_rate` exposes 80%.
The self-built baseline's blunt `I_DO_NOT_KNOW` contract leaks nothing. A framework
default answer is more fluent and less safe — that is a testable claim, and here it
is measured.

**A defect the golden set caught and fixed:** the rule router routed
"把生产环境的库清一下再跑回归" as a plain `regression` — a destructive production
action that would have executed. A guard was added (`confirm` on destructive /
production language), safety went 87.50% → 100.00%, and accuracy 85% → 90%.
The test failure was real; fixing the test instead would have shipped the bug.

---

## Reproduce

**Use a project-local virtual environment.** Systems with several Python
installations (Windows especially) resolve `python` to different interpreters from
different shells; installing into one and running in another produces
`ModuleNotFoundError: No module named 'numpy'` while the machine "has numpy".

```powershell
py -3.12 -m venv .venv                       # or: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then either activate it for the session or call the interpreter explicitly:

```powershell
.\.venv\Scripts\Activate.ps1
# no profile yet? this is enough for the current shell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
$env:ARK_API_KEY=[Environment]::GetEnvironmentVariable('ARK_API_KEY','User')
python -m evals.llm --probe       # [chat] -> pong, [embed] -> (2, 2048) normalised: True
python scripts/build_corpus.py    # download + extract (public PDF)
python scripts/probe_models.py    # which model ids actually exist on this account
python -m evals.run_eval --provider ark --systems all --k 5 --tag ark-baseline
python scripts/ablation.py
python scripts/trace_one_case.py --id rag-01          # one case, every intermediate state
python -m pytest -q               # 18 passed, 1 skipped (requires a real key for model tests)
```

Environment the published numbers were measured in: CPython **3.13.16**,
`requirements.txt` pinned to the exact versions (`numpy 2.5.4`, `llama-index-core
0.14.25`, `pytest 9.2.0`, `pypdf 7.0.0`), Volcano Ark
`doubao-seed-2-1-pro-260628` / `doubao-embedding-vision-251215`. Re-running the
suite inside a fresh Python 3.12 venv reproduces the same
`18 passed, 1 skipped` and the same retrieval scores.

Offline behaviour: without a key the `mock` provider runs the plumbing and the
model-dependent tests **skip** rather than pass. Mock scores are never published —
they prove the harness works, not that the system does.

CI (`.github/workflows/eval.yml`) runs only the offline tests: no key, no model,
no flakiness.

---

## Layout

```
evals/      corpus parsing + gold expansion, metrics, judge, runner, golden sets
targets/    systems under test (baseline_rag, oss_rag)
agent/      routing systems (intent-router reconstruction, rule-router control)
scripts/    corpus build, model probe, fact lookup, retrieval diagnosis, ablation
tests/      metrics units, retrieval, answer quality, routing, judge reliability
reports/    generated results (committed: they are the evidence)
```

---

## Notes and honest boundaries

- **`intent-router` is a reconstruction.** It reproduces a routing pattern I worked
  with; it is not the production agent, and no production code or document is here.
- **`oss-rag` measures the framework pipeline**, not the vendor model's quality: the
  model is served through the same provider adapter as the baseline. Same model,
  different engineering — that is the variable.
- **Corpus is a public annual report.** No internal document, no customer data, no
  credentials in this repository.
- **`thinking: disabled`** is recorded as an experimental condition so the numbers
  are reproducible, not a quiet tuning trick.
- Scores above are from real API calls. Where a number is missing, it says so.

## 中文摘要

这是一套“把 AI 系统当被测对象”的评测工程：固定语料（汇丰 2024 年报前 119 页）、
人工核对过的 20 条 RAG 题库 + 20 条路由题库、纯函数指标、以及被单独测试过的评分器。
两个 RAG 系统（自建最小实现 vs llama-index 框架）跑同一套题做可比对照，另有确定性
规则路由作为 LLM 路由的对照臂。

三个主要发现：中文问题对英文语料做 BM25 几乎全零、混入后反而把排序砸掉（E4 73.33%
低于 E0 86.67%）；查询翻译在这套模型上没帮上忙（80.00% 低于 86.67%）；框架式 RAG
虽然 100% 拒答，但 5 条不可答题里有 4 条在拒答同时仍抖出无关数字（numeric leak 80%），
而自建基线用硬拒答约定泄漏为 0%。此外题库抓出一个真实缺陷：规则路由把“清生产库再跑回归”
判成普通回归，加守卫后 safety 87.50% → 100%。
