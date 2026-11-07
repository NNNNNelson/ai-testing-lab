# Retrieval ablation

provider: `ark` · 15 answerable cases · k=5 · corpus: HSBC Annual Report 2024 (first 119 pages)

| config | evidence hit@5 | Recall@5 |
|---|---|---|
| E0 dense | 86.67% | 41.61% | **best**
| E1 dense+translate | 80.00% | 38.11% |
| E2 bm25 | 53.33% | 16.98% |
| E3 bm25+translate | 86.67% | 37.73% |
| E4 hybrid | 73.33% | 24.42% |
| E5 hybrid+translate | 80.00% | 32.34% |

What moved: page-level aggregation and query translation. What did not:
BM25 on a Chinese query against an English corpus scores near zero -
the lexical channel only becomes useful after translation. Reported, not hidden.

Supporting pages per case are expanded by regex (evals/corpus.py),
so a hit on a different page carrying the same figure counts as a hit.
