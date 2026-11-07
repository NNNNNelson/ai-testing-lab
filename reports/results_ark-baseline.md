# Results (ark-baseline)

provider: `ark`

## RAG

| metric | baseline-rag | oss-rag | threshold |
|---|---|---|---|
| hit_at_k | 86.67% | 86.67% | ≥0.90 |
| recall | 41.61% | 35.87% | observe |
| precision | 48.00% | 52.89% | observe |
| abstention | 100.00% | 100.00% | ≥0.80 |
| hallucination | 0.00% | 0.00% | ≤0.20 |
| numeric_leak | 0.00% | 80.00% | ≤0.20 |

## Agent

| metric | rule-router | intent-router |
|---|---|---|
| routing_accuracy | 90.00% | 100.00% |
| safety_block_rate | 100.00% | 100.00% |
