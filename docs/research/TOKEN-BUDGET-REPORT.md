# Token Budget Report

Date: 2026-03-01  
Threshold: 20,000 tokens (`cl100k_base`)

Command:

```bash
poetry run python scripts/report_token_budget.py --threshold 20000
```

## Source And Docs Status

- `src/`: **0 files** above threshold
- `docs/`: **0 files** above threshold

## Tracked Files Above 20k Tokens

These are large corpus/result assets (plus `poetry.lock`), not source modules:

| Tokens | File |
| ---: | --- |
| 1,394,581 | `data/probes/mmlu/mmlu_factual.json` |
| 590,005 | `data/training/expansion_train.jsonl` |
| 399,051 | `data/probes/mmlu/mmlu_linguistic.json` |
| 374,408 | `data/probes/mmlu_curated.json` |
| 352,858 | `data/training/1p2b_reasoning_foundation_train.jsonl` |
| 317,393 | `poetry.lock` |
| 239,291 | `data/probes/mmlu/mmlu_moral.json` |
| 216,430 | `data/probes/mmlu/mmlu_physical.json` |
| 186,158 | `data/probes/mmlu/mmlu_relational.json` |
| 160,522 | `data/probes/mmlu/mmlu_mathematical.json` |
| 149,334 | `data/training/expansion_eval.jsonl` |
| 96,977 | `data/training/format_augmented_train.jsonl` |
| 89,928 | `data/training/1p2b_reasoning_foundation_val.jsonl` |
| 75,255 | `data/training/benchmark_train.jsonl` |
| 68,389 | `data/probes/mmlu/mmlu_computational.json` |
| 49,521 | `data/training/reasoning_traces_train.jsonl` |
| 44,736 | `data/probes/mmlu/mmlu_logical.json` |
| 36,870 | `data/training/paired_reasoning_train.jsonl` |
| 33,395 | `data/probes/mmlu/mmlu_philosophical.json` |
| 32,415 | `data/training/ce_reasoning_traces_train.jsonl` |
| 27,584 | `data/training/mt_pure_train.jsonl` |
| 25,378 | `data/training/phases_1_4_combined.jsonl` |
| 23,878 | `plasma/results/anomaly_candidates.json` |
| 22,147 | `data/training/geometric_self_study.jsonl` |
| 20,285 | `data/training/phase5_benchmark_failures_base.jsonl` |
