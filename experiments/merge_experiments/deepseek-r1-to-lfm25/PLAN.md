# Geometric Merge: DeepSeek-R1 → LFM 2.5

## Objective

Transfer knowledge from DeepSeek-R1-0528-Qwen3-8B to LFM2.5-1.2B-Instruct using closed-form geometric alignment.

## Models

| Role | Model | Path | Hidden Dim | Layers | Params | Vocab |
|------|-------|------|------------|--------|--------|-------|
| SOURCE | DeepSeek-R1-0528-Qwen3-8B-bf16 | `/path/to/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16` | 4096 | 36 | 8.19B | 151936 |
| TARGET | LFM2.5-1.2B-Instruct-bf16 | `/path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16` | 2048 | 16 | 1.17B | 65536 |
| OUTPUT | deepseek-r1-lfm25-merged | `/path/to/models/merged/deepseek-r1-lfm25` | 2048 | 16 | ~1.17B | 65536 |

## Cross-Architecture Notes

- **Hidden dimension compression**: 4096 → 2048 (2x)
- **Layer compression**: 36 → 16 (2.25x)
- **Cross-vocabulary**: Different tokenizers (151936 vs 65536 tokens)
- **Architecture difference**: Qwen3 (standard transformer) → LFM2 (hybrid with conv layers)

## Experiment Directory

All outputs saved to: `experiments/merge_experiments/deepseek-r1-to-lfm25/`

## Pre-merge Validation

Before running the merge, we validate:

1. **Model dimensions** - Record hidden_dim, intermediate_dim, num_layers for both models
2. **Vocabulary compatibility** - Check if cross-vocab merge (different tokenizers)
3. **Rank requirements** - Verify atlas has sufficient probes (n > max(src_hidden_dim, tgt_hidden_dim))

## Execution Steps

### Step 1: Model Inspection

```bash
poetry run mc model probe /path/to/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 --output json > experiments/merge_experiments/deepseek-r1-to-lfm25/source_profile.json

poetry run mc model probe /path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 --output json > experiments/merge_experiments/deepseek-r1-to-lfm25/target_profile.json
```

### Step 2: Run Geometric Merge

```bash
poetry run mc merge run \
  -s /path/to/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 \
  -t /path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 \
  -o /path/to/models/merged/deepseek-r1-lfm25 \
  --log-level DEBUG \
  2>&1 | tee experiments/merge_experiments/deepseek-r1-to-lfm25/merge.log
```

### Step 3: Benchmark Merged Model

```bash
poetry run mc benchmark run \
  /path/to/models/merged/deepseek-r1-lfm25 \
  --output json > experiments/merge_experiments/deepseek-r1-to-lfm25/benchmark_merged.json
```

### Step 4: Compare to Baseline

```bash
poetry run mc benchmark run \
  /path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 \
  --output json > experiments/merge_experiments/deepseek-r1-to-lfm25/benchmark_baseline.json
```

## Success Criteria

The merge is successful if:
1. All 3 stages complete without exception (the math is closed-form)
2. Full rank coverage achieved during probe stage
3. Merged model produces coherent output on inference test
4. Benchmark scores >= baseline (knowledge was added, not lost)

## Logs and Artifacts

| File | Description |
|------|-------------|
| `PLAN.md` | This plan |
| `source_profile.json` | Source model dimensions and config |
| `target_profile.json` | Target model dimensions and config |
| `merge.log` | Complete merge output with DEBUG logging |
| `benchmark_merged.json` | Merged model benchmark results |
| `benchmark_baseline.json` | Target baseline benchmark results |
| `RESULTS.md` | Final analysis and conclusions |

## Principles

- **No heuristics**: Every step is mathematically derived
- **No failure handling**: If the math is right, it works. Crashes indicate bugs.
- **Full documentation**: Every output is saved to the experiment directory
- **Reproducible**: Commands are exact and can be re-run
