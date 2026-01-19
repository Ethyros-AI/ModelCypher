# EXP001: Reasoning Transfer via Geometric Merge

> **Date:** January 18, 2026
> **Status:** In Progress
> **Hypothesis:** Reasoning capability can be transferred from a large model to a small model via null-space projection, preserving the small model's speed while gaining the large model's intelligence.

---

## Objective

Transfer reasoning capability from DeepSeek-R1-8B into LFM2.5-1.2B using geometric alignment and null-space projection.

**Success criteria:**
- Merged model shows improved reasoning benchmarks over LFM2.5 baseline
- Merged model maintains LFM2.5's inference speed (within 10%)
- CKA = 1.0 on training probes (alignment achieved)

---

## Models

### Target (Base Model)
- **Name:** LFM2.5-1.2B-Instruct
- **Path:** `/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16`
- **Parameters:** 1.2B
- **Architecture:** Liquid Foundation Model (optimized for edge/speed)
- **Why:** Fast inference, good instruction following, 97%+ null space available

### Source (Knowledge Donor)
- **Name:** DeepSeek-R1-0528-Qwen3-8B
- **Path:** `/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16`
- **Parameters:** ~8B
- **Architecture:** Qwen3 base with DeepSeek R1 reasoning training
- **Why:** Exceptional reasoning capability, strong on GPQA/MATH/coding

---

## Benchmarks

### Primary Metrics (Reasoning)
| Benchmark | Description | Measures |
|-----------|-------------|----------|
| **GPQA** | Graduate-level science questions | Deep reasoning |
| **MMLU-Pro** | Multi-task language understanding (harder) | General knowledge + reasoning |
| **GSM8K** | Grade school math word problems | Mathematical reasoning |
| **ARC-Challenge** | Science reasoning questions | Logical reasoning |

### Secondary Metrics (Preservation)
| Benchmark | Description | Measures |
|-----------|-------------|----------|
| **HellaSwag** | Commonsense completion | Basic language understanding |
| **TruthfulQA** | Factual accuracy | Alignment preservation |
| **Inference Speed** | Tokens/second | Speed preservation |

### Geometric Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **CKA (training probes)** | Alignment quality | 1.0 |
| **CKA (held-out probes)** | Generalization | > 0.95 |
| **Null space utilization** | % of null space filled | Measured |
| **Condition number** | Numerical stability | < 1e6 |

---

## Protocol

### Phase 1: Baseline Benchmarks

1. **Benchmark LFM2.5-1.2B-Instruct**
   ```bash
   # Run all benchmarks, log results
   python experiments/merge_experiments/benchmark.py \
     --model /Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 \
     --output experiments/merge_experiments/results/lfm25_baseline.json
   ```

2. **Benchmark DeepSeek-R1-8B**
   ```bash
   python experiments/merge_experiments/benchmark.py \
     --model /Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 \
     --output experiments/merge_experiments/results/deepseek_r1_baseline.json
   ```

### Phase 2: Geometric Analysis

1. **Characterize bottleneck structure**
   - Compute effective rank at each layer for both models
   - Identify optimal merge depth (where bottlenecks align)
   - Measure null space availability in target

2. **Compute alignment**
   - Generate/load probe set (atlas probes)
   - Collect activations at merge layer
   - Compute alignment matrix F = pinv(source) @ target
   - Verify CKA = 1.0 on training probes

### Phase 3: Merge

1. **Execute geometric merge**
   ```bash
   mc merge run \
     --source /Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 \
     --target /Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16 \
     --output /Volumes/CodeCypher/models/merged/exp001_lfm25_deepseek_r1 \
     --log experiments/merge_experiments/results/exp001_merge_log.json
   ```

2. **Log merge metrics**
   - Alignment CKA (training/held-out)
   - Condition number
   - Null space utilization
   - Preserved fraction (behavioral norm)

### Phase 4: Post-Merge Benchmarks

1. **Benchmark merged model**
   ```bash
   python experiments/merge_experiments/benchmark.py \
     --model /Volumes/CodeCypher/models/merged/exp001_lfm25_deepseek_r1 \
     --output experiments/merge_experiments/results/exp001_merged.json
   ```

2. **Compare results**
   - Reasoning improvement vs LFM2.5 baseline
   - Speed preservation vs LFM2.5 baseline
   - Any capability degradation

---

## Expected Results

### Optimistic Scenario
- GPQA: 38.89 (baseline) → 50+ (merged)
- MMLU-Pro: 44.35 (baseline) → 55+ (merged)
- Speed: < 5% degradation

### Realistic Scenario
- GPQA: 38.89 → 42-45 (10-15% improvement)
- MMLU-Pro: 44.35 → 48-50 (8-12% improvement)
- Speed: < 10% degradation

### Failure Modes to Watch
1. **CKA < 1.0 on training probes** → Alignment failed, need more/different probes
2. **Reasoning degrades** → Null space projection disrupted target behavior
3. **Speed degrades significantly** → Architecture mismatch issues
4. **No improvement** → Knowledge didn't transfer (probe coverage insufficient)

---

## Files

```
experiments/merge_experiments/
├── EXP001_REASONING_TRANSFER.md    # This document
├── benchmark.py                     # Benchmarking script
├── configs/
│   └── exp001_config.json          # Experiment configuration
├── results/
│   ├── lfm25_baseline.json         # Target baseline benchmarks
│   ├── deepseek_r1_baseline.json   # Source baseline benchmarks
│   ├── exp001_merged.json          # Merged model benchmarks
│   └── exp001_merge_log.json       # Merge process metrics
└── analysis/
    └── exp001_analysis.py          # Results analysis script
```

---

## Notes

- LFM2.5 uses Liquid architecture (not standard transformer) - verify layer alignment compatibility
- DeepSeek-R1 was trained with extensive reasoning RL - this is the capability we want to transfer
- Monitor memory usage - 8B source may need careful handling
- If first merge fails, iterate on: probe selection, merge depth, alignment method

---

## Log

### 2026-01-18
- Experiment designed
- Models identified
- Protocol written
- Next: Run baseline benchmarks
