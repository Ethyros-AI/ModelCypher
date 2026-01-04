# Negative Results and Narrowed Hypotheses

**Document Purpose**: This file records experimental results that did not support initial hypotheses, following best practices for scientific transparency and reproducibility.

**Status**: Historical snapshot from 2025-12-25. Results have not been reproduced against current code paths. Treat numbers as provisional and rerun with recorded datasets and prompts before relying on them.

---

## 1. Semantic Primes Are Not More Special Than Random Words

**Original Claim (Paper 1)**:
> "Semantic primes achieve CKA = 0.82 ± 0.05 across model families, compared to CKA = 0.54 ± 0.08 for frequency-matched controls (p < 0.001)."

**Experiment Date**: 2025-12-25

**Methodology**:
- Extracted embeddings for 47 semantic primes common to both Qwen2.5-3B and Mistral-7B vocabularies
- Generated 200 null samples of 47 random words each
- Computed CKA using normalized Gram matrices (unit diagonal normalization)
- All comparisons used the same word sets across both models

**Results**:
| Metric | Semantic Primes | Random Words |
|--------|-----------------|--------------|
| CKA | 0.9175 | 0.9380 ± 0.0030 |
| 95th percentile | - | 0.9422 |
| Effect size (Cohen's d) | -6.76 | - |
| p-value | 1.0 | - |

**Observations**:
- In this run, semantic primes produced lower cross-model CKA than the random baseline.
- Both sets produced high CKA for this model pair.

**Possible Explanations / Follow-ups**:
1. The original claim may have used a different CKA normalization or centering scheme
2. Different model pairs may show different patterns
3. The semantic "primeness" may manifest in dimensions CKA doesn't capture
4. Random words may share more surface-level features (morphology, frequency) that boost CKA

**Working Hypothesis (to test)**:
> Cross-model CKA is uniformly high for most word sets, reflecting shared training dynamics and tokenization strategies rather than semantic structure specifically.

---

## 2. Scale Limits and Memory Constraints

**Experiment Date**: 2025-12-25
**Hardware**: Apple M4 Max, 128GB unified memory

### Observation: Memory Use From a Single-Machine Run

Unlike training (which requires gradient memory), geometric analysis and merging are inference-weight operations. This section records one machine's observed memory usage; it is not a general limit.

### Memory Test Results

| Model Combination | Combined Weight Size | Peak RAM Used | RAM Utilization |
|-------------------|---------------------|---------------|-----------------|
| Qwen2.5-3B + Mistral-7B | 9.6 GB | 9,774 MB | 7.5% |
| Qwen3-8B + Qwen2.5-3B | 10.1 GB | 10,280 MB | 7.8% |
| **Qwen3-80B + Mistral-7B** | **46 GB** | **46,655 MB** | **35.6%** |
| **Qwen3-80B + Qwen3-8B** | **47 GB** | **47,161 MB** | **36.0%** |
| **Qwen3-80B + Qwen2.5-3B-bf16** | **48 GB** | **48,653 MB** | **37.1%** |

### Notes

1. These measurements reflect the model weight footprint only (no training gradients).
2. These measurements are hardware- and configuration-specific; rerun on your system.

### Observed Limit on 128GB M4 Max

- **Observed working**: 80B + 8B models (47GB combined, 36% RAM)
- **Headroom remaining**: 82GB after largest test

### Performance Timings (Single-Run Snapshot)

| Operation | Model Pair | Time |
|-----------|-----------|------|
| Interference analysis (4 domains) | 0.5B + 3B | 76s |
| Interference analysis (4 domains) | 3B + 7B | 293s |
| Model loading only | 80B + 3B | 3.4s |

---

## How to Cite This Document

When reporting results from ModelCypher experiments, cite both positive and negative results:

```
Our experiments produced mixed results: while cross-model CKA exceeded 0.9
for semantic primes, random word baselines also achieved CKA > 0.93,
suggesting the high CKA may reflect general representation similarity
rather than semantic structure specifically (see NEGATIVE-RESULTS.md).
```

---

## Experimental Data

Raw data files are not stored in this repo. If you rerun these experiments, capture inputs and outputs under a local `experiments/` directory and note the paths here.
