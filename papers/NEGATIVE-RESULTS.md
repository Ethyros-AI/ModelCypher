# Negative Results and Narrowed Hypotheses

**Document Purpose**: This file records experimental results that did not support initial hypotheses, following best practices for scientific transparency and reproducibility.

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

**Interpretation**:
Semantic primes showed **lower** cross-model CKA than random words, not higher. The difference is statistically significant in the **wrong direction**.

**What This Means**:
1. ✅ **Supported**: Models share invariant structure (both primes and random words show CKA > 0.9)
2. ❌ **Not Supported**: Semantic primes are geometrically "special" compared to other words
3. ❌ **Not Supported**: The 52% improvement over controls claimed in Paper 1

**Possible Explanations**:
1. The original claim may have used a different CKA normalization or centering scheme
2. Different model pairs may show different patterns
3. The semantic "primeness" may manifest in dimensions CKA doesn't capture
4. Random words may share more surface-level features (morphology, frequency) that boost CKA

**Revised Hypothesis**:
> Cross-model CKA is uniformly high (>0.9) for most word sets, reflecting shared training dynamics and tokenization strategies rather than semantic structure specifically.

---

## 2. Scale Limits and Memory Constraints

**Experiment Date**: 2025-12-25
**Hardware**: Apple M4 Max, 128GB unified memory

### Key Finding: If You Can Run Inference, You Can Merge

Unlike training (which requires ~3x model size in RAM for gradients), geometric analysis and merging are computationally lightweight. The operations are simple matrix manipulations on embeddings.

### Memory Test Results

| Model Combination | Combined Weight Size | Peak RAM Used | RAM Utilization |
|-------------------|---------------------|---------------|-----------------|
| Qwen2.5-3B + Mistral-7B | 9.6 GB | 9,774 MB | 7.5% |
| Qwen3-8B + Qwen2.5-3B | 10.1 GB | 10,280 MB | 7.8% |
| **Qwen3-80B + Mistral-7B** | **46 GB** | **46,655 MB** | **35.6%** |
| **Qwen3-80B + Qwen3-8B** | **47 GB** | **47,161 MB** | **36.0%** |
| **Qwen3-80B + Qwen2.5-3B-bf16** | **48 GB** | **48,653 MB** | **37.1%** |

### Implications

1. **No "training overhead"** - Geometric operations use only the model weight memory
2. **80B models are practical** - An 80B 4-bit model uses only ~43GB, leaving 85GB for other operations
3. **Two 80B models could theoretically be loaded simultaneously** - 82GB headroom after 80B+3B
4. **Rule of thumb**: If you can load both models for inference, you can analyze and merge them

### Verified Limit on 128GB M4 Max

- **Confirmed working**: 80B + 8B models (47GB combined, 36% RAM)
- **Headroom remaining**: 82GB after largest test
- **Theoretical maximum**: ~110GB of combined model weights (accounting for system overhead)

### Performance Timings

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

Full data files are available in the `experiments/` directory after running validation commands.
