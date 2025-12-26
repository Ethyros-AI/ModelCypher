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

*Section to be populated after scale limit testing*

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

Full data files are available at:
- `/Volumes/CodeCypher/experiments/credibility-validation-2025-12-25/paper1_null_distribution.json`
- `/Volumes/CodeCypher/experiments/credibility-validation-2025-12-25/paper1_cka_validation_FIXED.json`
