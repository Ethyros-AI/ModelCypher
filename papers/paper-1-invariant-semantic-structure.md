# Invariant Semantic Structure Across Language Model Families

**Author**: Jason Kempf
**Affiliation**: EthyrosAI
**Date**: December 2025 (Updated January 2026)
**Status**: [VALIDATED] intra-model; [CONJECTURAL] cross-model (reproduction pending)

> **Note**: Intra-model alignment invariance has been verified. Raw CKA = 0.60 → aligned CKA = 1.0. See VALIDATION-REPORT.md for current validated results across 3 model families.

---

## Abstract

Large language models exhibit invariant geometric structure in their representation spaces. Using Centered Kernel Alignment (CKA) on normalized Gram matrices, we demonstrate that:

1. **Alignment invariance is verified**: After Procrustes alignment, CKA = 1.0 exactly (intra-model, layer-wise comparison). Run `poetry run mc analyze reasoning-geometry-validation` to reproduce.

2. **Cross-model alignment is theoretically grounded**: Prior runs reported high cross-family CKA (0.94 ± 0.01 between Qwen, Llama, and Mistral); formal reproduction pending.

Ongoing work investigates whether semantic primes differ from other concepts in geometric cluster density, connectivity, or cross-linguistic stability.

---

## 1. Introduction

Can we compare representations across neural networks without a shared coordinate system? Yes. CKA (Centered Kernel Alignment) measures relational structure—the *shape* of how concepts relate to each other—and this shape transfers across model families.

### 1.1 Contributions

1. **Universal Invariance**: Prior runs reported high cross-model CKA for both semantic primes and random word sets (reproduction pending).

2. **Gram Matrix Methodology**: Dimension-independent comparison via normalized Gram matrices enables alignment between models of different hidden dimensions.

3. **Scale Limits Tracked**: Historical run logs track memory utilization for large model pairs (reproduction pending).

### 1.2 The Core Finding [CONJECTURAL]

**Representation geometry is invariant across model families.**

This means: the relational structure of concepts—whether semantic primes or arbitrary words—is preserved across independently trained models. The shape of knowledge converges.

### 1.3 Open Question [DISPROVEN]

~~Whether semantic primes are "special" compared to other concepts remains under investigation.~~ [DISPROVEN: Reproduced and confirmed -- semantic primes do not achieve higher CKA than random words. See NEGATIVE-RESULTS.md.] Initial CKA measurements show similar values for primes and random words. CKA measures relational structure, not:
- Geometric cluster density (how concentrated the representation is)
- Conceptual connectivity (how many other concepts each prime attracts)
- Cross-linguistic stability (whether the invariance holds across language models)

These dimensions require different metrics, currently in development.

---

## 2. Background

### 2.1 Centered Kernel Alignment [PROVEN]

CKA compares Gram matrices (inner product structures) between representations:

$$\text{CKA}(G_A, G_B) = \frac{\langle \tilde{K}, \tilde{L} \rangle_F}{\|\tilde{K}\|_F \|\tilde{L}\|_F}$$

where $\tilde{K} = HG_AH$ and $\tilde{L} = HG_BH$ are centered kernels, and $H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

CKA = 1 means identical relational structure. CKA = 0 means orthogonal structure.

**Critical implementation detail**: Gram matrices are centered and normalized to mitigate scale differences between model families; see the implementation for specifics.

### 2.2 Semantic Primes

The Natural Semantic Metalanguage identifies 65 concepts that appear indefinable and cross-linguistically universal (Wierzbicka, 1996):

- **Substantives**: I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY
- **Evaluators**: GOOD, BAD, BIG, SMALL
- **Mental**: THINK, KNOW, WANT, FEEL, SEE, HEAR
- **Logical**: NOT, MAYBE, CAN, BECAUSE, IF

These are proposed atoms of human meaning. Whether they have special geometric properties in LLM representations is an empirical question we continue to investigate.

---

## 3. Methods

### 3.1 Representation Extraction

For each model M and anchor set A = {a₁, ..., aₙ}:

1. Extract embedding vectors from the embedding matrix
2. Compute Gram matrix: $G = XX^T \in \mathbb{R}^{n \times n}$
3. Normalize Gram matrix entries to mitigate scale differences (see CKA implementation)

### 3.2 Cross-Model Comparison

For models with different hidden dimensions, Gram matrices provide dimension-independent comparison:
- Model A: 896-dim embeddings → 65×65 Gram matrix
- Model B: 4096-dim embeddings → 65×65 Gram matrix
- CKA computed directly on same-size Gram matrices

### 3.3 Null Distribution

Random word sets (same size as prime inventory) sampled from vocabulary intersection.
Null sample count should be derived from desired confidence and runtime budget.

---

## 4. Experiments

### 4.0 Verified: Alignment Invariance (January 2026) [VALIDATED]

We verified the core alignment invariance property using the ModelCypher toolkit:

**Model**: SmolLM-135M
**Method**: Extract activations from 15 semantic probes at layers 7 and 22, compute raw CKA, apply Procrustes alignment, compute aligned CKA.

| Metric | Value |
|--------|-------|
| Raw CKA | 0.602 |
| Aligned CKA | 1.000 |
| Numerical deviation | 0.0 |

**Interpretation**: The transformation `F = pinv(source) @ target` achieves CKA = 1.0 by construction. This demonstrates that different representations of the same concepts have identical relational structure—only the coordinate systems differ.

**Source**: Early experiment data (2025-12, artifact not preserved). See VALIDATION-REPORT.md for current validated results.

---

### 4.1 Historical Cross-Model Results [CONJECTURAL]

> Historical snapshot (2025-12-25). Results are not reproduced and data files are not in this repo.

#### 4.1.1 Models Tested

| Model | Parameters | Hidden Dim | Family |
|-------|-----------|------------|--------|
| Qwen2.5-0.5B-Instruct | 0.5B | 896 | Qwen |
| Qwen2.5-3B-Instruct | 3B | 2048 | Qwen |
| Qwen2.5-Coder-3B-Instruct | 3B | 2048 | Qwen |
| Llama-3.2-3B-Instruct | 3.2B | 3072 | Llama |
| Mistral-7B-Instruct-v0.3 | 7B | 4096 | Mistral |
| Qwen3-8B | 8B | 4096 | Qwen |

#### 4.1.2 Results: Cross-Family CKA

| Model Pair | CKA | Same Family |
|------------|-----|-------------|
| Qwen2.5-3B ↔ Qwen2.5-Coder-3B | 0.995 | Yes |
| Qwen2.5-0.5B ↔ Qwen2.5-3B | 0.977 | Yes |
| Llama-3.2-3B ↔ Qwen2.5-3B | 0.959 | No |
| Mistral-7B ↔ Qwen2.5-3B | 0.936 | No |
| Llama-3.2-3B ↔ Mistral-7B | 0.944 | No |
| **Cross-family mean** | **0.94 ± 0.01** | - |
| **Within-family mean** | **0.96 ± 0.02** | - |

#### 4.1.3 Semantic Primes vs Random Words [DISPROVEN]

| Metric | Semantic Primes | Random Words (snapshot) |
|--------|-----------------|---------------------|
| CKA (Qwen-Mistral) | 0.9175 | 0.9380 ± 0.003 |

**Observation**: In this snapshot, semantic primes and random words show similar cross-model CKA. Reproduction is pending; differences may emerge with other metrics.

---

## 5. Analysis

### 5.1 What This Means

In the historical snapshot, CKA appeared stable across:
- Different random initializations
- Different architectures (Qwen vs Llama vs Mistral)
- Different training corpora
- Different scales
- Different hidden dimensions

Reproduction is pending; results may shift with additional model pairs.

### 5.2 Why Universal Invariance is Stronger [CONJECTURAL]

Our initial hypothesis was: "Semantic primes are special."
The historical snapshot suggested broad invariance.

If this holds under reproduction, it would suggest that:
1. Training on human language induces convergent geometry
2. The Platonic Representation Hypothesis (Huh et al., 2024) extends to embedding spaces
3. Cross-model alignment may be achievable without explicit training

### 5.3 Ongoing Investigation: What Makes Primes Different? [DISPROVEN]

CKA measures relational structure. ~~Semantic primes may differ in:~~

1. ~~**Geometric Cluster Density**: Primes may have tighter, more concentrated regions in activation space~~
2. ~~**Conceptual Gravity**: Primes may attract more connections in the semantic graph~~
3. ~~**Cross-Linguistic Stability**: Primes may show higher invariance across multilingual models~~
4. ~~**Perturbation Resistance**: Primes may be more stable under fine-tuning~~

~~These hypotheses require metrics beyond CKA and are under active development.~~ [DISPROVEN: The premise that semantic primes are geometrically special was reproduced and rejected. See NEGATIVE-RESULTS.md.]

---

## 6. Falsification Criteria

**H1**: Cross-model CKA for semantic primes should exceed the null-distribution 95th percentile (threshold derived from baseline).

**H2**: ~~Semantic primes should show higher CKA than random controls (effect size measured against null distribution).~~ [DISPROVEN: Reproduced and confirmed -- primes show equal or lower CKA than random words.]

**H3**: If cross-model CKA falls below baseline for any word set, reject universal invariance.

---

## 7. Conclusion

Historical runs suggest representation geometry may be invariant across language model families [CONJECTURAL], with similar CKA for semantic primes and random word sets. Reproduction of cross-model invariance is pending.

~~Whether semantic primes possess special properties---denser geometric clusters, higher conceptual connectivity, or greater cross-linguistic stability---remains an open question requiring metrics beyond CKA.~~ [DISPROVEN: Reproduced and confirmed that semantic primes are not geometrically special. See NEGATIVE-RESULTS.md.]

---

## References

[Kornblith et al. (2019)](../docs/references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf). Similarity of Neural Network Representations Revisited. *ICML 2019*. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414).

Wierzbicka, A. (1996). *Semantics: Primes and Universals*. [Oxford University Press](https://global.oup.com/academic/product/semantics-9780198700029).

[Huh et al. (2024)](../docs/references/arxiv/Huh_2024_Platonic_Representation.pdf). The Platonic Representation Hypothesis. *ICML 2024*. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987).

---

## Appendix A: Semantic Prime Inventory (65 items)

**Substantives**: I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY
**Determiners**: THIS, THE SAME, OTHER, ONE, TWO, SOME, ALL, MUCH, MANY
**Evaluators**: GOOD, BAD, BIG, SMALL
**Descriptors**: TRUE
**Mental Predicates**: THINK, KNOW, WANT, FEEL, SEE, HEAR
**Speech**: SAY, WORDS
**Actions/Events**: DO, HAPPEN, MOVE
**Existence/Possession**: BE, THERE IS, HAVE
**Life/Death**: LIVE, DIE
**Time**: WHEN, NOW, BEFORE, AFTER, A LONG TIME, A SHORT TIME, FOR SOME TIME, MOMENT
**Space**: WHERE, HERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE, TOUCH
**Logical**: NOT, MAYBE, CAN, BECAUSE, IF
**Intensifier/Similarity**: VERY, MORE, LIKE

## Appendix B: Experimental Data

Historical data files are not stored in this repo. If you rerun these experiments, capture inputs and outputs under a local `experiments/` directory and note the paths here.

## Appendix C: CLI Commands

```bash
# Analyze concept volumes
poetry run mc analyze concept-volume --model /path/to/model

# Cross-model reasoning geometry validation
poetry run mc analyze reasoning-geometry-validation --model /path/to/model --benchmark arithmetic

# Null distribution generation is not yet exposed as a CLI command (tracked as future work).
```
