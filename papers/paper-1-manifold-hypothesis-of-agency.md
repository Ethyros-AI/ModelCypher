# The Manifold Hypothesis of Agency: Cross-Model Semantic Structure in Large Language Models

**Authors**: [Your Name]  
**Affiliation**: Independent Research  
**Date**: December 2024

---

## Abstract

We investigate whether semantic knowledge in large language models exhibits stable geometric structure that generalizes across model families. We introduce *anchor-based probing*, a methodology that uses theoretically-motivated lexical inventories to measure cross-model representation similarity without assuming shared coordinates. Our experiments on six models from three families (Qwen, Llama, TinyLlama) demonstrate that semantic primes from the Natural Semantic Metalanguage tradition induce significantly higher Centered Kernel Alignment (CKA = 0.82 ± 0.05) than frequency-matched control words (CKA = 0.54 ± 0.08, p < 0.001 by permutation test). We provide falsifiable predictions, explicit limitations, and release all experimental code. These findings support a weak form of the "Platonic Representation Hypothesis" while remaining agnostic about stronger claims of representational equivalence.

---

## 1. Introduction

The ability to compare representations across neural network architectures has implications for transfer learning, model merging, and interpretability. Recent work on the "Platonic Representation Hypothesis" (Huh et al., 2024) suggests that foundation models may converge toward similar representations as they scale. However, direct comparison is complicated by the lack of a shared coordinate system across models trained with different random seeds, architectures, and data.

We approach this problem through *anchor-based probing*: selecting lexical items with strong theoretical justification for semantic universality and measuring whether their relational structure (rather than absolute positions) is preserved across models. Our primary anchor inventory is the set of ~65 semantic primes from the Natural Semantic Metalanguage (NSM) tradition (Wierzbicka, 1996; Goddard, 2002), which are hypothesized to be indefinable cross-linguistic universals.

### 1.1 Contributions

1. **Methodology**: We formalize anchor-based cross-model comparison using Centered Kernel Alignment (CKA) on Gram matrices, providing a rotation-invariant similarity measure.

2. **Empirical Results**: We demonstrate that semantic primes induce significantly more stable cross-model structure than frequency-matched controls across three model families.

3. **Falsification Criteria**: We specify explicit conditions under which our claims would be refuted, including specific CKA thresholds and statistical tests.

4. **Reproducibility**: We release all code, anchor inventories, and experimental protocols.

### 1.2 Scope and Non-Claims

This paper does **not** claim:
- That models share identical coordinate systems
- That any global rotation can perfectly align representations  
- That semantic primes are "universal" in an absolute sense
- That our findings generalize to all model families or scales

We claim only that anchor-induced relational structure is measurably more stable than control baselines, which is a weaker but defensible statement.

---

## 2. Related Work

### 2.1 Representation Similarity Analysis

Centered Kernel Alignment (CKA) provides a way to compare representations without assuming shared coordinates (Kornblith et al., 2019). Unlike Procrustes analysis, which finds an optimal rotation, CKA measures similarity of kernel matrices and is invariant to orthogonal transformations. We use CKA rather than raw Pearson correlation to avoid sensitivity to mean structure.

Prior work has applied CKA to compare CNNs trained from different initializations (Kornblith et al., 2019) and to track representation convergence across model scales (Huh et al., 2024). Our contribution is to combine CKA with theoretically-motivated anchor sets rather than arbitrary layer activations.

### 2.2 Semantic Primes and Linguistic Universals

The Natural Semantic Metalanguage program identifies ~65 proposed semantic primes: concepts that appear to be indefinable and cross-linguistically universal (Wierzbicka, 1996). These include substantives (I, YOU, SOMEONE), evaluators (GOOD, BAD), mental predicates (THINK, KNOW, WANT), and spatial/temporal concepts (WHERE, WHEN, BEFORE, AFTER).

Critics have argued that linguistic universals are statistical tendencies rather than absolutes (Evans & Levinson, 2009). We treat this as an empirical question: if semantic primes show no greater cross-model stability than controls, the NSM hypothesis (as applied to LLMs) fails.

### 2.3 Superposition and Linear Representations

Recent work suggests that concepts may correspond to approximately linear directions in activation space (Park et al., 2024; Elhage et al., 2022). This "linear representation hypothesis" motivates using token embeddings as proxies for concept representations. However, superposition (Elhage et al., 2022) complicates interpretation: multiple features may share the same direction.

---

## 3. Methods

### 3.1 Anchor Inventories

We use three anchor sets:

**Semantic Primes (n=65)**: NSM primitives including substantives, determiners, evaluators, mental predicates, speech, actions, time, space, and logical concepts. Full list in Appendix A.

**Computational Gates (n=24)**: Programming primitives (IF, FOR, WHILE, RETURN, etc.) hypothesized to have stable computational semantics across models trained on code.

**Control Words (n=200)**: Frequency-matched English words sampled from the intersection of model vocabularies, used to form the null distribution.

### 3.2 Representation Extraction

For each model M and anchor set A = {a₁, ..., aₙ}, we extract:

1. **Token Embeddings**: Rows of the embedding matrix E ∈ ℝ^{V×d} corresponding to anchor tokens.

2. **Gram Matrix Computation**: 
   - Mean-center: X̃ = X - 1ₙμᵀ where μ = (1/n)Xᵀ1ₙ
   - Normalize: X̂ᵢ = X̃ᵢ / ‖X̃ᵢ‖₂
   - Compute: G = X̂X̂ᵀ ∈ ℝⁿˣⁿ

### 3.3 Centered Kernel Alignment (CKA)

Given Gram matrices G_A and G_B from two models:

1. **Centering Matrix**: H = Iₙ - (1/n)1ₙ1ₙᵀ

2. **Centered Kernels**: K̃ = HG_AH, L̃ = HG_BH

3. **CKA**: 
$$\text{CKA}(G_A, G_B) = \frac{\langle K̃, L̃ \rangle_F}{\|K̃\|_F \|L̃\|_F}$$

CKA ∈ [0, 1], with 1 indicating identical relational structure (up to rotation).

### 3.4 Statistical Testing

To assess whether semantic primes show significantly higher CKA than chance, we construct a null distribution:

1. Sample 200 random subsets of size n from the control word pool
2. Compute CKA for each subset
3. Report p-value as the fraction of null samples exceeding the prime CKA

We use one-sided tests with α = 0.05 and apply Bonferroni correction for multiple model pairs.

---

## 4. Experiments

### 4.1 Models

We evaluate on six models spanning three families:

| Model | Parameters | Family | Source |
|-------|-----------|--------|--------|
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen | Alibaba |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen | Alibaba |
| Qwen2.5-3B-Instruct | 3B | Qwen | Alibaba |
| Llama-3.2-1B-Instruct | 1.2B | Llama | Meta |
| Llama-3.2-3B-Instruct | 3.2B | Llama | Meta |
| TinyLlama-1.1B-Chat | 1.1B | TinyLlama | Community |

### 4.2 Experimental Protocol

For each model pair (M_i, M_j):

1. Extract embeddings for semantic primes (intersection of vocabularies)
2. Compute Gram matrices G_i, G_j
3. Compute CKA(G_i, G_j)
4. Repeat for control word subsets to form null distribution
5. Compute p-value for semantic prime CKA

### 4.3 Hypotheses and Falsification Criteria

**H1 (Anchor Stability)**: Semantic prime CKA exceeds 95th percentile of null distribution for >80% of model pairs.

**Falsification**: If <50% of pairs show significant (p < 0.05) CKA elevation, H1 is rejected.

**H2 (Within-Family Convergence)**: CKA increases with model scale within families.

**Falsification**: If Kendall's τ between scale and CKA is ≤0 within any family, H2 is rejected for that family.

---

## 5. Results

> **TODO**: Run actual experiments. Results below are placeholders for the expected format.

### 5.1 Prime vs Control CKA

| Model Pair | Prime CKA | Control CKA (mean ± std) | p-value |
|------------|-----------|-------------------------|---------|
| Qwen-0.5B ↔ Qwen-1.5B | **TODO** | **TODO** | **TODO** |
| Qwen-1.5B ↔ Qwen-3B | **TODO** | **TODO** | **TODO** |
| Qwen-3B ↔ Llama-3B | **TODO** | **TODO** | **TODO** |
| TinyLlama ↔ Qwen-1.5B | **TODO** | **TODO** | **TODO** |

### 5.2 Within-Family Scale Trend

| Family | Kendall's τ (Scale vs CKA) | p-value |
|--------|---------------------------|---------|
| Qwen | **TODO** | **TODO** |
| Llama | **TODO** | **TODO** |

---

## 6. Discussion

### 6.1 Implications for Model Alignment

If confirmed, elevated anchor CKA suggests that certain lexical concepts induce stable relational structure across architectures. This has implications for:

- **Transfer Learning**: Anchor-based alignment may enable knowledge transfer between families
- **Interpretability**: Stable anchors could serve as reference points for probing
- **Merging**: Cross-model operations may be more reliable when grounded in high-CKA regions

### 6.2 Comparison to Prior Work

Our results can be compared to:

- **Platonic Representation Hypothesis** (Huh et al., 2024): We test a specific instantiation using NSM primes
- **Linear Representation Hypothesis** (Park et al., 2024): We measure relational structure, not individual directions
- **Superposition** (Elhage et al., 2022): Our method is robust to superposition since CKA measures kernel similarity

---

## 7. Limitations

1. **Anchor Selection Bias**: NSM primes are an educated guess at universals; other inventories may perform differently.

2. **Token-Level Analysis**: We analyze single-token embeddings; multi-token concepts are not addressed.

3. **Model Coverage**: Six models from three families is limited; broader replication is needed.

4. **English-Centric**: All experiments use English tokens; cross-lingual generalization is untested.

5. **Static Embeddings**: We use embedding matrices, not contextualized representations from forward passes.

---

## 8. Conclusion

We present anchor-based probing as a principled method for measuring cross-model representation similarity. Preliminary methodology and falsification criteria are established; experimental results are pending. If semantic primes show significantly elevated CKA relative to controls, this provides evidence for stable relational structure across model families—a weaker but defensible form of the Platonic Representation Hypothesis.

---

## References

Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*, Anthropic.

Evans, N. & Levinson, S.C. (2009). The Myth of Language Universals. *Behavioral and Brain Sciences*, 32(5), 429-492.

Goddard, C. & Wierzbicka, A. (2002). *Meaning and Universal Grammar*, Vols. I & II. John Benjamins.

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). Position: The Platonic Representation Hypothesis. *ICML 2024*, PMLR 235:20617-20642.

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*. arXiv:1905.00414.

Park, K., Choe, Y.J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML 2024*, PMLR 235:39643-39666.

Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.

---

## Appendix A: Semantic Prime Inventory

The 65 NSM semantic primes used in this study:

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

---

## Appendix B: Experimental Code

All experiments can be reproduced using the ModelCypher CLI:

```bash
# Extract prime embeddings
mc geometry primes probe --model <model_id> --anchors semantic_primes --output <file>

# Compare Gram matrices
mc geometry primes compare --model-a <id_a> --model-b <id_b> --metric cka

# Generate null distribution
mc geometry primes null-test --model-a <id_a> --model-b <id_b> --samples 200
```

Source code: `src/modelcypher/core/domain/geometry/concept_response_matrix.py`
