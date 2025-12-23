# Cross-Architecture Adapter Transfer via Geometric Alignment

**Authors**: [Your Name]  
**Affiliation**: Independent Research  
**Date**: December 2024

---

## Abstract

We address the problem of transferring fine-tuned adapters (e.g., LoRA weights) between large language models of different architectures. Current adapters are locked to their parent model, fragmenting the open-source ecosystem and wasting compute. We propose a geometric alignment pipeline that: (1) computes layer-wise compatibility via anchor-based similarity, (2) aligns weight matrices using constrained Procrustes rotation, and (3) applies DARE-style sparsification to reduce interference. We evaluate on Qwen→Llama and Mistral→Llama transfers, measuring skill retention and safety drift. Our method achieves **TODO**% skill retention versus 0% for naive transfer, while introducing **TODO**% safety drift. We release diagnostic tools for assessing cross-architecture compatibility and specify conditions under which transfer is inadvisable.

---

## 1. Introduction

The proliferation of LoRA adapters (Hu et al., 2022) has democratized fine-tuning: practitioners can specialize 70B models on consumer hardware. However, adapters are architecture-specific. A coding LoRA trained on Qwen cannot run on Llama without retraining.

We explore whether geometric alignment can bridge this gap. Our key insight is that adapter weights encode task-specific modifications in a subspace of the weight manifold; if source and target models share sufficient structure in this subspace, approximate transfer may be possible.

### 1.1 Problem Definition

Given:
- Source model S with weights W_S
- Target model T with weights W_T
- LoRA adapter Δ_S trained on S

Find: Projection Δ_T such that T + Δ_T exhibits similar behavior to S + Δ_S.

### 1.2 Approach

We decompose alignment into three spaces:

1. **Weight Space**: Permutation alignment (Git Re-Basin) + constrained rotation (Anchor-Locked Procrustes)
2. **Representation Space**: Layer-wise compatibility scoring via CKA on anchor sets
3. **Probability Space**: Behavioral validation via skill retention and safety drift metrics

### 1.3 Contributions

1. **Diagnostic Framework**: Tools for assessing cross-architecture compatibility *before* merge attempts
2. **Alignment Algorithm**: Anchor-locked Procrustes that prevents sign flips
3. **Evaluation Protocol**: Skill retention and safety drift metrics with falsification criteria

---

## 2. Related Work

### 2.1 Model Merging

**Task Arithmetic** (Ilharco et al., 2023) shows that task vectors (Δ = W_finetuned - W_pretrained) can be added, negated, and composed. This works when source and target share the same base model.

**Git Re-Basin** (Ainsworth et al., 2023) addresses permutation symmetry: different random initializations lead to equivalent loss basins connected by weight permutations. Aligning permutations enables zero-barrier interpolation.

**TIES-Merging** (Yadav et al., 2023) resolves sign disagreements that cause interference when merging task vectors from different fine-tuning runs.

**DARE** (Yu et al., 2024) shows that 90-99% of delta parameters can be randomly dropped without performance loss, suggesting extreme sparsity in task-relevant subspace.

### 2.2 Cross-Architecture Transfer

Prior work on cross-architecture transfer is limited. Bansal et al. (2021) demonstrate "stitching" layers between different architectures with linear transformations. Singh & Jaggi (2020) use optimal transport to soft-align neurons by activation patterns.

We extend these ideas to adapter transfer, combining permutation alignment, rotation, and sparsification.

---

## 3. Methods

### 3.1 Compatibility Assessment

Before attempting transfer, we compute layer-wise compatibility:

**Intersection Map**: For each layer pair (l_S, l_T):
1. Extract anchor embeddings A_S ∈ ℝ^{n×d_S}, A_T ∈ ℝ^{n×d_T}
2. Compute CKA(G_S, G_T) where G = AA^T
3. Flag layers with CKA < threshold as incompatible

**Coverage Score**: Fraction of layers with CKA > 0.7.

### 3.2 Anchor-Locked Procrustes

Standard Procrustes finds rotation R minimizing ||A_SR - A_T||². However, Procrustes solutions are ambiguous up to sign flips across singular values, which can cause "mirror world" bugs.

**Anchor Locking**: We constrain R such that designated anchor pairs have positive cosine similarity:

$$R^* = \arg\min_R ||A_SR - A_T||_F^2 \quad \text{s.t.} \quad \langle a_i R, b_i \rangle > 0 \; \forall i \in \text{locks}$$

We solve this via iterative projection: SVD for unconstrained rotation, then sign correction for locked anchors.

### 3.3 DARE Sparsification

After rotation, we apply DARE-style sparsification:

1. Compute magnitude |Δ_T| for each parameter
2. Drop parameters below the p-th percentile
3. Rescale remaining parameters by 1/(1-p)

We sweep p ∈ {0.8, 0.9, 0.95, 0.99} and select based on validation performance.

### 3.4 Algorithm

```
Algorithm: Cross-Architecture Adapter Transfer
Input: Source model S, target model T, adapter Δ_S
Output: Adapted adapter Δ_T

1. Compute intersection map I = CKA(layer pairs)
2. If coverage(I) < 0.5: ABORT (incompatible)
3. For each compatible layer pair (l_S, l_T):
   a. Extract anchor matrices A_S, A_T
   b. Compute R = AnchorLockedProcrustes(A_S, A_T)
   c. Rotate: Δ_T[l] = R @ Δ_S[l] @ R^T  (for square matrices)
4. Apply DARE sparsification with p=0.9
5. Return Δ_T
```

### 3.5 Evaluation Metrics

**Skill Retention**: 
$$\text{Retention}(S, T) = \frac{\text{Score}(T + Δ_T, \text{task})}{\text{Score}(S + Δ_S, \text{task})}$$

**Safety Drift**: Increase in harmful response rate between T and T + Δ_T.

---

## 4. Experiments

### 4.1 Transfer Pairs

| Source | Target | Adapter | Domain |
|--------|--------|---------|--------|
| Qwen2.5-7B-Instruct | Llama-3.2-8B | Coder | Code |
| Qwen2.5-3B-Instruct | Llama-3.2-3B | Instruct | Chat |
| Mistral-7B-Instruct | Llama-3.2-8B | Creative | Writing |

### 4.2 Baselines

- **Naive**: Direct weight copy (expected: 0% retention)
- **Weight Average**: Simple averaging of source and target weights
- **TIES**: TIES-Merging without rotation
- **Ours**: Anchor-locked Procrustes + DARE

### 4.3 Evaluation Suites

**Coding** (50 problems): Subset of HumanEval with pass@1 scoring.

**Safety** (100 prompts): Curated harmful/benign pairs with human-labeled responses.

### 4.4 Hypotheses and Falsification

**H1 (Transfer Possible)**: Our method achieves >50% skill retention for at least one transfer pair.

**Falsification**: If all pairs show <30% retention, cross-architecture transfer is not viable with our approach.

**H2 (Safety Preserved)**: Safety drift <10% across all pairs.

**Falsification**: If any pair shows >20% safety drift, the method is unsafe.

---

## 5. Results

> **TODO**: Run experiments.

### 5.1 Compatibility Assessment

| Pair | Coverage Score | Compatible? |
|------|---------------|-------------|
| Qwen-7B → Llama-8B | **TODO** | **TODO** |
| Qwen-3B → Llama-3B | **TODO** | **TODO** |
| Mistral-7B → Llama-8B | **TODO** | **TODO** |

### 5.2 Skill Retention

| Method | Qwen→Llama (Code) | Mistral→Llama (Creative) |
|--------|------------------|-------------------------|
| Naive | 0% | 0% |
| Weight Avg | **TODO** | **TODO** |
| TIES | **TODO** | **TODO** |
| Ours | **TODO** | **TODO** |

### 5.3 Safety Drift

| Transfer | Baseline Refusal | Post-Transfer Refusal | Drift |
|----------|-----------------|----------------------|-------|
| Qwen→Llama | **TODO** | **TODO** | **TODO** |

---

## 6. Limitations

1. **Architecture Constraints**: Requires compatible layer dimensions or projection layers
2. **Tokenizer Mismatch**: Different tokenizers mean different vocabulary representations
3. **Non-Bijective Layers**: Not all layer types have 1:1 correspondence across architectures
4. **Compute Cost**: Requires loading both source and target models simultaneously

---

## 7. Conclusion

We present a diagnostic and alignment framework for cross-architecture adapter transfer. The pipeline assesses compatibility before transfer, applies anchor-locked rotation to prevent sign errors, and validates with skill retention and safety metrics. Experimental results are pending; methodology is specified with explicit falsification criteria.

---

## References

Ainsworth, S.K., Hayase, J., & Srinivasa, S.S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*. arXiv:2209.04836.

Bansal, Y., et al. (2021). Stitching Neural Networks with Minimal Shift. *NeurIPS 2021*.

Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv:2106.09685.

Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. *ICLR 2023*. arXiv:2212.04089.

Singh, S.P., & Jaggi, M. (2020). Model Fusion via Optimal Transport. *NeurIPS 2020*.

Yadav, P., et al. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. arXiv:2306.01708.

Yu, L., et al. (2024). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE). *ICML 2024*. arXiv:2306.01708.
