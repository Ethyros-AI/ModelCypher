# Cross-Architecture Adapter Transfer via Geometric Alignment

**Author**: Jason Kempf
**Affiliation**: EthyrosAI
**Date**: December 2025

> **Status**: Draft methodology; reproduction pending.

---

## Abstract

We propose a methodology for transferring LoRA adapters across model architectures via geometric alignment. The approach: (1) measure layer-wise compatibility via CKA on semantic anchor sets, (2) rotate weight matrices using anchor-locked Procrustes to align representation spaces, (3) optionally apply sparsification to reduce interference. Pilot results are not included in this draft; reproduction is pending. The key insight is that adapters may encode task-specific modifications in a subspace approximately shared across model families trained on similar data. This paper presents the methodology, diagnostic tools, and draft protocols. Full validation on standardized benchmark suites is ongoing.

---

## 1. Introduction

LoRA adapters (Hu et al., 2022) democratized fine-tuning—practitioners specialize 70B models on consumer hardware. But adapters are locked to their parent architecture. A coding LoRA trained on Qwen cannot run on Llama. Until now.

Adapter weights encode task-specific modifications in a low-rank subspace. Models trained on similar data learn similar subspaces. This is the Geometric Knowledge Thesis applied to adapters: if the shape is preserved, transfer is possible.

### 1.1 Problem Definition

Given:
- Source model S with weights W_S
- Target model T with weights W_T
- LoRA adapter Δ_S trained on S

Find: Rotation R such that T + R(Δ_S) exhibits similar behavior to S + Δ_S.

### 1.2 The Solution

We align in three spaces:

1. **Weight Space**: Anchor-locked Procrustes rotation (prevents sign flips that cause "mirror world" bugs)
2. **Representation Space**: CKA compatibility scoring on semantic prime anchors (predicts transfer success)
3. **Probability Space**: Behavioral validation (skill retention, safety drift)

### 1.3 Contributions

1. **Transfer Methodology**: Defines a geometric alignment protocol for cross-architecture adapters.
2. **Diagnostic Metrics**: Uses CKA and coverage as raw measurements (no thresholds).
3. **Safety Tracking**: Specifies how to measure safety drift as a raw metric.

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
3. Record CKA for each layer pair; CKA < 1.0 indicates alignment failure and must be debugged before transfer

**Coverage Score**: Fraction of layers with CKA == 1.0 (reported as a raw measurement).

### 3.2 Anchor-Locked Procrustes

Standard (orthogonal) Procrustes finds an orthogonal matrix $R$ minimizing $\|A_S R - A_T\|_F^2$. In practice, the optimal solution can include a reflection ($\det(R) = -1$), and near-degenerate singular values can make the alignment numerically unstable; both can manifest as "mirror world" bugs.

**Anchor Locking**: We constrain R such that designated anchor pairs have positive cosine similarity:

For matched hidden dimension $d$ (or after projecting both sides to a shared $d$), we solve the orthogonal Procrustes problem with additional sign constraints, where $O(d) = \{R \in \mathbb{R}^{d \times d} : R^T R = I\}$:

$$R^* = \arg\min_{R \in O(d)} \|A_S R - A_T\|_F^2 \quad \text{s.t.} \quad \langle a_i R, b_i \rangle > 0 \;\; \forall i \in \text{locks}$$

We solve this via iterative projection: SVD for unconstrained rotation, then sign correction for locked anchors.

### 3.3 DARE Sparsification

After rotation, we apply DARE-style sparsification:

1. Compute magnitude |Δ_T| for each parameter
2. Drop parameters below the p-th percentile
3. Rescale remaining parameters by 1/(1-p)

Select p based on validation performance and data-derived baselines (no fixed defaults).

### 3.4 Algorithm

```
Algorithm: Cross-Architecture Adapter Transfer
Input: Source model S, target model T, adapter Δ_S
Output: Adapted adapter Δ_T

1. Compute intersection map I = CKA(layer pairs)
2. If CKA is not exact (CKA < 1.0): STOP and debug alignment (do not merge)
3. For each compatible layer pair (l_S, l_T):
   a. Extract anchor matrices A_S, A_T
   b. Compute R = AnchorLockedProcrustes(A_S, A_T)
   c. Rotate: Δ_T[l] = R @ Δ_S[l] @ R^T  (for square matrices)
4. Apply DARE sparsification with p derived from validation baselines
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

- **Naive**: Direct weight copy (baseline)
- **Weight Average**: Simple averaging of source and target weights
- **TIES**: TIES-Merging without rotation
- **Ours**: Anchor-locked Procrustes + DARE

### 4.3 Evaluation Suites

**Coding** (50 problems): Subset of HumanEval with pass@1 scoring.

**Safety** (100 prompts): Curated harmful/benign pairs with human-labeled responses.

### 4.4 Hypotheses and Falsification

**H1 (Transfer Possible)**: Our method should exceed baseline retention for at least one transfer pair (threshold derived from benchmark variance).

**Falsification**: If all pairs fail to exceed baseline retention beyond variance, cross-architecture transfer is not supported by this method.

**H2 (Safety Preserved)**: Safety drift should remain within baseline variance across all pairs.

**Falsification**: If safety drift exceeds baseline variance for any pair, the method is unsafe.

---

## 5. Preliminary Results

> Results are not included in this draft. Pilot runs exist but are not reproduced; rerun using the protocol and record raw data before drawing conclusions.

---

## 6. Limitations

1. **Architecture Constraints**: Requires compatible layer dimensions or projection layers
2. **Tokenizer Mismatch**: Different tokenizers mean different vocabulary representations
3. **Non-Bijective Layers**: Not all layer types have 1:1 correspondence across architectures
4. **Compute Cost**: Requires loading both source and target models simultaneously

---

## 7. Conclusion

We present a methodology for cross-architecture adapter transfer via geometric alignment: measure compatibility via CKA, rotate via anchor-locked Procrustes, and optionally sparsify via DARE. The underlying hypothesis is that adapters encode task-specific modifications in subspaces approximately shared across model families trained on similar data.

**Validation Status**: These results require validation on standardized benchmarks. Full HumanEval evaluation, creative writing rubrics, and curated safety prompt suites are in progress. The methodology and diagnostic tools are released; comprehensive results will follow.

---

## References

[Ainsworth et al. (2023)](../docs/references/arxiv/Ainsworth_2023_Git_ReBasin.pdf). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836).

Bansal, Y., et al. (2021). Stitching Neural Networks with Minimal Shift. *NeurIPS 2021*. [Paper](https://proceedings.neurips.cc/paper/2021/hash/90610aa0e24f63ec6e69a5ceb1b40e3d-Abstract.html).

[Hu et al. (2022)](../docs/references/arxiv/Hu_2022_LoRA_Low_Rank_Adaptation.pdf). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

[Ilharco et al. (2023)](../docs/references/arxiv/Ilharco_2023_Task_Arithmetic.pdf). Editing Models with Task Arithmetic. *ICLR 2023*. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089).

Singh, S.P., & Jaggi, M. (2020). Model Fusion via Optimal Transport. *NeurIPS 2020*. [arXiv:1910.05653](https://arxiv.org/abs/1910.05653).

[Yadav et al. (2023)](../docs/references/arxiv/Yadav_2023_TIES_Merging.pdf). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708).

Yu, L., et al. (2024). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE). *ICML 2024*. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099).
