# Task Singular Vectors (TSV)

> CVPR 2025: Separating skills from structure via SVD decomposition.

---

## Why This Matters for Model Merging

When merging fine-tuned models, **task interference** is a major problem: capabilities from different tasks conflict in the merged model. Task Singular Vectors provide a principled way to:
1. **Compress** task vectors to 10% size while retaining 99% accuracy
2. **Measure** interference between tasks via singular vector alignment
3. **Reduce** interference through decorrelation

**In ModelCypher**: Implemented in `task_singular_vectors.py` for low-rank task representation and interference-aware merging.

---

## The Core Insight

Fine-tuned model weights can be decomposed as:

$$W_{fine} = W_{base} + \Delta W$$

where $\Delta W$ is the "task vector". The key discovery:

> **Task vectors are inherently low-rank.** Only a small portion of singular vectors capture task-specific function.

---

## Formal Definition

### Task Matrix Decomposition

For a layer $l$, let $\Delta W^{(l)} \in \mathbb{R}^{m \times n}$ be the task matrix. SVD gives:

$$\Delta W^{(l)} = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

where:
- $U = [u_1, \ldots, u_r]$: left singular vectors
- $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$: singular values
- $V = [v_1, \ldots, v_r]$: right singular vectors (Task Singular Vectors)
- $r = \min(m, n)$: rank

### The Task Singular Vectors

The **right singular vectors** $v_i$ span the input space directions that the task modifies. These are the Task Singular Vectors (TSV).

---

## TSV-Compress: 10× Compression

### Algorithm

Given task matrix $\Delta W$ and compression ratio $k$:

```
1. Compute SVD: ΔW = UΣV^T
2. Keep top-k singular values: Σ_k = diag(σ₁, ..., σ_k)
3. Reconstruct: ΔW_compressed = U_k Σ_k V_k^T
```

### The Low-Rank Property

Gargiulo et al. (2025) show empirically that:
- **90% of singular values can be discarded** with <1% accuracy loss
- Task matrices have effective rank much smaller than full rank
- This holds across vision and language models

---

## TSV-Merge: Interference Reduction

### Measuring Interference

For two task matrices $\Delta W_A$ and $\Delta W_B$ with TSVs $V_A$ and $V_B$:

**Task Interference** = $\|V_A^T V_B\|_F$

High interference means tasks modify the same input directions.

### ZCA Whitening for Decorrelation

The TSV-Merge algorithm applies ZCA whitening to decorrelate:

```
1. Concatenate TSVs: V = [V_A, V_B]
2. Compute covariance: C = V^T V
3. Whitening transform: W_zca = C^(-1/2)
4. Apply to task matrices before merging
```

### The Full Algorithm

```
Input: Task matrices {ΔW₁, ..., ΔW_T}, compression ratio k
Output: Merged task matrix ΔW_merged

For each layer l:
  1. SVD each task: ΔW_i = U_i Σ_i V_i^T
  2. Compress to rank k
  3. Concatenate TSVs: V = [V₁, ..., V_T]
  4. ZCA whiten: V' = V · C^(-1/2)
  5. Reconstruct: ΔW'_i = U_i Σ_i (V'_i)^T
  6. Average: ΔW_merged = (1/T) Σᵢ ΔW'_i

Return W_merged = W_base + ΔW_merged
```

---

## Theoretical Foundation

### Why Low-Rank?

The low-rank property connects to:
1. **LoRA's success**: Low-rank adaptation works precisely because task matrices are low-rank
2. **Intrinsic dimensionality**: Tasks live in low-dimensional subspaces
3. **Information geometry**: Task-specific changes concentrate in few directions

### Relationship to LoRA

LoRA explicitly parameterizes $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$.

TSV shows this is **not just a convenient parameterization** but reflects the true structure of fine-tuning.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/task_singular_vectors.py`](../../../../src/modelcypher/core/domain/geometry/task_singular_vectors.py)

| Class/Function | Line | Description |
|----------------|------|-------------|
| `TaskVectorDecomposition` | 64 | Result with U, S, Vt, variance captured, effective rank |
| `SVDBlendConfig` | 103 | Configuration with numerical stability params |
| `_find_spectral_gap()` | 142 | Find natural rank cutoff from spectral gap |

**Design decisions**:
1. **Per-layer analysis**: TSV computed separately per layer
2. **Adaptive rank**: Rank chosen to capture specified variance (default 99%)
3. **Backend-agnostic**: Works with MLX, JAX, or any backend

---

## Experimental Results (CVPR 2025)

From Gargiulo et al. (2025):

| Method | Accuracy (8 tasks) | Compression |
|--------|-------------------|-------------|
| Task Arithmetic | 67.8% | 1× |
| TIES-Merging | 72.1% | 1× |
| TSV-Compress | 67.2% | **10×** |
| TSV-Merge | **75.4%** | 10× |

Key findings:
- 10× compression with <1% accuracy loss
- TSV-Merge outperforms all baselines despite compression
- Interference reduction is key to performance

---

## Citations

### Primary Reference

1. **Gargiulo, A.A., Crisostomi, D., Bucarelli, M.S., Scardapane, S., Silvestri, F., & Rodolà, E.** (2025). "Task Singular Vectors: Reducing Task Interference in Model Merging." *CVPR 2025*.
   arXiv: 2412.00081
   Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf
   - *The foundational TSV paper*

### Related Model Merging

2. **Ilharco, G., et al.** (2023). "Editing Models with Task Arithmetic." *ICLR 2023*.
   arXiv: 2212.04089
   - *Task vectors concept*

3. **Yadav, P., et al.** (2024). "TIES-Merging: Resolving Interference When Merging Models." *NeurIPS 2023*.
   arXiv: 2306.01708
   - *Ties-based merging baseline*

4. **Yu, L., et al.** (2024). "DARE: Drop and Rescale for Model Merging." *ICML 2024*.
   arXiv: 2311.03099
   - *Sparsification-based merging*

### SVD-Based Methods

5. **Lialin, V., et al.** (2025). "STAR: Spectral Truncation and Rescale for Model Merging." *NAACL 2025*.
   - *Adaptive rank truncation*

6. **Panariello, A., et al.** (2025). "Accurate and Efficient Low-Rank Model Merging in Core Space." *arXiv*.
   - *Low-rank merging in aligned space*

7. **Kim, J., et al.** (2025). "KnOTS: Model merging with SVD to tie the Knots." *ICLR 2025*.
   arXiv: 2410.19735
   - *SVD for LoRA alignment*

---

## Related Concepts

- [fisher_information.md](fisher_information.md) - Importance weighting (complementary to TSV)
- [procrustes_analysis.md](procrustes_analysis.md) - Alignment before decomposition
- [intrinsic_dimension.md](intrinsic_dimension.md) - Why low-rank works

---

*Task Singular Vectors reveal that fine-tuning is fundamentally low-rank. This insight enables both compression and interference reduction.*
