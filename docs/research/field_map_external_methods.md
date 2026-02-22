# External Methods Landscape: Geometry-Derived Training (2024-2026)

**Source:** Field map compiled 2026-02-22 from ~200 papers across 8 research threads.

**Key conclusion from the literature:** No unified system exists that derives all training hyperparameters from a single spectral/geometric analysis. ModelCypher is the only implementation attempting this. Individual pieces (spectral optimizers, rank derivation, layer targeting, stopping criteria) exist as isolated tools.

---

## Thread 1: Learning Rates from Curvature

**Field status:** Mature but treacherous. The consensus is that spectral information works best for **preconditioning**, not for directly setting step sizes.

### Key Methods

| Method | Paper | Approach | Adam-specific? |
|--------|-------|----------|----------------|
| **CDAT** | Roulet et al., NeurIPS 2024 | EMA-smoothed curvature, EoS-aware | No (general) |
| **Sophia** | Liu et al., ICLR 2024 | Diagonal Hessian + element-wise clipping | Yes |
| **Muon** | Jordan & Bernstein, 2024-2025 | Polar factor of gradient, spectral norm duality | No |
| **SOAP** | Vyas et al., ICLR 2025 | Adafactor in Shampoo's eigenbasis | No |
| **Spectra** | arXiv:2602.11185, Feb 2025 | Track gradient SV spike subspace, spectral shaping | No |
| **D-Adaptation** | Defazio & Mishchenko, ICML 2023 | Distance-to-solution lower bound | No (general) |
| **Prodigy** | arXiv:2306.06101, ICML 2024 | Weighted dual averaging on D | Built on Adam |
| **Schedule-Free** | Defazio et al., NeurIPS 2024 | Iterate averaging, eliminates schedules | No |

### ModelCypher Mapping

**MASS replaces this entire thread.** MASS sidesteps curvature estimation via SPS (per-step measured rate) + Weyl geometric bounds. The field map's key finding — (L₀,L₁)-relaxed smoothness makes HVP unreliable (Zhang ICLR 2020) — is exactly what ModelCypher's ablation confirmed empirically.

**Relevant theoretical context:**
- CDAT's EoS finding: greedy η = 2/λ_max breaks Edge-of-Stability dynamics. ModelCypher's old Lipschitz approach would have hit this.
- Zhang ICLR 2020: explains WHY the HVP-based Lipschitz measurement failed (loss surface is non-smooth)
- Abreu et al. (Oct 2025): full Gauss-Newton achieves 5.4× fewer iterations than SOAP/Muon at 150M — establishes a large gap between current methods and the theoretical oracle

**Fallback candidates:** D-Adaptation (if MASS ceiling fails at scale), Muon-inspired spectral-norm control (if per-layer control needed). See `lr_derivation_analysis.md`.

---

## Thread 2: LoRA Rank from Spectral Structure

**Field status:** Active and accelerating. Multiple approaches published 2024-2025.

### Key Methods

| Method | Paper | Rank Source | In Production? |
|--------|-------|------------|----------------|
| **SR-LoRA** | Zhang et al., June 2025 | Stable rank: ‖W‖²_F / ‖W‖²_2 | No |
| **EVA** | Paischer et al., ICLR 2025 | Incremental SVD on activation minibatches | **Yes (HF PEFT)** |
| **SARA** | Gu et al., Aug 2024 | Cumulative SV energy threshold | No |
| **GeLoRA** | Ed-dib et al., EMNLP 2025 | ID lower bound: r_i ≥ max(d_{i+1} - d_i, 0) | No |
| **rsLoRA** | arXiv:2312.03732 | Scaling fix: α/√r instead of α/r | Yes (HF PEFT) |

### ModelCypher Mapping

**ModelCypher's approach is unique:** `tail_dims = full_rank - floor(shannon_effective_rank)` — the null-space capacity derived from Shannon entropy of the squared singular value distribution (Roy & Vetterli 2007).

**Comparison:**

| Approach | What It Measures | ModelCypher Equivalent |
|----------|-----------------|----------------------|
| SR-LoRA (stable rank) | ‖W‖²_F / ‖W‖²_2 | Trivially computable from existing SVD. Worth comparing as a sanity check. |
| EVA (activation SVD) | Explained variance of activations | Complementary — data-driven vs weight-structure-driven. Not equivalent. |
| SARA (SV energy) | Cumulative energy threshold | Similar spirit to Shannon effective rank, but uses a threshold instead of entropy. |
| GeLoRA (ID bound) | Intrinsic dimension of hidden states | ModelCypher computes ID (TwoNN) but uses it for monitoring, not rank selection. |

**Key insight from rsLoRA:** Standard LoRA scaling (α/r) causes collapse of stable rank — the adapter's update concentrates into a single dominant direction regardless of nominal rank. ModelCypher's Cayley parameterization avoids this entirely: the semi-orthogonal constraint guarantees all rank dimensions contribute equally.

**Complicating finding:** Staats et al. (NeurIPS 2025) show small singular values carry surprising importance in MLP projections. Simple cumulative-energy thresholding may be insufficient. ModelCypher's tail_dims approach addresses this by construction: tail_dims measures null-space capacity, not the "least important" directions.

**Action item:** Compute SR-LoRA's stable rank for the 350M model as a comparison metric alongside tail_dims. The values are trivially available from existing SVD data.

---

## Thread 3: DeepSeek mHC and Birkhoff Polytope Constraints

**Field status:** Narrow but deep. Addresses signal amplification in multi-channel residual streams.

### Key Result

DeepSeek's mHC (arXiv:2512.24880, Dec 2025): Unconstrained residual mixing matrices produce 3000× signal gain across 60 layers. Constraining to the **Birkhoff polytope** (doubly stochastic matrices) via Sinkhorn-Knopp normalization reduces this to ~1.6×.

Properties of doubly stochastic constraint:
- ‖H‖₂ ≤ 1 (prevents signal explosion)
- Compositional closure (depth-independent stability)
- Row-sum = 1 (convex combination of streams, energy conservation)
- Column-sum = 1 (gradient flow conservation)

**Validated at 3B, 9B, 27B MoE** with +7.2 BBH, +6.9 DROP.

### ModelCypher Mapping

**Reference for multi-channel architecture research thread** (RESEARCH-ROADMAP.md). If multi-channel residual is explored, Sinkhorn-Knopp projection is the right constraint mechanism.

**Connection to residual scaling:** ModelCypher's residual scaling `α_i = σ_max(x) / σ_max(f(x))` achieves signal normalization per-layer. mHC achieves cross-channel signal conservation. These are complementary approaches to the same problem (preventing signal amplification through depth).

**Broader pattern:** Constraining parameters to geometrically appropriate manifolds improves training:
- Birkhoff polytope for mixing matrices (mHC)
- Stiefel manifold for LoRA factors (Cayley parameterization — ModelCypher)
- Simplex for attention weights (softmax — universal)

---

## Thread 4: No Unified System Exists

**Field status:** This is the gap ModelCypher fills.

### The Field's Assessment

The literature confirms that no published work combines:
- Adaptive learning rate
- Rank selection per layer
- Layer targeting
- Weight decay
- Stopping criteria

...all derived from the same geometric analysis. The closest competitors are individual tools that could be combined but never have been:

| What | External Tool | Status |
|------|-------------|--------|
| Layer selection | Spectrum (Marchenko-Pastur SNR) | In Axolotl |
| Rank redistribution | EVA (activation SVD) | In HF PEFT |
| LR derivation | Prodigy (distance geometry) | In HF Diffusers |
| Schedule elimination | Schedule-Free (iterate averaging) | PyPI package |

These have never been combined, and none shares a common geometric framework.

### ModelCypher's Position

ModelCypher derives **all 15 hyperparameters** from three sources (SVD, IEEE 754, measured data):

| # | Hyperparameter | Source |
|---|----------------|--------|
| 1 | Learning Rate | MASS (Weyl ceiling + SPS + Weyl displacement) |
| 2 | Adam Epsilon | Spectral noise floor: max(σ_k², √ε×σ_max²) |
| 3 | Momentum | Cayley-Riemannian natural gradient (replaces Adam) |
| 4 | Weight Decay | σ_k / σ_max per layer |
| 5 | Gradient Clipping | REMOVED (preconditioner bounds prevent) |
| 6 | Warmup | REMOVED (geometric LR stable from step 0) |
| 7 | LR Schedule | OPTIONAL (cosine marginal) |
| 8 | Batch Size | Gradient noise scale B_crit |
| 9 | Early Stopping | Loss stability + adapter saturation (Weyl) |
| 10 | LoRA Scale | σ_k(W) / ‖BA‖_spectral per layer |
| 11 | LoRA Rank | tail_dims (null-space capacity) |
| 12 | Target Modules | tail_dims > 0 |
| 13 | Dropout | redundancy × adapter_fraction (NB-LoRA: 0.0) |
| 14 | Weight Init | ‖BA‖_spectral = σ_k from step 0 |
| 15 | Residual Scaling | σ_max(x) / σ_max(f(x)) per layer |

This is, as of February 2026, **genuinely without public precedent**.

---

## Thread 5: Adam Dominance at Frontier

**Field status:** Every frontier lab uses AdamW except Google (Distributed Shampoo for Gemini 1.5 Flash).

### Key Results

- **FAdam** (Hwang, 2024): Proved Adam's v_t IS the diagonal empirical Fisher — a connection asserted without proof in the original Adam paper
- Adam uses √(FIM)⁻¹ rather than the true natural gradient's FIM⁻¹ (loses covariance information)
- **AdaFisher** (ICLR 2025): Block-diagonal Kronecker Fisher without the square root outperforms Adam, K-FAC, and Shampoo

### ModelCypher Mapping

ModelCypher's Cayley-Riemannian preconditioner (P = M M^T where M = I + Z) is a **full pullback metric inverse**, not a diagonal approximation. This is mathematically superior to Adam's diagonal Fisher:

- Adam: P_adam = diag(1/√v_t) — diagonal, loses all covariance
- K-FAC/Shampoo: P_kfac = Kronecker block-diagonal — captures within-layer covariance
- ModelCypher: P = (I+Z)(I+Z)^T — full rank-r inverse metric on the Stiefel manifold

The FAdam proof reinforces this choice: Adam IS a crude approximation to natural gradient. ModelCypher does the full computation (at rank-r cost, not full-parameter cost).

**Why frontier labs haven't adopted better preconditioners:** (1) $5-100M+ training costs make switching risky, (2) distributed implementations need custom CPU-accelerator pipelines, (3) models trained with one optimizer can't easily be fine-tuned with another.

---

## Thread 6: Geometric Convergence Criteria

**Field status:** Emerging but undeployed.

### Key Methods

| Method | Paper | Signal |
|--------|-------|--------|
| **Heavy-tailed spectral stopping** | He et al., Oct 2025 | Power-law α of value matrix spectrum converges to ~2.5 |
| **ε-rank staircase** | Yang et al., Dec 2024 | Effective rank jumps correlate with loss decreases |

### ModelCypher Mapping

**ModelCypher's geometric stopping certificate covers this domain** with 4 conditions:
1. Stationarity: ‖Pg‖ at numerical floor
2. Improvement bound: Δmax < CI width
3. Worst-group: per-batch improvement < noise
4. Mechanism drift: entropy/repetition within dtype bounds

**Complementary signals worth exploring:**
- Heavy-tailed α monitoring during training — could serve as an independent confirmation of convergence
- ε-rank staircase — the connection to data-rank ceiling is suggestive: if ε-rank plateaus at tail_dims, the adapter has learned all it can at this rank

**The "spectral budget" concept** (monitoring how much available spectral capacity has been consumed) **does not appear in the literature under any terminology**. ModelCypher's adapter_saturation_median_ratio (‖BA‖₂/σ_k) is a genuinely novel framing.

---

## Thread 7: Consumer Fine-Tuning Tools

**Field status:** Defaults, not derivations. No consumer tool derives training parameters from spectral analysis.

| Tool | Spectral Features | Training Parameter Derivation? |
|------|------------------|-------------------------------|
| **Unsloth** | Dynamic Quantization 2.0 (per-layer compression sensitivity) | Quantization only, not training |
| **Axolotl** | Spectrum method (Marchenko-Pastur SNR) | Layer selection only |
| **mlx-lm** | Basic LoRA/QLoRA | None |
| **Torchtune** | Recipe configs | None |
| **LLaMA-Factory** | GUI defaults | None |

**ModelCypher's position:** The only tool that derives rank, LR, layer targeting, weight decay, stopping from spectral analysis. The gap between what exists in the consumer space and what geometry makes possible is enormous.

---

## Thread 8: Diagnostic → Prescriptive Bridge

**Field status:** Crossing in progress.

### The Progression

| Level | What | Status | Examples |
|-------|------|--------|----------|
| **Mature** | Post-hoc diagnostics | Done | WeightWatcher (power-law α), CKA |
| **Mature** | Spectral-aware optimizers | Done | Muon, SOAP, Spectra |
| **Emerging** | Pre-training prescription | Active | Spectrum (layer selection), EVA (rank redistribution) |
| **Emerging** | Training-time adaptation | Active | AdaLoRA (adaptive rank), heavy-tailed stopping |
| **Missing** | Unified prescriptive system | Gap | Real-time ε-rank monitoring, spectral budget, integrated derivation |

### ModelCypher's Bridge Crossings

| Domain | Bridge Status |
|--------|--------------|
| Rank derivation | **Crossed** — tail_dims prescribes rank from weight SVD |
| Layer targeting | **Crossed** — tail_dims > 0 prescribes targets |
| Weight decay | **Crossed** — σ_k/σ_max prescribes per-layer decay |
| Stopping criteria | **Crossed** — adapter saturation + loss stability |
| Learning rate | **In progress** — MASS replaces broken Lipschitz |
| Spectral budget tracking | **Crossed** — adapter_saturation_median_ratio (novel) |

**Worth comparing:** Spectrum method's Marchenko-Pastur SNR-based layer targeting against ModelCypher's tail_dims > 0 approach. Both are spectral prescriptions from different angles — Spectrum looks at signal-to-noise, tail_dims looks at null-space capacity.

---

## Cross-Cutting Findings

### The Full Gauss-Newton Ceiling

Abreu et al. (Oct 2025): Full GN preconditioning achieves **5.4× fewer iterations** than SOAP/Muon at 150M. Layerwise GN nearly matches. This establishes a large gap between current approximate methods and the oracle.

**Implication for ModelCypher:** The Cayley-Riemannian preconditioner operates at rank-r (not full parameter count). It's closer to diagonal Adam than to full GN. There may be significant room for improvement in preconditioning quality, though at higher computational cost.

### Hyperparameter Transfer Across Scale

**Complete(d) Parameterisation** (Apple ML Research, Dec 2025): Transfers LR, weight decay, Adam params, and init scales across width, depth, batch size, and training duration simultaneously.

**Implication for ModelCypher:** ModelCypher derives per-model (not transferred). If a model's geometry is analyzed, the parameters are correct for that model. No small-scale sweep needed. This is arguably stronger than transfer (direct derivation vs extrapolation), but transfer methods provide a useful sanity check.

### The μP Connection

**μP** (Yang et al., Microsoft): Derives LR scaling from infinite-width theory. ModelCypher's per-layer σ_k/σ_max effectively achieves similar width-aware scaling without the infinite-width assumption — it's measured from the actual finite model, not extrapolated from a limit.

---

*Document created: 2026-02-22*
*This is a living reference. Update when new methods are published or ModelCypher's infrastructure changes.*
