# ModelCypher Paper Review — 2026-02-20

**Generated**: 2026-02-25 20:15 (deep-dive complete)
**Papers scanned**: 23
**Papers with geometric relevance**: 2 (2 survived deep-dive)

## Executive Summary

Both papers that passed the automated filter are genuinely relevant after deep-dive:

1. **Modular Addition Mechanisms** (score 6.0) — Complete spectral-mechanistic account of feature learning. The "lottery ticket via initial spectral magnitude" result directly validates ModelCypher's thesis that SVD structure is causal. Grokking decomposed into three spectral stages. **Addresses Q4 (geometry from architecture) and training dynamics → geometry.**

2. **CrispEdit** (score 5.5) — K-FAC projection for capability preservation during LLM editing. Complementary to ModelCypher's null-space projection: we project into the null-space of activations (geometric); CrispEdit projects into the low-curvature subspace of the loss landscape (functional). **Addresses Q1 (layer-wise invariants) and could tighten Weyl monitoring.**

---

## 1. On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking

**arXiv**: [2602.16849](https://arxiv.org/abs/2602.16849)
**Authors**: Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang
**HF Upvotes**: 6
**Geometric Relevance Score**: 6.0
**Code**: [https://github.com/Y-Agent/modular-addition-feature-learning](https://github.com/Y-Agent/modular-addition-feature-learning)

### Summary
We present a comprehensive analysis of how two-layer neural networks learn features to solve the modular addition task. Our work provides a full mechanistic interpretation of the learned model and a theoretical explanation of its training dynamics. While prior work has identified that individual neurons learn single-frequency Fourier features and phase alignment, it does not fully explain how these features combine into a global solution. We bridge this gap by formalizing a diversification condition that emerges during training when overparametrized, consisting of two parts: phase symmetry and frequency diversification. We prove that these properties allow the network to collectively approximate a flawed indicator function on the correct logic for the modular addition task. While individual neurons produce noisy signals, the phase symmetry enables a majority-voting scheme that cancels out noise, allowing the network to robustly identify the correct sum. Furthermore, we explain the emergence of these features under random initialization via a lottery ticket mechanism. Our gradient flow analysis proves that frequencies compete within each neuron, with the "winner" determined by its initial spectral magnitude and phase alignment. From a technical standpoint, we provide a rigorous characterization of the layer-wise phase coupling dynamics and formalize the competitive landscape using the ODE comparison lemma. Finally, we use these insights to demystify grokking, characterizing it as a three-stage process involving memorization followed by two generalization phases, driven by the competition between loss minimization and weight decay.

### Key Math

**Lottery ticket mechanism** (proven via gradient flow ODE): The winning frequency `k*` for neuron `j` is determined at initialization:
```
k* = argmin_k D̃_m^k(0)
```
where `D̃_m^k(0)` depends on initial spectral magnitude at frequency `k`. The neuron with the largest initial Fourier coefficient wins.

**Phase coupling** (proven): Output phase converges to 2× input phase: `ψ_j = 2φ_j`. Deterministic geometric coupling.

**Diversification condition**: (1) Phase symmetry — phases distribute uniformly, enabling majority-vote noise cancellation. (2) Frequency diversification — each neuron specializes to one frequency.

**Grokking = three spectral stages**: (1) Memorization: broad spectrum, scattered phases. (2) Sparsification: weight decay prunes frequencies (entropy drops). (3) Cleanup: magnitude refinement, perfect phase alignment. Transitions driven by `λ_decay / η_lr` ratio.

### Extractable Code

[GitHub repo](https://github.com/Y-Agent/modular-addition-feature-learning):
- `src/mechanism_base.py` — DFT magnitude/phase extraction per neuron, frequency competition analysis
- `precompute/generate_analytical.py` — ODE simulation of gradient flow dynamics (theoretical backbone)
- Gradio app with 9 analysis tabs including DFT heatmaps + phase coupling

### ModelCypher Integration Notes

1. **LoRA init insight**: Paper proves initial spectral magnitude determines which features survive. ModelCypher inits with `||BA||_spectral = σ_k` — magnitude is set. But phase alignment of initialization may also matter. Could inform refined init in `cayley_lora.py`.

2. **Spectral entropy as stopping signal**: Track spectral entropy of adapter weights during training. Memorization→sparsification→cleanup manifests as entropy drops. Could refine `geometric_early_stopping.py` — stop when entropy stabilizes.

3. **Open questions addressed**: Q4 (geometry from architecture) — proves architecture + init SVD determines learned features, at least for 2-layer nets. Training dynamics → geometry — the `λ_decay / η_lr` ratio controls spectral stage transitions.

### Evidence Level
[PROVEN] for 2-layer networks on modular arithmetic | [CONJECTURAL] for transformers

---

## 2. CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing

**arXiv**: [2602.15823](https://arxiv.org/abs/2602.15823)
**Authors**: Zarif Ikram, Arad Firouzkouhi, Stephen Tu, Mahdi Soltanolkotabi, Paria Rashidinejad
**HF Upvotes**: 3
**Geometric Relevance Score**: 5.5
**Code**: [https://github.com/zarifikram/CrispEdit](https://github.com/zarifikram/CrispEdit)

### Summary
A central challenge in large language model (LLM) editing is capability preservation: methods that successfully change targeted behavior can quietly game the editing proxy and corrupt general capabilities, producing degenerate behaviors reminiscent of proxy/reward hacking. We present CrispEdit, a scalable and principled second-order editing algorithm that treats capability preservation as an explicit constraint, unifying and generalizing several existing editing approaches. CrispEdit formulates editing as constrained optimization and enforces the constraint by projecting edit updates onto the low-curvature subspace of the capability-loss landscape. At the crux of CrispEdit is expressing capability constraint via Bregman divergence, whose quadratic form yields the Gauss-Newton Hessian exactly and even when the base model is not trained to convergence. We make this second-order procedure efficient at the LLM scale using Kronecker-factored approximate curvature (K-FAC) and a novel matrix-free projector that exploits Kronecker structure to avoid constructing massive projection matrices. Across standard model-editing benchmarks, CrispEdit achieves high edit success while keeping capability degradation below 1% on average across datasets, significantly improving over prior editors.

### Key Math

**Constrained editing (Eq. 1)**: `min_θ ℓ_edit(θ) s.t. D_B(θ, θ₀) ≤ ε` where `D_B` = Bregman divergence of capability loss.

**Proposition 2 — Bregman → Gauss-Newton Hessian (exact, no convergence assumption)**:
The Bregman divergence's quadratic approximation yields the Gauss-Newton Hessian:
```
D_B(θ₀ + Δθ, θ₀) = ½ Δθᵀ [Jᵀ H_ℓ J] Δθ + o(||Δθ||²)
```
where `J = ∂f_θ/∂θ` (parameter-output Jacobian), `H_ℓ = ∂²ℓ/∂u²` (loss Hessian w.r.t. outputs).
Critical property: `∇_a D_B(a, a) = 0` always — gradient vanishes at θ₀ regardless of whether θ₀ is a local minimum. The full Hessian can have negative eigenvalues away from minima, but `G = Jᵀ H_ℓ J` is always PSD when `H_ℓ` is PSD (true for cross-entropy). Valid for ANY checkpoint.

**Proposition 1 — AlphaEdit is strictly more conservative**:
```
Null(K_cap) ⊆ Null(G_cap)
```
where `K_cap = I ⊗ [a₁, ..., aₙ]` (stacked input activations). Proof: if `ΔW` is in Null(K_cap), then `ΔW aᵢ = 0` for all samples, so the per-sample Jacobian `Jᵢ Δw = 0`, and since `G = Σ Jᵢᵀ Hᵢ Jᵢ` with `Hᵢ ≥ 0`, we get `G Δw = 0`. **ModelCypher's `F = pinv(source) @ target` projects into Null(K_cap) — the AlphaEdit-equivalent space. CrispEdit's feasible region is strictly larger.**

**K-FAC factorization (Eq. 5)** per layer:
```
G_cap^(l) ≈ A_{l-1} ⊗ S_l
A_{l-1} = E[a_{l-1} a_{l-1}ᵀ]  (input activation covariance, d_in × d_in)
S_l     = E[g_l g_lᵀ]          (pre-activation pseudo-gradient covariance, d_out × d_out)
```
Storage: O(d_in² + d_out²) instead of O(d_in² · d_out²).

**Energy threshold** for gamma-approximate null space:
```
k = min{ r ∈ [p] | Σ_{i=1}^r σᵢ / Σ_{i=1}^p σᵢ ≥ γ }
P_γ = U_{>k} U_{>k}ᵀ  (projector onto low-curvature directions)
```

**Matrix-free projector (Prop. 3, Eq. 6)** via Kronecker structure:
```
Q_proj = U_out @ ((U_outᵀ @ Q @ U_in) ⊙ M) @ U_inᵀ
M_{ij} = 𝟙[λᵢ^out · λⱼ^in ≤ λ_γ]  (binary mask on Kronecker eigenvalue products)
```
where `U_in, U_out` = eigenvectors of A, S factors. Eigenvalues of `A ⊗ S` are products `λ_{A,i} · λ_{S,j}` with eigenvectors `u_{S,j} ⊗ u_{A,i}`.

### Extractable Code

[GitHub repo](https://github.com/zarifikram/CrispEdit):

Core projection cache (`utils.py`):
```python
def calculate_projection_cache_with_kfac(A, B, energy_threshold=0.9):
    Sa, Ua = torch.linalg.eigh(A)     # eigendecomposition of input covariance
    Sb, Ub = torch.linalg.eigh(B)     # eigendecomposition of gradient covariance
    M = torch.outer(Sa, Sb)           # Kronecker eigenvalue products
    rank, null_threshold = get_rank_and_threshold_by_energy_ratio(
        M.view(-1), percent=energy_threshold)
    M = M < null_threshold            # boolean mask: True = low curvature (KEEP)
    return {'Ua': Ua, 'Ub': Ub, 'M': M}
```

Core projection step (`projected_sgd.py`, `projected_adam.py`):
```python
grad_proj = U_B @ ((U_B.T @ grad @ U_A) * M.T) @ U_A.T
```

Config: gamma=0.99 for LLaMA-3-8B, last 5 MLP down_proj layers, 1000 Wikipedia samples for K-FAC, lr=5e-4, 25 steps.

### ModelCypher Integration Notes

**Direct connection — Proposition 1 is the key result.** ModelCypher's null-space projection via `F = pinv(source) @ target` operates in Null(K_cap) — the activation null space. CrispEdit proves this is a strict subset of Null(G_cap), the Gauss-Newton null space. We're being more conservative than necessary. The GNH null space is the mathematically optimal constraint: it preserves capabilities with a strictly larger feasible region for edits.

| CrispEdit | ModelCypher Analog | Relationship |
|---|---|---|
| Null(G_cap) — GNH null space | Null space of source activations | CrispEdit is strictly LARGER (Prop 1) |
| K-FAC factor A = E[aaᵀ] | CKA probes / activation covariance | Same object — we already compute this |
| K-FAC factor S = E[ggᵀ] | Not in ModelCypher | **Missing factor** — adds output gradient curvature |
| Energy threshold γ | SVD rank selection | Both spectral truncation, different spectra |
| Bregman divergence | CKA post-training | Bregman is valid pre-convergence; CKA is not a divergence |

**What we could gain:**
1. **S_l (output gradient covariance)** is the missing piece. We already compute A_l during activation snapshots. Collecting pseudo-gradients `g_l = ∂ℓ/∂s_l` alongside would give us the full K-FAC factorization for free during the existing probe pass.
2. **Kronecker eigenvalue mask** replaces uniform Weyl bounds with curvature-weighted per-direction bounds. High-curvature directions get tighter bounds, low-curvature get relaxed. Net effect: more adapter capacity without sacrificing capability preservation.
3. **Bregman divergence as stopping signal** — a proper parameter-space divergence that works even when the model isn't converged. Could replace or complement CKA in `geometric_early_stopping.py`.

**Implementation path**: Collect `E[g_l g_lᵀ]` during existing probe pass → eigendecompose both factors → compute Kronecker mask → use matrix-free projector during training. The eigendecomposition is O(d²) per factor, done once before training.

### Evidence Level
[VALIDATED] — Tested on Llama-3-8B across 3 benchmarks (ZsRE, COUNTERFACT, WikiBigEdit), <1% capability degradation. Proposition 1 is [PROVEN].

---

## Quick Reference

| # | Score | Paper | arXiv | Code |
|---|-------|-------|-------|------|
| 1 | 6.0 | On the Mechanism and Dynamics of Modular Addition: Fourier F... | [2602.16849](https://arxiv.org/abs/2602.16849) | [repo](https://github.com/Y-Agent/modular-addition-feature-learning) |
| 2 | 5.5 | CrispEdit: Low-Curvature Projections for Scalable Non-Destru... | [2602.15823](https://arxiv.org/abs/2602.15823) | [repo](https://github.com/zarifikram/CrispEdit) |
