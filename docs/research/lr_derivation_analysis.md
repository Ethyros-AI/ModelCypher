# Learning Rate Derivation: From Lipschitz to MASS

**Status:** MASS implemented (2026-02-22). Fallback candidates analyzed but not implemented.

---

## 1. The Failure Mode (Historical)

### What Broke

The original LR derivation used central-difference HVP + power iteration to estimate the Lipschitz constant L = λ_max(Hessian), then set η ≤ 2/(L × λ_max(P)) (Amari 1998, Nesterov 2004).

**Ablation evidence (REINFORCE ablation, 2026-02-22):**

| Exp | Config | Derived LR | Result (from 18/25 baseline) |
|-----|--------|-----------|-----|
| 0 | Default (CE+REINFORCE) | 0.996 | 5/25 (-13) |
| 1 | CE-only (no REINFORCE) | 1.64 | 13/25 (-5) |
| 2 | LR/10 | 0.072 | 16/25 (-2) |
| 3 | LR/100 | 0.0037 | 17/25 (-1) |
| 4 | Entropy floor 95% | 0.428 | 13/25 (-5) |
| 5 | REINFORCE-only | 0.366 | 15/25 (-3) |
| 8 | 10-batch Lipschitz | 1.13 | 11/25 (-7) |

**Key findings:**
1. Degradation is **monotonically correlated with LR magnitude** (not with training objective)
2. LR/100 ≈ 0.0037 nearly eliminates degradation (1 problem lost from baseline)
3. CE-only also degrades (exp 1) — root cause is LR, not REINFORCE
4. 10-batch stabilization doesn't help (exp 8) — median of 3-OOM-spread noise is noise

### Why Central-Difference HVP Fails

**Zhang et al. (ICLR 2020)** proved neural network loss functions are NOT globally Lipschitz smooth. They formalize **(L₀, L₁)-relaxed smoothness**:

```
L(θ) = L₀ + L₁ × ||∇f(θ)||
```

Local smoothness correlates positively with gradient norm. This means:
- When gradients are large (high loss, early training), L is large → η should be small
- When gradients are small (near convergence), L is small → η can be larger
- The Lipschitz constant varies by **orders of magnitude** along the training trajectory

**Empirical confirmation:** HVP values measured during ablation span 0.1 to 193 across minibatches within a single epoch. After 10 training steps, L jumps 10-25×. The loss landscape is non-smooth.

### Why the Per-Step Spectral Ceiling σ_k/σ_max Was Wrong

The per-step ceiling `eta_ceiling = σ_k_min / σ_max` prevents any single step from crossing a Weyl eigenvalue boundary. But over N steps per epoch, accumulated displacement scales as √N × per-step-displacement (Brownian scaling). For the 350M model:

- Per-step ceiling: σ_k_min/σ_max = 0.4005/3.7644 = 0.1064
- Per-step displacement: 0.1064 × 0.79 = 0.084
- Over 115 steps: √115 × 0.084 = 0.90 (2.25× past σ_k_min)
- The empirical sweet spot ≈ 0.003-0.004 (25-30× smaller than the per-step ceiling)

The fix: distribute the Weyl budget across epoch steps: `eta_ceiling = σ_k_min / (σ_max × √N)`. See §3 for validation results.

---

## 2. MASS — The Implemented Solution

**MASS (Measured-Adaptive Step Size)** replaces curvature estimation with per-step measurement + geometric bounds.

### Architecture

```
eta_step = min(eta_ceiling, eta_sps, eta_weyl)
```

**Layer 1: Static Weyl Ceiling (√N-corrected)**
```
eta_ceiling = σ_k_min / (σ_max × √N)
```
where N = batches per epoch.
- Derived from Weyl perturbation theory (Weyl 1912) + Brownian scaling
- Per-step bound σ_k_min/σ_max prevents single-step crossing
- √N correction distributes the Weyl budget across epoch steps
- Computed from pre-training SVD + dataset size
- Subject to validation-guided backoff: `eta_ceiling *= val_loss_ratio`, floor at √ε_f32

**Layer 2: Stochastic Polyak Step-size (SPS)**
```
eta_sps = f(x_t) / ||d_t||²
```
- Loizou et al. 2020
- Per-step measured rate derived from actual loss value and preconditioned gradient norm
- No curvature estimation — uses the loss function value directly
- Properties:
  - Naturally decreases as loss decreases (convergent behavior)
  - Naturally decreases as ||d|| increases (stabilizing under large gradients)
  - For quadratic objectives: η_sps = optimal step size (exact line search)
  - For non-convex: provides an upper bound on useful step size

**Layer 3: Weyl Displacement Bound**
```
eta_weyl = σ_k_min / ||d_t||
```
- Bounds per-step displacement relative to the Weyl crossing threshold
- Ensures no single step pushes the adapter's spectral contribution past the structural boundary
- Per-step adaptive (tracks ||d_t||)

**Layer 4: Validation Backoff**
```
eta_ceiling *= val_losses[-2] / val_losses[-1]  (if val loss increased)
floor: √ε_f32
```
- Only fires when validation loss increases
- Reduces ceiling multiplicatively
- Floor prevents ceiling from collapsing to zero

### Why MASS Works

1. **SPS measures what matters.** Instead of estimating curvature (which varies by 3 OOM), SPS uses the loss value itself — a direct measurement of how far the current point is from optimality. The step size η = f/||d||² interpolates toward the optimum for quadratic loss, and provides a reasonable bound for non-quadratic.

2. **SPS addresses (L₀,L₁) smoothness implicitly.** Under relaxed smoothness, η should decrease when ||g|| is large. SPS has η ∝ 1/||d||², which decreases quadratically with preconditioned gradient norm. This is actually MORE aggressive than the 1/L₀+L₁||g|| prescription — which may be why the Weyl displacement bound (η ∝ 1/||d||) serves as a softer alternative.

3. **Weyl bounds are geometric safety rails.** The ceiling and displacement bound are independent of the loss landscape. They constrain the adapter's effect on the base model's spectral structure regardless of optimization dynamics.

4. **No hyperparameters.** Every component is derived from:
   - SVD of base weights (σ_k, σ_max)
   - Per-step measurements (loss, gradient norm)
   - IEEE 754 precision (√ε_f32 floor)

---

## 3. MASS Validation Results (2026-02-22)

### Run 1: Per-step Ceiling Only (FAILED)

```
sigma_k_min = 0.4005, sigma_max = 3.7644
eta_ceiling = 0.4005 / 3.7644 = 0.1064
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| eta_step | 0.1064 | ~0.004 | 30× too high |
| eta_sps | 0.585 | binding | Not binding |
| eta_weyl | 0.505 | binding | Not binding |
| repetition | 0.603 | <0.1 | Catastrophic |
| entropy | 1.34 | >2.0 | Collapsing |
| adapter_sat (1 ep) | 0.67 | <0.5 | Too fast |

**Root cause:** The per-step ceiling prevents any single step from Weyl crossing, but accumulated displacement over an epoch scales as √N × per-step-displacement (Brownian scaling). Over 115 steps: `√115 × 0.084 = 0.90`, which is 2.25× past sigma_k_min. The budget is exhausted partway through the first epoch.

**Neither SPS nor Weyl displacement bind** because:
- SPS: `f(x)/||d||² = 0.37/0.79² = 0.59` — SPS assumes f* = 0, but for language model fine-tuning loss is never near zero
- Weyl displacement: `σ_k_min/||d|| = 0.40/0.79 = 0.51` — per-step gradient norm is small enough that displacement bound is easily satisfied

### Run 2: √N Epoch Budget Correction (HEALTHY)

**Fix:** `eta_ceiling /= √n_batches_per_epoch`

```
ceiling_step = 0.1064
n_batches = 46 (batch_size=20, 924 samples)
eta_ceiling = 0.1064 / √46 = 0.0157
```

| Epoch | eta_step | eta_sps | eta_weyl | train_loss | val_loss | rep | entropy | adapter_sat |
|-------|----------|---------|----------|------------|----------|-----|---------|-------------|
| 1 | 0.0157 | 1.35 | 0.44 | 1.11 | 2.40 | 0.055 | 2.49 | 0.23 |
| 2 | 0.0157 | 0.29 | 0.10 | 4.58 | 2.22 | 0.250 | 2.13 | 0.32 |
| 3 | 0.0157 | 0.88 | 0.24 | 2.39 | 2.07 | 0.178 | 2.41 | 0.45 |
| 4 | 0.0157 | 0.36 | 0.20 | 1.43 | 1.99 | 0.141 | 2.25 | 0.57 |

**Training is healthy:** monotonically decreasing val_loss (2.40→1.99 from baseline 2.73), modest repetition, good entropy, CKA min=0.965.

**But ceiling still binds at every step.** SPS and Weyl displacement remain 5-85× above the ceiling. They would need √N correction too, or a different f* assumption.

### Analysis: Why SPS Doesn't Bind

SPS formula: `η_sps = f(x)/||d||²` (Loizou et al. 2020, assumes f* = 0).

For language model fine-tuning, the irreducible loss is NOT zero — the baseline loss is ~2.7. SPS treats ALL of the loss as "distance to optimum," massively overestimating the useful step size. With f* = 2.7 instead of 0:
```
η_sps_corrected = max(0, f(x) - 2.7) / ||d||²
```
At epoch 1: `(2.49 - 2.7) / ||d||² < 0` → would give 0 (already past baseline).

This reveals a fundamental mismatch: SPS is designed for **convex optimization toward zero loss**. For fine-tuning where we're making small adjustments near baseline, SPS's f* = 0 assumption makes it non-binding.

### Key Finding: √N Correction Resolves Q11.2

The open question "√N budget distribution" (Q11 in OPEN-MATHEMATICAL-QUESTIONS.md) is now empirically confirmed:
- Per-step Weyl bound (eta × ||d|| ≤ σ_k) is necessary but insufficient
- Over N steps per epoch, accumulated displacement ≈ √N × eta × ||d|| (Brownian scaling)
- Epoch budget: `eta_ceiling = σ_k_min / (σ_max × √N)`
- For 350M: 0.0157 produces healthy training (vs 0.1064 producing catastrophic overfitting)
- Still 4× above the empirical sweet spot (0.004), but within an order of magnitude

### Run 3: MASS + REINFORCE (auto_regime) — DEGRADED

The critical test: does MASS + REINFORCE maintain the 350M baseline?

**Configuration:** auto_regime=True, regime_n_problems=25, max_iters=1000, no lr_override.

```
eta_ceiling = 0.0121 (√N-corrected, + val backoff)
Auto-regime baseline: 18/25 (72%) — selected hybrid (CE + REINFORCE)
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Online eval (epoch 0) | 16/25 (64%) | >= 18/25 | DEGRADED (-2) |
| eta_step | 0.0121 | ~0.004 | 3× above sweet spot |
| eta_sps | 0.983 | binding | Not binding |
| eta_weyl | 0.255 | binding | Not binding |
| repetition | 49.2% | <10% | High |
| adapter_saturation | 19.7% | — | Low (stopped early) |
| REINFORCE grad norm | 53.1 | — | Large |
| REINFORCE signal density | 0.28 | — | Low |
| train_loss | 5.63 → 2.43 | — | Rapid drop |
| Stop reason | online_eval_degraded | — | Early stopping worked |

**Comparison across LR methods:**

| Method | LR | Result | Delta from baseline |
|--------|-----|--------|---------------------|
| Lipschitz (broken) | 0.996 | 5/25 | -13 |
| MASS ceiling (CE+REINFORCE) | 0.012 | 16/25 | -2 |
| Manual LR/100 (CE+REINFORCE) | 0.004 | 17/25 | -1 |
| MASS ceiling (CE-only, Run 2) | 0.016 | healthy (4 epochs) | — |

**Key findings:**

1. **MASS is a massive improvement over Lipschitz** (from -13 to -2 problems), but still 3× above the empirical sweet spot.

2. **CE-only vs CE+REINFORCE:** MASS validation Run 2 (CE-only, η=0.016) was healthy over 4 epochs. The REINFORCE run (η=0.012, lower due to val backoff) degraded after 1 epoch. The combined CE + REINFORCE gradient pushes the model further than CE alone.

3. **The REINFORCE gradient is large:** `outcome_o_grad_norm = 53.1` vs typical CE gradient norms of ~1-4. Even though REINFORCE has its own tiny η (0.00022), the step norm (0.0114) is comparable to CE updates.

4. **Online eval gating worked correctly.** The system detected degradation (16/25 < 18/25) and stopped training. Without this gate, training would have continued to catastrophe.

**Open question:** Should MASS ceiling account for REINFORCE gradient magnitude? When REINFORCE is active, the effective total step is larger than CE alone. Options:
- Reduce ceiling when REINFORCE is active (by what factor?)
- Apply MASS ceiling to the total step (CE + REINFORCE) rather than to CE alone
- Use per-component MASS bounds

**Scripts:** `scripts/mass_validation.py`, `scripts/mass_reinforce_run.py`
**Results:** `/Volumes/CodeCypher/models/experiments/mass-reinforce/run1/`

---

## 4. Fallback Candidates (If MASS Proves Insufficient)

### (A) Distance-Geometry: D-Adaptation / Prodigy

**Core idea:** η ∝ D/G where D = ||θ* - θ₀|| (distance to solution), G = ||∑gₜ|| (accumulated gradient).

**D-Adaptation** (Defazio & Mishchenko, ICML 2023 Outstanding Paper):
- Maintains running lower bound d_k on D
- η_k = d_k / ||∑ĝ_t||
- Converges at optimal rate for convex objectives (up to constant factors)

**Prodigy** (arXiv:2306.06101, ICML 2024):
- Weighted dual averaging improvement on D-Adaptation
- Better convergence by O(√log(D/d₀))
- Recommended optimizer for HuggingFace Diffusers DreamBooth LoRA training

**Adaptation for Cayley-Riemannian:**
- Replace Adam's second moment with the Cayley-Riemannian preconditioner P
- Use ||Pg|| instead of Adam's effective step for gradient accumulation
- D = ||θ - θ₀|| measured in parameter space (A_tilde, B_tilde, S_raw)

**Open question:** Is D well-defined on the Stiefel manifold? The Cayley parameterization maps from unconstrained space (A_tilde, B_tilde) to the Stiefel manifold (A, B), so D measured in the unconstrained parameterization space is well-defined. But does it meaningfully capture distance-to-solution for the actual optimization?

**When to consider:** If MASS ceiling is systematically wrong across model scales — i.e., the static σ_k_min/σ_max bound doesn't adapt to the actual optimization trajectory.

### (D) Spectral-Norm Step Control (Muon-inspired)

**Core idea:** Instead of bounding η, directly bound ||δW||₂ ≤ c × σ_k per step.

**Muon** (Jordan, Bernstein et al., 2024-2025):
- Computes polar factor UV^T of gradient via Newton-Schulz iteration
- Maximizes linearized loss decrease subject to bounded spectral perturbation
- LR has direct geometric meaning: η controls output perturbation bound

**Adaptation for Cayley-Riemannian:**
- After Cayley preconditioning, compute δ(BA) = the resulting change in the adapter's effective weight
- Normalize: η = c × σ_k / ||δ(BA)||₂
- c derivable from Weyl: c = spectral_gap / σ_k (perturbation stays within crossing threshold)

**Computational cost:** Per-step spectral norm of δ(BA). Since BA is rank-r (typically r ≤ 64), the spectral norm can be computed via power iteration on the r×r matrix in O(r² × max(m,n)) — negligible for small rank.

**When to consider:** If per-step SPS gives insufficient control over update spectral norm — i.e., if SPS allows steps that are large in loss-reduction terms but catastrophic in spectral-perturbation terms.

### (C) CDAT-style EMA Smoothing — REJECTED

**Core idea:** Smooth the curvature estimate with EMA (β₂ ≈ 0.99).

**Why rejected:** MASS replaces the measurement entirely. CDAT smooths a noisy measurement of L; MASS doesn't measure L at all. Smoothing noise produces less noisy noise, not signal.

**Documented for completeness.** If future work returns to curvature estimation (e.g., for theoretical analysis), CDAT's EoS-aware framework (Roulet et al., NeurIPS 2024) is the right starting point. Key insight: greedy η = 2/λ_max actually **breaks** Edge-of-Stability dynamics.

---

## 5. Open Mathematical Questions

### Q11.1: Per-layer vs global η

MASS uses global minimums/maximums:
- `eta_ceiling = σ_k_min / σ_max` (min σ_k across layers, max σ_max across layers)
- This is conservative — the least-permissive layer constrains all layers

Per-layer alternative: `eta_ceiling_i = σ_k_i / σ_max_i` per layer.

The Cayley-Riemannian preconditioner already adapts per-layer (P_i = M_i M_i^T). If the preconditioner fully accounts for per-layer geometry, then global η may be sufficient. If not, per-layer η would allow layers with more spectral budget to take larger steps.

### Q11.2: SPS and (L₀,L₁)-relaxed smoothness

Under (L₀,L₁) smoothness (Zhang ICLR 2020), the optimal step size is:

```
η_opt ≤ 1 / (L₀ + L₁ × ||g||)
```

SPS gives:
```
η_sps = f / ||d||²
```

For a quadratic f = (1/2)||g||²/L, SPS gives η = ||g||²/(2L||d||²). If the preconditioner is identity (d = g), this is η = 1/(2L) — the optimal Nesterov step for smooth objectives.

Under (L₀,L₁): Is f/||d||² ≤ 1/(L₀ + L₁||g||)?

This requires f ≤ ||d||²/(L₀ + L₁||g||). Near a quadratic: f ≈ ||g||²/(2L₀), so f/||d||² ≈ ||g||²/(2L₀||d||²). If d = g, this is 1/(2L₀), which is ≤ 1/(L₀ + L₁||g||) only when ||g|| is small. When ||g|| is large, SPS may over-step.

**Implication:** SPS may need the Weyl displacement bound to compensate in early training when gradients are large. This is exactly what MASS provides.

### Q11.3: Convergence of min(ceiling, SPS, Weyl)

Each component converges individually. The min of convergent bounds converges. But the binding component may switch during training:

- Early training (high loss, moderate gradients): SPS likely binds
- Mid training (moderate loss, moderate gradients): SPS or Weyl binds
- Late training (low loss, small gradients): Ceiling likely binds

The question is whether switching between binding components causes oscillatory behavior. Empirically monitor which component is binding at each step.

### Q11.4: Interaction with preconditioner

The Cayley-Riemannian preconditioner transforms the gradient: d = Pg. This changes the effective step size per direction. Does preconditioning make SPS more or less optimal?

For natural gradient (P = Fisher inverse): the preconditioned step d = F⁻¹g moves in the steepest descent direction in KL-divergence geometry. SPS on the preconditioned direction gives η = f/||F⁻¹g||². This is related to the natural gradient step size, but the relationship depends on the curvature of the KL-divergence landscape.

---

## 6. Comparison Table

| Property | MASS | D-Adaptation | Muon-inspired | CDAT |
|----------|------|-------------|---------------|------|
| Step size source | SPS + Weyl bounds | Distance to solution | Spectral norm of update | Smoothed curvature |
| Curvature estimation | None | None | None | Yes (EMA) |
| Hyperparameters | Zero | Zero (but Adam-specific) | c from Weyl | β₂ (unless derived) |
| Per-layer | Global (open Q) | Global | Per-layer natural | Global |
| Convergence guarantee | SPS: yes (convex) | Yes (convex, up to log) | Linearized descent | Yes (EoS-aware) |
| Cayley-Riemannian compat | Native | Needs adaptation | Natural fit | Needs adaptation |
| Computational overhead | Negligible (loss + norm) | Negligible (distance + accum) | Moderate (rank-r SVD) | Same as Lipschitz |
| (L₀,L₁) robustness | Via ||d||² + Weyl floor | Via accumulated norm | Via spectral bound | Via EMA smoothing |
| When to prefer | Default | MASS ceiling wrong at scale | Need per-layer spectral control | Never (use MASS) |

---

## References

- Loizou, N. et al. (2020). "Stochastic Polyak Step-size for SGD." arXiv:2002.10542.
- Zhang, J. et al. (ICLR 2020). "Why Gradient Clipping Accelerates Training." (L₀,L₁)-relaxed smoothness.
- Defazio, A. & Mishchenko, K. (ICML 2023). "Learning-Rate-Free Learning by D-Adaptation." Outstanding Paper.
- Defazio, A. et al. (NeurIPS 2024). "The Road Less Scheduled." Schedule-Free optimization.
- Roulet, V. et al. (NeurIPS 2024). "CDAT: Curvature Dynamics Aware Tuning."
- Jordan, M. & Bernstein, J. et al. (2024-2025). "Muon: An Optimizer for Hidden Layers."
- Weyl, H. (1912). Perturbation bounds for singular values.
- Amari, S. (1998). Natural gradient.
- Nesterov, Y. (2004). Stability bounds for preconditioned gradient descent.

---

*Document created: 2026-02-22*
*Status: MASS implemented. Fallback candidates documented. Open questions in OPEN-MATHEMATICAL-QUESTIONS.md §11.*
