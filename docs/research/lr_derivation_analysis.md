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

### Why the Spectral Ceiling σ_k/σ_max Was Wrong

The ceiling `eta_ceiling = σ_k_min / σ_max` is a **condition ratio** of the base weight matrix — the ratio of the smallest preserved singular value to the spectral norm. This quantity has clear geometric meaning (it bounds how much the adapter can perturb relative to the smallest structure worth preserving), but it is NOT a step size:

- For the 350M model: σ_k_min/σ_max ≈ 0.3-1.0 (depending on layer selection)
- The empirical sweet spot is ≈ 0.003-0.004 (100-300× smaller)
- The ceiling was too permissive by two orders of magnitude

---

## 2. MASS — The Implemented Solution

**MASS (Measured-Adaptive Step Size)** replaces curvature estimation with per-step measurement + geometric bounds.

### Architecture

```
eta_step = min(eta_ceiling, eta_sps, eta_weyl)
```

**Layer 1: Static Weyl Ceiling**
```
eta_ceiling = σ_k_min / σ_max
```
- Derived from Weyl perturbation theory (Weyl 1912)
- Bounds total adapter contribution relative to base model's smallest preserved structure
- Computed once from pre-training SVD
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

## 3. Predicted Step Sizes (MASS Verification)

### Static Ceiling

For the 350M model (LFM2):
- σ_k_min varies by layer (from LayerGeometry)
- σ_max is the global spectral norm

The ceiling η_ceiling is typically in the range 0.3-1.0 — still too permissive on its own, but SPS and Weyl displacement are expected to produce smaller values.

### SPS and Weyl at Typical Training Values

**To verify MASS produces the right step sizes, measure these during the first few training steps:**

1. Record f(x_t) (loss value) at each step
2. Record ||d_t|| (preconditioned gradient norm) at each step
3. Compute η_sps = f/||d||² and η_weyl = σ_k_min/||d||
4. Compare η_step = min(ceiling, sps, weyl) against the empirical sweet spot ≈ 0.003-0.004

**Expected behavior:**
- At step 0 (high loss, moderate gradients): SPS should dominate, giving a moderate η
- As training progresses (loss decreases): SPS decreases, pulling η down
- If gradients spike: both SPS and Weyl displacement decrease, preventing large steps

**Verification protocol:**
```bash
# Run a few steps and inspect EpochMetrics
poetry run mc train run --model /path/to/350M --data /path/to/benchmark --output /tmp/mass-verify --max-iters 5
# Check: eta_step, eta_sps, eta_weyl, eta_ceiling in output
```

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
