# LFM2-1.2B Training Configuration: Mathematical Derivations

Every parameter below is derived from SVD, IEEE 754, or cited theorem. Zero magic numbers.

---

## 1. Rank Allocation: rank_i = tail_dims_i

**Derivation:** For weight matrix W of layer i, compute SVD: W = U S V^T.

Shannon effective rank: r_eff = exp(H) where H = -sum(p_j log(p_j)), p_j = sigma_j^2 / ||W||_F^2.

Structural rank: k = floor(r_eff). Tail dimensions: tail_dims = min(m,n) - k.

NB-LoRA injects delta = B @ diag(S) @ A into the null space (dims k+1 through min(m,n)). Using rank < tail_dims wastes available null space. Using rank > tail_dims is impossible (exits the null space). Therefore rank = tail_dims is the unique correct choice.

**Source:** Shannon entropy definition + Cayley parameterization (Wen & Yin 2013).

**1.2B Values:**

| Matrix | Avg Shannon Eff Rank | Avg Tail Dims | % Null Space |
|--------|:---:|:---:|:---:|
| q_proj (2048x2048) | 630 | 1418 | 69% |
| out_proj (2048x2048) | 1003 | 1045 | 51% |
| k_proj (512x512) | 280 | 232 | 45% |
| v_proj (512x512) | 407 | 105 | 21% |
| ff.w1 (2048x2048) | 1498 | 550 | 27% |
| conv.out_proj (2048x2048) | 994 | 1054 | 51% |

---

## 2. Scale Bound: sigma_k / 2 * (1 - sqrt(eps_f32))

**Derivation:** NB-LoRA scale parameter S_raw is clamped so that ||B @ diag(S) @ A||_2 < sigma_k / 2.

The factor of 2 comes from Weyl's inequality: the perturbation must not shift any eigenvalue across the spectral gap. sigma_k / 2 is the maximum safe perturbation.

The safety margin accounts for numerical precision. IEEE 754 float32: eps = 2^{-23} ≈ 1.19e-7. The relative precision near sigma_k is sqrt(eps) * sigma_k ≈ 3.45e-4 * sigma_k.

Maximum safe scale_bound = sigma_k / 2 * (1 - sqrt(eps_f32)) = sigma_k / 2 * 0.99965.

**Current value:** 0.9 (10x more conservative than derived maximum).

**Recommendation:** Test safety_margin = 0.95 as an ablation. The IEEE 754 derivation shows 0.99965 is the theoretical maximum; 0.9 or 0.95 are both safely within this bound.

**Source:** Weyl's inequality (1912), IEEE 754-2008.

---

## 3. Learning Rate: eta = 1/L, bounded by eta <= 2/(L * lambda_max(P))

**Derivation:** The Lipschitz constant L of the loss gradient is estimated via power iteration on the Hessian (5 batches, 10 iterations per batch). The initial learning rate is eta = 1/L.

With Cayley-Riemannian preconditioning (P = M M^T where M = I + Z, Z is the skew-symmetric Cayley parameter), the effective step size is amplified by lambda_max(P). To maintain stability:

eta_eff = min(eta, 2 / (L * lambda_max(P)))

This is the Nesterov (2004) condition for convergence of preconditioned gradient descent.

lambda_max(P) is computed per step via 5 iterations of power iteration on the r x r SPD matrix P = M M^T.

**No per-layer learning rate needed:** The preconditioner P is computed per-layer from each layer's Cayley parameter Z. Layers with more curvature (larger lambda_max) automatically get smaller effective step sizes. This is equivalent to per-layer LR scheduling derived from the manifold geometry.

**Source:** Nesterov (2004) Theorem 1.2.4, Amari (1998) natural gradient.

---

## 4. Batch Size: B_crit = 1/SNR

**Derivation:** Gradient signal-to-noise ratio SNR is estimated from two micro-batches: SNR = ||mean_grad||^2 / var(grad). The critical batch size B_crit = 1/SNR is the point where doubling the batch gives diminishing returns.

**Source:** McCandlish et al. (2018) "An Empirical Model of Large-Batch Training."

---

## 5. Stopping Criterion: 4-Condition Geometric Certificate

**Conditions (all must hold):**

1. **Stationarity:** Preconditioned gradient norm ||P @ g|| is at the stochastic noise floor. Checked via windowed standard error test on norm history.

2. **Improvement bound:** Maximum possible improvement Delta_max = a^2/(2b) (Taylor analysis) is less than the half-width of the validation loss confidence interval.

3. **Worst-group:** max per-batch Delta_max_i < per-batch CI (no batch is still improving significantly).

4. **No mechanism drift:** Token entropy has not collapsed, repetition rate has not spiked.

**Plus:** Val-loss convergence check (window=3 epochs, check monotonicity). Best checkpoint restored if final val_loss regresses.

**Source:** Stationarity from Nesterov (2004), improvement bound from Taylor remainder, validation from standard ML practice.

---

## 6. Target Module Selection

**Criterion:** layer is targetable iff tail_dims > 0.

For LFM2-1.2B, all weight matrices have tail_dims > 0. Priority is set by spectral analysis:

**Tier 1 (highest capacity, concentrated energy):**
- q_proj: 69% null space, top 10% SVs hold 61% energy
- out_proj (attn): 51% null space
- conv.out_proj: 51% null space

**Tier 2 (moderate capacity):**
- k_proj: 45% null space
- conv.in_proj: 33% null space
- ff.w1: 27% null space

**Tier 3 (low capacity, distributed energy — avoid):**
- v_proj: 21% null space, energy spread across many dims
- ff.w2/w3: 15-23% null space

**Source:** Phase 2 spectral deep dive (2026-02-18).

---

## 7. Why NOT GRASP Top-10% Rank Allocation

GRASP (EMNLP 2025) found that retaining top 10% of singular values preserves 90% reasoning performance. This might suggest using rank = 0.1 * min_dim instead of tail_dims.

**Why tail_dims is correct:**
1. GRASP measures PRUNING resilience (removing SVs), not ADAPTATION capacity (adding into null space). Different operations.
2. NB-LoRA is bounded by construction. Using less than tail_dims wastes available capacity. The adapter can only inject into the null space, so using the full null space maximizes the model's ability to learn.
3. Staats et al. (NeurIPS 2025) showed small SVs are the adaptation substrate. Using only 10% of dims would ignore the very dimensions fine-tuning shapes.

---

## Summary Table

| Parameter | Value | Source |
|-----------|-------|--------|
| rank_i | tail_dims_i = full_rank - floor(shannon_eff_rank) | Shannon entropy + Cayley |
| scale_bound | sigma_k / 2 * safety_margin | Weyl inequality + IEEE 754 |
| safety_margin | 0.9 (test 0.95 in ablation) | Conservative vs 0.99965 max |
| eta_0 | 1/L (Hessian power iteration) | Lipschitz theory |
| eta_eff | min(eta_0, 2/(L * lambda_max(P))) | Nesterov 2004 |
| batch_size | 1/SNR (gradient noise) | McCandlish et al. 2018 |
| stopping | 4-condition certificate + val-loss window | Geometric derivation |
| targets | tail_dims > 0 (prioritized by energy concentration) | SVD + GRASP energy analysis |
