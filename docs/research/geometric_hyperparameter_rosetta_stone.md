# Geometric Hyperparameter Rosetta Stone

**Status**: Reference Document
**Date**: 2026-02-07
**Authors**: Jason Kempf, Claude (Anthropic)

---

## Thesis

There are no "knobs" in LLMs. Temperature, top-p, top-k are noise injection mechanisms that obscure the deterministic geometric structure. The model's weights define a fixed high-dimensional landscape. With greedy decoding (temp=0), every input traces exactly one path through that landscape.

Every training hyperparameter is either:

1. **Derived** from the spectral structure of weight matrices
2. **Derived** from dtype machine precision (float32 constants)
3. **Removed** because the geometric optimizer makes it unnecessary

This document maps each traditional hyperparameter to its geometric replacement.

---

## Master Reference Table

| # | Hyperparameter | Industry Standard | Geometric Replacement | Formula | Status |
|---|---|---|---|---|---|
| | **Optimizer** | | | | |
| 1 | Learning Rate | `1e-4` (grid search) | Measured Lipschitz constant | `η = 1/L` where `L = λ_max(Hessian)` (measured) | Implemented |
| 2 | Adam Epsilon | `1e-8` (never questioned) | Spectral noise floor | `max(sigma_k^2, sqrt(eps) * sigma_max^2)` | Implemented |
| 3 | Adam/Momentum | `0.9 / 0.999` | **ScaledGD** preconditioning | `grad_A @ (BBᵀ+εI)⁻¹`, `(AᵀA+εI)⁻¹ @ grad_B` | Implemented |
| 4 | Weight Decay | `0.01` (uniform) | Condition-aware scaling | `sigma_k / sigma_max` | Implemented |
| 5 | Gradient Clipping | `clip=1.0` | **REMOVED** | ScaledGD + budget monitoring prevent explosion | Removed |
| | **Training Loop** | | | | |
| 6 | Warmup | 5-10% of steps | **REMOVED** | Geometric LR stable from step 0 | Removed |
| 7 | LR Schedule | Cosine decay | **OPTIONAL** | Condition ratio is static; cosine marginal | Optional |
| 8 | Batch Size | "As big as fits" | Gradient noise scale | `B_crit = Var(g) / \|\|E[g]\|\|^2` | Partial |
| 9 | Early Stopping | Val loss patience | Geometric convergence | Loss < sqrt(eps) OR spectral budget > 0.9 | Implemented |
| | **LoRA** | | | | |
| 10 | Scale | `alpha/rank = 2.0` | Spectral bound per-layer | `sigma_k(W) / \|\|BA\|\|_spectral` | Implemented |
| 11 | Rank | `8` (arbitrary) | Null-space capacity | `tail_dims = full_rank - effective_rank` | Implemented |
| 12 | Target Modules | `q_proj + v_proj` | Spectral decay analysis | Layers where `tail_dims > 0` | Implemented |
| 13 | Dropout | `0.1` (arbitrary) | Two spectral ratios | `redundancy * adapter_fraction` | Implemented |
| 14 | Weight Init | Random A, zeros B | Spectral normalized | `\|\|BA\|\|_spectral = sigma_k` from step 0 | Implemented |
| | **Architecture** | | | | |
| 15 | Residual Scaling | `alpha = 1` | Spectral ratio per-layer | `sigma_max(x) / sigma_max(f(x))` | Implemented |

---

## Constants Derived from Machine Precision

All formulas reference constants derived from IEEE 754 float32:

| Constant | Symbol | Value | Derivation | Used By |
|---|---|---|---|---|
| Machine epsilon | `eps` | `2^-23 = 1.19e-7` | Smallest float32 ULP at 1.0 | All formulas |
| Significance threshold | `sqrt(eps)` | `~3.45e-4` | SVD noise floor | Noise floor, sigma_k |
| LR minimum | `eps` | `1.19e-7` | Can't represent smaller changes | LR bounds |
| LR maximum | `1/sqrt(eps)` | `~2896` | Numerical stability ceiling | LR bounds |
| Eigengap threshold | - | `sigma_k/sigma_{k+1} > 2.0` | Meaningful spectral structure | Scale bound refinement |
| Residual alpha range | - | `[sqrt(eps), 1/sqrt(eps)]` | Precision-derived | Residual scaling clamp |

**Code**: `core/domain/training/hyperparameter_validation.py:45-46` (`_EPS`, `_SQRT_EPS`)

---

## Detailed Derivations

### 1. Learning Rate

**Industry**: `1e-4` or `3e-4`, chosen by grid search or "what worked last time."

**Geometric**: `η = 1/L` where `L = λ_max(Hessian)` is MEASURED via power iteration on the Hessian-vector product. This is Nesterov's (2004) optimal step size for gradient descent on a function with L-Lipschitz gradient.

**Mathematical Basis**: The optimal step size for GD on a function with L-Lipschitz gradient is `η = 1/L`, where `L = ||Hessian||_spectral`. This is not a heuristic — it's the step size that achieves the optimal convergence rate `O(1/k)` for convex problems and is locally optimal around critical points for non-convex problems.

**Measurement**: L is measured via power iteration on EVERY training batch, taking the maximum. L is a supremum (worst-case curvature), so max-over-batches gives the true L. Single-batch L can vary 100× due to per-batch curvature differences — this is noise in the measurement apparatus, not a property of the landscape. The all-batch max eliminates this noise.

**No re-measurement during training**: ScaledGD handles curvature changes as adapters grow — the preconditioner `(BBᵀ+εI)⁻¹` shrinks automatically as B grows, which is exactly the rate decrease that Mu & Klabjan (Dec 2025) require. Budget monitoring (Weyl's inequality) catches any spectral bound violation. The initial all-batch measurement + ScaledGD + budget monitoring closes all loops.

**Previous approaches** (superseded): Per-layer LR = `σ_k/σ_max` (condition ratio), `1/σ_max` (inverse spectral norm), `σ_k/σ_max²` — all guesses based on weight spectral norms, which are not the Lipschitz constant of the loss gradient. Single-batch L measurement with 2× drift re-measurement threshold — noise dressed as adaptation.

**Code**: `scripts/validate_geometric_training.py` (`measure_lipschitz_constant`), `hessian_estimator.py:280` (`top_eigenvalue`)

---

### 2. Adam Epsilon

**Industry**: `1e-8`, the default from Kingma & Ba (2014). Never questioned.

**Geometric**: `eps = max(sigma_k^2, sqrt(eps_mach) * sigma_max^2)`

Two floors, take the larger:
- `sigma_k^2`: the noise floor of the weight's eigenspectrum
- `sqrt(eps_mach) * sigma_max^2`: the numerical precision floor

**Mathematical Basis**: Epsilon prevents division by zero in the Adam denominator (`1 / (sqrt(v) + eps)`). It must be large enough for stability but small enough to preserve gradient signal. Both floors come from the weight's spectral structure.

**Code**: `geometric_optimizer.py:129` (`compute_geometric_epsilon`)

---

### 3. Adam / Momentum (Beta1/Beta2) → ScaledGD

**Industry**: `beta1=0.9, beta2=0.999`, empirically chosen by Kingma & Ba (2014).

**Geometric**: **Replaced by ScaledGD** (Tong, Ma, Chi — JMLR 2021). For factored low-rank problems `X = AB`, preconditioning each factor's gradient by the pseudoinverse of the other factor achieves condition-number-free convergence. This is Riemannian gradient descent on the rank-r manifold:

```
grad_A_preconditioned = grad_A @ (B Bᵀ + εI)⁻¹
grad_B_preconditioned = (Aᵀ A + εI)⁻¹ @ grad_B
```

This simultaneously satisfies three proven requirements:
- **LoRA+** (Hayou et al., ICML 2024): A and B provably need different learning rates (`η_B/η_A = Θ(n)`). ScaledGD produces this automatically — the preconditioning by the other factor's spectral structure creates asymmetric effective rates.
- **Mu & Klabjan (Dec 2025)**: Step size must scale as `1/(L × ||adapters||²)`. ScaledGD satisfies this — as one factor grows, the preconditioner shrinks the effective step for the other.
- **Condition-number-free convergence** (Tong et al.): No momentum or adaptive methods needed; the preconditioning normalizes the optimization landscape.

The ε regularization in the inverse uses the geometric epsilon `max(σ_k², √ε_mach × σ_max²)` — the same value computed for numerical stability throughout the pipeline.

**Code**: `scripts/validate_geometric_training.py` (`apply_scaled_gd`)

---

### 4. Weight Decay

**Industry**: `0.01`, applied uniformly to all parameters.

**Geometric**: `decay_scale = sigma_k / sigma_max` (condition ratio). Poorly-conditioned layers (high kappa = sigma_max / sigma_k) get less decay because their small singular values are already near the noise floor. Decaying them further destroys useful signal.

**Code**: `geometric_optimizer.py:157` (`compute_decay_scale`)

---

### 5. Gradient Clipping

**Industry**: `clip=1.0`, from Pascanu et al. (2013). No theoretical basis for the threshold.

**Geometric**: **REMOVED**. ScaledGD preconditioning normalizes gradient scale by construction. The spectral budget monitor (`check_budget_exhausted`) catches any violation and halts training — this is cleaner than clipping because halting respects the bound while clipping is a heuristic correction. Experiments showed 0% clipping events across all layers.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 1

---

### 6. Warmup

**Industry**: Linear warmup for 5-10% of total steps.

**Geometric**: **REMOVED**. Geometric LR is bounded by spectral structure from step 0. The condition ratio `sigma_k/sigma_max` is computed once from base weights and never changes. There is no "cold start" problem because the step size is not learned - it's measured. Ma & Yarats (AAAI 2021) showed warmup compensates for Adam's initial update magnitude starting at exactly alpha. The geometric optimizer has no such initialization artifact.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 2

---

### 7. LR Schedule

**Industry**: Cosine decay to 0 (standard in transformer training).

**Geometric**: **OPTIONAL**. The per-layer condition ratio is static (computed once from base weights), so there's no implicit schedule. Experiments showed cosine decay gave only marginal improvement (0.008 loss) over flat geometric LR. Defazio et al. (NeurIPS 2024) showed schedules are equivalent to iterate averaging. For the geometric optimizer, the static LR works because the condition ratio already captures the correct scale.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 3

---

### 8. Batch Size

**Industry**: "As big as fits in memory."

**Geometric**: `B_crit = Var(g) / ||E[g]||^2` (gradient noise scale). Below B_crit: linear speedup. Above B_crit: diminishing returns. The gradient covariance encodes sample redundancy. Low effective rank of gradient covariance means samples are redundant and larger batches are safe.

**Status**: Partial. Formula proven (McCandlish et al. 2018), not wired to training loop.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 4

---

### 9. Early Stopping

**Industry**: "Stop when validation loss hasn't improved for N epochs" (patience).

**Geometric**: Two criteria, no validation set required:

```
should_stop = loss_stable OR budget_exhausted
```

Where:
- `loss_stable`: Loss change below `sqrt(eps)` (numerical precision floor)
- `budget_exhausted`: Spectral bound ratio > 0.9 (90% of geometric budget consumed, i.e. `||BA||_spectral / sigma_k > 0.9`)

All thresholds are dtype-derived (`sqrt(eps)`) or geometry-derived (spectral bound ratio).

**Deep Dive**: `training_heuristics_analysis.md`, Phase 2b

---

### 10. LoRA Scale

**Industry**: `scale = alpha/rank` (typically alpha=16, rank=8, scale=2.0).

**Geometric**: `scale <= sigma_k(W) / ||B @ A||_spectral` per layer. All 9 tested adapters were 22-2700x over this bound. The standard scale of 2.0 caused catastrophic model degradation (gibberish output, repetitive loops). The geometric scale restored correct reasoning.

**Mathematical Basis**: By Weyl's inequality, `|sigma_i(W') - sigma_i(W)| <= ||scale * Delta||_2`. To preserve W's spectral structure, the perturbation must not exceed `sigma_k`, the smallest significant singular value. The threshold `sqrt(eps) * sigma_max` separates signal from numerical noise. When an eigengap exists (sigma_k / sigma_{k+1} > 2.0), the bound tightens via Davis-Kahan: `scale <= gap / (2 * ||Delta||_2)`.

**Code**: `lora_safety_service.py` (`compute_geometric_scale`, `apply_lora_geometric`)

**Deep Dive**: `lora_spectral_scale_bound.md` (empirics), `lora_spectral_theory.md` (3 theorems)

---

### 11. LoRA Rank

**Industry**: `8` or `16`, chosen arbitrarily.

**Geometric**: `rank = tail_dims = full_rank - effective_rank` per layer. The effective rank is the count of singular values above the noise floor `sqrt(eps) * sigma_max`. The tail dimensions are the null-space capacity where LoRA can add information without interfering with the base model's learned structure. Standard rank-8 is typically under-parameterized by geometry.

Per-layer adaptive rank: each layer gets its own rank based on its spectral structure. Layers with more null space get higher rank.

**Code**: `geometric_lora.py:251` (`compute_per_layer_ranks`), `:230` (`compute_geometric_rank`)

---

### 12. LoRA Target Modules

**Industry**: `q_proj + v_proj` (convention from Hu et al. 2021).

**Geometric**: Target layers where `tail_dims > 0` (non-zero null-space capacity). Spectral decay analysis of LFM2-350M attention:

| Projection | sigma_k | Decay Ratio | Scale Bound |
|---|---|---|---|
| v_proj | 0.46 | 10x | ~0.5 |
| k_proj | 0.30 | 42x | ~0.3 |
| q_proj | 0.005 | 2,810x | ~0.002 |
| o_proj | 0.003 | 2,508x | ~0.002 |

v_proj/k_proj have 100x more room for perturbation than q_proj/o_proj. The standard practice of targeting q_proj + v_proj is geometrically inconsistent.

**Code**: `geometric_lora.py:213` (`select_target_modules`)

**Deep Dive**: `lora_projection_targeting.md`

---

### 13. LoRA Dropout

**Industry**: `0.1`, arbitrary.

**Geometric**: Product of two spectral ratios, no arbitrary constants:

```
dropout = redundancy * adapter_fraction
```

Where:
- `redundancy = 1 - shannon_eff_rank / full_rank` (spectral concentration, 0 = flat spectrum, 1 = single dominant SV)
- `adapter_fraction = rank / full_rank` (how much of the weight's space LoRA occupies)
- `shannon_eff_rank = exp(H(sigma^2))` (Roy & Vetterli 2007)

Self-calibrating: layers with more null-space capacity get both higher rank AND higher dropout. The two ratios multiply to give values that scale with the geometry. NB-LoRA (Cayley transform) uses 0.0 because its spectral norm bound is strictly tighter than dropout's nuclear norm regularization.

Validated on 7 real models across 4 architectures. Dropout ranges from 0.001 (small rank) to 0.11 (large rank layers).

**Code**: `geometric_lora.py:283` (`compute_geometric_dropout`)

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 5

---

### 14. LoRA Weight Initialization

**Industry**: Random A (Gaussian), zeros B (Hu et al. 2021). Product B @ A starts at zero and must "grow into" the budget during training.

**Geometric**: Spectral normalized initialization:

```
||A||_spectral = sqrt(sigma_k)
||B||_spectral = sqrt(sigma_k)
||B @ A||_spectral = sigma_k
```

Uses the full geometric budget from step 0. Each matrix gets `sqrt(sigma_k)` spectral norm so the product respects the bound immediately.

**Code**: `spectral_init.py:162` (`spectral_normalized_lora_init`)

---

### 15. Residual Connection Scaling

**Industry**: `alpha = 1` (no scaling, standard residual `output = x + f(x)`).

**Geometric**: `alpha_i = sigma_max(x) / sigma_max(f(x))` per layer. Normalizes so `||alpha * f(x)|| ~ ||x||`, making residual contributions comparable across layers. Hook-based (non-invasive, no model modification). Clamped to precision-derived range `[sqrt(eps), 1/sqrt(eps)]`. Spectral norms computed via fast power iteration (3 iterations).

**Code**: `residual_scaling.py:184` (`compute_residual_scale`), `:39` (`spectral_norm_power_iteration`)

---

## What Was Removed and Why

| Heuristic | Why Removed / Replaced | Evidence |
|---|---|---|
| Gradient Clipping | ScaledGD normalizes gradient scale; budget monitor halts on violation | 0% clip events across all layers in experiments |
| Warmup | Measured η=1/L is correct from step 0, no cold start | No divergence without warmup; convergence immediate |
| Adam / Momentum | Replaced by ScaledGD (Tong et al. JMLR 2021) | Condition-number-free convergence; automatic asymmetric A/B rates |
| Per-layer LR heuristics | Replaced by measured Lipschitz constant η=1/L | σ_k/σ_max, 1/σ_max, σ_k/σ_max² were guesses, not measurements |

---

## Partially Implemented

| Heuristic | Formula | What Exists | What's Missing |
|---|---|---|---|
| Batch Size | `B_crit = Var(g) / \|\|E[g]\|\|^2` | Formula, estimation function | Not wired to training loop |
| Weight Decay | `sigma_k / sigma_max` | `compute_decay_scale()` in optimizer | Full integration across all training paths |

---

## Cross-Reference Map

| Topic | Document | Content |
|---|---|---|
| LoRA scale empirical validation | `lora_spectral_scale_bound.md` | 9 adapter analysis, GSM8K test case |
| 3 Theorems (Weyl, Davis-Kahan, Sufficiency) | `lora_spectral_theory.md` | Full proofs for necessity, eigengap, sufficiency |
| Rank/target/scale original derivation | `lora_geometric_derivation.md` | Original derivation (superseded by scale bound) |
| Projection targeting | `lora_projection_targeting.md` | q_proj vs v_proj spectral analysis |
| All 5 experiments + Phase 2b | `training_heuristics_analysis.md` | Phase-by-phase results, conclusions |

---

## References

1. Cavazza, J. et al. (2018). Dropout as a Low-Rank Regularizer for Matrix Factorization. AISTATS.
2. Davis, C. & Kahan, W.M. (1970). The Rotation of Eigenvectors by a Perturbation. III. SIAM J. Numer. Anal.
3. Defazio, A. et al. (2024). The Road Less Scheduled. NeurIPS.
4. Golub, G.H. & Van Loan, C.F. (2013). Matrix Computations (4th ed.). Johns Hopkins.
5. Hayou, S. et al. (2024). LoRA+: Efficient Low-Rank Adaptation of Large Models. ICML.
6. Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
7. Kingma, D.P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
8. Ma, J. & Yarats, D. (2021). On the Adequacy of Untuned Warmup for Adaptive Optimization. AAAI.
9. McCandlish, S. et al. (2018). An Empirical Model of Large-Batch Training. arXiv:1812.06162.
10. Miyato, T. et al. (2018). Spectral Normalization for Generative Adversarial Networks. ICLR.
11. Mu, T. & Klabjan, D. (2025). Convergence Analysis of LoRA Fine-Tuning. arXiv (Dec 2025).
12. Nesterov, Y. (2004). Introductory Lectures on Convex Optimization. Springer.
13. Pascanu, R. et al. (2013). On the difficulty of training recurrent neural networks. ICML.
14. Roy, O. & Vetterli, M. (2007). The effective rank: A measure of effective dimensionality. EUSIPCO.
15. Tong, T., Ma, C. & Chi, Y. (2021). Accelerating Ill-Conditioned Low-Rank Matrix Estimation via Scaled Gradient Descent. JMLR.
16. Tran, H. et al. (2025). Spectral Perturbation Bounds Under Eigengap Conditions. arXiv:2510.25670.
17. Wang, X. et al. (2025). NB-LoRA: Norm-Bounded Low-Rank Adaptation. arXiv:2501.19050.
18. Weyl, H. (1912). Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen. Math. Ann.

---

*All code paths relative to `src/modelcypher/`. All formulas verified against source.*
