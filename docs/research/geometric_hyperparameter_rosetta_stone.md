# Geometric Hyperparameter Rosetta Stone `[EMPIRICAL]`

**Status**: Reference Document
**Date**: 2026-02-07
**Authors**: Jason Kempf, Claude (Anthropic)

> Historical note (2026-02-22):
> This document captures a pre-MASS training-era synthesis.
> Any `eta = 1/L` language is historical and superseded.
> Active LR control is MASS (`eta_step = min(eta_ceiling, eta_sps, eta_weyl)`).
> Canonical LR history: `docs/research/lr_derivation_analysis.md`.
> Integration log: `docs/research/deep_research_integration_2026_02.md`.

---

## Thesis

There are no "knobs" in LLMs. Temperature, top-p, top-k are noise injection mechanisms that obscure the deterministic geometric structure. The model's weights define a fixed high-dimensional landscape. With greedy decoding (temp=0), every input traces exactly one path through that landscape.

Every training-control row is now labeled by evidence type:

1. **derived**: the cited mechanism yields the formula directly
2. **adopted**: the row uses a mainstream external method as-is
3. **convention**: the row uses a precision or engineering convention, not a
   validated behavioral theorem

Runtime status lives in
[`15-HYPERPARAMETER-RESEARCH-PROGRAM.md`](15-HYPERPARAMETER-RESEARCH-PROGRAM.md).
This document maps the historical formula claims and keeps the citation labels
honest.

---

## Master Reference Table

| # | Hyperparameter | Industry Standard | Replacement / Current Truth | Formula | Evidence label |
|---|---|---|---|---|---|
| | **Optimizer** | | | | |
| 1 | Learning Rate | `1e-4` (grid search) | MASS step-size controller on research modes; default path is calibrated AdamW `2e-4` cosine | `eta_step = min(eta_ceiling, eta_sps, eta_weyl)` | derived |
| 2 | Adam Epsilon | `1e-8` (never questioned) | Precision floor formula exists but has unresolved Adam-units mismatch | `max(sigma_k^2, sqrt(eps) * sigma_max^2)` | convention |
| 3 | Adam/Momentum | `0.9 / 0.999` | Default path adopts AdamW betas; research modes use Fisher/MASS moment logic | AdamW betas or Fisher/MASS state | adopted |
| 4 | Weight Decay | `0.01` (uniform) | Condition-aware formula exists; default `mc train run` passes `weight_decay=0.0` | `sigma_k / sigma_max` | derived |
| 5 | Gradient Clipping | `clip=1.0` | Removed from canonical path; MASS research modes bound updates | controller-bound displacement | derived |
| | **Training Loop** | | | | |
| 6 | Warmup | 5-10% of steps | No warmup flag on the canonical path; MASS research modes rely on measured ceilings | no separate warmup schedule | derived |
| 7 | LR Schedule | Cosine decay | Default path uses calibrated cosine; schedule-free framing is external work | Defazio/Schedule-Free equivalence, not a ModelCypher derivation | adopted |
| 8 | Batch Size | "As big as fits" | Gradient-noise scale is wired for logical batch sizing | `B_crit = Var(g) / ||E[g]||^2` | adopted |
| 9 | Early Stopping | Val loss patience | Certificate and loss windows are wired; sqrt-eps behavioral thresholds are under review | measured validation windows plus precision conventions | convention |
| | **LoRA** | | | | |
| 10 | Scale | `alpha/rank = 2.0` | Weyl structural bound is correct; sigma_k budget is diagnostic-only for behavioral damage per R2 | `sigma_k(W) / ||BA||_2` structural bound | derived |
| 11 | Rank | `8` (arbitrary) | Null-space capacity from Shannon effective rank | `tail_dims = full_rank - floor(shannon_effective_rank)` | derived |
| 12 | Target Modules | `q_proj + v_proj` | Spectral decay analysis selects target surface | layers where `tail_dims > 0` | derived |
| 13 | Dropout | `0.1` (arbitrary) | Formula exists for config payloads but shipped adapter does not apply it | `redundancy * adapter_fraction` | convention |
| 14 | Weight Init | Random A, zeros B | Runtime default is PiSSA; spectral-normalized init is historical/unwired | PiSSA or historical `||BA||_2 = sigma_k` | adopted |
| | **Architecture** | | | | |
| 15 | Residual Scaling | `alpha = 1` | Formula exists as dead code; no shipped training-path consumer | `sigma_max(x) / sigma_max(f(x))` | derived |

---

## Precision Conventions `[CONVENTION]`

Several formulas reference IEEE 754 float32 constants. These are valid machine
precision quantities, but using them as behavioral thresholds is a convention
unless the measurement operator's sampling noise has been bounded below that
scale.

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

### 1. Learning Rate (Historical Path Superseded by MASS) `[DISPROVEN]`

**Industry**: `1e-4` or `3e-4`, chosen by grid search or "what worked last time."

**Historical geometric path**: `η = 1/L` where `L = λ_max(Hessian)` is measured
via power iteration on Hessian-vector products.

**Why superseded**: This path is brittle under stochastic nonsmooth training and
is no longer the active controller in ModelCypher.

**Active replacement**: MASS (Weyl ceiling + SPS + Weyl displacement bound).

See `docs/research/lr_derivation_analysis.md` for ablations and derivation
history.

**Previous approaches** (superseded): Per-layer LR = `σ_k/σ_max` (condition ratio), `1/σ_max` (inverse spectral norm), `σ_k/σ_max²` — all guesses based on weight spectral norms, which are not the Lipschitz constant of the loss gradient. Single-batch L measurement with 2× drift re-measurement threshold — noise dressed as adaptation.

**Code**: `scripts/validate_geometric_training.py` (`measure_lipschitz_constant`), `hessian_estimator.py:280` (`top_eigenvalue`)

---

### 2. Adam Epsilon `[CONVENTION]`

**Industry**: `1e-8`, the default from Kingma & Ba (2014). Never questioned.

**Historical formula**: `eps = max(sigma_k^2, sqrt(eps_mach) * sigma_max^2)`

Two floors, take the larger:
- `sigma_k^2`: the noise floor of the weight's eigenspectrum
- `sqrt(eps_mach) * sigma_max^2`: the numerical precision floor

**Current status**: This is not a shipped optimizer claim. Adam epsilon is added
to `sqrt(v_t)`, so its units must match the measured second-moment state. The
weight-singular-value-squared formula is a precision convention until that unit
chain is derived and wired.

**Code**: `geometric_optimizer.py:129` (`compute_geometric_epsilon`)

---

### 3. Adam / Momentum (Beta1/Beta2) -> ScaledGD (Superseded)

> **Superseded (2026-02-23):** For NB-LoRA, Cayley-Stiefel retraction replaced ScaledGD. The Cayley constraint enforces orthogonality on NB-LoRA factors directly, making preconditioning unnecessary (weight space is Euclidean — P = MM^T ≈ I, Fisher degenerate). ScaledGD remains mathematically valid for standard LoRA but is not used in the active training pipeline. See `geometric_optimizer.py` docstring.

**Industry**: `beta1=0.9, beta2=0.999`, empirically chosen by Kingma & Ba (2014).

**Historical geometric path**: **ScaledGD** (Tong, Ma, Chi — JMLR 2021). For factored low-rank problems `X = AB`, preconditioning each factor's gradient by the pseudoinverse of the other factor achieves condition-number-free convergence. This is Riemannian gradient descent on the rank-r manifold:

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

### 4. Weight Decay `[DERIVED-FORMULA-UNWIRED]`

**Industry**: `0.01`, applied uniformly to all parameters.

**Geometric formula**: `decay_scale = sigma_k / sigma_max` (condition ratio).
Poorly-conditioned layers (high kappa = sigma_max / sigma_k) get less decay
under the formula.

**Current status**: The formula exists, but the canonical `mc train run`
runtime passes `weight_decay=0.0` by default. Do not cite this row as a shipped
replacement until it is wired and benchmarked.

**Code**: `geometric_optimizer.py:157` (`compute_decay_scale`)

---

### 5. Gradient Clipping `[DERIVED-RESEARCH-MODE]`

**Industry**: `clip=1.0`, from Pascanu et al. (2013). No theoretical basis for the threshold.

**Geometric**: **Removed from the canonical path**. MASS research modes bound
updates through controller terms rather than clipping a post-hoc gradient norm.
The old "0% clipping events" statement is historical experiment context, not a
current public efficacy claim.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 1

---

### 6. Warmup `[DERIVED-RESEARCH-MODE]`

**Industry**: Linear warmup for 5-10% of total steps.

**Geometric**: **Removed as a separate control** on MASS research modes because
the step ceiling is measured before use. The canonical path currently uses
calibrated AdamW/cosine from step 0. This row is not evidence that the default
optimizer is fully hyperparameter-free.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 2

---

### 7. LR Schedule `[ADOPTED]`

**Industry**: Cosine decay to 0 (standard in transformer training).

**Adopted reference**: Defazio et al. (NeurIPS 2024) is an external
schedule-free/iterate-averaging result. It does not by itself prove the
ModelCypher runtime can remove schedules.

**Current status**: The canonical path uses calibrated cosine decay. MASS
research modes avoid a separate schedule by choosing `eta_step` online.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 3

---

### 8. Batch Size `[ADOPTED]`

**Industry**: "As big as fits in memory."

**Geometric**: `B_crit = Var(g) / ||E[g]||^2` (gradient noise scale). Below B_crit: linear speedup. Above B_crit: diminishing returns. The gradient covariance encodes sample redundancy. Low effective rank of gradient covariance means samples are redundant and larger batches are safe.

**Status**: The gradient-noise scale model is adopted from McCandlish et al.
(2018) and is wired for logical batch sizing. Promotion still depends on
showing that the measured critical batch predicts useful training behavior in
the closure benchmark.

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 4

---

### 9. Early Stopping `[CONVENTION-UNDER-REVIEW]`

**Industry**: "Stop when validation loss hasn't improved for N epochs" (patience).

**Geometric**: Two criteria, no validation set required:

```
should_stop = loss_stable OR budget_exhausted
```

Where:
- `loss_stable`: measured validation-loss windows
- `budget_exhausted`: spectral budget telemetry

**Current status**: Any use of `sqrt(eps)` on sampled behavioral quantities is
a precision convention until baseline sampling variance is measured and shown
to be below that threshold. Do not promote dtype thresholds as behavioral
stopping theorems.

**Deep Dive**: `training_heuristics_analysis.md`, Phase 2b

---

### 10. LoRA Scale `[DERIVED-STRUCTURAL]`

**Industry**: `scale = alpha/rank` (typically alpha=16, rank=8, scale=2.0).

**Geometric**: `scale <= sigma_k(W) / ||B @ A||_spectral` per layer is the
structural Weyl budget used by `mc analyze lora-svd`.

**Mathematical basis**: By Weyl's inequality,
`|sigma_i(W') - sigma_i(W)| <= ||scale * Delta||_2`. To preserve W's
structural spectrum, the perturbation must remain below the chosen structural
budget. For crossing at an eigengap, the no-crossing condition is
`||E||_2 < gap_k / 2`.

**R2 finding**: The retained R2 report downgraded the structural `sigma_k`
budget to **diagnostic-only** for behavioral damage. It is a correct structural
measurement; it does not currently predict behavioral damage by itself.

**Code**: `lora_safety_service.py` (`compute_geometric_scale`, `apply_lora_geometric`)

**Deep Dive**: `lora_spectral_scale_bound.md` (empirics), `lora_spectral_theory.md` (3 theorems)

---

### 11. LoRA Rank `[EMPIRICAL]`

**Industry**: `8` or `16`, chosen arbitrarily.

**Geometric**: `rank = tail_dims = full_rank - floor(shannon_effective_rank)` per layer. The Shannon effective rank captures structural spectral utilization, while precision rank (`max(m,n) * eps * sigma_max`) is a secondary numerical diagnostic. The tail dimensions are the null-space capacity where LoRA can add information without interfering with the base model's learned structure. Standard rank-8 is typically under-parameterized by geometry.

Per-layer adaptive rank: each layer gets its own rank based on its spectral structure. Layers with more null space get higher rank.

**Code**: `geometric_lora.py:251` (`compute_per_layer_ranks`), `:230` (`compute_geometric_rank`)

---

### 12. LoRA Target Modules `[EMPIRICAL]`

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

### 13. LoRA Dropout `[CONVENTION-UNWIRED]`

**Industry**: `0.1`, arbitrary.

**Historical formula**: Product of two spectral ratios:

```
dropout = redundancy * adapter_fraction
```

Where:
- `redundancy = 1 - shannon_eff_rank / full_rank` (spectral concentration, 0 = flat spectrum, 1 = single dominant SV)
- `adapter_fraction = rank / full_rank` (how much of the weight's space LoRA occupies)
- `shannon_eff_rank = exp(H(sigma^2))` (Roy & Vetterli 2007)

Current status: this formula can appear in generated config payloads, but the
shipped training adapter does not apply it as runtime dropout. Treat the row as
unwired, not validated.

**Code**: `geometric_lora.py:283` (`compute_geometric_dropout`)

**Deep Dive**: `training_heuristics_analysis.md`, Experiment 5

---

### 14. LoRA Weight Initialization `[ADOPTED-DEFAULT]`

**Industry**: Random A (Gaussian), zeros B (Hu et al. 2021). Product B @ A starts at zero and must "grow into" the budget during training.

**Historical geometric formula**: Spectral normalized initialization:

```
||A||_spectral = sqrt(sigma_k)
||B||_spectral = sqrt(sigma_k)
||B @ A||_spectral = sigma_k
```

**Current status**: The runtime default is PiSSA. The spectral-normalized
`sigma_k` initialization is not the shipped default and should not be described
as the active weight-init derivation.

**Code**: `spectral_init.py:162` (`spectral_normalized_lora_init`)

---

### 15. Residual Connection Scaling `[DERIVED-FORMULA-DEAD-CODE]`

**Industry**: `alpha = 1` (no scaling, standard residual `output = x + f(x)`).

**Geometric formula**: `alpha_i = sigma_max(x) / sigma_max(f(x))` per layer.
The standalone formula exists, but no shipped training path consumes it.

**Code**: `residual_scaling.py:184` (`compute_residual_scale`), `:39` (`spectral_norm_power_iteration`)

---

## What Was Removed and Why

| Heuristic | Why Removed / Replaced | Evidence |
|---|---|---|
| Gradient Clipping | Removed from the canonical path; MASS research modes bound updates | Historical experiment note only; not a current public efficacy claim |
| Warmup | No separate warmup control on the canonical CLI | Default still uses calibrated AdamW/cosine, so this is not a fully replaced knob |
| Adam / Momentum | Default adopts AdamW betas; research modes use Fisher/MASS state | Promotion requires closure benchmark evidence |
| Per-layer LR heuristics | Replaced in research modes by MASS controller + Weyl bounds | `sigma_k/sigma_max`, `1/sigma_max`, `sigma_k/sigma_max^2` were guesses, not measured controls |

---

## Unwired Or Partially Implemented

| Heuristic | Formula | What Exists | What's Missing |
|---|---|---|---|
| Adam epsilon | `max(sigma_k^2, sqrt(eps) * sigma_max^2)` | `compute_geometric_epsilon()` | Unit-consistent Adam `v_t` derivation and runtime wiring |
| Weight Decay | `sigma_k / sigma_max` | `compute_decay_scale()` | Canonical runtime wiring and closure benchmark evidence |
| Dropout | `redundancy * adapter_fraction` | `compute_geometric_dropout()` | Adapter-runtime dropout application and validation |
| Residual scaling | `sigma_max(x) / sigma_max(f(x))` | `residual_scaling.py` | A shipped consumer or deletion |

---

## Cross-Reference Map

| Topic | Document | Content |
|---|---|---|
| LoRA scale structural diagnostic | `lora_spectral_scale_bound.md` | Adapter spectral analysis; behavioral prediction downgraded by R2 |
| 3 Theorems (Weyl, Weyl no-crossing, Sufficiency) | `lora_spectral_theory.md` | Full proofs for necessity, no-crossing, sufficiency |
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

*All code paths relative to `src/modelcypher/`. Runtime status must be checked
against the generated matrix before promoting any row.*
