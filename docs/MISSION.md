# ModelCypher Mission

## Mission Statement

**Train models better than they have ever been trained before, using only geometry.**

Every training decision — learning rate, rank, scale, convergence, batch size, weight decay, initialization, target selection, dropout, stopping — is derived from the spectral structure of weight matrices and the Riemannian geometry of activation manifolds. No grid search. No "what worked last time." No knobs.

Point any model at any dataset. Hit train. Get a LoRA that perfectly captures either the knowledge or the behavioral shapes contained in the data.

Deep-research integration context:
- `docs/research/deep_research_integration_2026_02.md`

## Why Geometry Instead of Standard Practice [PROVEN]

The ML industry is built on a fundamental category error: treating probability as a causal mechanism rather than an epistemic measurement. A forward pass is a deterministic geometric map from input to logits. Softmax normalizes the output for human interpretation. Probability describes uncertainty about outcomes; it does not produce them.

This is not a minor philosophical distinction. It is the foundation on which 8 years of "best practices" were built — learning rate schedules, dropout rates, temperature tuning, gradient clipping thresholds, early stopping patience, warmup periods. All derived from a framework that confuses the observer's description of the system with the system itself.

When you realize the foundation is wrong, you cannot trust the building. Every "standard" technique must be re-derived from the actual mechanism (geometry) or discarded. This is why ModelCypher derives every parameter from SVD, IEEE 754 machine precision, and measured data. Not because we're being difficult. Because the alternative is building on a mistake.

## Bedrock Mandate: Explain Causes, Not Correlates

ModelCypher does not accept correlation as explanation. We dig to bedrock: the causal mechanism and its exact mathematics.

For any claim used in design or decision-making, the required chain is:

1. **Causal operator** — identify the deterministic map that produces the behavior.
2. **Formal derivation** — write the governing equation/theorem.
3. **Direct measurement** — measure variables that test that mechanism, not proxies.
4. **Falsification condition** — define what observation would invalidate the claim.

If any link is missing, the claim is conjectural and cannot be promoted to doctrine or used as a hard threshold.

Probability remains epistemic bookkeeping at readout and evaluation boundaries. It can quantify uncertainty in our observations; it does not cause model trajectories.

## "Fine" Is the Enemy of Correct

A forward pass is a deterministic geometric map. Given parameters and input, there is exactly one output. There is no "fine." There is no "close enough." There is no "reasonable approximation." There is the correct answer, derived from the geometry, or there is a wrong answer you stopped questioning too early.

Every parameter in this system has exactly one correct value, derivable from the spectral structure of the weight matrices, IEEE 754 machine precision, or measured data. When an AI assistant says "this value is fine," it means it imported a guess from its training data instead of deriving the answer from the geometry. That is a bug, not a style preference.

The standard: if you cannot say "this is correct, and here is the derivation," then you do not have an answer yet. Keep looking.

## Canonical Inference Model (Object vs Shadow) [PROVEN]

ModelCypher treats transformer inference as **geometric composition**:

```text
h_0 = Embed(prefix)
h_{l+1} = T_l(h_l)
h_L = (T_{L-1} ∘ ... ∘ T_0)(h_0)
```

Not additive "layer 1 adds info, layer 2 adds more." The same evolving state is
transformed through ordered operators. Order is structural signal:
`T_1(T_2(h)) != T_2(T_1(h))`.

The output distribution is a readout of the terminal state:

```text
logits_t = W_out h_{L,t} + b
p(token_t | prefix) = softmax(logits_t)
```

Interpretation contract:
- **Object (mechanism):** manifold trajectory induced by composed transforms.
- **Shadow (observable):** softmax-ranked token probabilities at the readout layer.

Probability is required at the autoregressive loop boundary (token selection for
the next pass), but it is not the internal mechanism that produces the state
trajectory inside a pass.

### Probability Is Epistemic, Geometry Is Causal

In this project, probability is an observer-side summary, not a causal force.
Nothing inside a transformer forward pass "samples" or "guesses." Given
parameters and an input prefix, the map to logits is deterministic.

```text
h_L = F_theta(prefix)
logits = W_out h_L + b
```

Softmax is a normalization humans apply to interpret relative logit energies.
It does not create model behavior; it reports the result of geometric
transformations already completed by the network.

Cross-entropy is therefore interpreted geometrically:

```text
loss = log(sum_j exp(logit_j)) - logit_correct
```

This is a margin/energy misalignment measurement in readout space. Training
changes weight geometry so trajectories land at higher compatibility with
correct targets. Validation loss measures the same misalignment on held-out
trajectories.

```mermaid
graph LR
    A["Prefix state h0"] --> B["T0"]
    B --> C["T1"]
    C --> D["..."]
    D --> E["TL-1"]
    E --> F["Terminal state hL (object)"]
    F --> G["Readout logits"]
    G --> H["Softmax token distribution (shadow)"]
    H --> I["Decode next token"]
    I --> J["Next pass"]
```

---

## What "Done" Looks Like

ModelCypher is complete when the following command works on any model and any dataset, with zero manual configuration:

```bash
mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
```

And the resulting adapter:

1. **Captures the target knowledge or behavior** — measurable via held-out evaluation
2. **Preserves existing capabilities** — CKA alignment to base model within machine precision on sampled activations
3. **Respects the base model's spectral structure** — per-layer `||BA||_spectral <= sigma_k(W)` by construction
4. **Converges automatically** — stops when the data says to stop, not when a patience counter expires
5. **Required zero human hyperparameter choices** — every number came from the model's own geometry

---

## Guardrails

These conditions must ALL be true for the mission to be considered reached. If any fails, we're not done.

### G1: Zero Arbitrary Hyperparameters

Every training parameter must be derived from one of exactly three sources:

| Source | Examples |
|--------|----------|
| **Spectral structure of W** | sigma_max, sigma_k, effective_rank, tail_dims, spectral_gap |
| **IEEE 754 machine precision** | eps, sqrt(eps), machine_epsilon |
| **Measured from the data** | Per-step loss `f(x_t)`, preconditioned direction norm `||d_t||`, gradient noise scale, loss variance |

**Test**: Grep the training codepath for any literal number that isn't derived from one of these three sources. If you find one, it's a guardrail violation.

**Scope**: This applies to ALL code — production, experiment scripts, research prototypes. A magic number in a script is the same category of bug as a magic number in `train_loop`. "It's just an experiment" is not an exemption. If you cannot derive the threshold, mark it `# TODO: derive from [source]` and do not use it as a decision boundary until derived.

The 15 hyperparameters and their geometric replacements:

| # | Hyperparameter | Geometric Replacement | Formula |
|---|---|---|---|
| 1 | Learning Rate | MASS: Weyl ceiling + SPS + Weyl displacement | `eta_step = min(eta_ceiling, eta_sps, eta_weyl)` where `eta_ceiling = σ_k_min / (σ_max × √N)` (N = batches/epoch, √N = Brownian budget), `eta_sps = f(x_t) / \|\|d_t\|\|²` (Loizou 2020), `eta_weyl = σ_k_min / \|\|d_t\|\|` + val backoff. Replaces broken Lipschitz derivation. Ceiling binding in practice (SPS/Weyl non-binding for fine-tuning). See `docs/research/lr_derivation_analysis.md`. |
| 2 | Adam Epsilon | Spectral noise floor | `max(sigma_k^2, sqrt(eps) * sigma_max^2)` |
| 3 | Adam/Momentum | Cayley-Stiefel retraction | Cayley constraint (Stiefel surface enforcement). Pullback metric `P = (I+Z)(I+Z)^T` removed after falsification (2026-02-23: `P ≈ I`). Benefit is constraint, not curvature. |
| 4 | Weight Decay | Condition-aware scaling | `sigma_k / sigma_max` |
| 5 | Gradient Clipping | REMOVED | MASS step bound + budget monitoring prevent explosion |
| 6 | Warmup | REMOVED | Geometric LR stable from step 0 |
| 7 | LR Schedule | OPTIONAL | MASS ceiling binds throughout training on 350M-1.2B; cosine decay showed no measurable improvement in val loss |
| 8 | Batch Size | Gradient noise scale | `B_crit = Var(g) / ||E[g]||^2` |
| 9 | Early Stopping | Geometric convergence | `loss_stable(SE_diff)` OR `adapter_saturation_exhausted(Weyl)` |
| 10 | LoRA Scale | Spectral bound per-layer | `sigma_k(W) / ||BA||_spectral` |
| 11 | LoRA Rank | Null-space capacity | `tail_dims = full_rank - floor(shannon_effective_rank)` |
| 12 | Target Modules | Spectral decay analysis | Layers where `tail_dims > 0` |
| 13 | Dropout | Two spectral ratios | `redundancy * adapter_fraction` |
| 14 | Weight Init | Spectral normalized | `||BA||_spectral = sigma_k` from step 0 |
| 15 | Residual Scaling | Spectral ratio per-layer | `sigma_max(x) / sigma_max(f(x))` |

### G2: Spectral Safety by Construction

The adapter must NEVER violate the base model's spectral structure. This is not checked after the fact — it is guaranteed by the parameterization.

- **NB-LoRA (Cayley transform)**: `||W_lora||_spectral <= 2 * max(S)` where `S_i <= sigma_k(W_i)`
- **Weyl perturbation monitoring**: Per-layer `||scale * BA||_spectral / sigma_k` tracked every step
- **Budget exhaustion**: Training stops if ANY layer crosses `spectral_gap / (2 * sigma_k)` (Weyl 1912)

**Test**: Run SVD on every trained layer. `max(singular_values(BA)) <= sigma_k(W)` for all layers. No exceptions.

### G3: Data-Derived Convergence

Training stops when the DATA says to stop — not when an arbitrary counter expires.

Three independent stopping criteria (any one triggers):

| Criterion | What It Measures | Threshold |
|-----------|-----------------|-----------|
| **Loss threshold** | Absolute convergence | `loss < sqrt(machine_epsilon)` |
| **Loss stability** | Relative convergence | `|recent_mean - earlier_mean| < SE_diff` where SE_diff is measured from data variance |
| **Adapter saturation** | Spectral safety limit | Any layer's `||BA||_spectral / sigma_k > spectral_gap / (2 * sigma_k)` |

**Test**: The `stop_reason` field in `TrainResult` must be one of `convergence`, `stable_loss`, or `adapter_saturation_exhausted`. Never `max_steps` as the primary design — max_steps exists only as a circuit breaker, not as the intended stopping mechanism.

### G4: Preservation of Existing Capabilities

The adapter must not degrade what the model already knows. Measured, not hoped.

Three independent preservation signals — ALL must hold:

- **CKA alignment**: After merge, CKA between original and merged model on atlas probes >= 1.0 - sqrt(eps)
- **Mode connectivity barrier**: Interpolation path from base to adapted has barrier <= 1.0 + sqrt(eps)
- **Behavioral coherence (degeneration)**: Adapted model's n-gram repetition rate on a fixed probe set must not exceed the base model's measured rate plus sqrt(eps). Threshold is the base model's own repetition envelope, not a constant.

CKA and PPL recovery alone are insufficient. Hard-cutoff quantization correction demonstrated that CKA and PPL can improve while degeneration worsens — removing quantization error with a step-function projection eliminates noise in low-usage directions that acts as implicit regularization. Tikhonov-weighted correction (Marchenko-Pastur noise edge, 2026-02-27) improved all three simultaneously by using continuous eigenvalue weighting instead of a hard cutoff — but degeneration must still be tracked independently because the improvement magnitudes differ (+0.014 CKA mean vs -0.047 degeneration on Qwen3-1.7B). Cross-scale validation on Qwen3-8B confirmed larger gains at scale (+0.033 CKA mean, +0.181 CKA min, -0.04 PPL, -0.016 degeneration). Cross-architecture validation on Llama-3.2-3B confirmed PPL and degeneration improvement (-0.08 PPL, -0.056 degeneration) even when quantization damage is already minimal (baseline CKA 0.992).

**Test**: Run the verification suite on the merged model. CKA, mode connectivity, AND degeneration all within bounds. A model that passes CKA but fails degeneration is not preserved.

### G5: Reproducible Across Models and Datasets

The system works on ANY model architecture and ANY dataset. Not just the ones we tested on.

- **Tested model scales**: 350M, 700M, 1.2B (validated), 8B (mechanically validated: geometry, injection, spectral bounds, stopping — see below for efficacy status)
- **Tested data types**: Logical rules, behavioral patterns, domain knowledge, compositional reasoning
- **Architecture requirement**: Must have extractable weight matrices (attention + MLP projections)
- **No model-specific code in the training loop**: All adaptation flows through the Backend protocol

**Test**: Run the same training pipeline on a model we've never seen before. It should work without code changes. The geometry adapts; the code doesn't.

### G6: Verifiable Quality

Every claim is backed by a measurement. Every measurement has a geometric derivation.

The verification ecosystem must confirm:

| What | How | Module |
|------|-----|--------|
| Spectral safety | Per-layer SVD bounds | `lora_safety_service.py` |
| Knowledge capture | Concept volume separation | `concept_volume_service.py` |
| Behavioral consistency | Mode connectivity barrier | `lora_safety_service.py` |
| Entropy provenance | Baseline verification probe | `baseline_verification_probe.py` |
| Alignment integrity | CKA on atlas probes | `cka.py` |
| Checkpoint integrity | SHA256 checksums | `checkpoint_validation.py` |

### G7: Falsifiability Before Narrative

Broad claims (for example, dimensional tractability or universal stopping
rules) must be evaluated under a pre-registered falsification protocol.
No claim is promoted from observation to doctrine without explicit pass/fail
criteria and reproducible results.

Protocol reference:
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`

---

## Architecture

### One Training Pipeline

One command. One method. Geometry decides everything.

```
Dataset --> SVD(W) --> NB-LoRA (Cayley) --> ScaledGD --> Weyl Budget --> CKA Verify --> Adapter
```

```bash
mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
```

**What happens:**

1. **Geometry analysis** — SVD every weight matrix. Extract σ_max, σ_k, effective_rank, tail_dims, spectral_gap per layer.
2. **Target selection** — Layers where tail_dims > 0 get NB-LoRA. Rank = tail_dims.
3. **Optimizer config** — Per-layer ε = max(σ_k², √ε_mach × σ_max²), decay = σ_k / σ_max, spectral_gap = σ_{k-1} - σ_k.
4. **Base activation snapshot** — Collect per-layer hidden activations on eval probes (for CKA verification).
5. **NB-LoRA injection** — Cayley-parameterized: ||2 B^T diag(S) A||₂ ≤ σ_k by construction.
6. **MASS step size** — Three-layer adaptive: `eta_ceiling = σ_k_min / (σ_max × √N)` (Weyl bound, √N Brownian budget over N batches/epoch), `eta_sps = f(x_t) / ||d_t||²` (Loizou 2020, per-step measured), `eta_weyl = σ_k_min / ||d_t||` (per-step Weyl displacement). Final: `eta_step = min(ceiling, sps, weyl)` + validation-guided backoff.
7. **Training** — Cayley-Stiefel retraction (P removed, 2026-02-23 falsification: `P ≈ I`), MASS step sizing, Weyl budget monitoring per epoch, val loss convergence.
8. **Post-training verification** — Spectral bounds (by construction), CKA alignment to base model.

**Four stopping criteria (any one triggers):**

| Criterion | Source | What It Measures |
|-----------|--------|-----------------|
| Val loss stable | Data | `check_val_loss_converged()` — val loss plateau |
| Val loss increasing | Data | Overfitting detected |
| Budget exhausted | Weyl 1912 | Any layer's ||BA||₂/σ_k > gap/(2σ_k) |
| Max iterations | Circuit breaker | Safety cap only |

### Core Domain Modules (all wired into `mc train run`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `geometric_lora.py` | Weight analysis → LoRA config | `analyze_weight_geometries()`, `select_target_modules()` |
| `geometric_optimizer.py` | Per-layer optimizer params from SVD | `derive_optimizer_geometry_config()` |
| `scaled_gd.py` | Riemannian GD preconditioning | `precondition_lora_gradients()` |
| `spectral_budget.py` | Weyl-derived adapter saturation monitoring | `compute_budget_ratios()`, `is_budget_exhausted()` |
| `geometric_early_stopping.py` | Data-derived convergence detection | `check_loss_stable()`, `check_val_loss_converged()` |
| `cayley_lora.py` | NB-LoRA parameterization | `cayley_transform_full()`, `NBLoRALayer` |
| `cka.py` | Capability preservation verification | `compute_linear_cka_from_activations()` |
| `activation_provider.py` | Activation collection for CKA | `collect_hidden_activations()` |
| `marchenko_pastur.py` | Random-matrix noise edge for eigenvalue thresholding | `marchenko_pastur_noise_edge()`, `effective_dimension()` |

---

## Current State

### Implemented and Validated (all wired into `mc train run`)

- NB-LoRA via Cayley transform — spectral bounds by construction (Wang et al. 2025)
- Cayley-Stiefel retraction — Stiefel constraint via Cayley map (Wen & Yin 2013, Lezcano-Casado 2019). Pullback metric `P = (I+Z)(I+Z)^T` removed after cross-family falsification (2026-02-23: `P ≈ I`, Cohen's d = 0.12 at 200 steps). See `results/weight_geometry/trajectory_falsification_200_multiseed_summary.json`.
- Weyl budget monitoring — capacity usage tracking with `compute_budget_ratios()` (Weyl 1912, Shuttleworth et al. 2024)
- MASS step size — `eta_step = min(eta_ceiling, eta_sps, eta_weyl)`: Weyl ceiling (√N Brownian budget) + SPS (Loizou 2020) + Weyl displacement + val backoff. Replaces broken Lipschitz derivation (HVP spans 3 OOM across minibatches; see `docs/research/lr_derivation_analysis.md`)
- Validation-based stopping — val loss convergence via `check_val_loss_converged()` + best checkpoint restore (validated in 4-arm × 3-seed ablation, 2026-02-17)
- CKA verification — post-training capability preservation check against base model activations (Kornblith et al. 2019)
- Per-layer geometric optimizer config — ε, decay, spectral_gap from SVD
- Zero magic numbers in training codepath (all thresholds from SVD or IEEE 754)
- Training validated on 3 model scales (350M, 700M, 1.2B); 8B mechanically validated (spectral ratio 0.062, stopping catches degradation, geometry analysis + injection confirmed)
- Ablation-validated on 350M (2026-02-17): pure CE + Cayley-Stiefel is optimal; constrained training (invariance, separation, geodesic) monotonically hurts — disabled; available via service API for experiments only
- Backend abstraction (MLX, JAX, CUDA) — framework imports only in backend files
- bf16 SVD guard in `compute_per_layer_signal_ranks` (2026-02-27): activations cast to float32 before SVD; required for 8B bf16 models
- CI-based online eval degradation (2026-02-27): `degraded = degraded_significant` via Clopper-Pearson non-overlap at `alpha = 1/N`. Raw count and significance tracked independently. Replaces single-point comparison that locked in transient valleys.
- Quantization Weyl precheck (2026-02-27): per-layer `||E_q||_spectral >= gap/2` crossing detection before training. Blocks training on quantized models unless `research_allow_quantization_crossing=True`.
- Headroom CI preflight + ceiling override (2026-02-27): when `headroom_upper <= 1/n_total` (baseline at CI ceiling), forces CE-only regime — no REINFORCE, no entropy regularization. Prevents wasted compute at saturated baselines.
- Marchenko-Pastur domain module (2026-02-27): `marchenko_pastur.py` — noise-edge derivation from eigenvalue spectrum + sample ratio. Used by Tikhonov correction and MP-weighted null-space projector. 23 unit tests.
- MP-weighted null-space projector (2026-02-27): `compute_null_space_projector()` in `transplant.py` uses Tikhonov shrinkage weights `w_i = λ_i/(λ_i + α)` with α = MP noise edge, replacing binary eigenvalue mask. Diagnostic on real LFM2-350M activations: 60-67% of eigenvalues fall between IEEE threshold and MP edge in non-bottleneck layers. Shrinkage operator has eigenvalues in [0,1] (monotone, not idempotent). 146 null-space/transplant tests passing.
- MP-weighted null-space projector validated via A/B test (2026-02-28): Tikhonov won all 5 metrics vs binary eigenvalue mask — preserved fraction +35% (0.517 vs 0.384), degeneration 0.088 vs 0.759, PPL 17.93 vs 18.16, CKA 0.9997 vs 0.9996. Binary projector mode removed; Tikhonov is sole mode. `scripts/merge_ab_test.py` retained as validation harness.
- Quantization correction CLI promotion (2026-02-27): `mc quantize correct` promoted with sequential Tikhonov orchestration in `quantization_correction_service.py`. Command is now part of production CLI surface.
- ActivationProviderAdapter delegation fix (2026-02-28): 4 adapter methods (`collect_intermediate_activations`, `collect_probe_activations_batch`, `collect_intermediate_activations_batch`, `collect_gate_activations_batch`) now delegate to backend instead of silently returning hidden activations. Root cause of wrong-dimension intermediate/gate data in profiles. 9 delegation contract tests.
- 82%+ test coverage, 6809 tests passing (2026-02-28)

### Remaining Gaps

| Gap | What's Known | What's Missing | Impact |
|-----|-------------|---------------|--------|
| **8B training efficacy** | Mechanical validation passes (no crash, spectral ratio 0.062, stopping catches degradation). Multi-seed on non-ceiling eval (65% baseline): seed 42 done (3/5 gates: no_crash, spectral_ok, accuracy_ok pass; cka_ok fail min=0.925, degenerate_ok fail), seed 43 running. | Aggregate multi-seed results. Need 2+ seeds to assess variance and determine if CKA/degeneration failures are systematic or seed-dependent. | G5 mechanically closed, efficacy open |
| **Quantization correction ceiling** | Tikhonov closed-form correction validated cross-scale and cross-architecture (2026-02-27). **Qwen3-1.7B**: CKA +0.014/+0.18, PPL -0.06, degen -0.05. **Qwen3-8B**: CKA +0.033/+0.181, PPL -0.04, degen -0.016. **Llama-3.2-3B**: PPL -0.08, degen -0.056, CKA near-flat (baseline already 0.992). Gains scale with quantization damage magnitude. Marchenko-Pastur noise edge (one formula, no sweep). | CKA mean 0.909 (Qwen3-1.7B) still far from 0.9997 guardrail. 8B shows CKA 0.877 post-correction. Llama-3.2-3B shows CKA 0.992 baseline — 4-bit g64 affine causes minimal damage on this architecture. The CKA floor is architecture- and quantization-scheme dependent, not a universal constant. | G4 CKA guardrail not met on 4-bit; ceiling is architecture-dependent |
| **Scale bound tradeoff** | Scale A/B (2026-02-27): standard (1.0) → PPL 3.47, min_CKA 0.65; geometric (sigma_k-derived) → PPL 4.01, min_CKA 0.88. Both preserve spectral bounds. | Neither scale satisfies G4 CKA threshold (0.9997) on 4-bit models. Need to determine if this is a fundamental quantization limitation or a derivation gap. | G4 not achievable on 4-bit without further research |
| **Stopping oscillation sensitivity** | CI-based degradation gate is implemented end-to-end (`degraded = degraded_significant`) with Clopper-Pearson non-overlap at `alpha = 1/N`; raw-vs-significant telemetry is propagated in training + research artifacts. | Multi-seed confirmation that false stop/rollback events are eliminated under transient valleys in 8B non-ceiling runs. | G3 mechanism closed in code, empirical closure pending |
| **Merge pipeline end-to-end A/B closure [CLOSED 2026-02-28]** | Tikhonov won all 5 metrics: preserved fraction +35%, degeneration 12× better, PPL and CKA both improved. Binary projector mode removed. Pipeline dimension guard added for profiles with hidden-dim fallback intermediates. ActivationProviderAdapter root cause fixed. | None — closed. | Closed |

### What Closes the Gaps

1. **8B efficacy separation**: Build a fixed non-ceiling online-eval set (`scripts/g5_build_non_ceiling_eval_set.py`), then run 3 seeded validations with FP reference required (`scripts/g5_8b_multiseed_closure.py`). Current artifact: `results/g5_8b_validation/non_ceiling_eval_set_8b.json` (`13/20 = 65%`, generated 2026-02-27).
2. **Quantization correction cross-scale validation [CLOSED 2026-02-27]**: Validated on Qwen3-8B (CKA +0.033/+0.181, PPL -0.04, degen -0.016) and cross-architecture on Llama-3.2-3B (PPL -0.08, degen -0.056). Gains are proportional to quantization damage: 8B (larger damage → larger correction), Llama (minimal damage → minimal CKA change but PPL/degen still improve). CKA ceiling confirmed architecture-dependent.
3. **Quantization frontier mapping**: Join Weyl crossing severity with observed CKA floors and Tikhonov correction ceilings across bit-depths (4-bit, 8-bit) using non-crossing fraction, `max(error/(gap/2))`, and `min_cka`.
4. **Merge projector closure [CLOSED 2026-02-28]**: A/B test completed — Tikhonov dominated all 5 metrics. Binary projector mode retired. ActivationProviderAdapter delegation fixed (root cause of wrong-dimension intermediate/gate profiles).
5. **Significance-based stopping**: Implemented. Remaining work is empirical closure in multi-seed 8B non-ceiling runs.

---

## What This Is NOT

- **Not a training framework.** We don't compete with PyTorch Lightning or HuggingFace Trainer. We replace the decision-making, not the infrastructure.
- **Not approximate.** When we say `sigma_k`, we mean the actual singular value from SVD, not an estimate. When we say `machine_epsilon`, we mean `2^-23` for float32. Precision is the point.
- **Not a black box.** Every number in the training run has a derivation. Every derivation has a mathematical reference. Every reference is a published theorem or a measured quantity. If you can't trace a number back to SVD or IEEE 754, it's a bug.

---

## References

| Paper | What We Use |
|-------|-------------|
| Amari (1998) | Natural gradient: G^{-1} @ grad for Riemannian optimization |
| Nesterov (2004) | Stability bound: eta ≤ 2/(L * lambda_max(P)) for preconditioned GD |
| Wen & Yin (2013) | Cayley retraction on Stiefel manifold; feasible orthogonality-constrained optimization |
| Li, Fuxin, Todorovic (ICLR 2020) | Cayley SGD with convergence proof on Stiefel manifold |
| Lezcano-Casado (NeurIPS 2019) | Trivializations: Euclidean GD on phi(theta) = Riemannian GD with pullback metric |
| Tong et al. (JMLR 2021) | ScaledGD: condition-number-free convergence for unconstrained low-rank (not Stiefel) |
| Hayou et al. (ICML 2024) | Asymmetric LoRA convergence rates |
| Wang et al. (2025, arXiv:2501.19050) | NB-LoRA: Cayley parameterization for norm-bounded LoRA |
| Weyl (1912) | Perturbation bounds: \|sigma_i(W+E) - sigma_i(W)\| <= \|\|E\|\|_2 |
| Shuttleworth et al. (2024, arXiv:2410.21228) | Empirical confirmation of Weyl bounds for LoRA |
| Roy & Vetterli (2007) | Shannon effective rank: exp(H(sigma^2)) |
| de Silva & Tenenbaum (2004) | Landmark MDS for cross-manifold projection |
| Eckart-Young (1936) | Optimal low-rank approximation via SVD |
| Kornblith et al. (2019) | CKA: Centered Kernel Alignment for representation similarity |
| Marchenko & Pastur (1967) | Limiting spectral distribution of random matrices; noise edge `σ²(1+√(D/N))²` for eigenvalue thresholding |
| Loizou et al. (2020) | Stochastic Polyak Step-size (SPS): `η = f(x_t) / ||g_t||²` |
