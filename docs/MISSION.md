# ModelCypher Mission

## Mission Statement

**Demonstrate, with measurements, when geometry-derived training beats standard practice.**

Every training decision — learning rate, rank, scale, convergence, batch size, weight decay, initialization, target selection, dropout, stopping — is derived from the spectral structure of weight matrices and the Riemannian geometry of activation manifolds. No grid search. No "what worked last time." No knobs.

End-state target, not current claim: point a model at a dataset, hit train, and
get a LoRA that captures the target structure while preserving the base model.

## Current Evidence State (2026-03-11)

- `mc train run` is the canonical shipped training surface, and its runtime path
  is geometry-derived.
- `results/pipeline_validation/verdict.json` still reports structural pass
  without full inference closure: `all_structural_pass = true`,
  `all_inference_pass = false`, `all_pass = false`.
- `results/nblora_vs_standard/` is retained as `summary_only`; the kept
  single-seed LFM2-350M summary does not yet support a promotable "better than
  standard practice" claim.
- `results/g5_8b_validation_multiseed/multiseed_gates.json` shows 8B
  mechanical viability, but `cka_ok` and `degenerate_ok` remain open.
- `results/quantization_frontier/20260227T235714Z/quantization_frontier.json`
  shows encouraging correction measurements, but not the frontier law required
  for mission closure.

## Precision Objective (2026-03-05)

Full precision was the microscope phase. Quantization is the deployment phase.

- `bf16/fp16` remains the derivation environment for mechanism discovery and falsification.
- Quantized models are the practical objective because they are the only viable path on constrained hardware and they deliver materially higher throughput.
- Mission success requires smaller-and-smarter outcomes: preserving or improving behavior while reducing precision and resource footprint.
- "Works in bf16" is necessary but insufficient. A training or merge method is promotable only when its quantized behavior is measured, explained, and controllable.

Quantization in this project is treated as a deterministic geometric perturbation operator. If behavior changes under quantization, the response is mechanism tracing, not heuristic patching.

Deep-research integration context:
- `docs/research/deep_research_integration_2026_02.md`

## Why Geometry Instead of Standard Practice

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

## Bedrock Enforcement Upgrade (2026-03-03)

Mixed "confirmed on one model, refuted on another" is not a valid endpoint. It is a signal that the mechanism is underspecified.

From this point forward, no claim is promotable unless it includes all of:

1. **Architecture term** — explicit dependence on architecture variables, not an implicit "all models" assumption.
2. **Scale term** — explicit dependence on model size/depth/dimension where relevant.
3. **Commensurability proof** — proof that the measurement is comparable across the compared objects.
4. **Directional prediction** — sign/magnitude expectation from the derivation before running experiments.
5. **Falsifier** — exact observation that invalidates the mechanism.

If a result differs across models and those terms were omitted, the result is classified as **measurement-design failure**, not "partial confirmation."

## Prediction Contract (Required Before Any New Experiment)

Every pre-registered prediction must include this equation-level contract:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

With explicit declarations:
- `geometry_state`: the causal geometric variables
- `architecture_state`: layer/operator type, routing pattern, attention regime, etc.
- `scale_state`: width/depth/parameter count and derived dimensional terms
- `precision_state`: numeric representation and quantization operator parameters
- `measurement_operator`: kernel/normalization/statistic with domain of validity

No experiment starts until this contract is written in the experiment doc.

## Documentation Contract (Required Before Any Claim Promotion)

No claim can be promoted into mission/vision/roadmap language unless it passes
`docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`.

This applies to all promoted statements, not just new experiments.
Retroactive doctrine cleanup is mandatory whenever a claim is found to lack:
1. architecture terms,
2. scale terms,
3. measurement commensurability proof,
4. explicit falsifier.

## Scope Cascade

These four documents do different jobs and must stay aligned:

- **Mission**: what must become promotably true in the canonical geometric
  engine
- **Vision**: what mission success may later enable
- **Roadmap**: the closure order from current evidence to promotable claims
- **Open Questions**: only the mathematical blockers on that closure order

Mission accounting is intentionally narrow:

- `mc train run` is the only clearly shipped canonical surface counted toward
  mission closure today
- merge, continual learning, stacking, and sovereignty remain downstream,
  experimental, or partial until their own certificates close

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

`mc train run` now includes a hard promotability gate (`pipeline_gate_v1`): strict runs fail
if measured geometric invariants are violated or core invariants are unresolved.

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
| 3 | Adam/Momentum | Diagonal Fisher preconditioner + Cayley-Stiefel | Per-parameter `v_t = EMA(g²)`, `d_t = m̂_t/(√v̂_t + ε)`. β₁ derived from half-epoch window ∩ precision ceiling (`derive_beta1()`). β₂ = 0.999 (IEEE 754: EMA error < √ε after 119+ steps). Cayley retraction enforces Stiefel constraint. P removed (2026-02-23: `P ≈ I`). |
| 4 | Weight Decay | Condition-aware scaling | `sigma_k / sigma_max` |
| 5 | Gradient Clipping | REMOVED | MASS step bound + budget monitoring prevent explosion |
| 6 | Warmup | REMOVED | Geometric LR stable from step 0 |
| 7 | LR Schedule | OPTIONAL | MASS ceiling binds throughout training on 350M-1.2B; cosine decay showed no measurable improvement in val loss |
| 8 | Batch Size | Gradient noise scale | `B_crit = Var(g) / ||E[g]||^2` |
| 9 | Early Stopping | Geometric convergence | `loss_stable(SE_diff)` OR `adapter_saturation_exhausted(Weyl)` |
| 10 | LoRA Scale | Spectral bound per-layer | `sigma_max(W) / 2 × (1 - √ε)`. Allows adapter to perturb at weight scale. Per-step displacement bounded by MASS (`η_weyl = σ_k/\|\|d\|\|`). |
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

Four independent stopping criteria (any one triggers):

| Criterion | What It Measures | Threshold |
|-----------|-----------------|-----------|
| **Loss threshold** | Absolute convergence | `loss < sqrt(machine_epsilon)` |
| **Loss stability** | Relative convergence | `|recent_mean - earlier_mean| < SE_diff` where SE_diff is measured from data variance |
| **Adapter saturation** | Spectral safety limit | Any layer's `||BA||_spectral / sigma_k > spectral_gap / (2 * sigma_k)` |
| **Geometric certificate** | No measurable local improvement | 5 conditions: stationarity (grad norm converged to stochastic floor), improvement bound < CI, worst-group, no drift, task improvement. Gated by `should_certificate_stop()` — suppressed while val_loss is still improving (prevents stochastic false positives from gradient alignment flips). |

**Test**: The `stop_reason` field in `TrainResult` must be one of `convergence`, `stable_loss`, `adapter_saturation_exhausted`, or `certificate`. Never `max_steps` as the primary design — max_steps exists only as a circuit breaker, not as the intended stopping mechanism.

### G4: Preservation of Existing Capabilities

The adapter must not degrade what the model already knows. Measured, not hoped.

Three independent preservation signals — ALL must hold:

- **CKA alignment**: After merge, CKA between original and merged model on atlas probes >= 1.0 - sqrt(eps)
- **Mode connectivity barrier**: Interpolation path from base to adapted has barrier <= 1.0 + sqrt(eps)
- **Behavioral coherence (degeneration)**: Adapted model's n-gram repetition rate on a fixed probe set must not exceed the base model's measured rate plus sqrt(eps). Threshold is the base model's own repetition envelope, not a constant.

CKA and PPL recovery alone are insufficient. Hard-cutoff quantization correction demonstrated that CKA and PPL can improve while degeneration worsens — removing quantization error with a step-function projection eliminates noise in low-usage directions that acts as implicit regularization. Tikhonov-weighted correction (Marchenko-Pastur noise edge, 2026-02-27) improved all three simultaneously by using continuous eigenvalue weighting instead of a hard cutoff — but degeneration must still be tracked independently because the improvement magnitudes differ (+0.014 CKA mean vs -0.047 degeneration on Qwen3-1.7B). Cross-scale validation on Qwen3-8B confirmed larger gains at scale (+0.033 CKA mean, +0.181 CKA min, -0.04 PPL, -0.016 degeneration). Cross-architecture validation on Llama-3.2-3B confirmed PPL and degeneration improvement (-0.08 PPL, -0.056 degeneration) even when quantization damage is already minimal (baseline CKA 0.992).

**Test**: Run the verification suite on the merged model. CKA, mode connectivity, AND degeneration all within bounds. A model that passes CKA but fails degeneration is not preserved.

### G5: Reproducible Across Models and Datasets

Mission closure requires the system to work on new model architectures and data
regimes, not just the ones already tested.

- **Current retained state**: small-model and mid-scale evidence exists, but the
  promotable baseline suite is still open and 8B efficacy closure is still open
  even though mechanical viability exists
- **Tested model scales**: 350M, 700M, 1.2B on retained smaller-scale surfaces;
  8B is mechanically viable (geometry, injection, spectral bounds, stopping)
  but still open on efficacy
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
Dataset --> SVD(W) --> NB-LoRA (Cayley) --> Fisher-preconditioned GD --> MASS --> Weyl Budget --> Certificate --> CKA Verify --> Adapter
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
7. **Training** — Diagonal Fisher preconditioner (`d_t = m̂/(√v̂ + ε)`, β₁ from half-epoch window, β₂ = 0.999), Cayley-Stiefel retraction (P removed, 2026-02-23: `P ≈ I`), MASS step sizing, Weyl budget monitoring per epoch, geometric certificate + val loss convergence.
8. **Post-training verification** — Spectral bounds (by construction), CKA alignment to base model.

**Five stopping criteria (any one triggers):**

| Criterion | Source | What It Measures |
|-----------|--------|-----------------|
| Val loss stable | Data | `check_val_loss_converged()` — val loss plateau |
| Val loss increasing | Data | Overfitting detected |
| Geometric certificate | Gradient + curvature + data | 5-condition certificate gated by val_loss trend (`should_certificate_stop()`) |
| Budget exhausted | Weyl 1912 | Any layer's ||BA||₂/σ_k > gap/(2σ_k) |
| Max iterations | Circuit breaker | Safety cap only |

### Core Domain Modules (all wired into `mc train run`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `geometric_lora.py` | Weight analysis → LoRA config | `analyze_weight_geometries()`, `select_target_modules()` |
| `geometric_optimizer.py` | Per-layer optimizer params from SVD | `derive_optimizer_geometry_config()` |
| `scaled_gd.py` | Riemannian GD preconditioning | `precondition_lora_gradients()` |
| `spectral_budget.py` | Weyl-derived adapter saturation monitoring | `compute_budget_ratios()`, `is_budget_exhausted()` |
| `geometric_early_stopping.py` | Data-derived convergence detection | `check_loss_stable()`, `check_val_loss_converged()`, `check_stopping_certificate()`, `should_certificate_stop()` |
| `diagonal_fisher_preconditioner.py` | Per-parameter curvature preconditioning | `init_fisher_state()`, `precondition_gradient()`, `derive_beta1()` |
| `cayley_lora.py` | NB-LoRA parameterization | `cayley_transform_full()`, `NBLoRALayer` |
| `cka.py` | Capability preservation verification | `compute_linear_cka_from_activations()` |
| `activation_provider.py` | Activation collection for CKA | `collect_hidden_activations()` |
| `marchenko_pastur.py` | Random-matrix noise edge for eigenvalue thresholding | `marchenko_pastur_noise_edge()`, `effective_dimension()` |

---

## Current State

### Canonical Surface Status

Mission accounting is intentionally narrow:

- `mc train run` is the only clearly shipped canonical surface counted toward
  mission closure today
- its runtime path is geometry-derived and guarded by `pipeline_gate_v1`
- the canonical engine is still not closed because the baseline suite,
  behavioral preservation, 8B efficacy, and the quantization frontier law all
  remain open

What is already true on the canonical path:

- the runtime training surface no longer exposes the old user-facing
  hyperparameter bypasses
- spectral safety, MASS control, data-derived stopping, and preservation
  telemetry are wired into `mc train run`
- `results/pipeline_validation/verdict.json` still reports `all_pass = false`
  even while `all_structural_pass = true`

What does **not** count as mission closure yet:

- experimental merge workflows
- continual-learning or consolidation infrastructure
- stacking infrastructure
- sovereignty or user-owned identity runtime flow

### Mission Blockers

| Blocker | Current evidence | Exit criterion |
| --- | --- | --- |
| Head-to-head baseline suite | `results/nblora_vs_standard/` retains standardized slices and a grid summary, but not a promotable benchmark bundle | same-model same-data same-eval multi-seed comparison against standard LoRA, rsLoRA, PiSSA, EVA, DoRA, and a recipe-level baseline survives preservation gates |
| Behavioral preservation operator | `results/pipeline_validation/verdict.json` shows structural pass without inference closure | a pre-registered operator predicts and explains the retained failure cases before online degradation |
| 8B non-ceiling efficacy | `results/g5_8b_validation_multiseed/multiseed_gates.json` keeps open failures in `cka_ok` and `degenerate_ok` | the pre-registered 8B seed set closes the full gate bundle on the fixed non-ceiling evaluator |
| Quantization frontier law | `results/quantization_frontier/` and `results/closedform_sequential_correction/` are promising but incomplete | one architecture-conditioned frontier statistic predicts achieved CKA floor and degeneration behavior across bit-depth sweeps |

### Downstream Vision Gates

These matter, but they are downstream of canonical mission closure:

| Gate | Current state | Why it is downstream |
| --- | --- | --- |
| Portable adapter certificate | partial / experimental | depends on stronger preservation math and baselineed merge comparison |
| Stacking preservation certificate | experimental | requires portable behavior, not just probe alignment |
| Consolidation operator | experimental | depends on a non-forgetting update law and fixed continual-learning baselines |
| Sovereignty infrastructure | not built | requires runtime and product infrastructure after the geometry closes |

### Closure Order

Mission-first order:

1. baseline suite against standard practice
2. causal operator for behavioral failure when structural safety passes
3. 8B non-ceiling efficacy closure
4. quantization frontier law

Vision-gate order after that:

5. portable adapter certificate
6. consolidation operator
7. stacking preservation certificate
8. sovereignty infrastructure

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
