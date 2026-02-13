# ModelCypher Mission

## Mission Statement

**Train models better than they have ever been trained before, using only geometry.**

Every training decision — learning rate, rank, scale, convergence, batch size, weight decay, initialization, target selection, dropout, stopping — is derived from the spectral structure of weight matrices and the Riemannian geometry of activation manifolds. No grid search. No "what worked last time." No knobs.

Point any model at any dataset. Hit train. Get a LoRA that perfectly captures either the knowledge or the behavioral shapes contained in the data.

## Canonical Inference Model (Object vs Shadow)

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
| **Measured from the data** | Lipschitz constant L, gradient noise scale, loss variance |

**Test**: Grep the training codepath for any literal number that isn't derived from one of these three sources. If you find one, it's a guardrail violation.

The 15 hyperparameters and their geometric replacements:

| # | Hyperparameter | Geometric Replacement | Formula |
|---|---|---|---|
| 1 | Learning Rate | Measured Lipschitz constant | `eta = 1/L` where `L = lambda_max(Hessian)` |
| 2 | Adam Epsilon | Spectral noise floor | `max(sigma_k^2, sqrt(eps) * sigma_max^2)` |
| 3 | Adam/Momentum | ScaledGD preconditioning | `grad_A @ (BB^T+eI)^-1`, `(A^TA+eI)^-1 @ grad_B` |
| 4 | Weight Decay | Condition-aware scaling | `sigma_k / sigma_max` |
| 5 | Gradient Clipping | REMOVED | ScaledGD + budget monitoring prevent explosion |
| 6 | Warmup | REMOVED | Geometric LR stable from step 0 |
| 7 | LR Schedule | OPTIONAL | Condition ratio is static; cosine is marginal |
| 8 | Batch Size | Gradient noise scale | `B_crit = Var(g) / ||E[g]||^2` |
| 9 | Early Stopping | Geometric convergence | `loss_stable(SE_diff)` OR `budget_exhausted(Weyl)` |
| 10 | LoRA Scale | Spectral bound per-layer | `sigma_k(W) / ||BA||_spectral` |
| 11 | LoRA Rank | Null-space capacity | `tail_dims = full_rank - effective_rank` |
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
| **Budget exhaustion** | Spectral safety limit | Any layer's `||BA||_spectral / sigma_k > spectral_gap / (2 * sigma_k)` |

**Test**: The `stop_reason` field in `TrainResult` must be one of `convergence`, `stable_loss`, or `budget_exhausted`. Never `max_steps` as the primary design — max_steps exists only as a circuit breaker, not as the intended stopping mechanism.

### G4: Preservation of Existing Capabilities

The adapter must not degrade what the model already knows. Measured, not hoped.

- **Null-space projection**: Delta is projected into directions with low activation variance (GNSP/PNSP)
- **CKA alignment**: After merge, CKA between original and merged model on atlas probes >= 1.0 - sqrt(eps)
- **Mode connectivity barrier**: Interpolation path from base to adapted has barrier <= 1.0 + sqrt(eps)

**Test**: Run the verification suite on the merged model. All metrics within bounds.

### G5: Reproducible Across Models and Datasets

The system works on ANY model architecture and ANY dataset. Not just the ones we tested on.

- **Tested model scales**: 350M, 700M, 1.2B (validated), 8B (in progress)
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

---

## Architecture

### Two Training Paths

ModelCypher provides two independent paths from intent to adapter:

#### Path 1: Data-Driven Training
```
Dataset --> Activations --> LoRAMemoryStore --> NB-LoRA (Cayley) --> ScaledGD --> Trained Adapter
```

Traditional training loop, but every parameter is geometry-derived:
- Events accumulate as `(hidden_state, delta, confidence, heat)` tuples
- NB-LoRA parameterization guarantees spectral bounds
- ScaledGD replaces Adam with condition-number-free convergence
- Three geometric stopping criteria replace patience-based early stopping

#### Path 2: Geometry-Only Synthesis (No Training Data Required)
```
Source Model --> GeometricProfile --> CrossManifoldProjector --> TransferPoint --> GeometricLoRA
```

Direct LoRA synthesis from geometric measurements:
- Map activation manifolds via trajectory sampling (199 samples per 100-token text)
- Compute anchor distance profiles in source and target manifolds
- Project via stress minimization (landmark MDS)
- Solve `DeltaW = (y* - W@x) (x) x / ||x||^2` per layer
- Rank truncate via SVD

### Core Domain Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `geometric_lora.py` | Weight analysis -> LoRA config | `compute_layer_geometry()` |
| `geometric_optimizer.py` | Per-layer optimizer params from SVD | `derive_optimizer_geometry_config()` |
| `scaled_gd.py` | Riemannian GD preconditioning | `precondition_lora_gradients()` |
| `spectral_budget.py` | Weyl-derived budget monitoring | `compute_budget_ratios()` |
| `geometric_early_stopping.py` | Data-derived loss stability | `check_loss_stable()` |
| `cayley_lora.py` | NB-LoRA parameterization | `cayley_transform_full()`, `NBLoRALayer` |
| `direct_lora_synthesis.py` | Geometry -> weights | `GeometricLoRAGenerator` |
| `manifold_transfer.py` | Cross-model projection | `CrossManifoldProjector` |
| `lora_memory_store.py` | Event buffer + training loop | `accumulate()`, `train_step()`, `merge_to_base()` |

---

## Current State

### Implemented and Validated

- All 15 hyperparameter geometric replacements (Rosetta Stone complete)
- ScaledGD preconditioning (Tong et al. JMLR 2021, Hayou et al. ICML 2024)
- NB-LoRA via Cayley transform (Wang et al. 2025, arXiv:2501.19050)
- Spectral budget monitoring via Weyl perturbation theory (Weyl 1912, Shuttleworth et al. 2024)
- Data-derived early stopping via SE_diff
- Training validated on 3 model scales (350M, 700M, 1.2B)
- 17+ verification modules
- Backend abstraction (MLX, JAX, CUDA) — framework imports only in backend files
- Lipschitz constant measurement via exact Cayley pullback + HVP (Nesterov 2004)
- Geometric heat signal via EL2N relative perturbation (Paul et al. 2021)
- Spectral confidence from budget headroom × condition ratio
- Dataset training pipeline via `mc train run --data` (productized from validation script)
- 82%+ test coverage, 4400+ tests passing

### Remaining Gaps

| Gap | What's Missing | Impact |
|-----|---------------|--------|
| **Multi-LoRA stacking** | No verification for sequential/stacked adapters | Unknown interference effects |
| **Large-scale validation** | 8B+ models not yet fully validated | Guardrail G5 incomplete |

### What Closes the Gaps

The path from current state to mission-complete:

1. **8B+ validation**: Run the full pipeline on DeepSeek-R1-8B and Qwen3-8B. Either it works or we learn why not. The architecture shouldn't matter if the geometry is right.

---

## What This Is NOT

- **Not a training framework.** We don't compete with PyTorch Lightning or HuggingFace Trainer. We replace the decision-making, not the infrastructure.
- **Not approximate.** When we say `sigma_k`, we mean the actual singular value from SVD, not an estimate. When we say `machine_epsilon`, we mean `2^-23` for float32. Precision is the point.
- **Not a black box.** Every number in the training run has a derivation. Every derivation has a mathematical reference. Every reference is a published theorem or a measured quantity. If you can't trace a number back to SVD or IEEE 754, it's a bug.

---

## References

| Paper | What We Use |
|-------|-------------|
| Nesterov (2004) | Optimal step size eta = 1/L for L-Lipschitz gradient |
| Tong et al. (JMLR 2021) | ScaledGD: condition-number-free convergence for low-rank |
| Hayou et al. (ICML 2024) | Asymmetric LoRA convergence rates, ScaledGD for A/B |
| Wang et al. (2025, arXiv:2501.19050) | NB-LoRA: Cayley parameterization for norm-bounded LoRA |
| Weyl (1912) | Perturbation bounds: \|sigma_i(W+E) - sigma_i(W)\| <= \|\|E\|\|_2 |
| Shuttleworth et al. (2024, arXiv:2410.21228) | Empirical confirmation of Weyl bounds for LoRA |
| Roy & Vetterli (2007) | Shannon effective rank: exp(H(sigma^2)) |
| de Silva & Tenenbaum (2004) | Landmark MDS for cross-manifold projection |
| Eckart-Young (1936) | Optimal low-rank approximation via SVD |
| Kornblith et al. (2019) | CKA: Centered Kernel Alignment for representation similarity |
