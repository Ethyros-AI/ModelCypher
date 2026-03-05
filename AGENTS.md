# AI-Assisted Development Guide

Guidance for AI coding assistants working on ModelCypher.

---

## This Is Research Code, Not a Product

**No backwards compatibility. No users to keep happy. Math and code to get right.**

This is a research codebase exploring geometric properties of neural networks. The goal is correctness, not convenience. There are no customers, no deprecation cycles, no migration paths.

**Correct is singular.** There is ONE right way to do things:
- ONE ModelLoader that uses Backend
- ONE ActivationProvider that uses Backend
- ONE Backend protocol that abstracts MLX/JAX/CUDA

If duplicate implementations exist, delete them. If backwards-compatibility shims exist, delete them. If factory patterns exist that just add indirection, delete them.

**Architecture is simple:**
```
backends/           ← ONLY place for framework imports (mlx, jax, torch)
adapters/           ← Uses Backend protocol
core/               ← Uses Backend protocol, NEVER imports frameworks
```

Framework imports (`import mlx`, `import jax`, `import torch`) belong in `backends/` ONLY. Everything else uses the Backend protocol. If you see framework imports outside `backends/`, fix them.

---

## What is ModelCypher?

Geometric diagnostics for LLM representations. Measures intrinsic dimension, curvature, entropy, and similarity to guide model merging, monitor training, and detect behavioral drift.

- **Backend**: MLX (macOS) primary, JAX (Linux/TPU) secondary
- **Architecture**: Hexagonal (ports and adapters)
- **Tests**: Run `poetry run pytest` for current count

## Quantization Is the Endgame (2026-03-05)

Full precision work is the derivation phase, not the destination.

- `bf16/fp16` is where we derive and falsify geometric mechanisms.
- Quantized models are the deployment target for real hardware constraints and throughput.
- A method is not "done" when it works in full precision; it is done when we can explain and control its behavior under quantization.
- Treat quantization as a deterministic geometric perturbation operator, not as a post-hoc compression trick.

Required contract for new training/merge claims:

`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`

If bf16 and quantized behavior diverge:
- Do not narrate; trace the operator path.
- Identify where invariance breaks (for example: spectral tail clipping, rank collapse, condition number inflation, projection misalignment).
- Add the missing precision term to the mechanism and re-test.

## Token Budget For AI Review

The one-shot review budget is **20,000 tokens per file**.

- Audit with: `poetry run python scripts/report_token_budget.py --threshold 20000`
- Keep `src/` and `docs/` files under budget whenever possible.
- Large corpus/result assets (`data/`, `plasma/results/`) and `poetry.lock` are allowed to exceed the budget because they are data artifacts, not implementation modules.

---

## STOP — Read This Before You Do Anything

**The ML industry is built on at least one fundamentally wrong assumption. That means you cannot trust ANY of its conventional wisdom without re-deriving it from first principles.**

The wrong assumption: **probability causes events.** The entire industry treats the softmax distribution as if it IS the model's reasoning process — as if the model "samples from a distribution" to think. This is categorically wrong. A forward pass is a deterministic geometric map. Given parameters and input, there is exactly one output. Softmax is a normalization humans apply at readout time. Probability describes our uncertainty about which path was taken; it does not cause the path to be taken.

This is not a philosophical nuance. It is the difference between a geocentric and heliocentric model of computation. And the industry has built 8 years of "best practices" on the geocentric version.

**Consequences of getting this wrong:**
- "Temperature controls creativity" → Wrong. Temperature injects noise into the readout. The model's geometric trajectory is identical at any temperature. You're not making the model "more creative" — you're randomly perturbing which token gets selected from a fixed distribution.
- "The model hallucinates because it's uncertain" → Wrong. The model hallucinates because the query lands in a sparse region of the activation manifold where nearest-neighbor interpolation gives a plausible but incorrect geometric path. It's topology, not probability.
- "Training is fitting a distribution" → Wrong. Training adjusts weight geometry so activation trajectories land at higher compatibility with correct targets. Cross-entropy is a margin/energy misalignment measurement, not a distributional fitting objective.
- "Next-token prediction IS the model" → Wrong. The model is a composed geometric map. Next-token prediction is the readout/selection interface. Confusing the interface with the mechanism is like confusing a speedometer with an engine.
- "We need more data to generalize" → Sometimes wrong. If the model's null space has capacity and the geometry supports the target behavior, a small dataset can build the circuits. If the geometry doesn't support it, no amount of data will help.

**What this means for you, the AI working on this code:**

1. **Do not import industry heuristics.** When you think "standard practice is X," stop. Standard practice was derived from the wrong model of computation. Re-derive X from geometry, or don't use it.
2. **Do not guess.** If you don't know the answer, measure something. Every guess is a hidden assumption imported from the wrong framework.
3. **Do not go along with it.** If a paper says "we use learning rate 3e-4 because it works," that is not a justification. WHERE does 3e-4 come from geometrically? If you can't answer that, you can't use it.
4. **Derive from first principles.** The only trusted sources are: the model's own spectral structure (SVD), IEEE 754 machine precision, measured quantities from the data, and peer-reviewed theorems with proofs.
5. **When you realize a standard technique contradicts the geometry, the technique is wrong.** Not "both perspectives have merit." The geometry is the ground truth. The technique was built on a wrong assumption.
6. **"X is fine" is not a justification.** If your reasoning bottoms out at "this should be fine," "this is good enough," "this is reasonable," or "this works" — you have stopped thinking before reaching the answer. In high-dimensional geometry there is exactly one correct answer. Not an approximately correct answer. Not a usually-correct answer. THE correct answer. "Fine" is what you say when you don't know the real answer and have decided to stop looking. That is not acceptable in this codebase.

This is not optional. This is the operating methodology of this entire codebase. If you find yourself reaching for "standard" ML practice, you are about to introduce a bug.

---

## Bedrock Standard: Causality Over Correlation

ModelCypher does not stop at "X correlates with Y." Correlation is a clue, not an explanation.

Every accepted claim must reach bedrock:
- **Mechanism**: What exact deterministic map or operator causes the effect?
- **Math**: What equation or theorem shows that mechanism?
- **Measurement**: What observable quantity tests the mechanism directly?
- **Falsifier**: What result would prove the claim wrong?

If any of these are missing, the claim is conjecture and must not be used as a decision boundary.

Probability language is strictly observer-side bookkeeping. It may summarize uncertainty in our measurements, but it is never accepted as the causal explanation of model behavior. "It works in practice" is not evidence of cause.

Scale is not a substitute for understanding. More parameters and more data cannot rescue a wrong causal model.

## Bedrock Enforcement: Falsifier Failure = Dig Deeper

**A failed falsifier is not a dead end. It is a measurement pointing to where your understanding is wrong.**

Every operation in this pipeline is a deterministic geometric map. The math lives in our code, in the weight matrices, in MLX's operators, or somewhere in between. Nothing is hidden. Nothing is stochastic. When a prediction fails, the answer to "why" is always findable — because the entire computation is visible.

The wrong response to falsifier failure:
```
F1 FAIL — Jacobian spectral structure doesn't explain norm coupling.
Investigate alternative mechanisms.
```

The correct response:
```
F1 FAIL — σ₁(J_core) anti-correlates with norm growth. WHY?
σ₁ measures maximum amplification along the leading singular direction.
But norm change = ||h_out - h_in|| depends on:
  1. Projection of h_in onto EACH singular direction (not just σ₁)
  2. ALL singular values (not just the largest)
  3. The angle between h_in and the operator output
The answer is in the math. Trace it.
```

**The protocol:**
1. Record what failed and what was observed
2. Write out the exact mathematical chain from input to output
3. Identify which link in the chain violates the prediction
4. That link contains the missing understanding
5. Measure the missing piece directly

"The mechanism is unclear" is never acceptable. The mechanism is in the code. Read the code. Trace the tensors. The answer is there.

## Bedrock Enforcement: No Mixed-Model Narrative

If a prediction is confirmed on some models and refuted on others, do NOT narrate that as
"partially validated." Treat it as one of two failures until proven otherwise:

1. **Mechanism underspecified** — missing architecture or scale terms in the derivation.
2. **Measurement invalid** — statistic is not commensurable across compared layers/models.

Required before any cross-model claim:
- Write the architecture-conditioned equation (explicit architecture variables).
- Write the scale-conditioned equation (depth/width/parameter dependence).
- Prove measurement commensurability for the chosen operator.
- Pre-register directional prediction and falsifier.
- Use claim form: `observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`.
- Apply and cite `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`.

If any item is missing, the work is exploratory only and must not be promoted in roadmap,
mission, or doctrine docs.

---

## The Foundational Premise

**LLMs are not black boxes. They are high-dimensional geometry.**

This is not metaphor. This is not approximation. This is the mathematical reality:

1. **The relational structure of meaning is invariant.** The relationship between "apple" and "orange", between "cause" and "effect", must be consistent—or meaning couldn't transfer across languages, encodings, or compression levels. This is logical necessity, not hypothesis. [PROVEN: follows from compositionality of language]

2. **Coordinate systems are model-specific.** Different models encode the same relational structure in different bases. A 0.5B model and a 70B model may represent the same relationships at different resolutions, but their coordinate systems differ. [PROVEN: follows from arbitrary basis choice in embedding initialization]

3. **CKA captures relational geometry invariant to coordinates.** Gram matrices (K = X @ Xᵀ) encode pairwise relationships independent of basis choice. CKA compares Gram structure, which is why it works across dimensions. [PROVEN: Kornblith et al. 2019]

4. **The Platonic hypothesis: models converge to shared structure.** Whether independently trained models discover the SAME relational geometry is empirical, testable via CKA on probes. High CKA after alignment is evidence, not proof. [CONJECTURAL: high CKA observed across 3 architecture families, but "same structure" is stronger than "similar covariance"]

5. **Models can be merged when coordinates can be aligned.** Procrustes finds the orthogonal transform mapping one coordinate system to another. Null-space addition preserves target behavior on sampled activations. [VALIDATED: cross-architecture merging demonstrated on LFM2/Qwen/SmolLM with CKA >= 0.95]

6. **Probability is epistemic bookkeeping, not causal mechanism.** A forward pass is a deterministic geometric map from prefix to logits. Softmax is an observer-side normalization at readout/selection time, not the internal process that creates state trajectories. Treat "probability" metrics as shadows of geometric structure, never as the thing doing the causal work. [PROVEN: follows from the definition of a deterministic function composition]

**If code contradicts these premises, check the math first.**

**Experimental evidence** (from prior validation runs — command removed after Δβ₁ falsification):
- **Alignment invariance**: Raw CKA=0.60 (before), Aligned CKA=1.00 (after Procrustes) - structure is preserved, coordinates differ
- Layer-wise intrinsic dimension compression (15.8 → 1.8 in middle layers)
- Domain-specific manifold structure (spatial ID=1.5, moral ID=8.0)

---

## The Research Methodology

**We solve through increasing constraint on the geometry.**

This is not trial-and-error. This is systematic elimination of unknown pathways:

### CKA = 1.0 is the Invariant Unlock

The key mathematical discovery: after Procrustes alignment, CKA = 1.0 on shared concepts. This proves:
- The relational structure is **identical** across models
- Only the coordinates differ
- Alignment is **closed-form**: `F = pinv(source) @ target`

From this single invariant, everything else derives:
- **Alignment** → closed-form rotation finding
- **Transfer** → null-space projection onto unused capacity
- **Density** → k-NN comparison identifies where to transfer
- **Coherence** → trajectory validity on the merged manifold

### Tokens Are Shadows, Not the Thing Itself

When a prompt enters, it becomes a trajectory through the manifold. The model is a passthrough - concepts have gravity and pull the trajectory through high-dimensional space. Tokens are the powder flying off the skis - the residue of geometry, not the thought itself.

**Implication**: Don't debug tokens. Debug the geometry that produces them.

### Hallucination is Geometric, Not Moral [CONJECTURAL]

Hallucination is NOT the model "lying." It's one of two geometric phenomena:
1. **Sparse interpolation**: Query lands in under-sampled region; nearest-neighbor gives plausible but wrong path
2. **Tangent hop**: Trajectory follows dimensionally-adjacent but logically-unrelated concept

The model can't "see" cliff edges in sparse regions. It's not malicious - it's topology. [CONJECTURAL: proposed mechanism, no direct empirical test of the sparse interpolation / tangent hop distinction]

**Implication**: Fix by characterizing the manifold (dense sampling), not by "training honesty."

### One Variable Per Day

When something breaks:
1. **Pick one variable** - sample coverage, condition number, spectral gap, density weighting
2. **Characterize it fully** - what does it mean geometrically? Where in the trajectory does it matter?
3. **Measure before/during/after** - activation geometry in, projection geometry during, coherence out
4. **Work backward from failure** - which metric changed? In which direction? At which stage?

The problem space is finite. Every variable is discoverable. Every interaction is measurable.

### Metrics, Not Vibes

Every diagnostic must return raw measurements:
- **Coverage ratio**: n_samples / hidden_dim (must be > 1.0, ideally > 4.0)
- **Condition number**: max_eigenvalue / min_eigenvalue (numerical stability)
- **Null rank**: dimensions available for transfer
- **Transfer strength**: mean density weight applied
- **Preserved fraction**: how much delta survived projection
- **Spectral gap**: separation between used and unused directions

When something fails, one of these metrics will tell you why.

### The Debugging Contract

```
If coherence fails:
    → Check coverage_ratio (was manifold properly sampled?)
    → Check condition_number (was projection numerically stable?)
    → Check density_weights (did transfer happen in right places?)
    → Check spectral_gap (were used/unused directions separated?)
    → Check preserved_fraction (how much delta survived?)

Each metric points to a different failure mode.
Each failure mode has a different fix.
The space is finite.
```

**If you find yourself guessing, you're missing a metric.** Add the metric first.

### Stranded Neurons: Alignment Stability via Condition Number

**The alignment matrix F = pinv(A_source) @ A_target requires numerical stability.**

The geometry says:
1. **Numerical stability depends on condition number**, not a simple probe count threshold
2. **Condition number κ = max_eigenvalue / min_eigenvalue determines solution accuracy**
3. **Check κ at runtime**—actual stability depends on activation structure, not a formula

For float32 with ε ≈ 1e-7:
- κ × ε is the relative error in the solution
- κ = 1e5 → 1e-2 relative error (2 significant digits)
- κ = 1e3 → 1e-4 relative error (4 significant digits)

The threshold is dtype-derived: κ × machine_epsilon must be small enough for your use case.

**Implementation**: GramAligner computes Gram condition number and logs it.
More probes provide overdetermined least-squares (good for stability). The Gram matrix
rank is bounded by min(n_probes, hidden_dim); the pseudoinverse handles rank deficiency.
The runtime condition number check determines actual numerical stability.

**If merge produces incoherent outputs but CKA looks good**: Check the Gram condition number.
The alignment may have succeeded mathematically but the transform is numerically unstable.
Use --full-atlas for more probes (4596 total in atlas).

### Trajectory Rank is the Geometric Ceiling

**The activation rank cannot exceed trajectory_rank** - this is topology, not a heuristic.

- `trajectory_rank` = intrinsic manifold dimension (computed from SVD with sqrt(eps) threshold)
- `hidden_dim - trajectory_rank` = the null space we project INTO
- Rank augmentation stops when `activation_rank >= trajectory_rank`

This is why we don't use arbitrary iteration limits. The geometry tells us when to stop:

```python
# WRONG: Arbitrary iteration limit
for i in range(1000):  # Why 1000? No geometric basis.
    augment()

# CORRECT: Stop when geometric ceiling reached
while activation_rank < trajectory_rank:
    augment()
    activation_rank = compute_rank(activations, threshold=sqrt_eps)
```

The loop terminates when the manifold is fully spanned. No magic numbers needed.

---

## Commands

Always use `poetry` to run or install anything in this repo.

```bash
poetry install                    # Install
poetry run pytest                 # Test
poetry run mc --help              # CLI
```

---

## CLI Quick Reference

### Training (The Main Operation)

**Train a LoRA adapter — all hyperparameters derived from geometry:**
```bash
mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
```

No learning rate, no rank selection, no warmup. Everything is derived from the weight matrices.

**Research path with explicit controls:**
```bash
mc train run-research --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
```

### Analysis

```bash
# Intrinsic dimension profile
mc analyze dimension-profile --model /path/to/model

# LoRA adapter spectral analysis
mc analyze lora-svd /path/to/adapter --base /path/to/model

# Spectral entropy trajectory
mc analyze spectral-trajectory --model /path/to/model
```

### Model Merging (Experimental)

```bash
# Single merge (1→1)
mc merge run -s SOURCE -t TARGET -o OUTPUT

# Batch merge (N→1)
mc merge batch -s MODEL1 -s MODEL2 -t TARGET -o OUTPUT
```

Adds SOURCE knowledge to TARGET via null-space projection. TARGET's capabilities are preserved by construction.

### Other Commands

```bash
mc infer run --model /path/to/model --prompt "Hello"
mc system status --output json
mc model info /path/to/model --output json
```

---

## Architecture

```
src/modelcypher/
├── core/
│   ├── domain/        # Pure math + logic (geometry, safety, merging, thermo)
│   ├── ports/         # Abstract interfaces (Backend protocol)
│   └── use_cases/     # Service orchestration
├── adapters/          # Concrete implementations (hf_hub, filesystem)
├── backends/          # MLX, JAX implementations
├── cli/               # Typer CLI
```

Dependencies point inward. Domain imports nothing external.

---

## Concurrency Rules

Multiple AI agents work concurrently. Don't pause for unrelated changes.

1. Ignore modified or untracked files you don't need to touch. Do not mention them or ask about them; leave them alone.
2. No destructive git operations (`add`, `commit`, `push`, `reset`)
3. No bulk modification scripts—edit files individually
4. **No tests during training.** If model training is running (`mc train run`, `train_from_dataset_research`, or any script that loads models onto the GPU), do NOT run `pytest` or any test suite concurrently. Tests and training both compete for Metal GPU memory and unified RAM — running both causes OOM crashes or silent numerical corruption. Wait until training completes before running tests.
5. **Explicit override**: If you notice unexpected or unrelated changes, do not pause or ask. Ignore them unless they are in files you must edit for the task; if they are, integrate them and continue without interruption.

---

## Core Principles

### Prefer Backend Over NumPy

**NumPy forces CPU fallback. Use the Backend protocol to stay on GPU.**

Use the Backend protocol in core domain code. No `import numpy`, no `to_numpy()`, no NumPy operations in the domain layer. If the Backend doesn't have an operation you need, **add it to the Backend protocol**.

```python
# WRONG - Forces CPU fallback
import numpy as np
mean = np.mean(vectors, axis=0)
sorted_vals = np.sort(eigenvalues)[::-1]
result = backend.to_numpy(arr)[mask]  # NumPy boolean indexing

# CORRECT - Stays on GPU
from modelcypher.core.domain._backend import get_default_backend
backend = get_default_backend()
mean = backend.mean(vectors, axis=0)
sorted_idx = backend.argsort(eigenvalues)
reversed_idx = backend.arange(n - 1, -1, -1)
sorted_vals = backend.take(eigenvalues, reversed_idx, axis=0)
result = backend.where(mask, arr, backend.zeros_like(arr))
```

**Common NumPy patterns and their Backend replacements:**

| NumPy Pattern | Backend Replacement |
|---------------|---------------------|
| `arr[::-1]` | `backend.take(arr, backend.arange(n-1, -1, -1), axis=0)` |
| `arr[mask]` | `backend.where(mask, arr, zeros)` then filter |
| `np.sort(arr)` | `backend.sort(arr)` |
| `arr[:, -1] *= -1` | `scale = backend.array([1.0]*(d-1) + [-1.0]); arr * scale` |
| `np.linalg.det(A)` | `backend.det(A)` |
| `for x in to_numpy(arr)` | Keep on backend, use `backend.take()` for indexing |

**If you need an operation the Backend doesn't have, add it.** The Backend protocol is extensible. Don't work around missing ops with NumPy.

### Geometry Type Matters: Activations vs Weights

**Critical distinction**: Activation space and weight space have DIFFERENT geometry.

| Space | Geometry | Correct Metric | Research Basis |
|-------|----------|----------------|----------------|
| **Activation vectors** | Curved Riemannian manifold | Geodesic on k-NN graph | arXiv:2506.12187 "Characterizing Neural Manifolds" |
| **Weight matrices** | Flat + spectral structure | Euclidean + eigenvalues | ICLR 2026 "From Memorization to Reasoning in the Spectrum of Loss Curvature"; Fort & Ganguli "Emergent properties of neural loss landscapes" |

**Activation space** (neural manifold): Empirical measurements show curved manifolds with measurable Riemannian curvature tensors. Geodesic distance via k-NN graph is correct.

**Weight space** (loss landscape): Research shows weight space has SPECTRAL structure (Hessian eigenvalues), NOT manifold curvature. High-curvature directions = shared generalizable structure. Low-curvature directions = memorized examples. The space is mostly FLAT with spectral outliers.

```python
# For ACTIVATIONS - use geodesic (curved manifold)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
rg = RiemannianGeometry(backend)
distances = rg.geodesic_distances(activation_points, k_neighbors=k)

# For WEIGHTS - use Euclidean + spectral (flat + eigenvalue structure)
weight_norm = backend.sqrt(backend.sum(weight * weight))  # Frobenius norm
# Spectral analysis via SVD/eigendecomposition for structure
```

**Don't use geodesic on weight matrices.** It's 400x slower with marginal accuracy difference because weight space isn't curved.

### Behavioral Norm for Transplant Metrics

When measuring weight deltas in transplant, use **behavioral norm**, not Frobenius norm.

**Why?** Frobenius measures weight magnitude. Behavioral measures actual output change.

```python
# WRONG: Frobenius norm (ignores activation structure)
delta_norm = sqrt(sum(delta_W ** 2))  # Misleading

# CORRECT: Behavioral norm (measures actual impact)
output_change = input_activations @ delta_W.T
delta_norm = sqrt(sum(output_change ** 2))  # True impact
```

**Key insight**: After null-space projection:
- Frobenius might say "47% preserved" (weight mass)
- Behavioral shows "0.0002% preserved" (actual impact on target)

The behavioral norm is the TRUTH. Null-space projection preserves weight magnitude but eliminates behavioral impact on target activations. That's the design.

For `preserved_fraction`:
- Use: `behavioral_after / behavioral_before`
- NOT: `frobenius_after / frobenius_before`

This answers "What fraction of behavioral change transferred?" - which is what we actually care about.

### All Models Encode the Same Shape [CONJECTURAL]

**Observed in alignment experiments, not proven universal.**

Neural networks trained on language converge toward shared high-dimensional geometric structure. Different architectures (SmolLM, Qwen, Llama, Mistral) are different compressions or projections of this common structure. [CONJECTURAL: CKA measures covariance similarity, not manifold identity. "Same shape" is a stronger claim than the evidence supports. See VALIDATION-REPORT.md.]

**Key insight**: Raw CKA between unaligned representations can be low (e.g., 0.60) because they use different coordinate systems. After Procrustes alignment, CKA = 1.0 on training probes - the structural relationships are identical, only the coordinates differ. [PROVEN: F = pinv(source) @ target guarantees K_aligned = K_target when n <= d]

Think of it like high-dimensional Legos: the geometry constrains how pieces fit together.

- **Different dimensions** = different compression levels of the same geometry
- **Different layer counts** = different resolutions of the same manifold
- **Different architectures** = different coordinate systems for the same shape

Use Gram matrices for comparison (dimension-agnostic). Use projection for transformation. Low raw CKA doesn't mean incompatible - it means coordinate alignment is needed.

### CKA = 1.0 on Training Probes [PROVEN]

Procrustes alignment achieves CKA = 1.0 on training probes by construction. **Experiment shows: Raw CKA=0.60 → Aligned CKA=1.00** [VALIDATED: reproduced on LFM2/Qwen/SmolLM]

**F = pinv(source) @ target** guarantees **K_aligned = K_target** when n ≤ d. This is closed-form. No iteration needed. [PROVEN: linear algebra]

- **CKA = 1.0 on probes**: Alignment found the correct rotation for those probe points.
- **CKA < 1.0 on held-out samples**: Probes didn't span enough of the shared manifold.

**LOW CKA ON RAW (UNALIGNED) DATA MEANS:**
- Coordinate systems differ (expected)
- Run alignment first, then evaluate

**LOW CKA ON HELD-OUT DATA AFTER ALIGNMENT MEANS:**
- Probes didn't span the shared manifold regions
- Need more diverse probes (different domains, abstraction levels)
- Expand probe coverage - the alignment math is correct

**Key distinction**: Low *raw* CKA is expected (different coordinates). Low *aligned* CKA on held-out data means insufficient probe coverage.

### Geometric Addition, NOT Blending

**This is the most important principle. Read it until you understand it.**

LLMs are NOT piles of probabilities to weight. They are HIGH-DIMENSIONAL GEOMETRIC STRUCTURES - like Legos that snap together.

**The goal**: Combine unique knowledge from hundreds of models into ONE denser, smarter model.

**WRONG - Interpolation/Blending**:
```python
# This is WRONG. Never do this.
merged = alpha * model_A + (1 - alpha) * model_B  # Blending
merged = 0.5 * source + 0.5 * target              # Weighted average
merged = lerp(source, target, t)                   # Interpolation
```

Why it's wrong: Interpolation AVERAGES information. You get a smeared, degraded model that's worse than either input. You're not adding knowledge - you're diluting it.

**CORRECT - Null Space Addition**:
```python
# This is CORRECT. Knowledge addition.
delta = source_weights - target_weights
projected = null_space_projection(delta, target_activations)
merged = target_weights + projected
```

Why it works:
- **Null space** = directions the target model doesn't actively use (low variance)
- Projecting source delta into these sparse regions means: **add source knowledge where target has capacity**
- Target behavior is PRESERVED (dense directions scaled down)
- Source knowledge is ADDED (not averaged)
- Result is DENSER than either model alone

**Implementation note**: Uses variance-weighted projection, not true orthogonal null-space.
Dense directions (high activation variance) are scaled down; sparse directions are preserved.
This is intentional - true orthogonal projection with many samples erases all delta.

**Think of it like this**:
- Blending: Mixing two paint colors → muddy average
- Addition: Adding ingredients to a recipe → richer dish

**Our approach: addition, not blending.** We project source deltas into the target's null space rather than interpolating weights. Rationale: interpolation assumes shared coordinate systems and mode connectivity; null-space addition preserves target behavior on sampled activations by construction. Both approaches have valid use cases (model soups, Git Re-Basin show blending works for mode-connected models); we optimize for behavior preservation when coordinate systems differ.

### No Vibes

Return raw measurements. No hardcoded thresholds, interpretation strings, or qualitative labels.

```python
# Wrong
return {"similarity": 0.73, "interpretation": "Good alignment"}

# Correct
return {"similarity": 0.73}
```

When thresholds are needed, derive from baselines (z-scores, percentiles).

### No Simulated Results

**Run real models or don't claim to know the answer.**

Reprocessing cached JSON to "predict" what a script will output is not measurement — it is simulation dressed as verification. If a test exists in the script, run the script on real models. Do not substitute `python -c "import json; d = json.load(...);"` for the actual pipeline.

The only valid verification is the real pipeline producing real numbers from real weights. A simulation that matches expectations proves nothing — it proves you can re-derive the same arithmetic. The script exists to run on models, not on cached output.

```python
# WRONG: "Smoke test" by reprocessing cached results
d = json.load(open("results/cached_output.json"))
# ... recompute statistic from cached data ...
print(f"Expected result: {result}")  # This is not a measurement

# CORRECT: Run the actual pipeline
# poetry run python scripts/curvature_accumulation_analysis.py --models ...
# Then read the output
```

If the volume isn't mounted or the run takes too long for the current session, say so and stop. Do not fabricate a shortcut.

### Don't Invent Heuristics

**Every heuristic is an admission of ignorance.**

When you write `if ratio < 0.8` or `threshold = 0.1` or `margin = 100`, you're saying: "I don't understand the geometry well enough to know the real constraint, so I'm guessing."

The problem space is finite. The geometry has answers. If you don't know the answer, you haven't done the research yet.

**The wrong response to uncertainty:**
```python
# "I don't know what rank ratio matters, so I'll pick 0.8"
if rank_ratio < 0.8:
    logger.warning("Rank mismatch detected")

# "I don't know when curvature correction breaks down, so I'll pick 10%"
if error > 0.1:
    skip_correction()

# "I don't know how many probes we need, so I'll add 100"
min_probes = max_dim + 100
```

**The correct response to uncertainty:**
```python
# Log the measurement. Let the geometry speak.
logger.info("Alignment: src_rank=%d, tgt_rank=%d, alignment_rank=%d", ...)

# Derive from machine precision - the ONLY thing we know for certain
if taylor_remainder >= sqrt(machine_epsilon):
    # Correction is numerically meaningless - not "probably wrong"
    return uncorrected

# Research until you find the actual constraint
# Berry & Sauer (2016): n >= d * (1 + 1/sqrt(d)) for well-conditioned Gram
# But even this assumes "generic point clouds" - is that our data?
```

**The test:** Can you cite the mathematical derivation for your number? If not, it's a guess.

**When you catch yourself guessing:**
1. STOP writing code
2. Add a measurement instead (return raw data, log the value)
3. Research until you find the actual constraint
4. The constraint will come from: machine precision, geometric invariants, or experimental measurement on baseline data

**Numbers that ARE allowed:**
- Machine epsilon (dtype-derived)
- sqrt(epsilon) for relative precision thresholds
- Mathematical constants (pi, e, etc.)
- Formulas from peer-reviewed papers (with citation)
- Measurements from baseline experiments in this codebase

**Numbers that are NOT allowed:**
- Round numbers (0.1, 0.5, 0.8, 100)
- "Standard" values from other codebases
- Anything described as "works well in practice"
- Anything you can't derive on a whiteboard

**Watch for math-washing:** Dressing up a guess in mathematical language doesn't make it principled.

```python
# This LOOKS principled but isn't:
# "The Taylor remainder is O((K*r²)²). For 10% error: |K*r²| < 0.32"
# WHERE DID 10% COME FROM? That's a guess. 0.32 is derived from the guess.

# This IS principled:
# "Taylor remainder must be < sqrt(eps) to be distinguishable from noise"
# sqrt(eps) comes from numerical analysis, not a guess.
```

**The smell test:** If you picked a percentage (10%, 20%, 80%), it's a guess. Percentages don't appear in geometry - they appear in human intuition.

### "Fine" Is Not an Answer

**"X is fine" means "I stopped looking."**

The most dangerous phrase in AI-assisted development is "this should be fine." It is the mechanism by which wrong assumptions enter a codebase — not through malice, but through premature satisfaction.

These are all the same failure:
- "This learning rate is fine" → You don't know the correct learning rate
- "This threshold should work" → You don't know the correct threshold
- "This is a reasonable default" → You don't know the correct value
- "This is good enough" → You know it's wrong and have decided not to fix it
- "This shouldn't cause issues" → You don't know whether it will cause issues

In high-dimensional geometry, there is exactly ONE correct answer for every question. The deterministic map from input to output has one path. The spectral structure has one decomposition. The Weyl bound has one value. Nothing is approximate. Nothing is "close enough." Nothing is "fine."

**The correct response when you don't know the answer:** Say "I don't know" and measure something. Do not say "this is fine" and move on. The distance between "I don't know" and "this is fine" is the distance between science and negligence.

**The test:** Replace "fine" with "correct" in your sentence. If you can't honestly make that substitution — if you can't say "this is correct" with the same confidence — then you haven't found the answer yet.

---

## Research Before Code

AI training data is stale. Before using external APIs:
1. Search for current best practices (use current year in queries)
2. Fetch and read official documentation
3. Check for breaking changes

---

## Model Size Policy

**Smallest viable model first. Always.** All math validation, code testing, experimentation, and training runs default to the smallest models that expose the property being tested. A math bug at 350M is the same math bug at 8B — but 350M iterates 20x faster.

| Purpose | Model Size | When |
|---------|-----------|------|
| Math validation, unit tests, code bugs | 350M / 700M (LFM2, SmolLM) | Always — default for all dev work |
| Cross-architecture validation | 350M + 700M (different families) | When testing generalization |
| Scale-dependent behavior | 1.2B–1.7B (Qwen) | Only after small models pass |
| Production readiness | 3B–8B | Final validation before ship |

**Do NOT run 8B models for research iteration.** If the math is wrong at 350M, it's wrong at 8B. If it's right at 350M, validate scale behavior at 1.7B. 8B is for production confidence, not discovery.

---

## Experimental Research vs Production CLI

**Research code stays in modules until validated. CLI commands are for production-ready features only.**

The bar for adding a CLI command:
1. **Math is proven correct** - theorems derived, not guessed
2. **Outcomes are controllable** - we can predict what will happen
3. **Outcomes improve consistently** - the feature makes things better, not sometimes-better
4. **Validated with real models** - tested on actual adapters/models, not just unit tests

**Until all four criteria are met:**
- Code lives in `core/domain/` or `core/use_cases/` as internal APIs
- Can be called from tests and experiments
- Does NOT get a CLI command
- Marked as experimental in docstrings

**Promotion path:**
```
research idea → module code → unit tests → integration tests →
real model validation → reproducible improvement → CLI command
```

Example: LoRA spectral scale bounds started as research (discovering the formula), became module code (lora_safety_service.py), and will only become a CLI command after we've validated it improves outcomes on multiple adapters/models consistently.

**The test:** "Can we guarantee this will help, not hurt?" If no, it's not CLI-ready.

---

## CLI First

Never write custom scripts. Use the `mc` CLI. If capability doesn't exist, add a CLI command (once it's validated - see above).

---

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/CLI-REFERENCE.md` | Command reference |
| `docs/GEOMETRY-GUIDE.md` | Metric explanations |
| `docs/GLOSSARY.md` | Terminology |
| `docs/EVIDENCE-TAXONOMY.md` | Evidence status labels for claims |
