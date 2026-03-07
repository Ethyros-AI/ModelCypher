# AI-Assisted Development Guide

ModelCypher is research code for geometric diagnostics, training, and merging of
LLMs. The goal is not convenience. The goal is bedrock-correct math and code.

## 1. Purpose

### Research, Not Product

- No backwards compatibility.
- No deprecation shims.
- No duplicate implementations for convenience.
- Correct is singular.

There is one correct architecture:

- one `ModelLoader` that uses the Backend protocol
- one `ActivationProvider` that uses the Backend protocol
- one Backend abstraction for MLX, JAX, and CUDA

If duplicate paths exist, delete them.

### Mission

Train and merge models better than standard practice allows, using only
geometry.

Every design decision must be derived from:

- spectral structure of weights
- geometry of activations
- IEEE 754 machine precision
- direct measurements from the model and data

### Quantization Is The Endgame

`bf16/fp16` is the derivation phase. Quantized models are the deployment target.

Use this contract for all promotable claims:

`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`

If quantized behavior differs from full precision:

1. trace the operator path
2. find where invariance breaks
3. add the missing precision term
4. re-test

### Token Budget

Keep implementation and docs files under the one-shot review budget when
possible.

Audit with:

```bash
poetry run python scripts/report_token_budget.py --threshold 20000
```

Large data artifacts and lockfiles are exempt.

## 2. First Principles

### Deterministic Geometry, Not Probability

A forward pass is a deterministic geometric map. Softmax is an observer-side
normalization at readout. Probability is epistemic bookkeeping, not causal
mechanism.

Do not import ML convention just because it is common. Re-derive it from
geometry or discard it.

### Every Claim Must Reach Bedrock

Any claim used for design, promotion, or decision boundaries must include:

1. mechanism: the deterministic operator that causes the effect
2. math: the governing equation or theorem
3. measurement: the observable that tests the mechanism directly
4. falsifier: the observation that would prove the claim wrong

If any link is missing, the claim is exploratory only.

### No Mixed-Model Narrative

If a result differs across models, do not call it partial validation. Treat it
as one of:

1. mechanism underspecified
2. measurement invalid across compared objects

Before any cross-model claim, write:

- architecture-conditioned equation
- scale-conditioned equation
- commensurability argument for the measurement operator
- directional prediction
- explicit falsifier

Apply:

- `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`

## 3. Bedrock Workflow

### "What" Is Not An Answer

Never stop at what happened. Always continue to why and how.

After every meaningful experiment, answer:

```text
WHAT was observed?
WHY did that observable take that value?
HOW did the computation produce it, operator by operator?
WHAT exact quantity should be measured next?
```

Observations are not explanations:

- "the claim failed"
- "AICc did not generalize"
- "the metric is mixed"
- "the model is inconsistent"

Each must be followed by:

- the operator chain
- the link that broke the prediction
- the quantity that exposes that link
- the next falsifier targeting it directly

### Failed Falsifier = Dig Deeper

A failed falsifier is a pointer to missing understanding, not a stopping point.

Protocol:

1. record the failed prediction and the observed value
2. write the exact chain from input to reported observable
3. locate the link that violates the prediction
4. identify the missing quantity at that link
5. measure it directly

The answer is always in:

- the model weights
- the kernel or operator implementation
- the MLX/JAX/CUDA backend code
- the measurement code
- the tensors you can collect

"The mechanism is unclear" is never acceptable.

### One Variable Per Day

When something fails:

1. pick one variable
2. characterize it geometrically
3. measure it before, during, and after the operation
4. work backward from failure

If you are guessing, you are missing a measurement.

## 4. Architecture And Dependency Rules

### Directory Structure

```text
src/modelcypher/
├── core/
│   ├── domain/
│   ├── ports/
│   └── use_cases/
├── adapters/
├── backends/
└── cli/
```

Dependencies point inward.

- `backends/` is the only place allowed to import ML frameworks
- `adapters/` uses the Backend protocol
- `core/` uses ports and pure logic only

Framework imports such as `mlx`, `jax`, and `torch` belong in `backends/` only.

### Prefer Backend Over NumPy

NumPy in the domain layer is a bug because it forces CPU fallback and bypasses
the Backend protocol.

Rules:

- no `import numpy` in `core/domain`
- no `to_numpy()` workarounds in domain code
- if an operation is missing, add it to the Backend protocol

### Real Geometry Depends On The Object

Use the right geometry for the right object:

- activations: curved manifold, geodesic tools when justified
- weights: flat parameter space with spectral structure, Euclidean plus SVD

Do not use geodesic machinery on weight matrices by default.

### Measure Behavioral Impact, Not Weight Magnitude

For transplant and merge metrics, use behavioral norm, not Frobenius norm.

Use:

`||X delta_W^T||`

Do not use:

`||delta_W||_F`

If the question is behavioral preservation, measure behavioral change.

## 5. Merge And Alignment Guardrails

### CKA And Alignment

Raw CKA and aligned CKA are different questions.

- low raw CKA can mean coordinate mismatch
- aligned CKA tests whether probe geometry has been matched

CKA `= 1.0` on training probes after closed-form alignment is expected by
construction. Held-out failure means probe coverage failure, not alignment math
failure.

### Add, Do Not Blend

ModelCypher uses geometric addition, not interpolation.

Wrong:

```python
merged = alpha * A + (1 - alpha) * B
```

Right:

```python
delta = source - target
projected = null_space_projection(delta, target_activations)
merged = target + projected
```

The goal is to add source structure in directions the target does not already
occupy, while preserving target behavior on sampled activations.

## 6. Measurement Guardrails

### No Vibes

Diagnostics return raw measurements, not interpretation strings.

Wrong:

```python
{"similarity": 0.73, "interpretation": "good"}
```

Right:

```python
{"similarity": 0.73}
```

### No Simulated Results

Do not claim verification by reprocessing cached JSON or reproducing arithmetic
outside the real pipeline.

Verification means:

- run the actual script or command
- on real models or real artifacts produced by that script

If you cannot run the real pipeline, say so plainly.

### Don’t Invent Heuristics

Every arbitrary threshold is a confession that the mechanism is still unknown.

Allowed numbers:

- machine epsilon and functions of it
- mathematical constants
- values derived from spectral structure
- values measured from baseline data
- values from cited theorems or papers with direct applicability

Forbidden numbers:

- round human defaults like `0.1`, `0.5`, `0.8`, `100`
- "standard practice" values
- hand-picked percentages
- any cutoff you cannot derive

If you catch yourself writing a guessed threshold:

1. stop
2. log or return the raw quantity instead
3. derive the actual constraint or design a falsifier for it

### "Fine" Is Not An Answer

"This is fine," "good enough," or "reasonable default" are all signs that the
answer has not been derived yet.

If you cannot say "this is correct, and here is the derivation," you do not have
the answer yet.

## 7. Experiment And Implementation Rules

### Smallest Viable Model First

Default to the smallest model that exposes the property under test.

- 350M to 700M: math validation, debugging, rapid iteration
- 1B to 2B: scale checks after small models pass
- 3B to 8B: final validation, not discovery

If the math is wrong at 350M, it is wrong at 8B.

### Research Before Code

Before relying on external APIs, libraries, or external docs:

1. check current official documentation
2. verify the relevant version
3. look for breaking changes

### Research Code vs CLI

Use modules and scripts for exploratory work. Promote to CLI only when the
mechanism is derived and outcomes are controllable.

CLI promotion requires:

1. correct derivation
2. direct measurements
3. reproducible improvement on real models
4. no unresolved hard guardrail violations

Stable workflows should use `mc`. Exploratory falsifiers and one-off measurement
passes may live in `scripts/` until validated.

## 8. Operating Rules

### Commands

Use `poetry` for everything:

```bash
poetry install
poetry run pytest
poetry run mc --help
```

Useful commands:

```bash
mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
mc analyze dimension-profile --model /path/to/model
mc analyze lora-svd /path/to/adapter --base /path/to/model
mc merge run -s SOURCE -t TARGET -o OUTPUT
mc infer run --model /path/to/model --prompt "Hello"
```

### Concurrency

Multiple agents may be working at once.

Rules:

1. ignore unrelated modified or untracked files
2. do not ask about unrelated changes
3. no destructive git operations such as `reset`, `checkout --`, `add`, `commit`, `push`
4. no bulk modification scripts for code edits
5. do not run tests while training is using the GPU

If training is running, wait for it to finish before `pytest`.

### Documentation

Keep documentation concise, ordered, and falsifier-oriented.

When a claim is promotable, document:

- mechanism
- equation
- measurement operator
- falsifier
- evidence status

Useful references:

- `docs/CLI-REFERENCE.md`
- `docs/GEOMETRY-GUIDE.md`
- `docs/GLOSSARY.md`
- `docs/EVIDENCE-TAXONOMY.md`

## 9. Final Standard

The most dangerous phrase in this repository is:

> this should be fine

Do not use it.

Measure. Derive. Trace the operator chain. Keep going until the answer reaches
bedrock.
