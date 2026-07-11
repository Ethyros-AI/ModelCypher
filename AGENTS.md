# AI-Assisted Development Guide

ModelCypher is a measurement and observability workbench for open-source model
builders. The goal is to ship a tool that lets humans and frontier AI see what
models are doing below token level without requiring users to learn MLX
internals or hand-roll activation plumbing. Training remains shipped, but it is
downstream of the measurement layer. Every design decision is derived from the
model's geometry — that's what makes the measurements and auto-derivation
trustworthy, not a theoretical claim.

## 1. Purpose

### Principled Product

ModelCypher is a product built on derived math. Both halves of that matter.

There is one correct architecture:

- one `ModelLoader` that uses the Backend protocol
- one `ActivationProvider` that uses the Backend protocol
- one Backend abstraction for MLX, JAX, and CUDA

If duplicate paths exist, delete them.

**Ship beats prove.** A working adapter with a principled derivation beats a
theoretically superior adapter that hasn't shipped. Benchmark improvement on
real models is the arbiter.

**Product friction is a priority bug.** Stale docs, broken examples, missing
command coverage, and workflows that don't help users succeed get fixed before
new features get added.

### Mission

Make model behavior measurable below token level, then make downstream
fine-tuning accessible. Clear workflows, derived parameters, adapters that
work.

Every design decision must be derived from:

- spectral structure of weights
- geometry of activations
- IEEE 754 machine precision
- direct measurements from the model and data

### Scope Cascade

Keep these roles separate:

- **Mission** = the measurement workbench works end-to-end for real users and
  feeds the downstream training path
- **Vision** = what mission success may later enable
- **Roadmap** = the closure order
- **Open Questions** = only the mathematical blockers on that order

`mc analyze` is the clearest public entrypoint. `mc train run` remains a shipped
downstream surface. Merge, continual learning, stacking, and sovereignty remain
experimental, partial, or downstream.

### Canonical Training Identity

For doctrine, metadata, docs, and user-facing output, the shipped training
method is **geometry-derived LoRA**.

- canonical method id: `geometric_lora`
- current shipped runtime components:
  `init_method=pissa`, `optimizer=fisher_mass`, `controller=mass`,
  `stopping=geometric_certificate`
- target selection, rank derivation, MASS step sizing, and geometric stopping
  define the method
- names such as NB-LoRA, Cayley, and PiSSA should only be used when referring
  to the specific retained parameterization, helper, file, experiment arm, or
  historical result family they actually describe

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

When something fails or a design decision is being made, always answer why
and how — not just what.

When a training run degrades, when a claim doesn't hold, when a threshold
needs to be set:

```text
WHAT was observed?
WHY did that observable take that value?
HOW did the computation produce it, operator by operator?
WHAT exact quantity should be measured next?
```

This protocol applies when:
- diagnosing a failure or unexpected result
- deciding whether to promote a claim to doctrine
- choosing a threshold or parameter that will appear in production code

It does NOT apply as overhead on routine implementation tasks (writing a
test, fixing a bug, updating a doc). For those: just do it.

### Failed Falsifier = Dig Deeper

A failed prediction is a pointer to missing understanding, not a stopping point.

1. record the failed prediction and the observed value
2. write the chain from input to reported observable
3. locate the link that violates the prediction
4. identify the missing quantity at that link
5. measure it directly

The answer is always in the model weights, the backend code, or the
measurement code. "The mechanism is unclear" is never acceptable.

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
│   └── use_cases/
├── adapters/
├── backends/
├── cli/
├── experimental/
├── infrastructure/
├── ports/
└── utils/
```

Dependencies point inward.

- `backends/` is the only place allowed to import ML frameworks
- `adapters/` uses the Backend protocol
- `core/` uses `ports/` and pure logic only
- `experimental/` is not a production dependency unless a surface is explicitly
  labeled experimental

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
3. reproducible usefulness on real models
4. no unresolved hard guardrail violations

Stable workflows should use `mc`. Exploratory falsifiers and one-off measurement
passes may live in `scripts/` until validated. A surface does not need a grand
research claim to justify existing in the CLI; it does need to help users do
real work better.

### Link Every Experiment To A User Outcome

Before creating or extending any script, result family, or repeated run loop,
answer: "Does this help `mc analyze` reveal something users can act on, does it
help `mc train run` produce better adapters for users, or does it answer a
specific question about why one of those surfaces is currently failing to do
so?"

If the answer is no:

1. classify the work as parked exploration
2. do not create a new canonical script or result family
3. do not promote claims from it into mission, vision, or roadmap docs

Use `results/repo_research_inventory/` as the triage source of truth:

- `canonical` = live surface connected to a user-facing outcome
- `summary_only` = dormant unless explicitly reactivated
- `delete` = off-limits unless a human explicitly reopens the thread

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
mc analyze capture --model /path/to/model --prompt "Explain geodesics."
mc analyze family --model /path/to/model --manifest data/probes/prompt_family_minimal_pairs.json
mc analyze compare --left-model /path/to/base --right-model /path/to/base --right-adapter /path/to/adapter --manifest data/probes/prompt_family_minimal_pairs.json
mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
mc analyze reasoning-flow --model /path/to/model --prompt "Prove that sqrt(2) is irrational."
mc analyze lora-svd /path/to/adapter --base /path/to/model
mc merge run -s SOURCE -t TARGET -o OUTPUT
mc infer run --model /path/to/model --prompt "Hello"
```

Operational machine-local notes such as owner model paths, external-volume
locations, and active runbooks live in `OPERATIONS.md`. AGENTS.md remains the
single source for repository doctrine and coding rules.

### Concurrency

Multiple agents may be working at once.

Rules:

1. ignore unrelated modified or untracked files
2. do not ask about unrelated changes
3. no destructive git operations such as `reset`, `checkout --`, `add`, `commit`, `push`
4. no bulk modification scripts for code edits
5. do not run tests while training is using the GPU
6. **MANDATORY: check for GPU-using processes before any model work.** Before running training, inference, evaluation, or any script that loads a model, run `pgrep -af 'python|mlx' | grep -v grep` and confirm no other GPU-using processes are active. Multiple sessions running GPU work simultaneously will OOM and crash both. If processes are found, ask the user before proceeding.

If training is running, wait for it to finish before `pytest`.

### Autonomous Research Loops

When running repeated agent-driven experiments, follow:

- `docs/research/AUTONOMOUS-RESEARCH-PROTOCOL.md`

Minimum rules:

1. baseline first
2. freeze evaluator, probe set, and comparison budget for the whole run family
3. change one mutable surface per loop unless the operator requires a bundle
4. keep an append-only ledger for every run, including crashes and invalid measurements
5. advance only when the predicted observable survives its falsifier and no hard guardrail is violated

Retention rule for benchmark efficacy families:

- raw per-seed `gates.json`, `train_result.json`, and benchmark result JSON
  files may not be deleted until the aggregate verdict is computed and
  committed
- efficacy claims must report seed count; `[VALIDATED-EFF]` requires at least
  3 seeds with pooled effect outside 2*SE
- code, memory, artifact, and mechanics checks use `[VALIDATED-ENG]`; a
  single engineering run does not imply benchmark efficacy

For any new canonical research family, require all of:

1. `REPORT.md`
2. a machine-readable summary JSON
3. a run manifest or charter
4. an append-only ledger

Historical families do not need blanket backfill. This requirement applies when
a family is newly promoted or explicitly reactivated.

### Documentation

Keep documentation concise, ordered, and falsifier-oriented.

Lead user-facing docs with the workflow and the user outcome, not the abstract
research narrative. Broken examples, stale command signatures, and product copy
that hides what the CLI actually does are documentation bugs.

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
