# ModelCypher Mission

## Mission Statement

**Make model fine-tuning accessible: one command, derived parameters, honest evidence about whether the adapter helped.**

ModelCypher is a training workbench for open-source model builders. You prepare
data, derive the training plan, run training, and measure whether the model
improved without needing to know MLX internals, guess at LoRA rank, or cargo
cult a learning rate schedule.

Every training decision that enters the shipped path must come from the model
and the data:

- spectral structure of the weights
- geometry of the activations when relevant
- IEEE 754 machine precision
- direct measurements from the training run

The model tells us what it needs. The product's job is to measure that and make
it usable.

## What Users Should Be Able To Do

The core workflow is:

```bash
poetry run mc train run --model /path/to/model --data /path/to/data.jsonl --output /path/to/adapter
```

And then:

```bash
poetry run mc train evaluate --model /path/to/model --adapter /path/to/adapter --data /path/to/validation.jsonl
poetry run mc train compare --model /path/to/model --adapter-a /path/to/a --adapter-b /path/to/b --data /path/to/validation.jsonl
```

The workbench is doing its job when a developer can move through this sequence
without manual hyperparameter tuning:

`prepare -> inspect -> plan -> train -> evaluate -> compare -> export`

## Current Reality (2026-03-16)

What is true today:

- `mc train run` is a shipped training surface.
- The current runtime path is `geometry-derived LoRA`.
- The workbench can derive plans, train adapters, and evaluate or compare
  results.
- The CLI already exposes the surrounding workflow: `data prepare`,
  `model info`, `model capacity`, `train evaluate`, `train compare`,
  `train export`, and `train merge`.

Current limitations:

- The repo has **not** yet closed a promotable claim that the current training
  path beats standard practice head-to-head.
- Benchmark advantage is still something to measure and earn, not narrate into
  existence.
- Experimental merge, stacking, and continual-learning paths are not the core
  shipped promise.

That is the discipline of this mission: the workbench is real, the derivation
story is real, and benchmark superiority is still open.

### R2 Investigation Finding (2026-03-16)

The R2 "inference CKA collapse" investigation revealed that:

- Representation geometry is preserved during prompt processing (CKA mean
  0.93-0.98 on same inputs across all masking conditions).
- The benchmark failure was step-0 decode divergence caused by training data
  that stripped the reasoning chain needed for multi-step math.
- Chain-preserved retraining improved GSM8K from 1/10 to 4/10 and shifted
  the generation pattern toward the base model's reasoning-word frontier.
- The remaining gap is arithmetic-execution granularity: the model needs
  the intermediate computations decomposed at a finer grain than GSM8K chains
  provide.

This means the workbench's geometry-derived planning, CKA verification, and
spectral bounds are working as designed. The product gap is now in training
data quality — what we teach, at what granularity, in what order.

## Canonical Training Identity

The shipped training method is **geometry-derived LoRA**.

- runtime identity: `method=geometric_lora`
- current shipped components:
  `init_method=pissa`, `optimizer=fisher_mass`, `controller=mass`,
  `stopping=geometric_certificate`
- the method-defining surface is the derivation itself:
  target selection, rank derivation, MASS step sizing, preservation telemetry,
  and geometric stopping
- older experiment names remain historical labels, not the current user-facing
  product identity

## Why Geometry-Derived Training

Most fine-tuning advice in the open-source world is borrowed folklore:

- rank 8 or 16 usually works
- `2e-4` is a safe learning rate
- cosine decay helps
- warm up for a while
- stop when patience expires

Those are measurements somebody else made on different models, different data,
and different hardware.

ModelCypher's bet is simpler: the weights and the run already contain better
information than folklore defaults. SVD, spectral bounds, precision limits, and
measured controller quantities tell us more about what the model can absorb
than a copied recipe does.

Geometry matters here because it reduces guesswork. The user value is not
"research purity." The user value is: fewer knobs, clearer plans, and better
odds of producing a useful adapter.

## Three Product Pillars

### 1. Teach the Loop

Data is a control surface. Training examples do not just teach facts — they
teach answer contracts, loop structure, when to compress, and when to keep
uncertainty alive.

Chain granularity must match the model's current skill level. A model that
hasn't learned single-digit arithmetic won't benefit from word-problem chains
that assume `180/5 = 36` is a primitive. Curriculum from arithmetic
primitives → place-value → multi-digit operations → word problems.

### 2. Preserve the Loop at Decode

The adapter must not collapse intermediate computation. Greedy decode,
scratchpad exemplars, and tool use are different operators with different
failure modes. Bare-number training actively suppressed the base model's
chain-of-thought; chain-preserved training partially restored it.

### 3. Escalate to Tools

When internal looping exceeds reliable capacity, the model should reach for
tools (calculators, Python, etc.). The goal for small and medium models is
general-human competence plus tool awareness, not calculator replacement.

### Geometry as Instrumentation

Geometry is the instrumentation layer that makes all three pillars
trustworthy:

- **Pillar 1:** Does the data teach the right latent process? (CKA,
  activation geometry, corpus audit, logit divergence analysis)
- **Pillar 2:** Does decode collapse or preserve the internal computation?
  (step-0 divergence, chain CKA, generation trace analysis)
- **Pillar 3:** Is the model fighting or specializing within the prior?
  (spectral bounds, readout alignment)

The geometry work is the reason the training advice is trustworthy. It is not
an end in itself.

## Build Standards

These are implementation standards for the product. They are why the derived
surfaces are trustworthy.

### G1: Zero Arbitrary Hyperparameters

Every training parameter must come from exactly one of:

| Source | Examples |
| --- | --- |
| Spectral structure of `W` | `sigma_max`, `sigma_k`, effective rank, tail dimensions, spectral gaps |
| IEEE 754 precision | `eps`, `sqrt(eps)`, machine epsilon |
| Measured run state | loss, gradient statistics, controller quantities, validation signals |

If a number cannot be derived from one of these sources, it does not belong in
the shipped path.

Representative replacements:

| Decision | Derived surface |
| --- | --- |
| Learning rate | `eta_step = min(eta_ceiling, eta_sps, eta_weyl)` |
| LoRA rank | tail capacity from spectral structure |
| Target modules | layers with measured room for low-rank adaptation |
| Batch size | gradient-noise-derived quantity |
| Early stopping | measured convergence and certificate signals |
| Dropout | spectral ratios, not a copied constant |

### G2: Spectral Safety By Measured Control

The adapter must respect the base model's spectral structure. The training path
must monitor spectral budget and fail closed when the measured perturbation
exceeds the derived safe envelope.

### G3: Data-Derived Convergence

Training stops when the run shows convergence, not when a human-set patience
counter expires. Validation behavior, controller signals, and the geometric
certificate are part of the stopping surface.

### G4: Preservation Of Existing Capabilities

The adapter should not silently trash what the base model already knew.
Preservation checks belong in the workbench, not as an optional afterthought.

Current preservation surfaces include:

- CKA-style representation checks
- degeneration and repetition checks
- pipeline-gate verification

### G5: Works Across Models And Datasets

The workflow should adapt to the model rather than forcing model-specific logic
into the user experience. If the surface only works for one model family, it is
not a finished product behavior.

### G6: Verifiable Quality

Every claim about training quality must be backed by a direct measurement. Every
reported measurement must trace to a real command, real model, and real
artifact.

### G7: Falsifiability Before Narrative

No broad claim gets promoted into doctrine because it sounds right. Product
claims still need:

- mechanism
- math
- measurement
- falsifier

## Architecture

### One Training Pipeline

```text
dataset
  -> model inspection
  -> derived adaptation surface
  -> geometry-derived LoRA initialization
  -> Fisher-MASS optimization
  -> preservation and budget telemetry
  -> geometric stopping
  -> post-training verification
  -> adapter artifact
```

### Current Implementation Surfaces

| Surface | Role |
| --- | --- |
| `src/modelcypher/cli/commands/train.py` | user-facing train/evaluate/compare/export commands |
| `src/modelcypher/core/use_cases/geometry_training_service.py` | orchestrates the geometry-derived training workflow |
| `src/modelcypher/core/use_cases/training_comparison_service.py` | compares runs and adapters |
| `src/modelcypher/core/domain/training/geometric_lora.py` | derives LoRA surfaces from model structure |
| `src/modelcypher/core/domain/training/geometric_optimizer.py` | controller and optimizer derivation |
| `src/modelcypher/core/domain/training/spectral_budget.py` | budget monitoring |
| `src/modelcypher/core/domain/training/geometric_early_stopping.py` | convergence logic |
| `src/modelcypher/core/domain/training/pipeline_gate.py` | post-training gate |

## Canonical Inference Model

A forward pass is a deterministic geometric map:

```text
h_0 = Embed(prefix)
h_{l+1} = T_l(h_l)
h_L = (T_{L-1} o ... o T_0)(h_0)
logits = W_out h_L + b
```

Softmax is an observer-side normalization at readout. Probability is a way to
describe uncertainty at the interface; it is not the causal mechanism inside
the model.

This matters for training because the job is to change weight geometry so the
trajectory lands in a better place, not to inherit somebody else's tuning lore.

## "Fine" Is Not An Answer

"Fine," "good enough," and "reasonable default" are warnings that a number or
decision has not been derived yet.

The standard for anything that enters the shipped path is:

`this is correct, and here is the derivation`

If we do not have that, we keep measuring.

## References

| Paper | What We Use |
| --- | --- |
| Amari (1998) | Natural-gradient framing for curvature-aware optimization |
| Loizou et al. (2020) | Stochastic Polyak step-size intuition |
| Weyl (1912) | Spectral perturbation bounds |
| Shuttleworth et al. (2024) | LoRA perturbation analysis |
| Roy and Vetterli (2007) | Shannon effective rank |
| Kornblith et al. (2019) | CKA for representation similarity |
| Marchenko and Pastur (1967) | Random-matrix noise edge reasoning |
| Eckart-Young (1936) | Low-rank approximation via SVD |
