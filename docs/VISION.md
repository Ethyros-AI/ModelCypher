# Vision

## What We Are Building Toward

Any developer should be able to specialize an open-source model for their
domain without needing to know what LoRA rank means, what a learning rate
schedule does, or how MLX is wired.

The long-term product promise is simple:

- bring a model
- bring data
- let the workbench derive the training plan
- keep only the adapters that measurably help

That is the center of gravity. Geometry is the method. The user outcome is the
story.

## Current Ship Status

| Capability | Status | What It Means Right Now |
| --- | --- | --- |
| Training workbench | `SHIPPED` | `mc train run`, `mc train evaluate`, `mc train compare`, `mc train export`, `mc train merge` exist today |
| Geometry-derived planning | `SHIPPED` | derived ranks, target surfaces, controller quantities, and post-run verification are part of the workflow |
| Benchmark advantage over standard practice | `IN PROGRESS` | the workbench runs; the head-to-head proof is still being measured |
| Quantization-first deployment story | `PARTIAL` | quantization is a stated target and active work area, but the full behavior law is still open |
| Cross-architecture portability | `EXPERIMENTAL` | merge infrastructure exists, but there is no closed portable certificate |
| Adapter stacking | `EXPERIMENTAL` | infrastructure exists, preservation guarantee does not |
| User-owned portable identity layer | `NOT YET` | downstream of portability and stacking actually working |

## Near-Term Product Goal

Make `mc train run` an obvious win for OSS model builders:

- it should remove backend-specific trivia,
- it should remove hand-tuned folklore knobs,
- and it should produce adapters that survive real evaluation.

The next hard product milestone is not a manifesto milestone. It is a user
milestone:

**show measurable benchmark improvement on real models with a workflow ordinary open-source developers can run.**

## Why Geometry Still Matters

Geometry is not the branding gimmick. It is why the workbench can derive a plan
instead of punting choices back to the user.

The value is practical:

- ranks come from measured model capacity rather than cargo-cult defaults
- controller quantities come from the run rather than copied schedules
- stopping comes from measured convergence rather than patience folklore
- preservation checks are part of the workflow rather than a manual audit

If the geometry does not improve the shipped experience, it is research. If it
reduces guesswork and improves adapters, it becomes product.

## Quantization Is The Deployment Reality

`bf16/fp16` is the derivation regime. Quantized models are the deployment
regime.

The vision is not "train in full precision and accept whatever quantization
breaks later." The vision is smaller-and-smarter behavior under measured
control. If quantized behavior diverges, we trace the operator path, identify
the missing precision term, and re-test.

That makes geometry more useful, not more abstract.

## Longer-Term Directions

Once the training workbench reliably improves real adapters, the same approach
could unlock more ambitious product surfaces:

### Portable Adapters Across Architectures

Adapters or deltas that transfer across model families under a real behavior
certificate, not just an interesting probe-alignment story.

### Adapter Stacking Without Silent Drift

Multiple adapters composing without quietly degrading preserved behavior.

### User-Owned Portable Identity

A developer's fine-tuning work carried as geometry across model upgrades and
runtime environments.

These are downstream possibilities, not current claims.

## Why This Vision Still Holds

The repo already supports the direction:

- the training workbench exists and can derive plans instead of exposing raw
  tuning knobs
- evaluation and comparison surfaces exist so results can be measured rather
  than narrated
- quantization and merge work show where the practical frontier is, even when
  those surfaces are not yet finished products

The discipline is to keep the promises honest. We do not skip from "interesting
infrastructure" to "users can rely on this" without the measurements in between.

## What This File Does Not License

- claiming ModelCypher already beats standard practice
- treating experimental infrastructure as shipped capability
- describing portability, stacking, or sovereignty as if they are done
- using mixed-model results as "partial validation"
- leading with long-range identity narratives instead of the immediate product
  outcome
