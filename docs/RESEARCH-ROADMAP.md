# Research Roadmap

**Updated:** 2026-03-08

## What This File Is For

This file answers four operational questions:

1. How far are we from `docs/VISION.md`?
2. Where do we measurably improve on standard practice today?
3. Where are we still weaker, unproven, or blocked?
4. What do we clean up next so evidence is easier to read than noise?

This file is intentionally not the running dump of every solved or refuted
thread.
`docs/research/OPEN-MATHEMATICAL-QUESTIONS.md` now carries only the active
mathematical blockers. Historical literature mapping and broader field position
live in:

- `docs/research/SOTA-AUDIT-2026-03.md`
- `docs/research/field_map_external_methods.md`
- `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`

## Executive Read

- We are ahead on controller design. The canonical training path removes manual
  learning-rate, rank, scale, and target-selection overrides from runtime code.
- We are not yet ahead on end-user reliability. The canonical pipeline still has
  unresolved behavioral failures, and 8B closure is still open.
- The repository still has a large research surface, but it is now inventoried.
  Current generated counts are `133` scripts, `90` top-level result families,
  and about `1.40G` under `results/`.

The short version is: we have reduced guessing in the control plane more than
we have proven superiority in the outcome plane.

## Vision Scorecard

| Vision promise from `docs/VISION.md` | Current state | Hard evidence | What still blocks promotion |
| --- | --- | --- | --- |
| Geometry-derived training should be one command with no manual guessing | Partial | `mc train run` exists; `pipeline_gate_v1` exists; doctrine audit removed runtime LR/scale/quantization bypasses; `results/pipeline_validation/verdict.json` still reports `all_pass=false` with 3/5 inference passes on 350M | Behavioral preservation still fails in part of the canonical path |
| Quantized models are the real target, not an afterthought | Partial | Quantization frontier precheck is implemented; `results/closedform_sequential_correction/20260227T173057Z/closedform_correction.json` improved CKA, PPL, and degeneration simultaneously on Qwen3-1.7B | We still lack the architecture-conditioned frontier law and 8B closure |
| Cross-architecture portability should make the geometry the invariant | Partial | Merge and alignment machinery exist; internal SOTA audit classifies null-space merge claims as strong enough to keep pushing | No mandatory MergeBench-style comparison against standard merge baselines |
| Nightly consolidation should turn daily interaction into stable memory | Experimental | Continual-learning code and artifact families exist (`src/modelcypher/experimental/continual/`, `src/modelcypher/experimental/use_cases/consolidation_service.py`, `results/continual_learning/`) | No promotable closure that this is repeatable, beneficial, and non-forgetting |
| Adapter stacking should preserve identity across substrates | Infrastructure partial | `src/modelcypher/experimental/self_improve/lora_stacker.py` exists | No promoted preservation certificate or supported workflow |
| Adapter sovereignty should let the user own the identity layer | Not built | No serialization, access-control, or user-owned runtime flow exists yet | This is still infrastructure and product work, not closed research |

## Where We Actually Improve On Standard Practice Today

### 1. Canonical training removes manual knob turning in a way current public tools do not

Current public fine-tuning stacks still expose user-chosen ranks, alphas,
dropout, learning rates, warmup, schedulers, and target-module choices in their
official docs:

- Hugging Face PEFT LoRA config: `r`, `target_modules`, `lora_alpha`,
  `lora_dropout`, and more
- Axolotl config reference: `lora_r`, `lora_alpha`, `lora_dropout`,
  `learning_rate`, `warmup_steps`, scheduler, optimizer
- TorchTune LoRA recipes: config and CLI overrides for LoRA rank, alpha, epochs,
  and recipe parameters
- Unsloth LoRA guide: explicit learning-rate ranges, recommended ranks, alpha,
  dropout, warmup, scheduler, and batch-size heuristics

ModelCypher's differentiator is not "we also have a recipe". It is that the
canonical runtime path tries to derive the control surface from model geometry,
IEEE 754 limits, and measured data, then aggressively removes override paths
that reintroduce guessing.

Internal evidence:

- `docs/MISSION.md`
- `docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md`
- `src/modelcypher/cli/commands/train.py`
- `src/modelcypher/core/domain/training/mass_step_size.py`
- `src/modelcypher/core/use_cases/dataset_training_service.py`

### 2. We are stronger on measurement discipline than on raw leaderboard claims

The project now has a better answer than "it looked okay in eval":

- promotion contracts for architecture, scale, precision, and operator validity
- a maintained SOTA crosswalk in `results/sota_audit_2026_03/`
- runtime guardrails such as `pipeline_gate_v1`
- explicit rejection of mixed-model "partial validation" stories

This matters because the main failure mode in the repo is not lack of
experiments. It is over-promotion from fragmented evidence.

The current internal SOTA crosswalk classifies `21` tracked claims as:

- `8` `CUTTING_EDGE`
- `5` `ADAPT_OTHERS`
- `6` `PUSH_FURTHER`
- `2` `DEPRIORITIZE`

That is real progress, but most of the "cutting edge" material is still about
measurement, falsification, or merge geometry, not yet about a complete
end-user training win.

### 3. Quantized correction work shows a real, user-relevant advantage

The strongest current "smaller-and-smarter" evidence is the corrective
quantization thread.

In `results/closedform_sequential_correction/20260227T173057Z/closedform_correction.json`
on Qwen3-1.7B:

- mean CKA improved by `+0.0139976`
- min CKA improved by `+0.180729`
- perplexity improved by `-0.0633116`
- max 4-gram repetition improved by `-0.0471629`

That is materially stronger than "quality loss is the price of quantization."
It is one of the clearest places where the repository has something better than
standard acceptance of quantization damage.

## Where We Do Not Yet Beat Standard Practice

| Gap | Evidence | Why this is not promotable yet |
| --- | --- | --- |
| Zero-guess training is not yet reliably behavior-preserving | `results/pipeline_validation/verdict.json` reports `all_pass=false`; on 350M, structural pass is 5/5 but inference pass is only 3/5 | We do not yet have the causal operator that explains when structural safety fails to preserve behavior |
| 8B closure is still open | `results/g5_8b_validation_multiseed/multiseed_gates.json` reports `n_seeds=1`, `cka_ok=0`, `degenerate_ok=0`, `all_gates_all_seeds=false` | The user-facing claim "works on any model" remains unclosed |
| We do not yet have mandatory head-to-head baselines against the standard PEFT ecosystem | The repo has strong internal doctrine and many experiments, but no mandatory same-model same-data comparison suite against standard LoRA, rsLoRA, PiSSA, EVA, DoRA, or recipe-level baselines; the retained `results/nblora_vs_standard/` family is summary-only and no longer retains a usable head-to-head benchmark payload | Without these controls, "better than standard practice" is still a thesis, not a measured result |
| Merge claims are differentiated but not yet industry-positioned | Internal SOTA audit says keep pushing null-space merge, but also says import MergeBench-style evaluation | We have internal strength but not yet benchmark parity with how the field compares merge methods |
| Identity-layer claims are ahead of infrastructure and evidence | `docs/VISION.md` still correctly marks stacking as partial and sovereignty as not built | The long-term vision remains valid as direction, not yet as delivered capability |

## Why Progress Feels Hard To See

### The repository surface is much larger than the maintained inventories imply

Current generated counts from `results/repo_research_inventory/`:

- `133` scripts under `scripts/`
- `13` scripts with exact name-matched test files under `tests/scripts/` or
  `tests/experiments/`
- `90` top-level result families under `results/`
- about `1.40G` stored under `results/`

`scripts/INVENTORY.md` is now generated from
`scripts/report_research_inventory.py`, and the full machine-readable inventory
now lives in `results/repo_research_inventory/`.

Current generated status split after collapsing duplicate and superseded result
runs into retained summary bundles:

- scripts: `5` `canonical`, `34` `summary_only`, `94` `delete`
- results: `32` `canonical`, `58` `summary_only`, `0` `delete`

### A small number of experiment families dominate the artifact footprint

The largest current result families are:

| Result family | Size | Share of `results/` |
| --- | ---: | ---: |
| `four_bit_extension` | `1.02G` | `72.7%` |
| `g5_8b_validation_memtest` | `0.06G` | `4.6%` |
| `quantization_ab_survey` | `0.06G` | `4.0%` |
| `pipeline_validation_cert_350m` | `0.04G` | `2.7%` |

Those four families account for about `84.0%` of the current `results/`
directory. Most cleanup leverage is still concentrated there.

### "Checkpoint sprawl" is really "results-as-checkpoints"

There is no top-level `checkpoints/` tree right now. Saved adapters, run logs,
and family summaries still live together under `results/`, so cleanup still has
to treat `results/` as both evidence store and checkpoint store.

## Roadmap: What We Should Do Next

### R1. Prove the zero-guess training claim against standard baselines

This is the highest-priority user-facing gap.

Required controls on the same model, data, and eval slices:

- standard LoRA
- rsLoRA
- PiSSA
- EVA
- DoRA
- recipe-level baselines where practical (TorchTune, Axolotl, or equivalent)

Promotion rule:

- do not claim "better than standard practice" unless the comparison is
  same-model, same-data, same-eval, and survives the preservation gates

### R2. Close the preservation gap before widening the claim surface

Current evidence says the structural certificates are not enough by themselves.
The next measurement pass should focus on:

- the operator behind `pipeline_validation` failures
- the link between null-space accessibility, CKA blindness, and behavioral
  degradation
- the exact condition under which a structurally safe adapter still flips task
  behavior

This is the shortest path from "interesting geometry" to "training is easier for
people".

### R3. Close the quantized-first story with an actual law, not a good run

Quantized correction is promising, but the frontier claim still needs:

- an architecture-conditioned equation from crossing severity to CKA floor
- paired FP-to-quantized sweeps at multiple bit depths
- a repeatable closure at 8B

Until then, quantized-first remains one of our strongest directions, not yet a
closed platform advantage.

### R4. Put the identity-layer vision back under a hard evidence leash

Near-term order:

1. consolidation without forgetting
2. stacking with preservation certificates
3. only then portability and sovereignty claims at the user level

The project should not talk like the identity layer is operational until the
preservation math is stronger than the narrative.

### R5. Clean the research surface so signal can win

#### Scripts

- Classify every script as one of: `canonical`, `summary_only`, `delete`
- Every script that remains live in the repo must declare:
  - owner
  - claim or question served
  - expected artifact path
  - whether it has tests
- Keep `scripts/INVENTORY.md` generated from
  `scripts/report_research_inventory.py`, not hand-maintained

#### Results

- Keep one canonical summary bundle per experiment family:
  - `REPORT.md` or equivalent narrative
  - machine-readable summary JSON
  - one canonical artifact bundle if the experiment really produced a reusable
    adapter or model
- Delete per-run bulky adapters from the worktree once their summary metrics are
  extracted and any genuinely reusable canonical artifact is retained
- Start with the four largest result families listed above

#### Claims

- Maintain the generated registry that maps:
  - claim
  - script
  - artifact path
  - evidence status
  - next falsifier

Without this, the project keeps rediscovering its own past work.

## Exit Criteria For The Next Roadmap Update

We should update this document again only after at least one of these is true:

- A baseline suite proves or refutes that the canonical training path beats
  standard practice on matched comparisons
- The 8B closure has at least three complete seeds and either passes all gates
  or fails with a traced mechanism
- The preservation operator behind `pipeline_validation` failures is identified
  and re-tested
- The script and result inventories are generated and the top storage-heavy
  result families are reduced to canonical evidence bundles plus retained
  summaries, with raw runs deleted from the worktree

## References

- `docs/VISION.md`
- `docs/MISSION.md`
- `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`
- `docs/research/SOTA-AUDIT-2026-03.md`
- `docs/research/field_map_external_methods.md`
- `docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md`
- `results/sota_audit_2026_03/scorecard.md`
- `results/repo_research_inventory/README.md`
- `results/repo_research_inventory/retention_plan.md`
- `results/pipeline_validation/REPORT.md`
- `results/g5_8b_validation_multiseed/REPORT.md`
- `results/closedform_sequential_correction/20260227T173057Z/closedform_correction.json`
- [Hugging Face PEFT LoRA docs](https://huggingface.co/docs/peft/main/package_reference/lora)
- [Axolotl config reference](https://docs.axolotl.ai/docs/config-reference.html)
- [TorchTune LoRA single-device recipe](https://docs.pytorch.org/torchtune/0.6/recipes/lora_finetune_single_device.html)
- [Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
