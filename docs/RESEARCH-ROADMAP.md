# Research Roadmap

**Updated:** 2026-03-12

## What This File Is For

This file is the closure order for the repository.

It answers four questions:

1. what is actively blocking
   [MISSION.md](/Users/jasonkempf/ModelCypher/docs/MISSION.md),
2. what work is merely valuable but not currently blocking,
3. what is closed or archived,
4. what new agent work is allowed to create live repo surface.

This file is not a dumping ground for every interesting result. If a thread is
not on the active ladder, it is either parked or archived.

## Scope Cascade

- **Mission**: close the canonical geometric engine, centered on `mc train run`.
- **Vision**: downstream identity-layer consequences of that closure.
- **Roadmap**: the required order of work.
- **Open Questions**: only the mathematical blockers on that order.

Use this file together with:

- [MISSION.md](/Users/jasonkempf/ModelCypher/docs/MISSION.md)
- [VISION.md](/Users/jasonkempf/ModelCypher/docs/VISION.md)
- [OPEN-MATHEMATICAL-QUESTIONS.md](/Users/jasonkempf/ModelCypher/docs/research/OPEN-MATHEMATICAL-QUESTIONS.md)
- [AUTONOMOUS-RESEARCH-PROTOCOL.md](/Users/jasonkempf/ModelCypher/docs/research/AUTONOMOUS-RESEARCH-PROTOCOL.md)

## Repository Reality Check

Generated inventory on 2026-03-08 reports:

- `133` scripts under `scripts/`
- `90` top-level result families under `results/`
- `5` canonical scripts
- `31` canonical result families
- `59` `summary_only` result families
- `94` `delete` script candidates

The repo does not have an experiment shortage. It has a prioritization and
surface-area problem.

## Active

Only the items in this section are allowed to generate new canonical scripts,
new canonical result families, or repeated agent-driven run families.

| ID | Goal | Primary evidence family | Exit criterion |
| --- | --- | --- | --- |
| `R1` | Same-model same-data same-eval baseline suite against standard practice for the canonical geometry-derived LoRA path | `results/nblora_vs_standard/` | Pre-registered multi-seed comparison against standard LoRA, rsLoRA, PiSSA, EVA, DoRA, and at least one recipe-level baseline; promotion allowed only if preservation gates stay valid |
| `R2` | Causal operator for behavioral failure when structural safety passes | `results/pipeline_validation/`, `results/pipeline_validation_blindness_350M_t20/` | A pre-registered operator predicts failure before online degradation, survives intervention, and explains the retained 350M failure cases |
| `R3` | 8B non-ceiling efficacy closure | `results/g5_8b_validation_multiseed/` | The pre-registered seed set on the fixed non-ceiling eval bundle passes the declared gate set without mixed or measurement-invalid outcomes |
| `R4` | Quantization frontier law | `results/quantization_frontier/`, `results/closedform_sequential_correction/`, `results/quantization_ab_survey/` | One architecture-conditioned frontier statistic orders achieved CKA floor and degeneration across bit-depth sweeps and survives a held-out family |
| `R5` | Portable cross-architecture adapter certificate | `results/geometry_sota/` plus a MergeBench-style comparison family | A commensurable preservation certificate plus head-to-head merge baselines show portable behavior, not just probe alignment |
| `R6` | Consolidation operator that adds structure without forgetting | `results/continual_learning/` | A fixed update operator beats replay-style baselines on preservation and capacity under a frozen evaluator and comparison budget |

### R1. Baseline Suite Against Standard Practice

This is the first active blocker because "better than standard practice" is the
mission claim, not an optional benchmark.

Current state:

- `results/nblora_vs_standard/` retains standardized slices and a grid-search
  summary, but it is not yet a promotable benchmark bundle.
- The family name `results/nblora_vs_standard/` is historical. The doctrine
  question is whether the canonical `geometric_lora` path beats standard
  practice, not whether the old NB-LoRA label survives.
- The repo has no mandatory, stable same-model same-data same-eval suite across
  the PEFT baseline set.
- Immediate execution note (2026-03-12): freeze the next spend to the local
  LFM2-350M bf16 tuple with
  `data/training/benchmark_train.jsonl` /
  `data/training/benchmark_val.jsonl`; use canonical `mc train run`
  (`method=geometric_lora`, current `init_method=pissa`) as the primary arm on
  that tuple. Matched-surface controls may still use the historical harness arm
  names `standard_nb_surface`, `pissa_nb_surface`, and `dora_nb_surface`, but
  those are experiment labels, not doctrine. Only expand to additional seeds if
  the canonical geometry-derived LoRA path is competitive; otherwise pivot
  directly to `R2` on the same tuple.

Required controls:

- standard LoRA
- rsLoRA
- PiSSA
- EVA
- DoRA
- at least one recipe-level baseline such as TorchTune, Axolotl, or an
  equivalent fixed recipe

### R2. Behavioral Preservation Operator

`results/pipeline_validation/verdict.json` still reports:

- `all_pass = false`
- `all_structural_pass = true`
- `all_inference_pass = false`

This is the shortest path from "interesting geometry" to "the canonical
training path really preserves behavior."

The active work is not "collect more failures." It is to derive the operator
that links null-space access, CKA blindness, and answer degradation.

### R3. 8B Non-Ceiling Closure

Current state:

- 8B mechanical viability exists.
- `results/g5_8b_validation_multiseed/multiseed_gates.json` still has only one
  tracked retained seed and does not close the efficacy claim.

This item stays active because "works on any model" is still untrue without
pre-registered multi-seed 8B closure.

### R4. Quantization Frontier Law

Current state:

- corrective quantization is one of the strongest current directions,
- but the repo still lacks the law that predicts when geometry and behavior can
  be preserved under reduced precision.

This is the bridge between "quantized-first by doctrine" and "quantized-first
by measured control."

### R5. Portable Adapter Certificate

The merge work is important, but it is still experimental. Alignment on probes
is not yet a portable identity certificate.

This item becomes active only after R1-R4 because portability is downstream of
having a canonical engine and preservation math that already closes on a fixed
substrate.

### R6. Consolidation Operator

The continual-learning code exists, but the central question is still open:
what update law adds new structure without forgetting old structure?

This is a vision gate, not a current mission-closure substitute.

## Parked

These threads are scientifically valuable, but they are not allowed to displace
the active ladder unless they become a direct blocker for one of `R1`-`R6`.

| Thread | Why it is parked | Main artifacts |
| --- | --- | --- |
| Entropy-curvature middle chain | important for deeper theory, but not the shortest path to closing the canonical engine | `results/entropy_curvature_operator_split/`, `results/f5_sign_law_analysis_6models/`, [SOTA-AUDIT-2026-03.md](/Users/jasonkempf/ModelCypher/docs/research/SOTA-AUDIT-2026-03.md) |
| Local-ID mechanism work | mechanism-rich, but not the present blocker on mission promotion | `results/tangent_subspace_id_mechanism/` |
| DPI-compatible information replacement | important doctrine hygiene, not the immediate reason the canonical path is blocked | `results/information_bridge_linear_cka/`, [linear_accessible_information_derivation.md](/Users/jasonkempf/ModelCypher/docs/research/linear_accessible_information_derivation.md) |
| LKM capacity sweep | useful benchmark discipline and harness design | `results/lora_memory_capacity_validation/`, [LKM-AREA-1-RUN-MANIFEST.md](/Users/jasonkempf/ModelCypher/docs/research/LKM-AREA-1-RUN-MANIFEST.md) |
| Quantization A/B surface survey | good measurement surface inventory, not itself the frontier law | `results/quantization_ab_survey/` |

Parked means:

- keep summary artifacts,
- do not create new canonical work unless reactivated,
- classify new work as exploration unless it is explicitly tied back to an
  active blocker.

## Archived

Archived means "do not spend new canonical cycles here unless a later blocker
forces a return."

Examples:

- Shannon-style MI depth-decay claims
- the old Lipschitz LR derivation
- `beta_1` as a direct reasoning-success predictor
- mixed-model narratives already downgraded by the first-principles review pass

Historical material belongs in dedicated notes, archived result families, and
git history, not in the active ladder.

## Operating Rules

### No Linked Blocker, No Experiment

Every new experiment, script, or repeated run family must declare one of:

- an active roadmap item ID (`R1`-`R6`), or
- an active open-question ID from
  [OPEN-MATHEMATICAL-QUESTIONS.md](/Users/jasonkempf/ModelCypher/docs/research/OPEN-MATHEMATICAL-QUESTIONS.md).

If it cannot declare one, it is parked exploration and must not create a new
canonical repo surface.

### Inventory Status Is Binding

Use `results/repo_research_inventory/` as the triage source of truth.

- `canonical`: live evidence bucket tied to an active blocker
- `summary_only`: dormant unless explicitly reactivated by roadmap/OpenQ linkage
- `delete`: off-limits for agent resurrection unless a human explicitly
  reopens the thread

### Canonical Family Artifact Bundle

Every new canonical research family must emit:

1. `REPORT.md`
2. a machine-readable summary JSON
3. a run manifest or charter
4. an append-only ledger

Historical families do not need blanket backfill. This rule applies when a
family is newly promoted or reactivated.

## What We Should Stop Doing

- stop widening the active surface because a result is interesting
- stop treating exploratory merge or continual-learning code as mission closure
- stop creating free-range experiment families that are not tied to `R1`-`R6`
- stop promoting internal strength as field position without baseline controls

## Bottom Line

The repository should feel narrower after this file, not broader.

If a thread does not close:

1. the baseline suite,
2. the preservation operator,
3. 8B closure,
4. the quantization frontier law,
5. the portability certificate, or
6. the consolidation operator,

then it is not the work that should currently dominate the repo or the agents.
