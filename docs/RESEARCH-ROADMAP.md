# Research Roadmap

**Updated:** 2026-03-26

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

- **Mission**: close the measurement-first workbench so users can see what a
  model is doing below token level and turn those measurements into better
  downstream decisions.
- **Vision**: downstream training, portability, and identity-layer
  consequences of that closure.
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
| `A1` | Observation-bundle closure for workflow-first `mc analyze` | `results/measurement_atlas/`, `results/pipeline_validation/`, and CLI/service contract tests | `capture`, `family`, `compare`, and `report` are stable, documented, and produce commensurable observation bundles for prompt and target studies without legacy safety-first packaging or buried command paths |
| `R1` | Same-model same-data same-eval baseline suite against standard practice for the canonical geometry-derived LoRA path | `results/nblora_vs_standard/` | Pre-registered multi-seed comparison against standard LoRA, rsLoRA, PiSSA, EVA, DoRA, and at least one recipe-level baseline; promotion allowed only if preservation gates stay valid |
| `R2` | Causal operator for behavioral failure when structural safety passes | `results/pipeline_validation/`, `results/pipeline_validation_blindness_350M_t20/` | A pre-registered operator predicts failure before online degradation, survives intervention, and explains the retained 350M failure cases |
| `R3` | 8B non-ceiling efficacy closure | `results/g5_8b_validation_multiseed/` | The pre-registered seed set on the fixed non-ceiling eval bundle passes the declared gate set without mixed or measurement-invalid outcomes |
| `R4` | Quantization frontier law | `results/quantization_frontier/`, `results/closedform_sequential_correction/`, `results/quantization_ab_survey/` | One architecture-conditioned frontier statistic orders achieved CKA floor and degeneration across bit-depth sweeps and survives a held-out family |
| `R5` | Portable cross-architecture adapter certificate | `results/geometry_sota/` plus a MergeBench-style comparison family | A commensurable preservation certificate plus head-to-head merge baselines show portable behavior, not just probe alignment |
| `R6` | Consolidation operator that adds structure without forgetting | `results/continual_learning/` | A fixed update operator beats replay-style baselines on preservation and capacity under a frozen evaluator and comparison budget |

### A1. Workflow-First Observation Closure

This is the shortest path from "interesting geometry" to "a usable measurement
workbench." It closes whether users and agents can actually run controlled
prompt and target studies without stitching together one-off expert commands.

Current state (2026-04-02):

- `mc analyze capture`, `mc analyze family`, and `mc analyze compare` now define
  the canonical workflow surface.
- `mc analyze report --bundle ...` now closes the read-side loop for existing
  observation bundles and retained measurement-atlas artifacts without
  regenerating artifacts.
- The follow-on research surface for token-level prompt and generation tracing
  now lives in `scripts/run_measurement_atlas.py`, writing
  `results/measurement_atlas/<run_id>/` without promoting a new CLI verb yet.
- The retained 350M atlas rerun at
  `results/measurement_atlas/20260402T150954Z-measurement-atlas/` closes the
  live/replay replay-alignment bug from `20260402T145540Z`. Agreement improved
  from `0/4 -> 4/4`, `0/2 -> 2/2`, and `1/4 -> 4/4` across the shipped study
  pack while keeping `errorCount = 0`.
- The prompt-family manifest is explicit rather than transform-driven:
  `case_id`, `variant_id`, `text`, optional `tags`, optional
  `comparison_to`, and optional `annotations` for research studies.
- Every run is expected to emit the same observation bundle contract:
  `manifest.json`, `summary.json`, `REPORT.md`, `variants.jsonl`,
  `layer_metrics.jsonl`, `comparisons.jsonl`.
- Phase 1 scope is inference-first and checkpoint-comparison-first. Live
  training-stream telemetry remains deferred.
- Remaining atlas work is now presentation and study-pack refinement:
  keep the retained atlas reports compact and decision-friendly, expand or tune
  the frozen study pack only when a concrete read-side question needs it, and
  preserve the retained alignment-closure evidence in
  `results/measurement_atlas/REPORT.md`.

### R1. Baseline Suite Against Standard Practice

This remains the first downstream training blocker because "better than
standard practice" is a consequential training claim, not an optional
benchmark.

Current state (2026-03-12):

- **Stage A frozen tuple remains a no-go.** Canonical `nb_lora` won 0/7 tasks
  against every surface-matched baseline on the old benchmark pair, so seed
  expansion remains deferred.
- Fresh-session handoff for the active local 350M R1 thread lives at
  [results/nblora_vs_standard/REPORT.md](/Users/jasonkempf/ModelCypher/results/nblora_vs_standard/REPORT.md).
  Use that file as the single re-entry point before starting new work on this
  blocker.
- The old benchmark pair
  (`data/training/benchmark_train.jsonl` /
  `data/training/benchmark_val.jsonl`) now serves as a retained mechanical
  proof substrate, not the active R1 closure corpus.
- The live local spend is the quick-aligned tuple:
  `data/training/r1_quick_aligned_train.jsonl` /
  `data/training/r1_quick_aligned_val.jsonl` on the local LFM2-350M bf16 model
  copy.
- The default canonical controller is safer than the MASS matched-trace branch
  on that tuple, but neither clears R1. The next spend is the fixed 96-step
  matched-trace MASS diagnostic recorded in the handoff note, not seed
  expansion or benchmark widening.
- The family name `results/nblora_vs_standard/` is historical. The doctrine
  question is whether the canonical `geometric_lora` path beats standard
  practice, not whether the old NB-LoRA label survives.

Required controls:

- standard LoRA
- rsLoRA
- PiSSA
- EVA
- DoRA
- at least one recipe-level baseline such as TorchTune, Axolotl, or an
  equivalent fixed recipe

### R2. Behavioral Preservation Operator

This is the shortest path from "interesting geometry" to "the canonical
measurement and training path really preserves behavior."

Current state (2026-03-26):

Use [results/nblora_vs_standard/REPORT.md](/Users/jasonkempf/ModelCypher/results/nblora_vs_standard/REPORT.md)
as the source of truth for this thread. The older geometry-collapse summary
below has been superseded there by a data-format and arithmetic-granularity
mechanism.

All surface-level explanations for the canonical path's benchmark degradation
have been eliminated. The R2 falsifier chain:

| Falsifier | Status | Ledger row |
|-----------|--------|------------|
| Optimizer (Cayley vs AdamW) | closed | r2 behavioral probe / V3 structural |
| MASS matched-trace step sizing | closed | r2 behavioral probe adamw |
| Closed-loop layer freeze | closed | r2 closed loop cayley |
| Loop mechanics (seq_len, batch, iter cap, early stops) | closed | r2 loop parity |
| Cosine LR schedule drift | closed | r2 adamw cosine schedule audit |
| **Inference-representation collapse** | **ACTIVE** | — |

The surviving mechanism: train-space CKA stays healthy (0.94) while
inference-manifold CKA collapses (min 0.13). The adapter memorizes training
format while destroying inference geometry. This persists across all optimizer,
loop, and schedule configurations tested.

Next falsifier and exact command: see `results/nblora_vs_standard/REPORT.md`
§ Exact Next Falsifier. That file is the single canonical handoff for all
R1/R2 work.

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

1. workflow-first observation closure,
2. the baseline suite,
3. the preservation operator,
4. 8B closure,
5. the quantization frontier law,
6. the portability certificate, or
7. the consolidation operator,

then it is not the work that should currently dominate the repo or the agents.
