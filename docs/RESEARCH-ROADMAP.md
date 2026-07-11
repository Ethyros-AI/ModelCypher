# Research Roadmap

**Updated:** 2026-07-11

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
| `R1` | Same-model same-data same-eval baseline suite against standard practice for the canonical geometry-derived LoRA path | `results/nblora_vs_standard/` | Pre-registered multi-seed comparison against standard LoRA, rsLoRA, PiSSA, EVA, DoRA, and at least one recipe-level baseline; promotion allowed only if preservation gates stay valid |
| `R3` | 8B non-ceiling efficacy closure | `results/g5_8b_validation_multiseed/` | The pre-registered seed set on the fixed non-ceiling eval bundle passes the declared gate set without mixed or measurement-invalid outcomes |
| `R4` | Quantization frontier law | `results/quantization_frontier/`, `results/closedform_sequential_correction/`, `results/quantization_ab_survey/` | One architecture-conditioned frontier statistic orders achieved CKA floor, fixed-basis feature survival, and degeneration across bit-depth sweeps and survives a held-out family |
| `R5` | Portable cross-architecture adapter certificate | `results/geometry_sota/` plus a MergeBench-style comparison family | A commensurable preservation certificate plus head-to-head merge baselines show portable behavior, not just probe alignment |
| `R6` | Consolidation operator that adds structure without forgetting | `results/continual_learning/` | A fixed update operator beats replay-style baselines on preservation and capacity under a frozen evaluator and comparison budget |

### R1. Baseline Suite Against Standard Practice

This remains the first downstream training blocker because "better than
standard practice" is a consequential training claim, not an optional
benchmark.

Current state (2026-03-16):

- **Stage A frozen tuple remains a no-go.** Canonical `nb_lora` won 0/7 tasks
  against every surface-matched baseline on the old benchmark pair, so seed
  expansion remains deferred.
- Fresh-session handoff for the active local 350M R1 thread lives at
  [docs/research/reports/nblora_vs_standard/REPORT.md](research/reports/nblora_vs_standard/REPORT.md).
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
  on that tuple, but neither clears R1. The 96-step MASS diagnostic is complete
  and did not rescue the controller.
- The retained R2 failure mechanism is data-format mismatch: bare-answer data
  removed the token-space work tape. Chain-preserved data partially repaired
  GSM8K, leaving arithmetic-execution granularity as the next owner-run
  falsifier. Do not reopen the old inference-CKA geometry thread.
- The family name `results/nblora_vs_standard/` is historical. The doctrine
  question is whether the canonical `geometric_lora` path beats standard
  practice, not whether the old NB-LoRA label survives.

Required controls:

- standard LoRA
- rsLoRA
- PiSSA
- EVA
- DoRA
- an applicable schedule-free control, with operator-level compatibility
  recorded for ScheduleFree+ and SF-NorMuon
- HiP-LoRA or an explicit matched-budget incompatibility record
- at least one recipe-level baseline such as TorchTune, Axolotl, or an
  equivalent fixed recipe

### R3. 8B Non-Ceiling Closure

Current state:

- 8B mechanical viability exists.
- `results/g5_8b_validation_multiseed/multiseed_gates.json` still has only one
  tracked retained seed and does not close the efficacy claim.

This item stays active because "works on any model" is still untrue without
pre-registered multi-seed 8B closure.

### Validation Tag And Retention Policy

- `[VALIDATED-ENG]` is for code, memory, artifact, and mechanical checks. One
  successful run may close only that engineering claim.
- `[VALIDATED-EFF]` is for benchmark efficacy. It requires at least 3 seeds, a
  reported seed count, a pooled effect outside 2*SE, retained per-seed artifacts,
  and a committed aggregate verdict.
- Data-rank ceiling and gradient-accumulation support are `[VALIDATED-ENG]`
  unless a separate benchmark-efficacy claim clears the `[VALIDATED-EFF]` rule.
- Raw per-seed `gates.json`, `train_result.json`, and benchmark result JSON must
  be retained until the aggregate verdict is computed and committed.

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

## Closed

### A1. Workflow-First Observation Closure

Closed 2026-07-11 as `[VALIDATED-ENG]`, not benchmark efficacy.

- `mc analyze capture`, `family`, `compare`, and `report` are canonical,
  documented command paths with CLI and README contract tests.
- Observation bundles use the `mc.analyze.bundle.v2` contract. Prompt context,
  precision state, and measurement operator each carry a required identity;
  report loading rejects identity tampering.
- The read-side report supports both v2 bundles and retained v1 artifacts
  without rerunning models or rewriting historical evidence.
- The prompt-family schema remains explicit (`case_id`, `variant_id`, `text`,
  optional comparison metadata), and the shared bundle artifacts remain
  `manifest.json`, `summary.json`, `REPORT.md`, `variants.jsonl`,
  `layer_metrics.jsonl`, and `comparisons.jsonl`.
- The public README preserves the tracked negative and single-seed verdicts;
  A1 closure does not promote the failed pipeline bundle or 8B efficacy claim.
- Hosted GitHub Actions are retired by owner policy. The tracked local gate,
  `./scripts/run_local_ci.sh`, passed with `7,558` tests, Ruff, mypy, generated
  documentation checks, lock validation, and the token-budget audit.

Real-model replication for contextual curvature, local intrinsic dimension,
and fixed-basis feature survival remains an owner-run `WS4.2` task. Those
results are not implied by this engineering closure.

### R2. Retained 350M Behavioral Failure

Closed 2026-03-16. The apparent inference-representation collapse was a
measurement artifact: CKA compared divergent generated token sequences.
Same-input geometry remained healthy. The causal intervention was the corpus
format: replacing bare-answer GSM8K examples with chain-preserved examples
partially restored the base model's reasoning-word frontier and benchmark
behavior. The remaining arithmetic-granularity experiment belongs to `R1`.

The canonical chronology and exact owner handoff are in
[docs/research/reports/nblora_vs_standard/REPORT.md](research/reports/nblora_vs_standard/REPORT.md).
Reopen `R2` only for a new behavioral failure that survives same-input,
prompt-distribution, masking, cache, decode-divergence, and data-format
controls.

## Parked

These threads are scientifically valuable, but they are not allowed to displace
the active ladder unless they become a direct blocker for one of `R1`-`R6`.

| Thread | Why it is parked | Main artifacts |
| --- | --- | --- |
| Entropy-curvature middle chain | broad chain remains parked; only the published contextual-curvature replication is active under owner-run `WS4.2` | `results/entropy_curvature_operator_split/`, `results/f5_sign_law_analysis_6models/`, [SOTA-AUDIT-2026-07.md](research/SOTA-AUDIT-2026-07.md) |
| Jacobian lens / J-space | relevant to future same-input CKA blindness, but the retained R2 geometry thread is closed | [SOTA-AUDIT-2026-07.md](research/SOTA-AUDIT-2026-07.md) |
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

- an ID currently listed under **Active** above, or
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
- stop creating free-range experiment families that are not tied to an active
  roadmap or open-question ID
- stop promoting internal strength as field position without baseline controls

## Bottom Line

The repository should feel narrower after this file, not broader.

If a thread does not close:

1. workflow-first observation closure,
2. the baseline suite,
3. 8B closure,
4. the quantization frontier law,
5. the portability certificate, or
6. the consolidation operator,

then it is not the work that should currently dominate the repo or the agents.
