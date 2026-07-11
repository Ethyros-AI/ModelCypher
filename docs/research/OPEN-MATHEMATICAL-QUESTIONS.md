# Open Mathematical Questions

**Updated:** 2026-07-10

## What This File Is For

This file is the active mathematical blocker list.

A question belongs here only if all three are true:

1. it blocks promotion of a claim in
   [MISSION.md](/Users/jasonkempf/ModelCypher/docs/MISSION.md),
   [VISION.md](/Users/jasonkempf/ModelCypher/docs/VISION.md), or
   [RESEARCH-ROADMAP.md](/Users/jasonkempf/ModelCypher/docs/RESEARCH-ROADMAP.md),
2. it still needs a causal operator, equation, measurement operator, or
   falsifier,
3. it is on the active ladder in
   [RESEARCH-ROADMAP.md](/Users/jasonkempf/ModelCypher/docs/RESEARCH-ROADMAP.md).

If a topic is mainly about prioritization, repo cleanup, artifact retention, or
field positioning, it does not belong here.

## Active Questions

### Q2. When is a global MASS ceiling sufficient, and when is per-layer control required?

**Roadmap link:** `R3`

**Why this is open**

MASS is the active controller, but the scale law is still incomplete across
architectures and scales.

**Current evidence**

- the canonical controller is wired into `mc train run`
- the 350M controller no-go remains, while its retained behavioral-failure
  mechanism is closed as a data-format issue
- 8B mechanical viability exists, but efficacy closure is open

**What is missing**

- an architecture-conditioned and scale-conditioned law saying when the global
  controller is sufficient
- a falsifiable boundary between valid global control and required per-layer
  control

**Next falsifier**

Run matched global-vs-per-layer MASS experiments on the same model, data, and
preservation suite. If per-layer control does not change the failure regime,
the missing term is elsewhere.

### Q3. What is the quantization frontier law?

**Roadmap link:** `R4`

**Why this is open**

We have promising quantized correction results but not the law that predicts
when reduced precision can preserve geometry and behavior.

**Current evidence**

- `results/quantization_frontier/`
- `results/closedform_sequential_correction/`
- `results/quantization_ab_survey/`

**What is missing**

- an architecture-conditioned equation linking crossing severity to achievable
  CKA floor, fixed-basis feature survival, and degeneration behavior
- a commensurable operator across bit-depths and architectures

**Next falsifier**

Run paired FP-to-quantized sweeps across multiple bit-depths and families. If
the same frontier statistic does not order the achieved CKA floor and
degeneration outcomes, the statistic is incomplete.

### Q8. What measurements are sufficient to certify portable cross-architecture adapters?

**Roadmap link:** `R5`

**Why this is open**

We have merge machinery and portability language, but not yet a complete
certificate for transfer and stacking.

**Current evidence**

- `src/modelcypher/experimental/merge/`
- `src/modelcypher/experimental/self_improve/lora_stacker.py`
- `results/geometry_sota/analysis_summary.json`
- `results/sota_audit_2026_03/scorecard.md`
- `docs/research/merge_bedrock_audit_2026_03_10.md`

**What is missing**

- a commensurable preservation certificate across model families
- mandatory comparison against standard merge baselines
- a clear line between "alignment on probes" and "portable behavior"

**Next falsifier**

If a MergeBench-style comparison shows that null-space transfer does not match
or beat standard baselines on preserved behavior, portability remains an
internal hypothesis rather than a user-facing advantage.

The current scaffold for that run family is:

- `scripts/merge_portability_falsifier.py`
- `docs/research/merge_portability_falsifier_schema.json`

### Q9. What consolidation operator adds new user structure without forgetting old structure?

**Roadmap link:** `R6`

**Why this is open**

Nightly consolidation depends on a specific update law, not on the existence of
continual-learning code.

**Current evidence**

- `src/modelcypher/experimental/continual/`
- `src/modelcypher/experimental/use_cases/consolidation_service.py`
- `results/continual_learning/`

**What is missing**

- the exact additive or projected update operator
- a preservation metric strong enough to certify non-forgetting
- an intervention showing why the operator beats replay-style baselines

**Next falsifier**

Run before/after preservation tests under a fixed update operator and compare
to replay-style baselines. If preservation is not measurably better, the
consolidation story is still open.

## Secondary Research Threads

These are important, but they are not active blocker questions right now:

- entropy-curvature middle-chain derivation:
  [SOTA-AUDIT-2026-03.md](/Users/jasonkempf/ModelCypher/docs/research/SOTA-AUDIT-2026-03.md)
- local-ID mechanism work:
  `results/tangent_subspace_id_mechanism/`
- DPI-compatible information replacement:
  [linear_accessible_information_derivation.md](/Users/jasonkempf/ModelCypher/docs/research/linear_accessible_information_derivation.md)

If one of these becomes a direct blocker for `R1`-`R6`, it can be promoted back
into the active list. Otherwise it stays out of this file.

## Inclusion Rule Going Forward

When a question is solved, refuted, or downgraded to a roadmap-only item:

1. remove it from this file,
2. link the dedicated note or retained artifact family,
3. keep only the active blocker list here.

This file should stay short enough to read in one pass.
