# Baranov Replication Protocol (ModelCypher-native, 2026-02)

Date: 2026-02-26  
Scope: independent replication of selected Baranov claims under ModelCypher research constraints.  
Constraint baseline: `docs/MISSION.md` and `docs/EVIDENCE-TAXONOMY.md`.

## 1. Protocol Goals

1. Replicate externally reported effects without importing external implementation.
2. Convert candidate mechanisms into geometry-derived, heuristic-free experimental modules.
3. Produce claim-level outcomes that can be labeled `[VALIDATED]`, `[EMPIRICAL]`, or `[DISPROVEN]`.

## 2. Non-goals (Intake Phase)

- No production CLI command additions.
- No promotion of experimental interfaces to stable APIs.
- No acceptance of fixed constants that are not derived from precision, spectrum, or measured baselines.

## 3. Common Experimental Contract (all tracks)

### 3.1 Required controls

- Unmodified base model control.
- `LoRA-only` and `edit-only` controls where applicable.
- Hold-out fact split (no overlap with training facts).
- Seed sweep with reported confidence intervals (single-run is insufficient for `[VALIDATED]`).

### 3.2 Required metrics

- `CKA drift` (pre vs post intervention, in-distribution and hold-out probes).
- `preserved_fraction` (behavioral norm based; not Frobenius).
- `perplexity drift` (identity text and broad-domain reference set separated).
- `recall curves` split by mode:
  - `raw_completion`
  - `chat_template`
- Capacity and geometry diagnostics:
  - coverage ratio
  - condition number
  - null rank
  - transfer strength
  - spectral gap

### 3.3 Heuristic guardrails

- Disallow hardcoded values such as fixed degraded thresholds, static stage schedules, fixed rollback percentages, and arbitrary cycle/iteration caps.
- All acceptance boundaries must be:
  - dtype-derived, or
  - spectral-derived, or
  - baseline-distribution-derived from controls in the same run family.

### 3.4 Pre-registration requirements

- Define hypothesis and rejection condition before execution.
- Record exact model IDs, quantization, backend, commit SHAs, and dataset hashes.
- Store all raw outputs needed for independent re-analysis.

## 4. Track A: Alignment-Tax Replication

### 4.1 Research question

Does post-training alignment behavior reduce recoverable factual recall under parameter-efficient updates as model scale increases, when evaluated separately for `raw_completion` and `chat_template` pathways?

### 4.2 Experimental design

- Controlled LoRA injection on at least three model scales (as available).
- Same fact pool template and split logic across scales.
- Evaluate both recall modes and geometry drift after each intervention.
- Include pre/post benchmark measurements for general capability preservation.

### 4.3 Pass/fail criteria (pre-registered)

- Replication pass (external effect):
  - Directional trend of recall suppression with scale is observed in at least one mode split, with confidence interval excluding zero effect for that split.
- Mechanism pass (ModelCypher relevance):
  - Suppression must co-occur with measurable geometry signatures (CKA drift and/or preserved-fraction collapse) rather than only narrative interpretation.
- Fail:
  - No consistent trend across scales, or trend reverses under controls, or confidence intervals include no-effect across all splits.

### 4.4 Required artifacts

- `track_a_manifest.json`
- `track_a_metrics.csv` (one row per run/seed/split)
- `track_a_recall_curves.json`
- `track_a_geometry.json`
- `track_a_summary.md`

## 5. Track B: Capacity-Threshold Replication

### 5.1 Research question

Is there a measurable interference tipping point under incremental fact injection, and does onset align with null-space/capacity metrics?

### 5.2 Experimental design

- Incremental fact injection with per-step measurement.
- Change-point detection based on observed degradation distribution (no fixed "fact count" threshold baked in).
- Compare against null-space and spectral metrics at each step.

### 5.3 Pass/fail criteria (pre-registered)

- Replication pass (external effect):
  - A statistically distinguishable change-point in degradation trajectory appears relative to baseline variability.
- Geometry pass (ModelCypher relevance):
  - Change-point coincides with capacity diagnostics (null-rank compression, spectral-gap shift, preserved-fraction drop).
- Fail:
  - No robust change-point or no geometry correlation with the detected transition.

### 5.4 Required artifacts

- `track_b_manifest.json`
- `track_b_stepwise_metrics.csv`
- `track_b_change_point.json`
- `track_b_capacity_diagnostics.json`
- `track_b_summary.md`

## 6. Track C: Sleep Convergence + Per-fact Consolidation

### 6.1 Research question

Can staged consolidation with per-fact advancement/retreat recover degraded recall while preserving behavior and renewing usable capacity?

### 6.2 Experimental design

- Start from degraded-state setups produced by Track B.
- Apply per-fact staged consolidation in cycles.
- Evaluate:
  - cycles-to-recovery,
  - stage transitions per fact,
  - capacity renewal metrics,
  - preserved behavior metrics.

### 6.3 Pass/fail criteria (pre-registered)

- Replication pass (external effect):
  - Recovery of degraded recall to baseline envelope is observed with explicit stage-transition traces.
- Safety pass (ModelCypher constraints):
  - No unacceptable behavior regression under baseline-derived acceptance bounds.
- Capacity pass:
  - Demonstrable recovery of available capacity metrics after consolidation cycles.
- Fail:
  - Recovery requires heuristic thresholds/schedules, or behavior preservation fails, or capacity does not renew.

### 6.4 Required artifacts

- `track_c_manifest.json`
- `track_c_cycle_metrics.csv`
- `track_c_stage_transitions.json`
- `track_c_capacity_renewal.json`
- `track_c_summary.md`

## 7. Experimental Interfaces (experimental-only)

No production CLI promotion in this phase. Additions stay under `src/modelcypher/experimental/`.

### 7.1 MEMIT-style editor interface

Suggested experimental datamodel:

- `FactTriple(subject: str, relation: str, object: str, fact_id: str)`
- `EditState(edit_id: str, fact_ids: list[str], layer_ids: list[int], status: str, metrics: dict[str, float])`
- `ConsolidationStage(stage_index: int, transfer_weight: float, passed: bool)`

### 7.2 Recall evaluator interface

Required API shape:

- `evaluate_recall(facts, mode="raw_completion")`
- `evaluate_recall(facts, mode="chat_template")`

Output must include per-fact outcomes and aggregate confidence intervals.

### 7.3 Consolidation progression interface

Required behavior:

- Per-fact advancement and retreat transitions.
- Rollback-safe state transitions.
- Emission of raw metrics for each decision (no opaque "pass/fail only" logic).

## 8. Test Matrix

### 8.1 Unit

- Matrix update correctness for Woodbury-equivalent formulations.
- Null-space non-interference invariants.
- Per-fact stage transition correctness, including rollback branch.

### 8.2 Integration

- End-to-end wake -> sleep run on a small fixture model.
- End-to-end mode-split evaluator (`raw_completion` + `chat_template`) on same fact set.

### 8.3 Regression

- Numeric literal audit on touched files to detect newly introduced heuristic constants.
- Regression cases for known failure mode: consolidation/pruning cascade.

### 8.4 Reproducibility

- Re-running identical config reproduces qualitative outcomes and preserves effect direction.
- Artifact completeness check (manifest + metrics + curves + summary) must pass.

## 9. Artifact Schema (minimum)

```json
{
  "run_id": "string",
  "track": "A|B|C",
  "timestamp_utc": "ISO-8601",
  "model": {
    "id": "string",
    "quantization": "string",
    "backend": "string"
  },
  "code": {
    "modelcypher_commit": "string",
    "experiment_module_commit": "string"
  },
  "data": {
    "fact_pool_hash": "sha256",
    "split_manifest_hash": "sha256",
    "reference_corpus_hash": "sha256"
  },
  "controls": {
    "base_control": true,
    "lora_only_control": true,
    "edit_only_control": true
  },
  "metrics": {
    "cka_drift": "float",
    "preserved_fraction": "float",
    "perplexity_drift_identity": "float",
    "perplexity_drift_general": "float",
    "recall_raw_completion": "float",
    "recall_chat_template": "float",
    "null_rank": "float",
    "condition_number": "float",
    "spectral_gap": "float"
  },
  "pre_registered_decision": {
    "criteria_version": "string",
    "outcome": "pass|fail|inconclusive",
    "reason": "string"
  }
}
```

## 10. Promotion Rule

A claim can only move to `[VALIDATED]` and become CLI-promotion-eligible when:

- at least one track-specific pre-registered pass condition is met,
- controls are complete,
- artifacts are reproducible,
- and no heuristic constants are required for the successful path.

## 11. Execution Status

### Patchset 1: Experimental Interfaces & Scaffolding (2026-02-26)

**Status: Implemented.**

All experimental interfaces from Section 7 are implemented:

| Protocol Section | Implementation | Status |
|-----------------|---------------|--------|
| 7.1 MEMIT-style editor interface | `EditApplicator` protocol + `EditState` data model | Protocol only (no Woodbury impl yet) |
| 7.2 Recall evaluator interface | `RecallEvaluator` protocol + `compute_recall_aggregate` | Protocol only (no concrete evaluator yet) |
| 7.3 Consolidation progression | `FactConsolidationTracker` with per-fact advance/retreat | Done |
| Section 9 artifact schema | `ReplicationManifest` + `validate_manifest` | Done |
| Section 8 test matrix | Unit + integration + numeric literal audit | Done |

Heuristic guardrails (Section 3.3):
- `transfer_weight` is always caller-provided (no fixed schedules).
- CI alpha derived from `1/n` (no arbitrary constants).
- Numeric literal audit test guards against heuristic reintroduction.
- Manifest validation rejects NaN/inf metric values and empty strings.
- Artifact writers have collision guards (`FileExistsError` on duplicate output).

### Still Pending

- Concrete evaluator and editor implementations (patchset 2).
- Track A/B/C runner scripts with real model execution.
- Multi-seed runs with artifact collection for claim-level outcomes.
