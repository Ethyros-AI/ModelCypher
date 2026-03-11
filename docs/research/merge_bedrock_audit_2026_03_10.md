# Merge Bedrock Audit

**Date:** 2026-03-10

**Scope:** `R5 / Q8` only. This note is about merge algebra, geometry, and portability certification. It is not LoRA-process work and it is not a continual-learning implementation note.

## Why This Note Exists

Merge doctrine and merge evidence were out of sync.

- doctrine still says cross-architecture portability is `PARTIAL / EXPERIMENTAL`
- the claim registry had merge claims promoted beyond what their artifact bundles justified
- the current merge pipeline exposes real mechanisms, but still mixes them with exploratory selection policy

This note records the operator chain, the claim-status correction, and the falsifier contract required before any portable-behavior promotion.

## Claim Audit

| Claim ID | Current classification | Why it is not promotable yet |
| --- | --- | --- |
| `CR-MRG-001` | `[EXPLORATORY]` | The repo has a plausible additive merge operator and real integration evidence, but not yet a complete portable-behavior certificate with frozen evaluator, baseline set, precision-state accounting, and claim-form-complete artifact bundle. |
| `CR-MRG-002` | `[MEASUREMENT_INVALID]` | Probe-aligned structure similarity is not itself a commensurable certificate of preserved behavior across model families. Alignment on probes and portable behavior are different observables. |

## Exact Operator Chain

The merge story currently contains three distinct mathematical objects.

1. Coordinate-resolution operator `F` or `F_layer`
   Resolves source and target representation coordinates so relational structure can be compared or reconstructed in one basis.

2. Occupancy / preservation projector `P_null`
   Constrains additive weight updates to directions the target does not already occupy under sampled activations.

3. Behavior-preservation projector `P_beh`
   Uses per-probe Jacobians to preserve behavior under a gradient-defined local constraint instead of an activation-covariance null space.

The two merge cases are:

- same-dimensional transplant
  `W' = W_t + (W_s_aligned - W_t) P_null`

- cross-dimensional transplant
  reconstruct source behavior in target coordinates with `F_in` and `F_out`, then apply the same preservation operator to the reconstructed target-coordinate weight

The promotable observable remains:

`observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`

## Measurement Contract

Portable merge claims must report all of these on the same frozen evaluator bundle.

- transfer observable: did source structure appear in held-out target-coordinate behavior
- preservation observable: how much target behavior survived after the additive operator
- degeneration observable: repetition, collapse, or coherence loss
- quantized retention observable: delta between bf16 reference behavior and quantized behavior under the same measurement operator
- commensurability argument: why the measurement operator is comparable across the model families under test

Probe-aligned CKA may remain in the artifact set, but it is not by itself a portability certificate.

## Heuristic Audit

These branches remain exploratory policy unless and until they are derived from precision limits, spectral structure, or measured baselines.

| Surface | Current treatment in code |
| --- | --- |
| Transmission-layer ranking | raw `transmission_layer_scores` are now emitted; quartile/median selection remains explicitly exploratory policy |
| Density opportunity on skipped layers | raw `density_dominance_margin = 2 * max_weight - 1` is emitted instead of only a hidden `> 0.5` interpretation |
| MLP scale divergence | raw `mlp_scale_observations_by_layer` and `scale_divergence_spectrum` are emitted; automatic env-var correction / whole-layer reversion is removed from the active path |
| Projector identity | raw `projector_mode_by_layer`, `projector_mode_by_weight`, and `projector_mode_counts` are emitted |
| HOT coupling | raw `layer_coupling_mass_by_layer` and pipeline-level coupling summaries are emitted when available |
| Probe coverage | raw `probe_rank_coverage` is emitted from the frozen rank-augmentation measurements |

The remaining selection rules are still useful for research-time execution, but they are now clearly separated from promotable measurements.

## Falsifier Contract

The research scaffold for `Q8` is:

- script: `scripts/merge_portability_falsifier.py`
- contract module: `src/modelcypher/experimental/merge/falsifier_contract.py`
- schema note: `docs/research/merge_portability_falsifier_schema.json`

Required bundle files:

- `REPORT.md`
- `summary.json`
- `manifest.json`
- `ledger.jsonl`

The frozen manifest includes:

- one evaluator bundle
- one held-out probe set
- one comparison budget
- one quantization policy
- the exact arm set
- the exact measurement set
- the claim-form fields needed for promotion review

## Decision Rule

Promote only if projector-based merge beats applicable baselines on preserved behavior without violating degeneration or quantized-retention controls.

If it fails, classify the failure as one of:

- coordinate-resolution failure
- projector / operator failure
- measurement invalidity

"Interesting but mixed" is not a closure state for `Q8`.
