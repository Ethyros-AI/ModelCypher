# Quantization Frontier Precheck v1 Implementation (2026-03-05)

## Summary

Implemented `quantization_frontier_precheck_v1` as the new quantization gate for
training runs that provide an FP reference model.

The gate now measures activation-aware FP-vs-quantized divergence on shared probe
texts and blocks only when that centered-Gram operator cannot be measured. Raw
Weyl crossing is retained as nested telemetry and no longer blocks training by
itself.

This closes the implementation gap identified by:
- `docs/research/quantization_geometry_deep_dive.md`
- `docs/research/quantization_frontier_bedrock_review_2026_03_05.md`

## Mathematical Contract

The implemented precheck measures hidden-output probe geometry, not raw
weight-space crossing severity:

- Per-layer observables: `cka`, `gram_epsilon`, `cka_bound`
- Hidden-output probe spectrum: covariance eigenvalues from centered FP probe activations
- Effective dimension: `hidden_probe_d_eff = (sum lambda)^2 / sum(lambda^2)`
- Effective boundary: `hidden_probe_k_eff = ceil(D_eff)`
- Effective gap: `hidden_probe_gap_eff` at the `k_eff` singular boundary
- Output perturbation ratio: `hidden_probe_rho_out = ||Y_q - Y_fp||_2 / gap_eff`

Important scope constraint:

- `hidden_probe_d_eff` is derived from hidden-output probe activations and is not
  numerically comparable to the deep dive's input-covariance `D_eff ≈ 3`.
- The precheck predicts base FP-vs-quantized divergence, not correction reach.
  Corrective training can be compensatory rather than restorative.

## Code Changes

### New domain components

- `src/modelcypher/core/domain/training/quantization_frontier_precheck.py`
  - Adds `run_quantization_frontier_precheck_v1(...)`
  - Adds `make_quantization_frontier_precheck_payload_v1(...)` so invalid payload
    schema lives in one place
- `src/modelcypher/core/domain/training/matrix_norms.py`
  - Adds shared exact `compute_spectral_norm(...)`

### Updated training orchestration

- `src/modelcypher/core/use_cases/dataset_training_service.py`
  - Renames `research_allow_quantization_crossing` to
    `research_allow_quantization_frontier_invalid`
  - Renames result field `quantization_precheck` to
    `quantization_frontier_precheck`
  - Reorders flow to:
    1. load model
    2. load/split dataset
    3. derive probe texts / effective seq length
    4. run frontier precheck
    5. continue existing training flow
  - Fails closed only on invalid frontier measurement with
    `failure_class="quantization_frontier_unavailable"`
- `src/modelcypher/core/use_cases/_dataset_training_service_helpers_mixin.py`
  - Adds shared probe-text activation collection and stacking helpers

### Updated documentation

- `docs/research/quantization_geometry_deep_dive.md`
  - `gap_eff` explicitly conditioned on `D_eff`
  - precheck semantics clarified as base divergence, not correction reach
- `docs/research/quantization_frontier_bedrock_review_2026_03_05.md`
  - same two amendments added
  - previously questioned arXiv citations `2507.18553` and `2602.02001`
    verified on 2026-03-05 and retained
- `docs/MISSION.md`
  - mission text updated from raw-Weyl fail-closed gate to activation-aware
    frontier gate with raw-Weyl telemetry

## Verification

Tests run after implementation:

```bash
poetry run pytest \
  tests/domain/training/test_quantization_frontier_precheck.py \
  tests/domain/training/test_quantization_weyl_precheck.py \
  tests/test_dataset_training_service_strict.py \
  -k "quantization_frontier or quantization_precheck or quantization_weyl"
```

Result: `14 passed`

Covered behaviors:

- reducer nominal case
- insufficient probes
- no overlapping layers
- degenerate centered Gram
- non-finite metric handling
- canonical invalid payload construction
- ordering: frontier runs before init-adapter merge
- valid frontier proceeds even with severe raw Weyl crossings
- invalid frontier fails closed without override
- invalid frontier proceeds with explicit research override
- no reference path means no frontier precheck runs

Repo hygiene checks completed:

- old names removed repo-wide:
  - `research_allow_quantization_crossing`
  - bare `quantization_precheck`

## Remaining Work

Implementation is complete, but research closure is still open:

1. Run paired FP↔quant validation artifacts across bit-depths and models.
2. Map activation-aware frontier observables against observed CKA floors and
   correction ceilings.
3. If direct comparison to the deep dive's `D_eff ≈ 3` is required, add
   input-covariance telemetry at the exact q/k/v and up/gate input sites rather
   than using hidden-output probes alone.
