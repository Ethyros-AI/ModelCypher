# WS4.2 Owner-Run Replication Packet

**Status:** code and manifests prepared; real-model evidence not run

This packet turns the July SOTA review into three frozen, auditable owner runs.
It does not contain benchmark outcomes or validation tags.

| Packet | Active blocker | Executable surface | Owner-only reason |
| --- | --- | --- | --- |
| Contextual curvature and entropy | `A1` via `WS4.2` | `scripts/run_contextual_curvature_replication.py` | Requires MLX, model weights, and a long-context probe file |
| Global and local intrinsic dimension | `A1` via `WS4.2` | `mc analyze dimension-profile --local --with-mle --with-ci` | Requires MLX and local model weights |
| Fixed-basis feature survival | `R4` | `scripts/run_fixed_basis_feature_survival.py` | Requires a reference-fitted basis and real precision states |

## Preflight

Before any model-loading command, confirm no other Python or MLX process is
using the GPU:

```bash
pgrep -af 'python|mlx' | grep -v grep
```

Stop and resolve any active model process before continuing. Keep the model,
probe, precision, and operator identities emitted in each artifact bundle.

## Contextual Curvature

The tracked manifest contains the paper-specific window, fold count,
intervention scale, arm count, importance-reweighting settings, and citations.
The random seed is derived from the manifest digest.
This first local packet reweights against full-space perturbations at the
selected intervention layer. The paper pooled that reference across multiple
layers, so the emitted report must record this operator difference when results
are compared.

```bash
export MODEL_PATH=/path/to/model
export PROBE_PATH=/path/to/ordered-long-context-probes.jsonl
export RUN_ID=owner-$(date -u +%Y%m%dT%H%M%SZ)

MC_BACKEND=mlx poetry run python scripts/run_contextual_curvature_replication.py \
  --model "$MODEL_PATH" \
  --probes "$PROBE_PATH" \
  --output-dir "results/ws4_2_contextual_curvature/$RUN_ID"
```

Use `--observe-only` for the layer scan before committing the intervention
budget. The full run selects the minimum-mean-curvature layer unless
`--target-layer` was pre-registered explicitly. Review discrepancies against
King et al. before appending a decision to the emitted ledger.

## Intrinsic Dimension

The command reports the estimators separately. It does not collapse their
values into a named phenomenon or a shared ground truth.

```bash
export MODEL_PATH=/path/to/model
export PROBE_PATH=/path/to/ordered-probes.txt
export PROBE_COUNT=$(grep -cve '^[[:space:]]*$' "$PROBE_PATH")
export RUN_ID=owner-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "results/ws4_2_intrinsic_dimension/$RUN_ID"

MC_BACKEND=mlx poetry run mc --json analyze dimension-profile \
  --model "$MODEL_PATH" \
  --probes "$PROBE_PATH" \
  --samples "$PROBE_COUNT" \
  --local \
  --with-mle \
  --with-ci \
  > "results/ws4_2_intrinsic_dimension/$RUN_ID/dimension_profile.json"
```

Retain sample-convergence evidence and document disagreement between TwoNN,
MLE, and local ID rather than averaging it away.

## Fixed Basis

The safetensors basis must contain one matrix per measured layer under
`layer_<index>` or `layer.<index>`. Rows are frozen feature vectors fitted only
on the reference state.

```bash
export REFERENCE_MODEL=/path/to/full-precision-model
export CANDIDATE_MODEL=/path/to/quantized-model
export FROZEN_BASIS=/path/to/reference-basis.safetensors
export PROBE_PATH=/path/to/ordered-probes.jsonl
export RUN_ID=owner-$(date -u +%Y%m%dT%H%M%SZ)

MC_BACKEND=mlx poetry run python scripts/run_fixed_basis_feature_survival.py \
  --reference-model "$REFERENCE_MODEL" \
  --candidate-model "$CANDIDATE_MODEL" \
  --basis "$FROZEN_BASIS" \
  --probes "$PROBE_PATH" \
  --output-dir "results/r4_fixed_basis_feature_survival/$RUN_ID"
```

The runner rejects tokenization mismatch and emits raw reconstruction,
coefficient, and per-feature energy measurements without a survival threshold.

## Promotion Gate

For each completed run, retain `run_manifest.json`, `summary.json`, `REPORT.md`,
the raw JSONL file, and `ledger.tsv`. The ledger is emitted header-only because
the software cannot honestly choose `advance`, `discard`, `crash`, or
`measurement_invalid` before the owner reviews the real artifacts.
