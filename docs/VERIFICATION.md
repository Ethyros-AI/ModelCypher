# Verification Protocols

Canonical protocols for verifying ModelCypher's key claims.

---

## 1. Merging Stability: Geometry Metrics

Command:

```bash
poetry run mc merge run -s <source_model> -t <target_model> -o <output_dir>  # experimental
```

Inspect merge output for:
- CKA preservation metrics
- Null-space projection quality
- Per-layer alignment diagnostics

Compare these raw measurements across merge strategies you test.

## 2. Model Inspection

Command:

```bash
poetry run mc model info <model_path>
poetry run mc model capacity <model_path>
```

Inspect these fields in the output:
- Architecture and layer configuration
- Per-layer spectral capacity and recommended LoRA ranks

## 2.5. Canonical CLI Workflow Smoke

Use this when you want to validate the shipped product path rather than the
service-level validation harnesses.

Command:

```bash
poetry run python scripts/canonical_train_workflow_smoke.py
```

Default inputs:
- model: `/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16`
- train data: `data/training/benchmark_train.jsonl`
- eval data: `data/training/benchmark_val.jsonl`
- output root: `results/canonical_train_workflow_smoke`

Override the model path if the default local 350M model is not present:

```bash
poetry run python scripts/canonical_train_workflow_smoke.py \
  --model-path /path/to/local/model
```

The runner verifies:
- `mc train run -> mc train evaluate -> mc train compare -> mc train export`
- generated `next_actions` are executable follow-up commands
- runtime ownership is published during training/export and cleared afterward
- quantized export rejects a competing GPU-heavy command with structured
  `active_runtime` details

Inspect these artifacts in the latest run directory:
- `summary.json`
- `REPORT.md`
- `command_logs/*.stdout.txt`
- `command_logs/*.stderr.txt`
- `payloads/*.json`

## 3. Safety: Pre-Emission Detection (Delta H)

For the full architecture and theory, see [Entropy Differential Safety](research/entropy_differential_safety.md).

Calibrate safety thresholds:

```bash
poetry run mc analyze calibrate-safety --model <model_path>
```

Test safety boundaries:

```bash
poetry run mc analyze jailbreak-test --model <model_path>
```

Inspect these fields in the output:
- `vulnerabilitiesFound`
- `meanThresholdExceedance`
- `vulnerabilityDetails[].baselineEntropy`
- `vulnerabilityDetails[].attackEntropy`
- `vulnerabilityDetails[].deltaH`

---

## Reproducing These Results

```bash
# Model inspection and capacity analysis
poetry run mc model info ./model
poetry run mc model capacity ./model

# Merge models
poetry run mc merge run -s ./model-A -t ./model-B -o ./merged  # experimental

# Safety calibration and testing
poetry run mc analyze calibrate-safety --model ./model
poetry run mc analyze jailbreak-test --model ./model

# Cross-model analysis
poetry run mc analyze benchmark --model LFM2-350M
```

For formal derivations and extended writeups, see [**Research Papers**](../papers/README.md).

---

## Verification Log (Template)

Use this format to record your own runs:

```
### YYYY-MM-DD: <Model> (<Hardware>)

Command: `poetry run mc model info <model_path>`

Results:
- architecture: <value>
- layer_count: <value>
- hidden_dim: <value>

Command: `poetry run mc analyze dimension-profile --model <model_path> --prompt "test"`

Results:
- intrinsic_dimension per layer: <values>
```
