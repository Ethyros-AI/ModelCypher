# G5 8B Validation

Retained family status: `canonical`

## What This Bundle Keeps

- Per-seed gate payload:
  `results/g5_8b_validation/seed41/gates.json`
- Retained training summary:
  `results/g5_8b_validation/seed41/train_result.json`
- Generated eval-set artifacts:
  - `results/g5_8b_validation/non_ceiling_eval_set_8b.json`
  - `results/g5_8b_validation/non_ceiling_eval_set_smoke_1p7b.json`

This family now keeps the measured failure summaries and the generated eval-set
inputs while deleting the raw adapter, duplicated capacity scans, logs, traces,
and probe dump.

## Key Measurements

Retained seed summary (`seed41`):

- train iterations: `20`
- stop reason:
  `online_eval_degraded (stage=pre_outcome, 21/25 correct, epoch=0)`
- baseline perplexity: `13.458090900342528`
- post perplexity: `9.703324600537686`
- mean CKA: `0.984034806849911`
- min CKA: `0.9064829799300109`
- max spectral ratio: `0.06158391859755699`
- spectral bounds: `true`
- baseline online accuracy: `0.96` on `24 / 25`
- final online accuracy: `0.84` on `21 / 25`
- adapted max 4-gram repeat: `0.10747663551401865`

Gate verdict:

- `no_crash = true`
- `cka_ok = false`
- `spectral_ok = true`
- `accuracy_ok = false`
- `degenerate_ok = false`

Retained adapter fingerprint:

- `seed41/adapter/adapters.safetensors`
  SHA256 `571f4a5e90dbc2a55bf395cdc301369ae92cdf50eb59574e1a724bdec8ab15a9`

The deleted capacity payloads matched the same Qwen3-8B capacity scan already
seen in `results/g5_8b_validation_multiseed/`:

- capacity report SHA256:
  `3fd73a7713cdf67911cee038e1033fcc7612936d27001bb28ce330b97de4ab45`
- capacity checkpoint SHA256:
  `21e5e1b00ffeac49b39e2bd4f699a225764a019fd1a7289620f97d4d2296634b`

## Deleted Raw Artifacts

- `results/g5_8b_validation/seed41/adapter`
- `results/g5_8b_validation/seed41/capacity_report.json`
- `results/g5_8b_validation/seed41/capacity_checkpoint.json`
- `results/g5_8b_validation/seed41/memory_trace.json`
- `results/g5_8b_validation/seed41/probe_responses.json`
- `results/g5_8b_validation/seed41/run.log`
- `results/g5_8b_validation/seed41/seed41_rerun_capture.log`

The retained JSON summaries still contain historical paths to deleted raw
artifacts; those paths are preserved as provenance rather than live worktree
pointers.
