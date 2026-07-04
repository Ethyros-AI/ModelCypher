# G5 8B Validation Multiseed

Retained family status: `canonical`

## What This Bundle Keeps

- Aggregate gate summary:
  `results/g5_8b_validation_multiseed/multiseed_gates.json`
- Per-seed gate payload:
  `results/g5_8b_validation_multiseed/seed42/gates.json`
- Retained training summary:
  `results/g5_8b_validation_multiseed/seed42/train_result.json`

This family now keeps the measured gate outputs and deletes the raw adapter,
duplicate capacity scans, logs, traces, and probe dump that do not add new
scientific evidence beyond the retained summaries.

## Key Measurements

Aggregate verdict from `multiseed_gates.json`:

- tracked seeds in aggregate verdict: `1`
- `all_gates_all_seeds = false`
- gate pass counts:
  - `no_crash = 1`
  - `cka_ok = 0`
  - `spectral_ok = 1`
  - `accuracy_ok = 1`
  - `degenerate_ok = 0`
  - `quantization_precheck_ok = 1`

Retained seed summary (`seed42`):

- train iterations: `80`
- stop reason: `val_stable (threshold=2.3736e-02, epoch=1)`
- baseline perplexity: `13.458090900342528`
- post perplexity: `4.669175479112682`
- mean CKA: `0.9820636693563204`
- min CKA: `0.9252629183742433`
- max spectral ratio: `0.22345896180532418`
- spectral bounds: `true`
- final online accuracy: `1.0` on `20 / 20`
- baseline online accuracy: `0.85` on `17 / 20`
- adapted max 4-gram repeat: `0.4014869888475836`
- quantization precheck crossing layers: `0 / 252`

Retained adapter fingerprint:

- `seed42/adapter/adapters.safetensors`
  SHA256 `63e57ec3125b925a8d9619a76c566c58bb7b6c8cd7f6a58bf2e86456c140b928`

## Deleted Raw Artifacts

- `results/g5_8b_validation_multiseed/seed42/adapter`
- `results/g5_8b_validation_multiseed/seed42/capacity_report.json`
- `results/g5_8b_validation_multiseed/seed42/capacity_checkpoint.json`
- `results/g5_8b_validation_multiseed/seed42/memory_trace.json`
- `results/g5_8b_validation_multiseed/seed42/probe_responses.json`
- `results/g5_8b_validation_multiseed/seed42/run.log`
- `results/g5_8b_validation_multiseed/seed43`

`seed43/capacity_report.json` and `seed43/capacity_checkpoint.json` were exact
duplicates of the corresponding `seed42` files:

- capacity report SHA256:
  `3fd73a7713cdf67911cee038e1033fcc7612936d27001bb28ce330b97de4ab45`
- capacity checkpoint SHA256:
  `21e5e1b00ffeac49b39e2bd4f699a225764a019fd1a7289620f97d4d2296634b`

The retained JSON summaries still contain historical paths to deleted raw
artifacts; those paths are intentionally preserved as provenance, not as live
worktree pointers.
