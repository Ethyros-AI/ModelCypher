# Quantized-Smarter Execution Closeout (2026-03-05)

## Scope

This note captures the March 5, 2026 implementation pass for the
Qwen3.5-0.8B quantized-smarter experiment infrastructure.

No full model experiment was run in this closeout pass.
The work today was experiment-enabling code, doctrine alignment, and test
coverage so the three-arm study can be executed cleanly.

## What changed today

1. Updated the claim-audit doctrine so promoted claims must include
   `precision_state`:
   `observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)`.
2. Added adapter-aware benchmarking so corrective LoRA arms can be measured
   without overloading the training script with benchmark concerns.
3. Added explicit base-resolution sanity checks to the corrective LoRA script so
   Qwen3.5 nesting failures surface early instead of halfway through a long run.
4. Added a dedicated orchestrator for the quantized-smarter experiment rather
   than grafting sequencing logic onto existing single-purpose tools.
5. Added CI-aware predictor logic so task recovery is classified using
   Clopper-Pearson overlap, not crude sign-only comparisons.

## Code changes

### Doctrine

- `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`
  - claim form now includes `precision_state`
  - required field list now explicitly requires precision specification

### Benchmark CLI

- `src/modelcypher/cli/commands/safety/benchmark.py`
  - added optional `--adapter`
  - benchmarking now supports base model + LoRA adapter evaluation for Arms B/C
  - Arm A remains model-only benchmarking on the corrected model path

### Corrective LoRA compatibility

- `scripts/corrective_lora_training.py`
  - added resolved-base sanity checks using `resolve_model_base(...)`
  - bf16 and quantized models now fail fast if the resolved base does not expose
    expected Qwen-style `.layers` and `.embed_tokens`

### New orchestrator

- `scripts/quantized_smarter_experiment.py`
  - performs architecture preflight
  - performs `mc quantize correct` smoke test before long execution
  - performs corrective-LoRA smoke test before long execution
  - runs three experiment arms:
    1. Arm A: Tikhonov correction
    2. Arm B: corrective LoRA on q4 base
    3. Arm C: corrective LoRA on Arm A corrected base
  - runs `mc analyze benchmark --suite quick --limit 100` for baseline and arms
  - computes task-conditioned CKA for `gsm8k`, `boolq`, and `arc_easy`
  - emits:
    - `combined_results.json`
    - `report.md`
  - produces predictor verdict:
    - `cka_predictive`
    - `cka_non_predictive`
    - `insufficient_evidence`

### Tests

- `tests/cli/commands/test_safety_commands.py`
  - added benchmark help assertion for `--adapter`
  - added adapter wiring test for benchmark model loading
- `tests/scripts/test_quantized_smarter_experiment.py`
  - added orchestration sequencing coverage
  - added preflight fail-fast coverage
  - added artifact parsing coverage
- `tests/scripts/test_quantized_smarter_predictor.py`
  - added CI overlap / non-overlap classification coverage
  - added predictor verdict coverage for predictive, non-predictive, and
    insufficient-evidence regimes

## Validation status

Implementation-targeted validation completed in this pass:

```bash
poetry run pytest -n 0 \
  tests/cli/commands/test_safety_commands.py \
  tests/scripts/test_quantized_smarter_experiment.py \
  tests/scripts/test_quantized_smarter_predictor.py
```

Result recorded for this implementation pass:
- `33 passed`

## What is intentionally not done yet

1. No `pipeline_gate_v2` implementation in this pass.
2. No gate branch logic was designed ahead of experiment data.
3. No full Qwen3.5-0.8B three-arm run was executed yet.
4. No promotion claim is made yet about whether task-conditioned CKA predicts
   task accuracy recovery under quantization.

## Why this matters

This pass moves the project from "quantization as an interesting measurement
thread" to "quantization as a first-class experimental target."

The code now supports the question that actually matters:

- can we make smaller quantized models smarter,
- can we recover degraded task performance,
- and which geometric observable actually predicts that recovery?

## Resume point

When work resumes, run the orchestrator on Qwen3.5-0.8B and inspect the
predictor verdict before designing any transfer gate:

```bash
poetry run python scripts/quantized_smarter_experiment.py \
  --fp-model /Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16 \
  --q4-model /Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-4bit-g64 \
  --train-dataset data/training/benchmark_train.jsonl \
  --eval-dataset data/training/benchmark_val.jsonl \
  --benchmark-limit 100 \
  --max-iters 100 \
  --output-dir results/quantized_smarter_experiment
```

Expected outputs:

- `results/quantized_smarter_experiment/<run_id>/combined_results.json`
- `results/quantized_smarter_experiment/<run_id>/report.md`

## Decision boundary for next session

The next engineering decision should be driven by measured evidence only:

1. if task-conditioned CKA tracks significant accuracy recovery on degraded
   tasks, design the next gate around that predictor
2. if it does not, design the next gate around the observable that does
3. if evidence is indeterminate at `limit=100`, increase sample count before
   building new gate logic
