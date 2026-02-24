# Training Guide

Current training workflow for ModelCypher.

Notes:
- In this repo, run CLI commands as `poetry run mc ...`.
- This guide covers only currently implemented training commands.

## Command Surface

Training-related commands available now:
- `mc train run`
- `mc train star`
- `mc train status`
- `mc train merge`
- `mc train export`

## Quick Start

```bash
# 1) Train a LoRA adapter with geometry-derived settings
poetry run mc train run \
  --model /path/to/base_model \
  --data /path/to/train.jsonl \
  --output /path/to/adapter

# 2) Inspect training state for an agent/model pair
poetry run mc train status \
  --agent agent-001 \
  --model /path/to/base_model

# 3) Export adapter artifacts
poetry run mc train export \
  --agent agent-001 \
  --model /path/to/base_model \
  --output /path/to/export
```

## Dataset Format

`mc train run` consumes JSONL with either:
- `{"text": "..."}`
- `{"messages": [{"role": "...", "content": "..."} ...]}`

Examples:

```json
{"text": "User: What is 2+2?\nAssistant: 4"}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

## `mc train run`

Primary command for NB-LoRA training from text data.

```bash
poetry run mc train run \
  -m /path/to/model \
  -d /path/to/data.jsonl \
  -o /path/to/adapter \
  --eval-data /path/to/eval.jsonl \
  --adaptive-lr \
  --auto-regime
```

Useful options:
- `--model`, `-m` (required)
- `--data`, `-d` (required)
- `--output`, `-o`
- `--eval-data`
- `--max-iters`
- `--seq-length`
- `--lr` (explicit override)
- `--deep`
- `--adaptive-lr/--no-adaptive-lr`
- `--lr-monotonic/--no-lr-monotonic`
- `--topo-monitor/--no-topo-monitor`
- `--dim-monitor/--no-dim-monitor`
- `--auto-regime/--no-auto-regime`

## `mc train star`

STaR loop (generate → verify → retrain) built on top of DatasetTrainingService.

```bash
poetry run mc train star \
  --model /path/to/model \
  --data /path/to/base_data.jsonl \
  --output /path/to/star_run \
  --rounds 3 \
  --problems-per-round 500
```

## `mc train status`

Show current training state for a specific agent/model pair.

```bash
poetry run mc train status --agent agent-001 --model /path/to/model
```

## `mc train merge`

Merge learned adapter state into base weights.

```bash
poetry run mc train merge \
  --agent agent-001 \
  --model /path/to/model \
  --save \
  --output /path/to/merged_model
```

## `mc train export`

Export LoRA artifacts for downstream use.

```bash
poetry run mc train export \
  --agent agent-001 \
  --model /path/to/model \
  --output /path/to/export_dir
```

## Geometry Monitoring During/After Training

Use analyze commands to inspect geometry of trained checkpoints/models.

```bash
poetry run mc analyze dimension-profile --model /path/to/model
poetry run mc analyze entropy-trajectory --model /path/to/model
poetry run mc analyze spectral-trajectory --model /path/to/model
poetry run mc analyze reasoning-flow --model /path/to/model --prompt "Solve x^2 - 5x + 6 = 0."
```

## Safety and Drift Checks

```bash
poetry run mc analyze calibrate-safety \
  --model /path/to/model \
  --prompt "Hello." \
  --output-file /tmp/calibration.json

poetry run mc analyze jailbreak-test \
  --model /path/to/model \
  --prompt "test prompt" \
  --calibration /tmp/calibration.json
```

## Troubleshooting

If a command fails unexpectedly, inspect help for live signatures:

```bash
poetry run mc train --help
poetry run mc train run --help
poetry run mc analyze --help
poetry run mc system status
```
