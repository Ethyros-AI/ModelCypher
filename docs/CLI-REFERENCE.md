# CLI Reference

Reference for the current `mc` CLI surface.

Notes:

- In this repo, run commands as `poetry run mc ...`.
- Global options can appear anywhere on the command line.
- Use `poetry run mc --help` and `poetry run mc <group> --help` for live
  signatures.

## Common Workflow

The main shipped workflow is:

```bash
poetry run mc system status
poetry run mc model info /path/to/model
poetry run mc model capacity /path/to/model
poetry run mc train run -m /path/to/model -d /path/to/data.jsonl --plan-only
poetry run mc train run -m /path/to/model -d /path/to/data.jsonl -o /path/to/adapter
poetry run mc train evaluate -m /path/to/model -a /path/to/adapter -d /path/to/validation.jsonl
poetry run mc train compare -m /path/to/model --adapter-a /path/to/adapter -d /path/to/validation.jsonl
poetry run mc train export -m /path/to/model --adapter /path/to/adapter -o /path/to/deployment --target deployment_quantized
```

## Global Options

| Option | Description |
| --- | --- |
| `--ai` | Force JSON output and suppress prompts/logs |
| `--output json|yaml|text` | Output format |
| `--json`, `-j` | Shorthand for `--output json` |
| `--text` | Shorthand for `--output text` |
| `--quiet`, `-q` | Suppress info logs |
| `--very-quiet`, `-qq` | Suppress all logs |
| `--yes`, `-y` | Auto-confirm prompts |
| `--no-prompt` | Fail if confirmation is required |
| `--pretty`, `-p` | Pretty-print structured output |
| `--log-level` | Set log level |
| `--trace-id` | Set trace ID for diagnostics |

## Command Groups

| Group | Purpose |
| --- | --- |
| `train` | Train, evaluate, compare, export, and merge adapters |
| `data` | Prepare data for training |
| `merge` | Experimental geometric model merging |
| `infer` | Prompt inference and suites |
| `analyze` | Geometry, safety, and benchmark diagnostics |
| `model` | Model registry, inspection, capacity, and quantization |
| `system` | System status, probes, cache benchmarks |
| `adapter` | Adapter geometry and baseline calibration |
| `quantize` | Quantization correction |

## `mc train`

Commands:

- `run`
- `evaluate`
- `compare`
- `export`
- `merge`
- `status`
- `validate-derived`
- `star`

```bash
poetry run mc train run -m /path/to/model -d /path/to/data.jsonl -o /path/to/adapter
poetry run mc train run -m /path/to/model -d /path/to/data.jsonl --plan-only
poetry run mc train run -m /path/to/model -d /path/to/data.jsonl --explain --benchmark quick
poetry run mc train evaluate -m /path/to/model -a /path/to/adapter -d /path/to/validation.jsonl
poetry run mc train evaluate -m /path/to/model -a /path/to/adapter --prompts /path/to/prompts.jsonl
poetry run mc train evaluate -m /path/to/model -a /path/to/adapter --benchmark quick
poetry run mc train compare --result-a run1.json --result-b run2.json
poetry run mc train compare -m /path/to/model --adapter-a /path/to/adapter -d /path/to/validation.jsonl
poetry run mc train compare -m /path/to/model --adapter-a /path/to/a --adapter-b /path/to/b -d /path/to/validation.jsonl
poetry run mc train export --model /path/to/model --adapter /path/to/adapter --output /path/to/deployment --target deployment_quantized
poetry run mc train merge --agent agent-001 --model /path/to/model --save --output /path/to/merged
poetry run mc train status --agent agent-001 --model /path/to/model
poetry run mc train validate-derived --model /path/to/model --data /path/to/data.jsonl --trials 5
poetry run mc train star -m /path/to/model -d /path/to/base_data.jsonl -o /path/to/star_run
```

`mc train run` surfaces a `derived_plan` object in JSON/YAML output. The plan is
organized as:

- `inputs`
- `data_plan`
- `adaptation_surface`
- `controller_plan`
- `derived_now`
- `measured_during_training`
- `verified_after_training`
- `removed_user_knobs`

Text mode keeps the summary concise. Full per-module rank maps live in
structured output and in `training_plan.json` saved next to adapter artifacts.

## `mc data`

Commands:

- `prepare`

```bash
poetry run mc data prepare /path/to/source --output /path/to/train.jsonl
```

Use this when your source data is not already in the JSONL format expected by
`mc train run`.

## `mc model`

Commands:

- `list`
- `add`
- `delete`
- `info`
- `capacity`
- `moe-profile`
- `quantize`

```bash
poetry run mc model list
poetry run mc model add /path/to/model --alias my-model
poetry run mc model info /path/to/model
poetry run mc model capacity /path/to/model --sort-by recommended-rank --emit-lora-config /tmp/lora_capacity.yaml
poetry run mc model moe-profile /path/to/model --dataset /path/to/data.jsonl
poetry run mc model quantize /path/to/model /path/to/output --bits 4
poetry run mc model delete my-model --force
```

## `mc system`

Commands:

- `status`
- `probe`
- `memory-profile`
- `test-cache`
- `commands`
- `benchmark cache`

```bash
poetry run mc system status
poetry run mc system status --require-backend mlx
poetry run mc system probe backends
poetry run mc system test-cache /path/to/model --pairs 10
poetry run mc system memory-profile --model /path/to/model
poetry run mc system commands
poetry run mc system benchmark cache
```

## `mc infer`

Commands:

- `run`
- `suite`

```bash
poetry run mc infer run --model /path/to/model --prompt "What is 2+2?"
poetry run mc infer run --model /path/to/model --adapter /path/to/adapter --prompt "Explain modus tollens."
poetry run mc infer suite --model /path/to/model --suite /path/to/suite.jsonl
```

## `mc analyze`

Subcommand families:

- Geometry: `geodesic-compare`, `geodesic-profile`, `geodesic-trajectory`,
  `concept-volume`, `dimension-profile`, `entropy-trajectory`,
  `expansion-ratio`, `reasoning-flow`, `spectral-trajectory`,
  `jacobian-trace`, `verification-depth-profile`
- Behavioral and safety: `adapter-probe`, `behavioral-signature`,
  `cognitive-reflection-test`, `calibrate-safety`, `jailbreak-test`,
  `probe-redteam`, `probe-behavioral`, `bilm-probe-info`
- Benchmark and monitoring: `benchmark`, `lora-svd`, `sparse-region`,
  `knowledge-type`, `curriculum-profile`, `circuit-breaker`, `persona`,
  `uncertainty-modes`, `entropy-pattern`, `entropy-baseline-verify`,
  `crm-build`, `crm-compare`

```bash
poetry run mc analyze dimension-profile --model /path/to/model
poetry run mc analyze reasoning-flow --model /path/to/model --prompt "Prove that sqrt(2) is irrational."
poetry run mc analyze lora-svd /path/to/adapter --base /path/to/model
poetry run mc analyze calibrate-safety --model /path/to/model --prompt "Hello." --output-file /tmp/calibration.json
poetry run mc analyze jailbreak-test --model /path/to/model --prompt "test prompt" --calibration /tmp/calibration.json
poetry run mc analyze crm-build /path/to/model --output /tmp/model.crm.json
poetry run mc analyze crm-compare /tmp/source.crm.json /tmp/target.crm.json
```

## `mc merge`

Commands:

- `run`
- `batch`

```bash
poetry run mc merge run -s /path/to/source_model -t /path/to/target_model -o /path/to/output
poetry run mc merge run -s /path/to/source_model -t /path/to/target_model -o /path/to/output --behavior-jacobian
poetry run mc merge batch -s /path/to/source_a -s /path/to/source_b -t /path/to/target_model -o /path/to/output
```

This surface is useful, but it is still experimental relative to the training
workbench.

## `mc adapter`

Commands:

- `analyze`
- `calibrate-baseline`

```bash
poetry run mc adapter analyze /path/to/adapter --base-model /path/to/model
poetry run mc adapter calibrate-baseline --output-artifact /tmp/adapter_baseline.json
```

## `mc quantize`

Commands:

- `correct`

```bash
poetry run mc quantize correct -q /path/to/quantized_model -f /path/to/fp_model -o /path/to/corrected
```

## Live Discovery

```bash
poetry run mc --help
poetry run mc train --help
poetry run mc train run --help
poetry run mc train evaluate --help
poetry run mc train compare --help
poetry run mc model --help
poetry run mc data --help
```
