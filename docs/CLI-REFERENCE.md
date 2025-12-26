# CLI Reference

ModelCypher CLI is standalone. Use `mc` (or `modelcypher`) for all commands.

## Output + AI Mode

- `stdout` is structured output (JSON/YAML/text).
- `stderr` is diagnostics (logs, progress).
- `--ai` forces JSON output and suppresses prompts/logs; `MC_AI_MODE=1` enables the same.
- `MC_NO_AI=1` disables AI mode.

## Global Options

- `--output {text,json,yaml}`
- `--ai`
- `--pretty`
- `--quiet`, `--very-quiet`
- `--yes`, `--no-prompt`
- `--trace-id <value>`
- `--log-level {trace,debug,info,warn,error}`

Environment variables:
- `MC_AI_MODE`, `MC_NO_AI`
- `MC_OUTPUT`

## Command Map

Primary workflows:
- `mc train` (start/preflight/status/pause/resume/cancel/export/logs)
- `mc job` (list/show/attach/delete)
- `mc checkpoint` (list/delete/export)
- `mc model` (list/register/delete/fetch/merge/search/probe/validate-merge/validate-knowledge/analyze-alignment/vocab-compare)
- `mc dataset` (validate/preprocess/convert/preview/get-row/update-row/add-row/delete-row/list/delete/pack-asif/quality/auto-fix/format-analyze/chunk/template)
- `mc doc` (convert/validate)
- `mc infer` (run/suite)
- `mc storage` (status/usage/cleanup)
- `mc inventory`, `mc system`

Research + diagnostics:
- `mc geometry` (path/training/safety/adapter/atlas/primes/stitch/crm/metrics/sparse/refusal/persona/manifold/transport/refinement/invariant/emotion/merge-entropy/transfer/spatial/social/temporal/moral/waypoint/interference)
- `mc thermo` (analyze/path/entropy/measure/detect/detect-batch/ridge-detect/phase/sweep/benchmark/parity)
- `mc entropy` (analyze/detect-distress/verify-baseline/window/conversation-track/dual-path/calibrate)
- `mc safety` (adapter-probe/dataset-scan/lint-identity)
- `mc agent` (trace-import/trace-analyze/validate-action)
- `mc eval` (run/list/show)
- `mc compare` (run/list/show/checkpoints/baseline/score)
- `mc adapter` (inspect/project/wrap-mlx/smooth/merge)
- `mc calibration` (run/status/apply)
- `mc stability` (run/report)
- `mc agent-eval` (run/results)
- `mc dashboard` (metrics/export)
- `mc ensemble` (create/run/list/delete)
- `mc research` (sparse-region/afm)
- `mc help` (ask/completions/schema)
- `mc explain`
- `mc validate` (train/dataset), `mc estimate` (train)

## Geometry Atlas Commands
```bash
mc geometry atlas dimensionality <model_path> --layer <n>
mc geometry atlas dimensionality-study <model_path> --layer <n> [-l ...]
```

## Geometry Spatial Commands
```bash
mc geometry spatial anchors
mc geometry spatial probe-model <model_path>
mc geometry spatial analyze <activations_file>
mc geometry spatial euclidean <activations_file>
mc geometry spatial gravity <activations_file>
mc geometry spatial density <activations_file>
mc geometry spatial cross-grounding-feasibility <source_activations> <target_activations>
mc geometry spatial cross-grounding-transfer <source_activations> <target_activations> --concepts <file>
```

## Selected Commands

### Safety Commands
```bash
mc safety adapter-probe --adapter <path>    # Run adapter safety probes
mc safety dataset-scan --dataset <path>     # Scan dataset for safety issues
mc safety lint-identity --dataset <path>    # Lint for intrinsic identity issues
```

### Entropy Commands
```bash
mc entropy window '[[3.5, 0.2], [3.6, 0.1]]' --size 50          # Sliding window entropy tracking
mc entropy conversation-track --session <file>                  # Multi-turn conversation analysis
mc entropy dual-path '[{"base": [3.5, 0.2], "adapter": [3.8, 0.3]}]'  # Base vs adapter divergence
```

### Agent Commands
```bash
mc agent trace-import --file <path>         # Import OpenTelemetry/Monocle traces
mc agent trace-analyze --trace <file>       # Analyze agent traces
mc agent validate-action --action <json>    # Validate agent actions
```

### Dataset Commands
```bash
mc dataset format-analyze <path>            # Detect dataset format (text/chat/instruction/etc)
mc dataset chunk --file <path> --output-file <file> --size <n>   # Chunk documents for training
mc dataset template --model <family>        # Apply chat template
```

## Streaming

- `mc doc convert --stream` emits NDJSON events for conversion progress.
- `mc train logs --follow` tails training logs.

## Schemas + Completions

- `mc help schema <command>` emits JSON schema for a command.
- `mc help completions {bash,zsh,fish}` generates shell completions.
