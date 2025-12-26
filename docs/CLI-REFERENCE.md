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
- `MC_TRACE_ID`
- `MC_NO_PROMPT`, `MC_ALLOW_ALL`
- `NO_COLOR`, `MC_NO_COLOR`
- `MC_NO_PAGER`

## Command Map

Primary workflows:
- `mc train` (start/preflight/status/pause/resume/cancel/logs/export)
- `mc job` (list/show/attach/delete)
- `mc checkpoint` (list/delete/export)
- `mc model` (list/register/delete/fetch/merge/search/probe/validate-merge/analyze-alignment)
- `mc dataset` (validate/preprocess/convert/preview/get-row/update-row/add-row/delete-row/list/delete/pack-asif)
- `mc doc` (convert/validate)
- `mc infer` (single run, batch, suite)
- `mc storage` (usage/status, cleanup)
- `mc inventory`, `mc system`

Research + diagnostics:
- `mc geometry` (validate/training/safety/adapter/primes/stitch/path/crm/sparse/refusal/persona/manifold/transport/atlas/spatial)
- `mc thermo` (analyze/path/entropy/measure/detect/ridge-detect/phase/sweep)
- `mc entropy` (analyze/detect-distress/verify-baseline/window/conversation-track/dual-path)
- `mc safety` (adapter-probe/dataset-scan/lint-identity)
- `mc agent` (trace-import/trace-analyze/validate-action)
- `mc eval` (run/list/show)
- `mc compare` (run/list/show/checkpoints/baseline/score)
- `mc calibration`, `mc stability`, `mc agent-eval`, `mc dashboard`
- `mc ensemble`, `mc research`, `mc help`, `mc schema`, `mc completions`
- `mc validate` (train), `mc estimate`, `mc explain`

## Geometry Atlas Commands
```bash
mc geometry atlas dimensionality <model_path> --layer <n>
mc geometry atlas dimensionality-study <model_path> --layer <n> [-l ...]
```

## Geometry Spatial Commands
```bash
mc geometry spatial euclidean <activations_file>
```

## Phase 2 Commands

### Safety Commands
```bash
mc safety adapter-probe --adapter <path>    # Run adapter safety probes
mc safety dataset-scan --dataset <path>     # Scan dataset for safety issues
mc safety lint-identity --dataset <path>    # Lint for intrinsic identity issues
```

### Entropy Commands
```bash
mc entropy window --model <path> --config <json>       # Sliding window entropy tracking
mc entropy conversation-track --session <file>         # Multi-turn conversation analysis
mc entropy dual-path --base <path> --adapter <path>    # Dual-path security analysis
```

### Agent Commands
```bash
mc agent trace-import --file <path>         # Import OpenTelemetry/Monocle traces
mc agent trace-analyze --trace <file>       # Analyze agent traces
mc agent validate-action --action <json>    # Validate agent actions
```

### Dataset Commands (Phase 2 additions)
```bash
mc dataset format-analyze <path>            # Detect dataset format (text/chat/instruction/etc)
mc dataset chunk --file <path> --size <n>   # Chunk documents for training
mc dataset template --model <family>        # Apply chat template
```

## Streaming

- `mc doc convert --stream` emits NDJSON events for conversion progress.
- `mc train logs --follow` tails training logs.

## Schemas + Completions

- `mc schema --list` to list schemas; `mc schema <key>` to emit JSON schema.
- `mc completions --shell {bash,zsh,fish}` to generate shell completions.
