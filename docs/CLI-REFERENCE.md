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
- `mc geometry` (validate/training/safety/adapter/primes/stitch/path)
- `mc thermo` (analyze/path/entropy/measure/detect)
- `mc eval` (run/list/show)
- `mc compare` (run/list/show/checkpoints/baseline/score)
- `mc calibration`, `mc stability`, `mc agent-eval`, `mc dashboard`
- `mc ensemble`, `mc research`, `mc help`, `mc schema`, `mc completions`
- Optional: `mc rag` (build/query/list/delete)

## Streaming

- `mc doc convert --stream` emits NDJSON events for conversion progress.
- `mc train logs --follow` tails training logs.

## Schemas + Completions

- `mc schema --list` to list schemas; `mc schema <key>` to emit JSON schema.
- `mc completions --shell {bash,zsh,fish}` to generate shell completions.
