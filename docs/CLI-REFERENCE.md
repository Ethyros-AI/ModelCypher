# CLI Reference

ModelCypher CLI preserves TrainingCypher-style workflows for non-GUI operations. Use `tc` for parity, or `mc`/`modelcypher` as aliases.

## Output + AI Mode

- `stdout` is structured output (JSON/YAML/text).
- `stderr` is diagnostics (logs, progress).
- `--ai` forces JSON output and suppresses prompts/logs; `TC_AI_MODE=1` enables the same.
- `TC_NO_AI=1` disables AI mode.

## Global Options

- `--output {text,json,yaml}`
- `--ai`
- `--pretty`
- `--quiet`, `--very-quiet`
- `--yes`, `--no-prompt`
- `--trace-id <value>`
- `--log-level {trace,debug,info,warn,error}`

Environment variables:
- `TC_AI_MODE`, `TC_NO_AI`
- `TC_OUTPUT`
- `TC_TRACE_ID`
- `TC_NO_PROMPT`, `TC_ALLOW_ALL`
- `NO_COLOR`, `TC_NO_COLOR`
- `TC_NO_PAGER`

## Command Map

Primary workflows:
- `tc train` (start/preflight/status/pause/resume/cancel/logs/export)
- `tc job` (list/show/attach/delete)
- `tc checkpoint` (list/delete/export)
- `tc model` (list/register/delete/fetch/merge/search/probe/validate-merge/analyze-alignment)
- `tc dataset` (validate/preprocess/convert/preview/get-row/update-row/add-row/delete-row/list/delete/pack-asif)
- `tc doc` (convert/validate)
- `tc infer` (single run, batch, suite)
- `tc rag` (build/index/query/list/delete/status)
- `tc storage` (usage/status, cleanup)
- `tc inventory`, `tc system`

Research + diagnostics:
- `tc eval` (run/list/show)
- `tc compare` (run/list/show/checkpoints/baseline/score)
- `tc geometry` (validate/training/safety/adapter/primes/stitch/path)
- `tc thermo` (analyze/path/entropy/measure/detect)
- `tc calibration`, `tc stability`, `tc agent-eval`, `tc dashboard`
- `tc ensemble`, `tc research`, `tc help`, `tc schema`, `tc completions`

## Streaming

- `tc doc convert --stream` emits NDJSON events for conversion progress.
- `tc train logs --follow` tails training logs.

## Schemas + Completions

- `tc schema --list` to list schemas; `tc schema <key>` to emit JSON schema.
- `tc completions --shell {bash,zsh,fish}` to generate shell completions.
