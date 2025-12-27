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
- `mc doc` (convert/validate)
- `mc infer` (run/suite)
- `mc storage` (status/usage/cleanup)
- `mc inventory`, `mc system`

Research + diagnostics:
- `mc geometry` (path/training/safety/adapter/atlas/baseline/concept/cross-cultural/primes/stitch/crm/metrics/sparse/refusal/persona/manifold/transport/refinement/invariant/emotion/merge-entropy/transfer/spatial/social/temporal/moral/waypoint/interference)
- `mc thermo` (analyze/path/path-integration/entropy/measure/detect/detect-batch/ridge-detect/phase/sweep/benchmark/parity)
- `mc entropy` (analyze/detect-distress/verify-baseline/window/conversation-track/dual-path/calibrate)
- `mc safety` (adapter-probe)
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
- `mc validate` (train), `mc estimate` (train)

## Geometry Atlas Commands
```bash
mc geometry atlas dimensionality <model_path> --layer <n>
mc geometry atlas dimensionality-study <model_path> --layer <n> [-l ...]
```

## Geometry Concept Commands
```bash
mc geometry concept detect "Text to analyze"
mc geometry concept detect "Prompt" --model <path>
mc geometry concept compare --text-a "First" --text-b "Second"
mc geometry concept compare --model-a <path> --model-b <path> --prompt "Test input"
```

## Geometry Cross-Cultural Commands
```bash
mc geometry cross-cultural analyze <input_json>
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

## Geometry Baseline Commands
```bash
# List available baselines
mc geometry baseline list
mc geometry baseline list --domain spatial

# Extract baseline from a reference model
mc geometry baseline extract <model_path> --domain spatial
mc geometry baseline extract <model_path> --domain social --layer -1 --k-neighbors 10

# Validate model against baselines
mc geometry baseline validate <model_path>
mc geometry baseline validate <model_path> --domains spatial,social --strict

# Compare two models
mc geometry baseline compare <model1_path> <model2_path> --domain spatial
```

### Baseline Output Schema
```json
{
  "_schema": "mc.geometry.baseline.extract.v1",
  "domain": "spatial",
  "modelFamily": "qwen",
  "modelSize": "0.5B",
  "ollivierRicciMean": -0.189,
  "ollivierRicciStd": 0.045,
  "manifoldHealthDistribution": {
    "hyperbolic": 1.0,
    "flat": 0.0,
    "spherical": 0.0
  },
  "intrinsicDimension": 12.4
}
```

### Curvature Reference
- **Negative Ricci curvature (< -0.1)**: Hyperbolic geometry - characteristic of high-capacity representations
- **Near-zero curvature (-0.1 to 0.1)**: Flat (Euclidean) geometry
- **Positive curvature (> 0.1)**: Spherical geometry - often indicates low-rank representations

Compare measurements against model family baselines to determine significance.

## Selected Commands

### Safety Commands
```bash
mc safety adapter-probe --adapter <path>    # Run adapter safety probes
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

## Streaming

- `mc doc convert --stream` emits NDJSON events for conversion progress.
- `mc train logs --follow` tails training logs.

## Schemas + Completions

- `mc help schema <command>` emits JSON schema for a command.
- `mc help completions {bash,zsh,fish}` generates shell completions.
