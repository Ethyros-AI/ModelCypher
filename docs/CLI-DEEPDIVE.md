# CLI Deep Dive Audit

**Status**: Historical snapshot. Superseded by `docs/REPO-AUDIT-2025-01-04.md`.

Purpose: track CLI command coverage, help quality, and documentation alignment.

## Environment
- Host: macOS 26.2 (Apple Silicon)
- Repo: /path/to/ModelCypher
- Models (CodeCypher drive):
  - /path/to/models/mlx-community/Qwen2-0.5B
  - /path/to/models/mlx-community/LFM2-1.2B-MLX-8bit
  - /path/to/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit
- Drive: /path/to/storage
- Dependency baseline (latest):
  - mlx 0.30.1
  - mlx-lm 0.30.0
  - huggingface-hub 1.2.3
  - typer 0.21.0

## Scope
- CLI command help output
- CLI command runtime behavior
- Documentation alignment (docs + README + examples)

## Test Matrix
Status: pass | fail | blocked | needs-doc-update

| Command | Status | Notes |
| --- | --- | --- |
| mc --help | pass | Works after help-only backend initialization skip. |
| mc geometry --help | pass | Previously crashed due to MLX init on help; now OK. |
| mc system status | pass | MLX probe reports NSRangeException during runtime init. |
| mc inventory --output json | pass | Lists registry + system snapshot. |
| mc storage status --output json | pass | Returns disk usage breakdown. |
| mc job list --output json | pass | Returns empty list when no jobs recorded. |
| mc checkpoint list --output json | pass | Returns empty list when no checkpoints recorded. |
| mc eval list --output json | pass | Returns existing eval history entries. |
| mc compare list --output json | pass | Returns comparison sessions. |
| mc model list --output json | pass | Registry contains stale path (see Runtime Findings). |
| mc model probe /path/to/models/mlx-community/Qwen2-0.5B | blocked | MLX probe fails; now returns structured error. |
| mc model validate-merge --source /path/to/models/mlx-community/Qwen2-0.5B --target /path/to/models/mlx-community/Qwen2-0.5B | blocked | MLX probe fails; now returns structured error. |
| mc geometry validate | blocked | Backend auto-detection fails; now returns structured error. |

## Known Issues
- write_error signature mismatch: many call sites pass output_format/pretty but function does not accept them.
- CLI help text missing for many top-level groups (blank descriptions in mc --help).
- geometry metrics commands ignore global output format (write_output called without CLI context).
- Duplicate/unused mc infer command defined in app.py; mc infer group is the actual entrypoint.
- MLX runtime crash on macOS 26.2 with M-series GPU (see MLX issue #2691).

## Runtime Findings
- `mc model list` returns a model path that no longer exists: `/path/to/hf-cache/hub/models--mlx-community--Qwen2-0.5B/...`.
- `mc model probe` and `mc model validate-merge` blocked by MLX probe failure (now handled with structured errors).
- `mc geometry validate` blocked by MLX probe failure (now handled with structured errors).

## Docs Mismatches (Initial)
- docs/FAQ.md references `mc geometry manifold analyze` (command does not exist).
- docs/INFERENCE.md references `mc infer compare` (command not present in CLI).
- docs/PROFILING.md references `mc geometry crm-build` and `mc geometry density-profile` (CLI uses `mc geometry crm build` / `mc geometry density profile`).
- docs/TRAINING-GUIDE.md references `mc geometry training-status` and `training-history` (CLI uses `mc geometry training status/history`).
- docs/GLOSSARY.md references `mc geometry cka compute` (command does not exist).

## MLX Crash Details
- Repro (fails in venv): `poetry run python -c "import mlx.core as mx; mx.array(1.0)"`
- Crash: `NSRangeException` in `mlx::core::metal::Device` during device enumeration.
- Upstream: ml-explore/mlx issue #2691 (macOS 26.x, M4-class hardware).
- Environment vars `MLX_DEVICE=cpu` / `MLX_FORCE_CPU=1` did not bypass the crash.
