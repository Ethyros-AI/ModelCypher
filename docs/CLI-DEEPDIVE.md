# CLI Deep Dive Audit

Purpose: track CLI command coverage, help quality, and documentation alignment.

## Environment
- Host: macOS (Apple Silicon)
- Repo: /Users/jasonkempf/ModelCypher
- Models (CodeCypher drive):
  - /Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B
  - /Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-MLX-8bit
  - /Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit
- Drive: /Volumes/CodeCypher

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

## Known Issues
- write_error signature mismatch: many call sites pass output_format/pretty but function does not accept them.
- CLI help text missing for many top-level groups (blank descriptions in mc --help).
- geometry metrics commands ignore global output format (write_output called without CLI context).
- Duplicate/unused mc infer command defined in app.py; mc infer group is the actual entrypoint.

## Docs Mismatches (Initial)
- docs/FAQ.md references `mc geometry manifold analyze` (command does not exist).
- docs/INFERENCE.md references `mc infer compare` and `--max-tokens` (command/options not present in CLI).
- docs/PROFILING.md references `mc geometry crm-build` and `mc geometry density-profile` (CLI uses `mc geometry crm build` / `mc geometry density profile`).
- docs/TRAINING-GUIDE.md references `mc geometry training-status` and `training-history` (CLI uses `mc geometry training status/history`).
- docs/GLOSSARY.md references `mc geometry cka compute` (command does not exist).

