#!/usr/bin/env bash
# Run ModelCypher's non-model verification suite without hosted CI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MC_DISABLE_MLX="${MC_DISABLE_MLX:-1}" \
MC_BACKEND="${MC_BACKEND:-jax}" \
poetry run pytest -m "not real_model and not slow" -q
poetry run ruff check .
poetry run mypy src/modelcypher
poetry run python scripts/generate_knob_matrix.py --check
poetry run python scripts/update_test_count.py --check
poetry check --lock

# AGENTS.md defines 20,000 tokens as the one-shot review budget.
poetry run python scripts/report_token_budget.py --threshold 20000
