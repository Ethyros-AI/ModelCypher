from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_generator():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_knob_matrix.py"
    spec = importlib.util.spec_from_file_location("generate_knob_matrix", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_knob_matrix_readme_block_matches_training_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/generate_knob_matrix.py", "--check"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_knob_matrix_lr_row_names_canonical_adamw_default() -> None:
    generator = _load_generator()
    rows = generator.build_rows(generator.collect_facts())
    lr_row = rows[0]
    assert lr_row.number == 1
    assert (
        lr_row.current_truth
        == "default: calibrated AdamW 2e-4 cosine; MASS on research modes"
    )
    assert lr_row.status == "derived+research-mode-only"
