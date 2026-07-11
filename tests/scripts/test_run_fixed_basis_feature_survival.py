from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "run_fixed_basis_feature_survival.py"
    spec = importlib.util.spec_from_file_location("fixed_basis_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("key", "layer"),
    (("layer_0", 0), ("layer.17", 17)),
)
def test_basis_layer_key_contract(key: str, layer: int) -> None:
    script = _load_script_module()
    assert script._basis_layer_index(key) == layer


def test_basis_contract_rejects_ambiguous_or_non_matrix_entries() -> None:
    script = _load_script_module()
    with pytest.raises(ValueError, match="Unsupported"):
        script._basis_layer_index("encoder.block.0")
    with pytest.raises(ValueError, match="not a matrix"):
        script._resolve_basis_layers({"layer_0": np.ones(3)})


def test_help_exposes_all_frozen_inputs() -> None:
    script = _load_script_module()
    help_text = script.build_parser().format_help()
    for flag in (
        "--reference-model",
        "--candidate-model",
        "--basis",
        "--probes",
        "--output-dir",
    ):
        assert flag in help_text


def test_jsonl_probe_loader_accepts_text_rows(tmp_path: Path) -> None:
    script = _load_script_module()
    probes = tmp_path / "probes.jsonl"
    probes.write_text('{"text": "alpha"}\n{"text": "beta"}\n', encoding="utf-8")

    assert script._load_probes(probes) == ["alpha", "beta"]
