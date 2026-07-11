from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "run_contextual_curvature_replication.py"
    spec = importlib.util.spec_from_file_location("contextual_curvature_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_uniform_logits_have_log2_vocabulary_entropy() -> None:
    script = _load_script_module()
    logits = np.zeros((3, 8))

    entropy = script.next_token_entropy_bits(logits)

    np.testing.assert_allclose(entropy, np.full(3, math.log2(8)))


def test_cross_validated_ols_recovers_linear_signal() -> None:
    script = _load_script_module()
    predictor = np.linspace(-1.0, 1.0, 100)
    target = 4.0 * predictor - 2.0

    result = script.cross_validated_ols_correlation(
        predictor,
        target,
        folds=10,
        confidence=0.95,
        rng=np.random.default_rng(3),
    )

    assert result["pooledPearson"] == pytest.approx(1.0)
    assert len(result["foldPearson"]) == 10


def test_importance_weights_are_bounded_by_paper_cap() -> None:
    script = _load_script_module()
    reference = np.linspace(0.0, 1.0, 101)
    family = np.concatenate((np.zeros(50), np.ones(2)))

    weights = script.importance_weights(
        reference,
        family,
        bins=10,
        epsilon=1e-12,
        cap=10.0,
    )

    assert np.min(weights) >= 0.1
    assert np.max(weights) <= 10.0

    outside_reference = script.importance_weights(
        reference,
        np.array([-3.0, 4.0]),
        bins=10,
        epsilon=1e-12,
        cap=10.0,
    )
    assert np.all(np.isfinite(outside_reference))


def test_jsonl_probe_loader_accepts_text_rows(tmp_path: Path) -> None:
    script = _load_script_module()
    probes = tmp_path / "probes.jsonl"
    probes.write_text('{"text": "alpha"}\n{"text": "beta"}\n', encoding="utf-8")

    assert script._load_probes(probes) == ["alpha", "beta"]
