# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

import pytest

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.training.exceptions import TrainingDerivationError


class _DummyBackend:
    pass


class _DummyModel:
    def trainable_parameters(self):
        return {}


def test_entropy_baseline_unavailable_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=None,
            dataset_samples=3,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_unavailable"
    assert diagnostics["dataset_samples"] == 3


def test_entropy_baseline_non_positive_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=0.0,
            dataset_samples=7,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_non_positive"
    assert diagnostics["baseline_entropy_value"] == 0.0


def test_entropy_baseline_negative_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=-1.0,
            dataset_samples=9,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_non_positive"
    assert diagnostics["baseline_entropy_value"] == -1.0


def test_spectral_ceiling_nonpositive_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_spectral_ceiling(
            sigma_k_min=0.0,
            sigma_max_global=1.0,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_adapter_geometry"
    diagnostics = err.diagnostics or {}
    assert "sigma_k_min" in diagnostics
    assert "sigma_max_global" in diagnostics


def test_spectral_ceiling_nonfinite_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_spectral_ceiling(
            sigma_k_min=float("inf"),
            sigma_max_global=1.0,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_adapter_geometry"


def test_spectral_ceiling_computes_ratio():
    adapter = MLXTrainingAdapter(_DummyBackend())

    ceiling = adapter._derive_spectral_ceiling(
        sigma_k_min=0.05,
        sigma_max_global=10.0,
    )

    assert ceiling == pytest.approx(0.005, rel=1e-6)


def test_entropy_baseline_derives_floor():
    adapter = MLXTrainingAdapter(_DummyBackend())

    floor = adapter._derive_entropy_floor_or_fail(
        baseline_entropy=2.0,
        dataset_samples=8,
        scope="answer_masked_training",
    )

    expected_floor = 2.0 * (1.0 - math.sqrt(math.ldexp(1.0, -23)))
    assert floor == pytest.approx(expected_floor, rel=1e-6)








