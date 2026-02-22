# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

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


def test_initial_lr_derivation_nonfinite_fails_fast(monkeypatch):
    adapter = MLXTrainingAdapter(_DummyBackend())
    monkeypatch.setattr(
        adapter,
        "_measure_lipschitz_robust",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_initial_learning_rate_or_fail(
            model=_DummyModel(),
            train_dataset=[],
            batch_size=1,
            seq_length=8,
            lipschitz_loss_fn=None,
            lipschitz_batches=3,
            seed=42,
            sigma_max=1.0,
            lr_override=None,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_curvature_estimate"
    diagnostics = err.diagnostics or {}
    for key in (
        "lipschitz_nonfinite",
        "hvp_failed",
        "no_active_lora_layers",
        "trainable_param_nan_inf",
        "invalid_sigma_max",
    ):
        assert key in diagnostics
    assert diagnostics["lipschitz_nonfinite"] is True


def test_entropy_baseline_derives_floor():
    adapter = MLXTrainingAdapter(_DummyBackend())

    floor = adapter._derive_entropy_floor_or_fail(
        baseline_entropy=2.0,
        dataset_samples=8,
        scope="answer_masked_training",
    )

    assert floor > 0.0
    assert floor < 2.0
