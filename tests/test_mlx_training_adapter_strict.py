# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import pytest

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.training.exceptions import TrainingDerivationError


class _DummyBackend:
    pass


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


def test_entropy_baseline_derives_floor():
    adapter = MLXTrainingAdapter(_DummyBackend())

    floor = adapter._derive_entropy_floor_or_fail(
        baseline_entropy=2.0,
        dataset_samples=8,
        scope="answer_masked_training",
    )

    assert floor > 0.0
    assert floor < 2.0
