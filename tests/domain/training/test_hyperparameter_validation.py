# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.hyperparameter_validation import (
    TrainingHyperparameterValidator,
)
from modelcypher.core.domain.training.types import ComputePrecision, Hyperparameters


def _params(**overrides) -> Hyperparameters:
    base = dict(
        batch_size=2,
        learning_rate=1e-3,
        epochs=3,
        sequence_length=512,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        mixed_precision=True,
        compute_precision=ComputePrecision.FLOAT32,
        warmup_steps=10,
        weight_decay=0.01,
        seed=42,
        deterministic=False,
        optimizer_type="geometric",
    )
    base.update(overrides)
    return Hyperparameters(**base)


def test_comprehensive_violations_valid_case() -> None:
    violations = TrainingHyperparameterValidator.comprehensive_violations(_params())
    assert violations == []


def test_comprehensive_violations_blocking_constraints() -> None:
    bad = _params(
        batch_size=0,
        sequence_length=64,
        learning_rate=0.0,
        epochs=0,
        gradient_accumulation_steps=0,
        warmup_steps=-1,
        weight_decay=-0.1,
    )

    violations = TrainingHyperparameterValidator.comprehensive_violations(bad)

    fields = {v.field for v in violations}
    assert "batch_size" in fields
    assert "sequence_length" in fields
    assert "learning_rate" in fields
    assert "epochs" in fields
    assert "gradient_accumulation_steps" in fields
    assert "warmup_steps" in fields
    assert "weight_decay" in fields
    assert any(v.is_blocking for v in violations)


def test_comprehensive_violations_non_blocking_recommendations() -> None:
    warn = _params(
        batch_size=TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD + 1,
        sequence_length=TrainingHyperparameterValidator.SEQUENCE_WARNING + 1,
        learning_rate=TrainingHyperparameterValidator.LR_INFO_LOW / 2.0,
        epochs=TrainingHyperparameterValidator.EPOCHS_MAX_REC + 1,
        gradient_accumulation_steps=TrainingHyperparameterValidator.GRAD_ACCUM_MAX + 1,
    )

    violations = TrainingHyperparameterValidator.comprehensive_violations(warn)

    assert all(v.is_blocking is False for v in violations)
    messages = [v.message for v in violations]
    assert "Batch size exceeds recommended range" in messages
    assert "Sequence length exceeds recommended range" in messages
    assert "Learning rate below recommended range" in messages
    assert "Epochs exceed recommended maximum" in messages
    assert "Gradient accumulation steps exceed recommended maximum" in messages


def test_learning_rate_upper_bound_and_validate_for_engine() -> None:
    too_high = _params(learning_rate=TrainingHyperparameterValidator.LR_MAX * 2.0)
    violations = TrainingHyperparameterValidator.comprehensive_violations(too_high)

    assert any(v.field == "learning_rate" and v.is_blocking for v in violations)

    with pytest.raises(ValueError, match="Invalid Configuration"):
        TrainingHyperparameterValidator.validate_for_engine(
            _params(batch_size=0)
        )

    # Non-blocking warnings should not raise.
    TrainingHyperparameterValidator.validate_for_engine(
        _params(epochs=TrainingHyperparameterValidator.EPOCHS_MAX_REC + 1)
    )
