# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for training hyperparameter validation."""

import pytest

from modelcypher.core.domain.training.types import ComputePrecision, Hyperparameters
from modelcypher.core.domain.training.validation import (
    TrainingHyperparameterValidator,
    Violation,
)


def _hyperparameters(**overrides):
    values = {
        "batch_size": 4,
        "learning_rate": 3e-5,
        "epochs": 3,
        "sequence_length": 1024,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "mixed_precision": True,
        "compute_precision": ComputePrecision.FLOAT16,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "seed": 42,
        "deterministic": True,
        "optimizer_type": "adamw",
    }
    values.update(overrides)
    return Hyperparameters(**values)


class TestViolation:
    """Tests for Violation dataclass."""

    def test_violation_required_fields(self):
        v = Violation(
            field="batch_size",
            message="Invalid batch size",
            is_blocking=True,
        )
        assert v.field == "batch_size"
        assert v.message == "Invalid batch size"
        assert v.is_blocking is True
        assert v.suggestion is None

    def test_violation_with_suggestion(self):
        v = Violation(
            field="learning_rate",
            message="Too high",
            is_blocking=False,
            suggestion="Try 3e-5",
        )
        assert v.suggestion == "Try 3e-5"


class TestBatchSizeValidation:
    """Tests for batch size validation rules."""

    def test_batch_size_zero_is_blocking(self):
        params = _hyperparameters(batch_size=0)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].is_blocking is True

    def test_batch_size_negative_is_blocking(self):
        params = _hyperparameters(batch_size=-1)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].is_blocking is True

    def test_batch_size_1_to_4_no_violation(self):
        for bs in range(
            TrainingHyperparameterValidator.BATCH_SIZE_RANGE.start,
            TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD + 1,
        ):
            params = _hyperparameters(batch_size=bs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            batch_violations = [v for v in violations if v.field == "batch_size"]
            # 1-4 should have no violations
            assert len(batch_violations) == 0, f"batch_size={bs} should have no violations"

    def test_batch_size_5_to_8_non_blocking(self):
        for bs in range(
            TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD + 1,
            TrainingHyperparameterValidator.BATCH_SIZE_RANGE.stop,
        ):
            params = _hyperparameters(batch_size=bs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            batch_violations = [v for v in violations if v.field == "batch_size"]
            assert len(batch_violations) == 1, f"batch_size={bs} should have non-blocking violation"
            assert batch_violations[0].is_blocking is False

    def test_batch_size_9_is_blocking(self):
        params = _hyperparameters(batch_size=TrainingHyperparameterValidator.BATCH_SIZE_RANGE.stop)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].is_blocking is True


class TestSequenceLengthValidation:
    """Tests for sequence length validation rules."""

    def test_sequence_below_min_is_blocking(self):
        params = _hyperparameters(
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_MIN - 1
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].is_blocking is True

    def test_sequence_at_min_no_violation(self):
        params = _hyperparameters(sequence_length=TrainingHyperparameterValidator.SEQUENCE_MIN)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 0

    def test_sequence_normal_range_no_violation(self):
        mid = (
            TrainingHyperparameterValidator.SEQUENCE_MIN
            + TrainingHyperparameterValidator.SEQUENCE_WARNING
        ) // 2
        for seq in [
            TrainingHyperparameterValidator.SEQUENCE_MIN + 1,
            mid,
            TrainingHyperparameterValidator.SEQUENCE_WARNING,
        ]:
            params = _hyperparameters(sequence_length=seq)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            seq_violations = [v for v in violations if v.field == "sequence_length"]
            assert len(seq_violations) == 0, f"sequence_length={seq} should have no violations"

    def test_sequence_above_warning_threshold_non_blocking(self):
        params = _hyperparameters(
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_WARNING + 1
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].is_blocking is False

    def test_sequence_at_max_is_non_blocking(self):
        params = _hyperparameters(sequence_length=TrainingHyperparameterValidator.SEQUENCE_MAX)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        # 4096 is at max but > warning threshold, so non-blocking
        assert len(seq_violations) == 1
        assert seq_violations[0].is_blocking is False

    def test_sequence_above_max_is_blocking(self):
        params = _hyperparameters(
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_MAX + 1
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].is_blocking is True


class TestLearningRateValidation:
    """Tests for learning rate validation rules."""

    def test_lr_below_min_is_blocking(self):
        params = _hyperparameters(
            learning_rate=TrainingHyperparameterValidator.LR_MIN / 10
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].is_blocking is True

    def test_lr_at_min_is_non_blocking(self):
        params = _hyperparameters(learning_rate=TrainingHyperparameterValidator.LR_MIN)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        # 1e-6 is valid but < INFO threshold (1e-5), non-blocking
        assert len(lr_violations) == 1
        assert lr_violations[0].is_blocking is False

    def test_lr_normal_range_no_violation(self):
        mid = (
            TrainingHyperparameterValidator.LR_INFO_LOW
            + TrainingHyperparameterValidator.LR_WARN_HIGH
        ) / 2
        for lr in [
            TrainingHyperparameterValidator.LR_INFO_LOW * 2,
            mid,
            TrainingHyperparameterValidator.LR_WARN_HIGH,
        ]:
            params = _hyperparameters(learning_rate=lr)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            lr_violations = [v for v in violations if v.field == "learning_rate"]
            assert len(lr_violations) == 0, f"learning_rate={lr} should have no violations"

    def test_lr_above_warning_threshold_non_blocking(self):
        params = _hyperparameters(
            learning_rate=TrainingHyperparameterValidator.LR_WARN_HIGH
            + (TrainingHyperparameterValidator.LR_MAX - TrainingHyperparameterValidator.LR_WARN_HIGH)
            / 2
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].is_blocking is False

    def test_lr_at_max_is_non_blocking(self):
        params = _hyperparameters(learning_rate=TrainingHyperparameterValidator.LR_MAX)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        # 1e-3 is at max but > warning threshold, non-blocking
        assert len(lr_violations) == 1
        assert lr_violations[0].is_blocking is False

    def test_lr_above_max_is_blocking(self):
        params = _hyperparameters(
            learning_rate=TrainingHyperparameterValidator.LR_MAX * 2
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].is_blocking is True


class TestEpochsValidation:
    """Tests for epochs validation rules."""

    def test_epochs_zero_is_blocking(self):
        params = _hyperparameters(epochs=TrainingHyperparameterValidator.EPOCHS_MIN - 1)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].is_blocking is True

    def test_epochs_negative_is_blocking(self):
        params = _hyperparameters(epochs=TrainingHyperparameterValidator.EPOCHS_MIN - 2)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].is_blocking is True

    def test_epochs_1_to_10_no_violation(self):
        for epochs in range(
            TrainingHyperparameterValidator.EPOCHS_MIN,
            TrainingHyperparameterValidator.EPOCHS_MAX_REC + 1,
        ):
            params = _hyperparameters(epochs=epochs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            epoch_violations = [v for v in violations if v.field == "epochs"]
            assert len(epoch_violations) == 0, f"epochs={epochs} should have no violations"

    def test_epochs_above_max_is_non_blocking(self):
        params = _hyperparameters(epochs=TrainingHyperparameterValidator.EPOCHS_MAX_REC + 1)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].is_blocking is False


class TestValidateForEngine:
    """Tests for validate_for_engine() method."""

    def test_valid_config_does_not_raise(self):
        params = _hyperparameters(
            batch_size=TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD,
            learning_rate=TrainingHyperparameterValidator.LR_INFO_LOW * 2,
            epochs=TrainingHyperparameterValidator.EPOCHS_MIN + 1,
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_MIN + 1,
        )
        # Should not raise
        TrainingHyperparameterValidator.validate_for_engine(params)

    def test_blocking_violation_raises_value_error(self):
        params = _hyperparameters(batch_size=0)
        with pytest.raises(ValueError) as exc_info:
            TrainingHyperparameterValidator.validate_for_engine(params)
        assert "Invalid Configuration" in str(exc_info.value)

    def test_non_blocking_only_does_not_raise(self):
        # epochs=15 is non-blocking
        params = _hyperparameters(epochs=TrainingHyperparameterValidator.EPOCHS_MAX_REC + 1)
        # Should not raise (non-blocking violations don't block)
        TrainingHyperparameterValidator.validate_for_engine(params)

    def test_info_only_does_not_raise(self):
        # batch_size=5 is non-blocking
        params = _hyperparameters(
            batch_size=TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD + 1
        )
        # Should not raise
        TrainingHyperparameterValidator.validate_for_engine(params)


class TestComprehensiveViolations:
    """Tests for comprehensive_violations() edge cases."""

    def test_valid_config_returns_empty_list(self):
        params = _hyperparameters(
            batch_size=TrainingHyperparameterValidator.BATCH_SIZE_INFO_THRESHOLD,
            learning_rate=TrainingHyperparameterValidator.LR_INFO_LOW * 2,
            epochs=TrainingHyperparameterValidator.EPOCHS_MIN + 1,
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_MIN + 1,
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        assert violations == []

    def test_multiple_violations_all_returned(self):
        params = _hyperparameters(
            batch_size=TrainingHyperparameterValidator.BATCH_SIZE_RANGE.start - 1,  # blocking
            learning_rate=TrainingHyperparameterValidator.LR_MAX * 2,  # blocking
            epochs=TrainingHyperparameterValidator.EPOCHS_MIN - 1,  # blocking
            sequence_length=TrainingHyperparameterValidator.SEQUENCE_MIN - 1,  # blocking
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        blocking_count = sum(1 for v in violations if v.is_blocking)
        assert blocking_count == 4

    def test_thresholds_match_class_constants(self):
        # Verify thresholds are derived from numerical principles, not heuristics
        import math
        import numpy as np

        eps = float(np.finfo(np.float32).eps)
        sqrt_eps = math.sqrt(eps)

        assert TrainingHyperparameterValidator.BATCH_SIZE_RANGE == range(1, 9)
        assert TrainingHyperparameterValidator.SEQUENCE_MIN == 128
        assert TrainingHyperparameterValidator.SEQUENCE_MAX == 4096

        # LR bounds are DERIVED from machine epsilon, not hardcoded
        # LR_MIN = eps (can't represent smaller changes)
        # LR_MAX = 1/sqrt(eps) (stability bound)
        assert TrainingHyperparameterValidator.LR_MIN == eps
        assert abs(TrainingHyperparameterValidator.LR_MAX - 1.0 / sqrt_eps) < 1e-6

        assert TrainingHyperparameterValidator.EPOCHS_MIN == 1
        assert TrainingHyperparameterValidator.GRAD_ACCUM_MAX == 16
