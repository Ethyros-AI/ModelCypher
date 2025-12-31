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

from modelcypher.core.domain.training.types import Hyperparameters
from modelcypher.core.domain.training.validation import (
    TrainingHyperparameterValidator,
    ValidationSeverity,
    Violation,
)


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_error_value(self):
        assert ValidationSeverity.ERROR.value == "error"

    def test_warning_value(self):
        assert ValidationSeverity.WARNING.value == "warning"

    def test_info_value(self):
        assert ValidationSeverity.INFO.value == "info"

    def test_all_severities_unique(self):
        values = [s.value for s in ValidationSeverity]
        assert len(values) == len(set(values))


class TestViolation:
    """Tests for Violation dataclass."""

    def test_violation_required_fields(self):
        v = Violation(
            field="batch_size",
            message="Invalid batch size",
            severity=ValidationSeverity.ERROR,
        )
        assert v.field == "batch_size"
        assert v.message == "Invalid batch size"
        assert v.severity == ValidationSeverity.ERROR
        assert v.suggestion is None

    def test_violation_with_suggestion(self):
        v = Violation(
            field="learning_rate",
            message="Too high",
            severity=ValidationSeverity.WARNING,
            suggestion="Try 3e-5",
        )
        assert v.suggestion == "Try 3e-5"


class TestBatchSizeValidation:
    """Tests for batch size validation rules."""

    def test_batch_size_zero_is_error(self):
        params = Hyperparameters(batch_size=0)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].severity == ValidationSeverity.ERROR

    def test_batch_size_negative_is_error(self):
        params = Hyperparameters(batch_size=-1)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].severity == ValidationSeverity.ERROR

    def test_batch_size_1_to_4_no_violation(self):
        for bs in [1, 2, 3, 4]:
            params = Hyperparameters(batch_size=bs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            batch_violations = [v for v in violations if v.field == "batch_size"]
            # 1-4 should have no violations
            assert len(batch_violations) == 0, f"batch_size={bs} should have no violations"

    def test_batch_size_5_to_8_info(self):
        for bs in [5, 6, 7, 8]:
            params = Hyperparameters(batch_size=bs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            batch_violations = [v for v in violations if v.field == "batch_size"]
            assert len(batch_violations) == 1, f"batch_size={bs} should have INFO"
            assert batch_violations[0].severity == ValidationSeverity.INFO

    def test_batch_size_9_is_error(self):
        params = Hyperparameters(batch_size=9)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        batch_violations = [v for v in violations if v.field == "batch_size"]
        assert len(batch_violations) == 1
        assert batch_violations[0].severity == ValidationSeverity.ERROR


class TestSequenceLengthValidation:
    """Tests for sequence length validation rules."""

    def test_sequence_below_min_is_error(self):
        params = Hyperparameters(sequence_length=64)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].severity == ValidationSeverity.ERROR

    def test_sequence_at_min_no_violation(self):
        params = Hyperparameters(sequence_length=128)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 0

    def test_sequence_normal_range_no_violation(self):
        for seq in [256, 512, 1024, 2048]:
            params = Hyperparameters(sequence_length=seq)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            seq_violations = [v for v in violations if v.field == "sequence_length"]
            assert len(seq_violations) == 0, f"sequence_length={seq} should have no violations"

    def test_sequence_above_warning_threshold(self):
        params = Hyperparameters(sequence_length=3000)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].severity == ValidationSeverity.WARNING

    def test_sequence_at_max_is_warning(self):
        params = Hyperparameters(sequence_length=4096)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        # 4096 is at max but > warning threshold, so WARNING
        assert len(seq_violations) == 1
        assert seq_violations[0].severity == ValidationSeverity.WARNING

    def test_sequence_above_max_is_error(self):
        params = Hyperparameters(sequence_length=5000)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        seq_violations = [v for v in violations if v.field == "sequence_length"]
        assert len(seq_violations) == 1
        assert seq_violations[0].severity == ValidationSeverity.ERROR


class TestLearningRateValidation:
    """Tests for learning rate validation rules."""

    def test_lr_below_min_is_error(self):
        params = Hyperparameters(learning_rate=1e-7)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].severity == ValidationSeverity.ERROR

    def test_lr_at_min_is_info(self):
        params = Hyperparameters(learning_rate=1e-6)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        # 1e-6 is valid but < INFO threshold (1e-5)
        assert len(lr_violations) == 1
        assert lr_violations[0].severity == ValidationSeverity.INFO

    def test_lr_normal_range_no_violation(self):
        for lr in [3e-5, 5e-5, 1e-4, 3e-4]:
            params = Hyperparameters(learning_rate=lr)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            lr_violations = [v for v in violations if v.field == "learning_rate"]
            assert len(lr_violations) == 0, f"learning_rate={lr} should have no violations"

    def test_lr_above_warning_threshold(self):
        params = Hyperparameters(learning_rate=7e-4)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].severity == ValidationSeverity.WARNING

    def test_lr_at_max_is_warning(self):
        params = Hyperparameters(learning_rate=1e-3)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        # 1e-3 is at max but > warning threshold, so WARNING
        assert len(lr_violations) == 1
        assert lr_violations[0].severity == ValidationSeverity.WARNING

    def test_lr_above_max_is_error(self):
        params = Hyperparameters(learning_rate=2e-3)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        lr_violations = [v for v in violations if v.field == "learning_rate"]
        assert len(lr_violations) == 1
        assert lr_violations[0].severity == ValidationSeverity.ERROR


class TestEpochsValidation:
    """Tests for epochs validation rules."""

    def test_epochs_zero_is_error(self):
        params = Hyperparameters(epochs=0)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].severity == ValidationSeverity.ERROR

    def test_epochs_negative_is_error(self):
        params = Hyperparameters(epochs=-1)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].severity == ValidationSeverity.ERROR

    def test_epochs_1_to_10_no_violation(self):
        for epochs in [1, 3, 5, 8, 10]:
            params = Hyperparameters(epochs=epochs)
            violations = TrainingHyperparameterValidator.comprehensive_violations(params)
            epoch_violations = [v for v in violations if v.field == "epochs"]
            assert len(epoch_violations) == 0, f"epochs={epochs} should have no violations"

    def test_epochs_above_max_is_warning(self):
        params = Hyperparameters(epochs=15)
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        epoch_violations = [v for v in violations if v.field == "epochs"]
        assert len(epoch_violations) == 1
        assert epoch_violations[0].severity == ValidationSeverity.WARNING


class TestValidateForEngine:
    """Tests for validate_for_engine() method."""

    def test_valid_config_does_not_raise(self):
        params = Hyperparameters(
            batch_size=4,
            learning_rate=3e-5,
            epochs=3,
            sequence_length=512,
        )
        # Should not raise
        TrainingHyperparameterValidator.validate_for_engine(params)

    def test_error_violation_raises_value_error(self):
        params = Hyperparameters(batch_size=0)
        with pytest.raises(ValueError) as exc_info:
            TrainingHyperparameterValidator.validate_for_engine(params)
        assert "Invalid Configuration" in str(exc_info.value)

    def test_warning_only_does_not_raise(self):
        # epochs=15 gives WARNING, not ERROR
        params = Hyperparameters(epochs=15)
        # Should not raise (warnings don't block)
        TrainingHyperparameterValidator.validate_for_engine(params)

    def test_info_only_does_not_raise(self):
        # batch_size=5 gives INFO, not ERROR
        params = Hyperparameters(batch_size=5)
        # Should not raise
        TrainingHyperparameterValidator.validate_for_engine(params)


class TestComprehensiveViolations:
    """Tests for comprehensive_violations() edge cases."""

    def test_valid_config_returns_empty_list(self):
        params = Hyperparameters(
            batch_size=2,
            learning_rate=3e-5,
            epochs=3,
            sequence_length=512,
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        assert violations == []

    def test_multiple_violations_all_returned(self):
        params = Hyperparameters(
            batch_size=0,  # ERROR
            learning_rate=2e-3,  # ERROR
            epochs=0,  # ERROR
            sequence_length=50,  # ERROR
        )
        violations = TrainingHyperparameterValidator.comprehensive_violations(params)
        error_count = sum(1 for v in violations if v.severity == ValidationSeverity.ERROR)
        assert error_count >= 4

    def test_thresholds_match_class_constants(self):
        # Verify thresholds are as documented
        assert TrainingHyperparameterValidator.BATCH_SIZE_RANGE == range(1, 9)
        assert TrainingHyperparameterValidator.SEQUENCE_MIN == 128
        assert TrainingHyperparameterValidator.SEQUENCE_MAX == 4096
        assert TrainingHyperparameterValidator.LR_MIN == 1e-6
        assert TrainingHyperparameterValidator.LR_MAX == 1e-3
        assert TrainingHyperparameterValidator.EPOCHS_MIN == 1
        assert TrainingHyperparameterValidator.GRAD_ACCUM_MAX == 16
