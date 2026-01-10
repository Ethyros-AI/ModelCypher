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

from dataclasses import dataclass

from .types import Hyperparameters


@dataclass
class Violation:
    """Raw constraint violation measurement.

    NO VIBES: Returns raw data, caller interprets significance.
    - is_blocking: True if training will fail (computationally invalid)
    - is_blocking: False if training will work but may be suboptimal
    """

    field: str
    message: str
    is_blocking: bool  # True = training fails, False = training works but suboptimal
    suggestion: str | None = None


class TrainingHyperparameterValidator:
    """
    Validates training hyperparameters against reference settings for Apple Silicon (MLX).
    Ported from Swift TrainingHyperparameterValidator.
    """

    # Thresholds ported from Swift ParameterThresholds.swift
    BATCH_SIZE_RANGE = range(1, 9)  # 1-8 local envelope
    BATCH_SIZE_INFO_THRESHOLD = 4

    SEQUENCE_MIN = 128
    SEQUENCE_MAX = 4096  # MLX limitation/memory constraint
    SEQUENCE_WARNING = 2048

    LR_MIN = 1e-6
    LR_MAX = 1e-3
    LR_WARN_HIGH = 5e-4
    LR_INFO_LOW = 1e-5

    EPOCHS_MIN = 1
    EPOCHS_MAX_REC = 10

    GRAD_ACCUM_MAX = 16
    GRAD_ACCUM_WARN = 8

    @classmethod
    def comprehensive_violations(cls, params: Hyperparameters) -> list[Violation]:
        violations = []

        # Batch Size
        if params.batch_size <= 0:
            violations.append(
                Violation(
                    "batch_size",
                    "Batch size must be > 0",
                    is_blocking=True,
                    suggestion="Start with 2 or 4",
                )
            )
        elif params.batch_size not in cls.BATCH_SIZE_RANGE:
            violations.append(
                Violation(
                    "batch_size",
                    "Batch size must be between 1 and 8 for Apple Silicon",
                    is_blocking=True,
                    suggestion="Start with 2 or 4",
                )
            )
        elif params.batch_size > cls.BATCH_SIZE_INFO_THRESHOLD:
            violations.append(
                Violation(
                    "batch_size",
                    "Batch sizes > 4 require significant memory",
                    is_blocking=False,
                    suggestion="Monitor memory usage",
                )
            )

        # Sequence Length
        if params.sequence_length < cls.SEQUENCE_MIN:
            violations.append(
                Violation(
                    "sequence_length",
                    f"Sequence length must be at least {cls.SEQUENCE_MIN}",
                    is_blocking=True,
                )
            )
        elif params.sequence_length > cls.SEQUENCE_MAX:
            violations.append(
                Violation(
                    "sequence_length",
                    f"Sequence length > {cls.SEQUENCE_MAX} exceeds tested limits",
                    is_blocking=True,
                    suggestion="Reduce to 4096 or lower",
                )
            )
        elif params.sequence_length > cls.SEQUENCE_WARNING:
            violations.append(
                Violation(
                    "sequence_length",
                    f"Sequences > {cls.SEQUENCE_WARNING} increase memory usage significantly",
                    is_blocking=False,
                )
            )

        # Learning Rate
        if not (cls.LR_MIN <= params.learning_rate <= cls.LR_MAX):
            violations.append(
                Violation(
                    "learning_rate",
                    "Learning rate must be between 1e-6 and 1e-3",
                    is_blocking=True,
                    suggestion="Start with 3e-5 for LoRA",
                )
            )
        elif params.learning_rate > cls.LR_WARN_HIGH:
            violations.append(
                Violation(
                    "learning_rate",
                    "Learning rate above 5e-4 can destabilize training",
                    is_blocking=False,
                )
            )
        elif params.learning_rate < cls.LR_INFO_LOW:
            violations.append(
                Violation("learning_rate", "Learning rate is very low", is_blocking=False)
            )

        # Epochs
        if params.epochs < cls.EPOCHS_MIN:
            violations.append(
                Violation("epochs", "Epochs must be positive", is_blocking=True)
            )
        elif params.epochs > cls.EPOCHS_MAX_REC:
            violations.append(
                Violation(
                    "epochs",
                    "Epochs > 10 often overfit small datasets",
                    is_blocking=False,
                    suggestion="Keep within 3-8 epochs",
                )
            )

        return violations

    @classmethod
    def validate_for_engine(cls, params: Hyperparameters) -> None:
        """Throws ValueError on first blocking violation."""
        violations = cls.comprehensive_violations(params)
        blocking = [v for v in violations if v.is_blocking]
        if blocking:
            raise ValueError(f"Invalid Configuration: {blocking[0].message}")
