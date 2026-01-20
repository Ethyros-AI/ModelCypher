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
    Validates training hyperparameters for algebraic correctness only.
    """

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
                )
            )

        # Sequence Length
        if params.sequence_length < 2:
            violations.append(
                Violation(
                    "sequence_length",
                    "Sequence length must be >= 2",
                    is_blocking=True,
                )
            )

        # Learning Rate
        if params.learning_rate <= 0:
            violations.append(
                Violation(
                    "learning_rate",
                    "Learning rate must be > 0",
                    is_blocking=True,
                )
            )

        # Epochs
        if params.epochs < 1:
            violations.append(
                Violation("epochs", "Epochs must be positive", is_blocking=True)
            )

        # Gradient Accumulation
        if params.gradient_accumulation_steps <= 0:
            violations.append(
                Violation(
                    "gradient_accumulation_steps",
                    "Gradient accumulation steps must be > 0",
                    is_blocking=True,
                )
            )

        # Warmup Steps
        if params.warmup_steps < 0:
            violations.append(
                Violation(
                    "warmup_steps",
                    "Warmup steps must be >= 0",
                    is_blocking=True,
                )
            )

        # Weight Decay
        if params.weight_decay < 0:
            violations.append(
                Violation(
                    "weight_decay",
                    "Weight decay must be >= 0",
                    is_blocking=True,
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
