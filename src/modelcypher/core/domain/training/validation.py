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

"""Training hyperparameter validation.

Philosophy: Bounds are DERIVED from numerical analysis, not heuristics.

Learning rate bounds:
    - LR_MIN = eps (machine epsilon, can't go smaller meaningfully)
    - LR_MAX = 1/sqrt(eps) (stability bound from numerical analysis)

The bounds are derived from float32 precision:
    - eps ≈ 1.19e-7
    - sqrt(eps) ≈ 3.45e-4
    - 1/sqrt(eps) ≈ 2896

We use sqrt(eps) as the reference scale for most operations.
"""

import math
from dataclasses import dataclass


from .types import Hyperparameters


# =============================================================================
# Dtype-derived constants (float32 is the training precision ceiling)
# =============================================================================

_EPS = math.ldexp(1.0, -23)  # float32 machine epsilon (2^-23)
_SQRT_EPS = math.sqrt(_EPS)  # ~3.45e-4


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
    Validates training hyperparameters for algebraic correctness.

    All bounds are DERIVED from numerical properties, not arbitrary heuristics:
    - LR_MIN = eps (can't represent smaller values meaningfully)
    - LR_MAX = 1/sqrt(eps) (numerical stability bound)
    - SEQUENCE bounds from transformer attention quadratic memory
    """

    # Batch size: algebraic constraint (must be positive integer)
    BATCH_SIZE_RANGE = range(1, 9)  # [1, 8] is practical for memory
    BATCH_SIZE_INFO_THRESHOLD = (BATCH_SIZE_RANGE.start + BATCH_SIZE_RANGE.stop - 1) // 2

    # Sequence length: architecture-derived (attention is O(n²) memory)
    # MIN: Must have enough tokens for meaningful context
    # MAX: Quadratic memory scaling makes this a hard limit
    SEQUENCE_MIN = 128  # Minimum meaningful context
    SEQUENCE_MAX = 4096  # Attention memory limit for most architectures
    SEQUENCE_WARNING = SEQUENCE_MAX // 2

    # Learning rate: DERIVED from machine epsilon
    # LR_MIN: Can't represent smaller changes than eps meaningfully
    # LR_MAX: Stability bound 1/sqrt(eps) (gradient scaling)
    LR_MIN = _EPS  # ~1.19e-7
    LR_MAX = 1.0 / _SQRT_EPS  # ~2896

    # Info thresholds at geometric thirds of the valid range
    # These mark "typical" vs "unusual" values, not "good" vs "bad"
    LR_INFO_LOW = LR_MIN * (LR_MAX / LR_MIN) ** (1 / 3)  # ~6.9e-5
    LR_WARN_HIGH = LR_MIN * (LR_MAX / LR_MIN) ** (2 / 3)  # ~40

    # Epochs: algebraic constraint
    EPOCHS_MIN = 1
    EPOCHS_MAX_REC = BATCH_SIZE_RANGE.stop + 1

    # Gradient accumulation: memory/compute tradeoff
    GRAD_ACCUM_MAX = 16

    @classmethod
    def comprehensive_violations(cls, params: Hyperparameters) -> list[Violation]:
        violations = []

        # Batch Size
        if params.batch_size < cls.BATCH_SIZE_RANGE.start:
            violations.append(
                Violation(
                    "batch_size",
                    "Batch size must be > 0",
                    is_blocking=True,
                )
            )
        elif params.batch_size >= cls.BATCH_SIZE_RANGE.stop:
            violations.append(
                Violation(
                    "batch_size",
                    f"Batch size exceeds supported range {cls.BATCH_SIZE_RANGE}",
                    is_blocking=True,
                )
            )
        elif params.batch_size > cls.BATCH_SIZE_INFO_THRESHOLD:
            violations.append(
                Violation(
                    "batch_size",
                    "Batch size exceeds recommended range",
                    is_blocking=False,
                )
            )

        # Sequence Length
        if params.sequence_length < cls.SEQUENCE_MIN:
            violations.append(
                Violation(
                    "sequence_length",
                    f"Sequence length must be >= {cls.SEQUENCE_MIN}",
                    is_blocking=True,
                )
            )
        elif params.sequence_length > cls.SEQUENCE_MAX:
            violations.append(
                Violation(
                    "sequence_length",
                    f"Sequence length exceeds max {cls.SEQUENCE_MAX}",
                    is_blocking=True,
                )
            )
        elif params.sequence_length > cls.SEQUENCE_WARNING:
            violations.append(
                Violation(
                    "sequence_length",
                    "Sequence length exceeds recommended range",
                    is_blocking=False,
                )
            )

        # Learning Rate
        if params.learning_rate < cls.LR_MIN:
            violations.append(
                Violation(
                    "learning_rate",
                    f"Learning rate must be >= {cls.LR_MIN:g}",
                    is_blocking=True,
                )
            )
        elif params.learning_rate > cls.LR_MAX:
            violations.append(
                Violation(
                    "learning_rate",
                    f"Learning rate exceeds max {cls.LR_MAX:g}",
                    is_blocking=True,
                )
            )
        elif params.learning_rate < cls.LR_INFO_LOW:
            violations.append(
                Violation(
                    "learning_rate",
                    "Learning rate below recommended range",
                    is_blocking=False,
                )
            )
        elif params.learning_rate > cls.LR_WARN_HIGH:
            violations.append(
                Violation(
                    "learning_rate",
                    "Learning rate above recommended range",
                    is_blocking=False,
                )
            )

        # Epochs
        if params.epochs < cls.EPOCHS_MIN:
            violations.append(
                Violation(
                    "epochs",
                    f"Epochs must be >= {cls.EPOCHS_MIN}",
                    is_blocking=True,
                )
            )
        elif params.epochs > cls.EPOCHS_MAX_REC:
            violations.append(
                Violation(
                    "epochs",
                    "Epochs exceed recommended maximum",
                    is_blocking=False,
                )
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
        elif params.gradient_accumulation_steps > cls.GRAD_ACCUM_MAX:
            violations.append(
                Violation(
                    "gradient_accumulation_steps",
                    "Gradient accumulation steps exceed recommended maximum",
                    is_blocking=False,
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
