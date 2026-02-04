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

from __future__ import annotations

from typing import Any


class MergeError(Exception):
    """Exception raised for errors during the model merging process."""


class MergeValidationError(MergeError):
    """Base exception for merge validation failures.

    Provides structured error information including:
    - stage: Which pipeline stage failed (e.g., "GLOBAL_TRAJECTORY_ALIGNMENT")
    - weight_key: The specific weight that failed (if applicable)
    - context: Dictionary with diagnostic details (dimensions, CKA values, etc.)
    """

    def __init__(
        self,
        stage: str,
        weight_key: str | None,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.weight_key = weight_key
        self.context = context or {}

        # Build detailed error message
        full_message = f"[{stage}] {message}"
        if weight_key:
            full_message = f"[{stage}] Weight '{weight_key}': {message}"

        super().__init__(full_message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"stage={self.stage!r}, "
            f"weight_key={self.weight_key!r}, "
            f"context={self.context!r})"
        )


class AlignmentFailureError(MergeValidationError):
    """Alignment diagnostics reported a failure.

    Use for numerical failures (NaN/Inf) or missing prerequisites, not low CKA.
    CKA is recorded as a diagnostic metric.

    Context typically includes:
    - achieved_cka: The CKA value that was achieved
    - source_dim: Source activation dimension
    - target_dim: Target activation dimension
    - samples_used: Number of samples used for alignment
    - fix: Suggested debugging steps
    """



class AlignmentPrecisionError(MergeError):
    """Strict mode: alignment precision check failed.

    This error is reserved for numerical failures, not low CKA.
    """



class DimensionMismatchError(MergeValidationError):
    """Weight dimensions differ and stitch transformation was not applied.

    If this is raised, the pipeline failed to compute or apply the stitch.

    Context typically includes:
    - source_shape: Shape of source weight
    - target_shape: Shape of target weight
    - stitch_type: Which stitch was needed (hidden/intermediate/attention)
    """



class StitchUnavailableError(MergeValidationError):
    """Required stitch transformation was not computed.

    This indicates missing prerequisites or a failure in the stitch stage.

    Context typically includes:
    - stitch_type: The type of stitch needed
    - reason: Why the stitch computation failed
    """



class NullSpaceFilterError(MergeValidationError):
    """Null-space filtering could not be applied.

    This occurs when the activation dimension doesn't match the weight dimension
    and cross-dimensional null-space computation is not possible.

    Context typically includes:
    - activation_dim: Dimension of activations
    - weight_dim: Dimension of weight
    """



class WeightCountMismatchError(MergeValidationError):
    """Merged model has different weight count than target.

    This indicates weights were unexpectedly added or lost during the merge,
    which would produce an invalid model.

    Context typically includes:
    - expected: Number of weights expected
    - actual: Number of weights in merged model
    - missing: List of missing weight keys (if any)
    - extra: List of unexpected weight keys (if any)
    """



class PreconditionError(MergeValidationError):
    """A required precondition for the merge was not met.

    This is raised at the start of a pipeline stage when required
    inputs are missing or invalid.

    Context typically includes:
    - requirement: What was required
    - actual: What was actually provided
    """



class PostconditionError(MergeValidationError):
    """A postcondition was not met after processing.

    This is raised at the end of a pipeline stage when the output
    doesn't meet expected postconditions.

    Context typically includes:
    - postcondition: What should have been true
    - actual: What was actually observed
    """



class CalibrationRequiredError(MergeValidationError):
    """Calibration data is required but not provided.

    Operations requiring baseline or calibration data cannot proceed
    without proper calibration. The caller must provide calibration
    data or run a calibration step first.

    Context typically includes:
    - required_data: What calibration data was needed
    - suggestion: How to obtain the calibration data
    """



class EntropyMeasurementError(MergeValidationError):
    """Entropy measurement failed.

    Real entropy measurement is required for merge validation.
    Simulated profiles are not acceptable - they mask genuine
    measurement failures.

    Context typically includes:
    - model_path: Path to the model
    - failure_reason: Why measurement failed
    - fix: Suggested debugging steps
    """
