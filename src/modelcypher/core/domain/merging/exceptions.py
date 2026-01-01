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

    pass


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
    """GramAligner failed to achieve CKA >= 0.9999.

    This indicates the alignment transformation could not find a perfect
    geometric match between source and target activation spaces.

    Context typically includes:
    - achieved_cka: The CKA value that was achieved
    - source_dim: Source activation dimension
    - target_dim: Target activation dimension
    - samples_used: Number of samples used for alignment
    """

    pass


class DimensionMismatchError(MergeValidationError):
    """Weight dimensions incompatible and no stitch transformation available.

    This occurs when source and target weights have different shapes and
    the pipeline cannot find or apply a valid transformation.

    Context typically includes:
    - source_shape: Shape of source weight
    - target_shape: Shape of target weight
    - stitch_type: Which stitch was needed (hidden/intermediate/attention)
    """

    pass


class StitchUnavailableError(MergeValidationError):
    """Required stitch transformation (hidden/intermediate/attention) not available.

    This occurs when a cross-architecture merge requires a dimension transformation
    but the global alignment for that transformation type failed or was not computed.

    Context typically includes:
    - stitch_type: The type of stitch needed
    - reason: Why the stitch is unavailable
    """

    pass


class NullSpaceFilterError(MergeValidationError):
    """Null-space filtering could not be applied.

    This occurs when the activation dimension doesn't match the weight dimension
    and cross-dimensional null-space computation is not possible.

    Context typically includes:
    - activation_dim: Dimension of activations
    - weight_dim: Dimension of weight
    """

    pass


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

    pass


class PreconditionError(MergeValidationError):
    """A required precondition for the merge was not met.

    This is raised at the start of a pipeline stage when required
    inputs are missing or invalid.

    Context typically includes:
    - requirement: What was required
    - actual: What was actually provided
    """

    pass


class PostconditionError(MergeValidationError):
    """A postcondition guarantee was violated after processing.

    This is raised at the end of a pipeline stage when the output
    doesn't meet expected guarantees.

    Context typically includes:
    - guarantee: What should have been true
    - actual: What was actually observed
    """

    pass
