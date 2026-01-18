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

"""TransplantManifest - Tracks every weight's transformation status during merge.

This module provides structured tracking of what happened to each weight during
the transplant stage. Instead of relying on log messages to debug failures,
the manifest provides:

1. A complete record of every weight considered
2. The transformation status (success/failure/skip)
3. Diagnostic information (shapes, CKA values, error messages)
4. Aggregate statistics for validation

Usage:
    manifest = TransplantManifest()

    # Record each weight as processed
    manifest.record(key, WeightTransformRecord(
        key=key,
        status=WeightStatus.TRANSFORMED,
        source_shape=(960, 960),
        target_shape=(896, 896),
        stitch_type="hidden",
        preserved_fraction=0.73,
        cka_achieved=0.9999,
    ))

    # Validate at end of stage
    manifest.validate(target_weight_count=len(target_weights), strict=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modelcypher.core.domain.merging.exceptions import (
    PostconditionError,
    WeightCountMismatchError,
)


class WeightStatus(str, Enum):
    """Status of a weight after transplant processing."""

    # Success states
    TRANSFORMED = "transformed"  # Successfully transformed and transplanted
    IDENTITY = "identity"  # Same dimensions, no transformation needed

    # Expected skip states (not failures)
    SKIPPED_QUANTIZED = "skipped_quantized"  # Skipped due to quantization scales/zeros
    SKIPPED_NON_WEIGHT = "skipped_non_weight"  # Not a .weight tensor
    SKIPPED_NON_2D = "skipped_non_2d"  # Not a 2D weight matrix
    SKIPPED_EMBEDDING = "skipped_embedding"  # Embedding layer (handled separately)
    SKIPPED_UNMAPPED = "skipped_unmapped"  # No corresponding source weight
    SKIPPED_MISSING_ACTIVATIONS = "skipped_missing_activations"  # Missing activations
    SKIPPED_ACTIVATION_DIM_MISMATCH = "skipped_activation_dim_mismatch"  # Activation dim mismatch
    SKIPPED_DENSITY_FILTER = "skipped_density_filter"  # Density mask rejected layer
    SKIPPED_BOUNDARY = "skipped_boundary"  # Boundary-preserved layer (layer 0)

    # Failure states (numerical or missing prerequisites, not model incompatibility)
    FAILED_STITCH = "failed_stitch"  # Required stitch not available
    FAILED_ALIGNMENT = "failed_alignment"  # Alignment computation failed
    FAILED_DIMENSION = "failed_dimension"  # Dimension mismatch, no transform possible
    FAILED_NULL_SPACE = "failed_null_space"  # Null-space filtering failed
    FAILED_NUMERICAL = "failed_numerical"  # Numerical instability (NaN, Inf)
    FAILED_UNKNOWN = "failed_unknown"  # Unexpected error

    def is_success(self) -> bool:
        """Return True if this status represents successful processing."""
        return self in (WeightStatus.TRANSFORMED, WeightStatus.IDENTITY)

    def is_expected_skip(self) -> bool:
        """Return True if this status is an expected skip (not a failure)."""
        return self in (
            WeightStatus.SKIPPED_QUANTIZED,
            WeightStatus.SKIPPED_NON_WEIGHT,
            WeightStatus.SKIPPED_NON_2D,
            WeightStatus.SKIPPED_EMBEDDING,
            WeightStatus.SKIPPED_UNMAPPED,
            WeightStatus.SKIPPED_MISSING_ACTIVATIONS,
            WeightStatus.SKIPPED_ACTIVATION_DIM_MISMATCH,
            WeightStatus.SKIPPED_DENSITY_FILTER,
            WeightStatus.SKIPPED_BOUNDARY,
        )

    def is_failure(self) -> bool:
        """Return True if this status represents a failure."""
        return self.value.startswith("failed_")


@dataclass
class WeightTransformRecord:
    """Record of a single weight's transformation during transplant."""

    key: str
    status: WeightStatus
    source_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...] | None = None
    stitch_type: str | None = None  # "hidden", "intermediate", "attention", "dual"
    preserved_fraction: float | None = None  # Fraction preserved through null-space filter
    cka_achieved: float | None = None  # CKA achieved by alignment
    error_message: str | None = None  # Human-readable error if failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "key": self.key,
            "status": self.status.value,
            "source_shape": list(self.source_shape) if self.source_shape else None,
            "target_shape": list(self.target_shape) if self.target_shape else None,
            "stitch_type": self.stitch_type,
            "preserved_fraction": self.preserved_fraction,
            "cka_achieved": self.cka_achieved,
            "error_message": self.error_message,
        }


@dataclass
class TransplantManifest:
    """Tracks every weight's transformation status during the transplant stage.

    This manifest provides:
    - A complete record of every weight considered
    - Aggregate statistics for quick validation
    - Failure tracking with detailed diagnostics
    - Validation methods to ensure merge integrity
    """

    # All weight records, keyed by weight name
    weights: dict[str, WeightTransformRecord] = field(default_factory=dict)

    # Aggregate counters
    total_weights_considered: int = 0
    weights_transformed: int = 0  # TRANSFORMED or IDENTITY
    weights_skipped_expected: int = 0  # Expected skips (quantized, non-weight, etc.)
    weights_failed: int = 0  # Any failure status

    # Lists for quick access
    failed_weights: list[str] = field(default_factory=list)
    transformed_weights: list[str] = field(default_factory=list)

    # Quality metrics (computed from records)
    _total_preserved_fraction: float = 0.0
    _preserved_fraction_count: int = 0

    def record(self, key: str, record: WeightTransformRecord) -> None:
        """Record the transformation result for a weight.

        Args:
            key: The weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
            record: The transformation record
        """
        self.weights[key] = record
        self.total_weights_considered += 1

        if record.status.is_success():
            self.weights_transformed += 1
            self.transformed_weights.append(key)
        elif record.status.is_expected_skip():
            self.weights_skipped_expected += 1
        else:  # Failure
            self.weights_failed += 1
            self.failed_weights.append(key)

        # Track preserved fraction for averaging
        if record.preserved_fraction is not None:
            self._total_preserved_fraction += record.preserved_fraction
            self._preserved_fraction_count += 1

    def get_mean_preserved_fraction(self) -> float | None:
        """Get mean preserved fraction across all weights that had filtering applied."""
        if self._preserved_fraction_count == 0:
            return None
        return self._total_preserved_fraction / self._preserved_fraction_count

    def get_failure_summary(self) -> dict[str, list[str]]:
        """Get failures grouped by status type."""
        summary: dict[str, list[str]] = {}
        for key in self.failed_weights:
            record = self.weights[key]
            status_name = record.status.value
            if status_name not in summary:
                summary[status_name] = []
            summary[status_name].append(key)
        return summary

    def validate(
        self,
        target_weight_count: int,
        strict: bool = True,
        min_preserved_fraction: float = 0.1,
    ) -> None:
        """Validate the manifest against expectations.

        Args:
            target_weight_count: Number of weights in target model
            strict: If True, raise exceptions on validation failure
            min_preserved_fraction: Minimum acceptable mean preserved fraction

        Raises:
            WeightCountMismatchError: If transformed + skipped != expected
            PostconditionError: If any weights failed or preserved fraction too low
        """
        # Check 1: No unexpected failures
        if self.weights_failed > 0 and strict:
            failure_summary = self.get_failure_summary()
            sample_failures = self.failed_weights[:5]
            sample_errors = [
                f"{k}: {self.weights[k].error_message}" for k in sample_failures
            ]

            raise PostconditionError(
                stage="TRANSPLANT_VALIDATION",
                weight_key=None,
                message=f"{self.weights_failed} weights failed transformation",
                context={
                    "failed_count": self.weights_failed,
                    "failure_summary": failure_summary,
                    "sample_errors": sample_errors,
                },
            )

        # Check 2: All weights accounted for
        expected_transformed = target_weight_count - self.weights_skipped_expected
        if self.weights_transformed != expected_transformed and strict:
            raise WeightCountMismatchError(
                stage="TRANSPLANT_VALIDATION",
                weight_key=None,
                message=f"Expected {expected_transformed} transformed weights, got {self.weights_transformed}",
                context={
                    "expected": expected_transformed,
                    "actual": self.weights_transformed,
                    "skipped": self.weights_skipped_expected,
                    "failed": self.weights_failed,
                    "total_considered": self.total_weights_considered,
                },
            )

        # Check 3: Minimum preserved fraction (if filtering was applied)
        mean_preserved = self.get_mean_preserved_fraction()
        if mean_preserved is not None and mean_preserved < min_preserved_fraction:
            if strict:
                raise PostconditionError(
                    stage="TRANSPLANT_VALIDATION",
                    weight_key=None,
                    message=f"Mean preserved fraction {mean_preserved:.3f} < {min_preserved_fraction}",
                    context={
                        "mean_preserved_fraction": mean_preserved,
                        "min_required": min_preserved_fraction,
                        "count": self._preserved_fraction_count,
                    },
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_considered": self.total_weights_considered,
                "transformed": self.weights_transformed,
                "skipped_expected": self.weights_skipped_expected,
                "failed": self.weights_failed,
                "mean_preserved_fraction": self.get_mean_preserved_fraction(),
            },
            "failed_weights": self.failed_weights,
            "failure_summary": self.get_failure_summary(),
            "weights": {k: v.to_dict() for k, v in self.weights.items()},
        }

    def __repr__(self) -> str:
        return (
            f"TransplantManifest("
            f"transformed={self.weights_transformed}, "
            f"skipped={self.weights_skipped_expected}, "
            f"failed={self.weights_failed})"
        )
