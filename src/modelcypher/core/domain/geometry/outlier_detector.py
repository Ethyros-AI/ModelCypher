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

"""Outlier detection for multi-model consensus.

Identifies models that deviate from consensus using data-derived thresholds
over alignment error distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.cross_grounding_transfer import (
        RelationalStressProfile,
    )
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierResult:
    """Result of outlier detection."""

    consensus_indices: tuple[int, ...]  # Indices of models in consensus
    outlier_indices: tuple[int, ...]  # Indices of outlier models
    errors: tuple[float, ...]  # Per-model error values
    threshold: float  # Threshold used for detection
    mean_error: float  # Mean error across all models
    std_error: float  # Standard deviation of errors


class OutlierDetector:
    """Detect models that disagree with consensus."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize detector.

        Args:
            backend: Compute backend (defaults to system default).
        """
        self._backend = backend or get_default_backend()

    def detect_from_gpa(
        self,
        per_model_errors: list[float],
    ) -> OutlierResult:
        """Detect outliers from GPA per-model alignment errors.

        Args:
            per_model_errors: Per-model alignment errors from GPA result.

        Returns:
            OutlierResult with consensus and outlier indices.
        """
        b = self._backend
        n_models = len(per_model_errors)

        if n_models < 2:
            logger.warning("Need at least 2 models for outlier detection")
            return OutlierResult(
                consensus_indices=tuple(range(n_models)),
                outlier_indices=(),
                errors=tuple(per_model_errors),
                threshold=0.0,
                mean_error=per_model_errors[0] if per_model_errors else 0.0,
                std_error=0.0,
            )

        errors_arr = b.array(per_model_errors)
        eps = division_epsilon(b, errors_arr)

        # Compute statistics
        mean_err = float(b.mean(errors_arr))
        variance = float(b.mean((errors_arr - mean_err) ** 2))
        std_err = sqrt_scalar(variance, b)

        sorted_errors = sorted(per_model_errors)
        median_err = sorted_errors[n_models // 2]
        tail = [value for value in sorted_errors if value >= median_err]
        threshold = find_magnitude_gap_threshold(tail, eps=eps, backend=b)
        threshold = max(threshold, median_err + eps)

        # Detect outliers
        consensus_indices = []
        outlier_indices = []

        for i, err in enumerate(per_model_errors):
            if err > threshold + eps:
                outlier_indices.append(i)
            else:
                consensus_indices.append(i)

        logger.info(
            "Outlier detection: %d consensus, %d outliers "
            "(threshold=%.4f, median=%.4f)",
            len(consensus_indices),
            len(outlier_indices),
            threshold,
            median_err,
        )

        return OutlierResult(
            consensus_indices=tuple(consensus_indices),
            outlier_indices=tuple(outlier_indices),
            errors=tuple(per_model_errors),
            threshold=threshold,
            mean_error=mean_err,
            std_error=std_err,
        )

    def detect_from_stress_profiles(
        self,
        profiles: list["RelationalStressProfile"],
    ) -> OutlierResult:
        """Detect outliers from stress profile pairwise distances.

        For 3 models, uses efficient triangulation:
        - Find the closest pair (consensus candidates)
        - Check if the third is significantly farther from both
        - If so, the third is the outlier

        For 4+ models, uses mean distance with z-score detection.

        Args:
            profiles: List of RelationalStressProfile from each model.

        Returns:
            OutlierResult with consensus and outlier indices.
        """
        b = self._backend
        n_models = len(profiles)

        if n_models < 2:
            logger.warning("Need at least 2 profiles for outlier detection")
            return OutlierResult(
                consensus_indices=tuple(range(n_models)),
                outlier_indices=(),
                errors=(0.0,) * n_models,
                threshold=0.0,
                mean_error=0.0,
                std_error=0.0,
            )

        # Compute pairwise distances
        pairwise = [[0.0] * n_models for _ in range(n_models)]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                dist = profiles[i].distance_to(profiles[j])
                pairwise[i][j] = dist
                pairwise[j][i] = dist

        # Use mean distance approach for all model counts
        mean_distances = []
        for i in range(n_models):
            total = sum(pairwise[i])
            mean_dist = total / (n_models - 1)
            mean_distances.append(mean_dist)

        # Use GPA-style detection on mean distances
        return self.detect_from_gpa(mean_distances)

    def get_consensus_stress(
        self,
        profiles: list["RelationalStressProfile"],
        consensus_indices: tuple[int, ...],
    ) -> "Array":
        """Compute Fréchet mean of stress vectors from consensus models.

        This gives the consensus stress profile used for correction.

        Args:
            profiles: All stress profiles.
            consensus_indices: Indices of models in consensus.

        Returns:
            Mean stress vector from consensus models.
        """
        b = self._backend

        if not consensus_indices:
            raise ValueError("No consensus models to compute mean from")

        # Stack stress vectors from consensus models
        consensus_stresses = [
            b.array(profiles[i].stress_vector) for i in consensus_indices
        ]
        stacked = b.stack(consensus_stresses, axis=0)

        # Fréchet mean = arithmetic mean for stress vectors (Euclidean space)
        mean_stress = b.mean(stacked, axis=0)

        return mean_stress
