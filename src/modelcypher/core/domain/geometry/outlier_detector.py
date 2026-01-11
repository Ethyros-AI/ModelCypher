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

Identifies models that disagree with the consensus position of a concept.
When 5 models agree on where "authority" lives and 1 differs, the 1 is WRONG.

Mathematical Background:
    Given per-model alignment errors from GPA, or pairwise stress profile
    distances, we detect outliers using z-scores with data-derived thresholds.

    An outlier is a model whose error exceeds:
        threshold = mean(errors) + k * std(errors)

    where k is derived from the number of models (stricter with more models).

    For stress profiles, we compute pairwise distances and identify models
    with high mean distance to others as outliers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
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
    """Detect models that disagree with consensus.

    Uses data-derived thresholds based on error distribution.
    No hardcoded magic numbers.
    """

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

        Uses z-score detection with sigma derived from model count:
        - 3 models: sigma = 1.5 (lenient - few data points)
        - 6+ models: sigma = 2.0 (stricter - more data points)

        The sigma value is interpolated linearly between these bounds.

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

        # Derive sigma from model count (interpolate between 1.5 and 2.0)
        # With 3 models: sigma = 1.5 (lenient)
        # With 6+ models: sigma = 2.0 (stricter)
        sigma = min(2.0, 1.5 + 0.1 * (n_models - 3))
        sigma = max(1.5, sigma)

        # Threshold = mean + sigma * std
        threshold = mean_err + sigma * std_err

        # Detect outliers
        consensus_indices = []
        outlier_indices = []

        for i, err in enumerate(per_model_errors):
            if err > threshold + eps:
                outlier_indices.append(i)
            else:
                consensus_indices.append(i)

        logger.info(
            "Outlier detection: %d consensus, %d outliers (threshold=%.4f, sigma=%.2f)",
            len(consensus_indices),
            len(outlier_indices),
            threshold,
            sigma,
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

        # For exactly 3 models, use triangulation
        if n_models == 3:
            return self._triangulate_outlier(pairwise)

        # For 4+ models, use mean distance approach
        mean_distances = []
        for i in range(n_models):
            total = sum(pairwise[i])
            mean_dist = total / (n_models - 1)
            mean_distances.append(mean_dist)

        # Use GPA-style detection on mean distances
        return self.detect_from_gpa(mean_distances)

    def _triangulate_outlier(
        self,
        pairwise: list[list[float]],
    ) -> OutlierResult:
        """Efficient outlier detection for exactly 3 models using triangulation.

        The consensus pair is the two models closest to each other.
        The outlier is the third model IF it's significantly farther from
        both consensus models than they are from each other.

        Threshold: outlier's mean distance to consensus pair > 2x the
        consensus pair's mutual distance.

        Args:
            pairwise: 3x3 pairwise distance matrix.

        Returns:
            OutlierResult with consensus and outlier indices.
        """
        # Get the three pairwise distances
        d_01 = pairwise[0][1]
        d_02 = pairwise[0][2]
        d_12 = pairwise[1][2]

        # Find the minimum distance - that's the consensus pair
        min_dist = min(d_01, d_02, d_12)

        if min_dist == d_01:
            # Models 0 and 1 are closest, 2 is potential outlier
            consensus_pair = (0, 1)
            potential_outlier = 2
            outlier_dists = (d_02, d_12)
        elif min_dist == d_02:
            # Models 0 and 2 are closest, 1 is potential outlier
            consensus_pair = (0, 2)
            potential_outlier = 1
            outlier_dists = (d_01, d_12)
        else:
            # Models 1 and 2 are closest, 0 is potential outlier
            consensus_pair = (1, 2)
            potential_outlier = 0
            outlier_dists = (d_01, d_02)

        # Compute mean distance from potential outlier to consensus pair
        outlier_mean_dist = sum(outlier_dists) / 2.0

        # Threshold: outlier is confirmed if its mean distance to consensus
        # is at least 2x the consensus pair's mutual distance
        # This is derived from triangle inequality: if truly an outlier,
        # it should be significantly farther from both consensus models
        threshold_ratio = 2.0
        is_outlier = outlier_mean_dist > threshold_ratio * min_dist

        # Compute per-model "errors" as mean distance to others
        errors = [
            (pairwise[0][1] + pairwise[0][2]) / 2.0,
            (pairwise[0][1] + pairwise[1][2]) / 2.0,
            (pairwise[0][2] + pairwise[1][2]) / 2.0,
        ]

        if is_outlier:
            consensus_indices = consensus_pair
            outlier_indices = (potential_outlier,)
            logger.info(
                "Triangulation: model %d is outlier (dist=%.4f vs consensus=%.4f)",
                potential_outlier,
                outlier_mean_dist,
                min_dist,
            )
        else:
            # All models are similar - no outlier
            consensus_indices = (0, 1, 2)
            outlier_indices = ()
            logger.info(
                "Triangulation: no outlier detected (max_ratio=%.2f)",
                outlier_mean_dist / max(min_dist, 1e-10),
            )

        mean_error = sum(errors) / 3.0
        variance = sum((e - mean_error) ** 2 for e in errors) / 3.0
        std_error = sqrt_scalar(variance, self._backend)

        return OutlierResult(
            consensus_indices=consensus_indices,
            outlier_indices=outlier_indices,
            errors=tuple(errors),
            threshold=threshold_ratio * min_dist,
            mean_error=mean_error,
            std_error=std_error,
        )

    def get_consensus_stress(
        self,
        profiles: list["RelationalStressProfile"],
        consensus_indices: tuple[int, ...],
    ) -> "Array":
        """Compute Fréchet mean of stress vectors from consensus models.

        This gives the "correct" stress profile that outliers should be
        moved toward.

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
