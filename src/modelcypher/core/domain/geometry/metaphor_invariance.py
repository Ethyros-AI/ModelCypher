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

"""Cross-model metaphor invariance testing.

Tests whether metaphor geometry is invariant across different model architectures,
validating the Platonic Representation Hypothesis (arXiv 2405.07987).

The key insight: if models learn the same underlying conceptual structure,
then the geometric relationship between source and target domains in a
metaphor should be similar across models, even with different architectures.

Measurements:
- trajectory_cka: CKA between full layer-wise CKA trajectories
- convergence_layer_delta: Difference in convergence layers (normalized by depth)
- peak_cka_delta: Difference in maximum CKA achieved
- gram_cka_at_convergence: Gram-based CKA at each model's convergence layer
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import CMTFamily, CMTMapping
    from modelcypher.core.domain.geometry.metaphor_trajectory import MetaphorTrajectory
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaphorInvarianceResult:
    """Invariance measurement for a metaphor across two models.

    All values are raw geometric measurements, no interpretation.

    Attributes:
        metaphor_id: Unique identifier for the CMT mapping.
        metaphor_name: Human-readable name (e.g., "TIME IS MONEY").
        model_a: Identifier for the first model.
        model_b: Identifier for the second model.
        trajectory_cka: CKA between layer-wise CKA trajectories.
        convergence_layer_a: Convergence layer for model A.
        convergence_layer_b: Convergence layer for model B.
        convergence_layer_delta_normalized: Absolute difference in convergence
            layers, normalized by the maximum layer count.
        peak_cka_a: Peak CKA for model A.
        peak_cka_b: Peak CKA for model B.
        peak_cka_delta: Absolute difference in peak CKA values.
    """

    metaphor_id: str
    metaphor_name: str
    model_a: str
    model_b: str

    # Trajectory comparison
    trajectory_cka: float

    # Convergence layer comparison
    convergence_layer_a: int
    convergence_layer_b: int
    convergence_layer_delta_normalized: float

    # Peak CKA comparison
    peak_cka_a: float
    peak_cka_b: float
    peak_cka_delta: float


@dataclass(frozen=True)
class BatchInvarianceResult:
    """Aggregate invariance measurements across multiple metaphor-model pairs.

    Attributes:
        results: All individual invariance results.
        mean_trajectory_cka: Mean trajectory CKA across all pairs.
        std_trajectory_cka: Standard deviation of trajectory CKA.
        mean_convergence_delta: Mean normalized convergence layer delta.
        mean_peak_cka_delta: Mean peak CKA delta.
        per_metaphor_trajectory_cka: Mean trajectory CKA per metaphor type.
        per_family_trajectory_cka: Mean trajectory CKA per metaphor family.
    """

    results: list[MetaphorInvarianceResult]
    mean_trajectory_cka: float
    std_trajectory_cka: float
    mean_convergence_delta: float
    mean_peak_cka_delta: float
    per_metaphor_trajectory_cka: dict[str, float]
    per_family_trajectory_cka: dict[str, float]


class MetaphorInvarianceAnalyzer:
    """Test whether metaphor geometry is invariant across architectures.

    Uses Gram-based CKA to handle different hidden dimensions between models,
    comparing the shape of source→target convergence trajectories.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the invariance analyzer.

        Args:
            backend: Backend protocol implementation. If None, uses default.
        """
        self.backend = backend or get_default_backend()

    def compare_metaphor_geometry(
        self,
        trajectory_a: "MetaphorTrajectory",
        trajectory_b: "MetaphorTrajectory",
    ) -> MetaphorInvarianceResult:
        """Compare metaphor trajectories between two models.

        Uses the layer-wise CKA values as feature vectors and computes
        CKA between them to measure trajectory similarity.

        Args:
            trajectory_a: Metaphor trajectory from model A.
            trajectory_b: Metaphor trajectory from model B.

        Returns:
            MetaphorInvarianceResult with raw geometric measurements.

        Raises:
            ValueError: If trajectories are for different metaphors.
        """
        if trajectory_a.metaphor_id != trajectory_b.metaphor_id:
            raise ValueError(
                f"Cannot compare trajectories for different metaphors: "
                f"{trajectory_a.metaphor_id} vs {trajectory_b.metaphor_id}"
            )

        # Extract CKA trajectories as vectors
        cka_a = [p.cka_source_target for p in trajectory_a.points]
        cka_b = [p.cka_source_target for p in trajectory_b.points]

        # Compute trajectory CKA
        # Treat CKA trajectories as single-feature activation vectors
        trajectory_cka = self._compute_trajectory_cka(cka_a, cka_b)

        # Compute convergence layer delta (normalized by max layer count)
        max_layers = max(trajectory_a.layer_count, trajectory_b.layer_count)
        eps = 1e-10
        convergence_delta = abs(
            trajectory_a.convergence_layer - trajectory_b.convergence_layer
        )
        convergence_delta_normalized = convergence_delta / (max_layers + eps)

        # Compute peak CKA delta
        peak_cka_delta = abs(trajectory_a.peak_cka - trajectory_b.peak_cka)

        return MetaphorInvarianceResult(
            metaphor_id=trajectory_a.metaphor_id,
            metaphor_name=trajectory_a.metaphor_name,
            model_a=trajectory_a.model_id,
            model_b=trajectory_b.model_id,
            trajectory_cka=trajectory_cka,
            convergence_layer_a=trajectory_a.convergence_layer,
            convergence_layer_b=trajectory_b.convergence_layer,
            convergence_layer_delta_normalized=convergence_delta_normalized,
            peak_cka_a=trajectory_a.peak_cka,
            peak_cka_b=trajectory_b.peak_cka,
            peak_cka_delta=peak_cka_delta,
        )

    def _compute_trajectory_cka(
        self,
        trajectory_a: list[float],
        trajectory_b: list[float],
    ) -> float:
        """Compute CKA between two CKA trajectories.

        Handles different trajectory lengths by interpolating to a common
        length (the maximum of the two).

        Args:
            trajectory_a: CKA values at each layer for model A.
            trajectory_b: CKA values at each layer for model B.

        Returns:
            CKA similarity between the trajectories.
        """
        if not trajectory_a or not trajectory_b:
            return 0.0

        # Interpolate both trajectories to common length
        common_length = max(len(trajectory_a), len(trajectory_b))

        interp_a = self._interpolate_trajectory(trajectory_a, common_length)
        interp_b = self._interpolate_trajectory(trajectory_b, common_length)

        # Convert to arrays for CKA computation
        # Treat each trajectory as [n_points, 1] activation matrix
        arr_a = self.backend.array([[v] for v in interp_a])
        arr_b = self.backend.array([[v] for v in interp_b])

        result = compute_cka(arr_a, arr_b, backend=self.backend, use_linear_kernel=True)
        return result.cka

    def _interpolate_trajectory(
        self,
        trajectory: list[float],
        target_length: int,
    ) -> list[float]:
        """Interpolate trajectory to target length using linear interpolation.

        Args:
            trajectory: Original CKA trajectory.
            target_length: Desired length.

        Returns:
            Interpolated trajectory of target_length.
        """
        if len(trajectory) == target_length:
            return trajectory

        if len(trajectory) == 1:
            return [trajectory[0]] * target_length

        # Linear interpolation
        result = []
        for i in range(target_length):
            # Map target index to source index
            src_idx = i * (len(trajectory) - 1) / (target_length - 1)
            lower = int(src_idx)
            upper = min(lower + 1, len(trajectory) - 1)
            frac = src_idx - lower

            value = trajectory[lower] * (1 - frac) + trajectory[upper] * frac
            result.append(value)

        return result

    def batch_invariance_test(
        self,
        trajectories: dict[str, list["MetaphorTrajectory"]],
        metaphor_family_map: dict[str, str] | None = None,
    ) -> BatchInvarianceResult:
        """Test invariance across all model pairs for all metaphors.

        Args:
            trajectories: Dict mapping model_id to list of MetaphorTrajectories.
            metaphor_family_map: Optional dict mapping metaphor_id to family name.

        Returns:
            BatchInvarianceResult with aggregate statistics.
        """
        results: list[MetaphorInvarianceResult] = []
        per_metaphor_ckas: dict[str, list[float]] = {}
        per_family_ckas: dict[str, list[float]] = {}

        model_ids = list(trajectories.keys())

        # Compare all model pairs
        for i, model_a in enumerate(model_ids):
            for model_b in model_ids[i + 1 :]:
                # Get trajectories for each model
                trajs_a = {t.metaphor_id: t for t in trajectories[model_a]}
                trajs_b = {t.metaphor_id: t for t in trajectories[model_b]}

                # Compare common metaphors
                common_metaphors = set(trajs_a.keys()) & set(trajs_b.keys())
                for metaphor_id in common_metaphors:
                    result = self.compare_metaphor_geometry(
                        trajs_a[metaphor_id],
                        trajs_b[metaphor_id],
                    )
                    results.append(result)

                    # Track per-metaphor CKA
                    if metaphor_id not in per_metaphor_ckas:
                        per_metaphor_ckas[metaphor_id] = []
                    per_metaphor_ckas[metaphor_id].append(result.trajectory_cka)

                    # Track per-family CKA
                    if metaphor_family_map and metaphor_id in metaphor_family_map:
                        family = metaphor_family_map[metaphor_id]
                        if family not in per_family_ckas:
                            per_family_ckas[family] = []
                        per_family_ckas[family].append(result.trajectory_cka)

        # Compute aggregate statistics
        if results:
            all_ckas = [r.trajectory_cka for r in results]
            mean_cka = sum(all_ckas) / len(all_ckas)
            variance = sum((c - mean_cka) ** 2 for c in all_ckas) / len(all_ckas)
            std_cka = variance**0.5

            mean_conv_delta = sum(
                r.convergence_layer_delta_normalized for r in results
            ) / len(results)
            mean_peak_delta = sum(r.peak_cka_delta for r in results) / len(results)
        else:
            mean_cka = 0.0
            std_cka = 0.0
            mean_conv_delta = 0.0
            mean_peak_delta = 0.0

        # Compute per-metaphor means
        per_metaphor_means = {
            m: sum(ckas) / len(ckas) for m, ckas in per_metaphor_ckas.items() if ckas
        }

        # Compute per-family means
        per_family_means = {
            f: sum(ckas) / len(ckas) for f, ckas in per_family_ckas.items() if ckas
        }

        return BatchInvarianceResult(
            results=results,
            mean_trajectory_cka=mean_cka,
            std_trajectory_cka=std_cka,
            mean_convergence_delta=mean_conv_delta,
            mean_peak_cka_delta=mean_peak_delta,
            per_metaphor_trajectory_cka=per_metaphor_means,
            per_family_trajectory_cka=per_family_means,
        )


class PlatonicMetaphorValidator:
    """Validate Platonic Representation Hypothesis for metaphor geometry.

    Tests whether the same metaphors produce geometrically similar mappings
    across different model architectures, supporting the hypothesis that
    neural networks converge to similar representations.

    All outputs are raw measurements - no thresholds or interpretations.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the validator.

        Args:
            backend: Backend protocol implementation. If None, uses default.
        """
        self.backend = backend or get_default_backend()
        self.analyzer = MetaphorInvarianceAnalyzer(backend)

    def validate_cross_architecture(
        self,
        trajectories: dict[str, list["MetaphorTrajectory"]],
        metaphor_family_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Test whether same metaphors produce similar geometry across architectures.

        Args:
            trajectories: Dict mapping model_id to list of MetaphorTrajectories.
            metaphor_family_map: Optional dict mapping metaphor_id to family name.

        Returns:
            Dictionary with raw measurements:
                mean_trajectory_cka: Mean CKA across all model pairs
                std_trajectory_cka: Standard deviation
                per_metaphor_cka: CKA per metaphor type
                per_family_cka: CKA per metaphor family
                model_count: Number of models compared
                metaphor_count: Number of metaphors compared
                pair_count: Number of model pairs compared
        """
        result = self.analyzer.batch_invariance_test(trajectories, metaphor_family_map)

        return {
            "mean_trajectory_cka": result.mean_trajectory_cka,
            "std_trajectory_cka": result.std_trajectory_cka,
            "mean_convergence_delta": result.mean_convergence_delta,
            "mean_peak_cka_delta": result.mean_peak_cka_delta,
            "per_metaphor_cka": result.per_metaphor_trajectory_cka,
            "per_family_cka": result.per_family_trajectory_cka,
            "model_count": len(trajectories),
            "metaphor_count": len(result.per_metaphor_trajectory_cka),
            "pair_count": len(result.results),
        }


def invariance_result_to_dict(result: MetaphorInvarianceResult) -> dict[str, Any]:
    """Convert a MetaphorInvarianceResult to a dictionary for serialization.

    Args:
        result: The invariance result to convert.

    Returns:
        Dictionary representation of the result.
    """
    return {
        "metaphor_id": result.metaphor_id,
        "metaphor_name": result.metaphor_name,
        "model_a": result.model_a,
        "model_b": result.model_b,
        "trajectory_cka": result.trajectory_cka,
        "convergence_layer_a": result.convergence_layer_a,
        "convergence_layer_b": result.convergence_layer_b,
        "convergence_layer_delta_normalized": result.convergence_layer_delta_normalized,
        "peak_cka_a": result.peak_cka_a,
        "peak_cka_b": result.peak_cka_b,
        "peak_cka_delta": result.peak_cka_delta,
    }


def batch_result_to_dict(result: BatchInvarianceResult) -> dict[str, Any]:
    """Convert a BatchInvarianceResult to a dictionary for serialization.

    Args:
        result: The batch result to convert.

    Returns:
        Dictionary representation of the result.
    """
    return {
        "mean_trajectory_cka": result.mean_trajectory_cka,
        "std_trajectory_cka": result.std_trajectory_cka,
        "mean_convergence_delta": result.mean_convergence_delta,
        "mean_peak_cka_delta": result.mean_peak_cka_delta,
        "per_metaphor_trajectory_cka": result.per_metaphor_trajectory_cka,
        "per_family_trajectory_cka": result.per_family_trajectory_cka,
        "result_count": len(result.results),
        "results": [invariance_result_to_dict(r) for r in result.results],
    }
