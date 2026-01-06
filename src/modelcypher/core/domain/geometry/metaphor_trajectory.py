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

"""Metaphor trajectory analysis for layer-wise source→target convergence.

Tracks how source domain activations converge with target domain activations
through model layers. Based on research showing that intermediate layers
capture metaphor processing (arXiv 2505.22563).

The core measurement is CKA between source and target domain activations
at each layer. The "convergence layer" is where this CKA peaks - the layer
where the model maps source domain concepts to target domain concepts.

Example:
    For "TIME IS MONEY", we measure:
    - Source domain: ["dollar", "spend", "save", "invest", ...]
    - Target domain: ["hour", "minute", "schedule", "deadline", ...]
    - At each layer: CKA(source_activations, target_activations)
    - convergence_layer = argmax(CKA across layers)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.numerical_stability import is_nan
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import CMTMapping
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaphorTrajectoryPoint:
    """Single point in a metaphor activation trajectory.

    Captures the geometric relationship between source and target domain
    activations at a specific layer.

    Attributes:
        layer_index: Index of the transformer layer (0-based).
        cka_source_target: CKA between source and target activations at this layer.
        cosine_similarity: Cosine similarity between source/target centroids.
        source_centroid_norm: Geodesic norm of the source domain centroid.
        target_centroid_norm: Geodesic norm of the target domain centroid.
    """

    layer_index: int
    cka_source_target: float
    cosine_similarity: float
    source_centroid_norm: float
    target_centroid_norm: float


@dataclass(frozen=True)
class MetaphorTrajectory:
    """Complete trajectory of a metaphor mapping through model layers.

    Attributes:
        metaphor_id: Unique identifier for the CMT mapping.
        metaphor_name: Human-readable name (e.g., "TIME IS MONEY").
        source_domain: The concrete domain providing structure.
        target_domain: The abstract domain being understood.
        model_id: Identifier for the model that produced this trajectory.
        points: List of trajectory points, one per layer analyzed.
    """

    metaphor_id: str
    metaphor_name: str
    source_domain: str
    target_domain: str
    model_id: str
    points: tuple[MetaphorTrajectoryPoint, ...] = field(default_factory=tuple)

    @property
    def convergence_layer(self) -> int:
        """Layer where source/target become most similar (peak CKA)."""
        if not self.points:
            return -1
        return max(self.points, key=lambda p: p.cka_source_target).layer_index

    @property
    def peak_cka(self) -> float:
        """Maximum CKA achieved across all layers."""
        if not self.points:
            return 0.0
        return max(p.cka_source_target for p in self.points)

    @property
    def layer_count(self) -> int:
        """Number of layers analyzed."""
        return len(self.points)

    def cka_at_layer(self, layer_index: int) -> float | None:
        """Get CKA at a specific layer, or None if not available."""
        for point in self.points:
            if point.layer_index == layer_index:
                return point.cka_source_target
        return None


@dataclass(frozen=True)
class ConvergenceProfile:
    """Raw measurements describing source→target convergence.

    All values are raw geometric measurements, no interpretation.
    """

    convergence_layer: int
    """Layer index with peak CKA."""

    peak_cka: float
    """Maximum CKA achieved during trajectory."""

    early_layer_cka: float
    """Mean CKA in first 25% of layers."""

    mid_layer_cka: float
    """Mean CKA in middle 50% of layers."""

    late_layer_cka: float
    """Mean CKA in last 25% of layers."""

    trajectory_monotonicity: float
    """Geodesic correlation of layer index ranks with CKA (positive = increasing)."""

    layer_count: int
    """Total number of layers analyzed."""


def _compute_spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists.

    Args:
        x: First list of values.
        y: Second list of values.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)

    # Compute ranks
    def rank(values: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val)
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)

    # Spearman correlation using geodesic correlation on ranks.
    backend = get_default_backend()
    rank_x_arr = backend.array(rank_x)
    rank_y_arr = backend.array(rank_y)
    mean_x = backend.mean(rank_x_arr)
    mean_y = backend.mean(rank_y_arr)
    centered_x = rank_x_arr - mean_x
    centered_y = rank_y_arr - mean_y
    centered_x_mat = backend.reshape(centered_x, (1, -1))
    centered_y_mat = backend.reshape(centered_y, (1, -1))
    cos_arr, _ = geodesic_pairwise_metrics(centered_x_mat, centered_y_mat, backend)
    backend.eval(cos_arr)
    if cos_arr.size == 0:
        return 0.0
    corr = float(backend.to_scalar(cos_arr[0]))
    return 0.0 if is_nan(corr, backend) else corr


def compute_convergence_profile(trajectory: MetaphorTrajectory) -> ConvergenceProfile:
    """Compute raw measurements describing source→target convergence.

    Args:
        trajectory: Complete metaphor trajectory through model layers.

    Returns:
        ConvergenceProfile with raw geometric measurements.
    """
    if not trajectory.points:
        return ConvergenceProfile(
            convergence_layer=-1,
            peak_cka=0.0,
            early_layer_cka=0.0,
            mid_layer_cka=0.0,
            late_layer_cka=0.0,
            trajectory_monotonicity=0.0,
            layer_count=0,
        )

    n = len(trajectory.points)
    cka_values = [p.cka_source_target for p in trajectory.points]
    layer_indices = [float(p.layer_index) for p in trajectory.points]

    # Compute layer region boundaries
    early_end = max(1, n // 4)
    late_start = n - max(1, n // 4)

    early_ckas = cka_values[:early_end]
    mid_ckas = cka_values[early_end:late_start] if late_start > early_end else []
    late_ckas = cka_values[late_start:]

    early_mean = sum(early_ckas) / len(early_ckas) if early_ckas else 0.0
    mid_mean = sum(mid_ckas) / len(mid_ckas) if mid_ckas else 0.0
    late_mean = sum(late_ckas) / len(late_ckas) if late_ckas else 0.0

    # Compute monotonicity (Spearman correlation of layer index with CKA)
    monotonicity = _compute_spearman_correlation(layer_indices, cka_values)

    return ConvergenceProfile(
        convergence_layer=trajectory.convergence_layer,
        peak_cka=trajectory.peak_cka,
        early_layer_cka=early_mean,
        mid_layer_cka=mid_mean,
        late_layer_cka=late_mean,
        trajectory_monotonicity=monotonicity,
        layer_count=n,
    )


class MetaphorTrajectoryCollector:
    """Collect layer-wise activations for metaphor source→target mapping.

    Uses the activation extraction pattern from activation_stream.py but
    focused on collecting activations for specific word lists (source/target
    domain exemplars) and computing CKA between them at each layer.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the trajectory collector.

        Args:
            backend: Backend protocol implementation. If None, uses default.
        """
        self.backend = backend or get_default_backend()

    def collect_from_activations(
        self,
        cmt_mapping: "CMTMapping",
        model_id: str,
        layer_activations: dict[int, tuple["Array", "Array"]],
    ) -> MetaphorTrajectory:
        """Collect trajectory from pre-computed activations.

        This is the core method that computes CKA between source and target
        activations at each layer.

        Args:
            cmt_mapping: The CMT mapping being analyzed.
            model_id: Identifier for the model.
            layer_activations: Dict mapping layer_index to (source_acts, target_acts)
                where each is [n_samples, hidden_dim].

        Returns:
            MetaphorTrajectory with one point per layer.
        """
        points: list[MetaphorTrajectoryPoint] = []

        for layer_idx in sorted(layer_activations.keys()):
            source_acts, target_acts = layer_activations[layer_idx]

            # Compute CKA between source and target activations
            cka_result = compute_cka(
                source_acts,
                target_acts,
                backend=self.backend,
            )

            # Compute centroids and cosine similarity
            source_centroid = self.backend.mean(source_acts, axis=0)
            target_centroid = self.backend.mean(target_acts, axis=0)

            # Compute norms and cosine similarity using geodesic distances
            source_mat = self.backend.reshape(source_centroid, (1, -1))
            target_mat = self.backend.reshape(target_centroid, (1, -1))
            source_norm_arr = geodesic_norms(source_mat, self.backend)
            target_norm_arr = geodesic_norms(target_mat, self.backend)
            cos_arr, _ = geodesic_pairwise_metrics(source_mat, target_mat, self.backend)
            self.backend.eval(source_norm_arr, target_norm_arr, cos_arr)
            source_norm = float(self.backend.to_scalar(source_norm_arr[0]))
            target_norm = float(self.backend.to_scalar(target_norm_arr[0]))
            cosine_sim = float(self.backend.to_scalar(cos_arr[0]))

            points.append(
                MetaphorTrajectoryPoint(
                    layer_index=layer_idx,
                    cka_source_target=cka_result.cka,
                    cosine_similarity=cosine_sim,
                    source_centroid_norm=source_norm,
                    target_centroid_norm=target_norm,
                )
            )

        return MetaphorTrajectory(
            metaphor_id=cmt_mapping.id,
            metaphor_name=cmt_mapping.name,
            source_domain=cmt_mapping.source_domain,
            target_domain=cmt_mapping.target_domain,
            model_id=model_id,
            points=tuple(points),
        )

    def collect_trajectory(
        self,
        cmt_mapping: "CMTMapping",
        model_id: str,
        activation_fn: Callable[[list[str], list[int]], dict[int, "Array"]],
        layers: list[int] | None = None,
    ) -> MetaphorTrajectory:
        """Collect trajectory by running exemplars through a model.

        Args:
            cmt_mapping: The CMT mapping being analyzed.
            model_id: Identifier for the model.
            activation_fn: Function that takes (words, layers) and returns
                dict mapping layer_index to activations [n_words, hidden_dim].
            layers: List of layer indices to analyze. If None, function should
                return all layers.

        Returns:
            MetaphorTrajectory with one point per layer.
        """
        # Get activations for source domain exemplars
        source_words = list(cmt_mapping.source_exemplars)
        target_words = list(cmt_mapping.target_exemplars)

        source_activations = activation_fn(source_words, layers or [])
        target_activations = activation_fn(target_words, layers or [])

        # Combine into layer_activations dict
        all_layers = set(source_activations.keys()) & set(target_activations.keys())
        layer_activations = {
            layer: (source_activations[layer], target_activations[layer])
            for layer in all_layers
        }

        return self.collect_from_activations(cmt_mapping, model_id, layer_activations)


def trajectory_to_dict(trajectory: MetaphorTrajectory) -> dict[str, Any]:
    """Convert a MetaphorTrajectory to a dictionary for serialization.

    Args:
        trajectory: The trajectory to convert.

    Returns:
        Dictionary representation of the trajectory.
    """
    return {
        "metaphor_id": trajectory.metaphor_id,
        "metaphor_name": trajectory.metaphor_name,
        "source_domain": trajectory.source_domain,
        "target_domain": trajectory.target_domain,
        "model_id": trajectory.model_id,
        "convergence_layer": trajectory.convergence_layer,
        "peak_cka": trajectory.peak_cka,
        "layer_count": trajectory.layer_count,
        "points": [
            {
                "layer_index": p.layer_index,
                "cka_source_target": p.cka_source_target,
                "cosine_similarity": p.cosine_similarity,
                "source_centroid_norm": p.source_centroid_norm,
                "target_centroid_norm": p.target_centroid_norm,
            }
            for p in trajectory.points
        ],
    }


def convergence_profile_to_dict(profile: ConvergenceProfile) -> dict[str, Any]:
    """Convert a ConvergenceProfile to a dictionary for serialization.

    Args:
        profile: The profile to convert.

    Returns:
        Dictionary representation of the profile.
    """
    return {
        "convergence_layer": profile.convergence_layer,
        "peak_cka": profile.peak_cka,
        "early_layer_cka": profile.early_layer_cka,
        "mid_layer_cka": profile.mid_layer_cka,
        "late_layer_cka": profile.late_layer_cka,
        "trajectory_monotonicity": profile.trajectory_monotonicity,
        "layer_count": profile.layer_count,
    }
