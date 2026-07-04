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

"""Trajectory complexity metrics for measuring intra-layer dynamics.

Measures geometric properties of activation trajectories through model layers
to detect iterative refinement patterns ("looping") vs. direct feedforward flow.

Hypothesis: Reasoning/thinking models exhibit more complex trajectories with:
- Higher path length ratio (wandering vs direct path)
- Higher curvature (direction changes)
- Return visits to similar subspaces (non-adjacent layer similarity)
- Higher trajectory effective rank (exploring diverse directions)

All metrics are raw measurements. Numerical stability guards are
dtype-derived (IEEE 754 machine precision).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_metrics import _linear_cka
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class TrajectoryComplexityResult:
    """Raw trajectory complexity measurements across layers.

    All fields are raw measurements with no interpretation thresholds.
    NaN values indicate insufficient data for computation.
    """

    # Path geometry
    path_length: float
    """Total Euclidean distance traveled through activation space."""

    direct_distance: float
    """Direct (endpoint-to-endpoint) Euclidean distance."""

    path_length_ratio: float
    """Ratio of path_length to direct_distance. Higher = more wandering."""

    # Curvature
    mean_curvature: float
    """Mean turn angle in radians at each intermediate point."""

    max_curvature: float
    """Maximum curvature at any point. High = sharp direction change."""

    curvature_variance: float
    """Variance of curvature values. High = inconsistent direction changes."""

    # Return visits (non-adjacent layer similarity)
    mean_return_cka: float
    """Mean CKA between non-adjacent layers (skip-1 comparisons)."""

    max_return_cka: float
    """Maximum CKA between any non-adjacent layer pair."""

    return_visit_count: int
    """Number of non-adjacent pairs with CKA above sqrt(machine_eps)."""

    # Trajectory spectral properties
    trajectory_effective_rank: float
    """Shannon effective rank of trajectory matrix (layers as samples)."""

    trajectory_spectral_entropy: float
    """Spectral entropy of trajectory singular values."""

    # Metadata
    layer_count: int
    """Number of layers in trajectory."""

    layer_indices: tuple[int, ...]
    """Indices of layers analyzed."""

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path_length": self.path_length,
            "direct_distance": self.direct_distance,
            "path_length_ratio": self.path_length_ratio,
            "mean_curvature": self.mean_curvature,
            "max_curvature": self.max_curvature,
            "curvature_variance": self.curvature_variance,
            "mean_return_cka": self.mean_return_cka,
            "max_return_cka": self.max_return_cka,
            "return_visit_count": self.return_visit_count,
            "trajectory_effective_rank": self.trajectory_effective_rank,
            "trajectory_spectral_entropy": self.trajectory_spectral_entropy,
            "layer_count": self.layer_count,
            "layer_indices": list(self.layer_indices),
        }


class TrajectoryComplexity:
    """Compute trajectory complexity metrics from layer activations.

    Measures how "looping" or iterative the activation path is through layers,
    as opposed to direct feedforward flow.

    Example:
        >>> complexity = TrajectoryComplexity(backend)
        >>> result = complexity.compute(layer_activations)
        >>> print(f"Path ratio: {result.path_length_ratio}")  # Higher = more looping
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize with optional backend.

        Args:
            backend: Backend for tensor operations. If None, uses default.
        """
        self._backend = backend or get_default_backend()

    def compute(
        self,
        layer_activations: dict[int, "Array"],
    ) -> TrajectoryComplexityResult:
        """Compute trajectory complexity from layer activations.

        Args:
            layer_activations: Dict mapping layer_idx -> activation array.
                Arrays should be [hidden_dim] (single sample, mean-pooled)
                or [n_samples, hidden_dim] (will be mean-pooled).

        Returns:
            TrajectoryComplexityResult with raw measurements.
        """
        b = self._backend

        if not layer_activations:
            return self._empty_result()

        # Sort layers by index
        sorted_indices = sorted(layer_activations.keys())
        layer_count = len(sorted_indices)

        if layer_count < 2:
            return self._empty_result(
                layer_count=layer_count,
                layer_indices=tuple(sorted_indices),
            )

        # Normalize activations to 1D vectors (mean-pool if needed)
        points = []
        for idx in sorted_indices:
            act = layer_activations[idx]
            arr = b.array(act) if not hasattr(act, "shape") else act
            b.eval(arr)

            # Mean-pool if 2D
            if len(arr.shape) == 2:
                arr = b.mean(arr, axis=0)
                b.eval(arr)
            elif len(arr.shape) > 2:
                # Flatten to [n_samples, hidden_dim] then mean
                n_features = int(arr.shape[-1])
                flat = b.reshape(arr, (-1, n_features))
                arr = b.mean(flat, axis=0)
                b.eval(arr)

            points.append(arr)

        # Stack into trajectory matrix [n_layers, hidden_dim]
        trajectory = b.stack(points, axis=0)
        b.eval(trajectory)

        # Compute path geometry
        path_length, direct_distance, path_ratio = self._compute_path_geometry(
            points, trajectory
        )

        # Compute curvature
        mean_curv, max_curv, curv_var = self._compute_curvature(points)

        # Compute return visits (non-adjacent CKA)
        mean_return, max_return, return_count = self._compute_return_visits(
            points, sorted_indices
        )

        # Compute trajectory spectral properties
        eff_rank, spectral_entropy = self._compute_trajectory_spectrum(trajectory)

        return TrajectoryComplexityResult(
            path_length=path_length,
            direct_distance=direct_distance,
            path_length_ratio=path_ratio,
            mean_curvature=mean_curv,
            max_curvature=max_curv,
            curvature_variance=curv_var,
            mean_return_cka=mean_return,
            max_return_cka=max_return,
            return_visit_count=return_count,
            trajectory_effective_rank=eff_rank,
            trajectory_spectral_entropy=spectral_entropy,
            layer_count=layer_count,
            layer_indices=tuple(sorted_indices),
        )

    def _empty_result(
        self,
        layer_count: int = 0,
        layer_indices: tuple[int, ...] = (),
    ) -> TrajectoryComplexityResult:
        """Return result for degenerate input."""
        return TrajectoryComplexityResult(
            path_length=0.0,
            direct_distance=0.0,
            path_length_ratio=float("nan"),
            mean_curvature=float("nan"),
            max_curvature=float("nan"),
            curvature_variance=float("nan"),
            mean_return_cka=float("nan"),
            max_return_cka=float("nan"),
            return_visit_count=0,
            trajectory_effective_rank=0.0,
            trajectory_spectral_entropy=0.0,
            layer_count=layer_count,
            layer_indices=layer_indices,
        )

    def _compute_path_geometry(
        self,
        points: list["Array"],
        trajectory: "Array",
    ) -> tuple[float, float, float]:
        """Compute path length, direct distance, and ratio.

        Args:
            points: List of activation vectors for each layer.
            trajectory: Stacked trajectory matrix [n_layers, hidden_dim].

        Returns:
            Tuple of (path_length, direct_distance, path_length_ratio).
        """
        b = self._backend

        # Path length: sum of consecutive Euclidean distances
        path_length = 0.0
        for i in range(len(points) - 1):
            diff = points[i + 1] - points[i]
            dist = b.sqrt(b.sum(diff * diff))
            b.eval(dist)
            path_length += float(b.to_scalar(dist))

        # Direct distance: first to last
        if len(points) >= 2:
            diff = points[-1] - points[0]
            direct = b.sqrt(b.sum(diff * diff))
            b.eval(direct)
            direct_distance = float(b.to_scalar(direct))
        else:
            direct_distance = 0.0

        # Path ratio (with division protection)
        eps = division_epsilon(b, trajectory)
        if direct_distance > eps:
            path_ratio = path_length / direct_distance
        else:
            path_ratio = float("nan") if path_length > eps else 1.0

        return path_length, direct_distance, path_ratio

    def _compute_curvature(
        self,
        points: list["Array"],
    ) -> tuple[float, float, float]:
        """Compute curvature statistics along trajectory.

        Curvature at point i is the angle between vectors (p[i]-p[i-1]) and (p[i+1]-p[i]).

        Args:
            points: List of activation vectors for each layer.

        Returns:
            Tuple of (mean_curvature, max_curvature, curvature_variance).
        """
        b = self._backend

        if len(points) < 3:
            return float("nan"), float("nan"), float("nan")

        curvatures = []
        # Division guard: sqrt(eps_f32) — vector norms below this are
        # indistinguishable from zero in float32 precision.
        # IEEE 754: eps_f32 = 2^-23, sqrt(eps_f32) ≈ 3.45e-4.
        eps = math.sqrt(math.ldexp(1.0, -23))

        for i in range(1, len(points) - 1):
            # Vectors before and after point i
            v1 = points[i] - points[i - 1]  # incoming
            v2 = points[i + 1] - points[i]  # outgoing

            # Norms
            norm1 = b.sqrt(b.sum(v1 * v1))
            norm2 = b.sqrt(b.sum(v2 * v2))
            b.eval(norm1, norm2)

            norm1_val = float(b.to_scalar(norm1))
            norm2_val = float(b.to_scalar(norm2))

            if norm1_val < eps or norm2_val < eps:
                # Degenerate: no movement
                curvatures.append(0.0)
                continue

            # Cosine of angle between vectors
            dot = b.sum(v1 * v2)
            b.eval(dot)
            dot_val = float(b.to_scalar(dot))
            cos_angle = dot_val / (norm1_val * norm2_val)

            # Clamp for numerical stability
            cos_angle = max(-1.0, min(1.0, cos_angle))

            # Curvature = angle in radians (0 = straight, pi = reversal)
            angle = math.acos(cos_angle)
            curvatures.append(angle)

        if not curvatures:
            return float("nan"), float("nan"), float("nan")

        mean_curv = sum(curvatures) / len(curvatures)
        max_curv = max(curvatures)

        if len(curvatures) > 1:
            variance = sum((c - mean_curv) ** 2 for c in curvatures) / len(curvatures)
        else:
            variance = 0.0

        return mean_curv, max_curv, variance

    def _compute_return_visits(
        self,
        points: list["Array"],
        layer_indices: list[int],
    ) -> tuple[float, float, int]:
        """Compute return visit metrics (non-adjacent layer CKA).

        Args:
            points: List of activation vectors for each layer.
            layer_indices: Sorted layer indices.

        Returns:
            Tuple of (mean_return_cka, max_return_cka, return_visit_count).
        """
        b = self._backend

        if len(points) < 3:
            return float("nan"), float("nan"), 0

        # Compute CKA for non-adjacent pairs (skip-1 and beyond)
        cka_values = []
        eps = division_epsilon(b, points[0])

        for i in range(len(points)):
            for j in range(i + 2, len(points)):  # Skip adjacent (j = i+1)
                # Reshape to [1, hidden_dim] for CKA
                p1 = b.reshape(points[i], (1, -1))
                p2 = b.reshape(points[j], (1, -1))

                cka = _linear_cka(p1, p2, b)
                if not math.isnan(cka):
                    cka_values.append(cka)

        if not cka_values:
            return float("nan"), float("nan"), 0

        mean_cka = sum(cka_values) / len(cka_values)
        max_cka = max(cka_values)

        # Count "significant" returns (CKA above sqrt(machine_eps))
        threshold = math.sqrt(eps)
        return_count = sum(1 for c in cka_values if c > threshold)

        return mean_cka, max_cka, return_count

    def _compute_trajectory_spectrum(
        self,
        trajectory: "Array",
    ) -> tuple[float, float]:
        """Compute spectral properties of trajectory matrix.

        Treats trajectory as [n_layers, hidden_dim] and computes effective rank.

        Args:
            trajectory: Trajectory matrix [n_layers, hidden_dim].

        Returns:
            Tuple of (effective_rank, spectral_entropy).
        """
        b = self._backend

        n_layers = int(trajectory.shape[0])
        if n_layers < 2:
            return 0.0, 0.0

        # Center trajectory
        mean = b.mean(trajectory, axis=0)
        centered = trajectory - mean
        b.eval(centered)

        # SVD
        _, singular_values, _ = geodesic_svd(b, centered)
        b.eval(singular_values)

        n_sv = int(singular_values.shape[0])
        if n_sv == 0:
            return 0.0, 0.0

        # Eigenvalues from singular values
        eigvals = singular_values * singular_values
        sum_eig = b.sum(eigvals)
        b.eval(sum_eig)

        sum_eig_val = float(b.to_scalar(sum_eig))
        eps = division_epsilon(b, eigvals)

        if sum_eig_val <= eps:
            return 0.0, 0.0

        # Normalized energy distribution
        p = eigvals / sum_eig_val
        b.eval(p)

        # Shannon entropy with safe log
        log_eps = safe_log_epsilon(b, eigvals)
        eps_arr = b.full(p.shape, log_eps, dtype=p.dtype)
        p_safe = b.where(p > log_eps, p, eps_arr)
        entropy_terms = -p * b.log(p_safe)
        # Zero out terms where p was effectively zero
        entropy_terms = b.where(p > log_eps, entropy_terms, b.zeros_like(entropy_terms))
        entropy_arr = b.sum(entropy_terms)
        b.eval(entropy_arr)
        spectral_entropy = float(b.to_scalar(entropy_arr))

        # Shannon effective rank: exp(H)
        eff_rank_arr = b.exp(entropy_arr)
        b.eval(eff_rank_arr)
        effective_rank = float(b.to_scalar(eff_rank_arr))

        return effective_rank, spectral_entropy


def compute_trajectory_complexity(
    layer_activations: dict[int, "Array"],
    backend: "Backend | None" = None,
) -> TrajectoryComplexityResult:
    """Convenience function for computing trajectory complexity.

    Args:
        layer_activations: Dict mapping layer_idx -> activation array.
        backend: Optional backend. If None, uses default.

    Returns:
        TrajectoryComplexityResult with raw measurements.
    """
    tc = TrajectoryComplexity(backend)
    return tc.compute(layer_activations)


__all__ = [
    "TrajectoryComplexity",
    "TrajectoryComplexityResult",
    "compute_trajectory_complexity",
]
