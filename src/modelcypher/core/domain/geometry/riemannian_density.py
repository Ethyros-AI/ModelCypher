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

"""Riemannian density estimation on representation manifolds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    _mask_sum,
    _promote_precision,
    division_epsilon,
    e_value,
    exp_scalar,
    geodesic_svd,
    inf_value,
    lgamma_scalar,
    log_scalar,
    machine_epsilon,
    pi_value,
    power_iteration_eigh,
    regularization_epsilon,
    safe_inverse,
    sqrt_scalar,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Array, Backend

from .manifold_curvature import (
    LocalCurvature,
    SectionalCurvatureEstimator,
)
from .riemannian_utils import (
    GeodesicDistanceResult,
    RiemannianGeometry,
)

logger = logging.getLogger(__name__)


def _logdet_from_eigenvalues(
    backend: "Backend",
    eigenvalues: "Array",
) -> float | None:
    """Return log(det) from eigenvalues, or None if non-positive."""
    min_eig = backend.min(eigenvalues)
    backend.eval(min_eig)
    if float(backend.to_scalar(min_eig)) <= 0.0:
        return None
    tiny = tiny_value(backend, eigenvalues)
    clamped = backend.maximum(eigenvalues, backend.full(eigenvalues.shape, tiny))
    logdet_arr = backend.sum(backend.log(clamped))
    backend.eval(logdet_arr)
    return float(backend.to_scalar(logdet_arr))


class InfluenceType(str, Enum):
    """Type of probability density function for concept influence."""

    GAUSSIAN = "gaussian"  # Standard Riemannian Gaussian
    LAPLACIAN = "laplacian"  # Heavy-tailed, robust to outliers
    STUDENT_T = "student_t"  # Even heavier tails, df parameter
    UNIFORM = "uniform"  # Uniform within geodesic ball


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - influence_type: derived from data kurtosis
#   - kurtosis > 3 + eps → Student-t (heavy tails)
#   - kurtosis <= 3 + eps → Gaussian
# - student_t_df: derived from kurtosis: df = 4 + 6/(kurtosis - 3)
# - covariance_regularization: from machine epsilon
# - curvature_correction: enabled by default
# - k_neighbors: minimum k that yields a connected geodesic graph
# =============================================================================


@dataclass
class ConceptVolume:
    """Concept volume as a distribution on the manifold."""

    concept_id: str
    centroid: "Array"
    covariance: "Array"
    geodesic_radius: float
    local_curvature: LocalCurvature | None
    num_samples: int
    influence_type: InfluenceType = InfluenceType.GAUSSIAN
    student_t_df: float = 3.0

    # Optional raw activations for cross-dimensional CKA comparison
    # When comparing volumes of different dimensions, CKA uses these
    # to compute Gram matrices (n x n) which are dimension-agnostic.
    raw_activations: "Array | None" = field(default=None, repr=False)

    # Geodesic context for proper log map computation in density_at
    # Stores k-NN graph and distance matrix for curvature-aware density
    _geodesic_context: "GeodesicDistanceResult | None" = field(default=None, repr=False)

    # Cached values for efficiency
    _precision: "Array | None" = field(default=None, repr=False)
    _geo_from_centroid: "Array | None" = field(default=None, repr=False)
    _log_det_cov: float | None = field(default=None, repr=False)
    _cov_eigenvalues: "Array | None" = field(default=None, repr=False)
    _cov_eigenvectors: "Array | None" = field(default=None, repr=False)
    _raw_activations_sq: "Array | None" = field(default=None, repr=False)
    _raw_activations_t: "Array | None" = field(default=None, repr=False)
    _log_norm_gaussian: float | None = field(default=None, repr=False)
    _log_norm_student_t: float | None = field(default=None, repr=False)
    _cached_laplacian_norm: float | None = field(default=None, repr=False)
    _volume: float | None = field(default=None, repr=False)
    _cached_uniform_norm: float | None = field(default=None, repr=False)

    @property
    def dimension(self) -> int:
        """Feature dimension."""
        shape = self.centroid.shape
        return int(shape[0]) if len(shape) == 1 else int(shape[-1])

    @property
    def precision(self) -> "Array":
        """Inverse covariance."""
        if self._precision is None:
            backend = get_default_backend()
            # Use safe_inverse with automatic condition checking and regularization
            precision, _ = safe_inverse(backend, self.covariance, regularize=True)
            backend.eval(precision)
            object.__setattr__(self, "_precision", precision)
        return self._precision

    @property
    def log_det_covariance(self) -> float:
        """Log determinant of covariance."""
        if self._log_det_cov is None:
            backend = get_default_backend()
            # slogdet returns (sign, logdet) - we compute via eigenvalues (geodesic - GPU-only)
            eigenvalues, _ = self._covariance_eigendecomposition()
            logdet = _logdet_from_eigenvalues(backend, eigenvalues)
            if logdet is None:
                logdet = -inf_value(backend)
            object.__setattr__(self, "_log_det_cov", logdet)
        return self._log_det_cov

    @property
    def volume(self) -> float:
        """Approximate concept volume."""
        if self._volume is None:
            backend = get_default_backend()
            d = self.dimension
            if self.influence_type == InfluenceType.UNIFORM:
                # Volume of d-sphere: (pi^(d/2) / Gamma(d/2 + 1)) * r^d
                if self.geodesic_radius <= 0:
                    volume_val = 0.0
                else:
                    pi = pi_value(backend)
                    log_vol = (
                        (d / 2) * log_scalar(pi, backend)
                        - lgamma_scalar(d / 2 + 1.0, backend)
                        + d * log_scalar(self.geodesic_radius, backend)
                    )
                    volume_val = exp_scalar(log_vol, backend)
            else:
                # Gaussian effective volume
                pi = pi_value(backend)
                e = e_value(backend)
                volume_val = exp_scalar(
                    0.5 * self.log_det_covariance
                    + d / 2 * log_scalar(2 * pi * e, backend),
                    backend,
                )
            object.__setattr__(self, "_volume", volume_val)
        return self._volume

    @property
    def effective_radius(self) -> float:
        """Geometric-mean radius from covariance."""
        backend = get_default_backend()
        # Geodesic eigendecomposition (GPU-only)
        eigenvalues, _ = self._covariance_eigendecomposition()
        # Geometric mean via log: exp(mean(log(max(eig, tiny))))
        tiny = tiny_value(backend, eigenvalues)
        clamped = backend.maximum(eigenvalues, backend.full(eigenvalues.shape, tiny))
        backend.eval(clamped)
        # Verify positivity - if any values still non-positive, add regularization
        min_clamped = backend.min(clamped)
        backend.eval(min_clamped)
        if float(backend.to_scalar(min_clamped)) <= 0.0:
            reg = regularization_epsilon(backend, eigenvalues)
            clamped = clamped + reg
            backend.eval(clamped)
        mean_log = backend.mean(backend.log(clamped))
        backend.eval(mean_log)
        return exp_scalar(0.5 * float(backend.to_scalar(mean_log)), backend)

    def _covariance_eigendecomposition(self) -> tuple["Array", "Array"]:
        """Cache covariance eigendecomposition for reuse across metrics."""
        if self._cov_eigenvalues is None or self._cov_eigenvectors is None:
            backend = get_default_backend()
            n_cov = int(self.covariance.shape[0])
            eigenvalues, eigenvectors = power_iteration_eigh(backend, self.covariance, k=n_cov)
            backend.eval(eigenvalues, eigenvectors)
            object.__setattr__(self, "_cov_eigenvalues", eigenvalues)
            object.__setattr__(self, "_cov_eigenvectors", eigenvectors)
        return self._cov_eigenvalues, self._cov_eigenvectors

    def _gaussian_log_norm(self) -> float:
        if self._log_norm_gaussian is None:
            backend = get_default_backend()
            d = self.dimension
            pi = pi_value(backend)
            log_norm = -0.5 * (d * log_scalar(2 * pi, backend) + self.log_det_covariance)
            object.__setattr__(self, "_log_norm_gaussian", log_norm)
        return self._log_norm_gaussian

    def _student_t_log_norm(self) -> float:
        if self._log_norm_student_t is None:
            backend = get_default_backend()
            d = self.dimension
            nu = float(self.student_t_df)
            pi = pi_value(backend)
            log_norm = (
                lgamma_scalar((nu + d) / 2, backend)
                - lgamma_scalar(nu / 2, backend)
                - d / 2 * log_scalar(nu * pi, backend)
                - 0.5 * self.log_det_covariance
            )
            object.__setattr__(self, "_log_norm_student_t", log_norm)
        return self._log_norm_student_t

    def _laplacian_norm(self) -> float:
        if self._cached_laplacian_norm is None:
            backend = get_default_backend()
            d = self.dimension
            log_two = log_scalar(2.0, backend)
            norm_val = exp_scalar(-d * log_two, backend)
            object.__setattr__(self, "_cached_laplacian_norm", norm_val)
        return self._cached_laplacian_norm

    def _uniform_norm(self) -> float:
        if self._cached_uniform_norm is None:
            volume_val = self.volume
            inv = 1.0 / volume_val if volume_val > 0.0 else 0.0
            object.__setattr__(self, "_cached_uniform_norm", inv)
        return self._cached_uniform_norm

    def _geodesic_distance_from_centroid(
        self,
        point_arr: "Array",
        centroid_arr: "Array",
    ) -> float:
        """Compute geodesic distance from centroid using cached graph."""
        backend = get_default_backend()
        self._require_geodesic_context()

        rg = RiemannianGeometry(backend)
        geo_from_centroid = self._geo_from_centroid
        if geo_from_centroid is None:
            geo_from_centroid = rg._geodesic_distances_from_query(
                self.raw_activations,
                centroid_arr,
                geo_result=self._geodesic_context,
            )
            backend.eval(geo_from_centroid)
            self._geo_from_centroid = geo_from_centroid

        geo_from_query = rg._geodesic_distances_from_query(
            self.raw_activations,
            point_arr,
            geo_result=self._geodesic_context,
        )
        backend.eval(geo_from_query)
        total_dists = geo_from_centroid + geo_from_query
        geo_dist = backend.min(total_dists)
        backend.eval(geo_dist)
        return float(backend.to_scalar(geo_dist))

    def _require_geodesic_context(self) -> None:
        if self._geodesic_context is None or self.raw_activations is None:
            raise ValueError(
                "Geodesic context required for Riemannian log map. "
                "Build volumes with store_raw_activations=True."
            )

    def _compute_tangent_vector(self, point: "Array") -> "Array":
        """Compute log-map tangent from centroid to point."""
        backend = get_default_backend()
        point_arr = _promote_precision(backend.array(point), backend)
        centroid_arr = _promote_precision(self.centroid, backend)
        diff = point_arr - centroid_arr
        geo_dist_float = self._geodesic_distance_from_centroid(point_arr, centroid_arr)

        # Tangent norm via geodesic distance from origin
        diff_vec = backend.reshape(diff, (1, -1))
        diff_norm_arr = geodesic_norms(diff_vec, backend)
        backend.eval(diff_norm_arr)
        diff_norm = float(backend.to_scalar(diff_norm_arr[0]))

        # Scale factor: geodesic / chord
        # Use machine_epsilon for near-zero check
        if diff_norm < machine_epsilon(backend, diff):
            return diff  # Point is at centroid
        scale = geo_dist_float / diff_norm

        return diff * scale

    def _density_from_mahal_sq(self, mahal_sq: "Array") -> "Array":
        backend = get_default_backend()
        d = self.dimension

        if self.influence_type == InfluenceType.GAUSSIAN:
            log_norm = self._gaussian_log_norm()
            return backend.exp(backend.array(log_norm) - 0.5 * mahal_sq)

        if self.influence_type == InfluenceType.LAPLACIAN:
            mahal = backend.sqrt(backend.maximum(mahal_sq, backend.zeros_like(mahal_sq)))
            norm_val = self._laplacian_norm()
            return backend.exp(-mahal) * backend.array(norm_val)

        if self.influence_type == InfluenceType.STUDENT_T:
            nu = float(self.student_t_df)
            if nu <= 0:
                raise ValueError("student_t_df must be positive")
            log_norm = self._student_t_log_norm()
            # Use exp(exponent * log(base)) instead of pow since backend doesn't have pow
            base = 1 + mahal_sq / nu
            exponent = -(nu + d) / 2
            power_term = backend.exp(exponent * backend.log(base))
            return backend.exp(backend.array(log_norm)) * power_term

        if self.influence_type == InfluenceType.UNIFORM:
            inv_volume = self._uniform_norm()
            inside = mahal_sq <= self.geodesic_radius**2
            return backend.where(
                inside,
                backend.full(mahal_sq.shape, inv_volume),
                backend.zeros(mahal_sq.shape),
            )

        return backend.zeros(mahal_sq.shape)

    def density_at(self, point: "Array") -> float:
        """Compute density at a point (requires geodesic context)."""
        backend = get_default_backend()

        # Compute tangent vector (log map if geodesic context available)
        tangent = self._compute_tangent_vector(point)

        # mahal_sq = tangent @ precision @ tangent
        temp = backend.matmul(tangent, self.precision)
        mahal_sq_arr = backend.matmul(temp, tangent)
        mahal_sq_arr = backend.reshape(mahal_sq_arr, (1,))
        densities = self._density_from_mahal_sq(mahal_sq_arr)
        backend.eval(densities)
        return float(backend.to_scalar(densities))

    def mahalanobis_distance(self, point: "Array") -> float:
        """Mahalanobis distance from centroid."""
        backend = get_default_backend()
        diff = point - self.centroid
        # mahal_sq = diff @ precision @ diff
        temp = backend.matmul(diff, self.precision)
        mahal_sq_arr = backend.matmul(temp, diff)
        backend.eval(mahal_sq_arr)
        mahal_sq = float(backend.to_scalar(mahal_sq_arr))
        return sqrt_scalar(mahal_sq, backend)

    def geodesic_distance(self, point: "Array") -> float:
        """Geodesic distance from centroid."""
        return self.geodesic_distance_to(point)

    def geodesic_distance_to(self, point: "Array") -> float:
        """Geodesic distance from centroid using cached graph when available."""
        backend = get_default_backend()
        point_arr = backend.array(point)

        # Use cached geodesic context when available for precision and speed.
        if self._geodesic_context is not None and self.raw_activations is not None:
            centroid_arr = backend.array(self.centroid)
            return self._geodesic_distance_from_centroid(point_arr, centroid_arr)

        # Fallback for volumes without raw activations: 2-point geodesic.
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        centroid_2d = backend.reshape(self.centroid, (1, -1))
        point_2d = backend.reshape(point_arr, (1, -1))
        points = backend.concatenate([centroid_2d, point_2d], axis=0)
        points_arr = _promote_precision(points, backend)
        backend.eval(points_arr)

        geo_dist = geodesic_distance_matrix(points_arr, k_neighbors=1, backend=backend)
        geo_elem = geo_dist[0, 1]
        backend.eval(geo_elem)
        return float(backend.to_scalar(geo_elem))

    def contains(self, point: "Array") -> bool:
        """Check if point lies within the geodesic radius."""
        dist = self.geodesic_distance_to(point)
        return dist <= self.geodesic_radius

    def mahalanobis_distance_batch(self, points: "Array") -> "Array":
        """Mahalanobis distances from centroid for a batch."""
        backend = get_default_backend()
        # diff: (n, d)
        diff = points - self.centroid
        # mahal_sq = sum((diff @ precision) * diff, axis=1)
        temp = backend.matmul(diff, self.precision)  # (n, d)
        mahal_sq = backend.sum(temp * diff, axis=1)  # (n,)
        backend.eval(mahal_sq)
        return backend.sqrt(backend.maximum(mahal_sq, backend.zeros_like(mahal_sq)))

    def _compute_tangent_vectors_batch(self, points: "Array") -> "Array":
        """Compute log-map tangents from centroid for a batch."""
        backend = get_default_backend()
        points_arr = _promote_precision(backend.array(points), backend)
        centroid_arr = _promote_precision(self.centroid, backend)
        diff = points_arr - centroid_arr  # (n, d)

        # Geodesic context is required - no chord-only fallback on curved manifolds
        self._require_geodesic_context()

        backend.eval(points_arr)
        rg = RiemannianGeometry(backend)

        # Compute geodesic distances from centroid to all activation points
        geo_from_centroid = self._geo_from_centroid
        if geo_from_centroid is None:
            geo_from_centroid = rg._geodesic_distances_from_query(
                self.raw_activations,
                centroid_arr,
                geo_result=self._geodesic_context,
            )
            backend.eval(geo_from_centroid)
            self._geo_from_centroid = geo_from_centroid

        # Compute geodesic distances to centroid for all points at once.
        pts_sq = backend.sum(points_arr * points_arr, axis=1, keepdims=True)
        acts_sq = self._raw_activations_sq
        if acts_sq is None:
            acts_sq = backend.sum(
                self.raw_activations * self.raw_activations, axis=1, keepdims=True
            )
            backend.eval(acts_sq)
            self._raw_activations_sq = acts_sq
        raw_t = self._raw_activations_t
        if raw_t is None:
            raw_t = backend.transpose(self.raw_activations)
            backend.eval(raw_t)
            self._raw_activations_t = raw_t
        cross = backend.matmul(points_arr, raw_t)
        dist_sq = pts_sq + backend.transpose(acts_sq) - 2.0 * cross
        dist_sq = backend.maximum(dist_sq, backend.zeros_like(dist_sq))

        n_acts = int(self.raw_activations.shape[0])
        k_neighbors = (
            int(self._geodesic_context.k_neighbors)
            if self._geodesic_context is not None
            else n_acts
        )
        k_neighbors = max(1, min(k_neighbors, n_acts))

        if k_neighbors >= n_acts:
            chord_dist = backend.sqrt(dist_sq)
            total_dists = chord_dist + backend.reshape(geo_from_centroid, (1, -1))
            geo_dist = backend.min(total_dists, axis=1)
        else:
            tie_eps = machine_epsilon(backend, dist_sq)
            col_indices = backend.arange(n_acts)
            col_row = backend.reshape(col_indices, (1, n_acts))
            dist_for_sort = dist_sq + backend.astype(col_row, dist_sq.dtype) * tie_eps
            kth = max(0, min(k_neighbors - 1, n_acts - 1))
            partitioned = backend.argpartition(dist_for_sort, kth, axis=1)
            neighbor_idx = partitioned[:, :k_neighbors]

            neighbor_dist_sq = backend.take_along_axis(dist_sq, neighbor_idx, axis=1)
            neighbor_dist_sq = backend.maximum(
                neighbor_dist_sq, backend.zeros_like(neighbor_dist_sq)
            )
            neighbor_dists = backend.sqrt(neighbor_dist_sq)

            geo_row = backend.reshape(geo_from_centroid, (1, n_acts))
            geo_row = backend.broadcast_to(geo_row, dist_sq.shape)
            neighbor_geo = backend.take_along_axis(geo_row, neighbor_idx, axis=1)
            total_dists = neighbor_dists + neighbor_geo
            geo_dist = backend.min(total_dists, axis=1)

        # Chord length from centroid to each point (direction vector for tangent).
        diff_sq = backend.sum(diff * diff, axis=1)
        diff_sq = backend.maximum(diff_sq, backend.zeros_like(diff_sq))
        diff_norms = backend.sqrt(diff_sq)

        eps = machine_epsilon(backend, diff)
        safe_norms = backend.maximum(diff_norms, backend.full(diff_norms.shape, eps))
        scales_arr = geo_dist / safe_norms
        scales_arr = backend.where(diff_norms > eps, scales_arr, backend.ones_like(scales_arr))
        scales_col = backend.reshape(scales_arr, (-1, 1))
        tangent_vectors = diff * scales_col

        return tangent_vectors

    def density_at_batch(self, points: "Array") -> "Array":
        """Compute density at multiple points."""
        backend = get_default_backend()
        # Compute tangent vectors (log map if geodesic context available)
        tangent = self._compute_tangent_vectors_batch(points)

        # Mahalanobis squared using tangent vectors
        temp = backend.matmul(tangent, self.precision)  # (n, d)
        mahal_sq = backend.sum(temp * tangent, axis=1)  # (n,)

        densities = self._density_from_mahal_sq(mahal_sq)
        backend.eval(densities)
        return densities

    def contains_batch(self, points: "Array") -> "Array":
        """Check if multiple points are within the geodesic radius."""
        backend = get_default_backend()
        mahal_dist = self.mahalanobis_distance_batch(points)
        result = mahal_dist <= self.geodesic_radius
        backend.eval(result)
        return result


@dataclass
class ConceptVolumeRelation:
    """Relationship between two concept volumes."""

    volume_a: ConceptVolume
    volume_b: ConceptVolume
    # Overlap metrics
    overlap_coefficient: float  # Szymkiewicz-Simpson coefficient
    jaccard_index: float  # Intersection / Union volume ratio
    bhattacharyya_coefficient: float  # Distribution similarity
    # Distance metrics (geodesic, curvature-aware)
    centroid_distance: float  # Geodesic between centroids
    geodesic_centroid_distance: float  # Geodesic between centroids (same value)
    mahalanobis_distance_ab: float  # Mahal from A's perspective
    mahalanobis_distance_ba: float  # Mahal from B's perspective
    # Curvature mismatch
    curvature_divergence: float
    # Dimensionality overlap
    subspace_alignment: float  # Alignment of principal axes


class RiemannianDensityEstimator:
    """Estimate concept volumes with curvature awareness."""

    def __init__(self) -> None:
        self.curvature_estimator = SectionalCurvatureEstimator()

    def estimate_concept_volume(
        self,
        concept_id: str,
        activations: "Array",
        metric_fn: Callable[["Array"], "Array"] | None = None,
        store_raw_activations: bool = True,
    ) -> ConceptVolume:
        """Estimate a concept volume from activation samples."""
        backend = get_default_backend()
        # Convert to backend array if needed (handles numpy from tests)
        activations = _promote_precision(backend.array(activations), backend)
        backend.eval(activations)
        shape = activations.shape
        n, d = int(shape[0]), int(shape[1])

        if n < 2:
            # Single sample - return point mass with Gaussian influence
            geodesic_context = None
            if store_raw_activations:
                rg = RiemannianGeometry(backend)
                geodesic_context = rg.geodesic_distances(activations, k_neighbors=1)
            return ConceptVolume(
                concept_id=concept_id,
                centroid=activations[0],
                covariance=backend.eye(d) * regularization_epsilon(backend, activations),
                geodesic_radius=0.0,
                local_curvature=None,
                num_samples=n,
                influence_type=InfluenceType.GAUSSIAN,  # Default for single point
                student_t_df=0.0,
                raw_activations=activations if store_raw_activations else None,
                _geodesic_context=geodesic_context if store_raw_activations else None,
            )

        # Compute geodesic context for proper log map in density_at
        # k is derived from the data using elbow detection on k-NN distances
        rg = RiemannianGeometry(backend)
        geodesic_context = rg.geodesic_distances(activations, k_neighbors=None)
        k_neighbors = geodesic_context.k_neighbors

        # Compute centroid using Fréchet mean (minimizes squared geodesic distances).
        result = rg.frechet_mean(
            activations,
            tolerance=regularization_epsilon(backend, activations),
            k_neighbors=k_neighbors,
            geo_result=geodesic_context,
            # max_iterations auto-derived from n
        )
        centroid = result.mean
        backend.eval(centroid)

        geo_from_centroid = rg._geodesic_distances_from_query(
            activations,
            centroid,
            geo_result=geodesic_context,
        )
        backend.eval(geo_from_centroid)

        # Estimate local curvature at centroid when sample count allows.
        local_curvature = None
        if n >= d + 2:
            try:
                local_curvature = self.curvature_estimator.estimate_local_curvature(
                    centroid, activations, metric_fn
                )
            except Exception as e:
                logger.warning(f"Curvature estimation failed for {concept_id}: {e}")

        # Compute covariance with curvature correction
        covariance = self._estimate_covariance(
            activations,
            centroid,
            local_curvature,
            metric_fn,
            geo_result=geodesic_context,
            geo_from_centroid=geo_from_centroid,
        )

        # Compute geodesic radius (extent of activations from centroid)
        geodesic_radius = self._compute_geodesic_radius(
            activations,
            centroid,
            geo_from_centroid=geo_from_centroid,
            geo_result=geodesic_context,
        )

        # Derive influence type and student_t_df from data kurtosis
        # Formula: df = 4 + 6/(kurtosis - 3), where kurtosis > 3 for heavy tails
        # Student-t with df=3 has kurtosis=inf, df→∞ approaches Gaussian (kurtosis=3)
        # Compute excess kurtosis from centered activations
        centered = activations - centroid
        centered_sq = centered * centered
        var = backend.mean(centered_sq)
        fourth = backend.mean(centered_sq * centered_sq)
        backend.eval(var, fourth)
        var_val = float(backend.to_scalar(var))
        fourth_val = float(backend.to_scalar(fourth))
        div_eps = division_epsilon(backend, activations)
        if var_val > div_eps:
            kurtosis = fourth_val / (var_val * var_val)
        else:
            kurtosis = 3.0  # Default to Gaussian kurtosis

        # Derive influence_type and student_t_df from kurtosis.
        # Student-t kurtosis: 3 + 6/(df - 4) for df > 4.
        kurtosis_floor = 3.0 + division_epsilon(backend, activations)
        if kurtosis > kurtosis_floor:
            influence_type = InfluenceType.STUDENT_T
            student_t_df = 4.0 + 6.0 / (kurtosis - 3.0)
        else:
            influence_type = InfluenceType.GAUSSIAN
            student_t_df = 0.0

        return ConceptVolume(
            concept_id=concept_id,
            centroid=centroid,
            covariance=covariance,
            geodesic_radius=geodesic_radius,
            local_curvature=local_curvature,
            num_samples=n,
            influence_type=influence_type,
            student_t_df=student_t_df,
            raw_activations=activations if store_raw_activations else None,
            _geodesic_context=geodesic_context if store_raw_activations else None,
            _geo_from_centroid=geo_from_centroid if store_raw_activations else None,
        )

    def compute_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Compute overlap and distance metrics between two volumes."""
        # Use CKA for all comparisons when raw_activations available
        # CKA is dimension-agnostic and GPU-accelerated
        if volume_a.raw_activations is not None and volume_b.raw_activations is not None:
            return self._compute_cka_relation(volume_a, volume_b)

        # Fallback to geodesic-based comparison only when raw_activations not available
        # (e.g., when loading cached volumes without activations)
        return self._compute_geodesic_relation(volume_a, volume_b)

    def _compute_geodesic_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """Geodesic-only comparison for cached volumes without activations."""
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        backend = get_default_backend()

        # Must be same dimension for centroid comparison
        if volume_a.dimension != volume_b.dimension:
            raise ValueError(
                f"Geodesic comparison requires same dimensions. "
                f"Got {volume_a.dimension} vs {volume_b.dimension}. "
                f"Enable store_raw_activations=True for cross-dimensional comparison."
            )

        # Handle edge case: coincident centroids have geodesic distance 0 by definition
        diff = volume_a.centroid - volume_b.centroid
        diff_norm = geodesic_norms(backend.reshape(diff, (1, -1)), backend)
        backend.eval(diff_norm)
        centroid_diff = float(backend.to_scalar(diff_norm[0]))
        if centroid_diff < machine_epsilon(backend, diff):
            centroid_distance = 0.0
        else:
            centroid_a_2d = backend.reshape(volume_a.centroid, (1, -1))
            centroid_b_2d = backend.reshape(volume_b.centroid, (1, -1))
            centroids = backend.concatenate([centroid_a_2d, centroid_b_2d], axis=0)
            centroids_arr = _promote_precision(centroids, backend)

            geo_dist = geodesic_distance_matrix(centroids_arr, k_neighbors=1, backend=backend)
            geo_elem = geo_dist[0, 1]
            backend.eval(geo_elem)
            centroid_distance = float(backend.to_scalar(geo_elem))

        geodesic_centroid_distance = centroid_distance

        # Mahalanobis distances (asymmetric)
        mahal_ab = volume_a.mahalanobis_distance(volume_b.centroid)
        mahal_ba = volume_b.mahalanobis_distance(volume_a.centroid)

        # Bhattacharyya coefficient for Gaussians
        bhattacharyya = self._bhattacharyya_coefficient(volume_a, volume_b)

        # Overlap coefficient (Szymkiewicz-Simpson)
        overlap = self._overlap_coefficient(volume_a, volume_b)

        # Jaccard index
        jaccard = self._jaccard_index(volume_a, volume_b)

        # Curvature divergence
        curvature_div = self._curvature_divergence(volume_a, volume_b)

        # Subspace alignment
        subspace_align = self._subspace_alignment(volume_a, volume_b)

        return ConceptVolumeRelation(
            volume_a=volume_a,
            volume_b=volume_b,
            overlap_coefficient=overlap,
            jaccard_index=jaccard,
            bhattacharyya_coefficient=bhattacharyya,
            centroid_distance=centroid_distance,
            geodesic_centroid_distance=geodesic_centroid_distance,
            mahalanobis_distance_ab=mahal_ab,
            mahalanobis_distance_ba=mahal_ba,
            curvature_divergence=curvature_div,
            subspace_alignment=subspace_align,
        )

    def _compute_cka_relation(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> ConceptVolumeRelation:
        """CKA-based comparison with geodesic RBF kernels."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()

        # CKA requires raw activations - this should not happen if caller
        # follows the contract (store_raw_activations=True)
        if volume_a.raw_activations is None or volume_b.raw_activations is None:
            raise ValueError(
                f"CKA comparison requires raw_activations. "
                f"Volume {volume_a.concept_id} missing activations. "
                f"Enable store_raw_activations=True when creating volumes."
            )

        # Compute geodesic RBF CKA - this is dimension-agnostic AND curvature-aware
        # Uses geodesic distances in RBF kernel: K = exp(-d_geo²/2σ²)
        cka_result = compute_cka(
            volume_a.raw_activations,
            volume_b.raw_activations,
            backend,
        )
        cka_similarity = cka_result.cka if cka_result.is_valid else 0.0

        # CKA measures representational similarity:
        # - CKA ~ 1.0 = same representational structure = high overlap
        # - CKA ~ 0.0 = different representations = no overlap
        # - CKA in between = partial alignment

        # Map CKA to our metrics:
        # - overlap_coefficient: CKA directly measures overlap in representation space
        # - bhattacharyya: CKA approximates distribution overlap
        # - jaccard: CKA approximates concept intersection
        # - subspace_alignment: CKA measures alignment directly
        overlap = cka_similarity
        bhattacharyya = cka_similarity
        jaccard = cka_similarity
        subspace_align = cka_similarity

        # Distance is inverse of similarity: CKA=1→distance=0, CKA=0→distance=1
        # This is a "representational distance" not chord-length
        centroid_distance = 1.0 - cka_similarity
        geodesic_centroid_distance = centroid_distance

        # Mahalanobis doesn't apply across dimensions - use CKA-derived distance
        mahal_ab = centroid_distance
        mahal_ba = centroid_distance

        # Curvature: use average of local curvatures if available
        curvature_div = self._curvature_divergence(volume_a, volume_b)

        return ConceptVolumeRelation(
            volume_a=volume_a,
            volume_b=volume_b,
            overlap_coefficient=overlap,
            jaccard_index=jaccard,
            bhattacharyya_coefficient=bhattacharyya,
            centroid_distance=centroid_distance,
            geodesic_centroid_distance=geodesic_centroid_distance,
            mahalanobis_distance_ab=mahal_ab,
            mahalanobis_distance_ba=mahal_ba,
            curvature_divergence=curvature_div,
            subspace_alignment=subspace_align,
        )

    def _estimate_covariance(
        self,
        activations: "Array",
        centroid: "Array",
        local_curvature: LocalCurvature | None,
        metric_fn: Callable[["Array"], "Array"] | None,
        geo_result: GeodesicDistanceResult | None = None,
        geo_from_centroid: "Array | None" = None,
    ) -> "Array":
        """Estimate covariance in tangent space with optional metric correction."""
        backend = get_default_backend()
        shape = activations.shape
        d = int(shape[1])

        rg = RiemannianGeometry(backend)

        # Compute Riemannian covariance in tangent space at centroid
        # No fallback - if this fails, it's a bug we need to fix
        cov = rg.riemannian_covariance(
            activations,
            mean=centroid,
            geo_result=geo_result,
            geo_from_mean=geo_from_centroid,
        )

        # Regularize - always use machine epsilon
        reg_eps = regularization_epsilon(backend, cov)
        cov = cov + reg_eps * backend.eye(d)
        backend.eval(cov)

        # Metric correction if available
        if metric_fn is not None:
            cov = self._apply_metric_correction(cov, centroid, metric_fn)

        # Curvature correction — accounts for volume element distortion on curved manifold.
        # Applied AFTER metric correction so effective radius is geodesic, not coordinate.
        if local_curvature is not None:
            cov = self._apply_curvature_correction(cov, local_curvature)

        return cov

    def _apply_metric_correction(
        self,
        cov: "Array",
        centroid: "Array",
        metric_fn: Callable[["Array"], "Array"],
    ) -> "Array":
        """Apply metric-tensor correction to covariance."""
        backend = get_default_backend()
        try:
            metric = metric_fn(centroid)
            # Geodesic eigendecomposition (GPU-only)
            n_metric = int(metric.shape[0])
            eigenvalues, eigenvectors = power_iteration_eigh(backend, metric, k=n_metric)
            backend.eval(eigenvalues, eigenvectors)

            # Compute inverse sqrt of eigenvalues (use tiny_value to prevent sqrt(0))
            tiny = tiny_value(backend, eigenvalues)
            inv_sqrt_eigs = 1.0 / backend.sqrt(backend.maximum(eigenvalues, backend.array(tiny)))
            inv_sqrt_metric = backend.matmul(
                eigenvectors,
                backend.matmul(backend.diag(inv_sqrt_eigs), backend.transpose(eigenvectors)),
            )

            result = backend.matmul(inv_sqrt_metric, backend.matmul(cov, inv_sqrt_metric))
            backend.eval(result)
            return result
        except Exception:
            return cov  # Keep uncorrected covariance

    def _apply_curvature_correction(
        self,
        covariance: "Array",
        local_curvature: LocalCurvature,
    ) -> "Array":
        """Apply curvature-based correction when numerically meaningful."""
        backend = get_default_backend()
        K = local_curvature.mean_sectional
        eps = machine_epsilon(backend, covariance)

        # Curvature indistinguishable from zero
        if abs(K) < eps:
            return covariance

        # Compute effective radius from covariance trace
        trace_val = backend.trace(covariance)
        shape = covariance.shape
        r_arr = backend.sqrt(trace_val / int(shape[0]))
        backend.eval(r_arr)
        r = float(backend.to_scalar(r_arr))

        # Taylor remainder is (K*r²)²/36
        # Correction is meaningful only when remainder < sqrt(eps)
        # (sqrt(eps) is the precision threshold for relative error)
        kr_squared = abs(K) * r * r
        taylor_remainder = (kr_squared * kr_squared) / 36.0
        precision_threshold = sqrt_scalar(eps, backend)

        if taylor_remainder >= precision_threshold:
            # Correction would be numerically meaningless
            logger.debug(
                "Curvature correction skipped: Taylor remainder %.2e >= sqrt(eps)=%.2e",
                taylor_remainder,
                precision_threshold,
            )
            return covariance

        # Curvature correction factor (Taylor expansion is valid)
        # Based on comparison of volume elements in curved vs flat space
        # For sphere: dV_curved/dV_flat = (sin(r*sqrt(K))/(r*sqrt(K)))^(d-1)
        # For small curvature: ≈ 1 + K*r²/6 for positive K
        if K > 0:
            # Positive curvature - expand covariance
            correction = 1.0 + K * r * r / 6
        else:
            # Negative curvature - shrink covariance
            correction = 1.0 / (1.0 - K * r * r / 6)

        return covariance * correction

    def _compute_geodesic_radius(
        self,
        activations: "Array",
        centroid: "Array",
        geo_from_centroid: "Array | None" = None,
        geo_result: GeodesicDistanceResult | None = None,
    ) -> float:
        """Compute max geodesic radius from centroid."""
        backend = get_default_backend()
        shape = activations.shape
        int(shape[0])
        rg = RiemannianGeometry(backend)

        if geo_from_centroid is None:
            if geo_result is not None:
                geo_from_centroid = rg._geodesic_distances_from_query(
                    activations, centroid, geo_result=geo_result
                )
                backend.eval(geo_from_centroid)
            else:
                # Add centroid to points for distance computation
                centroid_2d = backend.reshape(centroid, (1, -1))
                points_with_centroid = backend.concatenate([centroid_2d, activations], axis=0)
                points_arr = _promote_precision(points_with_centroid, backend)
                backend.eval(points_arr)

                # Get geodesic distances from centroid (index 0) to all points
                geo_result = rg.geodesic_distances(points_arr, k_neighbors=None)
                geo_from_centroid = geo_result.distances[0, 1:]

        centroid_to_points = geo_from_centroid
        count = int(centroid_to_points.shape[0])
        if count == 0:
            return 0.0
        max_val = backend.max(centroid_to_points)
        backend.eval(max_val)
        return float(backend.to_scalar(max_val))

    def _bhattacharyya_coefficient(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Bhattacharyya coefficient between Gaussian volumes."""
        backend = get_default_backend()
        diff = volume_a.centroid - volume_b.centroid
        cov_avg = (volume_a.covariance + volume_b.covariance) / 2

        try:
            cov_avg_inv, _ = safe_inverse(backend, cov_avg, regularize=True)
            backend.eval(cov_avg_inv)

            # term1 = 0.125 * diff @ cov_avg_inv @ diff
            temp = backend.matmul(diff, cov_avg_inv)
            term1_arr = backend.matmul(temp, diff)
            backend.eval(term1_arr)
            term1 = 0.125 * float(backend.to_scalar(term1_arr))

            # Compute log det of cov_avg via eigenvalues (geodesic - GPU-only)
            n_cov_avg = int(cov_avg.shape[0])
            eigenvalues, _ = power_iteration_eigh(backend, cov_avg, k=n_cov_avg)
            backend.eval(eigenvalues)
            logdet_avg = _logdet_from_eigenvalues(backend, eigenvalues)
            if logdet_avg is None:
                return 0.0

            term2 = 0.5 * (
                logdet_avg - 0.5 * (volume_a.log_det_covariance + volume_b.log_det_covariance)
            )

            db = term1 + term2
            return exp_scalar(-db, backend)

        except Exception:
            return 0.0

    def _overlap_coefficient(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Estimate Szymkiewicz-Simpson overlap coefficient."""
        backend = get_default_backend()

        if volume_a.raw_activations is not None and volume_b.raw_activations is not None:
            a_in_b_mask = volume_b.contains_batch(volume_a.raw_activations)
            b_in_a_mask = volume_a.contains_batch(volume_b.raw_activations)

            a_in_b_sum = _mask_sum(a_in_b_mask, backend, dtype_source=volume_b.centroid)
            b_in_a_sum = _mask_sum(b_in_a_mask, backend, dtype_source=volume_a.centroid)
            backend.eval(a_in_b_sum, b_in_a_sum)

            n_a = max(1, int(volume_a.raw_activations.shape[0]))
            n_b = max(1, int(volume_b.raw_activations.shape[0]))
            frac_a = float(backend.to_scalar(a_in_b_sum)) / n_a
            frac_b = float(backend.to_scalar(b_in_a_sum)) / n_b
            return max(frac_a, frac_b)

        return self._bhattacharyya_coefficient(volume_a, volume_b)

    def _jaccard_index(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Estimate Jaccard index (intersection over union)."""
        # Use Bhattacharyya as proxy for intersection
        bc = self._bhattacharyya_coefficient(volume_a, volume_b)

        # For Gaussians: J ≈ BC / (2 - BC)
        # When BC is within machine epsilon of 1.0, the distributions are
        # effectively identical, so Jaccard = 1.0
        backend = get_default_backend()
        eps = machine_epsilon(backend, volume_a.centroid)
        if bc > 1.0 - eps:
            return 1.0
        return bc / (2 - bc)

    def _curvature_divergence(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute curvature mismatch between volumes."""
        if volume_a.local_curvature is None or volume_b.local_curvature is None:
            return 0.0

        K_a = volume_a.local_curvature.mean_sectional
        K_b = volume_b.local_curvature.mean_sectional

        # Normalized divergence (use division_epsilon for safe division)
        backend = get_default_backend()
        div_eps = division_epsilon(backend, volume_a.centroid)
        return abs(K_a - K_b) / (abs(K_a) + abs(K_b) + div_eps)

    def _subspace_alignment(
        self,
        volume_a: ConceptVolume,
        volume_b: ConceptVolume,
    ) -> float:
        """Compute alignment of principal subspaces."""
        backend = get_default_backend()

        dim_a = int(volume_a.covariance.shape[0])
        dim_b = int(volume_b.covariance.shape[0])

        if dim_a == dim_b:
            # Same dimension: use principal angles (geodesic - GPU-only)
            _, Va = volume_a._covariance_eigendecomposition()
            _, Vb = volume_b._covariance_eigendecomposition()

            # Compute singular values of Va^T @ Vb (geodesic - GPU-only)
            # These are cosines of principal angles
            M = backend.matmul(backend.transpose(Va), Vb)
            _, singular_values, _ = geodesic_svd(backend, M)
            backend.eval(singular_values)

            sq_vals = singular_values * singular_values
            alignment = backend.mean(sq_vals)
            backend.eval(alignment)
            result = float(backend.to_scalar(alignment))
        else:
            # Cross-architecture: use CKA on Gram matrices (dimension-agnostic)
            # Gram matrix K = X @ X^T has shape [n_samples, n_samples]
            # This is the SAME size regardless of feature dimension!
            from modelcypher.core.domain.geometry.cka import compute_cka_from_grams

            # Must use raw_activations to compute proper Gram matrices
            if volume_a.raw_activations is None or volume_b.raw_activations is None:
                logger.warning(
                    "Cross-architecture subspace alignment requires raw_activations. "
                    "Enable store_raw_activations=True when creating volumes."
                )
                return 0.0

            # Compute Gram matrices: K = X @ X^T → [n_samples, n_samples]
            # These have the SAME dimensions regardless of feature dimension
            gram_a = backend.matmul(
                volume_a.raw_activations,
                backend.transpose(volume_a.raw_activations),
            )
            gram_b = backend.matmul(
                volume_b.raw_activations,
                backend.transpose(volume_b.raw_activations),
            )
            backend.eval(gram_a, gram_b)

            result = compute_cka_from_grams(gram_a, gram_b, backend=backend)

        # Clamp to [0, 1] to handle floating point precision
        return max(0.0, min(1.0, result))


def batch_estimate_volumes(
    estimator: RiemannianDensityEstimator,
    concept_activations: dict[str, "Array"],
    metric_fn: Callable[["Array"], "Array"] | None = None,
) -> dict[str, ConceptVolume]:
    """Estimate volumes for multiple concepts."""
    volumes = {}
    for concept_id, activations in concept_activations.items():
        try:
            volumes[concept_id] = estimator.estimate_concept_volume(
                concept_id, activations, metric_fn
            )
        except Exception as e:
            logger.warning(f"Volume estimation failed for {concept_id}: {e}")
    return volumes


def compute_pairwise_relations(
    estimator: RiemannianDensityEstimator,
    volumes: dict[str, ConceptVolume],
) -> dict[tuple[str, str], ConceptVolumeRelation]:
    """Compute relations between all pairs of volumes."""
    relations = {}
    concept_ids = list(volumes.keys())

    for i, id_a in enumerate(concept_ids):
        for id_b in concept_ids[i + 1 :]:
            relation = estimator.compute_relation(volumes[id_a], volumes[id_b])
            relations[(id_a, id_b)] = relation

    return relations
