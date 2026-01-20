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

"""Manifold coverage acquisition using directional gaps and local intrinsic dimension.

This module implements acquisition scoring based on local manifold structure:
1. Directional coverage: identifies sparse tangent directions at corpus points
2. Local intrinsic dimension: prioritizes structurally complex regions

Research basis:
    - Huang et al. "Active Manifold Exploration" - HLLE-based summary points
    - Facco et al. 2017 "TwoNN" - intrinsic dimension estimation
    - Qiu & Miikkulainen 2024 "Semantic Density" - embedding space uncertainty

The key insight: regions with high local intrinsic dimension AND sparse
directional coverage represent the most valuable exploration targets.
These are regions where:
1. The manifold has rich local structure (high ID)
2. That structure is currently under-sampled (large angular gaps)

All thresholds derived from sqrt(eps) or modal statistics (no heuristics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.acquisition_protocols import (
    AcquisitionResult,
    AcquisitionScore,
    empty_acquisition_result,
    uniform_acquisition_result,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    LocalDimensionMap,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    infinity_threshold,
    machine_epsilon,
    pi_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class ManifoldCoverageConfig:
    """Configuration for manifold coverage acquisition.

    All parameters derived from geometry or data statistics.

    Attributes
    ----------
    k_neighbors : int | None
        k for geodesic graph. If None, uses minimum k for connectivity.
    refine_iterations : int
        Geodesic refinement iterations.
    """

    k_neighbors: int | None = None
    refine_iterations: int = 1


@dataclass(frozen=True)
class DirectionalGapScore:
    """Directional gap score for a corpus point.

    Attributes
    ----------
    point_idx : int
        Index of the corpus point.
    max_gap_angle : float
        Largest angular gap in tangent coverage [0, pi].
    sparse_direction : tuple[float, ...]
        Unit vector in the sparse direction.
    normalized_gap : float
        Gap angle normalized by pi (1.0 = half-sphere empty).
    """

    point_idx: int
    max_gap_angle: float
    sparse_direction: tuple[float, ...]
    normalized_gap: float


class ManifoldCoverageAcquisition:
    """Manifold coverage acquisition using directional gaps and local ID.

    Scores candidates based on:
    1. Alignment with sparse directions at nearby corpus points
    2. Local intrinsic dimension (complex regions = more valuable)

    The coverage contribution for a candidate is:
        coverage = alignment_with_sparse_direction × local_id_factor

    Where:
    - alignment_with_sparse_direction = cos(angle to sparse direction)
    - local_id_factor = local_id / modal_id (regions above modal are complex)

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to MLX.
    config : ManifoldCoverageConfig, optional
        Configuration for manifold analysis.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        config: ManifoldCoverageConfig | None = None,
    ) -> None:
        """Initialize manifold coverage acquisition."""
        self._backend = backend or get_default_backend()
        self._config = config or ManifoldCoverageConfig()

        # Derive precision threshold from machine epsilon
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

        # Geometry and dimension estimation
        self._geometry = RiemannianGeometry(backend=self._backend)
        self._id_estimator = IntrinsicDimension(backend=self._backend)

    @property
    def sqrt_eps(self) -> float:
        """Machine precision threshold."""
        return self._sqrt_eps

    def score(
        self,
        candidates: "Array",
        corpus: "Array",
        backend: "Backend | None" = None,
    ) -> AcquisitionResult:
        """Compute manifold coverage scores for candidates.

        For each candidate:
        1. Find nearest corpus point (geodesic)
        2. Compute alignment with sparse direction at that point
        3. Weight by local intrinsic dimension factor

        Parameters
        ----------
        candidates : Array
            Candidate activation vectors [n_candidates, hidden_dim].
        corpus : Array
            Existing corpus activation vectors [n_corpus, hidden_dim].
        backend : Backend, optional
            Compute backend. Uses instance backend if not specified.

        Returns
        -------
        AcquisitionResult
            Result with coverage-based scores.
        """
        b = backend or self._backend
        candidates = b.array(candidates)
        corpus = b.array(corpus)
        b.eval(candidates, corpus)

        n_candidates = int(candidates.shape[0])
        n_corpus = int(corpus.shape[0])

        # Edge cases
        if n_candidates == 0:
            return empty_acquisition_result()

        if n_corpus == 0:
            return uniform_acquisition_result(n_candidates)

        # Need enough points for directional analysis
        if n_corpus < 3:
            return uniform_acquisition_result(n_candidates)

        # Compute local dimension map for the corpus
        local_dim_map = self._id_estimator.local_dimension_map(corpus)
        modal_id = local_dim_map.modal_dimension
        mean_id = local_dim_map.mean_dimension

        # Compute directional gaps at each corpus point
        directional_gaps = self._compute_directional_gaps(corpus)

        # Combine corpus and candidates for geodesic nearest-neighbor lookup
        combined = b.concatenate([corpus, candidates], axis=0)
        b.eval(combined)

        # Compute geodesic distances
        geo_result = self._geometry.geodesic_distances(
            combined,
            k_neighbors=self._config.k_neighbors,
            refine_iterations=self._config.refine_iterations,
        )
        geo_dist = geo_result.distances
        b.eval(geo_dist)

        # For each candidate, find nearest corpus point
        candidate_to_corpus = b.take(
            geo_dist, b.arange(n_corpus, n_corpus + n_candidates), axis=0
        )
        candidate_to_corpus = b.take(candidate_to_corpus, b.arange(n_corpus), axis=1)
        b.eval(candidate_to_corpus)

        # Handle infinite distances
        inf_thresh = infinity_threshold(b, candidate_to_corpus)
        finite_mask = candidate_to_corpus < inf_thresh
        max_finite = b.max(b.where(finite_mask, candidate_to_corpus, b.zeros_like(candidate_to_corpus)))
        b.eval(max_finite)
        max_val = float(b.to_scalar(max_finite))
        sentinel = max_val * 2.0 if max_val > self._sqrt_eps else float("inf")

        dist_safe = b.where(finite_mask, candidate_to_corpus, b.full(candidate_to_corpus.shape, sentinel))
        nearest_idx_arr = b.argmin(dist_safe, axis=1)  # [n_candidates]
        b.eval(nearest_idx_arr)
        nearest_indices = b.tolist(nearest_idx_arr)

        # Compute scores
        scores: list[AcquisitionScore] = []
        local_dims = b.tolist(local_dim_map.dimensions)
        pi = pi_value(b)

        for i in range(n_candidates):
            nearest_corpus_idx = int(nearest_indices[i])

            # Get sparse direction and gap at nearest corpus point
            gap_info = directional_gaps.get(nearest_corpus_idx)
            if gap_info is None:
                # No directional info available
                scores.append(
                    AcquisitionScore(
                        probe_idx=i,
                        score=1.0,  # Default to uniform
                        coreset_contribution=0.0,
                        coverage_contribution=1.0,
                        density_contribution=0.0,
                    )
                )
                continue

            # Compute alignment with sparse direction
            candidate_vec = candidates[i]
            corpus_point = corpus[nearest_corpus_idx]
            tangent_vec = candidate_vec - corpus_point

            # Normalize tangent vector
            tangent_norm = b.sqrt(b.sum(tangent_vec * tangent_vec))
            b.eval(tangent_norm)
            norm_val = float(b.to_scalar(tangent_norm))

            if norm_val < self._sqrt_eps:
                # Candidate is on top of corpus point
                alignment = 0.0
            else:
                tangent_unit = tangent_vec / tangent_norm
                sparse_dir = b.array(gap_info.sparse_direction)

                # Cosine similarity (alignment)
                dot_prod = b.sum(tangent_unit * sparse_dir)
                b.eval(dot_prod)
                alignment = abs(float(b.to_scalar(dot_prod)))  # Absolute because direction is bidirectional

            # Local ID factor: how complex is the region?
            # Higher local ID relative to modal = more complex = more valuable
            local_id = local_dims[nearest_corpus_idx]
            if math.isnan(local_id) or modal_id < self._sqrt_eps:
                id_factor = 1.0
            else:
                id_factor = local_id / max(modal_id, self._sqrt_eps)

            # Coverage contribution = alignment × id_factor × gap_size
            # Gap size normalized by pi (1.0 = half hemisphere empty)
            gap_normalized = gap_info.max_gap_angle / pi if pi > 0 else 0.0
            coverage = alignment * id_factor * gap_normalized

            # Density contribution is the id_factor alone
            density = id_factor

            scores.append(
                AcquisitionScore(
                    probe_idx=i,
                    score=coverage,  # Coverage is the primary score
                    coreset_contribution=0.0,  # Filled by CoreSetAcquisition
                    coverage_contribution=coverage,
                    density_contribution=density,
                )
            )

        # Sort by score (descending)
        scores.sort(key=lambda s: s.score, reverse=True)

        # Compute sparse fraction: fraction of corpus with local ID above modal + sqrt(eps)
        sparse_count = sum(
            1 for d in local_dims
            if not math.isnan(d) and d > modal_id + self._sqrt_eps
        )
        sparse_fraction = sparse_count / n_corpus if n_corpus > 0 else 0.0

        return AcquisitionResult(
            scores=scores,
            coverage_radius=0.0,  # Filled by CoreSetAcquisition
            mean_local_id=mean_id,
            sparse_fraction=sparse_fraction,
        )

    def _compute_directional_gaps(
        self,
        corpus: "Array",
    ) -> dict[int, DirectionalGapScore]:
        """Compute directional gaps at each corpus point.

        Uses RiemannianGeometry.directional_coverage() to find the largest
        angular gap in the tangent sphere coverage at each point.

        Parameters
        ----------
        corpus : Array
            Corpus points [n_corpus, hidden_dim].

        Returns
        -------
        dict[int, DirectionalGapScore]
            Directional gap info for each corpus point.
        """
        b = self._backend
        n_corpus = int(corpus.shape[0])
        pi = pi_value(b)

        gaps: dict[int, DirectionalGapScore] = {}

        for i in range(n_corpus):
            try:
                coverage = self._geometry.directional_coverage(i, corpus)

                # Convert sparse direction to tuple
                sparse_dir = b.tolist(coverage.sparse_direction)
                sparse_tuple = tuple(float(x) for x in sparse_dir)

                # Normalized gap (0 = no gap, 1 = half hemisphere empty)
                normalized_gap = coverage.max_gap_angle / pi if pi > 0 else 0.0

                gaps[i] = DirectionalGapScore(
                    point_idx=i,
                    max_gap_angle=coverage.max_gap_angle,
                    sparse_direction=sparse_tuple,
                    normalized_gap=normalized_gap,
                )
            except Exception:
                # Skip points where directional coverage fails
                continue

        return gaps

    def get_sparse_directions(
        self,
        corpus: "Array",
    ) -> list[DirectionalGapScore]:
        """Get sparse directions for all corpus points.

        Useful for visualization and analysis.

        Parameters
        ----------
        corpus : Array
            Corpus points [n_corpus, hidden_dim].

        Returns
        -------
        list[DirectionalGapScore]
            Sparse direction info sorted by gap size (descending).
        """
        gaps = self._compute_directional_gaps(corpus)
        sorted_gaps = sorted(gaps.values(), key=lambda g: g.max_gap_angle, reverse=True)
        return sorted_gaps


def create_manifold_coverage_acquisition(
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    refine_iterations: int = 1,
) -> ManifoldCoverageAcquisition:
    """Create a manifold coverage acquisition function.

    Parameters
    ----------
    backend : Backend, optional
        Compute backend.
    k_neighbors : int, optional
        k for geodesic graph. None = auto.
    refine_iterations : int
        Geodesic refinement iterations.

    Returns
    -------
    ManifoldCoverageAcquisition
        Configured acquisition function.
    """
    config = ManifoldCoverageConfig(
        k_neighbors=k_neighbors,
        refine_iterations=refine_iterations,
    )
    return ManifoldCoverageAcquisition(backend=backend, config=config)
