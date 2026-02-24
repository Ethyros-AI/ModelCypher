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

"""Core-set acquisition function using geodesic k-center selection.

Implements the k-center acquisition function from Sener & Savarese 2018
"Active Learning for Convolutional Neural Networks: A Core-Set Approach".

The k-center problem:
    Find subset S ⊂ X that minimizes: max_{x ∈ X} min_{s ∈ S} d(x, s)

For acquisition, we compute for each candidate:
    score(x) = min_{s ∈ corpus} d_geo(x, s)

Higher score = farther from corpus = more informative = should acquire.

This uses geodesic distances (not chord) because the activation manifold
is curved. Chord distances would underestimate distances between points
on different parts of the manifold connected by curved paths.

Implementation leverages RiemannianGeometry.farthest_point_sampling() which
already implements geodesic FPS (maximin design).
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
from modelcypher.core.domain.geometry.numerical_stability import (
    infinity_threshold,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class CoreSetConfig:
    """Configuration for core-set acquisition.

    All parameters are derived from geometry or machine precision.
    No heuristic thresholds.

    Attributes
    ----------
    k_neighbors : int | None
        k for geodesic k-NN graph. If None, uses minimum k for connectivity.
    refine_iterations : int
        Geodesic refinement iterations (0 = chord bootstrap only).
    """

    k_neighbors: int | None = None
    refine_iterations: int = 1


class CoreSetAcquisition:
    """Core-set acquisition using geodesic k-center.

    Scores candidates by their minimum geodesic distance to the existing corpus.
    Higher scores indicate candidates that would maximize coverage if selected.

    The coverage radius (max min-distance) is the primary quality metric:
    - Large radius = sparse coverage = more candidates needed
    - Small radius = dense coverage = corpus is comprehensive

    All thresholds derived from machine precision (sqrt(eps)).

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to the system-selected backend.
    config : CoreSetConfig, optional
        Configuration for geodesic computation.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        config: CoreSetConfig | None = None,
    ) -> None:
        """Initialize core-set acquisition."""
        self._backend = backend or get_default_backend()
        self._config = config or CoreSetConfig()

        # Derive precision threshold from machine epsilon
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

        # Geometry computation
        self._geometry = RiemannianGeometry(backend=self._backend)

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
        """Compute core-set acquisition scores for candidates.

        For each candidate, computes:
            score(x) = min_{s ∈ corpus} d_geo(x, s)

        Higher score = farther from corpus = should acquire.

        Parameters
        ----------
        candidates : Array
            Activation vectors for candidate probes [n_candidates, hidden_dim].
        corpus : Array
            Activation vectors for existing corpus [n_corpus, hidden_dim].
        backend : Backend, optional
            Compute backend. Uses instance backend if not specified.

        Returns
        -------
        AcquisitionResult
            Result with ranked candidates and coverage statistics.
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
            # No corpus - all candidates are equally valuable
            return uniform_acquisition_result(n_candidates)

        # Combine corpus and candidates for geodesic computation
        # Corpus points come first, then candidates
        combined = b.concatenate([corpus, candidates], axis=0)
        b.eval(combined)

        # Compute geodesic distances on combined point cloud
        geo_result = self._geometry.geodesic_distances(
            combined,
            k_neighbors=self._config.k_neighbors,
            refine_iterations=self._config.refine_iterations,
        )
        geo_dist = geo_result.distances
        b.eval(geo_dist)

        # Extract candidate-to-corpus distances
        # geo_dist[i, j] = distance from point i to point j
        # We want geo_dist[n_corpus:, :n_corpus] = candidate rows, corpus columns
        candidate_rows = b.take(
            geo_dist, b.arange(n_corpus, n_corpus + n_candidates), axis=0
        )
        corpus_cols = b.take(candidate_rows, b.arange(n_corpus), axis=1)
        b.eval(corpus_cols)

        # For each candidate, find minimum distance to any corpus point
        # Handle infinite distances (disconnected points)
        inf_thresh = infinity_threshold(b, corpus_cols)
        finite_mask = corpus_cols < inf_thresh

        # Replace infinite with large finite value for min computation
        max_finite = b.max(b.where(finite_mask, corpus_cols, b.zeros_like(corpus_cols)))
        b.eval(max_finite)
        max_val = float(b.to_scalar(max_finite))
        sentinel = max_val * 2.0 if max_val > self._sqrt_eps else float("inf")

        distances_safe = b.where(finite_mask, corpus_cols, b.full(corpus_cols.shape, sentinel))
        min_distances = b.min(distances_safe, axis=1)  # [n_candidates]
        b.eval(min_distances)

        # Compute coverage radius (for existing corpus)
        # This is the max min-distance from any point to the corpus
        coverage_radius = self._compute_coverage_radius(corpus, geo_dist[:n_corpus, :n_corpus])

        # Build acquisition scores
        scores: list[AcquisitionScore] = []
        min_dist_list = b.tolist(min_distances)

        for i, dist in enumerate(min_dist_list):
            dist_val = float(dist)
            # Normalize score by coverage radius for comparability
            if coverage_radius > self._sqrt_eps:
                normalized_score = dist_val / coverage_radius
            else:
                normalized_score = dist_val

            scores.append(
                AcquisitionScore(
                    probe_idx=i,
                    score=normalized_score,
                    coreset_contribution=dist_val,
                    coverage_contribution=0.0,  # Filled by ManifoldCoverageAcquisition
                    density_contribution=0.0,  # Filled by ManifoldCoverageAcquisition
                )
            )

        # Sort by score (descending - higher is better)
        scores.sort(key=lambda s: s.score, reverse=True)

        return AcquisitionResult(
            scores=scores,
            coverage_radius=coverage_radius,
            mean_local_id=0.0,  # Filled by composite acquisition
            sparse_fraction=0.0,  # Filled by composite acquisition
        )

    def _compute_coverage_radius(
        self,
        corpus: "Array",
        corpus_geo_dist: "Array",
    ) -> float:
        """Compute the k-center coverage radius of the corpus.

        Coverage radius = max_{x ∈ corpus} min_{s ∈ corpus, s ≠ x} d(x, s)

        This is the largest "hole" in the corpus coverage.

        Parameters
        ----------
        corpus : Array
            Corpus points [n_corpus, hidden_dim].
        corpus_geo_dist : Array
            Pairwise geodesic distances within corpus [n_corpus, n_corpus].

        Returns
        -------
        float
            Coverage radius (max min-distance).
        """
        b = self._backend
        n = int(corpus.shape[0])

        if n <= 1:
            return float("inf")  # Single point has infinite coverage radius

        # Exclude self-distances by setting diagonal to infinity
        inf_val = float(b.finfo(corpus_geo_dist.dtype).max)
        eye = b.eye(n)
        dist_no_self = b.where(eye > 0, b.full((n, n), inf_val), corpus_geo_dist)

        # Handle infinite distances (disconnected points)
        inf_thresh = infinity_threshold(b, dist_no_self)
        finite_mask = dist_no_self < inf_thresh

        # For each point, find minimum distance to any other point
        max_finite = b.max(b.where(finite_mask, dist_no_self, b.zeros_like(dist_no_self)))
        b.eval(max_finite)
        max_val = float(b.to_scalar(max_finite))
        sentinel = max_val * 2.0 if max_val > self._sqrt_eps else inf_val

        dist_safe = b.where(finite_mask, dist_no_self, b.full(dist_no_self.shape, sentinel))
        min_distances = b.min(dist_safe, axis=1)  # [n]

        # Coverage radius is the maximum of these min-distances
        max_min_arr = b.max(min_distances)
        b.eval(max_min_arr)
        coverage_radius = float(b.to_scalar(max_min_arr))

        # If all points are disconnected, return infinity
        if coverage_radius >= sentinel:
            return float("inf")

        return coverage_radius

    def select_batch(
        self,
        candidates: "Array",
        corpus: "Array",
        batch_size: int,
    ) -> list[int]:
        """Select a batch of candidates using greedy k-center.

        This implements greedy k-center selection, which provides a 2-approximation
        to the optimal k-center solution (Gonzalez 1985).

        The algorithm iteratively:
        1. Select the candidate farthest from current corpus + selected
        2. Add it to the selected set
        3. Repeat until batch_size reached

        Parameters
        ----------
        candidates : Array
            Candidate points [n_candidates, hidden_dim].
        corpus : Array
            Existing corpus points [n_corpus, hidden_dim].
        batch_size : int
            Number of candidates to select.

        Returns
        -------
        list[int]
            Indices of selected candidates (in selection order).
        """
        b = self._backend
        candidates = b.array(candidates)
        corpus = b.array(corpus)
        b.eval(candidates, corpus)

        n_candidates = int(candidates.shape[0])
        n_corpus = int(corpus.shape[0])

        if n_candidates == 0:
            return []

        batch_size = min(batch_size, n_candidates)

        if n_corpus == 0:
            # No corpus - use FPS on candidates directly
            fps_result = self._geometry.farthest_point_sampling(candidates, batch_size)
            return fps_result.selected_indices

        # Combine corpus and candidates
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

        # Initialize: min distance from each candidate to corpus
        candidate_rows = b.take(
            geo_dist, b.arange(n_corpus, n_corpus + n_candidates), axis=0
        )
        corpus_cols = b.take(candidate_rows, b.arange(n_corpus), axis=1)
        b.eval(corpus_cols)

        inf_thresh = infinity_threshold(b, corpus_cols)
        finite_mask = corpus_cols < inf_thresh
        max_finite = b.max(b.where(finite_mask, corpus_cols, b.zeros_like(corpus_cols)))
        b.eval(max_finite)
        max_val = float(b.to_scalar(max_finite))
        sentinel = max_val * 2.0 if max_val > self._sqrt_eps else float("inf")

        distances_safe = b.where(finite_mask, corpus_cols, b.full(corpus_cols.shape, sentinel))
        min_to_corpus = b.min(distances_safe, axis=1)  # [n_candidates]
        b.eval(min_to_corpus)

        # Greedy selection
        selected: list[int] = []
        selected_mask = b.zeros((n_candidates,))

        for _ in range(batch_size):
            # Mask already selected
            neg_inf = b.full((n_candidates,), float("-inf"))
            masked_dist = b.where(selected_mask > 0, neg_inf, min_to_corpus)
            b.eval(masked_dist)

            # Select farthest
            farthest_idx_arr = b.argmax(masked_dist)
            b.eval(farthest_idx_arr)
            farthest_idx = int(b.to_scalar(farthest_idx_arr))

            selected.append(farthest_idx)

            # Update mask
            idx_array = b.arange(n_candidates)
            one_hot = b.astype(idx_array == farthest_idx, min_to_corpus.dtype)
            selected_mask = b.minimum(selected_mask + one_hot, b.ones_like(selected_mask))

            # Update min distances (include new selected point)
            # Distance from each candidate to newly selected
            new_point_idx = n_corpus + farthest_idx
            new_dists = b.take(geo_dist, b.array([new_point_idx]), axis=0)
            new_dists = b.squeeze(new_dists, axis=0)
            candidate_dists = b.take(new_dists, b.arange(n_corpus, n_corpus + n_candidates), axis=0)
            b.eval(candidate_dists)

            # Update min: element-wise minimum
            min_to_corpus = b.minimum(min_to_corpus, candidate_dists)
            b.eval(min_to_corpus)

        return selected


def create_coreset_acquisition(
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    refine_iterations: int = 1,
) -> CoreSetAcquisition:
    """Create a core-set acquisition function.

    Parameters
    ----------
    backend : Backend, optional
        Compute backend.
    k_neighbors : int, optional
        k for geodesic graph. None = auto (minimum for connectivity).
    refine_iterations : int
        Geodesic refinement iterations.

    Returns
    -------
    CoreSetAcquisition
        Configured acquisition function.
    """
    config = CoreSetConfig(
        k_neighbors=k_neighbors,
        refine_iterations=refine_iterations,
    )
    return CoreSetAcquisition(backend=backend, config=config)
