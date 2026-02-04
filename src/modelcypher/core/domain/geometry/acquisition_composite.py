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

"""Composite acquisition function combining core-set and manifold coverage.

This module implements the composite acquisition score from the Curiosity Daemon
design, combining:
1. Core-set contribution (global coverage via k-center)
2. Manifold coverage contribution (local exploration via directional gaps)
3. Density contribution (structural complexity via local intrinsic dimension)

The weighting is GEOMETRY-DERIVED (not heuristic):

    weight = 1 / (1 + coverage_radius / mean_local_id)

This automatically balances global vs local exploration:
- Large coverage_radius (sparse corpus) → weight → 0 → prioritize global (coreset)
- Small coverage_radius (dense corpus) → weight → 1 → prioritize local (manifold)
- High mean_local_id (complex manifold) → smaller ratio → more local exploration

The composite score:
    score = (1 - weight) × coreset + weight × (coverage + density)

All thresholds derived from sqrt(eps) - machine precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.acquisition_coreset import (
    CoreSetAcquisition,
    CoreSetConfig,
)
from modelcypher.core.domain.geometry.acquisition_manifold import (
    ManifoldCoverageAcquisition,
    ManifoldCoverageConfig,
)
from modelcypher.core.domain.geometry.acquisition_protocols import (
    AcquisitionResult,
    AcquisitionScore,
    empty_acquisition_result,
    uniform_acquisition_result,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class CompositeAcquisitionConfig:
    """Configuration for composite acquisition.

    All parameters derived from geometry or data.

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
class CompositeWeights:
    """Geometry-derived weights for composite acquisition.

    Attributes
    ----------
    coreset_weight : float
        Weight for global coverage (k-center).
    coverage_weight : float
        Weight for local exploration (directional gaps).
    density_weight : float
        Weight for structural complexity (local ID).
    coverage_radius : float
        Current k-center coverage radius.
    mean_local_id : float
        Mean local intrinsic dimension.
    """

    coreset_weight: float
    coverage_weight: float
    density_weight: float
    coverage_radius: float
    mean_local_id: float


class CompositeAcquisition:
    """Composite acquisition combining core-set and manifold coverage.

    The composite score balances global coverage with local exploration
    using geometry-derived weights:

        weight = 1 / (1 + coverage_radius / mean_local_id)
        score = (1 - weight) × coreset + weight × (coverage + density)

    When coverage_radius >> mean_local_id:
        - weight → 0
        - Global coverage dominates
        - Probes selected to fill large holes

    When coverage_radius << mean_local_id:
        - weight → 1
        - Local exploration dominates
        - Probes selected to explore complex regions

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to the system-selected backend.
    config : CompositeAcquisitionConfig, optional
        Configuration for acquisition functions.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        config: CompositeAcquisitionConfig | None = None,
    ) -> None:
        """Initialize composite acquisition."""
        self._backend = backend or get_default_backend()
        self._config = config or CompositeAcquisitionConfig()

        # Derive precision threshold
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

        # Component acquisition functions
        coreset_config = CoreSetConfig(
            k_neighbors=self._config.k_neighbors,
            refine_iterations=self._config.refine_iterations,
        )
        self._coreset = CoreSetAcquisition(
            backend=self._backend,
            config=coreset_config,
        )

        manifold_config = ManifoldCoverageConfig(
            k_neighbors=self._config.k_neighbors,
            refine_iterations=self._config.refine_iterations,
        )
        self._manifold = ManifoldCoverageAcquisition(
            backend=self._backend,
            config=manifold_config,
        )

    @property
    def sqrt_eps(self) -> float:
        """Machine precision threshold."""
        return self._sqrt_eps

    def compute_weights(
        self,
        coverage_radius: float,
        mean_local_id: float,
    ) -> CompositeWeights:
        """Compute geometry-derived weights for composite scoring.

        The weight formula:
            w = 1 / (1 + coverage_radius / mean_local_id)

        Derivation:
        - When corpus is sparse (large radius), prioritize global coverage
        - When corpus is dense (small radius), prioritize local exploration
        - Complex manifolds (high ID) warrant more local exploration

        Parameters
        ----------
        coverage_radius : float
            Current k-center coverage radius.
        mean_local_id : float
            Mean local intrinsic dimension across corpus.

        Returns
        -------
        CompositeWeights
            Geometry-derived weights for scoring.
        """
        # Handle edge cases
        if coverage_radius <= self._sqrt_eps:
            # Corpus is very dense - pure local exploration
            return CompositeWeights(
                coreset_weight=0.0,
                coverage_weight=0.5,
                density_weight=0.5,
                coverage_radius=coverage_radius,
                mean_local_id=mean_local_id,
            )

        if mean_local_id <= self._sqrt_eps:
            # Manifold is flat - pure global coverage
            return CompositeWeights(
                coreset_weight=1.0,
                coverage_weight=0.0,
                density_weight=0.0,
                coverage_radius=coverage_radius,
                mean_local_id=mean_local_id,
            )

        # Geometry-derived weight
        ratio = coverage_radius / mean_local_id
        w = 1.0 / (1.0 + ratio)

        # Coreset weight = 1 - w (global coverage)
        # Local weights split between coverage and density
        coreset_weight = 1.0 - w
        local_weight = w

        # Split local weight evenly between coverage and density
        coverage_weight = local_weight / 2.0
        density_weight = local_weight / 2.0

        return CompositeWeights(
            coreset_weight=coreset_weight,
            coverage_weight=coverage_weight,
            density_weight=density_weight,
            coverage_radius=coverage_radius,
            mean_local_id=mean_local_id,
        )

    def score(
        self,
        candidates: "Array",
        corpus: "Array",
        backend: "Backend | None" = None,
    ) -> AcquisitionResult:
        """Compute composite acquisition scores.

        Combines core-set and manifold coverage using geometry-derived weights.

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
            Result with composite scores and manifold statistics.
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

        # Compute component scores
        coreset_result = self._coreset.score(candidates, corpus, backend=b)
        manifold_result = self._manifold.score(candidates, corpus, backend=b)

        # Get manifold statistics
        coverage_radius = coreset_result.coverage_radius
        mean_local_id = manifold_result.mean_local_id
        sparse_fraction = manifold_result.sparse_fraction

        # Compute weights
        weights = self.compute_weights(coverage_radius, mean_local_id)

        # Index coreset and manifold scores by probe_idx
        coreset_by_idx = {s.probe_idx: s for s in coreset_result.scores}
        manifold_by_idx = {s.probe_idx: s for s in manifold_result.scores}

        # Compute composite scores
        composite_scores: list[AcquisitionScore] = []

        for i in range(n_candidates):
            coreset_score = coreset_by_idx.get(i)
            manifold_score = manifold_by_idx.get(i)

            if coreset_score is None or manifold_score is None:
                # Fallback to uniform
                composite_scores.append(
                    AcquisitionScore(
                        probe_idx=i,
                        score=1.0,
                        coreset_contribution=1.0,
                        coverage_contribution=0.0,
                        density_contribution=0.0,
                    )
                )
                continue

            # Composite score formula
            # Use coreset_score.score (normalized by coverage_radius) not coreset_contribution (raw)
            # This ensures all contributions are on comparable scales
            composite = (
                weights.coreset_weight * coreset_score.score  # Already normalized
                + weights.coverage_weight * manifold_score.coverage_contribution
                + weights.density_weight * manifold_score.density_contribution
            )

            composite_scores.append(
                AcquisitionScore(
                    probe_idx=i,
                    score=composite,
                    coreset_contribution=coreset_score.coreset_contribution,
                    coverage_contribution=manifold_score.coverage_contribution,
                    density_contribution=manifold_score.density_contribution,
                )
            )

        # Sort by composite score (descending)
        composite_scores.sort(key=lambda s: s.score, reverse=True)

        return AcquisitionResult(
            scores=composite_scores,
            coverage_radius=coverage_radius,
            mean_local_id=mean_local_id,
            sparse_fraction=sparse_fraction,
        )

    def select_batch(
        self,
        candidates: "Array",
        corpus: "Array",
        batch_size: int,
    ) -> list[int]:
        """Select a batch of candidates using composite acquisition.

        Parameters
        ----------
        candidates : Array
            Candidate activation vectors [n_candidates, hidden_dim].
        corpus : Array
            Existing corpus activation vectors [n_corpus, hidden_dim].
        batch_size : int
            Number of candidates to select.

        Returns
        -------
        list[int]
            Indices of selected candidates (in selection order).
        """
        result = self.score(candidates, corpus)
        return result.top_indices[:batch_size]

    def get_weights(
        self,
        corpus: "Array",
    ) -> CompositeWeights:
        """Get current geometry-derived weights for a corpus.

        Useful for analysis and visualization.

        Parameters
        ----------
        corpus : Array
            Corpus activation vectors [n_corpus, hidden_dim].

        Returns
        -------
        CompositeWeights
            Current weights based on corpus geometry.
        """
        b = self._backend
        corpus = b.array(corpus)
        b.eval(corpus)

        n_corpus = int(corpus.shape[0])

        if n_corpus == 0:
            return CompositeWeights(
                coreset_weight=1.0,
                coverage_weight=0.0,
                density_weight=0.0,
                coverage_radius=float("inf"),
                mean_local_id=0.0,
            )

        # Compute coverage radius via coreset acquisition
        # We need dummy candidates (just pass corpus as candidates)
        coreset_result = self._coreset.score(corpus, corpus, backend=b)
        coverage_radius = coreset_result.coverage_radius

        # Compute mean local ID via manifold acquisition
        manifold_result = self._manifold.score(corpus, corpus, backend=b)
        mean_local_id = manifold_result.mean_local_id

        return self.compute_weights(coverage_radius, mean_local_id)


def create_composite_acquisition(
    backend: "Backend | None" = None,
    k_neighbors: int | None = None,
    refine_iterations: int = 1,
) -> CompositeAcquisition:
    """Create a composite acquisition function.

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
    CompositeAcquisition
        Configured composite acquisition function.
    """
    config = CompositeAcquisitionConfig(
        k_neighbors=k_neighbors,
        refine_iterations=refine_iterations,
    )
    return CompositeAcquisition(backend=backend, config=config)
