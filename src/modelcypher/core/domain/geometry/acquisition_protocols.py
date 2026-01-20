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

"""Acquisition function protocols for manifold-aware probe selection.

This module defines the protocol and data structures for acquisition functions
used by the Curiosity Daemon to select optimal probes for manifold exploration.

Research basis:
    - Sener & Savarese 2018 "Active Learning for Convolutional Neural Networks:
      A Core-Set Approach" - k-center selection for maximum coverage
    - Huang et al. "Active Manifold Exploration" - HLLE-based summary points
    - Qiu & Miikkulainen 2024 "Semantic Density" - embedding space uncertainty

All implementations must:
    1. Accept candidate activations and existing corpus activations
    2. Return AcquisitionResult with scores for each candidate
    3. Use geodesic (not chord) distances for manifold geometry
    4. Derive all thresholds from sqrt(eps) or data

The acquisition score decomposition:
    score = coreset_contribution + coverage_contribution + density_contribution

Where:
    - coreset: k-center distance (global coverage)
    - coverage: directional gap alignment (local exploration)
    - density: local ID factor (structural complexity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class AcquisitionScore:
    """Score for a single candidate probe.

    Higher score = more informative = should acquire.
    All scores are raw geometric measurements, no interpretation.

    Attributes
    ----------
    probe_idx : int
        Index of the candidate in the input array.
    score : float
        Combined acquisition score (higher = better).
    coreset_contribution : float
        k-center distance contribution (global coverage).
        Measures minimum geodesic distance to existing corpus.
    coverage_contribution : float
        Directional coverage contribution (local exploration).
        Measures alignment with sparse directions at nearest corpus point.
    density_contribution : float
        Local intrinsic dimension factor (structural complexity).
        Higher local ID = more complex region = more valuable to sample.
    """

    probe_idx: int
    score: float
    coreset_contribution: float
    coverage_contribution: float
    density_contribution: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "probe_idx": self.probe_idx,
            "score": self.score,
            "coreset_contribution": self.coreset_contribution,
            "coverage_contribution": self.coverage_contribution,
            "density_contribution": self.density_contribution,
        }


@dataclass(frozen=True)
class AcquisitionResult:
    """Result of acquisition function evaluation.

    Contains ranked probes and aggregate manifold statistics.
    All values are raw measurements - no interpretation.

    Attributes
    ----------
    scores : list[AcquisitionScore]
        Scores for each candidate, sorted descending by score.
    coverage_radius : float
        Current k-center radius of the corpus.
        Maximum min-distance between any corpus point and the selected set.
    mean_local_id : float
        Mean local intrinsic dimension across corpus points.
    sparse_fraction : float
        Fraction of corpus points with local ID above modal ID + sqrt(eps).
        Indicates what fraction of the manifold is "sparse" (under-sampled).
    """

    scores: list[AcquisitionScore]
    coverage_radius: float
    mean_local_id: float
    sparse_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "scores": [s.to_dict() for s in self.scores],
            "coverage_radius": self.coverage_radius,
            "mean_local_id": self.mean_local_id,
            "sparse_fraction": self.sparse_fraction,
        }

    @property
    def top_score(self) -> AcquisitionScore | None:
        """Get the top-scoring candidate."""
        return self.scores[0] if self.scores else None

    @property
    def top_indices(self) -> list[int]:
        """Get indices of candidates sorted by score (descending)."""
        return [s.probe_idx for s in self.scores]

    def select_top_k(self, k: int) -> list[AcquisitionScore]:
        """Select top k candidates by score.

        Parameters
        ----------
        k : int
            Number of candidates to select.

        Returns
        -------
        list[AcquisitionScore]
            Top k candidates.
        """
        return self.scores[:k]


@runtime_checkable
class AcquisitionFunction(Protocol):
    """Protocol for manifold-aware acquisition functions.

    All implementations must:
    1. Accept candidate activations and existing corpus activations
    2. Return AcquisitionResult with scores for each candidate
    3. Use only geodesic (not chord) distances
    4. Derive all thresholds from sqrt(eps) or data

    The score method computes acquisition scores for candidate probes
    based on manifold geometry. Higher scores indicate candidates that
    would improve manifold coverage if selected.
    """

    def score(
        self,
        candidates: "Array",
        corpus: "Array",
        backend: "Backend | None" = None,
    ) -> AcquisitionResult:
        """Compute acquisition scores for candidate probes.

        Parameters
        ----------
        candidates : Array
            Activation vectors for candidate probes [n_candidates, hidden_dim].
        corpus : Array
            Activation vectors for existing corpus [n_corpus, hidden_dim].
        backend : Backend, optional
            Compute backend. Uses default if not specified.

        Returns
        -------
        AcquisitionResult
            Result with ranked candidates and manifold statistics.
        """
        ...


def empty_acquisition_result() -> AcquisitionResult:
    """Create an empty acquisition result.

    Returns
    -------
    AcquisitionResult
        Empty result with no candidates.
    """
    return AcquisitionResult(
        scores=[],
        coverage_radius=0.0,
        mean_local_id=0.0,
        sparse_fraction=0.0,
    )


def uniform_acquisition_result(n_candidates: int) -> AcquisitionResult:
    """Create a uniform acquisition result (all candidates equal).

    Used when corpus is empty or too small for meaningful comparison.

    Parameters
    ----------
    n_candidates : int
        Number of candidates.

    Returns
    -------
    AcquisitionResult
        Result with uniform scores of 1.0 for all candidates.
    """
    scores = [
        AcquisitionScore(
            probe_idx=i,
            score=1.0,
            coreset_contribution=1.0,
            coverage_contribution=0.0,
            density_contribution=0.0,
        )
        for i in range(n_candidates)
    ]
    return AcquisitionResult(
        scores=scores,
        coverage_radius=float("inf"),
        mean_local_id=0.0,
        sparse_fraction=1.0,
    )
