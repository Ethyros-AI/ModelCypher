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

"""Geodesic layer analyzer for compression.

Analyzes a layer's geodesic structure to estimate compressibility.
Computes:
- Euclidean rank (SVD-based)
- Geodesic rank (manifold intrinsic dimension)
- RMT signal rank (Marchenko-Pastur separation)
- Compressibility score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    svd_auto_rank,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeodesicLayerProfile:
    """Profile of a layer's geodesic structure.

    Attributes:
        euclidean_rank: SVD-based numerical rank.
        geodesic_rank: Manifold intrinsic dimension (from geodesic distances).
        rmt_signal_rank: Marchenko-Pastur signal components.
        null_space_dimension: Available capacity (d - signal_rank).
        frobenius_norm: Weight matrix Frobenius norm.
        top_singular_value: Largest singular value.
        top1_energy: Fraction of variance in top singular value.
        compressibility_score: Predicted compression success [0, 1].
        geodesic_distances: Pairwise geodesic distance matrix [n, n].
    """

    euclidean_rank: int
    geodesic_rank: int
    rmt_signal_rank: int
    null_space_dimension: int
    frobenius_norm: float
    top_singular_value: float
    top1_energy: float
    compressibility_score: float
    geodesic_distances: "Array"


class GeodesicLayerAnalyzer:
    """Analyzes a layer's geodesic structure for compression prediction.

    Uses existing infrastructure:
    - riemannian_core_geodesic.py for geodesic distances
    - rmt_signal_separation.py for signal/noise separation
    - numerical_stability.py for rank computation
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        activations: "Array",
        weight_matrix: "Array | None" = None,
    ) -> GeodesicLayerProfile:
        """Analyze layer activations for compression potential.

        Args:
            activations: Layer activations [n_samples, d_hidden].
            weight_matrix: Optional MLP weight matrix for Frobenius norm.

        Returns:
            GeodesicLayerProfile with all metrics.
        """
        b = self._backend

        activations = b.array(activations)
        b.eval(activations)

        n_samples = int(activations.shape[0])
        d_hidden = int(activations.shape[1])

        logger.info(
            "GEODESIC ANALYZER: Analyzing [%d, %d] activations",
            n_samples, d_hidden
        )

        # Step 1: Compute Euclidean rank via SVD
        euclidean_rank = self._compute_euclidean_rank(activations)

        # Step 2: Compute RMT signal rank
        rmt_signal_rank = self._compute_rmt_signal_rank(activations, n_samples, d_hidden)

        # Step 3: Compute geodesic distances and geodesic rank
        geodesic_distances, geodesic_rank = self._compute_geodesic_structure(activations)

        # Step 4: Compute weight metrics if provided
        if weight_matrix is not None:
            weight_matrix = b.array(weight_matrix)
            b.eval(weight_matrix)
            frobenius_norm, top_sv, top1_energy = self._compute_weight_metrics(weight_matrix)
        else:
            frobenius_norm = 0.0
            top_sv = 0.0
            top1_energy = 0.0

        # Step 5: Compute compressibility score
        # Based on exp2: frobenius_norm has r=-0.86 correlation with accuracy
        # Lower frobenius = higher compressibility
        compressibility_score = self._compute_compressibility_score(
            frobenius_norm, top1_energy, rmt_signal_rank, d_hidden
        )

        # Null space dimension
        null_space_dimension = d_hidden - rmt_signal_rank

        logger.info(
            "GEODESIC ANALYZER: euclidean_rank=%d, geodesic_rank=%d, "
            "rmt_signal_rank=%d, null_dim=%d, compressibility=%.3f",
            euclidean_rank, geodesic_rank, rmt_signal_rank,
            null_space_dimension, compressibility_score
        )

        return GeodesicLayerProfile(
            euclidean_rank=euclidean_rank,
            geodesic_rank=geodesic_rank,
            rmt_signal_rank=rmt_signal_rank,
            null_space_dimension=null_space_dimension,
            frobenius_norm=frobenius_norm,
            top_singular_value=top_sv,
            top1_energy=top1_energy,
            compressibility_score=compressibility_score,
            geodesic_distances=geodesic_distances,
        )

    def predict_compressibility(self, profile: GeodesicLayerProfile) -> float:
        """Predict compression success from profile.

        Returns:
            Float in [0, 1] where 1 = likely to compress losslessly.
        """
        return profile.compressibility_score

    def _compute_euclidean_rank(self, activations: "Array") -> int:
        """Compute numerical rank via SVD."""
        b = self._backend

        # Center activations
        mean = b.mean(activations, axis=0, keepdims=True)
        centered = activations - mean
        b.eval(centered)

        # SVD
        _, S, _ = b.svd(centered)
        b.eval(S)

        # Numerical rank
        rank = svd_auto_rank(S, b)
        return rank

    def _compute_rmt_signal_rank(
        self,
        activations: "Array",
        n_samples: int,
        d_hidden: int,
    ) -> int:
        """Compute signal rank via Marchenko-Pastur distribution."""
        from modelcypher.core.domain.geometry.rmt_signal_separation import (
            compute_signal_rank_from_singular_values,
        )

        b = self._backend

        # Center activations
        mean = b.mean(activations, axis=0, keepdims=True)
        centered = activations - mean
        b.eval(centered)

        # SVD for singular values
        _, S, _ = b.svd(centered)
        b.eval(S)

        # RMT signal/noise separation
        mp_result = compute_signal_rank_from_singular_values(
            S, n_samples=n_samples, n_features=d_hidden, backend=b
        )

        return max(1, int(mp_result.signal_rank))

    def _compute_geodesic_structure(
        self,
        activations: "Array",
    ) -> tuple["Array", int]:
        """Compute geodesic distance matrix and intrinsic dimension.

        Uses k-NN graph with Floyd-Warshall for geodesic distances.
        Estimates intrinsic dimension from geodesic distance scaling.
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )

        b = self._backend

        # Compute geodesic distances
        geodesic_distances = geodesic_distance_matrix(activations, backend=b)
        b.eval(geodesic_distances)

        # Estimate intrinsic dimension from geodesic distances
        # This tells us the "true" manifold dimension
        try:
            estimator = IntrinsicDimension(backend=b)
            id_result = estimator.compute(activations)
            geodesic_rank = max(1, int(id_result.intrinsic_dimension))
        except Exception as e:
            logger.warning("Intrinsic dimension estimation failed: %s", e)
            # Fallback: use ratio of geodesic spread to Euclidean spread
            geodesic_rank = self._estimate_geodesic_rank_fallback(
                activations, geodesic_distances
            )

        return geodesic_distances, geodesic_rank

    def _estimate_geodesic_rank_fallback(
        self,
        activations: "Array",
        geodesic_distances: "Array",
    ) -> int:
        """Fallback geodesic rank estimation from distance ratio."""
        b = self._backend

        # Compute Euclidean pairwise distances
        n = int(activations.shape[0])
        X_sq = b.sum(activations * activations, axis=1)
        XXT = b.matmul(activations, b.transpose(activations))
        D_sq = X_sq[:, None] + X_sq[None, :] - 2 * XXT
        D_sq = b.maximum(D_sq, b.zeros_like(D_sq))
        euclidean_distances = b.sqrt(D_sq)
        b.eval(euclidean_distances)

        # Compute ratio of geodesic to Euclidean median distances
        # Higher ratio = more curved manifold = lower effective dimension
        eps = float(division_epsilon(b, euclidean_distances))

        geo_median = b.median(geodesic_distances)
        euc_median = b.median(euclidean_distances)
        b.eval(geo_median, euc_median)

        geo_median_val = float(b.to_scalar(geo_median))
        euc_median_val = float(b.to_scalar(euc_median))

        ratio = geo_median_val / (euc_median_val + eps)

        # Heuristic: higher ratio means more curvature, lower rank
        # ratio ~1 means flat (Euclidean), ratio >> 1 means curved
        d_hidden = int(activations.shape[1])
        estimated_rank = int(d_hidden / max(ratio, 1.0))

        return max(1, min(estimated_rank, n - 1))

    def _compute_weight_metrics(
        self,
        weight_matrix: "Array",
    ) -> tuple[float, float, float]:
        """Compute weight matrix metrics.

        Returns:
            (frobenius_norm, top_singular_value, top1_energy)
        """
        b = self._backend

        # Frobenius norm
        frobenius_sq = b.sum(weight_matrix * weight_matrix)
        b.eval(frobenius_sq)
        frobenius_norm = float(b.to_scalar(b.sqrt(frobenius_sq)))

        # Singular value decomposition
        _, S, _ = b.svd(weight_matrix)
        b.eval(S)

        # Top singular value
        top_sv = float(b.to_scalar(S[0]))

        # Top-1 energy (fraction of variance in top singular value)
        S_sq = S * S
        total_var = b.sum(S_sq)
        b.eval(total_var)
        total_var_val = float(b.to_scalar(total_var))

        if total_var_val > 0:
            top1_energy = float(b.to_scalar(S_sq[0])) / total_var_val
        else:
            top1_energy = 0.0

        return frobenius_norm, top_sv, top1_energy

    def _compute_compressibility_score(
        self,
        frobenius_norm: float,
        top1_energy: float,
        rmt_signal_rank: int,
        d_hidden: int,
    ) -> float:
        """Compute compressibility score from metrics.

        Based on exp2 findings:
        - Lower frobenius_norm = higher compressibility (r=-0.86)
        - Lower top1_energy = higher compressibility (not a "gate" layer)
        - Lower signal_rank / d_hidden = more null space

        Returns score in [0, 1] where 1 = very compressible.
        """
        # Normalize frobenius norm to [0, 1] range.
        # Divisor 300.0 is empirical: observed range ~100-300 on 4096-dim layers.
        # TODO: Make configurable or derive from d_hidden (e.g., sqrt(d_hidden)).
        frob_score = max(0.0, 1.0 - frobenius_norm / 300.0)

        # Gate layers have high top-1 energy (>0.5). Invert so low energy = high score.
        gate_score = 1.0 - top1_energy

        # Null space ratio: larger null space = more compressible.
        null_ratio = 1.0 - rmt_signal_rank / d_hidden

        # Weighted combination.
        # Weights (0.5, 0.3, 0.2) are empirical, derived from regression on
        # compression success rate vs. these three metrics (internal experiment).
        # Not rigorously validated. Override for your use case.
        score = 0.5 * frob_score + 0.3 * gate_score + 0.2 * null_ratio

        return max(0.0, min(1.0, score))
