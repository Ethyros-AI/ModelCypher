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

"""
Geometry Metrics Service.

Exposes standalone geometry metrics as CLI/MCP-consumable operations.
These are the unique value propositions of ModelCypher - geometric
diagnostics that no other tool provides.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.geometry.geometry_metrics_cache import (
    CachedGWResult,
    CachedIDResult,
    CachedTopoResult,
    GeometryMetricsCache,
)
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    Config as GWConfig,
)
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
)
from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
    BootstrapConfiguration,
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)


@dataclass(frozen=True)
class GromovWassersteinResult:
    """Result of Gromov-Wasserstein distance computation."""

    distance: float
    normalized_distance: float
    compatibility_score: float
    converged: bool
    iterations: int
    coupling_shape: tuple[int, int]
    interpretation: str


@dataclass(frozen=True)
class IntrinsicDimensionResult:
    """Result of intrinsic dimension estimation."""

    dimension: float
    confidence_lower: float
    confidence_upper: float
    sample_count: int
    method: str
    interpretation: str


@dataclass(frozen=True)
class TopologicalFingerprintResult:
    """Result of topological fingerprint computation."""

    betti_0: int  # Connected components
    betti_1: int  # Loops/holes
    persistence_entropy: float
    total_persistence: float
    interpretation: str


class GeometryMetricsService:
    """
    Service for standalone geometry metrics.

    These metrics provide the unique geometric diagnostics that differentiate
    ModelCypher from other ML tools.

    Expensive computations are cached to ~/Library/Caches/ModelCypher/geometry_metrics/.
    """

    def __init__(self, cache: GeometryMetricsCache | None = None) -> None:
        """
        Initialize the service.

        Args:
            cache: Optional cache instance (uses shared singleton if None)
        """
        self._cache = cache or GeometryMetricsCache.shared()

    def compute_gromov_wasserstein(
        self,
        source_points: list[list[float]],
        target_points: list[list[float]],
        epsilon: float = 0.05,
        max_iterations: int = 50,
    ) -> GromovWassersteinResult:
        """
        Compute Gromov-Wasserstein distance between two point clouds.

        This measures the structural similarity of representation spaces
        without requiring point-to-point correspondence.

        Results are cached to avoid redundant O(n^3-n^4) computations.

        Args:
            source_points: First point cloud (N x D)
            target_points: Second point cloud (M x D)
            epsilon: Entropic regularization parameter
            max_iterations: Maximum outer iterations

        Returns:
            GromovWassersteinResult with distance and interpretation
        """
        # Check cache first
        cached = self._cache.get_gw_result(source_points, target_points, epsilon, max_iterations)
        if cached is not None:
            return self._gw_result_from_cached(cached)

        # Compute the expensive operation
        config = GWConfig(
            epsilon=epsilon,
            max_outer_iterations=max_iterations,
        )

        source_distances = GromovWassersteinDistance.compute_pairwise_distances(source_points)
        target_distances = GromovWassersteinDistance.compute_pairwise_distances(target_points)

        result = GromovWassersteinDistance.compute(
            source_distances=source_distances,
            target_distances=target_distances,
            config=config,
        )

        # Cache the result
        cached_result = CachedGWResult(
            distance=result.distance,
            normalized_distance=result.normalized_distance,
            compatibility_score=result.compatibility_score,
            converged=result.converged,
            iterations=result.iterations,
            coupling_shape=(len(source_points), len(target_points)),
        )
        self._cache.set_gw_result(
            source_points, target_points, epsilon, max_iterations, cached_result
        )

        return self._gw_result_from_cached(cached_result)

    def _gw_result_from_cached(self, cached: CachedGWResult) -> GromovWassersteinResult:
        """Convert cached GW result to full result with interpretation."""
        # Generate interpretation
        if cached.normalized_distance < 0.1:
            interpretation = (
                "Highly similar structure. Representation spaces are nearly isomorphic."
            )
        elif cached.normalized_distance < 0.3:
            interpretation = "Moderately similar. Core structure preserved with some divergence."
        elif cached.normalized_distance < 0.5:
            interpretation = (
                "Significant structural differences. Careful alignment needed before merging."
            )
        else:
            interpretation = "Very different structures. Merging may cause capability loss."

        if not cached.converged:
            interpretation += " Warning: solver did not converge; results may be approximate."

        return GromovWassersteinResult(
            distance=cached.distance,
            normalized_distance=cached.normalized_distance,
            compatibility_score=cached.compatibility_score,
            converged=cached.converged,
            iterations=cached.iterations,
            coupling_shape=cached.coupling_shape,
            interpretation=interpretation,
        )

    def estimate_intrinsic_dimension(
        self,
        points: list[list[float]],
        use_regression: bool = True,
        bootstrap_samples: int = 200,
    ) -> IntrinsicDimensionResult:
        """
        Estimate intrinsic dimension of a point cloud using TwoNN.

        This reveals the effective degrees of freedom in a representation
        space, which can indicate model capacity and generalization.

        Results are cached to avoid redundant bootstrap computations.

        Args:
            points: Point cloud (N x D)
            use_regression: Use regression method (more accurate)
            bootstrap_samples: Number of bootstrap iterations for confidence

        Returns:
            IntrinsicDimensionResult with dimension and confidence bounds
        """
        # Check cache first
        cached = self._cache.get_id_result(points, use_regression, bootstrap_samples)
        if cached is not None:
            return self._id_result_from_cached(cached, points)

        # Compute the expensive operation
        config = TwoNNConfiguration(
            use_regression=use_regression,
            bootstrap=BootstrapConfiguration(resamples=bootstrap_samples)
            if bootstrap_samples > 0
            else None,
        )

        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, config)

        # Extract confidence intervals if available
        if estimate.ci is not None:
            lower = estimate.ci.lower
            upper = estimate.ci.upper
        else:
            lower = estimate.intrinsic_dimension * 0.8
            upper = estimate.intrinsic_dimension * 1.2

        # Cache the result
        cached_result = CachedIDResult(
            dimension=estimate.intrinsic_dimension,
            confidence_lower=lower,
            confidence_upper=upper,
            sample_count=estimate.sample_count,
            use_regression=use_regression,
        )
        self._cache.set_id_result(points, use_regression, bootstrap_samples, cached_result)

        return self._id_result_from_cached(cached_result, points)

    def _id_result_from_cached(
        self, cached: CachedIDResult, points: list[list[float]]
    ) -> IntrinsicDimensionResult:
        """Convert cached ID result to full result with interpretation."""
        dimension = cached.dimension
        ambient_dim = len(points[0]) if points else 0
        ratio = dimension / ambient_dim if ambient_dim > 0 else 0

        if ratio < 0.1:
            interpretation = f"Low intrinsic dimension ({dimension:.1f}). Representations are highly structured/compressed."
        elif ratio < 0.3:
            interpretation = (
                f"Moderate intrinsic dimension ({dimension:.1f}). Balanced capacity utilization."
            )
        elif ratio < 0.6:
            interpretation = f"High intrinsic dimension ({dimension:.1f}). Rich representations with many degrees of freedom."
        else:
            interpretation = f"Very high intrinsic dimension ({dimension:.1f}). May indicate noise or overfitting."

        return IntrinsicDimensionResult(
            dimension=dimension,
            confidence_lower=cached.confidence_lower,
            confidence_upper=cached.confidence_upper,
            sample_count=cached.sample_count,
            method="TwoNN"
            + (" (regression)" if cached.use_regression else " (maximum likelihood)"),
            interpretation=interpretation,
        )

    def compute_topological_fingerprint(
        self,
        points: list[list[float]],
        max_dimension: int = 1,
        max_filtration: float | None = None,
        num_steps: int = 50,
    ) -> TopologicalFingerprintResult:
        """
        Compute topological fingerprint using persistent homology.

        This reveals the shape of the representation manifold, including
        connected components, loops, and voids.

        Results are cached to avoid redundant O(n^2 log n) computations.

        Args:
            points: Point cloud (N x D)
            max_dimension: Maximum homology dimension to compute
            max_filtration: Maximum filtration value for Rips complex
            num_steps: Number of filtration steps

        Returns:
            TopologicalFingerprintResult with Betti numbers and persistence
        """
        # Check cache first
        cached = self._cache.get_topo_result(points, max_dimension, max_filtration, num_steps)
        if cached is not None:
            return self._topo_result_from_cached(cached)

        # Compute the expensive operation
        fingerprint = TopologicalFingerprint.compute(
            points=points,
            max_dimension=max_dimension,
            max_filtration=max_filtration,
            num_steps=num_steps,
        )

        summary = fingerprint.summary
        betti = fingerprint.betti_numbers

        betti_0 = betti.get(0, summary.component_count)
        betti_1 = betti.get(1, summary.cycle_count)

        # Cache the result
        cached_result = CachedTopoResult(
            betti_0=betti_0,
            betti_1=betti_1,
            persistence_entropy=summary.persistence_entropy,
            total_persistence=summary.max_persistence,
        )
        self._cache.set_topo_result(points, max_dimension, max_filtration, num_steps, cached_result)

        return self._topo_result_from_cached(cached_result)

    def _topo_result_from_cached(self, cached: CachedTopoResult) -> TopologicalFingerprintResult:
        """Convert cached topological result to full result with interpretation."""
        betti_0 = cached.betti_0
        betti_1 = cached.betti_1

        # Generate interpretation
        if betti_0 == 1 and betti_1 == 0:
            interpretation = "Simple connected topology. Single coherent representation cluster."
        elif betti_0 > 1 and betti_1 == 0:
            interpretation = f"Fragmented topology ({betti_0} components). Multiple distinct representation clusters."
        elif betti_1 > 0:
            interpretation = f"Complex topology with {betti_1} loop(s). May indicate cyclic or periodic structure."
        else:
            interpretation = "Standard topology with moderate complexity."

        if cached.persistence_entropy > 0.8:
            interpretation += " High persistence entropy suggests stable features."
        elif cached.persistence_entropy < 0.3:
            interpretation += " Low persistence entropy indicates transient features."

        return TopologicalFingerprintResult(
            betti_0=betti_0,
            betti_1=betti_1,
            persistence_entropy=cached.persistence_entropy,
            total_persistence=cached.total_persistence,
            interpretation=interpretation,
        )

    @staticmethod
    def gromov_wasserstein_payload(result: GromovWassersteinResult) -> dict:
        """Convert GW result to CLI/MCP payload."""
        return {
            "distance": result.distance,
            "normalizedDistance": result.normalized_distance,
            "compatibilityScore": result.compatibility_score,
            "converged": result.converged,
            "iterations": result.iterations,
            "couplingShape": list(result.coupling_shape),
            "interpretation": result.interpretation,
        }

    @staticmethod
    def intrinsic_dimension_payload(result: IntrinsicDimensionResult) -> dict:
        """Convert ID result to CLI/MCP payload."""
        return {
            "dimension": result.dimension,
            "confidenceLower": result.confidence_lower,
            "confidenceUpper": result.confidence_upper,
            "sampleCount": result.sample_count,
            "method": result.method,
            "interpretation": result.interpretation,
        }

    @staticmethod
    def topological_fingerprint_payload(result: TopologicalFingerprintResult) -> dict:
        """Convert TF result to CLI/MCP payload."""
        return {
            "betti0": result.betti_0,
            "betti1": result.betti_1,
            "persistenceEntropy": result.persistence_entropy,
            "totalPersistence": result.total_persistence,
            "interpretation": result.interpretation,
        }
