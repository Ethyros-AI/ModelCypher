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
diagnostics that no other tool provides. Includes Gromov-Wasserstein,
intrinsic dimension, topological fingerprint, and spectral signature.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.geometry.geometry_metrics_cache import (
    CachedGWResult,
    CachedIDResult,
    CachedSpectralResult,
    CachedTopoResult,
    GeometryMetricsCache,
)
from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
    _MAX_OUTER_ITERATIONS,
    _SINKHORN_EPSILON,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    BootstrapConfiguration,
    IntrinsicDimension,
    TwoNNConfiguration,
)
from modelcypher.core.domain.geometry.spectral_signature import (
    SpectralSignature,
    SpectralSignatureConfig,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)


@dataclass(frozen=True)
class GromovWassersteinResult:
    """Result of Gromov-Wasserstein distance computation."""

    distance: float
    normalized_distance: float
    alignment_score: float
    converged: bool
    iterations: int
    coupling_shape: tuple[int, int]


@dataclass(frozen=True)
class IntrinsicDimensionResult:
    """Result of intrinsic dimension estimation."""

    dimension: float
    confidence_lower: float
    confidence_upper: float
    sample_count: int
    method: str


@dataclass(frozen=True)
class TopologicalFingerprintResult:
    """Result of topological fingerprint computation."""

    betti_0: int  # Connected components
    betti_1: int  # Loops/holes
    persistence_entropy: float
    total_persistence: float


@dataclass(frozen=True)
class SpectralSignatureResult:
    """Result of spectral signature computation."""

    eigenvalues: list[float]
    heat_trace: list[float]
    heat_times: list[float]
    spectral_entropy: float
    algebraic_connectivity: float
    component_count: int
    node_count: int
    edge_count: int
    k_neighbors: int
    kernel_bandwidth: float
    normalized_laplacian: bool
    connected: bool


@dataclass(frozen=True)
class DimensionConstraintInvarianceResult:
    """Result of dimension constraint invariance measurement."""

    base_dimension: int
    padded_dimension: int
    sample_count: int
    k_neighbors: int | None
    gram_cka: float
    geodesic_mean_abs_diff: float
    geodesic_max_abs_diff: float
    spectral_eigen_mean_abs_diff: float
    spectral_eigen_max_abs_diff: float
    spectral_entropy_base: float
    spectral_entropy_padded: float
    heat_trace_base: list[float]
    heat_trace_padded: list[float]
    heat_times: list[float]
    betti_numbers_base: dict[int, int]
    betti_numbers_padded: dict[int, int]
    component_count_base: int
    component_count_padded: int
    cycle_count_base: int
    cycle_count_padded: int
    persistence_entropy_base: float
    persistence_entropy_padded: float
    max_persistence_base: float
    max_persistence_padded: float




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
    ) -> GromovWassersteinResult:
        """
        Compute Gromov-Wasserstein distance between two point clouds.

        This measures the structural similarity of representation spaces
        without requiring point-to-point correspondence.

        Results are cached to avoid redundant O(n^3-n^4) computations.

        Entropic regularization and convergence parameters are determined
        by the domain layer defaults, which use adaptive convergence with
        thresholds derived from dtype precision.

        Args:
            source_points: First point cloud (N x D)
            target_points: Second point cloud (M x D)

        Returns:
            GromovWassersteinResult with distance metrics
        """
        # Check cache first (use module constants for cache key)
        cached = self._cache.get_gw_result(
            source_points,
            target_points,
            _SINKHORN_EPSILON,
            _MAX_OUTER_ITERATIONS,
        )
        if cached is not None:
            return self._gw_result_from_cached(cached)

        # Compute the expensive operation
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        gw = GromovWassersteinDistance(backend=backend)

        pts_source = backend.array(source_points)
        pts_target = backend.array(target_points)
        source_distances = gw.compute_pairwise_distances(pts_source)
        target_distances = gw.compute_pairwise_distances(pts_target)

        result = gw.compute(
            source_distances=source_distances,
            target_distances=target_distances,
        )

        # Cache the result
        cached_result = CachedGWResult(
            distance=result.distance,
            normalized_distance=result.normalized_distance,
            alignment_score=result.alignment_score,
            converged=result.converged,
            iterations=result.iterations,
            coupling_shape=(len(source_points), len(target_points)),
        )
        self._cache.set_gw_result(
            source_points,
            target_points,
            _SINKHORN_EPSILON,
            _MAX_OUTER_ITERATIONS,
            cached_result,
        )

        return self._gw_result_from_cached(cached_result)

    def _gw_result_from_cached(self, cached: CachedGWResult) -> GromovWassersteinResult:
        """Convert cached GW result to full result."""
        return GromovWassersteinResult(
            distance=cached.distance,
            normalized_distance=cached.normalized_distance,
            alignment_score=cached.alignment_score,
            converged=cached.converged,
            iterations=cached.iterations,
            coupling_shape=cached.coupling_shape,
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
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        config = TwoNNConfiguration(use_regression=use_regression)
        bootstrap_config = (
            BootstrapConfiguration(resamples=bootstrap_samples)
            if bootstrap_samples > 0
            else None
        )

        computer = IntrinsicDimension(backend)
        pts = backend.array(points)
        estimate = computer.compute(pts, config, bootstrap=bootstrap_config)

        # Extract confidence intervals if available
        if estimate.ci is None:
            raise ValueError(
                "Intrinsic dimension confidence intervals require bootstrap_samples > 0."
            )
        lower = estimate.ci.lower
        upper = estimate.ci.upper

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
        """Convert cached ID result to full result."""
        dimension = cached.dimension
        return IntrinsicDimensionResult(
            dimension=dimension,
            confidence_lower=cached.confidence_lower,
            confidence_upper=cached.confidence_upper,
            sample_count=cached.sample_count,
            method="TwoNN"
            + (" (regression)" if cached.use_regression else " (maximum likelihood)"),
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
        """Convert cached topological result to full result."""
        betti_0 = cached.betti_0
        betti_1 = cached.betti_1
        return TopologicalFingerprintResult(
            betti_0=betti_0,
            betti_1=betti_1,
            persistence_entropy=cached.persistence_entropy,
            total_persistence=cached.total_persistence,
        )

    def compute_spectral_signature(
        self,
        points: list[list[float]],
        k_neighbors: int | None = None,
        kernel_bandwidth: float | None = None,
        normalized_laplacian: bool = True,
        heat_times: list[float] | None = None,
    ) -> SpectralSignatureResult:
        """
        Compute geodesic spectral signature from a point cloud.

        Builds a k-NN geodesic graph, constructs a Laplacian, and reports
        raw spectral metrics (eigenvalues, heat trace, entropy).
        """
        times = (
            tuple(heat_times)
            if heat_times is not None
            else SpectralSignatureConfig().heat_trace_times
        )

        cached = self._cache.get_spectral_result(
            points, k_neighbors, kernel_bandwidth, normalized_laplacian, times
        )
        if cached is not None:
            return self._spectral_result_from_cached(cached)

        config = SpectralSignatureConfig(
            k_neighbors=k_neighbors,
            kernel_bandwidth=kernel_bandwidth,
            normalized_laplacian=normalized_laplacian,
            heat_trace_times=times,
        )

        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        computer = SpectralSignature(backend=backend)
        signature = computer.compute(points=points, config=config)

        cached_result = CachedSpectralResult(
            eigenvalues=signature.eigenvalues,
            heat_trace=signature.heat_trace,
            heat_times=signature.heat_times,
            spectral_entropy=signature.spectral_entropy,
            algebraic_connectivity=signature.algebraic_connectivity,
            component_count=signature.component_count,
            node_count=signature.node_count,
            edge_count=signature.edge_count,
            k_neighbors=signature.k_neighbors,
            kernel_bandwidth=signature.kernel_bandwidth,
            normalized_laplacian=signature.normalized_laplacian,
            connected=signature.connected,
        )
        self._cache.set_spectral_result(
            points, k_neighbors, kernel_bandwidth, normalized_laplacian, times, cached_result
        )

        return self._spectral_result_from_cached(cached_result)

    def _spectral_result_from_cached(
        self, cached: CachedSpectralResult
    ) -> SpectralSignatureResult:
        """Convert cached spectral result to full result."""
        return SpectralSignatureResult(
            eigenvalues=cached.eigenvalues,
            heat_trace=cached.heat_trace,
            heat_times=cached.heat_times,
            spectral_entropy=cached.spectral_entropy,
            algebraic_connectivity=cached.algebraic_connectivity,
            component_count=cached.component_count,
            node_count=cached.node_count,
            edge_count=cached.edge_count,
            k_neighbors=cached.k_neighbors,
            kernel_bandwidth=cached.kernel_bandwidth,
            normalized_laplacian=cached.normalized_laplacian,
            connected=cached.connected,
        )

    def compute_dimension_constraint_invariance(
        self,
        points: list[list[float]],
        padded_dimension: int,
        k_neighbors: int | None = None,
        heat_times: list[float] | None = None,
    ) -> DimensionConstraintInvarianceResult:
        """Measure invariance under zero-padding dimension constraints."""
        if not points:
            raise ValueError("points must be non-empty")

        base_dim = len(points[0])
        if padded_dimension < base_dim:
            raise ValueError("padded_dimension must be >= base dimension")

        sample_count = len(points)
        padded_points = [
            row + [0.0] * (padded_dimension - base_dim) for row in points
        ]

        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        base_arr = backend.array(points)
        padded_arr = backend.array(padded_points)
        backend.eval(base_arr, padded_arr)

        gram_base = backend.matmul(base_arr, backend.transpose(base_arr))
        gram_padded = backend.matmul(padded_arr, backend.transpose(padded_arr))
        backend.eval(gram_base, gram_padded)
        gram_cka = compute_cka_from_grams(gram_base, gram_padded, backend=backend)

        geometry = RiemannianGeometry(backend)
        geo_base = geometry.geodesic_distances(points, k_neighbors=k_neighbors)
        geo_padded = geometry.geodesic_distances(padded_points, k_neighbors=k_neighbors)

        geo_diff = backend.abs(geo_base.distances - geo_padded.distances)
        geo_mean = backend.mean(geo_diff)
        geo_max = backend.max(geo_diff)
        backend.eval(geo_mean, geo_max)
        geodesic_mean_abs_diff = float(backend.to_scalar(geo_mean))
        geodesic_max_abs_diff = float(backend.to_scalar(geo_max))

        times = (
            tuple(heat_times)
            if heat_times is not None
            else SpectralSignatureConfig().heat_trace_times
        )
        spectral_config = SpectralSignatureConfig(
            k_neighbors=k_neighbors,
            normalized_laplacian=True,
            heat_trace_times=times,
        )
        spectral = SpectralSignature(backend)
        sig_base = spectral.compute(points=points, config=spectral_config)
        sig_padded = spectral.compute(points=padded_points, config=spectral_config)

        eigen_diffs = [
            abs(a - b) for a, b in zip(sig_base.eigenvalues, sig_padded.eigenvalues)
        ]
        spectral_eigen_mean_abs_diff = (
            sum(eigen_diffs) / len(eigen_diffs) if eigen_diffs else 0.0
        )
        spectral_eigen_max_abs_diff = max(eigen_diffs) if eigen_diffs else 0.0

        fp_base = TopologicalFingerprint.compute(points, max_dimension=1)
        fp_padded = TopologicalFingerprint.compute(padded_points, max_dimension=1)

        return DimensionConstraintInvarianceResult(
            base_dimension=base_dim,
            padded_dimension=padded_dimension,
            sample_count=sample_count,
            k_neighbors=k_neighbors,
            gram_cka=float(gram_cka),
            geodesic_mean_abs_diff=geodesic_mean_abs_diff,
            geodesic_max_abs_diff=geodesic_max_abs_diff,
            spectral_eigen_mean_abs_diff=spectral_eigen_mean_abs_diff,
            spectral_eigen_max_abs_diff=spectral_eigen_max_abs_diff,
            spectral_entropy_base=sig_base.spectral_entropy,
            spectral_entropy_padded=sig_padded.spectral_entropy,
            heat_trace_base=sig_base.heat_trace,
            heat_trace_padded=sig_padded.heat_trace,
            heat_times=list(sig_base.heat_times),
            betti_numbers_base=fp_base.betti_numbers,
            betti_numbers_padded=fp_padded.betti_numbers,
            component_count_base=fp_base.summary.component_count,
            component_count_padded=fp_padded.summary.component_count,
            cycle_count_base=fp_base.summary.cycle_count,
            cycle_count_padded=fp_padded.summary.cycle_count,
            persistence_entropy_base=fp_base.summary.persistence_entropy,
            persistence_entropy_padded=fp_padded.summary.persistence_entropy,
            max_persistence_base=fp_base.summary.max_persistence,
            max_persistence_padded=fp_padded.summary.max_persistence,
        )

    @staticmethod
    def gromov_wasserstein_payload(result: GromovWassersteinResult) -> dict:
        """Convert GW result to CLI/MCP payload."""
        return {
            "distance": result.distance,
            "normalizedDistance": result.normalized_distance,
            "alignmentScore": result.alignment_score,
            "converged": result.converged,
            "iterations": result.iterations,
            "couplingShape": list(result.coupling_shape),
        }

    @staticmethod
    def intrinsic_dimension_payload(result: IntrinsicDimensionResult) -> dict:
        """Convert ID result to CLI/MCP payload."""
        return {
            "intrinsicDimension": result.dimension,
            "confidenceLower": result.confidence_lower,
            "confidenceUpper": result.confidence_upper,
            "sampleCount": result.sample_count,
            "method": result.method,
        }

    @staticmethod
    def topological_fingerprint_payload(result: TopologicalFingerprintResult) -> dict:
        """Convert TF result to CLI/MCP payload."""
        return {
            "betti0": result.betti_0,
            "betti1": result.betti_1,
            "persistenceEntropy": result.persistence_entropy,
            "totalPersistence": result.total_persistence,
        }

    @staticmethod
    def spectral_signature_payload(
        result: SpectralSignatureResult,
        max_eigenvalues: int | None = None,
    ) -> dict:
        """Convert spectral signature result to CLI/MCP payload."""
        eigenvalues = result.eigenvalues
        truncated = False
        if max_eigenvalues is not None and max_eigenvalues >= 0:
            if len(eigenvalues) > max_eigenvalues:
                eigenvalues = eigenvalues[:max_eigenvalues]
                truncated = True
        return {
            "eigenvalues": eigenvalues,
            "eigenvalueCount": len(result.eigenvalues),
            "eigenvaluesTruncated": truncated,
            "heatTrace": result.heat_trace,
            "heatTimes": result.heat_times,
            "spectralEntropy": result.spectral_entropy,
            "algebraicConnectivity": result.algebraic_connectivity,
            "componentCount": result.component_count,
            "nodeCount": result.node_count,
            "edgeCount": result.edge_count,
            "kNeighbors": result.k_neighbors,
            "kernelBandwidth": result.kernel_bandwidth,
            "normalizedLaplacian": result.normalized_laplacian,
            "connected": result.connected,
        }

    @staticmethod
    def dimension_constraint_invariance_payload(
        result: DimensionConstraintInvarianceResult,
    ) -> dict:
        """Convert dimension constraint invariance to CLI/MCP payload."""
        return {
            "baseDimension": result.base_dimension,
            "paddedDimension": result.padded_dimension,
            "sampleCount": result.sample_count,
            "kNeighbors": result.k_neighbors,
            "gramCka": result.gram_cka,
            "geodesicDiff": {
                "meanAbs": result.geodesic_mean_abs_diff,
                "maxAbs": result.geodesic_max_abs_diff,
            },
            "spectral": {
                "eigenMeanAbsDiff": result.spectral_eigen_mean_abs_diff,
                "eigenMaxAbsDiff": result.spectral_eigen_max_abs_diff,
                "spectralEntropyBase": result.spectral_entropy_base,
                "spectralEntropyPadded": result.spectral_entropy_padded,
                "heatTraceBase": result.heat_trace_base,
                "heatTracePadded": result.heat_trace_padded,
                "heatTimes": result.heat_times,
            },
            "topology": {
                "bettiNumbersBase": result.betti_numbers_base,
                "bettiNumbersPadded": result.betti_numbers_padded,
                "componentCountBase": result.component_count_base,
                "componentCountPadded": result.component_count_padded,
                "cycleCountBase": result.cycle_count_base,
                "cycleCountPadded": result.cycle_count_padded,
                "persistenceEntropyBase": result.persistence_entropy_base,
                "persistenceEntropyPadded": result.persistence_entropy_padded,
                "maxPersistenceBase": result.max_persistence_base,
                "maxPersistencePadded": result.max_persistence_padded,
            },
        }
