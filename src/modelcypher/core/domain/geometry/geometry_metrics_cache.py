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

"""Cache for expensive geometry metrics computations.

Provides caching for Gromov-Wasserstein distance, intrinsic dimension,
topological fingerprint, and spectral signature computations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from modelcypher.core.domain.cache import TwoLevelCache, content_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedGWResult:
    """Cached Gromov-Wasserstein result."""

    distance: float
    normalized_distance: float
    converged: bool
    iterations: int
    coupling_shape: tuple[int, int]


@dataclass(frozen=True)
class CachedIDResult:
    """Cached intrinsic dimension result."""

    dimension: float
    confidence_lower: float
    confidence_upper: float
    sample_count: int
    use_regression: bool


@dataclass(frozen=True)
class CachedTopoResult:
    """Cached topological fingerprint result."""

    betti_0: int
    betti_1: int
    persistence_entropy: float
    total_persistence: float


@dataclass(frozen=True)
class CachedSpectralResult:
    """Cached spectral signature result."""

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


class GeometryMetricsCache:
    """
    Two-level cache for expensive geometry metric computations.

    Provides caching for:
    - Gromov-Wasserstein distance (O(n^3-n^4))
    - Intrinsic dimension estimation (O(n log n) with bootstrap)
    - Topological fingerprints (O(n^2 log n))
    - Spectral signatures (O(n^3) for geodesic distances)

    Cache is stored in ~/Library/Caches/ModelCypher/geometry_metrics/
    """

    CACHE_VERSION = 2
    _shared_instance: "GeometryMetricsCache" | None = None

    @classmethod
    def shared(cls) -> "GeometryMetricsCache":
        """Get the shared singleton instance."""
        if cls._shared_instance is None:
            cls._shared_instance = GeometryMetricsCache()
        return cls._shared_instance

    def __init__(self, cache_directory: Path | None = None) -> None:
        """
        Initialize the cache.

        Args:
            cache_directory: Override default cache directory
        """
        base = cache_directory or (
            Path.home() / "Library" / "Caches" / "ModelCypher" / "geometry_metrics"
        )

        self._gw_cache: TwoLevelCache[CachedGWResult] = TwoLevelCache(
            cache_directory=base / "gromov_wasserstein",
            serializer=self._serialize_gw,
            deserializer=self._deserialize_gw,
            memory_limit=100,
            disk_ttl_seconds=7 * 24 * 60 * 60,  # 7 days
            cache_version=self.CACHE_VERSION,
        )

        self._id_cache: TwoLevelCache[CachedIDResult] = TwoLevelCache(
            cache_directory=base / "intrinsic_dimension",
            serializer=self._serialize_id,
            deserializer=self._deserialize_id,
            memory_limit=100,
            disk_ttl_seconds=7 * 24 * 60 * 60,  # 7 days
            cache_version=self.CACHE_VERSION,
        )

        self._topo_cache: TwoLevelCache[CachedTopoResult] = TwoLevelCache(
            cache_directory=base / "topological",
            serializer=self._serialize_topo,
            deserializer=self._deserialize_topo,
            memory_limit=100,
            disk_ttl_seconds=7 * 24 * 60 * 60,  # 7 days
            cache_version=self.CACHE_VERSION,
        )

        self._spectral_cache: TwoLevelCache[CachedSpectralResult] = TwoLevelCache(
            cache_directory=base / "spectral_signature",
            serializer=self._serialize_spectral,
            deserializer=self._deserialize_spectral,
            memory_limit=100,
            disk_ttl_seconds=7 * 24 * 60 * 60,  # 7 days
            cache_version=self.CACHE_VERSION,
        )

    # --- Gromov-Wasserstein ---

    def get_gw_result(
        self,
        source_points: list[list[float]],
        target_points: list[list[float]],
    ) -> CachedGWResult | None:
        """
        Get cached Gromov-Wasserstein result.

        Args:
            source_points: Source point cloud
            target_points: Target point cloud

        Returns:
            Cached result or None if not found
        """
        key = self._make_gw_key(source_points, target_points)
        return self._gw_cache.get(key)

    def set_gw_result(
        self,
        source_points: list[list[float]],
        target_points: list[list[float]],
        result: CachedGWResult,
    ) -> None:
        """
        Cache Gromov-Wasserstein result.

        Args:
            source_points: Source point cloud
            target_points: Target point cloud
            result: Result to cache
        """
        key = self._make_gw_key(source_points, target_points)
        self._gw_cache.set(key, result)

    def _make_gw_key(
        self,
        source_points: list[list[float]],
        target_points: list[list[float]],
    ) -> str:
        """Create cache key for GW computation.

        Note: GW algorithm parameters are derived from numerical precision,
        so only the point data determines the cache key.
        """
        return content_hash(
            {
                "source": sorted([tuple(p) for p in source_points]),
                "target": sorted([tuple(p) for p in target_points]),
            }
        )

    @staticmethod
    def _serialize_gw(result: CachedGWResult) -> dict:
        return {
            "distance": result.distance,
            "normalized_distance": result.normalized_distance,
            "converged": result.converged,
            "iterations": result.iterations,
            "coupling_shape": list(result.coupling_shape),
        }

    @staticmethod
    def _deserialize_gw(data: dict) -> CachedGWResult:
        return CachedGWResult(
            distance=float(data["distance"]),
            normalized_distance=float(data["normalized_distance"]),
            converged=bool(data["converged"]),
            iterations=int(data["iterations"]),
            coupling_shape=tuple(data["coupling_shape"]),
        )

    # --- Intrinsic Dimension ---

    def get_id_result(
        self,
        points: list[list[float]],
        use_regression: bool,
        bootstrap_samples: int,
    ) -> CachedIDResult | None:
        """
        Get cached intrinsic dimension result.

        Args:
            points: Point cloud
            use_regression: Whether regression method was used
            bootstrap_samples: Number of bootstrap samples

        Returns:
            Cached result or None if not found
        """
        key = self._make_id_key(points, use_regression, bootstrap_samples)
        return self._id_cache.get(key)

    def set_id_result(
        self,
        points: list[list[float]],
        use_regression: bool,
        bootstrap_samples: int,
        result: CachedIDResult,
    ) -> None:
        """
        Cache intrinsic dimension result.

        Args:
            points: Point cloud
            use_regression: Whether regression method was used
            bootstrap_samples: Number of bootstrap samples
            result: Result to cache
        """
        key = self._make_id_key(points, use_regression, bootstrap_samples)
        self._id_cache.set(key, result)

    def _make_id_key(
        self,
        points: list[list[float]],
        use_regression: bool,
        bootstrap_samples: int,
    ) -> str:
        """Create cache key for ID computation."""
        return content_hash(
            {
                "points": sorted([tuple(p) for p in points]),
                "use_regression": use_regression,
                "bootstrap_samples": bootstrap_samples,
            }
        )

    @staticmethod
    def _serialize_id(result: CachedIDResult) -> dict:
        return {
            "dimension": result.dimension,
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
            "sample_count": result.sample_count,
            "use_regression": result.use_regression,
        }

    @staticmethod
    def _deserialize_id(data: dict) -> CachedIDResult:
        return CachedIDResult(
            dimension=float(data["dimension"]),
            confidence_lower=float(data["confidence_lower"]),
            confidence_upper=float(data["confidence_upper"]),
            sample_count=int(data["sample_count"]),
            use_regression=bool(data["use_regression"]),
        )

    # --- Topological Fingerprint ---

    def get_topo_result(
        self,
        points: list[list[float]],
    ) -> CachedTopoResult | None:
        """
        Get cached topological fingerprint result.

        Args:
            points: Point cloud

        Returns:
            Cached result or None if not found
        """
        key = self._make_topo_key(points)
        return self._topo_cache.get(key)

    def set_topo_result(
        self,
        points: list[list[float]],
        result: CachedTopoResult,
    ) -> None:
        """
        Cache topological fingerprint result.

        Args:
            points: Point cloud
            result: Result to cache
        """
        key = self._make_topo_key(points)
        self._topo_cache.set(key, result)

    def _make_topo_key(
        self,
        points: list[list[float]],
    ) -> str:
        """Create cache key for topological computation."""
        return content_hash(
            {
                "points": sorted([tuple(p) for p in points]),
            }
        )

    @staticmethod
    def _serialize_topo(result: CachedTopoResult) -> dict:
        return {
            "betti_0": result.betti_0,
            "betti_1": result.betti_1,
            "persistence_entropy": result.persistence_entropy,
            "total_persistence": result.total_persistence,
        }

    @staticmethod
    def _deserialize_topo(data: dict) -> CachedTopoResult:
        return CachedTopoResult(
            betti_0=int(data["betti_0"]),
            betti_1=int(data["betti_1"]),
            persistence_entropy=float(data["persistence_entropy"]),
            total_persistence=float(data["total_persistence"]),
        )

    # --- Spectral Signature ---

    def get_spectral_result(
        self,
        points: list[list[float]],
    ) -> CachedSpectralResult | None:
        """
        Get cached spectral signature result.

        Args:
            points: Point cloud

        Returns:
            Cached result or None if not found
        """
        key = self._make_spectral_key(points)
        return self._spectral_cache.get(key)

    def set_spectral_result(
        self,
        points: list[list[float]],
        result: CachedSpectralResult,
    ) -> None:
        """
        Cache spectral signature result.

        Args:
            points: Point cloud
            result: Result to cache
        """
        key = self._make_spectral_key(points)
        self._spectral_cache.set(key, result)

    def _make_spectral_key(
        self,
        points: list[list[float]],
    ) -> str:
        """Create cache key for spectral signature computation."""
        return content_hash(
            {
                "points": sorted([tuple(p) for p in points]),
            }
        )

    @staticmethod
    def _serialize_spectral(result: CachedSpectralResult) -> dict:
        return {
            "eigenvalues": result.eigenvalues,
            "heat_trace": result.heat_trace,
            "heat_times": result.heat_times,
            "spectral_entropy": result.spectral_entropy,
            "algebraic_connectivity": result.algebraic_connectivity,
            "component_count": result.component_count,
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "k_neighbors": result.k_neighbors,
            "kernel_bandwidth": result.kernel_bandwidth,
            "normalized_laplacian": result.normalized_laplacian,
            "connected": result.connected,
        }

    @staticmethod
    def _deserialize_spectral(data: dict) -> CachedSpectralResult:
        return CachedSpectralResult(
            eigenvalues=[float(v) for v in data["eigenvalues"]],
            heat_trace=[float(v) for v in data["heat_trace"]],
            heat_times=[float(v) for v in data["heat_times"]],
            spectral_entropy=float(data["spectral_entropy"]),
            algebraic_connectivity=float(data["algebraic_connectivity"]),
            component_count=int(data["component_count"]),
            node_count=int(data["node_count"]),
            edge_count=int(data["edge_count"]),
            k_neighbors=int(data["k_neighbors"]),
            kernel_bandwidth=float(data["kernel_bandwidth"]),
            normalized_laplacian=bool(data["normalized_laplacian"]),
            connected=bool(data["connected"]),
        )

    # --- Utilities ---

    def clear_all(self) -> None:
        """Clear all geometry metrics caches."""
        self._gw_cache.clear_all()
        self._id_cache.clear_all()
        self._topo_cache.clear_all()
        self._spectral_cache.clear_all()
        logger.info("Cleared all geometry metrics caches")
