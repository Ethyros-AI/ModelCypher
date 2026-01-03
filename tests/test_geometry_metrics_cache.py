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

"""Tests for geometry metrics cache."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

from modelcypher.core.domain.geometry.geometry_metrics_cache import (
    CachedGWResult,
    CachedIDResult,
    CachedSpectralResult,
    CachedTopoResult,
    GeometryMetricsCache,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestCachedGWResult:
    """Tests for CachedGWResult dataclass."""

    def test_fields(self):
        result = CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            alignment_score=0.8,
            converged=True,
            iterations=100,
            coupling_shape=(50, 60),
        )
        eps = _eps(result.distance, result.normalized_distance, result.alignment_score)
        assert abs(result.distance - 0.5) <= eps
        assert abs(result.normalized_distance - 0.25) <= eps
        assert abs(result.alignment_score - 0.8) <= eps
        assert result.converged is True
        assert result.iterations == 100
        assert result.coupling_shape == (50, 60)

    def test_frozen(self):
        result = CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            alignment_score=0.8,
            converged=True,
            iterations=100,
            coupling_shape=(50, 60),
        )
        with pytest.raises(AttributeError):
            result.distance = 0.7

    def test_hashable(self):
        result = CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            alignment_score=0.8,
            converged=True,
            iterations=100,
            coupling_shape=(50, 60),
        )
        h = hash(result)
        assert isinstance(h, int)


class TestCachedIDResult:
    """Tests for CachedIDResult dataclass."""

    def test_fields(self):
        result = CachedIDResult(
            dimension=3.5,
            confidence_lower=3.0,
            confidence_upper=4.0,
            sample_count=100,
            use_regression=True,
        )
        eps = _eps(result.dimension, result.confidence_lower, result.confidence_upper)
        assert abs(result.dimension - 3.5) <= eps
        assert abs(result.confidence_lower - 3.0) <= eps
        assert abs(result.confidence_upper - 4.0) <= eps
        assert result.sample_count == 100
        assert result.use_regression is True

    def test_frozen(self):
        result = CachedIDResult(
            dimension=3.5,
            confidence_lower=3.0,
            confidence_upper=4.0,
            sample_count=100,
            use_regression=True,
        )
        with pytest.raises(AttributeError):
            result.dimension = 4.0

    def test_hashable(self):
        result = CachedIDResult(
            dimension=3.5,
            confidence_lower=3.0,
            confidence_upper=4.0,
            sample_count=100,
            use_regression=True,
        )
        h = hash(result)
        assert isinstance(h, int)


class TestCachedTopoResult:
    """Tests for CachedTopoResult dataclass."""

    def test_fields(self):
        result = CachedTopoResult(
            betti_0=1,
            betti_1=2,
            persistence_entropy=0.5,
            total_persistence=1.5,
        )
        assert result.betti_0 == 1
        assert result.betti_1 == 2
        eps = _eps(result.persistence_entropy, result.total_persistence)
        assert abs(result.persistence_entropy - 0.5) <= eps
        assert abs(result.total_persistence - 1.5) <= eps

    def test_frozen(self):
        result = CachedTopoResult(
            betti_0=1,
            betti_1=2,
            persistence_entropy=0.5,
            total_persistence=1.5,
        )
        with pytest.raises(AttributeError):
            result.betti_0 = 5

    def test_hashable(self):
        result = CachedTopoResult(
            betti_0=1,
            betti_1=2,
            persistence_entropy=0.5,
            total_persistence=1.5,
        )
        h = hash(result)
        assert isinstance(h, int)


class TestCachedSpectralResult:
    """Tests for CachedSpectralResult dataclass."""

    def test_fields(self):
        result = CachedSpectralResult(
            eigenvalues=[0.0, 0.1, 0.5, 1.0],
            heat_trace=[1.0, 0.9, 0.8],
            heat_times=[0.1, 1.0, 10.0],
            spectral_entropy=2.5,
            algebraic_connectivity=0.1,
            component_count=1,
            node_count=100,
            edge_count=500,
            k_neighbors=10,
            kernel_bandwidth=1.0,
            normalized_laplacian=True,
            connected=True,
        )
        assert result.eigenvalues == [0.0, 0.1, 0.5, 1.0]
        assert result.heat_trace == [1.0, 0.9, 0.8]
        assert abs(result.spectral_entropy - 2.5) <= _eps(result.spectral_entropy, 2.5)
        assert result.node_count == 100
        assert result.connected is True

    def test_frozen(self):
        result = CachedSpectralResult(
            eigenvalues=[0.0],
            heat_trace=[1.0],
            heat_times=[0.1],
            spectral_entropy=1.0,
            algebraic_connectivity=0.1,
            component_count=1,
            node_count=10,
            edge_count=20,
            k_neighbors=5,
            kernel_bandwidth=1.0,
            normalized_laplacian=False,
            connected=True,
        )
        with pytest.raises(AttributeError):
            result.node_count = 50


class TestGeometryMetricsCacheInit:
    """Tests for GeometryMetricsCache initialization."""

    def test_init_default_directory(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "metrics")
        assert cache._gw_cache is not None
        assert cache._id_cache is not None
        assert cache._topo_cache is not None
        assert cache._spectral_cache is not None

    def test_shared_singleton(self):
        # Reset singleton for testing
        GeometryMetricsCache._shared_instance = None
        instance1 = GeometryMetricsCache.shared()
        instance2 = GeometryMetricsCache.shared()
        assert instance1 is instance2
        # Reset after test
        GeometryMetricsCache._shared_instance = None


class TestGeometryMetricsCacheGW:
    """Tests for Gromov-Wasserstein caching."""

    @pytest.fixture
    def cache(self, tmp_path):
        return GeometryMetricsCache(cache_directory=tmp_path / "metrics")

    @pytest.fixture
    def sample_points(self):
        source = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        target = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        return source, target

    @pytest.fixture
    def sample_gw_result(self):
        return CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            alignment_score=0.8,
            converged=True,
            iterations=50,
            coupling_shape=(3, 3),
        )

    def test_set_and_get(self, cache, sample_points, sample_gw_result):
        source, target = sample_points
        cache.set_gw_result(source, target, epsilon=0.01, max_iterations=100, result=sample_gw_result)

        loaded = cache.get_gw_result(source, target, epsilon=0.01, max_iterations=100)
        assert loaded is not None
        assert abs(loaded.distance - sample_gw_result.distance) <= _eps(
            loaded.distance, sample_gw_result.distance
        )
        assert loaded.converged == sample_gw_result.converged

    def test_get_uncached(self, cache, sample_points):
        source, target = sample_points
        result = cache.get_gw_result(source, target, epsilon=0.01, max_iterations=100)
        assert result is None

    def test_different_epsilon_different_key(self, cache, sample_points, sample_gw_result):
        source, target = sample_points
        cache.set_gw_result(source, target, epsilon=0.01, max_iterations=100, result=sample_gw_result)

        # Different epsilon should not find the cached result
        result = cache.get_gw_result(source, target, epsilon=0.02, max_iterations=100)
        assert result is None

    def test_different_points_different_key(self, cache, sample_gw_result):
        source1 = [[0.0, 0.0], [1.0, 0.0]]
        source2 = [[0.0, 0.0], [2.0, 0.0]]
        target = [[0.0, 0.0], [1.0, 1.0]]

        cache.set_gw_result(source1, target, epsilon=0.01, max_iterations=100, result=sample_gw_result)

        # Different source should not find the cached result
        result = cache.get_gw_result(source2, target, epsilon=0.01, max_iterations=100)
        assert result is None


class TestGeometryMetricsCacheID:
    """Tests for intrinsic dimension caching."""

    @pytest.fixture
    def cache(self, tmp_path):
        return GeometryMetricsCache(cache_directory=tmp_path / "metrics")

    @pytest.fixture
    def sample_points(self):
        return [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    @pytest.fixture
    def sample_id_result(self):
        return CachedIDResult(
            dimension=2.5,
            confidence_lower=2.0,
            confidence_upper=3.0,
            sample_count=100,
            use_regression=True,
        )

    def test_set_and_get(self, cache, sample_points, sample_id_result):
        cache.set_id_result(sample_points, use_regression=True, bootstrap_samples=50, result=sample_id_result)

        loaded = cache.get_id_result(sample_points, use_regression=True, bootstrap_samples=50)
        assert loaded is not None
        assert abs(loaded.dimension - sample_id_result.dimension) <= _eps(
            loaded.dimension, sample_id_result.dimension
        )

    def test_get_uncached(self, cache, sample_points):
        result = cache.get_id_result(sample_points, use_regression=True, bootstrap_samples=50)
        assert result is None

    def test_different_regression_different_key(self, cache, sample_points, sample_id_result):
        cache.set_id_result(sample_points, use_regression=True, bootstrap_samples=50, result=sample_id_result)

        # Different use_regression should not find the cached result
        result = cache.get_id_result(sample_points, use_regression=False, bootstrap_samples=50)
        assert result is None


class TestGeometryMetricsCacheTopo:
    """Tests for topological fingerprint caching."""

    @pytest.fixture
    def cache(self, tmp_path):
        return GeometryMetricsCache(cache_directory=tmp_path / "metrics")

    @pytest.fixture
    def sample_points(self):
        return [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]

    @pytest.fixture
    def sample_topo_result(self):
        return CachedTopoResult(
            betti_0=1,
            betti_1=0,
            persistence_entropy=0.3,
            total_persistence=2.5,
        )

    def test_set_and_get(self, cache, sample_points, sample_topo_result):
        # Cache key is now derived from points only (algorithm params are data-derived)
        cache.set_topo_result(sample_points, sample_topo_result)

        loaded = cache.get_topo_result(sample_points)
        assert loaded is not None
        assert loaded.betti_0 == sample_topo_result.betti_0
        assert abs(loaded.persistence_entropy - sample_topo_result.persistence_entropy) <= _eps(
            loaded.persistence_entropy, sample_topo_result.persistence_entropy
        )

    def test_get_uncached(self, cache, sample_points):
        result = cache.get_topo_result(sample_points)
        assert result is None

    def test_different_points_different_key(self, cache, sample_points, sample_topo_result):
        cache.set_topo_result(sample_points, sample_topo_result)

        # Different points should give different key
        different_points = [[0.0, 0.0], [2.0, 0.0]]
        result = cache.get_topo_result(different_points)
        assert result is None


class TestGeometryMetricsCacheSpectral:
    """Tests for spectral signature caching."""

    @pytest.fixture
    def cache(self, tmp_path):
        return GeometryMetricsCache(cache_directory=tmp_path / "metrics")

    @pytest.fixture
    def sample_points(self):
        return [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    @pytest.fixture
    def sample_spectral_result(self):
        return CachedSpectralResult(
            eigenvalues=[0.0, 0.5, 1.0, 2.0],
            heat_trace=[4.0, 3.5, 3.0],
            heat_times=[0.1, 1.0, 10.0],
            spectral_entropy=1.5,
            algebraic_connectivity=0.5,
            component_count=1,
            node_count=4,
            edge_count=6,
            k_neighbors=2,
            kernel_bandwidth=1.0,
            normalized_laplacian=True,
            connected=True,
        )

    def test_set_and_get(self, cache, sample_points, sample_spectral_result):
        # Cache key is now derived from points only (algorithm params are data-derived)
        cache.set_spectral_result(sample_points, sample_spectral_result)

        loaded = cache.get_spectral_result(sample_points)
        assert loaded is not None
        assert loaded.eigenvalues == sample_spectral_result.eigenvalues
        assert abs(loaded.spectral_entropy - sample_spectral_result.spectral_entropy) <= _eps(
            loaded.spectral_entropy, sample_spectral_result.spectral_entropy
        )

    def test_get_uncached(self, cache, sample_points):
        result = cache.get_spectral_result(sample_points)
        assert result is None

    def test_different_points_different_key(self, cache, sample_points, sample_spectral_result):
        cache.set_spectral_result(sample_points, sample_spectral_result)

        # Different points should give different key
        different_points = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        result = cache.get_spectral_result(different_points)
        assert result is None


class TestGeometryMetricsCacheSerialization:
    """Tests for serialization/deserialization methods."""

    def test_gw_serialization_roundtrip(self):
        original = CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            alignment_score=0.8,
            converged=True,
            iterations=100,
            coupling_shape=(50, 60),
        )
        serialized = GeometryMetricsCache._serialize_gw(original)
        restored = GeometryMetricsCache._deserialize_gw(serialized)

        eps = _eps(restored.distance, restored.normalized_distance)
        assert abs(restored.distance - original.distance) <= eps
        assert abs(restored.normalized_distance - original.normalized_distance) <= eps
        assert restored.converged == original.converged
        assert restored.coupling_shape == original.coupling_shape

    def test_id_serialization_roundtrip(self):
        original = CachedIDResult(
            dimension=3.5,
            confidence_lower=3.0,
            confidence_upper=4.0,
            sample_count=100,
            use_regression=True,
        )
        serialized = GeometryMetricsCache._serialize_id(original)
        restored = GeometryMetricsCache._deserialize_id(serialized)

        eps = _eps(restored.dimension, restored.confidence_lower)
        assert abs(restored.dimension - original.dimension) <= eps
        assert abs(restored.confidence_lower - original.confidence_lower) <= eps
        assert restored.use_regression == original.use_regression

    def test_topo_serialization_roundtrip(self):
        original = CachedTopoResult(
            betti_0=1,
            betti_1=2,
            persistence_entropy=0.5,
            total_persistence=1.5,
        )
        serialized = GeometryMetricsCache._serialize_topo(original)
        restored = GeometryMetricsCache._deserialize_topo(serialized)

        assert restored.betti_0 == original.betti_0
        assert restored.betti_1 == original.betti_1
        assert abs(restored.persistence_entropy - original.persistence_entropy) <= _eps(
            restored.persistence_entropy, original.persistence_entropy
        )

    def test_spectral_serialization_roundtrip(self):
        original = CachedSpectralResult(
            eigenvalues=[0.0, 0.5, 1.0],
            heat_trace=[1.0, 0.9, 0.8],
            heat_times=[0.1, 1.0, 10.0],
            spectral_entropy=2.0,
            algebraic_connectivity=0.5,
            component_count=1,
            node_count=100,
            edge_count=500,
            k_neighbors=10,
            kernel_bandwidth=1.0,
            normalized_laplacian=True,
            connected=True,
        )
        serialized = GeometryMetricsCache._serialize_spectral(original)
        restored = GeometryMetricsCache._deserialize_spectral(serialized)

        assert restored.eigenvalues == original.eigenvalues
        assert restored.heat_trace == original.heat_trace
        assert abs(restored.spectral_entropy - original.spectral_entropy) <= _eps(
            restored.spectral_entropy, original.spectral_entropy
        )
        assert restored.connected == original.connected


class TestGeometryMetricsCacheClearAll:
    """Tests for clear_all method."""

    def test_clear_all(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "metrics")

        # Add items to each cache
        points = [[0.0, 0.0], [1.0, 0.0]]

        cache.set_gw_result(
            points, points, epsilon=0.01, max_iterations=100,
            result=CachedGWResult(0.1, 0.05, 0.9, True, 10, (2, 2))
        )
        cache.set_id_result(
            points, use_regression=True, bootstrap_samples=50,
            result=CachedIDResult(2.0, 1.5, 2.5, 10, True)
        )
        # Topo and spectral cache keys are now derived from points only
        cache.set_topo_result(points, CachedTopoResult(1, 0, 0.1, 0.5))
        cache.set_spectral_result(
            points,
            CachedSpectralResult([0.0], [1.0], [0.1], 0.5, 0.1, 1, 2, 1, 2, 1.0, True, True),
        )

        # Verify items exist
        assert cache.get_gw_result(points, points, 0.01, 100) is not None
        assert cache.get_id_result(points, True, 50) is not None
        assert cache.get_topo_result(points) is not None
        assert cache.get_spectral_result(points) is not None

        # Clear all
        cache.clear_all()

        # Verify all cleared
        assert cache.get_gw_result(points, points, 0.01, 100) is None
        assert cache.get_id_result(points, True, 50) is None
        assert cache.get_topo_result(points) is None
        assert cache.get_spectral_result(points) is None
