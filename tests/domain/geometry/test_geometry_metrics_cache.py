# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for geometry_metrics_cache module.

Covers all cached result dataclasses and GeometryMetricsCache get/set
round-trips for GW, ID, Topo, Spectral, and Entanglement results.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.geometry_metrics_cache import (
    CachedEntanglementResult,
    CachedGWResult,
    CachedIDResult,
    CachedSpectralResult,
    CachedTopoResult,
    GeometryMetricsCache,
)


# ---------------------------------------------------------------------------
# Cached result dataclasses
# ---------------------------------------------------------------------------


class TestCachedGWResult:
    def test_instantiation(self):
        r = CachedGWResult(
            distance=0.5,
            normalized_distance=0.25,
            converged=True,
            iterations=42,
            coupling_shape=(10, 10),
        )
        assert r.distance == 0.5
        assert r.normalized_distance == 0.25
        assert r.converged is True
        assert r.iterations == 42
        assert r.coupling_shape == (10, 10)

    def test_frozen(self):
        r = CachedGWResult(0.1, 0.05, True, 10, (5, 5))
        with pytest.raises(AttributeError):
            r.distance = 99.0  # type: ignore[misc]


class TestCachedIDResult:
    def test_instantiation(self):
        r = CachedIDResult(
            dimension=3.14,
            confidence_lower=2.8,
            confidence_upper=3.5,
            sample_count=100,
            use_regression=True,
        )
        assert r.dimension == 3.14
        assert r.confidence_lower == 2.8
        assert r.confidence_upper == 3.5
        assert r.sample_count == 100
        assert r.use_regression is True

    def test_frozen(self):
        r = CachedIDResult(1.0, 0.5, 1.5, 50, False)
        with pytest.raises(AttributeError):
            r.dimension = 99.0  # type: ignore[misc]


class TestCachedTopoResult:
    def test_instantiation(self):
        r = CachedTopoResult(
            betti_0=3,
            betti_1=1,
            persistence_entropy=0.8,
            total_persistence=2.5,
        )
        assert r.betti_0 == 3
        assert r.betti_1 == 1
        assert r.persistence_entropy == 0.8
        assert r.total_persistence == 2.5

    def test_frozen(self):
        r = CachedTopoResult(1, 0, 0.5, 1.0)
        with pytest.raises(AttributeError):
            r.betti_0 = 99  # type: ignore[misc]


class TestCachedSpectralResult:
    def test_instantiation(self):
        r = CachedSpectralResult(
            eigenvalues=[0.0, 0.1, 0.5],
            heat_trace=[1.0, 0.9, 0.8],
            heat_times=[0.01, 0.1, 1.0],
            spectral_entropy=1.2,
            algebraic_connectivity=0.1,
            component_count=1,
            node_count=10,
            edge_count=20,
            k_neighbors=5,
            kernel_bandwidth=0.5,
            normalized_laplacian=True,
            connected=True,
        )
        assert r.eigenvalues == [0.0, 0.1, 0.5]
        assert r.heat_trace == [1.0, 0.9, 0.8]
        assert r.heat_times == [0.01, 0.1, 1.0]
        assert r.spectral_entropy == 1.2
        assert r.algebraic_connectivity == 0.1
        assert r.component_count == 1
        assert r.node_count == 10
        assert r.edge_count == 20
        assert r.k_neighbors == 5
        assert r.kernel_bandwidth == 0.5
        assert r.normalized_laplacian is True
        assert r.connected is True

    def test_frozen(self):
        r = CachedSpectralResult(
            [0.0], [1.0], [0.01], 0.5, 0.1, 1, 5, 8, 3, 0.3, False, True,
        )
        with pytest.raises(AttributeError):
            r.spectral_entropy = 99.0  # type: ignore[misc]


class TestCachedEntanglementResult:
    def test_instantiation(self):
        r = CachedEntanglementResult(
            canonical_correlations=[0.99, 0.85, 0.5],
            entanglement_entropy=1.5,
            effective_rank_shannon=2.3,
            effective_rank_renyi=2.1,
            correlation_count=3,
            condition_number=10.0,
        )
        assert r.canonical_correlations == [0.99, 0.85, 0.5]
        assert r.entanglement_entropy == 1.5
        assert r.effective_rank_shannon == 2.3
        assert r.effective_rank_renyi == 2.1
        assert r.correlation_count == 3
        assert r.condition_number == 10.0

    def test_frozen(self):
        r = CachedEntanglementResult([1.0], 0.5, 1.0, 1.0, 1, 1.0)
        with pytest.raises(AttributeError):
            r.entanglement_entropy = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GeometryMetricsCache
# ---------------------------------------------------------------------------

# Shared test point clouds
_SOURCE_PTS = [[1.0, 2.0], [3.0, 4.0]]
_TARGET_PTS = [[5.0, 6.0], [7.0, 8.0]]
_POINTS = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


class TestGeometryMetricsCacheInit:
    def test_constructor_with_tmp_path(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "gm_cache")
        assert cache is not None

    def test_shared_singleton(self, monkeypatch):
        monkeypatch.setattr(GeometryMetricsCache, "_shared_instance", None)
        a = GeometryMetricsCache.shared()
        b = GeometryMetricsCache.shared()
        assert a is b

    def test_shared_singleton_reset(self, monkeypatch):
        monkeypatch.setattr(GeometryMetricsCache, "_shared_instance", None)
        first = GeometryMetricsCache.shared()
        monkeypatch.setattr(GeometryMetricsCache, "_shared_instance", None)
        second = GeometryMetricsCache.shared()
        assert first is not second


class TestGWCacheRoundTrip:
    def test_miss_returns_none(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "gw_miss")
        assert cache.get_gw_result(_SOURCE_PTS, _TARGET_PTS) is None

    def test_set_then_get(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "gw_rt")
        result = CachedGWResult(
            distance=0.3,
            normalized_distance=0.15,
            converged=True,
            iterations=20,
            coupling_shape=(2, 2),
        )
        cache.set_gw_result(_SOURCE_PTS, _TARGET_PTS, result)
        loaded = cache.get_gw_result(_SOURCE_PTS, _TARGET_PTS)

        assert loaded is not None
        assert loaded.distance == 0.3
        assert loaded.normalized_distance == 0.15
        assert loaded.converged is True
        assert loaded.iterations == 20
        assert loaded.coupling_shape == (2, 2)

    def test_different_inputs_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "gw_diff")
        result = CachedGWResult(0.3, 0.15, True, 20, (2, 2))
        cache.set_gw_result(_SOURCE_PTS, _TARGET_PTS, result)
        # Different source points should miss
        other_src = [[9.0, 10.0], [11.0, 12.0]]
        assert cache.get_gw_result(other_src, _TARGET_PTS) is None


class TestIDCacheRoundTrip:
    def test_miss_returns_none(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "id_miss")
        assert cache.get_id_result(_POINTS, True, 100) is None

    def test_set_then_get(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "id_rt")
        result = CachedIDResult(
            dimension=2.5,
            confidence_lower=2.0,
            confidence_upper=3.0,
            sample_count=50,
            use_regression=True,
        )
        cache.set_id_result(_POINTS, True, 100, result)
        loaded = cache.get_id_result(_POINTS, True, 100)

        assert loaded is not None
        assert loaded.dimension == 2.5
        assert loaded.use_regression is True

    def test_different_regression_flag_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "id_reg")
        result = CachedIDResult(2.5, 2.0, 3.0, 50, True)
        cache.set_id_result(_POINTS, True, 100, result)
        # use_regression=False should produce a different key
        assert cache.get_id_result(_POINTS, False, 100) is None

    def test_different_bootstrap_samples_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "id_bs")
        result = CachedIDResult(2.5, 2.0, 3.0, 50, True)
        cache.set_id_result(_POINTS, True, 100, result)
        assert cache.get_id_result(_POINTS, True, 200) is None


class TestTopoCacheRoundTrip:
    def test_miss_returns_none(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "topo_miss")
        assert cache.get_topo_result(_POINTS) is None

    def test_set_then_get(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "topo_rt")
        result = CachedTopoResult(
            betti_0=2,
            betti_1=0,
            persistence_entropy=0.7,
            total_persistence=1.5,
        )
        cache.set_topo_result(_POINTS, result)
        loaded = cache.get_topo_result(_POINTS)

        assert loaded is not None
        assert loaded.betti_0 == 2
        assert loaded.betti_1 == 0
        assert loaded.persistence_entropy == 0.7

    def test_different_points_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "topo_diff")
        result = CachedTopoResult(2, 0, 0.7, 1.5)
        cache.set_topo_result(_POINTS, result)
        other = [[99.0, 99.0]]
        assert cache.get_topo_result(other) is None


class TestSpectralCacheRoundTrip:
    def test_miss_returns_none(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "spec_miss")
        assert cache.get_spectral_result(_POINTS) is None

    def test_set_then_get(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "spec_rt")
        result = CachedSpectralResult(
            eigenvalues=[0.0, 0.5, 1.0],
            heat_trace=[3.0, 2.5, 2.0],
            heat_times=[0.01, 0.1, 1.0],
            spectral_entropy=0.9,
            algebraic_connectivity=0.5,
            component_count=1,
            node_count=3,
            edge_count=3,
            k_neighbors=2,
            kernel_bandwidth=0.3,
            normalized_laplacian=True,
            connected=True,
        )
        cache.set_spectral_result(_POINTS, result)
        loaded = cache.get_spectral_result(_POINTS)

        assert loaded is not None
        assert loaded.eigenvalues == [0.0, 0.5, 1.0]
        assert loaded.spectral_entropy == 0.9
        assert loaded.algebraic_connectivity == 0.5
        assert loaded.connected is True
        assert loaded.normalized_laplacian is True

    def test_different_points_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "spec_diff")
        result = CachedSpectralResult(
            [0.0], [1.0], [0.01], 0.5, 0.1, 1, 3, 3, 2, 0.3, True, True,
        )
        cache.set_spectral_result(_POINTS, result)
        assert cache.get_spectral_result([[0.0, 0.0]]) is None


class TestEntanglementCacheRoundTrip:
    def test_miss_returns_none(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "ent_miss")
        assert cache.get_entanglement_result(_SOURCE_PTS, _TARGET_PTS) is None

    def test_set_then_get(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "ent_rt")
        result = CachedEntanglementResult(
            canonical_correlations=[0.99, 0.8],
            entanglement_entropy=1.2,
            effective_rank_shannon=1.8,
            effective_rank_renyi=1.6,
            correlation_count=2,
            condition_number=5.0,
        )
        cache.set_entanglement_result(_SOURCE_PTS, _TARGET_PTS, result)
        loaded = cache.get_entanglement_result(_SOURCE_PTS, _TARGET_PTS)

        assert loaded is not None
        assert loaded.canonical_correlations == [0.99, 0.8]
        assert loaded.entanglement_entropy == 1.2
        assert loaded.correlation_count == 2

    def test_different_inputs_miss(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "ent_diff")
        result = CachedEntanglementResult([1.0], 0.5, 1.0, 1.0, 1, 1.0)
        cache.set_entanglement_result(_SOURCE_PTS, _TARGET_PTS, result)
        other_tgt = [[0.0, 0.0], [1.0, 1.0]]
        assert cache.get_entanglement_result(_SOURCE_PTS, other_tgt) is None


class TestClearAll:
    def test_clear_all_empties_caches(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "clear")

        # Populate each sub-cache
        cache.set_gw_result(
            _SOURCE_PTS,
            _TARGET_PTS,
            CachedGWResult(0.1, 0.05, True, 5, (2, 2)),
        )
        cache.set_id_result(
            _POINTS,
            True,
            100,
            CachedIDResult(2.0, 1.5, 2.5, 50, True),
        )
        cache.set_topo_result(
            _POINTS,
            CachedTopoResult(1, 0, 0.5, 1.0),
        )
        cache.set_spectral_result(
            _POINTS,
            CachedSpectralResult(
                [0.0], [1.0], [0.01], 0.5, 0.1, 1, 3, 3, 2, 0.3, True, True,
            ),
        )
        cache.set_entanglement_result(
            _SOURCE_PTS,
            _TARGET_PTS,
            CachedEntanglementResult([1.0], 0.5, 1.0, 1.0, 1, 1.0),
        )

        cache.clear_all()

        # All should be gone from memory
        assert cache.get_gw_result(_SOURCE_PTS, _TARGET_PTS) is None
        assert cache.get_id_result(_POINTS, True, 100) is None
        assert cache.get_topo_result(_POINTS) is None
        assert cache.get_spectral_result(_POINTS) is None
        assert cache.get_entanglement_result(_SOURCE_PTS, _TARGET_PTS) is None

    def test_clear_all_on_empty_no_error(self, tmp_path):
        cache = GeometryMetricsCache(cache_directory=tmp_path / "clear_empty")
        cache.clear_all()
