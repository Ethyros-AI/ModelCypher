# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Comprehensive tests for Riemannian geometry type definitions."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

import modelcypher.core.domain.geometry.riemannian_types as mod

# =============================================================================
# Helper: lightweight array-like objects for testing frozen dataclasses
# =============================================================================


class _FakeArray:
    """Minimal array-like object with .shape attribute for testing."""

    def __init__(self, data: list, shape: tuple[int, ...] | None = None) -> None:
        self._data = data
        if shape is not None:
            self.shape = shape
        else:
            # Infer 1D shape
            self.shape = (len(data),)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FakeArray):
            return self._data == other._data
        return NotImplemented

    def __repr__(self) -> str:
        return f"_FakeArray({self._data}, shape={self.shape})"


def _make_distance_matrix(n: int) -> _FakeArray:
    """Create a fake n x n distance matrix."""
    return _FakeArray([[0.0] * n for _ in range(n)], shape=(n, n))


def _make_points(n: int, d: int) -> _FakeArray:
    """Create a fake n x d point matrix."""
    return _FakeArray([[0.0] * d for _ in range(n)], shape=(n, d))


# =============================================================================
# FrechetMeanResult (frozen dataclass)
# =============================================================================


class TestFrechetMeanResult:
    def test_instantiation(self) -> None:
        mean = _FakeArray([1.0, 2.0, 3.0])
        result = mod.FrechetMeanResult(
            mean=mean,
            iterations=15,
            converged=True,
            final_variance=0.01,
        )
        assert result.mean is mean
        assert result.iterations == 15
        assert result.converged is True
        assert result.final_variance == 0.01

    def test_not_converged(self) -> None:
        result = mod.FrechetMeanResult(
            mean=_FakeArray([0.0]),
            iterations=100,
            converged=False,
            final_variance=5.0,
        )
        assert result.converged is False
        assert result.iterations == 100
        assert result.final_variance == 5.0

    def test_frozen_enforcement(self) -> None:
        result = mod.FrechetMeanResult(
            mean=_FakeArray([1.0]),
            iterations=10,
            converged=True,
            final_variance=0.0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.converged = False  # type: ignore[misc]

    def test_frozen_cannot_set_mean(self) -> None:
        result = mod.FrechetMeanResult(
            mean=_FakeArray([1.0]),
            iterations=1,
            converged=True,
            final_variance=0.0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.mean = _FakeArray([2.0])  # type: ignore[misc]


# =============================================================================
# GeodesicDistanceResult (frozen dataclass)
# =============================================================================


class TestGeodesicDistanceResult:
    def test_instantiation(self) -> None:
        distances = _make_distance_matrix(5)
        adjacency = _make_distance_matrix(5)
        result = mod.GeodesicDistanceResult(
            distances=distances,
            adjacency=adjacency,
            inf_value=1e10,
            k_neighbors=3,
            connected=True,
        )
        assert result.distances is distances
        assert result.adjacency is adjacency
        assert result.inf_value == 1e10
        assert result.k_neighbors == 3
        assert result.connected is True

    def test_disconnected_graph(self) -> None:
        result = mod.GeodesicDistanceResult(
            distances=_make_distance_matrix(4),
            adjacency=_make_distance_matrix(4),
            inf_value=float("inf"),
            k_neighbors=2,
            connected=False,
        )
        assert result.connected is False
        assert result.inf_value == float("inf")

    def test_frozen_enforcement(self) -> None:
        result = mod.GeodesicDistanceResult(
            distances=_make_distance_matrix(3),
            adjacency=_make_distance_matrix(3),
            inf_value=1e12,
            k_neighbors=2,
            connected=True,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.connected = False  # type: ignore[misc]

    def test_frozen_cannot_set_k_neighbors(self) -> None:
        result = mod.GeodesicDistanceResult(
            distances=_make_distance_matrix(3),
            adjacency=_make_distance_matrix(3),
            inf_value=1e12,
            k_neighbors=2,
            connected=True,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.k_neighbors = 5  # type: ignore[misc]


# =============================================================================
# CurvatureEstimate (frozen dataclass)
# =============================================================================


class TestCurvatureEstimate:
    def test_positive_curvature(self) -> None:
        est = mod.CurvatureEstimate(
            sectional_curvature=0.5,
            is_positive=True,
            is_negative=False,
            confidence=0.9,
        )
        assert est.sectional_curvature == 0.5
        assert est.is_positive is True
        assert est.is_negative is False
        assert est.confidence == 0.9

    def test_negative_curvature(self) -> None:
        est = mod.CurvatureEstimate(
            sectional_curvature=-1.2,
            is_positive=False,
            is_negative=True,
            confidence=0.75,
        )
        assert est.sectional_curvature == -1.2
        assert est.is_positive is False
        assert est.is_negative is True

    def test_flat_curvature(self) -> None:
        est = mod.CurvatureEstimate(
            sectional_curvature=0.0,
            is_positive=False,
            is_negative=False,
            confidence=1.0,
        )
        assert est.sectional_curvature == 0.0
        assert est.is_positive is False
        assert est.is_negative is False

    def test_frozen_enforcement(self) -> None:
        est = mod.CurvatureEstimate(
            sectional_curvature=0.1,
            is_positive=True,
            is_negative=False,
            confidence=0.5,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            est.confidence = 0.99  # type: ignore[misc]


# =============================================================================
# DirectionalCoverage (frozen dataclass)
# =============================================================================


class TestDirectionalCoverage:
    def test_instantiation(self) -> None:
        sparse_dir = _FakeArray([1.0, 0.0, 0.0])
        neighbor_dirs = _FakeArray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            shape=(2, 3),
        )
        cov = mod.DirectionalCoverage(
            sparse_direction=sparse_dir,
            max_gap_angle=1.57,
            coverage_variance=0.02,
            neighbor_directions=neighbor_dirs,
            point_idx=7,
        )
        assert cov.sparse_direction is sparse_dir
        assert cov.max_gap_angle == pytest.approx(1.57)
        assert cov.coverage_variance == 0.02
        assert cov.neighbor_directions is neighbor_dirs
        assert cov.point_idx == 7

    def test_frozen_enforcement(self) -> None:
        cov = mod.DirectionalCoverage(
            sparse_direction=_FakeArray([1.0]),
            max_gap_angle=0.5,
            coverage_variance=0.1,
            neighbor_directions=_FakeArray([[1.0]], shape=(1, 1)),
            point_idx=0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            cov.point_idx = 3  # type: ignore[misc]


# =============================================================================
# FarthestPointSamplingResult (frozen dataclass)
# =============================================================================


class TestFarthestPointSamplingResult:
    def test_instantiation(self) -> None:
        result = mod.FarthestPointSamplingResult(
            selected_indices=[0, 5, 12, 8],
            min_distances=_FakeArray([0.1, 0.2, 0.3, 0.4, 0.5]),
            coverage_radius=0.5,
        )
        assert result.selected_indices == [0, 5, 12, 8]
        assert result.coverage_radius == 0.5

    def test_single_point(self) -> None:
        result = mod.FarthestPointSamplingResult(
            selected_indices=[0],
            min_distances=_FakeArray([0.0]),
            coverage_radius=0.0,
        )
        assert len(result.selected_indices) == 1
        assert result.coverage_radius == 0.0

    def test_frozen_enforcement(self) -> None:
        result = mod.FarthestPointSamplingResult(
            selected_indices=[0, 1],
            min_distances=_FakeArray([0.1]),
            coverage_radius=0.3,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.coverage_radius = 1.0  # type: ignore[misc]

    def test_frozen_cannot_replace_selected_indices(self) -> None:
        result = mod.FarthestPointSamplingResult(
            selected_indices=[0],
            min_distances=_FakeArray([0.0]),
            coverage_radius=0.0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.selected_indices = [1, 2]  # type: ignore[misc]


# =============================================================================
# ManifoldContext (frozen dataclass with properties)
# =============================================================================


class TestManifoldContext:
    def _make_geo_result(
        self,
        n: int = 10,
        k: int = 3,
        connected: bool = True,
    ) -> mod.GeodesicDistanceResult:
        return mod.GeodesicDistanceResult(
            distances=_make_distance_matrix(n),
            adjacency=_make_distance_matrix(n),
            inf_value=1e10,
            k_neighbors=k,
            connected=connected,
        )

    def test_instantiation_without_sigma(self) -> None:
        geo = self._make_geo_result(n=10, k=3)
        points = _make_points(10, 5)
        ctx = mod.ManifoldContext(
            geo_result=geo,
            points=points,
        )
        assert ctx.geo_result is geo
        assert ctx.points is points
        assert ctx.sigma is None

    def test_instantiation_with_sigma(self) -> None:
        geo = self._make_geo_result(n=10, k=3)
        points = _make_points(10, 5)
        ctx = mod.ManifoldContext(
            geo_result=geo,
            points=points,
            sigma=2.5,
        )
        assert ctx.sigma == 2.5

    def test_n_points_property(self) -> None:
        geo = self._make_geo_result(n=20)
        points = _make_points(20, 8)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.n_points == 20

    def test_k_neighbors_property(self) -> None:
        geo = self._make_geo_result(n=10, k=5)
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.k_neighbors == 5

    def test_is_connected_property_true(self) -> None:
        geo = self._make_geo_result(connected=True)
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.is_connected is True

    def test_is_connected_property_false(self) -> None:
        geo = self._make_geo_result(connected=False)
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.is_connected is False

    def test_distances_property_delegates_to_geo_result(self) -> None:
        geo = self._make_geo_result(n=10)
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.distances is geo.distances

    def test_frozen_enforcement(self) -> None:
        geo = self._make_geo_result()
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            ctx.sigma = 1.0  # type: ignore[misc]

    def test_frozen_cannot_replace_geo_result(self) -> None:
        geo = self._make_geo_result()
        points = _make_points(10, 3)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            ctx.geo_result = self._make_geo_result()  # type: ignore[misc]

    def test_properties_with_different_dimensions(self) -> None:
        geo = self._make_geo_result(n=50, k=7)
        points = _make_points(50, 128)
        ctx = mod.ManifoldContext(geo_result=geo, points=points)
        assert ctx.n_points == 50
        assert ctx.k_neighbors == 7
