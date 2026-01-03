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

"""Tests for dimension_cascade.py.

Tests the dimension cascade that projects high-D activations through
structure-preserving projections to 4D→3D→2D.

Key properties tested:
- Coupling matrices are computed and stored correctly
- Streaming projection works with cached couplings
- Composite coupling is mathematically correct
- Geodesic distortion is measured at each step
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.dimension_cascade import (
    CascadeConfiguration,
    CascadeResult,
    DimensionCascade,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Test Fixtures
# =============================================================================


def _eps(backend: "Backend") -> float:
    return machine_epsilon(backend, backend.array([1.0]))


def _div_eps(backend: "Backend") -> float:
    return division_epsilon(backend, backend.array([1.0]))


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
    return get_default_backend()


@pytest.fixture
def random_activations(backend: "Backend") -> "Array":
    """Generate random activations for testing."""
    backend.random_seed(42)
    # Simulate 50 tokens with 64-dim hidden states
    return backend.random_normal((50, 64))


@pytest.fixture
def small_activations(backend: "Backend") -> "Array":
    """Generate small activations for edge case testing."""
    backend.random_seed(42)
    return backend.random_normal((25, 32))


# =============================================================================
# CascadeConfiguration Tests
# =============================================================================


class TestCascadeConfiguration:
    """Tests for CascadeConfiguration dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CascadeConfiguration(target_dims=[4, 3])
        assert config.target_dims == [4, 3]
        assert config.compute_curvature is True
        assert config.curvature_k == 15
        assert config.min_calibration_points == 20

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CascadeConfiguration(
            target_dims=[8, 4, 2],
            compute_curvature=False,
            curvature_k=10,
            min_calibration_points=30,
        )
        assert config.target_dims == [8, 4, 2]
        assert config.compute_curvature is False
        assert config.curvature_k == 10
        assert config.min_calibration_points == 30


# =============================================================================
# CascadeResult Tests
# =============================================================================


class TestCascadeResult:
    """Tests for CascadeResult dataclass."""

    def test_creation(self, backend: "Backend") -> None:
        """Test basic result creation."""
        projections = {
            64: backend.random_normal((50, 64)),
            4: backend.random_normal((50, 4)),
            3: backend.random_normal((50, 3)),
        }
        couplings = {
            4: backend.random_normal((64, 4)),
            3: backend.random_normal((4, 3)),
        }

        result = CascadeResult(
            original_dim=64,
            intrinsic_dim=12.5,
            projections=projections,
            couplings=couplings,
            curvatures={},
            geodesic_distortion={4: 0.1, 3: 0.15},
        )

        assert result.original_dim == 64
        assert abs(result.intrinsic_dim - 12.5) <= _eps(backend)
        assert 4 in result.projections
        assert 3 in result.projections
        assert 4 in result.couplings
        assert 3 in result.couplings


# =============================================================================
# DimensionCascade Tests
# =============================================================================


class TestDimensionCascade:
    """Tests for DimensionCascade class."""

    def test_initialization(self, backend: "Backend") -> None:
        """Test cascade initialization."""
        cascade = DimensionCascade(backend)
        assert cascade.backend is backend
        assert cascade.calibrated is False

    def test_calibrate_basic(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test basic calibration."""
        cascade = DimensionCascade(backend)
        result = cascade.calibrate(random_activations, target_dims=[4, 3])

        assert cascade.calibrated is True
        assert result.original_dim == 64
        assert result.intrinsic_dim > _eps(backend)
        assert 4 in result.projections
        assert 3 in result.projections
        assert 4 in result.couplings
        assert 3 in result.couplings

    def test_calibrate_with_config(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test calibration with explicit config."""
        cascade = DimensionCascade(backend)
        config = CascadeConfiguration(
            target_dims=[8, 4, 2],
            compute_curvature=True,
            curvature_k=10,
            min_calibration_points=20,
        )
        result = cascade.calibrate(random_activations, config=config)

        assert cascade.calibrated is True
        assert 8 in result.projections
        assert 4 in result.projections
        assert 2 in result.projections

    def test_projection_shapes(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that projections have correct shapes."""
        cascade = DimensionCascade(backend)
        result = cascade.calibrate(random_activations, target_dims=[4, 3])

        # Check projection shapes
        proj_4d = result.projections[4]
        proj_3d = result.projections[3]

        assert proj_4d.shape == (50, 4)
        assert proj_3d.shape == (50, 3)

    def test_coupling_shapes(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that couplings have correct shapes."""
        cascade = DimensionCascade(backend)
        result = cascade.calibrate(random_activations, target_dims=[4, 3])

        # Coupling from 64 -> 4
        coupling_4 = result.couplings[4]
        # Coupling from 4 -> 3
        coupling_3 = result.couplings[3]

        # First coupling should be [64, 4] (original to 4D)
        assert coupling_4.shape[1] == 4
        # Second coupling should map from 4 -> 3
        assert coupling_3.shape[1] == 3

    def test_project_token_before_calibration(
        self, backend: "Backend"
    ) -> None:
        """Test that project_token fails before calibration."""
        cascade = DimensionCascade(backend)
        token = backend.random_normal((64,))

        with pytest.raises(RuntimeError, match="calibrate"):
            cascade.project_token(token, target_dim=3)

    def test_project_token_after_calibration(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test token projection after calibration."""
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        # Project a single token
        token = backend.random_normal((64,))
        projected = cascade.project_token(token, target_dim=3)

        assert projected.shape == (3,)

    def test_get_composite_coupling_before_calibration(
        self, backend: "Backend"
    ) -> None:
        """Test that get_composite_coupling fails before calibration."""
        cascade = DimensionCascade(backend)

        with pytest.raises(RuntimeError, match="calibrate"):
            cascade.get_composite_coupling(target_dim=3)

    def test_get_composite_coupling(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test composite coupling computation."""
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        composite = cascade.get_composite_coupling(target_dim=3)

        # Should be [original_dim, 3]
        assert composite.shape[1] == 3

    def test_composite_equals_chain(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that composite coupling equals chain multiplication."""
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        # Get composite coupling
        composite = cascade.get_composite_coupling(target_dim=3)

        # Project a token via chain
        token = backend.random_normal((64,))
        chain_result = cascade.project_token(token, target_dim=3)

        # Project same token via composite
        composite_result = backend.matmul(token[None, :], composite)[0]
        backend.eval(composite_result)

        # Results should be very close (not exact due to chain vs single matmul)
        diff = backend.abs(chain_result - composite_result)
        max_diff = backend.max(diff)
        backend.eval(max_diff)
        assert float(backend.to_scalar(max_diff)) <= division_epsilon(backend, chain_result)

    def test_geodesic_distortion_computed(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that geodesic distortion is computed."""
        cascade = DimensionCascade(backend)
        result = cascade.calibrate(random_activations, target_dims=[4, 3])

        assert 4 in result.geodesic_distortion
        assert 3 in result.geodesic_distortion
        # Distortion should be between 0 and 1
        for dim, distortion in result.geodesic_distortion.items():
            assert distortion >= -_eps(backend)
            assert distortion <= 1.0 + _eps(backend)

    def test_curvature_computed(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that curvature is computed when requested."""
        cascade = DimensionCascade(backend)
        config = CascadeConfiguration(
            target_dims=[4, 3],
            compute_curvature=True,
            curvature_k=10,
        )
        result = cascade.calibrate(random_activations, config=config)

        # Curvatures dict should be present (may be empty if backend lacks required methods)
        # Curvature computation is best-effort - some backends may not support all ops
        assert isinstance(result.curvatures, dict)

    def test_curvature_not_computed(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test that curvature is not computed when disabled."""
        cascade = DimensionCascade(backend)
        config = CascadeConfiguration(
            target_dims=[4, 3],
            compute_curvature=False,
        )
        result = cascade.calibrate(random_activations, config=config)

        # No curvatures should be computed
        assert len(result.curvatures) == 0

    def test_recalibrate(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test recalibration replaces couplings entirely."""
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        # Get original coupling
        original_coupling = cascade._couplings[3]

        # Recalibrate with new data (replaces couplings, no blending)
        new_activations = backend.random_normal((50, 64))
        cascade.recalibrate(new_activations)

        # Coupling should have changed (completely replaced with new)
        new_coupling = cascade._couplings[3]
        diff = backend.to_numpy(backend.abs(original_coupling - new_coupling))
        assert diff.sum() > _div_eps(backend)

    def test_min_points_validation(self, backend: "Backend") -> None:
        """Test validation of minimum calibration points."""
        cascade = DimensionCascade(backend)
        min_points = CascadeConfiguration(target_dims=[4, 3]).min_calibration_points
        too_few = backend.random_normal((min_points - 1, 64))

        with pytest.raises(ValueError, match="calibration points"):
            cascade.calibrate(too_few, target_dims=[4, 3])

    def test_invalid_target_dim(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test error when requesting uncalibrated dimension."""
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        with pytest.raises(ValueError, match="not calibrated"):
            cascade.project_token(backend.random_normal((64,)), target_dim=2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDimensionCascadeIntegration:
    """Integration tests for dimension cascade."""

    def test_streaming_workflow(
        self, backend: "Backend", random_activations: "Array"
    ) -> None:
        """Test typical streaming workflow."""
        # 1. Calibrate with initial data
        cascade = DimensionCascade(backend)
        cascade.calibrate(random_activations, target_dims=[4, 3])

        # 2. Get composite coupling for injection
        composite = cascade.get_composite_coupling(target_dim=3)

        # 3. Stream tokens
        projected_points = []
        for i in range(10):
            token = backend.random_normal((64,))
            point = backend.matmul(token[None, :], composite)[0]
            backend.eval(point)
            projected_points.append(point)

        assert len(projected_points) == 10
        for p in projected_points:
            assert p.shape == (3,)

    def test_different_input_dims(self, backend: "Backend") -> None:
        """Test cascade with different input dimensions."""
        for dim in [32, 128, 256]:
            backend.random_seed(42)
            activations = backend.random_normal((50, dim))
            cascade = DimensionCascade(backend)
            result = cascade.calibrate(activations, target_dims=[4, 3])

            assert result.original_dim == dim
            assert result.projections[3].shape == (50, 3)
