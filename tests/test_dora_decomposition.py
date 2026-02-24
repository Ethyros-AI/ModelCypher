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

"""Tests for dora_decomposition.py - DoRA magnitude/direction decomposition.

Tests cover:
- ChangeType enum
- MagnitudeDirectionMetrics, DecompositionResult dataclasses
- DoRADecomposition.decompose() method
- DoRADecomposition.analyze_adapter() method
- to_metrics_dict() function
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.dora_decomposition import (
    ChangeType,
    DecompositionResult,
    DoRADecomposition,
    DoRAMetricKey,
    MagnitudeDirectionMetrics,
    to_metrics_dict,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

# =============================================================================
# ChangeType Tests
# =============================================================================


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_enum_values(self):
        """ChangeType has expected values."""
        assert ChangeType.MAGNITUDE_DOMINATED.value == "magnitude_dominated"
        assert ChangeType.DIRECTION_DOMINATED.value == "direction_dominated"
        assert ChangeType.BALANCED.value == "balanced"
        assert ChangeType.MINIMAL.value == "minimal"


# =============================================================================
# MagnitudeDirectionMetrics Tests
# =============================================================================


class TestMagnitudeDirectionMetrics:
    """Tests for MagnitudeDirectionMetrics dataclass."""

    def test_fields_stored(self):
        """MagnitudeDirectionMetrics stores all fields."""
        metrics = MagnitudeDirectionMetrics(
            layer_name="layer_0",
            base_magnitude=1.0,
            current_magnitude=1.2,
            magnitude_ratio=1.2,
            direction_cosine=0.98,
            directional_drift=0.02,
            absolute_magnitude_change=0.2,
            relative_magnitude_change=0.2,
        )
        assert metrics.layer_name == "layer_0"
        assert metrics.magnitude_ratio == 1.2
        assert metrics.direction_cosine == 0.98


# =============================================================================
# DoRADecomposition.decompose Tests
# =============================================================================


class TestDoRADecompose:
    """Tests for DoRADecomposition.decompose method."""

    def test_decompose_identical_weights(self):
        """decompose with identical weights shows minimal change."""
        backend = get_default_backend()
        weight = backend.array([[1.0, 0.0], [0.0, 1.0]])

        dora = DoRADecomposition(backend)
        result = dora.decompose(weight, weight, "test_layer")

        assert result is not None
        eps = division_epsilon(backend, weight)
        assert abs(result.direction_cosine - 1.0) <= eps
        assert abs(result.relative_magnitude_change) <= eps

    def test_decompose_scaled_weights(self):
        """decompose with scaled weights detects magnitude change."""
        backend = get_default_backend()
        base = backend.array([[1.0, 0.0], [0.0, 1.0]])
        scaled = backend.array([[2.0, 0.0], [0.0, 2.0]])

        dora = DoRADecomposition(backend)
        result = dora.decompose(base, scaled, "scaled_layer")

        assert result is not None
        from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

        base_mag = geodesic_norms(backend.reshape(base, (1, -1)), backend)
        scaled_mag = geodesic_norms(backend.reshape(scaled, (1, -1)), backend)
        backend.eval(base_mag, scaled_mag)
        expected_ratio = float(backend.to_scalar(scaled_mag)) / float(
            backend.to_scalar(base_mag)
        )
        eps = division_epsilon(backend, base)
        assert result.magnitude_ratio == pytest.approx(expected_ratio, rel=eps)

    def test_decompose_returns_layer_name(self):
        """decompose preserves layer name."""
        backend = get_default_backend()
        weight = backend.array([[1.0, 2.0]])

        dora = DoRADecomposition(backend)
        result = dora.decompose(weight, weight, "my_layer")

        assert result.layer_name == "my_layer"


# =============================================================================
# DoRADecomposition.analyze_adapter Tests
# =============================================================================


class TestDoRAAnalyzeAdapter:
    """Tests for DoRADecomposition.analyze_adapter method."""

    def test_analyze_adapter_returns_result(self):
        """analyze_adapter returns DecompositionResult."""
        backend = get_default_backend()
        base_weights = {
            "layer_0": backend.array([[1.0, 0.0], [0.0, 1.0]]),
            "layer_1": backend.array([[0.5, 0.5]]),
        }
        current_weights = {
            "layer_0": backend.array([[1.1, 0.0], [0.0, 1.1]]),
            "layer_1": backend.array([[0.6, 0.4]]),
        }

        dora = DoRADecomposition(backend)
        result = dora.analyze_adapter(base_weights, current_weights)

        assert isinstance(result, DecompositionResult)
        assert "layer_0" in result.per_layer_metrics

    def test_analyze_adapter_empty(self):
        """analyze_adapter handles empty weights."""
        backend = get_default_backend()

        dora = DoRADecomposition(backend)
        result = dora.analyze_adapter({}, {})

        assert isinstance(result, DecompositionResult)
        assert result.dominant_change_type == ChangeType.MINIMAL


# =============================================================================
# to_metrics_dict Tests
# =============================================================================


class TestToMetricsDict:
    """Tests for to_metrics_dict function."""

    def test_to_metrics_dict_keys(self):
        """to_metrics_dict returns expected keys."""
        backend = get_default_backend()
        base = {"layer": backend.array([[1.0]])}
        current = {"layer": backend.array([[1.1]])}

        dora = DoRADecomposition(backend)
        result = dora.analyze_adapter(base, current)
        metrics = to_metrics_dict(result)

        assert DoRAMetricKey.MAGNITUDE_CHANGE in metrics
        assert DoRAMetricKey.DIRECTIONAL_DRIFT in metrics
