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

"""Tests for safety_polytope.py - Transformation determination for merging.

Tests cover:
- TransformationType enum
- DiagnosticVector dataclass and methods
- PolytopeBounds.from_baseline_metrics() class method
- SafetyPolytope.analyze_layer() and analyze_model_pair() methods
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.safety_polytope import (
    DiagnosticVector,
    LayerTransformationResult,
    ModelTransformationProfile,
    PolytopeBounds,
    SafetyPolytope,
    TransformationType,
)


# =============================================================================
# TransformationType Tests
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_enum_values(self):
        """TransformationType has expected values."""
        assert TransformationType.GEODESIC_NULL_SPACE.value == "geodesic_null_space"
        assert TransformationType.SPECTRAL_CLAMP.value == "spectral_clamp"
        assert TransformationType.LAYER_SKIP.value == "layer_skip"


# =============================================================================
# DiagnosticVector Tests
# =============================================================================


class TestDiagnosticVector:
    """Tests for DiagnosticVector dataclass."""

    def test_fields_stored(self):
        """DiagnosticVector stores all fields."""
        dv = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.5,
            instability_score=0.2,
            complexity_score=0.4,
        )
        assert dv.interference_score == 0.3
        assert dv.importance_score == 0.5

    def test_vector_returns_list(self):
        """vector returns 4D list."""
        dv = DiagnosticVector(0.1, 0.2, 0.3, 0.4)
        vec = dv.vector
        
        assert len(vec) == 4
        assert vec[0] == 0.1

    def test_magnitude(self):
        """magnitude returns geodesic norm."""
        dv = DiagnosticVector(0.0, 0.0, 0.0, 0.0)
        assert dv.magnitude == 0.0

    def test_max_dimension(self):
        """max_dimension returns dimension name."""
        dv = DiagnosticVector(0.1, 0.5, 0.2, 0.3)
        max_dim = dv.max_dimension
        
        assert max_dim == "importance"


# =============================================================================
# PolytopeBounds Tests
# =============================================================================


class TestPolytopeBounds:
    """Tests for PolytopeBounds dataclass."""

    def test_from_baseline_metrics(self):
        """from_baseline_metrics derives thresholds."""
        bounds = PolytopeBounds.from_baseline_metrics(
            interference_samples=[0.1, 0.2, 0.3, 0.4, 0.5],
            instability_samples=[0.05, 0.1, 0.15, 0.2, 0.25],
            complexity_samples=[0.2, 0.3, 0.4, 0.5, 0.6],
            magnitude_samples=[0.5, 0.6, 0.7, 0.8, 0.9],
        )
        
        assert isinstance(bounds, PolytopeBounds)
        assert bounds.interference_threshold > 0


# =============================================================================
# SafetyPolytope Tests
# =============================================================================


class TestSafetyPolytope:
    """Tests for SafetyPolytope class."""

    def test_init(self):
        """SafetyPolytope initializes with bounds."""
        bounds = PolytopeBounds(
            interference_threshold=0.5,
            importance_threshold=0.5,
            instability_threshold=0.5,
            complexity_threshold=0.5,
            magnitude_threshold=0.5,
            high_instability_threshold=0.9,
            high_interference_threshold=0.9,
        )
        polytope = SafetyPolytope(bounds)
        
        assert polytope is not None

    def test_analyze_layer_returns_result(self):
        """analyze_layer returns LayerTransformationResult."""
        bounds = PolytopeBounds(0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9)
        polytope = SafetyPolytope(bounds)
        diag = DiagnosticVector(0.1, 0.2, 0.1, 0.2)
        
        result = polytope.analyze_layer(diag, layer=0)
        
        assert isinstance(result, LayerTransformationResult)
        assert result.layer == 0

    def test_analyze_model_pair_returns_profile(self):
        """analyze_model_pair returns ModelTransformationProfile."""
        bounds = PolytopeBounds(0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9)
        polytope = SafetyPolytope(bounds)
        
        layer_diags = {
            0: DiagnosticVector(0.1, 0.1, 0.1, 0.1),
            1: DiagnosticVector(0.2, 0.2, 0.2, 0.2),
        }
        
        profile = polytope.analyze_model_pair(layer_diags)
        
        assert isinstance(profile, ModelTransformationProfile)
        assert len(profile.per_layer) == 2
