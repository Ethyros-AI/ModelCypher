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

"""Tests for social_geometry.py - Social geometry analysis.

Tests cover:
- AxisOrthogonality, GradientConsistency, SocialGeometryReport dataclasses
- SocialGeometryAnalyzer class methods
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.social_geometry import (
    AxisOrthogonality,
    GradientConsistency,
    PowerGradientResult,
    SocialGeometryAnalyzer,
    SocialGeometryReport,
)


# =============================================================================
# AxisOrthogonality Tests
# =============================================================================


class TestAxisOrthogonality:
    """Tests for AxisOrthogonality dataclass."""

    def test_fields_stored(self):
        """AxisOrthogonality stores all fields."""
        orthog = AxisOrthogonality(
            power_kinship=0.9,
            power_formality=0.85,
            kinship_formality=0.88,
            mean_orthogonality=0.88,
        )
        assert orthog.power_kinship == 0.9
        assert orthog.mean_orthogonality == 0.88


# =============================================================================
# GradientConsistency Tests
# =============================================================================


class TestGradientConsistency:
    """Tests for GradientConsistency dataclass."""

    def test_fields_stored(self):
        """GradientConsistency stores all fields."""
        grad = GradientConsistency(
            power_monotonic=True,
            power_correlation=0.95,
            kinship_monotonic=True,
            kinship_correlation=0.92,
            formality_monotonic=False,
            formality_correlation=0.78,
        )
        assert grad.power_monotonic is True
        assert grad.formality_monotonic is False


# =============================================================================
# SocialGeometryReport Tests
# =============================================================================


class TestSocialGeometryReport:
    """Tests for SocialGeometryReport dataclass."""

    def test_to_dict(self):
        """to_dict returns dictionary."""
        backend = get_default_backend()
        orthog = AxisOrthogonality(0.9, 0.85, 0.88, 0.88)
        grad = GradientConsistency(True, 0.9, True, 0.9, True, 0.9)
        power = PowerGradientResult(
            power_axis_detected=True,
            power_direction=backend.array([1.0, 0.0]),
            status_correlation=0.95,
            high_status_anchors=("CEO", "Director"),
            low_status_anchors=("Intern",),
        )
        report = SocialGeometryReport(
            social_manifold_score=0.9,
            axis_orthogonality=orthog,
            gradient_consistency=grad,
            power_gradient=power,
            principal_components_variance=(0.4, 0.3, 0.2),
            anchor_count=10,
        )
        
        d = report.to_dict()
        
        assert isinstance(d, dict)
        assert "social_manifold_score" in d


# =============================================================================
# SocialGeometryAnalyzer Tests
# =============================================================================


class TestSocialGeometryAnalyzer:
    """Tests for SocialGeometryAnalyzer class."""

    def test_init(self):
        """SocialGeometryAnalyzer initializes."""
        backend = get_default_backend()
        analyzer = SocialGeometryAnalyzer(backend)
        
        assert analyzer is not None
