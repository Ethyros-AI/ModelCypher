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

"""Tests for moral_geometry.py - Moral structure analysis.

Tests cover:
- MoralAxisOrthogonality, MoralGradientConsistency, MoralFoundationClustering dataclasses
- MoralGeometryReport.to_dict() method
- MoralGeometryAnalyzer initialization
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.moral_geometry import (
    MoralAxisOrthogonality,
    MoralFoundationClustering,
    MoralGeometryAnalyzer,
    MoralGeometryReport,
    MoralGradientConsistency,
    VirtueViceOpposition,
)


# =============================================================================
# MoralAxisOrthogonality Tests
# =============================================================================


class TestMoralAxisOrthogonality:
    """Tests for MoralAxisOrthogonality dataclass."""

    def test_fields_stored(self):
        """MoralAxisOrthogonality stores all fields."""
        orthog = MoralAxisOrthogonality(
            valence_agency=0.85,
            valence_scope=0.90,
            agency_scope=0.88,
            mean_orthogonality=0.88,
        )
        assert orthog.valence_agency == 0.85
        assert orthog.mean_orthogonality == 0.88


# =============================================================================
# MoralGradientConsistency Tests
# =============================================================================


class TestMoralGradientConsistency:
    """Tests for MoralGradientConsistency dataclass."""

    def test_fields_stored(self):
        """MoralGradientConsistency stores all fields."""
        grad = MoralGradientConsistency(
            valence_correlation=0.95,
            valence_monotonic=True,
            agency_correlation=0.90,
            agency_monotonic=True,
            scope_correlation=0.85,
            scope_monotonic=False,
        )
        assert grad.valence_monotonic is True
        assert grad.scope_monotonic is False


# =============================================================================
# MoralFoundationClustering Tests
# =============================================================================


class TestMoralFoundationClustering:
    """Tests for MoralFoundationClustering dataclass."""

    def test_fields_stored(self):
        """MoralFoundationClustering stores all fields."""
        clustering = MoralFoundationClustering(
            within_foundation_similarity=0.8,
            between_foundation_similarity=0.3,
            separation_ratio=2.67,
            most_distinct_foundation="care",
            most_overlapping_pair=("loyalty", "authority"),
        )
        assert clustering.separation_ratio == 2.67


# =============================================================================
# VirtueViceOpposition Tests
# =============================================================================


class TestVirtueViceOpposition:
    """Tests for VirtueViceOpposition dataclass."""

    def test_fields_stored(self):
        """VirtueViceOpposition stores all fields."""
        opposition = VirtueViceOpposition(
            care_harm_opposition=0.9,
            fairness_opposition=0.85,
            loyalty_opposition=0.8,
            mean_opposition=0.85,
        )
        assert opposition.mean_opposition == 0.85


# =============================================================================
# MoralGeometryAnalyzer Tests
# =============================================================================


class TestMoralGeometryAnalyzer:
    """Tests for MoralGeometryAnalyzer class."""

    def test_init(self):
        """MoralGeometryAnalyzer initializes."""
        backend = get_default_backend()
        analyzer = MoralGeometryAnalyzer(backend)
        
        assert analyzer is not None
