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

"""Tests for manifold_transfer.py - Cross-manifold projection via anchors.

Tests cover:
- AnchorDistanceProfile dataclass
- TransferConfidenceComponents dataclass
- CrossManifoldProjector.compute_distance_profile() method
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    CrossManifoldProjector,
    TransferConfidenceComponents,
)


# =============================================================================
# TransferConfidenceComponents Tests
# =============================================================================


class TestTransferConfidenceComponents:
    """Tests for TransferConfidenceComponents dataclass."""

    def test_fields_stored(self):
        """TransferConfidenceComponents stores all fields."""
        conf = TransferConfidenceComponents(
            stress_factor=0.9,
            anchor_factor=0.8,
            curvature_factor=0.7,
        )
        assert conf.stress_factor == 0.9
        assert conf.anchor_factor == 0.8
        assert conf.curvature_factor == 0.7


# =============================================================================
# AnchorDistanceProfile Tests
# =============================================================================


class TestAnchorDistanceProfile:
    """Tests for AnchorDistanceProfile dataclass."""

    def test_fields_stored(self):
        """AnchorDistanceProfile stores all fields."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="test_concept",
            anchor_ids=["a1", "a2", "a3"],
            distances=backend.array([0.5, 0.8, 0.3]),
            weights=backend.array([1.0, 1.0, 1.0]),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.concept_id == "test_concept"
        assert len(profile.anchor_ids) == 3

    def test_num_anchors(self):
        """num_anchors returns correct count."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="c",
            anchor_ids=["a", "b"],
            distances=backend.array([1.0, 2.0]),
            weights=backend.array([1.0, 1.0]),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.num_anchors == 2

    def test_distance_to(self):
        """distance_to returns distance for specific anchor."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="c",
            anchor_ids=["anchor_1", "anchor_2"],
            distances=backend.array([0.75, 1.25]),
            weights=backend.array([1.0, 1.0]),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.distance_to("anchor_1") == 0.75


# =============================================================================
# CrossManifoldProjector Tests
# =============================================================================


class TestCrossManifoldProjector:
    """Tests for CrossManifoldProjector class."""

    def test_init(self):
        """CrossManifoldProjector initializes."""
        projector = CrossManifoldProjector()
        assert projector is not None

    def test_compute_distance_profile_returns_profile(self):
        """compute_distance_profile returns AnchorDistanceProfile."""
        backend = get_default_backend()
        projector = CrossManifoldProjector()
        
        # Need at least 3 anchors for triangulation
        concept = backend.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        anchors = {
            "anchor_a": backend.array([[0.0, 0.0], [0.1, 0.1]]),
            "anchor_b": backend.array([[1.0, 1.0], [0.9, 0.9]]),
            "anchor_c": backend.array([[0.5, 0.0], [0.5, 0.1]]),
        }
        
        profile = projector.compute_distance_profile(
            concept_activations=concept,
            concept_id="test",
            anchor_activations=anchors,
        )
        
        assert isinstance(profile, AnchorDistanceProfile)
        assert profile.concept_id == "test"
