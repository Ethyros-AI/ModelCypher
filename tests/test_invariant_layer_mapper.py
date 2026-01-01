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

"""Tests for InvariantLayerMapper.

Tests the layer mapping between models using fingerprints and triangulation.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.agents.unified_atlas import (
    AtlasDomain,
    AtlasSource,
    UnifiedAtlasInventory,
)
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    InvariantLayerMapper,
    LayerMapping,
    LayerProfile,
    ModelFingerprints,
    Report,
    Summary,
    TriangulationProfile,
)


# ===========================================================================
# Domain Model Tests
# ===========================================================================


def test_triangulation_profile_dataclass():
    """Test TriangulationProfile dataclass structure."""
    profile = TriangulationProfile(
        layer_index=5,
        domains_detected=3,
        cross_domain_multiplier=1.5,
        coherence_bonus=0.2,
    )

    assert profile.layer_index == 5
    assert profile.domains_detected == 3
    assert profile.cross_domain_multiplier == 1.5
    assert profile.coherence_bonus == 0.2


def test_layer_profile_dataclass():
    """Test LayerProfile dataclass structure."""
    profile = LayerProfile(
        layer_index=3,
        confidence=0.9,
        coverage=0.8,
        strength=0.7,
        collapsed=False,
        triangulation=None,
    )

    assert profile.layer_index == 3
    assert profile.confidence == 0.9
    assert profile.collapsed is False


def test_layer_mapping_dataclass():
    """Test LayerMapping dataclass structure."""
    mapping = LayerMapping(
        source_layer=0,
        target_layer=1,
        similarity=0.95,
    )

    assert mapping.source_layer == 0
    assert mapping.target_layer == 1
    assert mapping.similarity == 0.95


def test_model_fingerprints_dataclass():
    """Test ModelFingerprints dataclass structure."""
    fingerprints = ModelFingerprints(
        model_id="test-model",
        layer_count=12,
        fingerprints={},
    )

    assert fingerprints.model_id == "test-model"
    assert fingerprints.layer_count == 12


def test_summary_dataclass():
    """Test Summary dataclass structure."""
    summary = Summary(
        mapped_layers=10,
        mean_similarity=0.85,
        alignment_quality=0.9,
        source_collapsed_layers=[],
        target_collapsed_layers=[],
        mean_triangulation_multiplier=1.0,
        atlas_sources_detected=["semantic_primes"],
        atlas_domains_detected=["logical"],
        total_probes_used=50,
    )

    assert summary.mean_similarity == 0.85
    assert summary.mapped_layers == 10


def test_report_dataclass():
    """Test Report dataclass has expected fields."""
    # Report requires many fields for full model comparison
    # Just verify the class is importable and has key fields
    assert hasattr(Report, "__dataclass_fields__")
    fields = Report.__dataclass_fields__
    assert "mappings" in fields
    assert "summary" in fields
    assert "source_model" in fields
    assert "target_model" in fields


# ===========================================================================
# Atlas Integration Tests
# ===========================================================================


def test_unified_atlas_inventory_total_probes():
    """Test UnifiedAtlasInventory returns total probe count."""
    inventory = UnifiedAtlasInventory()
    total = inventory.total_probe_count()

    assert total > 0


def test_unified_atlas_inventory_probe_counts_by_source():
    """Test UnifiedAtlasInventory breaks down probes by source."""
    inventory = UnifiedAtlasInventory()
    # Use SEQUENCE_INVARIANT (singular) - the correct enum value
    probes_by_source = inventory.probes_by_source(AtlasSource.SEQUENCE_INVARIANT)

    # Should return probes or empty list
    assert isinstance(probes_by_source, list)


def test_unified_atlas_filter_by_domain():
    """Test UnifiedAtlasInventory can filter by domain."""
    inventory = UnifiedAtlasInventory()

    # Filter should return list
    filtered = inventory.probes_by_domain(AtlasDomain.MATHEMATICAL)
    assert isinstance(filtered, list)
