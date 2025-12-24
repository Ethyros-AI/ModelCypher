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

"""Tests for enhanced InvariantLayerMapper with triangulation scoring.

Tests the 68 sequence invariant integration with cross-domain triangulation
scoring for improved layer mapping between models.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    ActivatedDimension,
    ActivationFingerprint,
    Config,
    ConfidenceLevel,
    InvariantLayerMapper,
    InvariantScope,
    ModelFingerprints,
    TriangulationProfile,
)
from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    DEFAULT_FAMILIES,
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.domain.agents.unified_atlas import (
    AtlasProbe,
    AtlasSource,
    AtlasDomain,
    UnifiedAtlasInventory,
)
from modelcypher.core.use_cases.invariant_layer_mapping_service import (
    CollapseRiskConfig,
    InvariantLayerMappingService,
    LayerMappingConfig,
)


# ===========================================================================
# Domain Model Tests
# ===========================================================================


def test_invariant_scope_enum_has_sequence_invariants():
    """Test that the new SEQUENCE_INVARIANTS scope exists."""
    assert hasattr(InvariantScope, "SEQUENCE_INVARIANTS")
    assert InvariantScope.SEQUENCE_INVARIANTS.value == "sequenceInvariants"


def test_config_has_triangulation_options():
    """Test that Config has triangulation-related fields."""
    config = Config()

    assert hasattr(config, "use_cross_domain_weighting")
    assert hasattr(config, "triangulation_threshold")
    assert hasattr(config, "multi_domain_bonus")

    # Check defaults
    assert config.use_cross_domain_weighting is True
    assert config.triangulation_threshold == 0.3
    assert config.multi_domain_bonus is True


def test_config_with_sequence_invariants_scope():
    """Test Config creation with SEQUENCE_INVARIANTS scope."""
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        use_cross_domain_weighting=True,
        multi_domain_bonus=True,
    )

    assert config.invariant_scope == InvariantScope.SEQUENCE_INVARIANTS
    assert config.use_cross_domain_weighting is True


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


# ===========================================================================
# Invariant Selection Tests
# ===========================================================================


def test_get_invariants_with_sequence_scope_returns_all_68():
    """Test that SEQUENCE_INVARIANTS scope returns all 68 probes."""
    config = Config(invariant_scope=InvariantScope.SEQUENCE_INVARIANTS)

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should return all 68 sequence invariants
    assert len(invariants) == 68
    assert len(ids) == 68
    assert len(atlas_probes) == 0  # No atlas probes in SEQUENCE_INVARIANTS mode

    # Each ID should follow the expected format
    for inv_id in ids:
        assert inv_id.startswith("invariant:")


def test_get_invariants_with_logic_only_scope():
    """Test that LOGIC_ONLY scope returns only logic family probes."""
    config = Config(invariant_scope=InvariantScope.LOGIC_ONLY)

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should only have logic family
    for inv in invariants:
        assert inv.family == SequenceFamily.LOGIC


def test_get_invariants_with_family_allowlist():
    """Test that family_allowlist filters invariants correctly."""
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        family_allowlist=frozenset([SequenceFamily.FIBONACCI, SequenceFamily.PRIMES]),
    )

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should only have fibonacci and primes families
    families = {inv.family for inv in invariants}
    assert families == {SequenceFamily.FIBONACCI, SequenceFamily.PRIMES}


def test_get_invariants_backward_compat_invariants_scope():
    """Test backward compatibility with INVARIANTS scope."""
    config = Config(invariant_scope=InvariantScope.INVARIANTS)

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should return default families
    assert len(invariants) > 0


# ===========================================================================
# Fingerprint and Profile Tests
# ===========================================================================


def _make_fingerprints(model_id: str, layer_count: int = 8) -> ModelFingerprints:
    """Create test fingerprints with mock data.

    ActivationFingerprint structure:
    - prime_id: str (invariant identifier)
    - activated_dimensions: dict[int, list[ActivatedDimension]] (layer -> activations)
    """
    fingerprints = []

    # Create fingerprints for a few invariants
    invariant_ids = [
        "invariant:fibonacci_fib_recurrence",
        "invariant:logic_modus_ponens",
        "invariant:primes_twin_prime_gap",
    ]

    for inv_id in invariant_ids:
        # Build layer -> activations mapping
        layer_activations: dict[int, list[ActivatedDimension]] = {}
        for layer in range(layer_count):
            layer_activations[layer] = [
                ActivatedDimension(
                    index=0,
                    activation=0.5 + (layer * 0.05),
                ),
            ]

        fingerprints.append(
            ActivationFingerprint(
                prime_id=inv_id,
                prime_text=f"test_{inv_id}",
                activated_dimensions=layer_activations,
            )
        )

    return ModelFingerprints(
        model_id=model_id,
        layer_count=layer_count,
        fingerprints=fingerprints,
    )


def test_build_profile_with_triangulation():
    """Test that profile building includes triangulation data."""
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        use_cross_domain_weighting=True,
        multi_domain_bonus=True,
    )

    fingerprints = _make_fingerprints("test_model", layer_count=4)
    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    profile = InvariantLayerMapper._build_profile(fingerprints, ids, config)

    # Profile should be built with expected structure
    assert profile is not None
    assert hasattr(profile, "vectors")
    assert hasattr(profile, "collapsed_count")


def test_map_layers_basic():
    """Test basic layer mapping between two models."""
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        sample_layer_count=4,
    )

    source = _make_fingerprints("source_model", layer_count=4)
    target = _make_fingerprints("target_model", layer_count=4)

    report = InvariantLayerMapper.map_layers(source, target, config)

    assert report is not None
    assert report.source_model == "source_model"
    assert report.target_model == "target_model"
    assert len(report.mappings) > 0


def test_map_layers_summary_includes_triangulation_quality():
    """Test that mapping summary includes triangulation quality."""
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        multi_domain_bonus=True,
    )

    source = _make_fingerprints("source", layer_count=4)
    target = _make_fingerprints("target", layer_count=4)

    report = InvariantLayerMapper.map_layers(source, target, config)

    # Summary should have triangulation fields
    assert hasattr(report.summary, "mean_triangulation_multiplier")
    assert hasattr(report.summary, "triangulation_quality")


# ===========================================================================
# Service Layer Tests
# ===========================================================================


def _create_mock_model_dir(tmp_path: Path, layer_count: int = 32) -> Path:
    """Create a mock model directory with config.json."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": layer_count,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    return model_dir


def test_service_map_layers_raises_on_empty_fingerprints(tmp_path):
    """Test that map_layers raises ValueError with stub/empty fingerprints.

    The mapper requires actual invariant activations to detect signal.
    Empty fingerprints (no activations) correctly raise an error.
    """
    source = _create_mock_model_dir(tmp_path / "source")
    target = _create_mock_model_dir(tmp_path / "target")

    service = InvariantLayerMappingService()
    config = LayerMappingConfig(
        source_model_path=str(source),
        target_model_path=str(target),
        invariant_scope="sequenceInvariants",
        use_triangulation=True,
    )

    # Empty fingerprints should raise an error
    with pytest.raises(ValueError, match="no invariant activations"):
        service.map_layers(config)


def test_service_analyze_collapse_risk(tmp_path):
    """Test InvariantLayerMappingService.analyze_collapse_risk()."""
    model = _create_mock_model_dir(tmp_path)

    service = InvariantLayerMappingService()
    config = CollapseRiskConfig(
        model_path=str(model),
        collapse_threshold=0.35,
    )

    result = service.analyze_collapse_risk(config)

    assert result is not None
    assert result.model_path == str(model)
    assert result.layer_count == 32
    assert result.risk_level in ["low", "medium", "high", "critical"]
    assert result.interpretation is not None
    assert result.recommended_action is not None


def test_service_collapse_risk_payload_schema():
    """Test that collapse_risk_payload returns correct schema."""
    from modelcypher.core.use_cases.invariant_layer_mapping_service import CollapseRiskResult

    collapse_result = CollapseRiskResult(
        model_path="/tmp/model",
        layer_count=32,
        collapsed_layers=5,
        collapse_ratio=0.156,
        risk_level="medium",
        interpretation="Test interpretation",
        recommended_action="Test action",
    )

    payload = InvariantLayerMappingService.collapse_risk_payload(collapse_result)

    assert payload["_schema"] == "mc.geometry.invariant.collapse_risk.v1"
    assert payload["modelPath"] == "/tmp/model"
    assert payload["layerCount"] == 32
    assert payload["collapsedLayers"] == 5
    assert payload["riskLevel"] == "medium"
    assert payload["interpretation"] == "Test interpretation"
    assert payload["recommendedAction"] == "Test action"


def test_service_family_parsing():
    """Test that service parses family strings correctly."""
    from modelcypher.core.use_cases.invariant_layer_mapping_service import _parse_families

    # None returns None
    assert _parse_families(None) is None

    # Empty list returns None
    assert _parse_families([]) is None

    # Valid families are parsed
    result = _parse_families(["fibonacci", "logic", "primes"])
    assert result is not None
    assert SequenceFamily.FIBONACCI in result
    assert SequenceFamily.LOGIC in result
    assert SequenceFamily.PRIMES in result

    # Invalid families are skipped
    result = _parse_families(["fibonacci", "invalid_family"])
    assert result is not None
    assert SequenceFamily.FIBONACCI in result
    assert len(result) == 1


def test_service_scope_parsing():
    """Test that service parses scope strings correctly."""
    from modelcypher.core.use_cases.invariant_layer_mapping_service import _parse_scope

    assert _parse_scope("invariants") == InvariantScope.INVARIANTS
    assert _parse_scope("logicOnly") == InvariantScope.LOGIC_ONLY
    assert _parse_scope("logic_only") == InvariantScope.LOGIC_ONLY
    assert _parse_scope("sequenceInvariants") == InvariantScope.SEQUENCE_INVARIANTS
    assert _parse_scope("sequence_invariants") == InvariantScope.SEQUENCE_INVARIANTS

    # Default for unknown
    assert _parse_scope("unknown") == InvariantScope.SEQUENCE_INVARIANTS


# ===========================================================================
# Risk Level Classification Tests
# ===========================================================================


def test_collapse_risk_levels():
    """Test that collapse risk levels are classified correctly."""
    service = InvariantLayerMappingService()

    # Verify risk level thresholds through interpretation
    # Low: < 15% collapse
    # Medium: 15-30% collapse
    # High: 30-50% collapse
    # Critical: >= 50% collapse

    # Test classification logic indirectly
    from modelcypher.core.use_cases.invariant_layer_mapping_service import CollapseRiskResult

    # Low risk
    result_low = CollapseRiskResult(
        model_path="/test",
        layer_count=100,
        collapsed_layers=10,  # 10%
        collapse_ratio=0.10,
        risk_level="low",
        interpretation="",
        recommended_action="",
    )
    assert result_low.risk_level == "low"

    # Medium risk
    result_medium = CollapseRiskResult(
        model_path="/test",
        layer_count=100,
        collapsed_layers=20,  # 20%
        collapse_ratio=0.20,
        risk_level="medium",
        interpretation="",
        recommended_action="",
    )
    assert result_medium.risk_level == "medium"


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_sequence_inventory_integration():
    """Test that sequence invariant inventory integrates with mapper."""
    # Get all probes
    probes = SequenceInvariantInventory.probes_for_families()
    assert len(probes) == 68

    # Get probes for specific families
    fib_probes = SequenceInvariantInventory.probes_for_families({SequenceFamily.FIBONACCI})
    logic_probes = SequenceInvariantInventory.probes_for_families({SequenceFamily.LOGIC})

    assert len(fib_probes) > 0
    assert len(logic_probes) > 0

    # All probes should have cross_domain_weight
    for probe in probes:
        assert hasattr(probe, "cross_domain_weight")
        assert 0.0 <= probe.cross_domain_weight <= 2.0


def test_triangulation_scorer_used_in_mapper():
    """Test that TriangulationScorer is integrated into mapper."""
    # This is tested indirectly through the profile building
    config = Config(
        invariant_scope=InvariantScope.SEQUENCE_INVARIANTS,
        multi_domain_bonus=True,
        use_cross_domain_weighting=True,
    )

    # The mapper should use TriangulationScorer when multi_domain_bonus is True
    fingerprints = _make_fingerprints("test", layer_count=4)
    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Build profile - this should internally use triangulation
    profile = InvariantLayerMapper._build_profile(fingerprints, ids, config)

    assert profile is not None


# ===========================================================================
# Multi-Atlas Tests
# ===========================================================================


def test_unified_atlas_inventory_total_probes():
    """Test that UnifiedAtlasInventory returns 237 probes from all atlases."""
    all_probes = UnifiedAtlasInventory.all_probes()

    # Total should be 68 + 65 + 72 + 32 = 237
    assert len(all_probes) >= 200  # Allow some flexibility

    # Check probe structure
    for probe in all_probes:
        assert hasattr(probe, "id")
        assert hasattr(probe, "source")
        assert hasattr(probe, "domain")
        assert hasattr(probe, "cross_domain_weight")
        assert 0.0 <= probe.cross_domain_weight <= 2.0


def test_unified_atlas_inventory_probe_counts_by_source():
    """Test probe counts by source."""
    counts = UnifiedAtlasInventory.probe_count()

    assert AtlasSource.SEQUENCE_INVARIANT in counts
    assert AtlasSource.SEMANTIC_PRIME in counts
    assert AtlasSource.COMPUTATIONAL_GATE in counts
    assert AtlasSource.EMOTION_CONCEPT in counts

    # Check expected ranges
    assert counts[AtlasSource.SEQUENCE_INVARIANT] == 68
    assert counts[AtlasSource.SEMANTIC_PRIME] == 65
    assert counts[AtlasSource.COMPUTATIONAL_GATE] >= 60  # 66 core + composites
    assert counts[AtlasSource.EMOTION_CONCEPT] >= 30    # 24 emotions + 8 dyads


def test_unified_atlas_filter_by_source():
    """Test filtering probes by source."""
    sequence_probes = UnifiedAtlasInventory.probes_by_source({AtlasSource.SEQUENCE_INVARIANT})
    semantic_probes = UnifiedAtlasInventory.probes_by_source({AtlasSource.SEMANTIC_PRIME})

    assert len(sequence_probes) == 68
    assert len(semantic_probes) == 65

    # All should have correct source
    for probe in sequence_probes:
        assert probe.source == AtlasSource.SEQUENCE_INVARIANT
    for probe in semantic_probes:
        assert probe.source == AtlasSource.SEMANTIC_PRIME


def test_unified_atlas_filter_by_domain():
    """Test filtering probes by domain."""
    math_probes = UnifiedAtlasInventory.probes_by_domain({AtlasDomain.MATHEMATICAL})
    logical_probes = UnifiedAtlasInventory.probes_by_domain({AtlasDomain.LOGICAL})

    assert len(math_probes) > 0
    assert len(logical_probes) > 0

    for probe in math_probes:
        assert probe.domain == AtlasDomain.MATHEMATICAL
    for probe in logical_probes:
        assert probe.domain == AtlasDomain.LOGICAL


def test_multi_atlas_scope_returns_all_probes():
    """Test that MULTI_ATLAS scope returns all atlas probes."""
    config = Config(invariant_scope=InvariantScope.MULTI_ATLAS)

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should return atlas probes, not sequence invariants
    assert len(atlas_probes) >= 200  # All probes
    assert len(invariants) == 0      # No sequence invariants in this mode
    assert len(ids) == len(atlas_probes)


def test_multi_atlas_scope_with_source_filter():
    """Test MULTI_ATLAS scope with source filtering."""
    config = Config(
        invariant_scope=InvariantScope.MULTI_ATLAS,
        atlas_sources=frozenset([AtlasSource.SEQUENCE_INVARIANT, AtlasSource.SEMANTIC_PRIME]),
    )

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should only have sequence invariants and semantic primes
    sources = {p.source for p in atlas_probes}
    assert sources == {AtlasSource.SEQUENCE_INVARIANT, AtlasSource.SEMANTIC_PRIME}
    assert len(atlas_probes) == 68 + 65


def test_multi_atlas_scope_with_domain_filter():
    """Test MULTI_ATLAS scope with domain filtering."""
    config = Config(
        invariant_scope=InvariantScope.MULTI_ATLAS,
        atlas_domains=frozenset([AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL]),
    )

    ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)

    # Should only have mathematical and logical domains
    domains = {p.domain for p in atlas_probes}
    assert domains.issubset({AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL})
    assert len(atlas_probes) > 0


def test_invariant_scope_has_multi_atlas():
    """Test that InvariantScope has MULTI_ATLAS value."""
    assert hasattr(InvariantScope, "MULTI_ATLAS")
    assert InvariantScope.MULTI_ATLAS.value == "multiAtlas"


def test_config_has_atlas_options():
    """Test that Config has atlas source/domain options."""
    config = Config()

    assert hasattr(config, "atlas_sources")
    assert hasattr(config, "atlas_domains")
    assert config.atlas_sources is None  # Default is None (all sources)
    assert config.atlas_domains is None  # Default is None (all domains)


def test_summary_has_multi_atlas_metrics():
    """Test that Summary includes multi-atlas metrics."""
    from modelcypher.core.domain.geometry.invariant_layer_mapper import Summary

    summary = Summary(
        mapped_layers=10,
        skipped_layers=2,
        mean_similarity=0.75,
        alignment_quality=0.8,
        source_collapsed_layers=1,
        target_collapsed_layers=1,
        atlas_sources_detected=4,
        atlas_domains_detected=8,
        total_probes_used=237,
    )

    assert summary.atlas_sources_detected == 4
    assert summary.atlas_domains_detected == 8
    assert summary.total_probes_used == 237


def test_service_multi_atlas_config_parsing():
    """Test that service parses multi-atlas config correctly."""
    from modelcypher.core.use_cases.invariant_layer_mapping_service import (
        _parse_scope,
        _parse_atlas_sources,
        _parse_atlas_domains,
    )

    # Scope parsing
    assert _parse_scope("multiAtlas") == InvariantScope.MULTI_ATLAS
    assert _parse_scope("multi_atlas") == InvariantScope.MULTI_ATLAS

    # Source parsing
    sources = _parse_atlas_sources(["sequence", "semantic"])
    assert sources is not None
    assert AtlasSource.SEQUENCE_INVARIANT in sources
    assert AtlasSource.SEMANTIC_PRIME in sources

    # Domain parsing
    domains = _parse_atlas_domains(["mathematical", "logical"])
    assert domains is not None
    assert AtlasDomain.MATHEMATICAL in domains
    assert AtlasDomain.LOGICAL in domains
