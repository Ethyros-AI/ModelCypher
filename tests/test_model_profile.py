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

"""Comprehensive tests for ModelProfile unified schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.domain.geometry.model_profile import (
    DensitySummary,
    LayerProfile,
    ManifoldRegion,
    ModelProfile,
    ProfileSection,
    SCHEMA_VERSION,
    SemanticSignature,
    TopologySummary,
)


class TestProfileSection:
    """Tests for ProfileSection enum."""

    def test_all_sections_have_values(self) -> None:
        """All sections should have string values."""
        assert ProfileSection.IDENTITY.value == "identity"
        assert ProfileSection.GEOMETRY.value == "geometry"
        assert ProfileSection.TOPOLOGY.value == "topology"
        assert ProfileSection.SEMANTIC.value == "semantic"
        assert ProfileSection.DENSITY.value == "density"
        assert ProfileSection.ENTROPY.value == "entropy"

    def test_section_count(self) -> None:
        """Should have exactly 6 sections."""
        assert len(ProfileSection) == 6


class TestManifoldRegion:
    """Tests for ManifoldRegion dataclass."""

    def test_create_region(self) -> None:
        """Should create ManifoldRegion with all fields."""
        region = ManifoldRegion(
            start_position=0.0,
            end_position=0.3,
            mean_entropy=1.5,
        )
        assert region.start_position == 0.0
        assert region.end_position == 0.3
        assert region.mean_entropy == 1.5


class TestLayerProfile:
    """Tests for LayerProfile dataclass."""

    def test_create_minimal_layer_profile(self) -> None:
        """Should create LayerProfile with only required field."""
        lp = LayerProfile(layer_idx=0)
        assert lp.layer_idx == 0
        assert lp.layer_name == ""
        assert lp.sectional_curvature_mean == 0.0
        assert lp.dominant_curvature_sign == "unknown"

    def test_create_full_layer_profile(self) -> None:
        """Should create LayerProfile with all fields."""
        lp = LayerProfile(
            layer_idx=5,
            layer_name="layers.5.self_attn",
            sectional_curvature_mean=-0.15,
            sectional_curvature_std=0.05,
            ollivier_ricci_mean=-0.2,
            ollivier_ricci_std=0.08,
            dominant_curvature_sign="negative",
            intrinsic_dimension=64.5,
            intrinsic_dimension_uncertainty=2.3,
            intrinsic_dimension_method="mle",
            shannon_entropy=3.2,
            renyi_entropy_alpha2=2.8,
            betti_0=1,
            betti_1=3,
            max_persistence=0.45,
            gradient_norm=0.001,
            condition_number=150.0,
            manifold_regions=[
                ManifoldRegion(0.0, 0.3, 1.5),
                ManifoldRegion(0.3, 0.7, 2.5),
                ManifoldRegion(0.7, 1.0, 3.5),
            ],
        )
        assert lp.layer_idx == 5
        assert lp.layer_name == "layers.5.self_attn"
        assert lp.sectional_curvature_mean == -0.15
        assert lp.dominant_curvature_sign == "negative"
        assert lp.shannon_entropy == 3.2
        assert lp.betti_0 == 1
        assert lp.betti_1 == 3
        assert len(lp.manifold_regions) == 3

    def test_layer_profile_to_dict(self) -> None:
        """Should serialize LayerProfile to dict."""
        lp = LayerProfile(
            layer_idx=0,
            sectional_curvature_mean=-0.1,
            betti_0=2,
        )
        d = lp.to_dict()
        assert d["layer_idx"] == 0
        assert d["sectional_curvature_mean"] == -0.1
        assert d["betti_0"] == 2

    def test_layer_profile_from_dict(self) -> None:
        """Should deserialize LayerProfile from dict."""
        d = {
            "layer_idx": 3,
            "layer_name": "layers.3.mlp",
            "sectional_curvature_mean": -0.2,
            "ollivier_ricci_mean": -0.15,
            "dominant_curvature_sign": "negative",
            "intrinsic_dimension": 50.0,
            "betti_0": 1,
        }
        lp = LayerProfile.from_dict(d)
        assert lp.layer_idx == 3
        assert lp.layer_name == "layers.3.mlp"
        assert lp.sectional_curvature_mean == -0.2
        assert lp.dominant_curvature_sign == "negative"

    def test_layer_profile_roundtrip(self) -> None:
        """Should survive serialization roundtrip."""
        original = LayerProfile(
            layer_idx=7,
            layer_name="layers.7.self_attn",
            sectional_curvature_mean=-0.12,
            ollivier_ricci_mean=-0.18,
            intrinsic_dimension=72.5,
            shannon_entropy=3.1,
            betti_0=1,
            betti_1=2,
            manifold_regions=[
                ManifoldRegion(0.0, 0.5, 1.8),
            ],
        )
        d = original.to_dict()
        restored = LayerProfile.from_dict(d)

        assert restored.layer_idx == original.layer_idx
        assert restored.layer_name == original.layer_name
        assert restored.sectional_curvature_mean == original.sectional_curvature_mean
        assert restored.ollivier_ricci_mean == original.ollivier_ricci_mean
        assert restored.intrinsic_dimension == original.intrinsic_dimension
        assert restored.shannon_entropy == original.shannon_entropy
        assert restored.betti_0 == original.betti_0
        assert len(restored.manifold_regions) == 1

    def test_layer_profile_handles_nan(self) -> None:
        """Should convert NaN to None in to_dict."""
        lp = LayerProfile(
            layer_idx=0,
            sectional_curvature_mean=float("nan"),
            ollivier_ricci_mean=float("inf"),
        )
        d = lp.to_dict()
        assert d["sectional_curvature_mean"] is None
        assert d["ollivier_ricci_mean"] is None

    def test_layer_profile_handles_none_in_from_dict(self) -> None:
        """Should convert None to defaults in from_dict."""
        d = {
            "layer_idx": 0,
            "sectional_curvature_mean": None,
            "ollivier_ricci_mean": None,
        }
        lp = LayerProfile.from_dict(d)
        assert lp.sectional_curvature_mean == 0.0
        assert lp.ollivier_ricci_mean == 0.0


class TestTopologySummary:
    """Tests for TopologySummary dataclass."""

    def test_create_default(self) -> None:
        """Should create TopologySummary with defaults."""
        ts = TopologySummary()
        assert ts.component_count == 1
        assert ts.cycle_count == 0

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip TopologySummary."""
        original = TopologySummary(
            component_count=2,
            cycle_count=5,
            average_persistence=0.3,
            max_persistence=0.8,
            persistence_entropy=1.2,
            betti_numbers={0: 2, 1: 5},
        )
        d = original.to_dict()
        restored = TopologySummary.from_dict(d)

        assert restored.component_count == original.component_count
        assert restored.cycle_count == original.cycle_count
        assert restored.betti_numbers == original.betti_numbers


class TestSemanticSignature:
    """Tests for SemanticSignature dataclass."""

    def test_create_default(self) -> None:
        """Should create SemanticSignature with defaults."""
        ss = SemanticSignature()
        assert ss.vector == []
        assert ss.dominant_primes == []

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip SemanticSignature."""
        original = SemanticSignature(
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            dominant_primes=["I", "YOU", "THINK", "KNOW", "WANT"],
        )
        d = original.to_dict()
        restored = SemanticSignature.from_dict(d)

        assert restored.vector == original.vector
        assert restored.dominant_primes == original.dominant_primes


class TestDensitySummary:
    """Tests for DensitySummary dataclass."""

    def test_create_default(self) -> None:
        """Should create DensitySummary with defaults."""
        ds = DensitySummary()
        assert ds.overall_density == 0.0
        assert ds.domain_densities == {}

    def test_to_dict_and_from_dict(self) -> None:
        """Should roundtrip DensitySummary."""
        original = DensitySummary(
            overall_density=0.75,
            domain_densities={"math": 0.8, "language": 0.7},
        )
        d = original.to_dict()
        restored = DensitySummary.from_dict(d)

        assert restored.overall_density == original.overall_density
        assert restored.domain_densities == original.domain_densities


class TestModelProfile:
    """Tests for ModelProfile dataclass."""

    def test_create_minimal_profile(self) -> None:
        """Should create ModelProfile with only required field."""
        mp = ModelProfile(model_path="/path/to/model")
        assert mp.model_path == "/path/to/model"
        assert mp.profile_version == SCHEMA_VERSION
        assert mp.model_family == "unknown"
        assert mp.layer_profiles == []
        assert mp.computed_sections == []

    def test_create_full_profile(self) -> None:
        """Should create ModelProfile with all fields."""
        mp = ModelProfile(
            model_path="/path/to/Qwen2.5-3B-Instruct",
            model_family="qwen",
            architecture="qwen2",
            parameter_count=3_000_000_000,
            hidden_dim=2048,
            num_layers=32,
            num_attention_heads=16,
            vocab_size=151936,
            layer_profiles=[
                LayerProfile(layer_idx=0, sectional_curvature_mean=-0.1),
                LayerProfile(layer_idx=1, sectional_curvature_mean=-0.12),
            ],
            global_sectional_mean=-0.11,
            global_ollivier_ricci_mean=-0.15,
            topology_summary=TopologySummary(component_count=1, cycle_count=3),
            semantic_signature=SemanticSignature(vector=[0.1, 0.2]),
            density_summary=DensitySummary(overall_density=0.8),
            computed_sections=["geometry", "topology"],
            backend_used="mlx",
        )
        assert mp.model_family == "qwen"
        assert mp.architecture == "qwen2"
        assert len(mp.layer_profiles) == 2
        assert mp.topology_summary is not None
        assert "geometry" in mp.computed_sections

    def test_has_section(self) -> None:
        """Should correctly check for computed sections."""
        mp = ModelProfile(
            model_path="/path/to/model",
            computed_sections=["geometry", "topology"],
        )
        assert mp.has_section(ProfileSection.GEOMETRY)
        assert mp.has_section(ProfileSection.TOPOLOGY)
        assert not mp.has_section(ProfileSection.SEMANTIC)
        assert not mp.has_section(ProfileSection.DENSITY)

    def test_add_section(self) -> None:
        """Should add section to computed_sections."""
        mp = ModelProfile(model_path="/path/to/model")
        assert not mp.has_section(ProfileSection.GEOMETRY)

        mp.add_section(ProfileSection.GEOMETRY)
        assert mp.has_section(ProfileSection.GEOMETRY)

        # Adding again should not duplicate
        mp.add_section(ProfileSection.GEOMETRY)
        assert mp.computed_sections.count("geometry") == 1

    def test_to_dict(self) -> None:
        """Should serialize ModelProfile to dict."""
        mp = ModelProfile(
            model_path="/path/to/model",
            model_id="model_123",
            model_family="qwen",
            hidden_dim=2048,
            layer_profiles=[LayerProfile(layer_idx=0)],
            probe_cache={"atlas:hash": {"probe_mode": "atlas"}},
        )
        d = mp.to_dict()

        assert d["_schema"] == SCHEMA_VERSION
        assert d["model_path"] == "/path/to/model"
        assert d["model_id"] == "model_123"
        assert d["model_family"] == "qwen"
        assert d["hidden_dim"] == 2048
        assert len(d["layer_profiles"]) == 1

    def test_from_dict(self) -> None:
        """Should deserialize ModelProfile from dict."""
        d = {
            "model_path": "/path/to/model",
            "model_id": "model_abc",
            "model_family": "llama",
            "architecture": "llama",
            "hidden_dim": 4096,
            "num_layers": 32,
            "layer_profiles": [
                {"layer_idx": 0, "sectional_curvature_mean": -0.1},
                {"layer_idx": 1, "sectional_curvature_mean": -0.15},
            ],
            "topology_summary": {"component_count": 1, "cycle_count": 2},
            "probe_cache": {"atlas:hash": {"probe_mode": "atlas"}},
        }
        mp = ModelProfile.from_dict(d)

        assert mp.model_path == "/path/to/model"
        assert mp.model_id == "model_abc"
        assert mp.model_family == "llama"
        assert mp.hidden_dim == 4096
        assert len(mp.layer_profiles) == 2
        assert mp.topology_summary is not None
        assert mp.topology_summary.cycle_count == 2

    def test_profile_roundtrip(self) -> None:
        """Should survive serialization roundtrip."""
        original = ModelProfile(
            model_path="/path/to/Qwen2.5-3B-Instruct",
            model_family="qwen",
            architecture="qwen2",
            parameter_count=3_000_000_000,
            hidden_dim=2048,
            num_layers=32,
            num_attention_heads=16,
            vocab_size=151936,
            layer_profiles=[
                LayerProfile(
                    layer_idx=0,
                    sectional_curvature_mean=-0.1,
                    ollivier_ricci_mean=-0.15,
                    intrinsic_dimension=50.0,
                    shannon_entropy=3.0,
                ),
            ],
            global_sectional_mean=-0.1,
            global_ollivier_ricci_mean=-0.15,
            topology_summary=TopologySummary(component_count=1, cycle_count=3),
            semantic_signature=SemanticSignature(
                vector=[0.1, 0.2, 0.3],
                dominant_primes=["I", "YOU"],
            ),
            density_summary=DensitySummary(overall_density=0.75),
            computed_sections=["geometry", "topology", "semantic"],
            backend_used="mlx",
        )

        d = original.to_dict()
        restored = ModelProfile.from_dict(d)

        assert restored.model_path == original.model_path
        assert restored.model_family == original.model_family
        assert restored.architecture == original.architecture
        assert restored.parameter_count == original.parameter_count
        assert len(restored.layer_profiles) == 1
        assert restored.layer_profiles[0].sectional_curvature_mean == -0.1
        assert restored.topology_summary is not None
        assert restored.semantic_signature is not None
        assert len(restored.computed_sections) == 3

    def test_save_and_load(self) -> None:
        """Should save and load from JSON file."""
        mp = ModelProfile(
            model_path="/path/to/model",
            model_family="qwen",
            layer_profiles=[LayerProfile(layer_idx=0)],
            computed_sections=["geometry"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            mp.save(path)

            # Verify file exists
            assert path.exists()

            # Verify JSON is valid
            with open(path) as f:
                data = json.load(f)
            assert data["model_path"] == "/path/to/model"

            # Load and verify
            loaded = ModelProfile.load(path)
            assert loaded.model_path == mp.model_path
            assert loaded.model_family == mp.model_family
            assert len(loaded.layer_profiles) == 1

    def test_merge_with_empty_target(self) -> None:
        """Should merge into empty profile."""
        base = ModelProfile(model_path="/path/to/model")
        other = ModelProfile(
            model_path="/path/to/model",
            model_family="qwen",
            hidden_dim=2048,
            layer_profiles=[LayerProfile(layer_idx=0)],
            computed_sections=["geometry"],
        )

        merged = base.merge_with(other)
        assert merged.model_family == "qwen"
        assert merged.hidden_dim == 2048
        assert len(merged.layer_profiles) == 1
        assert "geometry" in merged.computed_sections

    def test_merge_with_overlapping_layers(self) -> None:
        """Should merge layer data properly."""
        base = ModelProfile(
            model_path="/path/to/model",
            layer_profiles=[
                LayerProfile(layer_idx=0, sectional_curvature_mean=-0.1),
                LayerProfile(layer_idx=1, sectional_curvature_mean=-0.15),
            ],
            computed_sections=["geometry"],
        )
        other = ModelProfile(
            model_path="/path/to/model",
            layer_profiles=[
                LayerProfile(layer_idx=0, shannon_entropy=3.0, betti_0=1),
                LayerProfile(layer_idx=1, shannon_entropy=3.2, betti_0=1),
            ],
            computed_sections=["entropy", "topology"],
        )

        merged = base.merge_with(other)
        assert len(merged.layer_profiles) == 2

        # Layer 0 should have curvature from base and entropy from other
        lp0 = merged.layer_profiles[0]
        assert lp0.sectional_curvature_mean == -0.1
        assert lp0.shannon_entropy == 3.0
        assert lp0.betti_0 == 1

        # All sections should be present
        assert "geometry" in merged.computed_sections
        assert "entropy" in merged.computed_sections
        assert "topology" in merged.computed_sections

    def test_merge_topology_and_semantic(self) -> None:
        """Should merge optional sections properly."""
        base = ModelProfile(
            model_path="/path/to/model",
            topology_summary=TopologySummary(component_count=1, cycle_count=2),
        )
        other = ModelProfile(
            model_path="/path/to/model",
            semantic_signature=SemanticSignature(vector=[0.1, 0.2]),
            density_summary=DensitySummary(overall_density=0.8),
        )

        merged = base.merge_with(other)
        assert merged.topology_summary is not None
        assert merged.topology_summary.cycle_count == 2
        assert merged.semantic_signature is not None
        assert merged.density_summary is not None
        assert merged.density_summary.overall_density == 0.8


class TestModelProfileImport:
    """Tests for importing from existing profile formats."""

    def test_from_curvature_profile(self) -> None:
        """Should import from CurvatureProfile."""
        from modelcypher.core.domain.geometry.curvature_profile import (
            CurvatureProfile,
            LayerCurvature,
        )

        curvature = CurvatureProfile(
            model_path="/path/to/Qwen2.5-3B-Instruct",
            model_family="qwen",
            model_size="3B",
            layer_curvatures=[
                LayerCurvature(
                    layer_idx=0,
                    sectional_mean=-0.1,
                    sectional_std=0.02,
                    ollivier_ricci_mean=-0.15,
                    ollivier_ricci_std=0.03,
                    intrinsic_dimension=50.0,
                    dominant_sign="negative",
                ),
                LayerCurvature(
                    layer_idx=1,
                    sectional_mean=-0.12,
                    sectional_std=0.025,
                    ollivier_ricci_mean=-0.18,
                    ollivier_ricci_std=0.035,
                    intrinsic_dimension=55.0,
                    dominant_sign="negative",
                ),
            ],
            total_layers=32,
            global_sectional_mean=-0.11,
            global_sectional_std=0.022,
            global_ollivier_ricci_mean=-0.165,
            global_ollivier_ricci_std=0.032,
            global_intrinsic_dimension_mean=52.5,
        )

        profile = ModelProfile.from_curvature_profile(curvature)

        assert profile.model_path == "/path/to/Qwen2.5-3B-Instruct"
        assert profile.model_family == "qwen"
        assert profile.num_layers == 32
        assert len(profile.layer_profiles) == 2
        assert profile.layer_profiles[0].sectional_curvature_mean == -0.1
        assert profile.layer_profiles[0].ollivier_ricci_mean == -0.15
        assert profile.layer_profiles[0].intrinsic_dimension == 50.0
        assert profile.global_sectional_mean == -0.11
        assert ProfileSection.GEOMETRY.value in profile.computed_sections


class TestSchemaVersioning:
    """Tests for schema versioning."""

    def test_schema_version_in_export(self) -> None:
        """Should include schema version in export."""
        mp = ModelProfile(model_path="/path/to/model")
        d = mp.to_dict()
        assert d["_schema"] == SCHEMA_VERSION
        assert d["profile_version"] == SCHEMA_VERSION

    def test_load_unknown_version(self) -> None:
        """Should load profiles with unknown version."""
        d = {
            "_schema": "mc.model_profile.v99",
            "model_path": "/path/to/model",
            "profile_version": "mc.model_profile.v99",
        }
        mp = ModelProfile.from_dict(d)
        assert mp.profile_version == "mc.model_profile.v99"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_layer_profiles(self) -> None:
        """Should handle empty layer profiles."""
        mp = ModelProfile(model_path="/path/to/model")
        d = mp.to_dict()
        restored = ModelProfile.from_dict(d)
        assert restored.layer_profiles == []

    def test_nan_and_inf_handling(self) -> None:
        """Should handle NaN and Inf in global curvature."""
        mp = ModelProfile(
            model_path="/path/to/model",
            global_sectional_mean=float("nan"),
            global_ollivier_ricci_mean=float("inf"),
        )
        d = mp.to_dict()
        assert d["global_sectional_mean"] is None
        assert d["global_ollivier_ricci_mean"] is None

        # Should load back with defaults
        restored = ModelProfile.from_dict(d)
        assert restored.global_sectional_mean == 0.0
        assert restored.global_ollivier_ricci_mean == 0.0

    def test_computed_at_auto_set(self) -> None:
        """Should auto-set computed_at if not provided."""
        mp = ModelProfile(model_path="/path/to/model")
        assert mp.computed_at != ""
        # Should be ISO format datetime
        assert "T" in mp.computed_at

    def test_partial_layer_dict(self) -> None:
        """Should handle partial layer dict with missing keys."""
        d = {
            "layer_idx": 0,
            # Missing most fields
        }
        lp = LayerProfile.from_dict(d)
        assert lp.layer_idx == 0
        assert lp.sectional_curvature_mean == 0.0
        assert lp.dominant_curvature_sign == "unknown"
