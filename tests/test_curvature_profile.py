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

"""Comprehensive tests for curvature profile module.

Tests cover:
- LayerCurvature serialization
- CurvatureProfile creation and serialization
- FamilyBaseline aggregation
- CurvatureAlignment computation
- Model info parsing
- Edge cases and error handling
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.domain.geometry.curvature_profile import (
    CurvatureAlignment,
    CurvatureProfile,
    FamilyBaseline,
    LayerCurvature,
    build_family_baseline,
    compute_curvature_alignment,
    parse_model_info,
    SCHEMA_VERSION,
)


# =============================================================================
# LayerCurvature Tests
# =============================================================================


class TestLayerCurvature:
    """Tests for LayerCurvature dataclass."""

    def test_default_values(self):
        """LayerCurvature initializes with sensible defaults."""
        lc = LayerCurvature(layer_idx=5)
        assert lc.layer_idx == 5
        assert lc.sectional_mean == 0.0
        assert lc.ollivier_ricci_mean == 0.0
        assert lc.intrinsic_dimension == 0.0
        assert lc.dominant_sign == "unknown"
        assert lc.manifold_health == "unknown"

    def test_custom_values(self):
        """LayerCurvature stores custom values correctly."""
        lc = LayerCurvature(
            layer_idx=10,
            sectional_mean=-0.05,
            sectional_std=0.02,
            sectional_min=-0.15,
            sectional_max=0.01,
            dominant_sign="negative",
            ollivier_ricci_mean=-0.12,
            ollivier_ricci_std=0.03,
            intrinsic_dimension=256.5,
            intrinsic_dimension_uncertainty=12.3,
            manifold_health="healthy",
        )
        assert lc.layer_idx == 10
        assert lc.sectional_mean == -0.05
        assert lc.dominant_sign == "negative"
        assert lc.ollivier_ricci_mean == -0.12
        assert lc.intrinsic_dimension == 256.5
        assert lc.manifold_health == "healthy"

    def test_to_dict(self):
        """LayerCurvature serializes to dictionary."""
        lc = LayerCurvature(
            layer_idx=3,
            sectional_mean=0.1,
            ollivier_ricci_mean=-0.2,
            dominant_sign="positive",
        )
        d = lc.to_dict()
        assert d["layer_idx"] == 3
        assert d["sectional_mean"] == 0.1
        assert d["ollivier_ricci_mean"] == -0.2
        assert d["dominant_sign"] == "positive"

    def test_from_dict(self):
        """LayerCurvature deserializes from dictionary."""
        d = {
            "layer_idx": 7,
            "sectional_mean": -0.3,
            "sectional_std": 0.05,
            "sectional_min": -0.5,
            "sectional_max": -0.1,
            "dominant_sign": "negative",
            "ollivier_ricci_mean": -0.15,
            "ollivier_ricci_std": 0.02,
            "intrinsic_dimension": 128.0,
            "intrinsic_dimension_uncertainty": 5.0,
            "manifold_health": "healthy",
        }
        lc = LayerCurvature.from_dict(d)
        assert lc.layer_idx == 7
        assert lc.sectional_mean == -0.3
        assert lc.dominant_sign == "negative"

    def test_from_dict_missing_optional_fields(self):
        """LayerCurvature handles missing optional fields gracefully."""
        d = {"layer_idx": 0}
        lc = LayerCurvature.from_dict(d)
        assert lc.layer_idx == 0
        assert lc.sectional_mean == 0.0
        assert lc.dominant_sign == "unknown"

    def test_roundtrip_serialization(self):
        """LayerCurvature survives dict roundtrip."""
        original = LayerCurvature(
            layer_idx=15,
            sectional_mean=-0.08,
            sectional_std=0.03,
            sectional_min=-0.2,
            sectional_max=0.04,
            dominant_sign="mixed",
            ollivier_ricci_mean=-0.1,
            ollivier_ricci_std=0.02,
            intrinsic_dimension=192.0,
            intrinsic_dimension_uncertainty=8.5,
            manifold_health="degenerate",
        )
        restored = LayerCurvature.from_dict(original.to_dict())
        assert restored.layer_idx == original.layer_idx
        assert restored.sectional_mean == original.sectional_mean
        assert restored.dominant_sign == original.dominant_sign
        assert restored.manifold_health == original.manifold_health


# =============================================================================
# CurvatureProfile Tests
# =============================================================================


class TestCurvatureProfile:
    """Tests for CurvatureProfile dataclass."""

    def test_default_values(self):
        """CurvatureProfile initializes with defaults."""
        profile = CurvatureProfile(
            model_path="/path/to/model",
            model_family="qwen",
            model_size="0.5B",
        )
        assert profile.model_path == "/path/to/model"
        assert profile.model_family == "qwen"
        assert profile.model_size == "0.5B"
        assert profile.layer_curvatures == []
        assert profile.total_layers == 0
        assert profile.global_sectional_mean == 0.0

    def test_with_layer_curvatures(self):
        """CurvatureProfile stores layer curvatures."""
        layers = [
            LayerCurvature(layer_idx=i, sectional_mean=-0.05 + i * 0.01)
            for i in range(5)
        ]
        profile = CurvatureProfile(
            model_path="/path/to/model",
            model_family="llama",
            model_size="7B",
            layer_curvatures=layers,
            total_layers=32,
            global_sectional_mean=-0.03,
            global_ollivier_ricci_mean=-0.1,
        )
        assert len(profile.layer_curvatures) == 5
        assert profile.total_layers == 32
        assert profile.global_sectional_mean == -0.03

    def test_to_dict_includes_schema(self):
        """CurvatureProfile serialization includes schema version."""
        profile = CurvatureProfile(
            model_path="/path/to/model",
            model_family="qwen",
            model_size="3B",
        )
        d = profile.to_dict()
        assert "_schema" in d
        assert d["_schema"] == SCHEMA_VERSION

    def test_roundtrip_serialization(self):
        """CurvatureProfile survives dict roundtrip."""
        layers = [
            LayerCurvature(
                layer_idx=i,
                sectional_mean=-0.1 + i * 0.02,
                ollivier_ricci_mean=-0.15,
                intrinsic_dimension=100.0 + i * 10,
            )
            for i in range(3)
        ]
        original = CurvatureProfile(
            model_path="/models/qwen-0.5b",
            model_family="qwen",
            model_size="0.5B",
            layer_curvatures=layers,
            total_layers=24,
            global_sectional_mean=-0.08,
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.15,
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=110.0,
            extraction_date="2025-12-31T12:00:00",
            extraction_config={"k_neighbors": 10},
        )
        restored = CurvatureProfile.from_dict(original.to_dict())
        assert restored.model_path == original.model_path
        assert restored.model_family == original.model_family
        assert len(restored.layer_curvatures) == 3
        assert restored.global_sectional_mean == original.global_sectional_mean

    def test_save_and_load(self):
        """CurvatureProfile saves to and loads from file."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="mistral",
            model_size="7B",
            layer_curvatures=[LayerCurvature(layer_idx=0, sectional_mean=-0.1)],
            total_layers=32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            profile.save(path)

            assert path.exists()

            loaded = CurvatureProfile.load(path)
            assert loaded.model_path == profile.model_path
            assert loaded.model_family == profile.model_family
            assert len(loaded.layer_curvatures) == 1


# =============================================================================
# FamilyBaseline Tests
# =============================================================================


class TestFamilyBaseline:
    """Tests for FamilyBaseline dataclass."""

    def test_default_values(self):
        """FamilyBaseline initializes with defaults."""
        baseline = FamilyBaseline(family="qwen")
        assert baseline.family == "qwen"
        assert baseline.layer_positions == []
        assert baseline.sample_count == 0

    def test_with_values(self):
        """FamilyBaseline stores provided values."""
        baseline = FamilyBaseline(
            family="llama",
            layer_positions=[0.0, 0.5, 1.0],
            sectional_mean_by_position=[-0.1, -0.05, -0.02],
            sectional_std_by_position=[0.02, 0.03, 0.01],
            ollivier_ricci_mean_by_position=[-0.15, -0.12, -0.1],
            ollivier_ricci_std_by_position=[0.03, 0.02, 0.02],
            intrinsic_dimension_by_position=[100.0, 150.0, 200.0],
            contributing_models=["/path/a", "/path/b"],
            sample_count=2,
        )
        assert baseline.family == "llama"
        assert len(baseline.layer_positions) == 3
        assert baseline.sample_count == 2

    def test_roundtrip_serialization(self):
        """FamilyBaseline survives dict roundtrip."""
        original = FamilyBaseline(
            family="qwen",
            layer_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
            sectional_mean_by_position=[-0.1, -0.08, -0.06, -0.04, -0.02],
            sectional_std_by_position=[0.02] * 5,
            ollivier_ricci_mean_by_position=[-0.15] * 5,
            ollivier_ricci_std_by_position=[0.03] * 5,
            intrinsic_dimension_by_position=[100.0] * 5,
            contributing_models=["model1", "model2"],
            sample_count=2,
            created_date="2025-12-31",
        )
        restored = FamilyBaseline.from_dict(original.to_dict())
        assert restored.family == original.family
        assert restored.layer_positions == original.layer_positions
        assert restored.sample_count == original.sample_count

    def test_save_and_load(self):
        """FamilyBaseline saves to and loads from file."""
        baseline = FamilyBaseline(
            family="mistral",
            layer_positions=[0.0, 1.0],
            sectional_mean_by_position=[-0.1, -0.05],
            sample_count=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            baseline.save(path)

            assert path.exists()

            loaded = FamilyBaseline.load(path)
            assert loaded.family == baseline.family


# =============================================================================
# build_family_baseline Tests
# =============================================================================


class TestBuildFamilyBaseline:
    """Tests for build_family_baseline function."""

    def test_empty_profiles(self):
        """build_family_baseline handles empty list."""
        baseline = build_family_baseline([], "qwen")
        assert baseline.family == "qwen"
        assert baseline.sample_count == 0
        assert len(baseline.layer_positions) == 11  # Default num_positions

    def test_single_profile(self):
        """build_family_baseline handles single profile."""
        profile = CurvatureProfile(
            model_path="/models/qwen-0.5b",
            model_family="qwen",
            model_size="0.5B",
            layer_curvatures=[
                LayerCurvature(layer_idx=i, sectional_mean=-0.1, ollivier_ricci_mean=-0.15)
                for i in range(10)
            ],
            total_layers=10,
        )
        baseline = build_family_baseline([profile], "qwen")
        assert baseline.family == "qwen"
        assert baseline.sample_count == 1
        assert len(baseline.contributing_models) == 1

    def test_multiple_profiles(self):
        """build_family_baseline aggregates multiple profiles."""
        profiles = []
        for size, mean in [("0.5B", -0.1), ("3B", -0.08), ("7B", -0.06)]:
            profiles.append(
                CurvatureProfile(
                    model_path=f"/models/qwen-{size}",
                    model_family="qwen",
                    model_size=size,
                    layer_curvatures=[
                        LayerCurvature(layer_idx=i, sectional_mean=mean, ollivier_ricci_mean=-0.15)
                        for i in range(24)
                    ],
                    total_layers=24,
                )
            )

        baseline = build_family_baseline(profiles, "qwen")
        assert baseline.family == "qwen"
        assert baseline.sample_count == 3
        assert len(baseline.contributing_models) == 3

        # Check that means are averaged
        assert len(baseline.sectional_mean_by_position) == 11
        for mean in baseline.sectional_mean_by_position:
            assert -0.12 < mean < -0.04  # Between min and max profile means

    def test_different_layer_counts(self):
        """build_family_baseline handles models with different layer counts."""
        profiles = [
            CurvatureProfile(
                model_path="/models/small",
                model_family="qwen",
                model_size="0.5B",
                layer_curvatures=[
                    LayerCurvature(layer_idx=i, sectional_mean=-0.1)
                    for i in range(24)
                ],
                total_layers=24,
            ),
            CurvatureProfile(
                model_path="/models/large",
                model_family="qwen",
                model_size="7B",
                layer_curvatures=[
                    LayerCurvature(layer_idx=i, sectional_mean=-0.08)
                    for i in range(32)
                ],
                total_layers=32,
            ),
        ]

        baseline = build_family_baseline(profiles, "qwen")
        assert baseline.sample_count == 2
        # Should still have 11 position samples
        assert len(baseline.layer_positions) == 11

    def test_custom_num_positions(self):
        """build_family_baseline respects custom num_positions."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="test",
            model_size="1B",
            layer_curvatures=[
                LayerCurvature(layer_idx=i, sectional_mean=-0.1)
                for i in range(10)
            ],
            total_layers=10,
        )
        baseline = build_family_baseline([profile], "test", num_positions=5)
        assert len(baseline.layer_positions) == 5
        assert baseline.layer_positions == [0.0, 0.25, 0.5, 0.75, 1.0]


# =============================================================================
# compute_curvature_alignment Tests
# =============================================================================


class TestComputeCurvatureAlignment:
    """Tests for compute_curvature_alignment function."""

    def test_identical_profiles(self):
        """Identical profiles have perfect alignment."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="qwen",
            model_size="0.5B",
            global_sectional_mean=-0.1,
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.15,
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=100.0,
        )

        alignment = compute_curvature_alignment(profile, profile)
        assert alignment.score == 1.0
        assert alignment.sectional_alignment == 1.0
        assert alignment.ollivier_ricci_alignment == 1.0
        assert alignment.sectional_z_score == 0.0
        assert alignment.ollivier_ricci_z_score == 0.0

    def test_different_profiles_no_baseline(self):
        """Different profiles produce lower alignment without baseline."""
        source = CurvatureProfile(
            model_path="/models/source",
            model_family="qwen",
            model_size="0.5B",
            global_sectional_mean=-0.1,
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.15,
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=100.0,
        )
        target = CurvatureProfile(
            model_path="/models/target",
            model_family="qwen",
            model_size="3B",
            global_sectional_mean=-0.05,  # Different
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.10,  # Different
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=150.0,  # Different
        )

        alignment = compute_curvature_alignment(source, target)
        assert 0.0 < alignment.score < 1.0
        assert alignment.sectional_z_score > 0.0
        assert alignment.baseline_family == "none"

    def test_with_baseline(self):
        """Alignment uses baseline for z-score when provided."""
        source = CurvatureProfile(
            model_path="/models/source",
            model_family="qwen",
            model_size="0.5B",
            global_sectional_mean=-0.1,
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.15,
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=100.0,
        )
        target = CurvatureProfile(
            model_path="/models/target",
            model_family="qwen",
            model_size="3B",
            global_sectional_mean=-0.08,
            global_sectional_std=0.02,
            global_ollivier_ricci_mean=-0.13,
            global_ollivier_ricci_std=0.03,
            global_intrinsic_dimension_mean=120.0,
        )
        baseline = FamilyBaseline(
            family="qwen",
            layer_positions=[0.0, 0.5, 1.0],
            sectional_mean_by_position=[-0.1, -0.08, -0.06],
            sectional_std_by_position=[0.05, 0.05, 0.05],  # Large std = smaller z-score
            ollivier_ricci_mean_by_position=[-0.15, -0.13, -0.11],
            ollivier_ricci_std_by_position=[0.05, 0.05, 0.05],
            sample_count=3,
        )

        alignment = compute_curvature_alignment(source, target, baseline)
        assert alignment.baseline_family == "qwen"
        assert alignment.baseline_model_count == 3
        # With larger baseline std, z-scores should be smaller, alignment higher
        assert alignment.score > 0.5

    def test_very_different_profiles(self):
        """Very different profiles have low alignment."""
        source = CurvatureProfile(
            model_path="/models/source",
            model_family="qwen",
            model_size="0.5B",
            global_sectional_mean=-0.1,
            global_sectional_std=0.01,  # Small std
            global_ollivier_ricci_mean=-0.15,
            global_ollivier_ricci_std=0.01,
            global_intrinsic_dimension_mean=100.0,
        )
        target = CurvatureProfile(
            model_path="/models/target",
            model_family="llama",  # Different family
            model_size="70B",
            global_sectional_mean=0.1,  # Very different (positive vs negative)
            global_sectional_std=0.01,
            global_ollivier_ricci_mean=0.05,  # Very different (positive = collapsed)
            global_ollivier_ricci_std=0.01,
            global_intrinsic_dimension_mean=500.0,  # Very different
        )

        alignment = compute_curvature_alignment(source, target)
        # Score should be very low due to large differences
        assert alignment.score < 0.5
        assert alignment.sectional_z_score > 3.0  # More than 3 sigma

    def test_alignment_score_bounded(self):
        """Alignment score is bounded to [0, 1]."""
        source = CurvatureProfile(
            model_path="/models/source",
            model_family="test",
            model_size="1B",
            global_sectional_mean=1000.0,  # Extreme
            global_sectional_std=0.001,
            global_ollivier_ricci_mean=1000.0,
            global_ollivier_ricci_std=0.001,
        )
        target = CurvatureProfile(
            model_path="/models/target",
            model_family="test",
            model_size="1B",
            global_sectional_mean=-1000.0,  # Extreme opposite
            global_sectional_std=0.001,
            global_ollivier_ricci_mean=-1000.0,
            global_ollivier_ricci_std=0.001,
        )

        alignment = compute_curvature_alignment(source, target)
        assert 0.0 <= alignment.score <= 1.0
        assert 0.0 <= alignment.sectional_alignment <= 1.0
        assert 0.0 <= alignment.ollivier_ricci_alignment <= 1.0


# =============================================================================
# parse_model_info Tests
# =============================================================================


class TestParseModelInfo:
    """Tests for parse_model_info function."""

    def test_qwen_models(self):
        """parse_model_info recognizes Qwen models."""
        assert parse_model_info("/path/to/Qwen2.5-0.5B-Instruct-bf16") == ("qwen", "0.5B")
        assert parse_model_info("/path/to/Qwen2-0.5B-Instruct-4bit") == ("qwen", "0.5B")
        assert parse_model_info("/path/to/Qwen2.5-3B-Instruct-bf16") == ("qwen", "3B")
        assert parse_model_info("/path/to/Qwen3-8B-4bit") == ("qwen", "8B")
        assert parse_model_info("/path/to/Qwen2.5-72B-Instruct") == ("qwen", "72B")

    def test_llama_models(self):
        """parse_model_info recognizes Llama models."""
        assert parse_model_info("/path/to/Llama-3.2-3B-Instruct-4bit") == ("llama", "3B")
        assert parse_model_info("/path/to/Llama-3-8B") == ("llama", "8B")
        assert parse_model_info("/path/to/Llama-2-70B") == ("llama", "70B")

    def test_mistral_models(self):
        """parse_model_info recognizes Mistral models."""
        assert parse_model_info("/path/to/Mistral-7B-Instruct-v0.3-4bit") == ("mistral", "7B")
        assert parse_model_info("/path/to/mathstral-7B-v0.1-8bit") == ("mistral", "7B")

    def test_smollm_models(self):
        """parse_model_info recognizes SmolLM models."""
        assert parse_model_info("/path/to/SmolLM-360M-Instruct-4bit")[0] == "smollm"

    def test_granite_models(self):
        """parse_model_info recognizes Granite models."""
        assert parse_model_info("/path/to/granite-3b-code")[0] == "granite"

    def test_unknown_model(self):
        """parse_model_info returns 'unknown' for unrecognized models."""
        family, size = parse_model_info("/path/to/SomeNewModel-10B")
        assert family == "unknown"
        assert size == "unknown"

    def test_case_insensitive(self):
        """parse_model_info is case insensitive."""
        assert parse_model_info("/path/to/QWEN2-0.5b-instruct")[0] == "qwen"
        assert parse_model_info("/path/to/LLAMA-3B")[0] == "llama"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_values_in_layer_curvature(self):
        """LayerCurvature handles NaN values by converting to None for JSON."""
        lc = LayerCurvature(
            layer_idx=0,
            sectional_mean=float("nan"),
            ollivier_ricci_mean=float("nan"),
        )
        d = lc.to_dict()
        # NaN is converted to None for JSON serialization safety
        assert d["sectional_mean"] is None
        assert d["ollivier_ricci_mean"] is None

    def test_empty_layer_curvatures_in_profile(self):
        """CurvatureProfile handles empty layer_curvatures."""
        profile = CurvatureProfile(
            model_path="/path",
            model_family="test",
            model_size="1B",
            layer_curvatures=[],
        )
        d = profile.to_dict()
        assert d["layer_curvatures"] == []

    def test_baseline_with_single_model_has_zero_std(self):
        """Baseline from single model has zero std (variance undefined)."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="test",
            model_size="1B",
            layer_curvatures=[
                LayerCurvature(layer_idx=0, sectional_mean=-0.1)
            ],
            total_layers=1,
        )
        baseline = build_family_baseline([profile], "test")
        # Single sample means std is 0
        for std in baseline.sectional_std_by_position:
            assert std == 0.0

    def test_alignment_with_zero_std_profile(self):
        """compute_curvature_alignment handles zero std profiles."""
        profile = CurvatureProfile(
            model_path="/path",
            model_family="test",
            model_size="1B",
            global_sectional_mean=0.0,
            global_sectional_std=0.0,  # Zero std
            global_ollivier_ricci_mean=0.0,
            global_ollivier_ricci_std=0.0,  # Zero std
        )
        # Should not raise, uses fallback epsilon
        alignment = compute_curvature_alignment(profile, profile)
        assert alignment.score == 1.0

    def test_alignment_dataclass_is_frozen(self):
        """CurvatureAlignment is immutable."""
        alignment = compute_curvature_alignment(
            CurvatureProfile(
                model_path="/a",
                model_family="test",
                model_size="1B",
            ),
            CurvatureProfile(
                model_path="/b",
                model_family="test",
                model_size="1B",
            ),
        )
        with pytest.raises(AttributeError):
            alignment.score = 0.5  # Should fail - frozen dataclass

    def test_json_serialization(self):
        """Profiles serialize to valid JSON."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="qwen",
            model_size="0.5B",
            layer_curvatures=[
                LayerCurvature(layer_idx=i, sectional_mean=-0.1 + i * 0.01)
                for i in range(5)
            ],
            total_layers=5,
        )
        # Should not raise
        json_str = json.dumps(profile.to_dict())
        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["model_family"] == "qwen"

    def test_profile_with_metadata(self):
        """CurvatureProfile stores extraction metadata."""
        profile = CurvatureProfile(
            model_path="/models/test",
            model_family="test",
            model_size="1B",
            extraction_date="2025-12-31T12:00:00",
            extraction_config={"k_neighbors": 10, "num_probes": 100},
        )
        d = profile.to_dict()
        assert d["extraction_date"] == "2025-12-31T12:00:00"
        assert d["extraction_config"]["k_neighbors"] == 10
