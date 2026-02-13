# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Comprehensive tests for DoRA decomposition module."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

import modelcypher.core.domain.geometry.dora_decomposition as mod


# =============================================================================
# ChangeType enum
# =============================================================================


class TestChangeType:
    def test_all_four_values_accessible(self) -> None:
        expected = {
            "MAGNITUDE_DOMINATED": "magnitude_dominated",
            "DIRECTION_DOMINATED": "direction_dominated",
            "BALANCED": "balanced",
            "MINIMAL": "minimal",
        }
        for name, value in expected.items():
            member = mod.ChangeType[name]
            assert member.value == value

    def test_exactly_four_members(self) -> None:
        assert len(mod.ChangeType) == 4

    def test_is_str_subclass(self) -> None:
        for member in mod.ChangeType:
            assert isinstance(member, str)

    def test_str_comparison(self) -> None:
        assert mod.ChangeType.MINIMAL == "minimal"
        assert mod.ChangeType.BALANCED == "balanced"

    def test_string_operations(self) -> None:
        # str subclass means string methods work
        assert mod.ChangeType.MAGNITUDE_DOMINATED.upper() == "MAGNITUDE_DOMINATED"
        assert mod.ChangeType.DIRECTION_DOMINATED.startswith("direction")


# =============================================================================
# MagnitudeDirectionMetrics (mutable dataclass)
# =============================================================================


class TestMagnitudeDirectionMetrics:
    def test_instantiation(self) -> None:
        m = mod.MagnitudeDirectionMetrics(
            layer_name="model.layers.0.self_attn.q_proj",
            base_magnitude=1.5,
            current_magnitude=1.6,
            magnitude_ratio=1.6 / 1.5,
            direction_cosine=0.98,
            directional_drift=0.02,
            absolute_magnitude_change=0.1,
            relative_magnitude_change=0.1 / 1.5,
        )
        assert m.layer_name == "model.layers.0.self_attn.q_proj"
        assert m.base_magnitude == 1.5
        assert m.current_magnitude == 1.6
        assert m.magnitude_ratio == pytest.approx(1.6 / 1.5)
        assert m.direction_cosine == 0.98
        assert m.directional_drift == 0.02
        assert m.absolute_magnitude_change == 0.1
        assert m.relative_magnitude_change == pytest.approx(0.1 / 1.5)

    def test_is_mutable(self) -> None:
        m = mod.MagnitudeDirectionMetrics(
            layer_name="test",
            base_magnitude=1.0,
            current_magnitude=1.0,
            magnitude_ratio=1.0,
            direction_cosine=1.0,
            directional_drift=0.0,
            absolute_magnitude_change=0.0,
            relative_magnitude_change=0.0,
        )
        m.directional_drift = 0.5
        assert m.directional_drift == 0.5

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(mod.MagnitudeDirectionMetrics)


# =============================================================================
# DecompositionResult (mutable dataclass)
# =============================================================================


class TestDecompositionResult:
    def test_instantiation(self) -> None:
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.05,
            overall_directional_drift=0.02,
            dominant_change_type=mod.ChangeType.MAGNITUDE_DOMINATED,
            magnitude_to_direction_ratio=2.5,
            layers_with_significant_direction_change=["layer.0"],
            layers_with_significant_magnitude_change=["layer.0", "layer.1"],
        )
        assert result.overall_magnitude_change == 0.05
        assert result.overall_directional_drift == 0.02
        assert result.dominant_change_type == mod.ChangeType.MAGNITUDE_DOMINATED
        assert result.magnitude_to_direction_ratio == 2.5
        assert result.layers_with_significant_direction_change == ["layer.0"]
        assert result.layers_with_significant_magnitude_change == ["layer.0", "layer.1"]

    def test_computed_at_default(self) -> None:
        before = datetime.now()
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=mod.ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        after = datetime.now()
        assert before <= result.computed_at <= after

    def test_computed_at_can_be_overridden(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0)
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=mod.ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
            computed_at=ts,
        )
        assert result.computed_at == ts

    def test_is_mutable(self) -> None:
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=mod.ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        result.dominant_change_type = mod.ChangeType.BALANCED
        assert result.dominant_change_type == mod.ChangeType.BALANCED

    def test_per_layer_metrics_dict(self) -> None:
        metrics = mod.MagnitudeDirectionMetrics(
            layer_name="layer.0",
            base_magnitude=1.0,
            current_magnitude=1.1,
            magnitude_ratio=1.1,
            direction_cosine=0.99,
            directional_drift=0.01,
            absolute_magnitude_change=0.1,
            relative_magnitude_change=0.1,
        )
        result = mod.DecompositionResult(
            per_layer_metrics={"layer.0": metrics},
            overall_magnitude_change=0.1,
            overall_directional_drift=0.01,
            dominant_change_type=mod.ChangeType.MAGNITUDE_DOMINATED,
            magnitude_to_direction_ratio=10.0,
            layers_with_significant_direction_change=["layer.0"],
            layers_with_significant_magnitude_change=["layer.0"],
        )
        assert "layer.0" in result.per_layer_metrics
        assert result.per_layer_metrics["layer.0"].direction_cosine == 0.99


# =============================================================================
# DoRAMetricKey string constants
# =============================================================================


class TestDoRAMetricKey:
    def test_magnitude_change(self) -> None:
        assert mod.DoRAMetricKey.MAGNITUDE_CHANGE == "geometry/dora_magnitude_change"

    def test_directional_drift(self) -> None:
        assert mod.DoRAMetricKey.DIRECTIONAL_DRIFT == "geometry/dora_directional_drift"

    def test_mag_dir_ratio(self) -> None:
        assert mod.DoRAMetricKey.MAG_DIR_RATIO == "geometry/dora_mag_dir_ratio"

    def test_dominant_type(self) -> None:
        assert mod.DoRAMetricKey.DOMINANT_TYPE == "geometry/dora_dominant_type"

    def test_all_keys_are_strings(self) -> None:
        for key in [
            mod.DoRAMetricKey.MAGNITUDE_CHANGE,
            mod.DoRAMetricKey.DIRECTIONAL_DRIFT,
            mod.DoRAMetricKey.MAG_DIR_RATIO,
            mod.DoRAMetricKey.DOMINANT_TYPE,
        ]:
            assert isinstance(key, str)

    def test_all_keys_start_with_geometry_prefix(self) -> None:
        for key in [
            mod.DoRAMetricKey.MAGNITUDE_CHANGE,
            mod.DoRAMetricKey.DIRECTIONAL_DRIFT,
            mod.DoRAMetricKey.MAG_DIR_RATIO,
            mod.DoRAMetricKey.DOMINANT_TYPE,
        ]:
            assert key.startswith("geometry/dora_")


# =============================================================================
# to_metrics_dict
# =============================================================================


class TestToMetricsDict:
    def test_converts_result_to_dict(self) -> None:
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.12,
            overall_directional_drift=0.03,
            dominant_change_type=mod.ChangeType.MAGNITUDE_DOMINATED,
            magnitude_to_direction_ratio=4.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        d = mod.to_metrics_dict(result)
        assert d[mod.DoRAMetricKey.MAGNITUDE_CHANGE] == 0.12
        assert d[mod.DoRAMetricKey.DIRECTIONAL_DRIFT] == 0.03
        assert d[mod.DoRAMetricKey.MAG_DIR_RATIO] == 4.0

    def test_keys_present(self) -> None:
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=mod.ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        d = mod.to_metrics_dict(result)
        assert set(d.keys()) == {
            mod.DoRAMetricKey.MAGNITUDE_CHANGE,
            mod.DoRAMetricKey.DIRECTIONAL_DRIFT,
            mod.DoRAMetricKey.MAG_DIR_RATIO,
        }

    def test_does_not_include_dominant_type(self) -> None:
        # DOMINANT_TYPE is defined but not included in to_metrics_dict output
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=mod.ChangeType.MINIMAL,
            magnitude_to_direction_ratio=0.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        d = mod.to_metrics_dict(result)
        assert mod.DoRAMetricKey.DOMINANT_TYPE not in d

    def test_all_values_are_float(self) -> None:
        result = mod.DecompositionResult(
            per_layer_metrics={},
            overall_magnitude_change=1.0,
            overall_directional_drift=2.0,
            dominant_change_type=mod.ChangeType.BALANCED,
            magnitude_to_direction_ratio=0.5,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )
        d = mod.to_metrics_dict(result)
        for v in d.values():
            assert isinstance(v, float)


# =============================================================================
# DoRADecomposition (requires backend)
# =============================================================================


class TestDoRADecomposition:
    """Tests requiring any_backend fixture for real array operations."""

    def test_constructor_with_explicit_backend(self, any_backend) -> None:
        dora = mod.DoRADecomposition(backend=any_backend)
        assert dora._backend is any_backend

    def test_constructor_default_backend(self) -> None:
        # Uses get_default_backend(), should not raise
        dora = mod.DoRADecomposition()
        assert dora._backend is not None

    def test_decompose_identical_weights_minimal_change(self, any_backend) -> None:
        """When base == current, directional_drift should be ~0 and magnitude_ratio ~1."""
        b = any_backend
        weight = b.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dora = mod.DoRADecomposition(backend=b)
        metrics = dora.decompose(weight, weight, layer_name="test_layer")
        assert metrics is not None
        assert metrics.layer_name == "test_layer"
        assert metrics.magnitude_ratio == pytest.approx(1.0, abs=1e-4)
        assert metrics.directional_drift == pytest.approx(0.0, abs=1e-4)
        assert metrics.direction_cosine == pytest.approx(1.0, abs=1e-4)
        assert metrics.absolute_magnitude_change == pytest.approx(0.0, abs=1e-4)
        assert metrics.relative_magnitude_change == pytest.approx(0.0, abs=1e-4)

    def test_decompose_shape_mismatch_returns_none(self, any_backend) -> None:
        b = any_backend
        base = b.array([[1.0, 2.0]])
        current = b.array([[1.0, 2.0, 3.0]])
        dora = mod.DoRADecomposition(backend=b)
        result = dora.decompose(base, current)
        assert result is None

    def test_decompose_zero_base_returns_none(self, any_backend) -> None:
        """Zero-magnitude base weight should return None (division protection)."""
        b = any_backend
        base = b.array([[0.0, 0.0, 0.0]])
        current = b.array([[1.0, 2.0, 3.0]])
        dora = mod.DoRADecomposition(backend=b)
        result = dora.decompose(base, current)
        assert result is None

    def test_decompose_scaled_weight_magnitude_dominated(self, any_backend) -> None:
        """Scaling weight by constant changes magnitude but not direction."""
        b = any_backend
        base = b.array([[1.0, 0.0, 0.0, 0.0]])
        current = b.array([[2.0, 0.0, 0.0, 0.0]])
        dora = mod.DoRADecomposition(backend=b)
        metrics = dora.decompose(base, current, "scaled_layer")
        assert metrics is not None
        assert metrics.magnitude_ratio == pytest.approx(2.0, rel=0.01)
        # Direction should be the same (cosine near 1.0)
        assert metrics.direction_cosine == pytest.approx(1.0, abs=0.01)

    def test_analyze_adapter_empty_weights(self, any_backend) -> None:
        """Empty weight dicts should produce an empty/minimal result."""
        dora = mod.DoRADecomposition(backend=any_backend)
        result = dora.analyze_adapter({}, {})
        assert result.dominant_change_type == mod.ChangeType.MINIMAL
        assert result.per_layer_metrics == {}
        assert result.overall_magnitude_change == 0.0
        assert result.overall_directional_drift == 0.0
        assert result.magnitude_to_direction_ratio == 0.0
        assert result.layers_with_significant_direction_change == []
        assert result.layers_with_significant_magnitude_change == []

    def test_analyze_adapter_no_matching_keys(self, any_backend) -> None:
        """Base and current share no keys -> empty result."""
        b = any_backend
        base = {"layer_a": b.array([[1.0, 2.0]])}
        current = {"layer_b": b.array([[3.0, 4.0]])}
        dora = mod.DoRADecomposition(backend=b)
        result = dora.analyze_adapter(base, current)
        assert result.dominant_change_type == mod.ChangeType.MINIMAL

    def test_analyze_adapter_identical_weights(self, any_backend) -> None:
        """Identical weights -> MINIMAL change type."""
        b = any_backend
        w = b.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        base = {"layer.0": w}
        current = {"layer.0": w}
        dora = mod.DoRADecomposition(backend=b)
        result = dora.analyze_adapter(base, current)
        assert result.dominant_change_type == mod.ChangeType.MINIMAL
        assert result.overall_directional_drift == pytest.approx(0.0, abs=1e-4)

    def test_analyze_adapter_result_is_decomposition_result(self, any_backend) -> None:
        b = any_backend
        w = b.array([[1.0, 0.0]])
        dora = mod.DoRADecomposition(backend=b)
        result = dora.analyze_adapter({"l": w}, {"l": w})
        assert isinstance(result, mod.DecompositionResult)

    def test_analyze_adapter_significant_lists_sorted(self, any_backend) -> None:
        """Significant layer lists should be sorted."""
        b = any_backend
        base = {
            "z_layer": b.array([[1.0, 0.0]]),
            "a_layer": b.array([[0.0, 1.0]]),
        }
        current = {
            "z_layer": b.array([[2.0, 0.0]]),
            "a_layer": b.array([[0.0, 2.0]]),
        }
        dora = mod.DoRADecomposition(backend=b)
        result = dora.analyze_adapter(base, current)
        assert result.layers_with_significant_magnitude_change == sorted(
            result.layers_with_significant_magnitude_change
        )
        assert result.layers_with_significant_direction_change == sorted(
            result.layers_with_significant_direction_change
        )

    def test_classify_change_type_indirectly_via_scaled_weights(self, any_backend) -> None:
        """Scaling all weights by 2x should produce magnitude-dominated or balanced change."""
        b = any_backend
        base = {"layer.0": b.array([[1.0, 2.0, 3.0]])}
        current = {"layer.0": b.array([[2.0, 4.0, 6.0]])}
        dora = mod.DoRADecomposition(backend=b)
        result = dora.analyze_adapter(base, current)
        # Pure scaling -> magnitude change with no directional drift
        # Should be MAGNITUDE_DOMINATED or BALANCED depending on epsilon
        assert result.dominant_change_type in (
            mod.ChangeType.MAGNITUDE_DOMINATED,
            mod.ChangeType.BALANCED,
        )
