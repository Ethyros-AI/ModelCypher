# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the adapters module.

Tests AdapterBlender and Signal functionality.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.adapters.adapter_blender import (
    AdapterBlender,
    BlendResult,
)
from modelcypher.core.domain.adapters.signal import (
    InputFormat,
    OutputFormat,
    PayloadValue,
    Priority,
    QuerySubtype,
    Signal,
    SignalType,
    SystemEvent,
    TaskType,
    matches_capability,
    normalize_capability_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    """Get the default backend."""
    return get_default_backend()


# =============================================================================
# AdapterBlender Tests
# =============================================================================


class TestAdapterBlenderWeights:
    """Tests for AdapterBlender weight operations."""

    def test_blend_weights_empty_returns_none(self) -> None:
        """Empty weights should return None."""
        result = AdapterBlender.blend_weights([])
        assert result is None

    def test_blend_weights_single_returns_scaled(self, backend: "Backend") -> None:
        """Single weight should return scaled matrix."""
        matrix = backend.ones((4, 4))
        backend.eval(matrix)

        result = AdapterBlender.blend_weights([(matrix, 0.5)])
        backend.eval(result)

        result_np = backend.to_numpy(result)
        # All values should be 0.5 (1.0 * 0.5)
        assert result_np[0, 0] == pytest.approx(0.5)

    def test_blend_weights_multiple_weighted_sum(self, backend: "Backend") -> None:
        """Multiple weights should compute weighted sum."""
        m1 = backend.ones((4, 4)) * 2.0
        m2 = backend.ones((4, 4)) * 4.0
        backend.eval(m1, m2)

        # 0.5 * 2 + 0.5 * 4 = 3
        result = AdapterBlender.blend_weights([(m1, 0.5), (m2, 0.5)])
        backend.eval(result)

        result_np = backend.to_numpy(result)
        assert result_np[0, 0] == pytest.approx(3.0)

    def test_normalize_weights_sums_to_one(self) -> None:
        """Normalized weights should sum to 1.0."""
        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        weights = {id1: 2.0, id2: 3.0, id3: 5.0}

        normalized = AdapterBlender.normalize_weights(weights)

        total = sum(normalized.values())
        assert total == pytest.approx(1.0)
        assert normalized[id1] == pytest.approx(0.2)
        assert normalized[id2] == pytest.approx(0.3)
        assert normalized[id3] == pytest.approx(0.5)

    def test_normalize_weights_zero_total_unchanged(self) -> None:
        """Zero total should return unchanged weights."""
        id1 = uuid4()
        weights = {id1: 0.0}

        normalized = AdapterBlender.normalize_weights(weights)
        assert normalized[id1] == 0.0

    def test_softmax_weights_distribution(self) -> None:
        """Softmax should produce valid probability distribution."""
        id1, id2 = uuid4(), uuid4()
        weights = {id1: 1.0, id2: 2.0}

        softmax = AdapterBlender.softmax_weights(weights, temperature=1.0)

        # Should sum to 1
        total = sum(softmax.values())
        assert total == pytest.approx(1.0)

        # Higher input should have higher probability
        assert softmax[id2] > softmax[id1]

    def test_softmax_weights_temperature_effect(self) -> None:
        """Lower temperature should sharpen distribution."""
        id1, id2 = uuid4(), uuid4()
        weights = {id1: 1.0, id2: 2.0}

        # Higher temperature - more uniform
        high_temp = AdapterBlender.softmax_weights(weights, temperature=10.0)
        # Lower temperature - more peaked
        low_temp = AdapterBlender.softmax_weights(weights, temperature=0.1)

        # Low temp should have more extreme difference
        diff_high = abs(high_temp[id2] - high_temp[id1])
        diff_low = abs(low_temp[id2] - low_temp[id1])

        assert diff_low > diff_high

    def test_apply_weight_floor(self) -> None:
        """Weight floor should ensure minimum participation."""
        id1, id2 = uuid4(), uuid4()
        weights = {id1: 0.0, id2: 1.0}

        floored = AdapterBlender.apply_weight_floor(weights, floor=0.1)

        # Both should be at least 0.1 before normalization
        assert floored[id1] > 0.0
        # Should still sum to 1
        assert sum(floored.values()) == pytest.approx(1.0)


class TestAdapterBlenderLoRA:
    """Tests for AdapterBlender LoRA operations."""

    def test_separate_lora_weights(self, backend: "Backend") -> None:
        """Should correctly separate A and B matrices."""
        weights = {
            "layer.0.lora_a": backend.ones((8, 4)),
            "layer.0.lora_b": backend.ones((16, 8)),
            "layer.1.lora_a": backend.ones((8, 4)),
            "layer.1.lora_b": backend.ones((16, 8)),
        }

        a_matrices, b_matrices = AdapterBlender.separate_lora_weights(weights)

        assert len(a_matrices) == 2
        assert len(b_matrices) == 2
        assert "layer.0" in a_matrices
        assert "layer.1" in a_matrices
        assert "layer.0" in b_matrices
        assert "layer.1" in b_matrices

    def test_recombine_lora_weights(self, backend: "Backend") -> None:
        """Should correctly recombine A and B matrices."""
        a_matrices = {
            "layer.0": backend.ones((8, 4)),
            "layer.1": backend.ones((8, 4)),
        }
        b_matrices = {
            "layer.0": backend.ones((16, 8)),
            "layer.1": backend.ones((16, 8)),
        }

        combined = AdapterBlender.recombine_lora_weights(a_matrices, b_matrices)

        assert len(combined) == 4
        assert "layer.0.lora_a" in combined
        assert "layer.0.lora_b" in combined
        assert "layer.1.lora_a" in combined
        assert "layer.1.lora_b" in combined

    def test_blend_lora_matrices_empty(self) -> None:
        """Empty input should return empty result."""
        result = AdapterBlender.blend_lora_matrices([], [])
        assert result == {}

    def test_blend_lora_matrices_mismatched_length(self, backend: "Backend") -> None:
        """Mismatched lengths should return empty result."""
        a_matrices = [{"layer.0": backend.ones((8, 4))}]
        weights = [0.5, 0.5]  # 2 weights for 1 matrix

        result = AdapterBlender.blend_lora_matrices(a_matrices, weights)
        assert result == {}

    def test_blend_lora_matrices_weighted_sum(self, backend: "Backend") -> None:
        """Should compute weighted sum of matrices."""
        m1 = {"layer.0": backend.ones((4, 4)) * 2.0}
        m2 = {"layer.0": backend.ones((4, 4)) * 4.0}
        backend.eval(m1["layer.0"], m2["layer.0"])

        result = AdapterBlender.blend_lora_matrices([m1, m2], [0.5, 0.5])

        assert "layer.0" in result
        backend.eval(result["layer.0"])
        result_np = backend.to_numpy(result["layer.0"])
        # 0.5 * 2 + 0.5 * 4 = 3
        assert result_np[0, 0] == pytest.approx(3.0)

    def test_blend_complete_adapters_empty(self) -> None:
        """Empty adapters should return None."""
        result = AdapterBlender.blend_complete_adapters([])
        assert result is None

    def test_blend_complete_adapters_single(self, backend: "Backend") -> None:
        """Single adapter should return scaled weights."""
        weights = {
            "layer.0.lora_a": backend.ones((8, 4)) * 2.0,
            "layer.0.lora_b": backend.ones((16, 8)) * 2.0,
        }
        backend.eval(*weights.values())

        result = AdapterBlender.blend_complete_adapters([(weights, 0.5)])

        assert result is not None
        backend.eval(*result.values())
        # Values should be scaled by 0.5
        a_np = backend.to_numpy(result["layer.0.lora_a"])
        assert a_np[0, 0] == pytest.approx(1.0)

    def test_compute_geometric_weights(self) -> None:
        """Geometric weights should normalize compatibility scores."""
        id1, id2 = uuid4(), uuid4()
        scores = {id1: 0.8, id2: 0.2}

        weights = AdapterBlender.compute_geometric_weights(scores)

        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights[id1] == pytest.approx(0.8)
        assert weights[id2] == pytest.approx(0.2)

    def test_compute_fidelity_weights_with_fallback(self) -> None:
        """Should use fallback for None scores."""
        id1, id2 = uuid4(), uuid4()
        scores = {id1: 0.9, id2: None}

        weights = AdapterBlender.compute_fidelity_weights(scores, fallback=0.5)

        assert sum(weights.values()) == pytest.approx(1.0)
        # id1 should have higher weight (0.9 vs 0.5 fallback)
        assert weights[id1] > weights[id2]


# =============================================================================
# Signal Tests
# =============================================================================


class TestPayloadValue:
    """Tests for PayloadValue functionality."""

    def test_string_value(self) -> None:
        """String payload should store and retrieve correctly."""
        pv = PayloadValue.string("hello")
        assert pv.kind == "string"
        assert pv.string_value == "hello"
        assert pv.int_value is None

    def test_int_value(self) -> None:
        """Int payload should store and retrieve correctly."""
        pv = PayloadValue.int(42)
        assert pv.kind == "int"
        assert pv.int_value == 42
        assert pv.string_value is None

    def test_double_value(self) -> None:
        """Double payload should store and retrieve correctly."""
        pv = PayloadValue.double(3.14)
        assert pv.kind == "double"
        assert pv.double_value == pytest.approx(3.14)

    def test_double_value_from_int(self) -> None:
        """Int should convert to double when accessed as double."""
        pv = PayloadValue.int(42)
        assert pv.double_value == pytest.approx(42.0)

    def test_bool_value(self) -> None:
        """Bool payload should store and retrieve correctly."""
        pv = PayloadValue.bool(True)
        assert pv.kind == "bool"
        assert pv.bool_value is True

    def test_null_value(self) -> None:
        """Null payload should have null kind."""
        pv = PayloadValue.null()
        assert pv.kind == "null"
        assert pv.value is None


class TestSignalType:
    """Tests for SignalType functionality."""

    def test_query_signal_type(self) -> None:
        """Query signal type should have correct capability string."""
        st = SignalType.query(QuerySubtype.infer)
        assert st.namespace == "query"
        assert st.value == "infer"
        assert st.capability_string == "query:infer"

    def test_input_format_signal_type(self) -> None:
        """Input format signal type should have correct capability string."""
        st = SignalType.input_format(InputFormat.json)
        assert st.capability_string == "input:json"

    def test_output_format_signal_type(self) -> None:
        """Output format signal type should have correct capability string."""
        st = SignalType.output_format(OutputFormat.markdown)
        assert st.capability_string == "output:markdown"

    def test_domain_signal_type(self) -> None:
        """Domain signal type should have correct capability string."""
        st = SignalType.domain("finance")
        assert st.capability_string == "domain:finance"

    def test_system_event_signal_type(self) -> None:
        """System event signal type should have correct capability string."""
        st = SignalType.system_event(SystemEvent.model_loaded)
        assert st.capability_string == "system:modelLoaded"

    def test_custom_signal_type(self) -> None:
        """Custom signal type should include custom name."""
        st = SignalType.custom("myns", "myevent")
        assert st.capability_string == "myns:myevent"


class TestSignal:
    """Tests for Signal functionality."""

    def test_signal_has_id(self) -> None:
        """Signal should have unique ID."""
        s1 = Signal()
        s2 = Signal()
        assert s1.id != s2.id

    def test_signal_has_timestamp(self) -> None:
        """Signal should have timestamp."""
        before = datetime.utcnow()
        s = Signal()
        after = datetime.utcnow()

        assert before <= s.timestamp <= after

    def test_signal_query_factory(self) -> None:
        """Query factory should create correct signal."""
        s = Signal.query(QuerySubtype.chat, "hello world")

        assert s.type.namespace == "query"
        assert s.type.value == "chat"
        assert s.payload["text"].string_value == "hello world"

    def test_signal_domain_factory(self) -> None:
        """Domain factory should create correct signal."""
        s = Signal.domain("finance", action="analyze")

        assert s.type.namespace == "domain"
        assert s.type.value == "finance"
        assert s.payload["action"].string_value == "analyze"

    def test_signal_system_factory(self) -> None:
        """System factory should create correct signal."""
        s = Signal.system(
            SystemEvent.memory_pressure,
            payload={"level": PayloadValue.double(0.9)},
            priority=Priority.critical,
        )

        assert s.type.namespace == "system"
        assert s.priority == Priority.critical
        assert s.payload["level"].double_value == pytest.approx(0.9)

    def test_signal_is_expired_no_ttl(self) -> None:
        """Signal without TTL should never expire."""
        s = Signal(ttl=None)
        assert s.is_expired is False

    def test_signal_is_expired_with_ttl(self) -> None:
        """Signal with exceeded TTL should be expired."""
        # Create signal with expired timestamp
        old_time = datetime.utcnow() - timedelta(seconds=10)
        s = Signal(timestamp=old_time, ttl=5.0)  # 5 second TTL, 10 seconds ago

        assert s.is_expired is True

    def test_signal_not_expired_within_ttl(self) -> None:
        """Signal within TTL should not be expired."""
        s = Signal(ttl=3600.0)  # 1 hour TTL
        assert s.is_expired is False


class TestCapabilityMatching:
    """Tests for capability matching functionality."""

    def test_normalize_capability_value(self) -> None:
        """Should normalize capability values consistently."""
        assert normalize_capability_value("Hello_World") == "hello-world"
        assert normalize_capability_value("FOO.BAR") == "foo-bar"
        assert normalize_capability_value("  test  ") == "test"

    def test_matches_capability_exact_match(self) -> None:
        """Exact capability match should succeed."""
        s = Signal.query(QuerySubtype.infer, "test")
        assert matches_capability(s, "query:infer") is True

    def test_matches_capability_wildcard(self) -> None:
        """Wildcard should match any value in namespace."""
        s = Signal.query(QuerySubtype.chat, "test")
        assert matches_capability(s, "query:*") is True

    def test_matches_capability_wrong_namespace(self) -> None:
        """Wrong namespace should not match."""
        s = Signal.query(QuerySubtype.infer, "test")
        assert matches_capability(s, "domain:infer") is False

    def test_matches_capability_wrong_value(self) -> None:
        """Wrong value should not match."""
        s = Signal.query(QuerySubtype.infer, "test")
        assert matches_capability(s, "query:chat") is False

    def test_matches_capability_invalid_pattern(self) -> None:
        """Invalid pattern without colon should not match."""
        s = Signal.query(QuerySubtype.infer, "test")
        assert matches_capability(s, "invalid") is False
