# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.hidden_state_extractor import (
    CapturedState,
    ExtractionSummary,
    HiddenStateExtractor,
)


class TestHiddenStateExtractorConstruction:
    """Tests for HiddenStateExtractor initialization and validation."""

    def test_empty_target_layers_raises_value_error(self) -> None:
        """Empty target_layers raises ValueError."""
        with pytest.raises(ValueError, match="target_layers must be specified"):
            HiddenStateExtractor(target_layers=set())

    def test_valid_target_layers(self) -> None:
        """Construction with valid target_layers succeeds."""
        extractor = HiddenStateExtractor(target_layers={0, 1, 2})
        assert extractor.target_layers == {0, 1, 2}

    def test_with_expected_hidden_dim(self) -> None:
        """Construction with expected_hidden_dim succeeds."""
        extractor = HiddenStateExtractor(target_layers={5}, expected_hidden_dim=768)
        assert extractor is not None


class TestHiddenStateExtractorInitialState:
    """Tests for initial state before any session activity."""

    def test_not_active_initially(self) -> None:
        """Extractor is not active before start_session."""
        extractor = HiddenStateExtractor(target_layers={0, 1})
        assert extractor.is_active is False

    def test_no_captures_initially(self) -> None:
        """No captures exist before any session."""
        extractor = HiddenStateExtractor(target_layers={0, 1})
        assert extractor.state_count == 0
        assert extractor.captured_layers == set()

    def test_has_all_target_layers_false_initially(self) -> None:
        """has_all_target_layers is False with no captures."""
        extractor = HiddenStateExtractor(target_layers={0, 1, 2})
        assert extractor.has_all_target_layers is False


class TestSessionLifecycle:
    """Tests for start_session / end_session lifecycle."""

    def test_start_session_activates(self) -> None:
        """start_session sets is_active to True."""
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()
        assert extractor.is_active is True

    def test_end_session_deactivates(self) -> None:
        """end_session sets is_active to False."""
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()
        extractor.end_session()
        assert extractor.is_active is False

    def test_end_session_returns_summary(self) -> None:
        """end_session returns an ExtractionSummary."""
        extractor = HiddenStateExtractor(target_layers={0, 1})
        extractor.start_session()
        summary = extractor.end_session()

        assert isinstance(summary, ExtractionSummary)
        assert summary.total_captures == 0
        assert summary.tokens_processed == 0
        assert summary.duration >= 0.0

    def test_end_session_summary_reflects_captures(self) -> None:
        """ExtractionSummary reflects actual capture counts."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0, 1})
        extractor.start_session()

        hidden = b.array([1.0, 2.0, 3.0, 4.0])
        extractor.capture(hidden, layer=0, token_index=0)
        extractor.capture(hidden, layer=1, token_index=0)

        summary = extractor.end_session()

        assert summary.total_captures == 2
        assert summary.layers_captured == {0, 1}


class TestCaptureBehavior:
    """Tests for the capture method."""

    def test_capture_when_not_active_is_ignored(self) -> None:
        """Captures are silently ignored when session is not active."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0})
        hidden = b.array([1.0, 2.0, 3.0, 4.0])

        extractor.capture(hidden, layer=0, token_index=0)

        assert extractor.state_count == 0

    def test_capture_for_non_target_layer_is_ignored(self) -> None:
        """Captures for layers not in target_layers are ignored."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0, 1})
        extractor.start_session()

        hidden = b.array([1.0, 2.0, 3.0, 4.0])
        extractor.capture(hidden, layer=5, token_index=0)

        assert extractor.state_count == 0

    def test_capture_accumulates_for_target_layers(self) -> None:
        """Captures accumulate states for target layers."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0, 1, 2})
        extractor.start_session()

        hidden = b.array([1.0, 2.0, 3.0, 4.0])
        extractor.capture(hidden, layer=0, token_index=0)
        extractor.capture(hidden, layer=1, token_index=0)

        assert extractor.state_count == 2
        assert extractor.captured_layers == {0, 1}

    def test_has_all_target_layers_when_all_captured(self) -> None:
        """has_all_target_layers is True when every target layer has a capture."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0, 1})
        extractor.start_session()

        hidden = b.array([1.0, 2.0, 3.0])
        extractor.capture(hidden, layer=0, token_index=0)
        assert extractor.has_all_target_layers is False

        extractor.capture(hidden, layer=1, token_index=0)
        assert extractor.has_all_target_layers is True

    def test_new_token_index_clears_previous_states(self) -> None:
        """Moving to a new token_index clears captures from the previous token."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0, 1})
        extractor.start_session()

        hidden = b.array([1.0, 2.0, 3.0])
        extractor.capture(hidden, layer=0, token_index=0)
        extractor.capture(hidden, layer=1, token_index=0)
        assert extractor.state_count == 2

        # New token index clears old states
        extractor.capture(hidden, layer=0, token_index=1)
        assert extractor.state_count == 1
        assert extractor.captured_layers == {0}


class TestExtractedStates:
    """Tests for extracted_states method."""

    def test_returns_copy(self) -> None:
        """extracted_states returns a copy, not the internal dict."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()

        hidden = b.array([1.0, 2.0])
        extractor.capture(hidden, layer=0, token_index=0)

        states_a = extractor.extracted_states()
        states_b = extractor.extracted_states()

        assert states_a is not states_b
        assert set(states_a.keys()) == set(states_b.keys())

    def test_empty_when_no_captures(self) -> None:
        """extracted_states returns empty dict when nothing is captured."""
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()

        states = extractor.extracted_states()
        assert states == {}


class TestClearAndReset:
    """Tests for clear_states and reset methods."""

    def test_clear_states_clears_without_ending_session(self) -> None:
        """clear_states removes captures but keeps session active."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()

        hidden = b.array([1.0, 2.0])
        extractor.capture(hidden, layer=0, token_index=0)
        assert extractor.state_count == 1

        extractor.clear_states()

        assert extractor.state_count == 0
        assert extractor.is_active is True

    def test_reset_clears_everything(self) -> None:
        """reset clears captures and ends session."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()

        hidden = b.array([1.0, 2.0])
        extractor.capture(hidden, layer=0, token_index=0)

        extractor.reset()

        assert extractor.state_count == 0
        assert extractor.is_active is False


class TestDebugInfo:
    """Tests for debug_info method."""

    def test_returns_string(self) -> None:
        """debug_info returns a non-empty string."""
        extractor = HiddenStateExtractor(target_layers={0, 5, 10})
        info = extractor.debug_info()

        assert isinstance(info, str)
        assert len(info) > 0

    def test_contains_state_information(self) -> None:
        """debug_info includes session and layer information."""
        extractor = HiddenStateExtractor(target_layers={3, 7})
        info = extractor.debug_info()

        assert "Active" in info
        assert "Target Layers" in info
        assert "Captured Layers" in info


class TestNeuronCollection:
    """Tests for neuron collection methods."""

    def test_enable_neuron_collection(self) -> None:
        """enable_neuron_collection does not raise."""
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.enable_neuron_collection(True)

    def test_start_neuron_collection_starts_session(self) -> None:
        """start_neuron_collection starts a session if not active."""
        extractor = HiddenStateExtractor(target_layers={0})
        assert extractor.is_active is False

        extractor.start_neuron_collection()

        assert extractor.is_active is True

    def test_finalize_prompt_activations_without_collection(self) -> None:
        """finalize_prompt_activations is a no-op when collection is disabled."""
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_session()

        # Should not raise even with no collection enabled
        extractor.finalize_prompt_activations()

    def test_neuron_activation_summary_no_data(self) -> None:
        """get_neuron_activation_summary returns no_data when nothing collected."""
        extractor = HiddenStateExtractor(target_layers={0})
        summary = extractor.get_neuron_activation_summary()

        assert summary["status"] == "no_data"
        assert summary["prompt_count"] == 0

    def test_neuron_collection_lifecycle(self) -> None:
        """Full neuron collection lifecycle captures and summarizes data."""
        b = get_default_backend()
        extractor = HiddenStateExtractor(target_layers={0})
        extractor.start_neuron_collection()

        hidden = b.array([1.0, 2.0, 3.0])
        extractor.capture(hidden, layer=0, token_index=0)
        extractor.finalize_prompt_activations()

        summary = extractor.get_neuron_activation_summary()

        assert summary["status"] == "collected"
        assert summary["prompt_count"] == 1
        assert summary["layers_collected"] == 1
        assert 0 in summary["layer_indices"]


class TestDataclasses:
    """Tests for CapturedState and ExtractionSummary dataclasses."""

    def test_captured_state_fields(self) -> None:
        """CapturedState holds expected fields."""
        b = get_default_backend()
        state = CapturedState(
            layer=5,
            token_index=10,
            state=b.array([1.0, 2.0]),
        )
        assert state.layer == 5
        assert state.token_index == 10
        assert state.captured_at is not None

    def test_extraction_summary_fields(self) -> None:
        """ExtractionSummary holds expected fields."""
        summary = ExtractionSummary(
            total_captures=42,
            tokens_processed=7,
            layers_captured={0, 1, 2},
            duration=1.5,
        )
        assert summary.total_captures == 42
        assert summary.tokens_processed == 7
        assert summary.layers_captured == {0, 1, 2}
        assert summary.duration == pytest.approx(1.5)
