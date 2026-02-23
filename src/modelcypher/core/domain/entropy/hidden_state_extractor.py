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

"""
Hidden State Extractor for Layer-wise Activations.

Extracts hidden states from transformer layers during inference.
The caller specifies which layers to extract - layer selection is
determined by upstream geometric analysis (CKA alignment, density
profiling, etc.), not by this module.

This is a utility class. The geometry decides which layers matter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


@dataclass
class CapturedState:
    """Container for a captured hidden state."""

    layer: int
    token_index: int
    state: "Array"
    captured_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractionSummary:
    """Summary of extraction session."""

    total_captures: int
    tokens_processed: int
    layers_captured: set[int]
    duration: float


class HiddenStateExtractor:
    """
    Extracts hidden states from specified transformer layers.

    Layer selection is NOT this module's responsibility. The caller
    determines which layers to extract based on geometric analysis:
    - CKA alignment identifies corresponding layers across models
    - Density profiling identifies graft opportunities
    - The merge pipeline knows which layers need extraction

    This class just captures what it's told to capture.

    Usage:
        # Caller knows which layers from geometric analysis
        aligned_layers = {24, 25, 26, 27, 28}  # From CKA alignment

        extractor = HiddenStateExtractor(target_layers=aligned_layers)
        extractor.start_session()

        # During forward pass:
        extractor.capture(hidden_state, layer=25, token_index=0)

        states = extractor.extracted_states()
        extractor.end_session()
    """

    def __init__(
        self,
        target_layers: set[int],
        expected_hidden_dim: int | None = None,
    ) -> None:
        """Create hidden state extractor.

        Args:
            target_layers: Layer indices to capture (from geometric analysis).
            expected_hidden_dim: Expected hidden dimension for validation.
        """
        if not target_layers:
            raise ValueError("target_layers must be specified by caller")

        self._target_layers = target_layers
        self._expected_hidden_dim = expected_hidden_dim

        # History depth derived from layer count — memory scales linearly with depth
        layer_count = max(target_layers) + 1 if target_layers else 32
        backend = get_default_backend()
        self._max_history_tokens = max(5, int(sqrt_scalar(float(layer_count), backend)))
        self._keep_history = False

        # Session state
        self._current_states: "dict[int, Array]" = {}
        self._state_history: list[dict[int, CapturedState]] = []
        self._current_token_index: int = -1
        self._is_active: bool = False
        self._capture_count: int = 0
        self._session_start: datetime | None = None

        # Per-neuron analysis storage
        self._neuron_activations: dict[int, list[list[float]]] = {}
        self._prompt_count: int = 0
        self._collect_for_neuron_analysis: bool = False

    @property
    def target_layers(self) -> set[int]:
        """Target layers for extraction."""
        return self._target_layers

    @property
    def is_active(self) -> bool:
        """Whether extraction session is active."""
        return self._is_active

    @property
    def captured_layers(self) -> set[int]:
        """Layers captured in current token."""
        return set(self._current_states.keys())

    @property
    def has_all_target_layers(self) -> bool:
        """Whether all target layers have been captured."""
        return self._target_layers.issubset(self.captured_layers)

    @property
    def state_count(self) -> int:
        """Number of states captured for current token."""
        return len(self._current_states)

    def start_session(self) -> None:
        """Start a new extraction session."""
        self._is_active = True
        self._current_states.clear()
        self._state_history.clear()
        self._current_token_index = -1
        self._capture_count = 0
        self._session_start = datetime.now()

    def end_session(self) -> ExtractionSummary:
        """End session and return summary."""
        self._is_active = False

        duration = 0.0
        if self._session_start:
            duration = (datetime.now() - self._session_start).total_seconds()

        return ExtractionSummary(
            total_captures=self._capture_count,
            tokens_processed=self._current_token_index + 1,
            layers_captured=set(self._current_states.keys()),
            duration=duration,
        )

    def capture(self, hidden_state: "Array", layer: int, token_index: int) -> None:
        """
        Capture a hidden state from a specific layer.

        Args:
            hidden_state: The hidden state tensor.
            layer: Layer index (0-indexed).
            token_index: Current token index in generation.
        """
        if not self._is_active:
            return

        if layer not in self._target_layers:
            return

        # Validate hidden dimension
        if self._expected_hidden_dim is not None:
            actual_dim = hidden_state.shape[-1]
            if actual_dim != self._expected_hidden_dim:
                logger.warning(
                    "Hidden dim mismatch: expected %s, got %s",
                    self._expected_hidden_dim,
                    actual_dim,
                )

        # Handle token transition
        if token_index != self._current_token_index:
            if self._keep_history and self._current_states:
                historical = {
                    layer: CapturedState(
                        layer=layer, token_index=self._current_token_index, state=state
                    )
                    for layer, state in self._current_states.items()
                }
                self._state_history.append(historical)

                if len(self._state_history) > self._max_history_tokens:
                    self._state_history.pop(0)

            self._current_states.clear()
            self._current_token_index = token_index

        # Extract last token if sequence
        h: "Array"
        if hidden_state.ndim > 2:
            h = hidden_state[0, -1]
        elif hidden_state.ndim == 2:
            h = hidden_state[-1]
        else:
            h = hidden_state

        self._current_states[layer] = h
        self._capture_count += 1

    def extracted_states(self) -> "dict[int, Array]":
        """Get extracted hidden states for current token."""
        return self._current_states.copy()

    def states_for_token(self, token_index: int) -> "dict[int, Array] | None":
        """Get states for a specific token (if history enabled)."""
        if not self._keep_history:
            return None

        if token_index == self._current_token_index:
            return self._current_states.copy()

        for historical in self._state_history:
            first_state = next(iter(historical.values()), None)
            if first_state and first_state.token_index == token_index:
                return {layer: s.state for layer, s in historical.items()}

        return None

    def enable_history(self, enable: bool = True) -> None:
        """Enable or disable token history."""
        self._keep_history = enable

    def enable_neuron_collection(self, enable: bool = True) -> None:
        """Enable collection for per-neuron analysis."""
        self._collect_for_neuron_analysis = enable

    def clear_states(self) -> None:
        """Clear captured states without ending session."""
        self._current_states.clear()
        self._state_history.clear()
        self._current_token_index = -1

    def reset(self) -> None:
        """Full reset including session state."""
        self.clear_states()
        self._is_active = False
        self._capture_count = 0
        self._session_start = None

    def debug_info(self) -> str:
        """Debug information string."""
        return f"""HiddenStateExtractor Debug:
  Active: {self._is_active}
  Target Layers: {sorted(self._target_layers)}
  Current Token: {self._current_token_index}
  Captured Layers: {sorted(self.captured_layers)}
  Total Captures: {self._capture_count}
  History Tokens: {len(self._state_history) if self._keep_history else "N/A"}"""

    # =========================================================================
    # Per-Neuron Analysis
    # =========================================================================

    def start_neuron_collection(self) -> None:
        """Start collecting activations for per-neuron analysis."""
        self._neuron_activations.clear()
        self._prompt_count = 0
        self._collect_for_neuron_analysis = True
        if not self._is_active:
            self.start_session()

    def finalize_prompt_activations(self) -> None:
        """Save current token's activations for neuron analysis."""
        if not self._collect_for_neuron_analysis:
            return

        for layer, state in self._current_states.items():
            if layer not in self._neuron_activations:
                self._neuron_activations[layer] = []

            backend = get_default_backend()
            try:
                activation_vector = backend.tolist(state)
            except Exception:
                activation_vector = state.tolist()
            self._neuron_activations[layer].append(activation_vector)

        self._prompt_count += 1
        self._current_states.clear()
        self._current_token_index = -1

    def get_neuron_activations(self) -> dict[int, list[list[float]]]:
        """Get collected per-neuron activations."""
        return self._neuron_activations.copy()

    def get_neuron_activation_summary(self) -> dict[str, any]:
        """Get summary of collected neuron activations."""
        if not self._neuron_activations:
            return {"status": "no_data", "prompt_count": 0}

        return {
            "status": "collected",
            "prompt_count": self._prompt_count,
            "layers_collected": len(self._neuron_activations),
            "layer_indices": sorted(self._neuron_activations.keys()),
            "activations_per_layer": {
                layer: len(acts) for layer, acts in self._neuron_activations.items()
            },
        }

    def clear_neuron_activations(self) -> None:
        """Clear collected neuron activations."""
        self._neuron_activations.clear()
        self._prompt_count = 0
