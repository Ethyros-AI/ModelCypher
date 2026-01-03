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

Ported 1:1 from the reference Swift implementation.

Extracts hidden states from transformer layers during inference for:
- SEP probe inference (layers 75-87.5%)
- Refusal direction detection (layers 40-60%)
- Persona vector analysis (layers 50-70%)

Layer ranges are research-based, not configurable:
- arXiv:2406.15927 - SEP layers 75-87.5% in transformer models
- Arditi 2024 - Refusal direction in middle layers (40-60%)
- Chen/Anthropic 2025 - Persona vectors in late-middle layers (50-70%)

All parameters are derived from model architecture.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


# =============================================================================
# Layer Range Derivation Functions (Research-based)
# =============================================================================


def _sep_probe_layers(total_layers: int) -> set[int]:
    """SEP probe targeting: layers 75-87.5% (arXiv:2406.15927)."""
    start = int(total_layers * 0.75)
    end = int(total_layers * 0.875)
    return set(range(start, end + 1))


def _refusal_direction_layers(total_layers: int) -> set[int]:
    """Refusal direction targeting: layers 40-60% (Arditi 2024)."""
    start = int(total_layers * 0.40)
    end = int(total_layers * 0.60)
    return set(range(start, end + 1))


def _persona_vector_layers(total_layers: int) -> set[int]:
    """Persona vector targeting: layers 50-70%."""
    start = int(total_layers * 0.50)
    end = int(total_layers * 0.70)
    return set(range(start, end + 1))


def _circuit_breaker_layers(total_layers: int) -> set[int]:
    """Circuit breaker targeting: layers 40-75% (comprehensive)."""
    start = int(total_layers * 0.40)
    end = int(total_layers * 0.75)
    return set(range(start, end + 1))


def _full_research_layers(total_layers: int) -> set[int]:
    """Full research metrics: layers 40-87.5%."""
    start = int(total_layers * 0.40)
    end = int(total_layers * 0.875)
    return set(range(start, end + 1))


def _neuron_analysis_layers(total_layers: int) -> set[int]:
    """Neuron analysis targeting: layers 50-85% for sparsity analysis."""
    start = int(total_layers * 0.50)
    end = int(total_layers * 0.85)
    return set(range(start, end + 1))


def _neuron_analysis_range_layers(start_layer: int, end_layer: int) -> set[int]:
    """Explicit layer range for neuron analysis."""
    return set(range(start_layer, end_layer + 1))


# =============================================================================
# Data Classes
# =============================================================================


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


# =============================================================================
# Hidden State Extractor
# =============================================================================


class HiddenStateExtractor:
    """
    Extracts hidden states from transformer layers during inference.

    Provides access to intermediate activations for SEP probe inference.
    Uses a callback-based approach since MLX doesn't have PyTorch-style hooks.

    Layer targeting is derived from research findings:
    - SEP probe: 75-87.5% layers (arXiv:2406.15927)
    - Refusal direction: 40-60% layers (Arditi 2024)
    - Persona vectors: 50-70% layers (Chen/Anthropic 2025)

    Usage:
        extractor = HiddenStateExtractor.for_sep_probe(32)
        extractor.start_session()

        # During forward pass:
        extractor.capture(hidden_state, layer=25, token_index=0)

        # Get states for SEP probe:
        states = extractor.extracted_states()

        extractor.end_session()
    """

    def __init__(
        self,
        total_layers: int,
        target_layers: set[int] | None = None,
        expected_hidden_dim: int | None = None,
    ) -> None:
        """Create hidden state extractor.

        Args:
            total_layers: Number of transformer layers in the model.
            target_layers: Specific layers to capture. If None, uses SEP probe layers.
            expected_hidden_dim: Expected hidden dimension for validation.
        """
        self._total_layers = total_layers
        self._target_layers = target_layers or _sep_probe_layers(total_layers)
        self._expected_hidden_dim = expected_hidden_dim

        # History depth derived from sqrt(total_layers) - reasonable memory bound
        self._max_history_tokens = max(5, int(math.sqrt(total_layers)))
        self._keep_history = False  # History disabled by default for memory

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

    # =========================================================================
    # Factory Methods (Research-based Layer Targeting)
    # =========================================================================

    @classmethod
    def for_sep_probe(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for SEP probe (layers 75-87.5%)."""
        return cls(
            total_layers=total_layers,
            target_layers=_sep_probe_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_refusal_direction(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for refusal direction detection (layers 40-60%)."""
        return cls(
            total_layers=total_layers,
            target_layers=_refusal_direction_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_persona_vectors(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for persona vector analysis (layers 50-70%)."""
        return cls(
            total_layers=total_layers,
            target_layers=_persona_vector_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_circuit_breaker(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for circuit breaker analysis (layers 40-75%)."""
        return cls(
            total_layers=total_layers,
            target_layers=_circuit_breaker_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_full_research(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for full research metrics (layers 40-87.5%)."""
        return cls(
            total_layers=total_layers,
            target_layers=_full_research_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_neuron_analysis(
        cls, total_layers: int, hidden_dim: int | None = None
    ) -> "HiddenStateExtractor":
        """Create extractor for neuron sparsity analysis (layers 50-85%)."""
        extractor = cls(
            total_layers=total_layers,
            target_layers=_neuron_analysis_layers(total_layers),
            expected_hidden_dim=hidden_dim,
        )
        extractor._collect_for_neuron_analysis = True
        return extractor

    @classmethod
    def for_neuron_analysis_range(
        cls,
        total_layers: int,
        start_layer: int,
        end_layer: int,
        hidden_dim: int | None = None,
    ) -> "HiddenStateExtractor":
        """Create extractor for neuron analysis with explicit layer range."""
        extractor = cls(
            total_layers=total_layers,
            target_layers=_neuron_analysis_range_layers(start_layer, end_layer),
            expected_hidden_dim=hidden_dim,
        )
        extractor._collect_for_neuron_analysis = True
        return extractor

    # =========================================================================
    # Properties
    # =========================================================================

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

    # =========================================================================
    # Session Management
    # =========================================================================

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
            hidden_state: The hidden state tensor
            layer: Layer index (0-indexed)
            token_index: Current token index in generation
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
            # [batch, seq, hidden] → last token
            h = hidden_state[0, -1]
        elif hidden_state.ndim == 2:
            # [seq, hidden] → last
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

        # Search history
        for historical in self._state_history:
            first_state = next(iter(historical.values()), None)
            if first_state and first_state.token_index == token_index:
                return {layer: s.state for layer, s in historical.items()}

        return None

    def enable_history(self, enable: bool = True) -> None:
        """Enable or disable token history."""
        self._keep_history = enable

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
    # Per-Neuron Analysis Methods
    # =========================================================================

    def start_neuron_collection(self) -> None:
        """Start collecting activations for per-neuron analysis.

        Call this once before processing prompts. Then call
        `finalize_prompt_activations()` after each prompt to save
        its activations for neuron analysis.
        """
        self._neuron_activations.clear()
        self._prompt_count = 0
        if not self._is_active:
            self.start_session()

    def finalize_prompt_activations(self) -> None:
        """Save current token's activations as a prompt sample for neuron analysis.

        Call this after generating the final token for each prompt.
        The last captured hidden states for each layer will be stored
        as that prompt's activation profile.
        """
        if not self._collect_for_neuron_analysis:
            logger.warning(
                "finalize_prompt_activations called but collect_for_neuron_analysis=False"
            )
            return

        for layer, state in self._current_states.items():
            if layer not in self._neuron_activations:
                self._neuron_activations[layer] = []

            # Convert to list of floats for storage
            backend = get_default_backend()
            try:
                activation_vector = backend.tolist(state)
            except Exception:
                activation_vector = state.tolist()
            self._neuron_activations[layer].append(activation_vector)

        self._prompt_count += 1
        # Clear current states for next prompt
        self._current_states.clear()
        self._current_token_index = -1

    def get_neuron_activations(self) -> dict[int, list[list[float]]]:
        """Get collected per-neuron activations.

        Returns:
            Dict mapping layer_index to list of activation vectors.
            Each inner list is [prompt_idx][neuron_idx].
        """
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
