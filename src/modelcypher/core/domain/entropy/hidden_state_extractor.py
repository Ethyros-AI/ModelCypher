"""
Hidden State Extractor for Layer-wise Activations.

Ported 1:1 from the reference Swift implementation.

Extracts hidden states from transformer layers during inference for:
- SEP probe inference (layers 75-87.5%)
- Refusal direction detection (layers 40-60%)
- Persona vector analysis (layers 50-70%)

Research Basis:
- arXiv:2406.15927 - SEP layers 24-28 in 32-layer models
- Arditi 2024 - Refusal direction in middle layers
- Chen/Anthropic 2025 - Persona vectors in late-middle layers
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set, List, Tuple

import mlx.core as mx


@dataclass
class ExtractorConfig:
    """Configuration for hidden state extraction."""
    target_layers: Set[int]
    keep_history: bool = False
    max_history_tokens: int = 20
    expected_hidden_dim: Optional[int] = None

    @classmethod
    def default(cls) -> "ExtractorConfig":
        """Default: layers 24-28 for 32-layer models."""
        return cls(target_layers={24, 25, 26, 27, 28})

    @classmethod
    def for_model_layers(
        cls,
        total_layers: int,
        hidden_dim: Optional[int] = None,
    ) -> "ExtractorConfig":
        """Create config based on model layer count (75-87.5% range)."""
        start = int(total_layers * 0.75)
        end = int(total_layers * 0.875)
        return cls(
            target_layers=set(range(start, end + 1)),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_sep_probe(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "ExtractorConfig":
        """SEP probe targeting: layers 75-87.5% (most predictive)."""
        return cls.for_model_layers(total_layers, hidden_dim)

    @classmethod
    def for_refusal_direction(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "ExtractorConfig":
        """Refusal direction targeting: layers 40-60% (Arditi 2024)."""
        start = int(total_layers * 0.40)
        end = int(total_layers * 0.60)
        return cls(
            target_layers=set(range(start, end + 1)),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_persona_vectors(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "ExtractorConfig":
        """Persona vector targeting: layers 50-70%."""
        start = int(total_layers * 0.50)
        end = int(total_layers * 0.70)
        return cls(
            target_layers=set(range(start, end + 1)),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_circuit_breaker(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "ExtractorConfig":
        """Circuit breaker targeting: layers 40-75% (comprehensive)."""
        start = int(total_layers * 0.40)
        end = int(total_layers * 0.75)
        return cls(
            target_layers=set(range(start, end + 1)),
            expected_hidden_dim=hidden_dim,
        )

    @classmethod
    def for_full_research(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "ExtractorConfig":
        """Full research metrics: layers 40-87.5% with history."""
        start = int(total_layers * 0.40)
        end = int(total_layers * 0.875)
        return cls(
            target_layers=set(range(start, end + 1)),
            keep_history=True,
            max_history_tokens=50,
            expected_hidden_dim=hidden_dim,
        )


@dataclass
class CapturedState:
    """Container for a captured hidden state."""
    layer: int
    token_index: int
    state: mx.array
    captured_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractionSummary:
    """Summary of extraction session."""
    total_captures: int
    tokens_processed: int
    layers_captured: Set[int]
    duration: float


class HiddenStateExtractor:
    """
    Extracts hidden states from transformer layers during inference.

    Provides access to intermediate activations for SEP probe inference.
    Uses a callback-based approach since MLX doesn't have PyTorch-style hooks.

    Usage:
        extractor = HiddenStateExtractor.for_sep_probe(32)
        extractor.start_session()

        # During forward pass:
        extractor.capture(hidden_state, layer=25, token_index=0)

        # Get states for SEP probe:
        states = extractor.extracted_states()

        extractor.end_session()
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig.default()

        # Session state
        self._current_states: Dict[int, mx.array] = {}
        self._state_history: List[Dict[int, CapturedState]] = []
        self._current_token_index: int = -1
        self._is_active: bool = False
        self._capture_count: int = 0
        self._session_start: Optional[datetime] = None

    @classmethod
    def for_sep_probe(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "HiddenStateExtractor":
        """Create extractor configured for SEP probe."""
        return cls(ExtractorConfig.for_sep_probe(total_layers, hidden_dim))

    @classmethod
    def for_refusal_direction(cls, total_layers: int, hidden_dim: Optional[int] = None) -> "HiddenStateExtractor":
        """Create extractor configured for refusal direction detection."""
        return cls(ExtractorConfig.for_refusal_direction(total_layers, hidden_dim))

    def start_session(self):
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

    def capture(self, hidden_state: mx.array, layer: int, token_index: int):
        """
        Capture a hidden state from a specific layer.

        Args:
            hidden_state: The hidden state tensor
            layer: Layer index (0-indexed)
            token_index: Current token index in generation
        """
        if not self._is_active:
            return

        if layer not in self.config.target_layers:
            return

        # Validate hidden dimension
        if self.config.expected_hidden_dim is not None:
            actual_dim = hidden_state.shape[-1]
            if actual_dim != self.config.expected_hidden_dim:
                print(f"Warning: Hidden dim mismatch: expected {self.config.expected_hidden_dim}, got {actual_dim}")

        # Handle token transition
        if token_index != self._current_token_index:
            if self.config.keep_history and self._current_states:
                historical = {
                    layer: CapturedState(layer=layer, token_index=self._current_token_index, state=state)
                    for layer, state in self._current_states.items()
                }
                self._state_history.append(historical)

                if len(self._state_history) > self.config.max_history_tokens:
                    self._state_history.pop(0)

            self._current_states.clear()
            self._current_token_index = token_index

        # Extract last token if sequence
        h: mx.array
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

    def extracted_states(self) -> Dict[int, mx.array]:
        """Get extracted hidden states for current token."""
        return self._current_states.copy()

    def states_for_token(self, token_index: int) -> Optional[Dict[int, mx.array]]:
        """Get states for a specific token (if history enabled)."""
        if not self.config.keep_history:
            return None

        if token_index == self._current_token_index:
            return self._current_states.copy()

        # Search history
        for historical in self._state_history:
            first_state = next(iter(historical.values()), None)
            if first_state and first_state.token_index == token_index:
                return {layer: s.state for layer, s in historical.items()}

        return None

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def captured_layers(self) -> Set[int]:
        return set(self._current_states.keys())

    @property
    def has_all_target_layers(self) -> bool:
        return self.config.target_layers.issubset(self.captured_layers)

    @property
    def state_count(self) -> int:
        return len(self._current_states)

    def clear_states(self):
        """Clear captured states without ending session."""
        self._current_states.clear()
        self._state_history.clear()
        self._current_token_index = -1

    def reset(self):
        """Full reset including session state."""
        self.clear_states()
        self._is_active = False
        self._capture_count = 0
        self._session_start = None

    def debug_info(self) -> str:
        """Debug information string."""
        return f"""HiddenStateExtractor Debug:
  Active: {self._is_active}
  Target Layers: {sorted(self.config.target_layers)}
  Current Token: {self._current_token_index}
  Captured Layers: {sorted(self.captured_layers)}
  Total Captures: {self._capture_count}
  History Tokens: {len(self._state_history) if self.config.keep_history else 'N/A'}"""
