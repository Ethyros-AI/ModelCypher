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

"""Entropy-Learning Bridge: Connect fog bank detector to continual learning.

This module bridges Phase 1 (Entropy Sense) to Phase 2 (Adaptive Geometry):

1. **Signal Conversion**: Converts EntropySignal (fog bank) to EntropyState (continual)
2. **Surprise Integration**: Routes WARN signals to SurpriseDetector for learning
3. **Sparsity Marking**: Marks sparse manifold regions for consolidation
4. **Confidence Feedback**: Encodes uncertainty back into generation via ConfidenceEmbedding

The bridge enables a feedback cycle:
    Generate → Entropy Detection → Update Learning Signals → Adjust Embeddings → Generate

Architecture:
    ┌─────────────────┐         ┌──────────────────┐
    │  EntropyMonitor │────────▶│  EntropyLearning │
    │  (Fog Bank)     │         │  Bridge          │
    └─────────────────┘         └────────┬─────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
    ┌──────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
    │ SurpriseDetector │     │ ConfidenceEmbed   │     │ NullSpaceTracker    │
    │ (Learning Signal)│     │ (Feedback Loop)   │     │ (Sparsity Marking)  │
    └──────────────────┘     └───────────────────┘     └─────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


from modelcypher.core.domain.continual.confidence_embedding import ConfidenceEmbedding
from modelcypher.core.domain.continual.entropy_analyzer import EntropyState
from modelcypher.core.domain.continual.lora_memory_store import HeatSignal
from modelcypher.core.domain.continual.surprise_detector import (
    SurpriseDetector,
    SurpriseEvent,
)
from modelcypher.core.use_cases.entropy_monitor import (
    EntropySignal,
    UncertaintyAction,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy_learning_bridge")


@dataclass
class SparsityEvent:
    """A detected sparse manifold region for consolidation.

    Attributes
    ----------
    token_index : int
        Position in sequence where sparsity detected.
    eigenscore : float
        Manifold sparsity at detection time.
    refusal_projection : float
        Refusal activation at detection time.
    action : UncertaintyAction
        The action that triggered this event.
    hidden_state_hash : int
        Hash of hidden state for deduplication.
    layer_index : int
        Layer where sparsity was most pronounced (-1 if unknown).
    manifold_coordinates : list[float] | None
        WHERE in activation space the sparsity was detected.
        This bridges the "optic nerve" to the "hands" - when RETRIEVE is
        triggered, these coordinates tell the Universal Translator where
        to fetch knowledge from the Source Model.
    hidden_state_key : str | None
        Key for retrieving the actual hidden state tensor from the bridge.
        Only populated when retain_hidden_states=True.
    """

    token_index: int
    eigenscore: float
    refusal_projection: float
    action: UncertaintyAction
    hidden_state_hash: int
    layer_index: int = -1
    manifold_coordinates: list[float] | None = None
    hidden_state_key: str | None = None


@dataclass
class BridgeStats:
    """Statistics from the entropy-learning bridge.

    Attributes
    ----------
    signals_processed : int
        Total EntropySignals processed.
    warn_events : int
        Number of WARN (hallucination risk) signals.
    sparsity_events : int
        Number of sparsity events queued for consolidation.
    confidence_injections : int
        Number of confidence embeddings injected.
    """

    signals_processed: int = 0
    warn_events: int = 0
    sparsity_events: int = 0
    confidence_injections: int = 0


class EntropyLearningBridge:
    """Bridge between fog bank detector and continual learning system.

    This service connects Phase 1 (Entropy Sense) to Phase 2 (Adaptive Geometry),
    enabling the model to learn from its own uncertainty signals.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension for confidence embedding.
    backend : Backend, optional
        Compute backend.
    null_space_tracker : NullSpaceTracker, optional
        Tracker for marking sparse regions.

    Examples
    --------
    Basic usage in generation loop:

        bridge = EntropyLearningBridge(hidden_dim=576)

        for token_idx, (logits, hidden_state) in enumerate(generation_loop):
            # Phase 1: Fog bank detection
            signal = monitor.compute_signal(
                token_index=token_idx,
                token_id=token_id,
                token_text=text,
                logits=logits,
                hidden_states=hidden_state,
            )

            # Bridge: Process signal and get feedback
            feedback = bridge.process_signal(
                signal=signal,
                logits=logits,
                actual_token_id=token_id,
                hidden_state=hidden_state,
            )

            # Inject confidence embedding if available
            if feedback.confidence_embedding is not None:
                hidden_state = hidden_state + feedback.confidence_embedding

            # Act on recommended action
            if signal.action == UncertaintyAction.WARN:
                logger.warning("Hallucination risk detected!")
    """

    def __init__(
        self,
        hidden_dim: int,
        backend: "Backend",
        null_space_tracker: "NullSpaceTracker | None" = None,
        retain_hidden_states: bool = False,
    ) -> None:
        self._backend = backend
        self._hidden_dim = hidden_dim
        self._retain_hidden_states = retain_hidden_states

        # Initialize components
        self._surprise_detector = SurpriseDetector(backend=self._backend)
        self._confidence_embedding = ConfidenceEmbedding(
            hidden_dim=hidden_dim,
            backend=self._backend,
        )
        self._null_space_tracker = null_space_tracker

        # State tracking
        self._sparsity_queue: list[SparsityEvent] = []
        self._stats = BridgeStats()
        self._previous_entropy_derivative = 0.0

        # Hidden state storage for LoRA memory (only when retain_hidden_states=True)
        self._hidden_states: dict[str, "Array"] = {}

    def process_signal(
        self,
        signal: EntropySignal,
        logits: "Array",
        actual_token_id: int,
        hidden_state: "Array | None" = None,
    ) -> "BridgeFeedback":
        """Process an EntropySignal and route to appropriate learning systems.

        Parameters
        ----------
        signal : EntropySignal
            Fog bank detection signal.
        logits : Array
            Model logits for this token.
        actual_token_id : int
            The actual token that was generated/selected.
        hidden_state : Array, optional
            Hidden state for surprise detection and confidence embedding.

        Returns
        -------
        BridgeFeedback
            Feedback including confidence embedding and surprise event.
        """
        self._stats.signals_processed += 1

        # Convert to EntropyState for downstream components
        entropy_state = self._signal_to_state(signal)

        # Detect surprise for learning
        surprise_event = self._surprise_detector.detect(
            logits=logits,
            actual_token_id=actual_token_id,
            hidden_state=hidden_state,
        )

        # Handle hallucination risk signals
        if signal.action == UncertaintyAction.WARN:
            self._stats.warn_events += 1
            self._handle_hallucination_risk(signal, hidden_state)

        # Generate confidence embedding for feedback loop
        confidence_embedding = None
        if hidden_state is not None:
            confidence_embedding = self._confidence_embedding.encode(entropy_state)
            self._stats.confidence_injections += 1

        # Update derivative for next iteration
        self._previous_entropy_derivative = signal.normalized_entropy

        return BridgeFeedback(
            entropy_state=entropy_state,
            surprise_event=surprise_event,
            confidence_embedding=confidence_embedding,
            is_hallucination_risk=(signal.action == UncertaintyAction.WARN),
            sparsity_queued=len(self._sparsity_queue),
            manifold_coordinates=signal.manifold_coordinates,
        )

    def _signal_to_state(self, signal: EntropySignal) -> EntropyState:
        """Convert EntropySignal to EntropyState for continual learning components.

        The EntropyState format is used by ConfidenceEmbedding and other
        continual learning components.
        """
        # Compute derivative (change from previous)
        derivative = signal.normalized_entropy - self._previous_entropy_derivative

        return EntropyState(
            entropy=signal.shannon_entropy,
            entropy_normalized=signal.normalized_entropy,
            entropy_derivative=derivative,
            entropy_acceleration=0.0,  # Would need more history
            logit_variance=signal.combined_uncertainty,  # Use combined as proxy
            vocab_size=32000,  # Default, could be passed in
            timestep=signal.token_index,
        )

    def _handle_hallucination_risk(
        self,
        signal: EntropySignal,
        hidden_state: "Array | None",
    ) -> None:
        """Handle a hallucination risk signal by marking for consolidation."""
        # Compute hash for deduplication
        if hidden_state is not None:
            h_flat = self._backend.reshape(hidden_state, (-1,))
            self._backend.eval(h_flat)
            # Simple hash: sum of absolute values
            hash_val = int(abs(float(self._backend.sum(self._backend.abs(h_flat)))) * 1e6)
        else:
            hash_val = hash((signal.token_index, signal.eigenscore))

        # Generate hidden state key for retrieval (if retaining)
        hidden_state_key: str | None = None
        if self._retain_hidden_states and hidden_state is not None:
            hidden_state_key = f"sparsity_{signal.token_index}_{hash_val}"
            self._hidden_states[hidden_state_key] = hidden_state

        # Create sparsity event with manifold coordinates
        # This wires the "optic nerve" to the "hands" - coordinates are passed
        # from EntropySignal through to SparsityEvent for retrieval targeting
        event = SparsityEvent(
            token_index=signal.token_index,
            eigenscore=signal.eigenscore,
            refusal_projection=signal.refusal_projection,
            action=signal.action,
            hidden_state_hash=hash_val,
            hidden_state_key=hidden_state_key,
            manifold_coordinates=signal.manifold_coordinates,
        )

        self._sparsity_queue.append(event)
        self._stats.sparsity_events += 1

        logger.info(
            "Sparsity event queued: token=%d, eigenscore=%.3f, refusal=%.3f%s",
            signal.token_index,
            signal.eigenscore,
            signal.refusal_projection,
            " (state retained)" if hidden_state_key else "",
        )

        # Mark in null space tracker if available
        if self._null_space_tracker is not None and hidden_state is not None:
            # Add to all layers (we don't know which layer is most sparse)
            # This is a simple approach; could be refined
            for layer_id in range(self._null_space_tracker._n_layers):
                self._null_space_tracker.add_activation(layer_id, hidden_state)

    def get_sparsity_queue(self) -> list[SparsityEvent]:
        """Get queued sparsity events for consolidation."""
        return list(self._sparsity_queue)

    def get_hidden_states(self) -> dict[str, "Array"]:
        """Get retained hidden states for LoRA memory.

        Only populated when retain_hidden_states=True was passed to __init__.

        Returns
        -------
        dict[str, Array]
            Mapping from hidden_state_key to actual tensor.
        """
        return dict(self._hidden_states)

    def clear_sparsity_queue(self) -> int:
        """Clear the sparsity queue, returning number of events cleared."""
        count = len(self._sparsity_queue)
        self._sparsity_queue.clear()
        return count

    def clear_hidden_states(self) -> int:
        """Clear retained hidden states, returning number cleared."""
        count = len(self._hidden_states)
        self._hidden_states.clear()
        return count

    def get_stats(self) -> BridgeStats:
        """Get bridge statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset bridge state for new generation."""
        self._surprise_detector.reset()
        self._previous_entropy_derivative = 0.0
        self._stats = BridgeStats()
        # Note: sparsity queue is NOT cleared - it persists for consolidation
        # Note: hidden states are NOT cleared - they persist for LoRA memory

    def inject_confidence(
        self,
        hidden_state: "Array",
        signal: EntropySignal,
    ) -> "Array":
        """Convenience method to inject confidence embedding into hidden state.

        Parameters
        ----------
        hidden_state : Array
            Current hidden state.
        signal : EntropySignal
            Current entropy signal.

        Returns
        -------
        Array
            Hidden state with confidence embedding injected.
        """
        entropy_state = self._signal_to_state(signal)
        return self._confidence_embedding.inject_into_residual(
            hidden_state=hidden_state,
            entropy_state=entropy_state,
        )

    def compute_heat_signal(
        self,
        surprise_event: SurpriseEvent,
        entropy_state: EntropyState,
        preserved_fraction: float,
        capacity_fraction: float = 0.0,
        eigenscore: float = 0.0,
    ) -> HeatSignal:
        """Compute heat signal for memory promotion decisions.

        Heat measures learning opportunity:
            HEAT = surprise_percentile × preserved_fraction × entropy_stability

        Where:
        - surprise_percentile ∈ [0, 1]: How novel this event was
        - preserved_fraction ∈ [0, 1]: What survived null-space projection
        - entropy_stability ∈ [0, 1]: H × (1 - |dH/dt| / max(H, √ε))

        All signals are raw measurements. No thresholds - heat is continuous [0, 1].

        Parameters
        ----------
        surprise_event : SurpriseEvent
            Surprise detection result with percentile.
        entropy_state : EntropyState
            Current entropy state with derivative.
        preserved_fraction : float
            Behavioral norm ratio from null-space projection [0, 1].
        capacity_fraction : float
            Null space available at this layer [0, 1].
        eigenscore : float
            Manifold sparsity at event time.

        Returns
        -------
        HeatSignal
            Computed heat with all component signals.
        """
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        b = self._backend
        eps = float(machine_epsilon(b, b.array([1.0])))
        sqrt_eps = eps ** 0.5

        # Surprise in [0, 1] from percentile
        surprise_percentile = max(0.0, min(1.0, surprise_event.percentile))

        # Preserved fraction in [0, 1]
        preserved = max(0.0, min(1.0, preserved_fraction))

        # Entropy stability: high entropy but not rapidly converging
        # stability = H × (1 - |dH/dt| / max(H, sqrt_eps))
        H = entropy_state.entropy_normalized
        dH_dt = abs(entropy_state.entropy_derivative)

        if H > sqrt_eps:
            # Normalize derivative by entropy (relative rate of change)
            relative_derivative = dH_dt / H
            # Stability is high when entropy is high but not changing fast
            stability = H * max(0.0, 1.0 - relative_derivative)
        else:
            # Very low entropy = confident, not a learning opportunity
            stability = 0.0

        # Heat is the product of all three factors
        heat = surprise_percentile * preserved * stability

        return HeatSignal(
            timestamp=entropy_state.timestep,
            surprise_percentile=surprise_percentile,
            preserved_fraction=preserved,
            entropy_normalized=H,
            entropy_derivative_abs=dH_dt,
            heat=heat,
            eigenscore=eigenscore,
            capacity_fraction=capacity_fraction,
        )


@dataclass
class BridgeFeedback:
    """Feedback from processing an EntropySignal.

    Attributes
    ----------
    entropy_state : EntropyState
        Converted entropy state for downstream use.
    surprise_event : SurpriseEvent
        Detected surprise for learning.
    confidence_embedding : Array or None
        Embedding to inject into residual stream.
    is_hallucination_risk : bool
        Whether this signal indicates hallucination risk.
    sparsity_queued : int
        Number of sparsity events in queue.
    manifold_coordinates : list[float] | None
        WHERE in activation space this signal occurred.
        When RETRIEVE action is recommended, these coordinates enable
        the retrieval system to fetch the right concept from the Source Model.
    """

    entropy_state: EntropyState
    surprise_event: SurpriseEvent
    confidence_embedding: Any  # Array or None
    is_hallucination_risk: bool
    sparsity_queued: int
    manifold_coordinates: list[float] | None = None


def create_entropy_learning_bridge(
    hidden_dim: int,
    null_space_tracker: "NullSpaceTracker | None" = None,
    retain_hidden_states: bool = False,
) -> EntropyLearningBridge:
    """Create an entropy-learning bridge.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    null_space_tracker : NullSpaceTracker, optional
        Tracker for marking sparse regions.
    retain_hidden_states : bool, default False
        If True, retain actual hidden state tensors for LoRA memory.
        This uses more memory but enables later consolidation.

    Returns
    -------
    EntropyLearningBridge
        Configured bridge.
    """
    return EntropyLearningBridge(
        hidden_dim=hidden_dim,
        null_space_tracker=null_space_tracker,
        retain_hidden_states=retain_hidden_states,
    )
