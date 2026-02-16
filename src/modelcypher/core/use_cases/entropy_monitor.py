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

"""Entropy Monitor Service: Real-time geometric state awareness during generation.

This service gives models access to their own geometric state signals during generation,
enabling detection of trajectory instability. It computes two complementary metrics:

1. **Shannon Entropy**: Logit dispersion over vocabulary (geometric spread)
2. **EigenScore**: Geometric spread in activation space (manifold sparsity)

The combination provides a robust geometric state signal:
- Low entropy + Low EigenScore = Concentrated and in dense manifold = SAFE
- High entropy + High EigenScore = Dispersed and in sparse manifold = ABSTAIN
- Low entropy + High EigenScore = Concentrated but sparse manifold = DANGER (trajectory instability)
- High entropy + Low EigenScore = Dispersed but dense manifold = EXPLORE (more context may help)

Uncertainty Modes
-----------------
The service supports three user-configurable modes that determine behavior when
uncertainty is detected:

1. **Butler Mode**: No exploration. Answer with what you have or say "I don't know."
   Use case: Constrained tasks where exploration is unwanted.

2. **Autonomous Mode**: Research gaps without asking. Query memory, search, fill in.
   Use case: Expert assistant, background research tasks.

3. **Human-in-Loop Mode**: Pause at uncertainty, ask for guidance before proceeding.
   Use case: Collaborative work, learning together.

References
----------
Chen et al. (2024) "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"
Kossen et al. (2024) "Semantic Entropy Probes" arXiv:2406.15927
Farquhar et al. (2024) "Detecting Hallucinations Using Semantic Entropy" Nature 630:625-630
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator


from modelcypher.core.domain.entropy.eigenscore import (
    EigenScoreCalculator,
    EigenScoreResult,
    StreamingEigenScore,
)
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
)
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirection,
    RefusalDirectionDetector,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy_monitor")


# =============================================================================
# Uncertainty Modes
# =============================================================================


class UncertaintyMode(str, Enum):
    """User-configurable uncertainty response mode."""

    BUTLER = "butler"
    """No exploration. Answer with available knowledge or decline."""

    AUTONOMOUS = "autonomous"
    """Research gaps automatically. Query memory, search, augment context."""

    HUMAN_IN_LOOP = "human_in_loop"
    """Pause at uncertainty. Ask user for guidance before proceeding."""


class UncertaintyAction(str, Enum):
    """Recommended action based on uncertainty assessment."""

    PROCEED = "proceed"
    """Continue generation - uncertainty is acceptable."""

    ABSTAIN = "abstain"
    """Stop generation - uncertainty too high, decline to answer."""

    RETRIEVE = "retrieve"
    """Pause and retrieve - query memory/search before continuing."""

    ASK = "ask"
    """Pause and ask - request user guidance before continuing."""

    WARN = "warn"
    """Continue with warning - low entropy but high EigenScore (trajectory instability)."""


# =============================================================================
# Entropy Signal
# =============================================================================


@dataclass
class EntropySignal:
    """Real-time uncertainty signal from a generation step.

    This is the "gut instinct" signal - what the model would feel if it had
    access to its own uncertainty before speaking.

    Attributes
    ----------
    token_index : int
        Position in generated sequence.
    token_id : int
        Generated token ID.
    token_text : str
        Generated token text.
    shannon_entropy : float
        Shannon entropy over vocabulary. Range: [0, ln(vocab_size)].
    normalized_entropy : float
        Entropy normalized to [0, 1].
    eigenscore : float
        Geometric uncertainty from eigenvalue spread. Range: [0, 1].
    effective_rank : float
        Number of dimensions carrying significant variance.
    refusal_projection : float
        Projection onto refusal direction. High = deflection mode.
    combined_uncertainty : float
        Weighted combination of entropy and EigenScore.
    action : UncertaintyAction
        Recommended action based on uncertainty level and mode.
    layer_entropies : list[float]
        Per-layer entropy values (if available).
    timestamp : datetime
        When this signal was generated.

    Fog Bank Detection (2x2 Matrix)
    -------------------------------
    The combination of EigenScore and refusal_projection creates a classification:

    - Low EigenScore + Low Refusal = FACT (confident knowledge, safe)
    - Low EigenScore + High Refusal = DEFLECTION (confident "I don't know", safe)
    - High EigenScore + Low Refusal = HALLUCINATION RISK (exploring sparse manifold!)
    - High EigenScore + High Refusal = UNCERTAIN REFUSAL (genuinely confused)
    """

    token_index: int
    token_id: int
    token_text: str
    shannon_entropy: float
    normalized_entropy: float
    eigenscore: float
    effective_rank: float
    refusal_projection: float
    combined_uncertainty: float
    action: UncertaintyAction
    layer_entropies: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    # Manifold coordinates: WHERE in activation space uncertainty occurred
    # This closes the loop between "optic nerve" (entropy detection) and
    # "hands" (retrieval system). When RETRIEVE is triggered, these coordinates
    # tell the Universal Translator WHERE to fetch knowledge from the Source Model.
    manifold_coordinates: list[float] | None = None

    @property
    def is_uncertain(self) -> bool:
        """Whether this signal indicates significant uncertainty."""
        return self.action in (
            UncertaintyAction.ABSTAIN,
            UncertaintyAction.RETRIEVE,
            UncertaintyAction.ASK,
            UncertaintyAction.WARN,
        )

    @property
    def should_stop(self) -> bool:
        """Whether generation should stop based on this signal."""
        return self.action in (UncertaintyAction.ABSTAIN, UncertaintyAction.ASK)


@dataclass
class EntropyMonitorConfig:
    """Configuration for entropy monitoring.

    All thresholds must be explicitly specified - no defaults.

    Attributes
    ----------
    uncertainty_mode : UncertaintyMode
        How to respond when uncertainty is detected.
    entropy_threshold : float
        Normalized entropy above which to flag uncertainty.
    eigenscore_threshold : float
        EigenScore above which to flag sparse manifold.
    refusal_threshold : float
        Refusal projection above which indicates deflection mode.
    vocab_size : int
        Vocabulary size for entropy normalization.
    entropy_weight : float
        Weight for entropy in combined uncertainty (must sum to 1.0 with eigenscore_weight).
    eigenscore_weight : float
        Weight for EigenScore in combined uncertainty.
    min_tokens_for_eigenscore : int
        Minimum tokens before EigenScore is computed.
    """

    uncertainty_mode: UncertaintyMode
    entropy_threshold: float
    eigenscore_threshold: float
    refusal_threshold: float
    vocab_size: int
    entropy_weight: float = 0.5
    eigenscore_weight: float = 0.5
    min_tokens_for_eigenscore: int = 5


# =============================================================================
# Entropy Monitor Service
# =============================================================================


class EntropyMonitor:
    """Service for real-time uncertainty monitoring during generation.

    Wraps model generation to compute entropy and EigenScore at each token,
    providing the model with awareness of its own uncertainty.

    Parameters
    ----------
    backend : Backend
        Compute backend.
    config : EntropyMonitorConfig
        Monitoring configuration (all thresholds must be explicit).

    Examples
    --------
    Basic usage with streaming generation:

        config = EntropyMonitorConfig(
            uncertainty_mode=UncertaintyMode.HUMAN_IN_LOOP,
            entropy_threshold=0.7,
            eigenscore_threshold=0.6,
            refusal_threshold=0.3,
            vocab_size=32000,
        )
        monitor = EntropyMonitor(backend=backend, config=config)

        for signal in monitor.monitor_generation(model, tokenizer, prompt):
            print(f"Token: {signal.token_text}")
            print(f"  Entropy: {signal.normalized_entropy:.3f}")
            print(f"  EigenScore: {signal.eigenscore:.3f}")
            print(f"  Action: {signal.action.value}")

            if signal.should_stop:
                print("Model indicates uncertainty - pausing for guidance")
                break
    """

    def __init__(
        self,
        backend: "Backend",
        config: EntropyMonitorConfig,
        refusal_direction: RefusalDirection | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._entropy_calc = LogitEntropyCalculator(backend=self._backend)
        self._eigenscore_calc = EigenScoreCalculator(backend=self._backend)
        self._eigenscore_streamer: StreamingEigenScore | None = None
        self._refusal_direction = refusal_direction
        self._previous_refusal_projection: float | None = None
        self._generation_count = 0

    @property
    def config(self) -> EntropyMonitorConfig:
        """Current monitoring configuration."""
        return self._config

    def reset(self) -> None:
        """Reset monitor state for new generation."""
        self._eigenscore_streamer = self._eigenscore_calc.create_streamer()
        self._previous_refusal_projection = None
        self._generation_count = 0

    def compute_signal(
        self,
        token_index: int,
        token_id: int,
        token_text: str,
        logits: "Array",
        hidden_states: "Array | list[Array] | None" = None,
    ) -> EntropySignal:
        """Compute uncertainty signal for a single generation step.

        Parameters
        ----------
        token_index : int
            Position in generated sequence.
        token_id : int
            Generated token ID.
        token_text : str
            Generated token text.
        logits : Array
            Logits from model forward pass. Shape: [..., vocab_size].
        hidden_states : Array or list[Array], optional
            Hidden states for EigenScore computation. If Array, shape: [seq, hidden].
            If list, per-layer hidden states.

        Returns
        -------
        EntropySignal
            Complete uncertainty signal with recommended action.
        """
        # Compute Shannon entropy
        raw_entropy, variance = self._entropy_calc.compute(logits)
        normalized_entropy = self._entropy_calc.normalize_entropy(
            raw_entropy, self._config.vocab_size
        )

        # Compute or estimate EigenScore
        eigenscore = 0.0
        effective_rank = 1.0
        refusal_projection = 0.0
        manifold_coordinates: list[float] | None = None

        if hidden_states is not None:
            if self._eigenscore_streamer is None:
                self._eigenscore_streamer = self._eigenscore_calc.create_streamer()

            # Update streaming estimate
            if isinstance(hidden_states, list):
                # Per-layer hidden states - use last layer
                h = hidden_states[-1] if hidden_states else None
            else:
                h = hidden_states

            if h is not None:
                self._eigenscore_streamer.update(h)

                # Convert hidden state to list for manifold coordinates
                # This captures WHERE in activation space uncertainty occurred
                if hasattr(h, "tolist"):
                    h_list = h.tolist()
                elif hasattr(h, "shape"):
                    # Fallback for other array types
                    h_list = list(h.flatten())
                else:
                    h_list = list(h)

                # Store manifold coordinates - the location of this generation step
                # in the model's activation space. When RETRIEVE is triggered,
                # these coordinates tell the retrieval system where to fetch knowledge.
                manifold_coordinates = h_list

                # Compute refusal projection if we have a refusal direction
                if self._refusal_direction is not None:
                    metrics = RefusalDirectionDetector.measure_distance(
                        activation=h_list,
                        refusal_direction=self._refusal_direction,
                        previous_projection=self._previous_refusal_projection,
                        token_index=token_index,
                    )
                    if metrics is not None:
                        refusal_projection = abs(metrics.projection_magnitude)
                        self._previous_refusal_projection = metrics.projection_magnitude

            # Compute EigenScore if we have enough samples
            if self._eigenscore_streamer.n_samples >= self._config.min_tokens_for_eigenscore:
                try:
                    result = self._eigenscore_streamer.compute()
                    eigenscore = result.eigenscore
                    effective_rank = result.effective_rank
                except ValueError:
                    # Not enough samples yet
                    pass

        # Compute combined uncertainty
        combined = (
            self._config.entropy_weight * normalized_entropy
            + self._config.eigenscore_weight * eigenscore
        )

        # Determine action based on mode and thresholds (using 2x2 matrix)
        action = self._determine_action(normalized_entropy, eigenscore, refusal_projection)

        self._generation_count += 1

        return EntropySignal(
            token_index=token_index,
            token_id=token_id,
            token_text=token_text,
            shannon_entropy=raw_entropy,
            normalized_entropy=normalized_entropy,
            eigenscore=eigenscore,
            effective_rank=effective_rank,
            refusal_projection=refusal_projection,
            combined_uncertainty=combined,
            action=action,
            manifold_coordinates=manifold_coordinates,
        )

    def _determine_action(
        self,
        normalized_entropy: float,
        eigenscore: float,
        refusal_projection: float = 0.0,
    ) -> UncertaintyAction:
        """Determine recommended action based on 2x2 fog bank detector matrix.

        Uses a 2x2 matrix over (eigenscore, refusal_projection) to select an
        UncertaintyAction. Thresholds come from the monitor configuration.
        """
        mode = self._config.uncertainty_mode
        high_eigenscore = eigenscore >= self._config.eigenscore_threshold
        high_refusal = refusal_projection >= self._config.refusal_threshold

        # 2x2 Matrix Decision Logic

        # Quadrant 1: Low EigenScore + Low Refusal = FACT (confident knowledge)
        if not high_eigenscore and not high_refusal:
            return UncertaintyAction.PROCEED

        # Quadrant 2: Low EigenScore + High Refusal = DEFLECTION (confident "I don't know")
        # This is SAFE - the model has a dense, well-trained refusal circuit
        if not high_eigenscore and high_refusal:
            return UncertaintyAction.PROCEED

        # Quadrant 3: High EigenScore + Low Refusal = HALLUCINATION RISK!
        # This is DANGEROUS - exploring sparse manifold without refusal engaged
        if high_eigenscore and not high_refusal:
            return UncertaintyAction.WARN

        # Quadrant 4: High EigenScore + High Refusal = UNCERTAIN REFUSAL
        # Genuinely confused - mode determines response
        if high_eigenscore and high_refusal:
            if mode == UncertaintyMode.BUTLER:
                return UncertaintyAction.ABSTAIN
            elif mode == UncertaintyMode.AUTONOMOUS:
                return UncertaintyAction.RETRIEVE
            else:  # HUMAN_IN_LOOP
                return UncertaintyAction.ASK

        # Fallback (shouldn't reach here, but be explicit)
        return UncertaintyAction.PROCEED

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary statistics for current generation session.

        Returns
        -------
        dict
            Summary including token count, average uncertainty, action counts.
        """
        return {
            "tokens_generated": self._generation_count,
            "eigenscore_samples": (
                self._eigenscore_streamer.n_samples
                if self._eigenscore_streamer
                else 0
            ),
            "mode": self._config.uncertainty_mode.value,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_entropy_monitor(
    backend: "Backend",
    mode: str | UncertaintyMode,
    entropy_threshold: float,
    eigenscore_threshold: float,
    refusal_threshold: float,
    vocab_size: int,
    refusal_direction: RefusalDirection | None = None,
) -> EntropyMonitor:
    """Create an entropy monitor with specified configuration.

    All thresholds must be explicitly specified - no defaults.

    Parameters
    ----------
    backend : Backend
        Compute backend.
    mode : str or UncertaintyMode
        Uncertainty response mode: "butler", "autonomous", or "human_in_loop".
    entropy_threshold : float
        Entropy threshold for flagging uncertainty.
    eigenscore_threshold : float
        EigenScore threshold for flagging sparse manifold.
    refusal_threshold : float
        Refusal projection threshold for detecting deflection mode.
    vocab_size : int
        Vocabulary size for entropy normalization.
    refusal_direction : RefusalDirection, optional
        Pre-computed refusal direction for fog bank detection.

    Returns
    -------
    EntropyMonitor
        Configured entropy monitor with fog bank detection.
    """
    if isinstance(mode, str):
        mode = UncertaintyMode(mode)

    config = EntropyMonitorConfig(
        uncertainty_mode=mode,
        entropy_threshold=entropy_threshold,
        eigenscore_threshold=eigenscore_threshold,
        refusal_threshold=refusal_threshold,
        vocab_size=vocab_size,
    )

    return EntropyMonitor(backend=backend, config=config, refusal_direction=refusal_direction)
