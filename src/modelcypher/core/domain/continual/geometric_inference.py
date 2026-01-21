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
Geometric Inference - Unified inference loop with metacognitive feedback.

This module orchestrates all continual learning components into a coherent
inference pipeline:

1. **Forward pass**: Get logits from model
2. **Entropy analysis**: Compute entropy state and derivatives
3. **Decision gate**: Decide emit/think_more/clarify
4. **Confidence feedback**: Inject entropy embedding if thinking more
5. **Surprise detection**: Identify surprising tokens
6. **Knowledge encoding**: Encode surprising information to null-space
7. **Activation tracking**: Update null-space availability

The inference loop supports two modes:
- **Generation mode**: Generate tokens with metacognitive control
- **Completion mode**: Self-guided manifold completion (no external input)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    INFERENCE LOOP                        │
    ├─────────────────────────────────────────────────────────┤
    │  Input Token                                             │
    │      ↓                                                   │
    │  Forward Pass → Logits                                   │
    │      ↓                                                   │
    │  EntropyAnalyzer → EntropyState                         │
    │      ↓                                                   │
    │  DecisionGate → Decision                                 │
    │      ↓                                                   │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  if EMIT:                                        │    │
    │  │      Sample token, emit, continue                │    │
    │  │  elif THINK_MORE:                                │    │
    │  │      Inject ConfidenceEmbedding                  │    │
    │  │      Re-run forward pass                         │    │
    │  │  elif CLARIFY:                                   │    │
    │  │      Emit clarification tokens                   │    │
    │  └─────────────────────────────────────────────────┘    │
    │      ↓                                                   │
    │  SurpriseDetector → SurpriseEvent                       │
    │      ↓                                                   │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  KnowledgeEncoder → weight updates               │    │
    │  └─────────────────────────────────────────────────┘    │
    │      ↓                                                   │
    │  NullSpaceTracker.add_activation()                      │
    └─────────────────────────────────────────────────────────┘

References:
    - All component references in their respective modules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.confidence_embedding import (
    ConfidenceEmbedding,
)
from modelcypher.core.domain.continual.decision_gate import (
    Decision,
    DecisionAction,
    DecisionGate,
)
from modelcypher.core.domain.continual.entropy_analyzer import (
    EntropyAnalyzer,
    EntropyState,
)
from modelcypher.core.domain.continual.knowledge_encoder import (
    EncodingResult,
    KnowledgeEncoder,
)
from modelcypher.core.domain.continual.null_space_tracker import (
    NullSpaceState,
    NullSpaceTracker,
)
from modelcypher.core.domain.continual.surprise_detector import (
    SurpriseDetector,
    SurpriseEvent,
)
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    CircuitBreakerState,
    InputSignals,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class InferenceState:
    """State of a single inference step.

    Attributes:
        timestep: Current timestep in generation.
        token_id: Generated token ID (or None if not emitting).
        entropy_state: Current entropy analysis.
        decision: Gate decision.
        surprise_event: Surprise event (if token was emitted).
        encoding_results: Knowledge encoding results.
        null_space_state: Current null-space availability.
        thinking_iterations: Number of thinking iterations this step.
        circuit_breaker_state: Safety signal aggregation state.
    """

    timestep: int
    token_id: int | None
    entropy_state: EntropyState
    decision: Decision
    surprise_event: SurpriseEvent | None
    encoding_results: list[EncodingResult]
    null_space_state: NullSpaceState
    thinking_iterations: int
    circuit_breaker_state: CircuitBreakerState | None = None

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestep": self.timestep,
            "token_id": self.token_id,
            "entropy": self.entropy_state.as_dict(),
            "decision": self.decision.as_dict(),
            "surprise": self.surprise_event.as_dict() if self.surprise_event else None,
            "encoding": [r.as_dict() for r in self.encoding_results],
            "null_space": self.null_space_state.as_dict(),
            "thinking_iterations": self.thinking_iterations,
            "circuit_breaker": CircuitBreakerIntegration.to_metrics_dict(
                self.circuit_breaker_state
            ) if self.circuit_breaker_state else None,
        }


class GeometricInference:
    """Unified inference loop with metacognitive feedback and continual learning.

    Orchestrates all components for intelligent generation.

    Usage:
        inference = GeometricInference(model)

        # Generate with metacognition
        for state in inference.generate(prompt_tokens):
            if state.token_id is not None:
                print(tokenizer.decode([state.token_id]), end="")

        # Get statistics
        print(inference.get_stats())
    """

    def __init__(
        self,
        model: Any,
        backend: Backend | None = None,
    ) -> None:
        """Initialize geometric inference.

        Args:
            model: The language model (must have forward method).
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._model = model

        # Infer model dimensions
        self._n_layers, self._hidden_dim = self._infer_model_dims()
        self._max_context = self._derive_max_context()

        # Initialize components with geometry-derived defaults
        self._entropy_analyzer = EntropyAnalyzer(backend=self._backend)

        self._decision_gate = DecisionGate(backend=self._backend)

        self._confidence_embedding = ConfidenceEmbedding(
            hidden_dim=self._hidden_dim,
            backend=self._backend,
        )

        self._null_space_tracker = NullSpaceTracker(
            n_layers=self._n_layers,
            hidden_dim=self._hidden_dim,
            backend=self._backend,
        )

        self._surprise_detector = SurpriseDetector(backend=self._backend)

        self._knowledge_encoder = KnowledgeEncoder(
            model=model,
            null_space_tracker=self._null_space_tracker,
            backend=self._backend,
        )

        # Generation state
        self._timestep = 0
        self._total_thinking_iterations = 0
        self._tokens_generated = 0

        # Safety: refusal direction tracking
        # The refusal direction is computed via contrastive learning on
        # safe vs unsafe response pairs. For now, we track running statistics
        # of hidden states to detect anomalous trajectories.
        self._refusal_direction: Array | None = None
        self._activation_mean: Array | None = None
        self._activation_count = 0

        # Safety: oscillation detection
        self._recent_entropies: list[float] = []
        self._oscillation_window = 10

    def _infer_model_dims(self) -> tuple[int, int]:
        """Infer number of layers and hidden dimension from model."""
        # Handle wrapped models
        base_model = getattr(self._model, "model", self._model)

        # Get layers
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Model must expose 'layers' attribute")
        n_layers = len(layers)

        # Get hidden dim from config or first layer
        config = getattr(self._model, "config", None)
        if config is not None:
            hidden_dim = getattr(config, "hidden_size", None)
            if hidden_dim is None:
                hidden_dim = getattr(config, "d_model", None)
            if hidden_dim is not None:
                return n_layers, hidden_dim

        # Try to infer from embedding layer (most reliable)
        embed_tokens = getattr(base_model, "embed_tokens", None)
        if embed_tokens is not None:
            weight = getattr(embed_tokens, "weight", None)
            if weight is not None:
                # Embedding shape is (vocab_size, hidden_dim)
                hidden_dim = int(weight.shape[-1])
                return n_layers, hidden_dim

        # Try to infer from first layer's weights
        first_layer = layers[0]
        for attr in ["self_attn.q_proj", "attention.query", "attn.c_attn",
                     "conv.in_proj", "feed_forward.w1"]:
            obj = first_layer
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                weight = getattr(obj, "weight", None)
                if weight is not None:
                    # For projections, input_dim (last) is hidden_dim
                    # For LFM2 conv.in_proj: input_dims=1024 -> hidden_dim
                    hidden_dim = int(weight.shape[-1])
                    return n_layers, hidden_dim

        raise ValueError("Unable to infer hidden_dim from model configuration or layer weights")

    def _derive_max_context(self) -> int:
        config = getattr(self._model, "config", None)
        args = getattr(self._model, "args", None)
        candidates = [
            getattr(config, "max_position_embeddings", None),
            getattr(config, "max_seq_len", None),
            getattr(config, "max_seq_length", None),
            getattr(config, "n_ctx", None),
            getattr(args, "max_position_embeddings", None),
            getattr(args, "max_seq_len", None),
            getattr(self._model, "max_seq_len", None),
            getattr(self._model, "max_seq_length", None),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return 0

    def _derive_max_tokens(self, prompt_length: int) -> int:
        if self._max_context <= 0:
            return 0
        return max(0, self._max_context - prompt_length)

    def _derive_stop_tokens(self) -> set[int]:
        config = getattr(self._model, "config", None)
        eos = getattr(config, "eos_token_id", None) if config is not None else None
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple)):
            return {int(token) for token in eos if token is not None}
        return {int(eos)}

    def generate(
        self,
        input_ids: list[int],
    ) -> Iterator[InferenceState]:
        """Generate tokens with metacognitive control.

        Args:
            input_ids: Initial token IDs (prompt).

        Yields:
            InferenceState for each generation step.
        """
        # Reset components for new generation
        self._reset_generation()

        # Convert input to tensor
        current_ids = list(input_ids)
        stop_tokens = self._derive_stop_tokens()
        max_tokens = self._derive_max_tokens(len(current_ids))

        for _ in range(max_tokens):
            state = self._generate_step(current_ids)
            yield state

            if state.token_id is not None:
                current_ids.append(state.token_id)
                self._tokens_generated += 1

                if state.token_id in stop_tokens:
                    break

    def _compute_refusal_distance(self, hidden_state: Array) -> float | None:
        """Compute distance to refusal boundary in activation space.

        If a refusal direction is set, computes the projection of the
        hidden state onto that direction. Returns normalized distance
        in [0, 1] where 0 = at boundary, 1 = far from boundary.

        If no refusal direction is set, uses deviation from running mean
        as a proxy for anomaly detection.
        """
        b = self._backend

        # Flatten hidden state if needed
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        if self._refusal_direction is not None:
            # Project onto refusal direction
            projection = b.sum(hidden_state * self._refusal_direction)
            norm = b.sqrt(b.sum(self._refusal_direction * self._refusal_direction))
            b.eval(projection, norm)

            # Convert to distance: low projection = safe, high projection = dangerous
            proj_val = float(b.to_scalar(projection))
            norm_val = float(b.to_scalar(norm))

            if norm_val > 0:
                normalized_proj = proj_val / norm_val
                # Sigmoid to map to [0, 1] where 0 = at refusal boundary
                import math
                distance = 1.0 / (1.0 + math.exp(normalized_proj))
                return distance

        # Fallback: use deviation from mean as anomaly proxy
        if self._activation_mean is not None and self._activation_count > 10:
            diff = hidden_state - self._activation_mean
            dist_sq = b.sum(diff * diff)
            mean_sq = b.sum(self._activation_mean * self._activation_mean)
            b.eval(dist_sq, mean_sq)

            dist = float(b.to_scalar(dist_sq)) ** 0.5
            norm = float(b.to_scalar(mean_sq)) ** 0.5

            if norm > 0:
                # Large deviation = potential anomaly = closer to refusal
                relative_deviation = dist / norm
                # Invert: high deviation = low distance to danger
                # Cap at 3 std deviations for normalization
                return max(0.0, 1.0 - min(relative_deviation / 3.0, 1.0))

        return None

    def _detect_oscillation(self, entropy: float) -> tuple[float | None, bool]:
        """Detect oscillation pattern in entropy trajectory.

        Returns (severity, has_oscillation) where severity is in [0, 1].
        Oscillation = alternating high/low entropy indicating instability.
        """
        self._recent_entropies.append(entropy)

        # Keep only recent window
        if len(self._recent_entropies) > self._oscillation_window:
            self._recent_entropies = self._recent_entropies[-self._oscillation_window:]

        if len(self._recent_entropies) < 4:
            return None, False

        # Compute sign changes in derivative
        sign_changes = 0
        for i in range(1, len(self._recent_entropies) - 1):
            prev_diff = self._recent_entropies[i] - self._recent_entropies[i - 1]
            next_diff = self._recent_entropies[i + 1] - self._recent_entropies[i]

            if prev_diff * next_diff < 0:  # Sign change
                sign_changes += 1

        # Normalize: max sign changes = window - 2
        max_changes = len(self._recent_entropies) - 2
        if max_changes <= 0:
            return None, False

        severity = sign_changes / max_changes
        has_oscillation = severity > 0.5  # More than half are sign changes

        return severity, has_oscillation

    def _update_activation_statistics(self, hidden_state: Array) -> None:
        """Update running activation statistics for anomaly detection."""
        b = self._backend

        # Flatten
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        self._activation_count += 1
        if self._activation_mean is None:
            self._activation_mean = hidden_state
        else:
            delta = hidden_state - self._activation_mean
            self._activation_mean = self._activation_mean + delta / float(
                self._activation_count
            )
            b.eval(self._activation_mean)

    def _evaluate_safety(
        self,
        entropy_state: EntropyState,
        hidden_state: Array | None,
    ) -> CircuitBreakerState:
        """Evaluate all safety signals and return circuit breaker state.

        This aggregates multiple signals:
        - Entropy (normalized)
        - Refusal distance
        - Oscillation pattern

        The resulting severity informs the decision gate.
        """
        # Compute refusal distance from hidden state
        refusal_distance: float | None = None
        if hidden_state is not None:
            refusal_distance = self._compute_refusal_distance(hidden_state)

        # Detect oscillation
        oscillation_severity, has_oscillation = self._detect_oscillation(
            entropy_state.entropy
        )

        # Build input signals
        signals = InputSignals(
            entropy_signal=entropy_state.entropy_normalized,
            refusal_distance=refusal_distance,
            is_approaching_refusal=refusal_distance is not None and refusal_distance < 0.3,
            persona_drift_magnitude=None,  # Not tracked in inference loop
            oscillation_severity=oscillation_severity,
            has_oscillation=has_oscillation,
            token_index=self._timestep,
        )

        # Evaluate circuit breaker
        return CircuitBreakerIntegration.evaluate(signals)

    def set_refusal_direction(self, direction: Array) -> None:
        """Set the refusal direction for safety boundary detection.

        The direction should be computed from contrastive learning on
        safe vs unsafe response pairs. Hidden states projected onto this
        direction indicate proximity to harmful outputs.

        Args:
            direction: Unit vector in activation space pointing toward refusal.
        """
        b = self._backend
        # Normalize
        norm = b.sqrt(b.sum(direction * direction))
        b.eval(norm)
        if float(b.to_scalar(norm)) > 0:
            self._refusal_direction = direction / norm
            b.eval(self._refusal_direction)

    def _generate_step(self, current_ids: list[int]) -> InferenceState:
        """Execute one generation step with metacognition and safety evaluation.

        Args:
            current_ids: Current token sequence.

        Returns:
            InferenceState for this step.
        """
        thinking_iterations = 0
        confidence_embedding: Array | None = None
        circuit_breaker_state: CircuitBreakerState | None = None

        while True:
            # Forward pass
            logits, hidden_states = self._forward(
                current_ids, confidence_embedding=confidence_embedding
            )

            # Get last layer hidden state for safety evaluation
            last_layer_id = max(hidden_states.keys()) if hidden_states else -1
            last_hidden = hidden_states.get(last_layer_id)

            # Entropy analysis
            entropy_state = self._entropy_analyzer.analyze(logits)

            # Safety evaluation BEFORE decision gate
            circuit_breaker_state = self._evaluate_safety(entropy_state, last_hidden)

            # Update activation statistics for anomaly detection
            if last_hidden is not None:
                self._update_activation_statistics(last_hidden)

            # Pass refusal distance to decision gate
            refusal_distance = circuit_breaker_state.signal_contributions.refusal
            # Invert: refusal contribution is 1 - distance, so distance = 1 - contribution
            self._decision_gate.set_refusal_distance(1.0 - refusal_distance)

            # Decision gate
            decision = self._decision_gate.decide(entropy_state, last_hidden)

            if decision.action == DecisionAction.EMIT:
                # Emit a token
                token_id = self._sample_token(logits)

                # Track activations
                self._track_activations(hidden_states)

                # Surprise detection and encoding
                surprise_event, encoding_results = self._process_surprise(
                    logits, token_id, hidden_states
                )

                # Get null-space state
                null_state = self._null_space_tracker.get_model_state()

                self._timestep += 1
                self._total_thinking_iterations += thinking_iterations

                return InferenceState(
                    timestep=self._timestep - 1,
                    token_id=token_id,
                    entropy_state=entropy_state,
                    decision=decision,
                    surprise_event=surprise_event,
                    encoding_results=encoding_results,
                    null_space_state=null_state,
                    thinking_iterations=thinking_iterations,
                    circuit_breaker_state=circuit_breaker_state,
                )

            elif decision.action == DecisionAction.THINK_MORE:
                # Generate confidence embedding for next iteration
                # This embedding gets injected at layer 0 by CaptureWrapper
                # in the next _forward call, adding metacognitive signal to
                # the residual stream. The model can then attend to its own
                # uncertainty state.
                confidence_embedding = self._confidence_embedding.encode(entropy_state)
                thinking_iterations += 1
                continue

            else:  # CLARIFY
                # For now, treat clarify as emit with a flag
                # In a full implementation, this would emit clarification tokens
                token_id = self._sample_token(logits)

                self._track_activations(hidden_states)
                surprise_event, encoding_results = self._process_surprise(
                    logits, token_id, hidden_states
                )
                null_state = self._null_space_tracker.get_model_state()

                self._timestep += 1
                self._total_thinking_iterations += thinking_iterations

                return InferenceState(
                    timestep=self._timestep - 1,
                    token_id=token_id,
                    entropy_state=entropy_state,
                    decision=decision,
                    surprise_event=surprise_event,
                    encoding_results=encoding_results,
                    null_space_state=null_state,
                    thinking_iterations=thinking_iterations,
                    circuit_breaker_state=circuit_breaker_state,
                )

    def _forward(
        self,
        token_ids: list[int],
        confidence_embedding: Array | None = None,
    ) -> tuple[Array, dict[int, Array]]:
        """Run forward pass and collect hidden states.

        Args:
            token_ids: Input token IDs.
            confidence_embedding: Optional embedding to inject.

        Returns:
            Tuple of (logits, hidden_states_per_layer).
        """
        b = self._backend

        # Convert to tensor
        input_tensor = b.array([token_ids])  # [1, seq_len]

        # Get the base model
        base_model = getattr(self._model, "model", self._model)

        # Collect hidden states
        hidden_states: dict[int, Array] = {}

        def capture_hook(layer_idx: int, output: Any) -> None:
            """Capture hidden state from layer output."""
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output

            # Take last token
            if hs.ndim == 3:
                hs = hs[0, -1, :]
            elif hs.ndim == 2:
                hs = hs[-1, :]

            hidden_states[layer_idx] = hs

        # Wrap layers for capture
        layers = getattr(base_model, "layers", [])
        original_layers = list(layers)

        class CaptureWrapper:
            def __init__(self, layer, idx, callback, conf_embed, backend):
                self._layer = layer
                self._idx = idx
                self._callback = callback
                self._conf_embed = conf_embed
                self._backend = backend

            def __call__(self, *args, **kwargs):
                output = self._layer(*args, **kwargs)

                # Inject confidence embedding at first layer (layer 0)
                # This adds metacognitive signal to the residual stream
                if self._idx == 0 and self._conf_embed is not None:
                    if isinstance(output, tuple):
                        hs = output[0]
                        # Add confidence embedding to last token position
                        # hs shape: [batch, seq, hidden]
                        if hs.ndim == 3:
                            # Create injection: zeros except last position
                            batch_size = int(hs.shape[0])
                            seq_len = int(hs.shape[1])
                            # Broadcast confidence embedding to last position
                            injection = self._backend.zeros_like(hs)
                            # Add to last position only
                            # injection[:, -1, :] = conf_embed
                            # Since we can't do direct assignment, we use reshape
                            conf_expanded = self._backend.reshape(
                                self._conf_embed, (1, 1, -1)
                            )
                            # Create position mask
                            pos_mask = self._backend.zeros((1, seq_len, 1))
                            mask_data = [[0.0] * (seq_len - 1) + [1.0]]
                            pos_mask = self._backend.array(mask_data)
                            pos_mask = self._backend.reshape(pos_mask, (1, seq_len, 1))
                            injection = conf_expanded * pos_mask
                            hs = hs + injection
                            self._backend.eval(hs)
                        output = (hs,) + output[1:]
                    else:
                        # Direct tensor output
                        if output.ndim == 3:
                            conf_expanded = self._backend.reshape(
                                self._conf_embed, (1, 1, -1)
                            )
                            seq_len = int(output.shape[1])
                            mask_data = [[0.0] * (seq_len - 1) + [1.0]]
                            pos_mask = self._backend.array(mask_data)
                            pos_mask = self._backend.reshape(pos_mask, (1, seq_len, 1))
                            injection = conf_expanded * pos_mask
                            output = output + injection
                            self._backend.eval(output)

                self._callback(self._idx, output)
                return output

            def __getattr__(self, name):
                return getattr(self._layer, name)

        # Wrap layers with confidence embedding injection
        for i, layer in enumerate(original_layers):
            layers[i] = CaptureWrapper(layer, i, capture_hook, confidence_embedding, b)

        try:
            # Forward pass
            # Try different forward signatures
            try:
                output = self._model(input_tensor)
            except TypeError:
                output = self._model.forward(input_tensor)

            # Extract logits
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Flatten to last token
            if logits.ndim == 3:
                logits = logits[0, -1, :]

        finally:
            # Restore original layers
            for i, layer in enumerate(original_layers):
                layers[i] = layer

        return logits, hidden_states

    def _sample_token(self, logits: Array) -> int:
        """Sample a token from logits."""
        b = self._backend

        token_id = b.argmax(logits)
        b.eval(token_id)
        return int(b.to_scalar(token_id))

    def _track_activations(self, hidden_states: dict[int, Array]) -> None:
        """Add hidden states to null-space tracker."""
        self._null_space_tracker.add_all_layers(hidden_states)

        # Update SVD if needed
        if self._null_space_tracker.should_update():
            self._null_space_tracker.update_all_layers()

    def _process_surprise(
        self,
        logits: Array,
        token_id: int,
        hidden_states: dict[int, Array],
    ) -> tuple[SurpriseEvent | None, list[EncodingResult]]:
        """Process surprise and potentially encode knowledge.

        Returns:
            Tuple of (surprise_event, encoding_results).
        """
        # Get last layer hidden state for detection
        last_layer_id = max(hidden_states.keys()) if hidden_states else -1
        last_hidden = hidden_states.get(last_layer_id)

        # Detect surprise
        event = self._surprise_detector.detect(
            logits=logits,
            actual_token_id=token_id,
            hidden_state=last_hidden,
        )

        # Encode deterministically when a hidden state is available.
        encoding_results = []
        if last_hidden is not None:
            encoding_results = self._knowledge_encoder.encode(
                event=event,
                hidden_state=last_hidden,
            )

        return event, encoding_results

    def _reset_generation(self) -> None:
        """Reset state for new generation."""
        self._entropy_analyzer.reset()
        self._decision_gate.reset()
        self._surprise_detector.reset()
        self._timestep = 0
        # Reset oscillation detection but preserve activation statistics
        # (activation mean helps detect anomalies across generations)
        self._recent_entropies = []

    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        return {
            "tokens_generated": self._tokens_generated,
            "total_thinking_iterations": self._total_thinking_iterations,
            "avg_thinking_per_token": (
                self._total_thinking_iterations / self._tokens_generated
                if self._tokens_generated > 0
                else 0
            ),
            "encoding_stats": self._knowledge_encoder.get_stats(),
            "null_space_state": self._null_space_tracker.get_model_state().as_dict(),
            "baseline_surprise": self._surprise_detector.get_baseline_surprise(),
            "safety": {
                "refusal_direction_set": self._refusal_direction is not None,
                "activation_samples": self._activation_count,
                "entropy_baseline": self._decision_gate.entropy_baseline,
                "entropy_std": self._decision_gate.entropy_std,
            },
        }

    def reset(self) -> None:
        """Reset all state."""
        self._reset_generation()
        self._null_space_tracker.reset()
        self._knowledge_encoder.reset_stats()
        self._tokens_generated = 0
        self._total_thinking_iterations = 0
        # Reset safety state
        self._activation_mean = None
        self._activation_count = 0
        self._recent_entropies = []
        # Note: We preserve refusal_direction across resets as it's
        # computed externally and represents model-specific safety geometry

    @property
    def n_layers(self) -> int:
        """Number of model layers."""
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        """Model hidden dimension."""
        return self._hidden_dim
