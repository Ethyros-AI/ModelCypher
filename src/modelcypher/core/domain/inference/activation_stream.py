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
Streaming activation extraction from model forward pass.

MLX does NOT support PyTorch-style hooks (register_forward_hook).
We use the callback-based wrapper pattern from local_inference.py
which temporarily wraps layers with callbacks during inference.

The key insight for real-time visualization:
1. Coupling matrices (π_col) computed via GRAM_TRANSPORT are REUSABLE
2. For streaming: token @ π_composite = 3D point (single matmul!)
3. No recomputation of GW needed - geometry is FIXED after calibration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ActivationFrame:
    """Single frame of activation data during inference.

    Captures hidden state at a specific layer and token position,
    optionally with real-time projection to low-dimensional space.

    Attributes:
        layer_id: Index of the transformer layer (0-based)
        token_idx: Index of the token being processed
        hidden_state: The hidden state tensor [hidden_dim]
        projected_3d: Optional 3D projection [3] if coupling is set
        entropy: Local entropy of the hidden state distribution
        timestamp: Time of capture for animation sequencing
    """

    layer_id: int
    token_idx: int
    hidden_state: "Array"  # [hidden_dim] - last token's activation
    projected_3d: "Array | None" = None  # [3] - if cascade is active
    entropy: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ActivationStream:
    """
    Stream activations from model forward pass using callback-based capture.

    MLX does NOT support PyTorch-style hooks. We use the _LayerCapture pattern
    from local_inference.py which temporarily wraps layers with callbacks.

    The streaming pipeline:
    1. Calibration phase: Capture initial tokens, compute coupling matrices
    2. Inject composite coupling: π_composite = [hidden_dim, 3]
    3. Streaming phase: Each layer activation is projected via single matmul

    Usage:
        stream = ActivationStream(model, backend)
        stream.add_subscriber(on_frame)  # Called for each layer activation

        # After calibration, inject the composite coupling
        stream.set_projection_coupling(cascade.get_composite_coupling(target_dim=3))

        with stream.capture(target_layers={0, 16, 32}):
            for token in generate(prompt):
                stream.advance_token()  # Increment token index
    """

    def __init__(
        self,
        model: Any,
        backend: "Backend | None" = None,
        projection_coupling: "Array | None" = None,
    ) -> None:
        """
        Initialize the activation stream.

        Args:
            model: The MLX model to capture activations from
            backend: Backend for tensor operations (defaults to MLX)
            projection_coupling: Optional precomputed coupling matrix [d, 3]
                for real-time projection during capture
        """
        self.model = model
        self.backend = backend or get_default_backend()
        self._projection_coupling = projection_coupling

        self._buffer: list[ActivationFrame] = []
        self._subscribers: list[Callable[[ActivationFrame], None]] = []
        self._current_token: int = 0

    def set_projection_coupling(self, coupling: "Array") -> None:
        """
        Set the coupling matrix for streaming projection.

        Once set, all captured activations will be projected to target dimension
        with a single matmul: hidden @ coupling

        The coupling matrix should be computed via DimensionCascade.calibrate()
        and retrieved via DimensionCascade.get_composite_coupling().

        Args:
            coupling: [d_source, d_target] coupling from GRAM_TRANSPORT
        """
        self._projection_coupling = coupling
        logger.debug(
            "Set projection coupling: [%d, %d]",
            coupling.shape[0],
            coupling.shape[1],
        )

    def add_subscriber(self, callback: Callable[[ActivationFrame], None]) -> None:
        """
        Add callback to be invoked for each captured activation frame.

        Subscribers receive frames in real-time as layers are processed.
        Use this for live visualization updates.

        Args:
            callback: Function taking an ActivationFrame
        """
        self._subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[ActivationFrame], None]) -> None:
        """Remove a previously added subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def advance_token(self) -> None:
        """
        Increment token index. Call BEFORE each token generation.

        This tracks which token the activations correspond to,
        enabling proper trajectory visualization.
        """
        self._current_token += 1

    def reset(self) -> None:
        """Reset buffer and token index for new session."""
        self._buffer.clear()
        self._current_token = 0
        logger.debug("Activation stream reset")

    def _capture_callback(self, layer_index: int, output: Any) -> None:
        """
        Callback invoked by _LayerWrapper for each layer forward pass.

        Extracts last token's hidden state and optionally projects to 3D.
        This is the hot path during streaming - must be efficient.

        Args:
            layer_index: Index of the layer that produced this output
            output: Layer output (tuple or tensor depending on model)
        """
        b = self.backend

        # Extract hidden state for last token
        # MLX output format varies by model architecture:
        # - Llama/Qwen: (hidden_states, ...) where hidden_states is [batch, seq, hidden]
        # - Some models return just the tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Handle different tensor shapes
        if len(hidden_states.shape) == 3:
            # [batch, seq, hidden] -> [hidden]
            hidden = hidden_states[0, -1, :]
        elif len(hidden_states.shape) == 2:
            # [seq, hidden] -> [hidden]
            hidden = hidden_states[-1, :]
        else:
            # Already 1D [hidden]
            hidden = hidden_states

        b.eval(hidden)

        # Project to 3D if coupling matrix is set
        # This is the STREAMING path - single matmul O(d×3)
        projected_3d = None
        if self._projection_coupling is not None:
            # Ensure proper shape for matmul
            if len(hidden.shape) == 1:
                projected_3d = b.matmul(hidden[None, :], self._projection_coupling)[0]
            else:
                projected_3d = b.matmul(hidden, self._projection_coupling)
            b.eval(projected_3d)

        # Compute local entropy for MAP compatibility
        entropy = self._compute_entropy(hidden)

        frame = ActivationFrame(
            layer_id=layer_index,
            token_idx=self._current_token,
            hidden_state=hidden,
            projected_3d=projected_3d,
            entropy=entropy,
            timestamp=time.time(),
        )

        self._buffer.append(frame)

        # Emit to subscribers for real-time updates
        for callback in self._subscribers:
            try:
                callback(frame)
            except Exception as exc:
                logger.warning("Subscriber callback failed: %s", exc)

    def _compute_entropy(self, hidden: "Array") -> float:
        """
        Compute local entropy from hidden state magnitude distribution.

        Uses Shannon entropy on the normalized absolute values of the
        hidden state. Higher entropy = more uniform distribution = less
        confident/specialized representation.

        Args:
            hidden: Hidden state tensor [hidden_dim]

        Returns:
            Shannon entropy of the magnitude distribution
        """
        b = self.backend

        # Normalize absolute values to probability-like distribution
        abs_hidden = b.abs(hidden)
        total = b.sum(abs_hidden)
        b.eval(total)

        # Avoid division by zero
        if float(b.to_numpy(total)) < 1e-10:
            return 0.0

        probs = abs_hidden / total

        # Shannon entropy: -Σ p log p
        # Add small epsilon to avoid log(0)
        log_probs = b.log(probs + 1e-10)
        entropy_tensor = -b.sum(probs * log_probs)
        b.eval(entropy_tensor)

        return float(b.to_numpy(entropy_tensor))

    def capture(self, target_layers: set[int] | None = None) -> "_LayerCaptureContext":
        """
        Context manager for capturing activations during inference.

        Temporarily wraps model layers with callbacks, restoring them on exit.
        Use with 'with' statement to ensure proper cleanup.

        Args:
            target_layers: Set of layer indices to capture (None = all layers).
                For efficiency, specify only the layers you need.

        Returns:
            Context manager that wraps layers and restores them on exit

        Raises:
            RuntimeError: If model doesn't expose transformer layers

        Example:
            with stream.capture(target_layers={0, 16, 31}):
                output = model(input_ids)
        """
        # Get model's layers list - handle wrapped models
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)

        if layers is None:
            raise RuntimeError(
                "Model does not expose transformer layers for capture. "
                "Expected model.layers or model.model.layers attribute."
            )

        logger.debug(
            "Creating capture context for %d layers (targeting %s)",
            len(layers),
            target_layers if target_layers else "all",
        )

        return _LayerCaptureContext(layers, self._capture_callback, target_layers)

    @property
    def frames(self) -> list[ActivationFrame]:
        """Return all captured frames in order."""
        return self._buffer

    def get_trajectory(self, layer_id: int) -> list[ActivationFrame]:
        """
        Get activation trajectory for a specific layer.

        Returns frames for the specified layer across all captured tokens,
        useful for visualizing how representations evolve during generation.

        Args:
            layer_id: Index of the layer to get trajectory for

        Returns:
            List of ActivationFrames for that layer, ordered by token_idx
        """
        return sorted(
            [f for f in self._buffer if f.layer_id == layer_id],
            key=lambda f: f.token_idx,
        )

    def get_layer_snapshot(self, token_idx: int) -> list[ActivationFrame]:
        """
        Get activations across all layers for a specific token.

        Returns frames for all captured layers at the specified token position,
        useful for visualizing the full layer trajectory.

        Args:
            token_idx: Index of the token to get snapshot for

        Returns:
            List of ActivationFrames for that token, ordered by layer_id
        """
        return sorted(
            [f for f in self._buffer if f.token_idx == token_idx],
            key=lambda f: f.layer_id,
        )


class _LayerCaptureContext:
    """
    Context manager for temporary layer wrapping.

    Implements the callback-based capture pattern from local_inference.py
    but optimized for streaming visualization.

    On enter: Wraps target layers with _StreamingLayerWrapper
    On exit: Restores original layer references
    """

    def __init__(
        self,
        layers: list[Any],
        capture: Callable[[int, Any], None],
        target_layers: set[int] | None = None,
    ) -> None:
        self._layers = layers
        self._capture = capture
        self._target_layers = target_layers
        self._original: list[Any] | None = None

    def __enter__(self) -> "_LayerCaptureContext":
        # Save original layer references
        self._original = list(self._layers)

        # Wrap target layers with capture callbacks
        wrapped: list[Any] = []
        for idx, layer in enumerate(self._layers):
            if self._target_layers is not None and idx not in self._target_layers:
                wrapped.append(layer)  # Skip non-target layers
            else:
                wrapped.append(_StreamingLayerWrapper(layer, idx, self._capture))

        # In-place replacement of layers list
        self._layers[:] = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Restore original layers
        if self._original is not None:
            self._layers[:] = self._original


class _StreamingLayerWrapper:
    """
    Wraps a layer to intercept output and invoke callback.

    Delegates all attributes to the wrapped layer, only intercepting
    the __call__ method to capture activations.

    This is the hot path during inference - implementation is minimal
    to avoid overhead.
    """

    __slots__ = ("_layer", "_layer_index", "_capture")

    def __init__(
        self,
        layer: Any,
        layer_index: int,
        capture: Callable[[int, Any], None],
    ) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_layer_index", layer_index)
        object.__setattr__(self, "_capture", capture)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._layer(*args, **kwargs)
        self._capture(self._layer_index, output)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_layer", "_layer_index", "_capture"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._layer, name, value)
