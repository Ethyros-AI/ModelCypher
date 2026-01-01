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

"""Tests for activation_stream.py.

Tests the streaming activation capture using callback-based layer wrapping.

Note: Full integration tests require a loaded model, so these tests focus on
the data structures, helper methods, and callback patterns.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.inference.activation_stream import (
    ActivationFrame,
    ActivationStream,
    _LayerCaptureContext,
    _StreamingLayerWrapper,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Provide backend for tests."""
    return get_default_backend()


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model with layers."""
    model = MagicMock()
    model.model = MagicMock()
    model.model.layers = [MagicMock() for _ in range(3)]
    return model


# =============================================================================
# ActivationFrame Tests
# =============================================================================


class TestActivationFrame:
    """Tests for ActivationFrame dataclass."""

    def test_creation(self, backend: "Backend") -> None:
        """Test basic frame creation."""
        hidden = backend.random_normal((64,))
        frame = ActivationFrame(
            layer_id=5,
            token_idx=10,
            hidden_state=hidden,
            entropy=2.5,
        )

        assert frame.layer_id == 5
        assert frame.token_idx == 10
        assert frame.hidden_state.shape == (64,)
        assert frame.entropy == 2.5
        assert frame.projected_3d is None

    def test_with_projection(self, backend: "Backend") -> None:
        """Test frame with 3D projection."""
        hidden = backend.random_normal((64,))
        projected = backend.random_normal((3,))
        frame = ActivationFrame(
            layer_id=5,
            token_idx=10,
            hidden_state=hidden,
            projected_3d=projected,
            entropy=2.5,
        )

        assert frame.projected_3d is not None
        assert frame.projected_3d.shape == (3,)

    def test_timestamp_auto_generated(self, backend: "Backend") -> None:
        """Test that timestamp is auto-generated."""
        hidden = backend.random_normal((64,))
        before = time.time()
        frame = ActivationFrame(
            layer_id=0,
            token_idx=0,
            hidden_state=hidden,
        )
        after = time.time()

        assert before <= frame.timestamp <= after


# =============================================================================
# ActivationStream Tests
# =============================================================================


class TestActivationStream:
    """Tests for ActivationStream class."""

    def test_initialization(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test stream initialization."""
        stream = ActivationStream(mock_model, backend)
        assert stream.model is mock_model
        assert stream.backend is backend
        assert len(stream.frames) == 0

    def test_set_projection_coupling(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test setting projection coupling."""
        stream = ActivationStream(mock_model, backend)
        coupling = backend.random_normal((64, 3))
        stream.set_projection_coupling(coupling)

        assert stream._projection_coupling is not None
        assert stream._projection_coupling.shape == (64, 3)

    def test_add_subscriber(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test adding subscribers."""
        stream = ActivationStream(mock_model, backend)

        callback1 = MagicMock()
        callback2 = MagicMock()

        stream.add_subscriber(callback1)
        stream.add_subscriber(callback2)

        assert callback1 in stream._subscribers
        assert callback2 in stream._subscribers

    def test_remove_subscriber(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test removing subscribers."""
        stream = ActivationStream(mock_model, backend)
        callback = MagicMock()

        stream.add_subscriber(callback)
        stream.remove_subscriber(callback)

        assert callback not in stream._subscribers

    def test_advance_token(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test token advancement."""
        stream = ActivationStream(mock_model, backend)

        assert stream._current_token == 0
        stream.advance_token()
        assert stream._current_token == 1
        stream.advance_token()
        assert stream._current_token == 2

    def test_reset(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test stream reset."""
        stream = ActivationStream(mock_model, backend)

        stream.advance_token()
        stream.advance_token()
        # Manually add a frame for testing
        stream._buffer.append(
            ActivationFrame(
                layer_id=0,
                token_idx=0,
                hidden_state=backend.random_normal((64,)),
            )
        )

        stream.reset()

        assert stream._current_token == 0
        assert len(stream.frames) == 0

    def test_compute_entropy(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test entropy computation."""
        stream = ActivationStream(mock_model, backend)

        # Create a hidden state
        hidden = backend.random_normal((64,))
        entropy = stream._compute_entropy(hidden)

        # Entropy should be positive for non-zero hidden states
        assert entropy > 0

    def test_compute_entropy_zero(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test entropy computation for zero vector."""
        stream = ActivationStream(mock_model, backend)

        hidden = backend.zeros((64,))
        entropy = stream._compute_entropy(hidden)

        # Should handle zero vector gracefully
        assert entropy == 0.0

    def test_get_trajectory(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test trajectory retrieval."""
        stream = ActivationStream(mock_model, backend)

        # Manually add frames for testing
        for i in range(5):
            stream._buffer.append(
                ActivationFrame(
                    layer_id=0,
                    token_idx=i,
                    hidden_state=backend.random_normal((64,)),
                )
            )
            stream._buffer.append(
                ActivationFrame(
                    layer_id=1,
                    token_idx=i,
                    hidden_state=backend.random_normal((64,)),
                )
            )

        trajectory = stream.get_trajectory(layer_id=0)
        assert len(trajectory) == 5
        assert all(f.layer_id == 0 for f in trajectory)
        # Should be sorted by token_idx
        for i in range(len(trajectory) - 1):
            assert trajectory[i].token_idx <= trajectory[i + 1].token_idx

    def test_get_layer_snapshot(
        self, backend: "Backend", mock_model: MagicMock
    ) -> None:
        """Test layer snapshot retrieval."""
        stream = ActivationStream(mock_model, backend)

        # Add frames for token 2 across layers
        for layer in range(3):
            stream._buffer.append(
                ActivationFrame(
                    layer_id=layer,
                    token_idx=2,
                    hidden_state=backend.random_normal((64,)),
                )
            )

        snapshot = stream.get_layer_snapshot(token_idx=2)
        assert len(snapshot) == 3
        assert all(f.token_idx == 2 for f in snapshot)
        # Should be sorted by layer_id
        for i in range(len(snapshot) - 1):
            assert snapshot[i].layer_id <= snapshot[i + 1].layer_id


# =============================================================================
# _StreamingLayerWrapper Tests
# =============================================================================


class TestStreamingLayerWrapper:
    """Tests for _StreamingLayerWrapper class."""

    def test_call_delegates_to_layer(self) -> None:
        """Test that __call__ delegates to wrapped layer."""
        mock_layer = MagicMock()
        mock_layer.return_value = "output"
        capture = MagicMock()

        wrapper = _StreamingLayerWrapper(mock_layer, layer_index=0, capture=capture)
        result = wrapper("input")

        mock_layer.assert_called_once_with("input")
        assert result == "output"

    def test_call_invokes_capture(self) -> None:
        """Test that __call__ invokes capture callback."""
        mock_layer = MagicMock()
        mock_layer.return_value = "output"
        capture = MagicMock()

        wrapper = _StreamingLayerWrapper(mock_layer, layer_index=5, capture=capture)
        wrapper("input")

        capture.assert_called_once_with(5, "output")

    def test_getattr_delegates(self) -> None:
        """Test that attribute access delegates to wrapped layer."""
        mock_layer = MagicMock()
        mock_layer.some_attr = "value"
        capture = MagicMock()

        wrapper = _StreamingLayerWrapper(mock_layer, layer_index=0, capture=capture)

        assert wrapper.some_attr == "value"

    def test_setattr_delegates(self) -> None:
        """Test that attribute setting delegates to wrapped layer."""
        mock_layer = MagicMock()
        capture = MagicMock()

        wrapper = _StreamingLayerWrapper(mock_layer, layer_index=0, capture=capture)
        wrapper.some_attr = "new_value"

        assert mock_layer.some_attr == "new_value"


# =============================================================================
# _LayerCaptureContext Tests
# =============================================================================


class TestLayerCaptureContext:
    """Tests for _LayerCaptureContext class."""

    def test_enter_wraps_layers(self) -> None:
        """Test that entering context wraps layers."""
        layers = [MagicMock() for _ in range(3)]
        original_layers = list(layers)
        capture = MagicMock()

        ctx = _LayerCaptureContext(layers, capture, target_layers=None)
        ctx.__enter__()

        # All layers should be wrapped
        for i, layer in enumerate(layers):
            assert isinstance(layer, _StreamingLayerWrapper)

    def test_exit_restores_layers(self) -> None:
        """Test that exiting context restores original layers."""
        original = [MagicMock() for _ in range(3)]
        layers = list(original)
        capture = MagicMock()

        ctx = _LayerCaptureContext(layers, capture, target_layers=None)
        ctx.__enter__()
        ctx.__exit__(None, None, None)

        # Original layers should be restored
        for i, layer in enumerate(layers):
            assert layer is original[i]

    def test_target_layers_filter(self) -> None:
        """Test that only target layers are wrapped."""
        layers = [MagicMock() for _ in range(5)]
        original = list(layers)
        capture = MagicMock()

        # Only wrap layers 1 and 3
        ctx = _LayerCaptureContext(layers, capture, target_layers={1, 3})
        ctx.__enter__()

        # Layer 0, 2, 4 should not be wrapped
        assert layers[0] is original[0]
        assert layers[2] is original[2]
        assert layers[4] is original[4]

        # Layer 1 and 3 should be wrapped
        assert isinstance(layers[1], _StreamingLayerWrapper)
        assert isinstance(layers[3], _StreamingLayerWrapper)
