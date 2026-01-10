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

"""Tests for Visual Concept Injection pipeline.

All geometric parameters are AUTO-DERIVED from the data.
No user-configurable knobs.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.multimodal.visual_injection import (
    VisualConceptInjector,
    VisualMemoryToken,
    InjectionResult,
)


class TestVisualConceptInjector:
    """Tests for VisualConceptInjector class."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def injector(self, backend):
        """Create injector instance."""
        return VisualConceptInjector(backend, architecture="LFM2")

    @pytest.fixture
    def mock_weights(self, backend):
        """Create mock bridge weights."""
        # Identity-ish transformation for testing
        W = backend.eye(1024, dtype="float32")
        b = backend.zeros((1024,), dtype="float32")
        return W, b

    @pytest.fixture
    def mock_vocab(self, backend):
        """Create mock vocabulary embeddings."""
        # 100 random vocabulary embeddings
        vocab = backend.random_normal((100, 1024))
        # Normalize
        norms = backend.sqrt(backend.sum(vocab * vocab, axis=1, keepdims=True))
        vocab = vocab / (norms + 1e-8)
        backend.eval(vocab)
        return vocab

    def test_init_with_known_architecture(self, backend) -> None:
        """Should recognize LFM2 architecture."""
        injector = VisualConceptInjector(backend, architecture="LFM2")
        assert injector._layer_config is not None
        assert injector._layer_config.n_layers == 16

    def test_init_with_unknown_architecture(self, backend) -> None:
        """Should handle unknown architecture gracefully."""
        injector = VisualConceptInjector(backend, architecture="Unknown")
        assert injector._layer_config is None

    def test_is_ready_false_initially(self, injector) -> None:
        """Should not be ready without weights and vocab."""
        assert not injector.is_ready

    def test_create_visual_memory_requires_setup(self, backend, injector) -> None:
        """Should raise error if not set up."""
        embed = backend.random_normal((1, 1024))

        with pytest.raises(RuntimeError, match="load_bridge_weights"):
            injector.create_visual_memory(embed)

    def test_set_vocabulary(self, backend, injector, mock_vocab) -> None:
        """Should set vocabulary correctly."""
        injector.set_vocabulary(mock_vocab)
        assert injector._vocabulary_set

    def test_compute_null_basis_auto_derives_rank(self, backend, injector) -> None:
        """Should auto-derive null rank from SVD variance analysis."""
        activations = backend.random_normal((50, 1024))
        # No null_rank parameter - auto-derived
        injector.compute_null_basis_from_activations(activations)
        assert injector._null_basis is not None
        # Rank should be auto-determined, not fixed

    def test_get_optimal_injection_layers(self, injector) -> None:
        """Should return attention layers in semantic highway."""
        layers = injector.get_optimal_injection_layers()
        # LFM2 has attention at 8 in semantic highway (7-9)
        assert 8 in layers


class TestVisualConceptInjectorIntegration:
    """Integration tests with mock weights."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def ready_injector(self, backend):
        """Create fully configured injector."""
        injector = VisualConceptInjector(backend, architecture="LFM2")

        # Set up mock bridge (identity transformation)
        W = backend.eye(1024, dtype="float32")
        b = backend.zeros((1024,), dtype="float32")
        injector._bridge.load_affine_weights(W, b)
        injector._bridge_loaded = True

        # Set up mock vocabulary
        vocab = backend.random_normal((100, 1024))
        norms = backend.sqrt(backend.sum(vocab * vocab, axis=1, keepdims=True))
        vocab = vocab / (norms + 1e-8)
        backend.eval(vocab)
        injector.set_vocabulary(vocab)

        # Compute null basis with calibration activations (for scale derivation)
        activations = backend.random_normal((50, 1024))
        injector.compute_null_basis_from_activations(activations)

        return injector

    def test_create_visual_memory_returns_token(self, backend, ready_injector) -> None:
        """Should create visual memory token with auto-derived parameters."""
        embed = backend.random_normal((1, 1024))
        backend.eval(embed)

        # No scale/temperature parameters - all auto-derived
        memory = ready_injector.create_visual_memory(embed)

        assert isinstance(memory, VisualMemoryToken)
        # Scale and temperature should be auto-derived (non-default values)
        assert memory.scale > 0  # Auto-derived from activation norms
        assert memory.temperature > 0  # Auto-derived from similarity std
        assert len(memory.nearest_token_ids) > 0

    def test_create_visual_memory_handles_1d_input(self, backend, ready_injector) -> None:
        """Should handle 1D embedding input."""
        embed = backend.random_normal((1024,))
        backend.eval(embed)

        memory = ready_injector.create_visual_memory(embed)
        assert memory is not None

    def test_inject_memory_returns_result(self, backend, ready_injector) -> None:
        """Should inject memory into hidden states."""
        # Create memory (no parameters - all auto-derived)
        embed = backend.random_normal((1, 1024))
        memory = ready_injector.create_visual_memory(embed)

        # Create hidden states
        hidden = backend.random_normal((1, 10, 1024))
        backend.eval(hidden)

        # Inject (no layer_idx parameter - auto-determined)
        result = ready_injector.inject_memory(hidden, memory)

        assert isinstance(result, InjectionResult)
        assert result.hidden_states is not None
        # Layer should be auto-determined from architecture
        assert 8 in result.injection_layers

    def test_null_space_projection_always_applied(self, backend, ready_injector) -> None:
        """Should always project into null-space when basis is available."""
        # ready_injector already has null basis computed
        embed = backend.random_normal((1, 1024))
        memory = ready_injector.create_visual_memory(embed)

        # No use_null_space parameter - always applied if basis exists
        assert memory.null_space_projected


class TestVisualMemoryToken:
    """Tests for VisualMemoryToken dataclass."""

    def test_fields_accessible(self) -> None:
        """Should have all required fields."""
        token = VisualMemoryToken(
            embedding=[1.0, 2.0],
            nearest_token_ids=[0, 1, 2],
            attention_weights=[0.5, 0.3, 0.2],
            scale=10.0,  # Auto-derived, exposed for diagnostics
            temperature=0.087,  # Auto-derived, exposed for diagnostics
            null_space_projected=True,
            source_type="clip_image",
        )

        assert token.scale == 10.0
        assert token.temperature == 0.087
        assert token.source_type == "clip_image"
        assert len(token.nearest_token_ids) == 3


class TestInjectionResult:
    """Tests for InjectionResult dataclass."""

    def test_fields_accessible(self) -> None:
        """Should have all required fields."""
        result = InjectionResult(
            hidden_states=[1.0, 2.0],
            injection_layers=[8],  # Auto-determined
            is_safe=True,
            safety_message="Safe",
        )

        assert result.is_safe
        assert 8 in result.injection_layers
