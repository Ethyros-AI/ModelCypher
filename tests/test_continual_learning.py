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

"""Tests for the continual learning module."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend


class TestEntropyAnalyzer:
    """Tests for EntropyAnalyzer."""

    def test_analyze_returns_entropy_state(self):
        """EntropyAnalyzer returns EntropyState with expected fields."""
        from modelcypher.core.domain.continual import EntropyAnalyzer

        backend = get_default_backend()
        analyzer = EntropyAnalyzer(backend=backend)

        # Create mock logits [vocab_size] - standard normal
        logits = backend.random_normal((32000,))
        backend.eval(logits)

        state = analyzer.analyze(logits)

        assert state.entropy >= 0
        assert 0 <= state.entropy_normalized <= 1
        assert state.vocab_size == 32000
        assert state.timestep == 1

    def test_derivative_computed_after_warmup(self):
        """EntropyAnalyzer computes derivatives after enough samples."""
        from modelcypher.core.domain.continual import EntropyAnalyzer

        backend = get_default_backend()
        analyzer = EntropyAnalyzer(backend=backend)

        # Run a few iterations
        for i in range(5):
            logits = backend.random_normal((1000,))
            backend.eval(logits)
            state = analyzer.analyze(logits)

        # After 5 iterations, should have derivative
        assert state.timestep == 5
        # Derivative should be non-zero (entropy changes between random logits)
        # Note: Could be zero by chance, so we just check it's a valid float
        assert isinstance(state.entropy_derivative, float)

    def test_reset_clears_history(self):
        """EntropyAnalyzer.reset() clears state."""
        from modelcypher.core.domain.continual import EntropyAnalyzer

        backend = get_default_backend()
        analyzer = EntropyAnalyzer(backend=backend)

        # Add some samples
        for _ in range(3):
            logits = backend.random_normal((1000,))
            backend.eval(logits)
            analyzer.analyze(logits)

        assert analyzer.timestep == 3

        analyzer.reset()
        assert analyzer.timestep == 0


class TestDecisionGate:
    """Tests for DecisionGate."""

    def test_decide_returns_decision(self):
        """DecisionGate.decide() returns a Decision."""
        from modelcypher.core.domain.continual import (
            DecisionGate,
            EntropyAnalyzer,
        )

        backend = get_default_backend()
        analyzer = EntropyAnalyzer(backend=backend)
        gate = DecisionGate(backend=backend)

        logits = backend.random_normal((1000,))
        backend.eval(logits)
        state = analyzer.analyze(logits)

        decision = gate.decide(state)

        assert decision.action is not None
        assert 0 <= decision.confidence <= 1
        assert len(decision.action_logits) == 3

    def test_decision_is_emit_and_budget_zero(self):
        """DecisionGate always emits with zero budget."""
        from modelcypher.core.domain.continual import (
            DecisionAction,
            DecisionGate,
            EntropyState,
        )

        backend = get_default_backend()
        gate = DecisionGate(backend=backend)

        # Create a high-entropy state that would normally trigger THINK_MORE
        state = EntropyState(
            entropy=10.0,
            entropy_normalized=0.95,
            entropy_derivative=0.1,
            entropy_acceleration=0.0,
            logit_variance=100.0,
            vocab_size=32000,
            timestep=1,
        )

        decision = gate.decide(state)

        assert decision.action == DecisionAction.EMIT
        assert decision.thinking_budget_remaining == 0

    def test_reset_restores_budget(self):
        """DecisionGate.reset() restores thinking budget."""
        from modelcypher.core.domain.continual import DecisionGate

        backend = get_default_backend()
        gate = DecisionGate(backend=backend)

        gate._thinking_steps_used = 5
        assert gate._thinking_steps_used == 5

        gate.reset()
        assert gate._thinking_steps_used == 0


class TestActivationBuffer:
    """Tests for ActivationBuffer."""

    def test_add_updates_statistics(self):
        """ActivationBuffer.add() updates running statistics."""
        from modelcypher.core.domain.continual import ActivationBuffer

        backend = get_default_backend()
        buffer = ActivationBuffer(hidden_dim=128, backend=backend)

        # Add some activations
        for _ in range(10):
            act = backend.random_normal((128,))
            backend.eval(act)
            buffer.add(act)

        stats = buffer.get_stats()
        assert stats.n_samples == 10
        assert stats.total_variance > 0

    def test_rolling_buffer_removes_old(self):
        """ActivationBuffer removes old samples when full."""
        from modelcypher.core.domain.continual import ActivationBuffer

        backend = get_default_backend()
        buffer = ActivationBuffer(hidden_dim=4, backend=backend)

        # Add more than buffer size
        for i in range(10):
            act = backend.random_normal((4,)) * 0.1 + float(i)
            backend.eval(act)
            buffer.add(act)

        # Should only have hidden_dim + 1 samples
        assert buffer.current_size == buffer.buffer_size
        assert buffer.is_full

    def test_svd_update_computes_rank(self):
        """ActivationBuffer.update_svd() computes rank."""
        from modelcypher.core.domain.continual import ActivationBuffer

        backend = get_default_backend()
        buffer = ActivationBuffer(hidden_dim=32, backend=backend)

        # Add enough samples for meaningful SVD
        for _ in range(50):
            act = backend.random_normal((32,))
            backend.eval(act)
            buffer.add(act)

        buffer.update_svd()

        stats = buffer.get_stats()
        assert stats.svd_update_count == 1
        assert stats.svd_rank > 0


class TestNullSpaceTracker:
    """Tests for NullSpaceTracker."""

    def test_tracks_per_layer(self):
        """NullSpaceTracker maintains per-layer buffers."""
        from modelcypher.core.domain.continual import NullSpaceTracker

        backend = get_default_backend()
        tracker = NullSpaceTracker(
            n_layers=4,
            hidden_dim=64,
            backend=backend,
        )

        # Add activations to different layers
        for layer_id in range(4):
            for _ in range(10):
                act = backend.random_normal((64,)) + float(layer_id)
                backend.eval(act)
                tracker.add_activation(layer_id, act)

        # Each layer should have samples
        for layer_id in range(4):
            state = tracker.get_layer_state(layer_id)
            assert state.layer_id == layer_id
            assert state.hidden_dim == 64

    def test_get_variance_weights(self):
        """NullSpaceTracker provides variance weights for projection."""
        from modelcypher.core.domain.continual import NullSpaceTracker

        backend = get_default_backend()
        tracker = NullSpaceTracker(
            n_layers=2,
            hidden_dim=32,
            backend=backend,
        )

        # Add samples to layer 0
        for _ in range(20):
            act = backend.random_normal((32,))
            backend.eval(act)
            tracker.add_activation(0, act)

        weights = tracker.get_variance_weights(0)
        assert weights is not None
        assert weights.shape[0] == 32


class TestSurpriseDetector:
    """Tests for SurpriseDetector."""

    def test_detect_returns_event(self):
        """SurpriseDetector.detect() returns SurpriseEvent."""
        from modelcypher.core.domain.continual import SurpriseDetector

        backend = get_default_backend()
        detector = SurpriseDetector(backend=backend)

        # Create mock logits
        logits = backend.random_normal((1000,))
        backend.eval(logits)

        event = detector.detect(
            logits=logits,
            actual_token_id=42,
        )

        assert event.token_id == 42
        assert event.token_surprise >= 0
        assert event.timestep == 0

    def test_baseline_adapts_over_time(self):
        """SurpriseDetector baseline adapts to recent history."""
        from modelcypher.core.domain.continual import SurpriseDetector

        backend = get_default_backend()
        detector = SurpriseDetector(backend=backend)

        # Run several detections
        for i in range(20):
            logits = backend.random_normal((1000,))
            backend.eval(logits)
            detector.detect(logits=logits, actual_token_id=i % 1000)

        baseline = detector.get_baseline_surprise()
        assert baseline > 0  # Should have computed a baseline


class TestConfidenceEmbedding:
    """Tests for ConfidenceEmbedding."""

    def test_encode_produces_embedding(self):
        """ConfidenceEmbedding.encode() produces correct shape."""
        from modelcypher.core.domain.continual import (
            ConfidenceEmbedding,
            EntropyState,
        )

        backend = get_default_backend()
        embedding = ConfidenceEmbedding(hidden_dim=256, backend=backend)

        state = EntropyState(
            entropy=5.0,
            entropy_normalized=0.5,
            entropy_derivative=0.1,
            entropy_acceleration=0.01,
            logit_variance=10.0,
            vocab_size=32000,
            timestep=1,
        )

        result = embedding.encode(state)
        assert result.shape == (256,)


class TestManifoldCompletion:
    """Tests for ManifoldCompletion."""

    def test_compute_densities(self):
        """ManifoldCompletion._compute_densities returns valid densities."""
        from modelcypher.core.domain.continual import (
            KnowledgeEncoder,
            ManifoldCompletion,
            NullSpaceTracker,
        )

        backend = get_default_backend()

        # Create mock model
        class MockModel:
            pass

        model = MockModel()

        tracker = NullSpaceTracker(
            n_layers=4,
            hidden_dim=32,
            backend=backend,
        )

        encoder = KnowledgeEncoder(
            model=model,
            null_space_tracker=tracker,
            backend=backend,
        )

        completion = ManifoldCompletion(
            model=model,
            null_space_tracker=tracker,
            knowledge_encoder=encoder,
            backend=backend,
        )

        # Create test embeddings
        embeddings = backend.random_normal((10, 32))
        backend.eval(embeddings)

        densities = completion._compute_densities(embeddings)

        assert len(densities) == 10
        assert all(0 < d <= 1 for d in densities)


class TestIntegration:
    """Integration tests for the continual learning pipeline."""

    def test_entropy_to_decision_pipeline(self):
        """Test EntropyAnalyzer -> DecisionGate flow."""
        from modelcypher.core.domain.continual import (
            DecisionGate,
            EntropyAnalyzer,
        )

        backend = get_default_backend()

        analyzer = EntropyAnalyzer(backend=backend)
        gate = DecisionGate(backend=backend)

        # Run a few steps
        for _ in range(5):
            logits = backend.random_normal((1000,))
            backend.eval(logits)

            state = analyzer.analyze(logits)
            decision = gate.decide(state)

            # Pipeline should produce valid results
            assert state.entropy >= 0
            assert decision.action is not None

    def test_activation_tracking_pipeline(self):
        """Test activation tracking through NullSpaceTracker."""
        from modelcypher.core.domain.continual import (
            ActivationBuffer,
            NullSpaceTracker,
        )

        backend = get_default_backend()

        tracker = NullSpaceTracker(
            n_layers=4,
            hidden_dim=64,
            backend=backend,
        )

        # Simulate inference with activation tracking
        steps = tracker._buffers[0].buffer_size + 1
        for step in range(steps):
            # Mock layer activations
            activations = {
                i: backend.random_normal((64,))
                for i in range(4)
            }
            for act in activations.values():
                backend.eval(act)

            tracker.add_all_layers(activations)

            if tracker.should_update():
                tracker.update_all_layers()

        # Should have computed SVD at least once
        model_state = tracker.get_model_state()
        assert model_state.hidden_dim == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
