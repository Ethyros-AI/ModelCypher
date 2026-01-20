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

"""Tests for continual learning domain modules with geometry-derived behavior."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.activation_buffer import ActivationBuffer
from modelcypher.core.domain.continual.manifold_completion import ManifoldCompletion
from modelcypher.core.domain.continual.surprise_detector import SurpriseDetector
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)


class TestManifoldCompletionDerivedParams:
    """Tests for ManifoldCompletion geometry-derived parameters."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def mock_completion(self, backend):
        """Create ManifoldCompletion with minimal mocks."""
        class MockTracker:
            """Minimal tracker stub."""

        class MockEncoder:
            def encode(self, event, hidden_state):
                return []

        class MockModel:
            """Minimal model stub."""

        return ManifoldCompletion(
            model=MockModel(),
            null_space_tracker=MockTracker(),
            knowledge_encoder=MockEncoder(),
            backend=backend,
        )

    def test_convergence_threshold_derived_from_sqrt_eps(self, mock_completion, backend):
        """Convergence threshold should be sqrt(machine_epsilon)."""
        sqrt_eps = mock_completion._sqrt_eps

        sample = backend.array([1.0])
        eps = machine_epsilon(backend, sample)
        expected = sqrt_scalar(eps, backend)

        assert abs(sqrt_eps - expected) < 1e-10

    def test_k_neighbors_scales_with_intrinsic_dimension(self, mock_completion, backend):
        """k_neighbors should scale with estimated intrinsic dimension."""
        n_samples = 100
        ambient_dim = 64

        low_dim = backend.array([[i / 10.0, j / 10.0] for i in range(10) for j in range(10)])
        padding = backend.zeros((n_samples, ambient_dim - 2))
        embeddings = backend.concatenate([low_dim, padding], axis=1)
        backend.eval(embeddings)

        k = mock_completion._get_k_neighbors(embeddings)

        assert k >= 2
        assert k <= 10

    def test_step_size_derived_from_condition_number(self, mock_completion, backend):
        """Step size should be 1/condition_number."""
        n_samples = 50
        dim = 32
        embeddings = backend.array([
            [float(i == j % dim) for j in range(dim)]
            for i in range(n_samples)
        ])
        backend.eval(embeddings)

        step = mock_completion._get_step_size(embeddings)

        assert step > 0
        assert step <= 1.0


class TestSurpriseDetector:
    """Tests for SurpriseDetector with running baselines."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_detect_returns_raw_metrics(self, backend):
        """detect() should return raw metrics without hardcoded thresholds."""
        detector = SurpriseDetector(backend=backend)

        vocab_size = 100
        logits = backend.zeros((vocab_size,))
        logits = logits + backend.array([float(i) for i in range(vocab_size)])
        backend.eval(logits)

        event = detector.detect(logits, actual_token_id=50)

        assert hasattr(event, "token_surprise")
        assert hasattr(event, "token_surprise_zscore")
        assert hasattr(event, "percentile")
        assert hasattr(event, "rank_surprise")
        assert isinstance(event.token_surprise, float)
        assert isinstance(event.percentile, float)

    def test_baseline_tracks_running_mean(self, backend):
        """Baseline surprise should track the running mean of history."""
        detector = SurpriseDetector(backend=backend)

        surprises = []
        for i in range(5):
            logits = backend.random_normal((100,)) + float(i)
            backend.eval(logits)
            event = detector.detect(logits, actual_token_id=i)
            surprises.append(event.token_surprise)

        expected = sum(surprises) / len(surprises)
        assert abs(detector.get_baseline_surprise() - expected) < 1e-6

    def test_activation_surprise_reflects_shift(self, backend):
        """Activation surprise should increase when activations shift."""
        detector = SurpriseDetector(backend=backend)

        logits = backend.random_normal((50,))
        backend.eval(logits)

        # Use non-zero baseline activations (zeros cause norm=0 → early return)
        hidden_a = backend.ones((16,))
        # Shifted activation - significantly different from baseline
        hidden_b = backend.ones((16,)) * 10.0
        backend.eval(hidden_a, hidden_b)

        # First call: activation_count=0, so activation_surprise=0.0
        event_a = detector.detect(logits, actual_token_id=1, hidden_state=hidden_a)
        # Second call: activation_count=1, still < 2, so activation_surprise=0.0
        event_b = detector.detect(logits, actual_token_id=2, hidden_state=hidden_a)
        # Third call: activation_count=2, now we can compare against mean
        event_c = detector.detect(logits, actual_token_id=3, hidden_state=hidden_b)

        # First two calls return 0.0 because we need at least 2 samples for baseline
        assert event_a.activation_surprise == 0.0
        assert event_b.activation_surprise == 0.0
        # Third call with different activation should show surprise
        assert event_c.activation_surprise > 0.0


class TestActivationBuffer:
    """Tests for ActivationBuffer coverage-based SVD updates."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_should_update_svd_requires_full_coverage(self, backend):
        """SVD should only update after reaching algebraic minimum coverage."""
        hidden_dim = 8
        buffer = ActivationBuffer(hidden_dim=hidden_dim, backend=backend)

        for _ in range(buffer.buffer_size - 1):
            activation = backend.zeros((hidden_dim,))
            backend.eval(activation)
            buffer.add(activation)

        assert not buffer.should_update_svd()

        activation = backend.zeros((hidden_dim,))
        backend.eval(activation)
        buffer.add(activation)

        assert buffer.should_update_svd()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
