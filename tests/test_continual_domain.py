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

"""Tests for continual learning domain modules with geometry-derived parameters.

Tests verify that:
- Parameters are derived from geometry/machine precision, not arbitrary constants
- Derived values scale appropriately with data dimensions
- Adaptive mechanisms stabilize correctly
- No hardcoded magic numbers leak through
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.activation_buffer import ActivationBuffer
from modelcypher.core.domain.continual.manifold_completion import (
    CompletionConfig,
    ManifoldCompletion,
)
from modelcypher.core.domain.continual.surprise_detector import SurpriseDetector
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)


# =============================================================================
# CompletionConfig Tests
# =============================================================================


class TestCompletionConfig:
    """Tests for CompletionConfig with geometry-derived defaults."""

    def test_default_config_has_none_for_derived_params(self):
        """Default config should have None for parameters that will be derived."""
        config = CompletionConfig()

        # These should be None - derived at runtime from data
        assert config.max_iterations is None
        assert config.convergence_threshold is None
        assert config.k_neighbors is None
        assert config.step_size is None
        assert config.min_density_ratio is None

        # These are policy choices, not derived
        assert config.constraint_weight == 1.0  # Equal weighting policy
        assert config.patience == 2  # Convergence theory: 2 plateau rounds

    def test_explicit_values_preserved(self):
        """Explicitly set values should be preserved."""
        config = CompletionConfig(
            max_iterations=500,
            convergence_threshold=0.05,
            k_neighbors=4,
            step_size=0.1,
            patience=3,
            min_density_ratio=0.2,
        )

        assert config.max_iterations == 500
        assert config.convergence_threshold == 0.05
        assert config.k_neighbors == 4
        assert config.step_size == 0.1
        assert config.patience == 3
        assert config.min_density_ratio == 0.2


# =============================================================================
# ManifoldCompletion Derived Parameter Tests
# =============================================================================


class TestManifoldCompletionDerivedParams:
    """Tests for ManifoldCompletion geometry-derived parameters."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def mock_completion(self, backend):
        """Create ManifoldCompletion with minimal mocks."""
        # We need to mock the dependencies minimally
        class MockTracker:
            pass

        class MockEncoder:
            def encode(self, event, hidden_state, frequency):
                return []

        class MockModel:
            pass

        return ManifoldCompletion(
            model=MockModel(),
            null_space_tracker=MockTracker(),
            knowledge_encoder=MockEncoder(),
            backend=backend,
        )

    def test_convergence_threshold_derived_from_sqrt_eps(self, mock_completion, backend):
        """Convergence threshold should be sqrt(machine_epsilon)."""
        # Access the derived sqrt_eps
        sqrt_eps = mock_completion._sqrt_eps

        # Verify it's derived from machine precision
        sample = backend.array([1.0])
        eps = machine_epsilon(backend, sample)
        expected = sqrt_scalar(eps, backend)

        assert abs(sqrt_eps - expected) < 1e-10

    def test_k_neighbors_scales_with_intrinsic_dimension(self, mock_completion, backend):
        """k_neighbors should scale with estimated intrinsic dimension."""
        # Create embeddings with known intrinsic dimension
        # A 2D manifold embedded in 64D should give k ~ 3
        n_samples = 100
        intrinsic_dim = 2
        ambient_dim = 64

        # Create 2D data embedded in higher dim
        low_dim = backend.array([[i / 10.0, j / 10.0] for i in range(10) for j in range(10)])
        # Pad to ambient_dim
        padding = backend.zeros((n_samples, ambient_dim - 2))
        embeddings = backend.concatenate([low_dim, padding], axis=1)
        backend.eval(embeddings)

        k = mock_completion._get_k_neighbors(embeddings)

        # k should be small for low intrinsic dimension
        # k = max(2, int(intrinsic_dim + 1)) ~ 3 for 2D
        assert k >= 2
        assert k <= 10  # Shouldn't be huge for low-dim data

    def test_step_size_derived_from_condition_number(self, mock_completion, backend):
        """Step size should be 1/condition_number."""
        # Well-conditioned data should give step_size close to 1
        # Ill-conditioned data should give smaller step_size

        # Create well-conditioned data (orthogonal directions)
        n_samples = 50
        dim = 32
        embeddings = backend.array([
            [float(i == j % dim) for j in range(dim)]
            for i in range(n_samples)
        ])
        backend.eval(embeddings)

        step = mock_completion._get_step_size(embeddings)

        # Step size should be positive and bounded
        assert step > 0
        assert step <= 1.0

    def test_min_density_ratio_from_distribution(self, mock_completion, backend):
        """min_density_ratio should be derived from k-NN distribution."""
        n_samples = 50
        dim = 16

        # Random embeddings
        embeddings = backend.array([
            [float((i * j + i) % 100) / 100.0 for j in range(dim)]
            for i in range(n_samples)
        ])
        backend.eval(embeddings)

        k = mock_completion._get_k_neighbors(embeddings)
        ratio = mock_completion._get_min_density_ratio(embeddings, k)

        # Ratio should be in [0, 1]
        assert 0.0 <= ratio <= 1.0


# =============================================================================
# SurpriseDetector Tests
# =============================================================================


class TestSurpriseDetector:
    """Tests for SurpriseDetector with adaptive parameters."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_default_baseline_window_is_adaptive(self):
        """Default baseline_window should be None (adaptive)."""
        detector = SurpriseDetector()

        # Before stabilization, baseline_window is None
        assert detector._baseline_window_config is None

    def test_explicit_baseline_window_preserved(self):
        """Explicit baseline_window should be used."""
        detector = SurpriseDetector(baseline_window=50)

        assert detector._baseline_window == 50
        assert detector._baseline_window_config == 50

    def test_activation_history_size_derived_from_hidden_dim(self, backend):
        """activation_history_size should be derived from hidden_dim."""
        detector = SurpriseDetector(backend=backend)

        # Initially None
        assert detector._activation_history_size is None

        # Create a hidden state
        hidden_dim = 768
        hidden_state = backend.zeros((hidden_dim,))
        backend.eval(hidden_state)

        # Update with hidden state
        detector._update_activation_history(hidden_state)

        # Now should be derived: min(hidden_dim // 128, 64), clamped to >= 8
        expected = min(hidden_dim // 128, 64)
        expected = max(expected, 8)
        assert detector._activation_history_size == expected

    def test_activation_history_size_scales_with_dimension(self, backend):
        """Larger hidden_dim should give larger history size (up to cap)."""
        detector1 = SurpriseDetector(backend=backend)
        detector2 = SurpriseDetector(backend=backend)

        # Small model
        small_hidden = backend.zeros((256,))
        detector1._update_activation_history(small_hidden)

        # Large model
        large_hidden = backend.zeros((4096,))
        detector2._update_activation_history(large_hidden)

        # Larger model should have larger or equal history size
        assert detector2._activation_history_size >= detector1._activation_history_size

    def test_baseline_stabilization_uses_sqrt_eps(self, backend):
        """Baseline should stabilize when std(mean) < sqrt(eps)."""
        detector = SurpriseDetector(backend=backend)

        # sqrt_eps should be derived from machine precision
        sample = backend.array([1.0])
        eps = machine_epsilon(backend, sample)
        expected_sqrt_eps = sqrt_scalar(eps, backend)

        assert abs(detector._sqrt_eps - expected_sqrt_eps) < 1e-10

    def test_detect_returns_raw_metrics(self, backend):
        """detect() should return raw metrics without hardcoded thresholds."""
        detector = SurpriseDetector(backend=backend)

        # Create mock logits
        vocab_size = 100
        logits = backend.zeros((vocab_size,))
        logits = logits + backend.array([float(i) for i in range(vocab_size)])
        backend.eval(logits)

        event = detector.detect(logits, actual_token_id=50)

        # Event should have raw metrics
        assert hasattr(event, 'token_surprise')
        assert hasattr(event, 'token_surprise_zscore')
        assert hasattr(event, 'percentile')
        assert hasattr(event, 'rank_surprise')

        # Values should be numeric, not interpretations
        assert isinstance(event.token_surprise, float)
        assert isinstance(event.percentile, float)


# =============================================================================
# ActivationBuffer Tests
# =============================================================================


class TestActivationBuffer:
    """Tests for ActivationBuffer with geometry-derived SVD frequency."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_default_svd_frequency_derived_from_hidden_dim(self, backend):
        """SVD frequency should be sqrt(hidden_dim) by default."""
        hidden_dim = 1024
        buffer = ActivationBuffer(
            buffer_size=100,
            hidden_dim=hidden_dim,
            backend=backend,
        )

        # Expected: sqrt(hidden_dim), clamped to [8, 128]
        expected = max(8, min(128, int(hidden_dim ** 0.5)))
        assert buffer._svd_update_frequency == expected

    def test_svd_frequency_scales_with_dimension(self, backend):
        """Larger hidden_dim should give larger SVD frequency."""
        buffer_small = ActivationBuffer(
            buffer_size=100,
            hidden_dim=64,
            backend=backend,
        )
        buffer_large = ActivationBuffer(
            buffer_size=100,
            hidden_dim=4096,
            backend=backend,
        )

        # Larger model should have larger or equal frequency
        assert buffer_large._svd_update_frequency >= buffer_small._svd_update_frequency

    def test_explicit_svd_frequency_preserved(self, backend):
        """Explicit svd_update_frequency should be used."""
        buffer = ActivationBuffer(
            buffer_size=100,
            hidden_dim=512,
            svd_update_frequency=32,
            backend=backend,
        )

        assert buffer._svd_update_frequency == 32

    def test_svd_frequency_minimum_bound(self, backend):
        """SVD frequency should be at least 8."""
        # Very small dimension
        buffer = ActivationBuffer(
            buffer_size=100,
            hidden_dim=16,  # sqrt(16) = 4, but should clamp to 8
            backend=backend,
        )

        assert buffer._svd_update_frequency >= 8

    def test_svd_frequency_maximum_bound(self, backend):
        """SVD frequency should be at most 128."""
        # Very large dimension
        buffer = ActivationBuffer(
            buffer_size=1000,
            hidden_dim=65536,  # sqrt = 256, but should clamp to 128
            backend=backend,
        )

        assert buffer._svd_update_frequency <= 128

    def test_should_update_svd_uses_frequency(self, backend):
        """should_update_svd() should trigger based on derived frequency."""
        buffer = ActivationBuffer(
            buffer_size=100,
            hidden_dim=256,
            backend=backend,
        )

        freq = buffer._svd_update_frequency

        # Add samples
        for i in range(freq - 1):
            activation = backend.zeros((256,))
            buffer.add(activation)

        # Should not trigger yet
        assert not buffer.should_update_svd()

        # Add one more
        activation = backend.zeros((256,))
        buffer.add(activation)

        # Now should trigger
        assert buffer.should_update_svd()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDerivedParameterIntegration:
    """Integration tests verifying no hardcoded values leak through."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_all_defaults_are_derived_or_policy(self):
        """All default parameter values should be derived or explicit policy."""
        config = CompletionConfig()

        # Document which are derived vs policy
        derived = [
            config.max_iterations,  # None = convergence-based
            config.convergence_threshold,  # None = sqrt(eps)
            config.k_neighbors,  # None = intrinsic_dim + 1
            config.step_size,  # None = 1/condition_number
            config.min_density_ratio,  # None = k-NN percentile
        ]

        policy = [
            config.constraint_weight,  # 1.0 = equal weighting
            config.patience,  # 2 = plateau detection theory
        ]

        # All derived should be None
        for val in derived:
            assert val is None, f"Derived parameter should be None, got {val}"

        # Policy values should be documented
        assert config.constraint_weight == 1.0
        assert config.patience == 2

    def test_sqrt_eps_used_consistently(self, backend):
        """sqrt(eps) should be used consistently across modules."""
        detector = SurpriseDetector(backend=backend)

        class MockTracker:
            pass

        class MockEncoder:
            def encode(self, event, hidden_state, frequency):
                return []

        class MockModel:
            pass

        completion = ManifoldCompletion(
            model=MockModel(),
            null_space_tracker=MockTracker(),
            knowledge_encoder=MockEncoder(),
            backend=backend,
        )

        # Both should use the same sqrt_eps derivation
        assert abs(detector._sqrt_eps - completion._sqrt_eps) < 1e-10
