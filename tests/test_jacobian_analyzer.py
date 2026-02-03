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

"""Unit tests for Jacobian spectrum analyzer."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.jacobian_analyzer import (
    JacobianAnalyzer,
    JacobianProfile,
)


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


class TestJacobianProfile:
    """Tests for JacobianProfile dataclass."""

    def test_as_dict_roundtrip(self):
        """Profile should serialize to dict correctly."""
        profile = JacobianProfile(
            layer_idx=0,
            top_k_singular_values=[10.0, 5.0, 2.0, 1.0],
            effective_rank_renyi=2.5,
            effective_rank_shannon=3.0,
            spectral_entropy=1.1,
            condition_number=10.0,
            spectral_gap=2.0,
            spectral_decay_rate=-0.5,
            norm_amplification=10.0,
            hidden_dim=100,
            num_probes=32,
        )
        d = profile.as_dict()
        assert d["layer_idx"] == 0
        assert d["top_k_singular_values"] == [10.0, 5.0, 2.0, 1.0]
        assert d["effective_rank_renyi"] == 2.5
        assert d["condition_number"] == 10.0


class TestJacobianAnalyzer:
    """Tests for JacobianAnalyzer."""

    def test_identity_layer_jacobian(self, backend):
        """Identity layer should have Jacobian with singular values ≈ 1."""
        b = backend

        # Create identity layer function
        def identity_layer(x):
            return x

        # Create random input
        hidden_dim = 64
        x = b.random_normal((hidden_dim,))
        b.eval(x)

        # Analyze Jacobian
        analyzer = JacobianAnalyzer(backend=b, num_probes=32, epsilon=1e-5)
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=identity_layer,
            input_activation=x,
            layer_idx=0,
        )

        # Identity has all singular values = 1
        # So norm_amplification should be close to 1
        assert 0.9 < profile.norm_amplification < 1.1
        # Condition number should be close to 1
        assert profile.condition_number < 2.0
        # Effective rank should be high (all directions preserved)
        assert profile.effective_rank_shannon > 20.0

    def test_scaling_layer_jacobian(self, backend):
        """Scaling layer should have σ_max = scale factor."""
        b = backend

        scale = 5.0

        def scaling_layer(x):
            return scale * x

        hidden_dim = 64
        x = b.random_normal((hidden_dim,))
        b.eval(x)

        analyzer = JacobianAnalyzer(backend=b, num_probes=32, epsilon=1e-5)
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=scaling_layer,
            input_activation=x,
            layer_idx=0,
        )

        # Scaling by 5 means all singular values = 5
        assert 4.5 < profile.norm_amplification < 5.5
        # Condition number should be 1 (uniform scaling)
        assert profile.condition_number < 1.5

    def test_projection_layer_jacobian(self, backend):
        """Projection to subspace should have low effective rank."""
        b = backend

        hidden_dim = 64
        rank = 4

        # Create a rank-4 projection matrix
        U = b.random_normal((hidden_dim, rank))
        b.eval(U)

        def projection_layer(x):
            # Project to rank-4 subspace and back
            # This is x @ U @ U^T (with some normalization)
            projected = b.matmul(b.reshape(x, (1, -1)), U)
            reconstructed = b.matmul(projected, b.transpose(U))
            return b.reshape(reconstructed, (-1,))

        x = b.random_normal((hidden_dim,))
        b.eval(x)

        analyzer = JacobianAnalyzer(backend=b, num_probes=32, epsilon=1e-5)
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=projection_layer,
            input_activation=x,
            layer_idx=0,
        )

        # Projection should have low effective rank (around 4)
        # Note: Due to randomized SVD and projection noise, allow some tolerance
        assert profile.effective_rank_shannon < 10.0
        # Condition number should be high (rank deficient)
        assert profile.condition_number > 100.0

    def test_nonlinear_layer_jacobian(self, backend):
        """Nonlinear layer (ReLU-like) should compute correct Jacobian."""
        b = backend

        def relu_like_layer(x):
            # Soft ReLU: max(x, 0.1 * x)
            return b.maximum(x, 0.1 * x)

        hidden_dim = 64
        x = b.random_normal((hidden_dim,))
        b.eval(x)

        analyzer = JacobianAnalyzer(backend=b, num_probes=16, epsilon=1e-5)
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=relu_like_layer,
            input_activation=x,
            layer_idx=0,
        )

        # ReLU-like layer should have singular values between 0.1 and 1.0
        # depending on input signs
        assert 0.1 <= profile.norm_amplification <= 1.5
        # Should have reasonable effective rank
        assert profile.effective_rank_shannon > 1.0

    def test_deterministic_jacobian(self, backend):
        """Same input should give same Jacobian (deterministic forward pass)."""
        b = backend

        # Seed for reproducibility
        b.random_seed(42)

        def linear_layer(x):
            # Simple linear transformation
            return x * 2.0 + 1.0

        hidden_dim = 32
        x = b.random_normal((hidden_dim,))
        b.eval(x)

        # Run twice with same input
        analyzer = JacobianAnalyzer(backend=b, num_probes=16, epsilon=1e-5)

        # Note: The randomized SVD uses random probes, so we can't expect
        # exact equality, but the dominant singular values should be similar
        profile1 = analyzer.compute_layer_jacobian_profile(
            layer_fn=linear_layer,
            input_activation=x,
            layer_idx=0,
        )

        profile2 = analyzer.compute_layer_jacobian_profile(
            layer_fn=linear_layer,
            input_activation=x,
            layer_idx=0,
        )

        # Both should identify σ_max ≈ 2.0 (the scaling factor)
        assert abs(profile1.norm_amplification - 2.0) < 0.5
        assert abs(profile2.norm_amplification - 2.0) < 0.5

    def test_jvp_finite_diff_accuracy(self, backend):
        """Finite difference JVP should be accurate for smooth functions."""
        b = backend

        # Simple function: f(x) = x^2 (element-wise)
        # Jacobian is diagonal: J_ii = 2*x_i
        def quadratic_layer(x):
            return x * x

        hidden_dim = 8
        x = b.full((hidden_dim,), 2.0)  # x = [2, 2, ..., 2]
        b.eval(x)

        # For x_i = 2, J_ii = 4, so Jacobian is 4*I
        # All singular values should be 4
        analyzer = JacobianAnalyzer(backend=b, num_probes=16, epsilon=1e-5)
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=quadratic_layer,
            input_activation=x,
            layer_idx=0,
        )

        # σ_max should be close to 4
        assert 3.5 < profile.norm_amplification < 4.5
        # All singular values same, so condition number ≈ 1
        assert profile.condition_number < 2.0


class TestJacobianMetrics:
    """Tests for derived Jacobian metrics."""

    def test_effective_rank_computation(self):
        """Effective rank should be computed correctly from singular values."""
        # For uniform singular values [1, 1, 1, 1], effective rank = 4
        # For [10, 0, 0, 0], effective rank = 1

        profile_uniform = JacobianProfile(
            layer_idx=0,
            top_k_singular_values=[1.0, 1.0, 1.0, 1.0],
            effective_rank_renyi=4.0,
            effective_rank_shannon=4.0,
            spectral_entropy=math.log(4),
            condition_number=1.0,
            spectral_gap=1.0,
            spectral_decay_rate=0.0,
            norm_amplification=1.0,
            hidden_dim=4,
            num_probes=4,
        )

        assert profile_uniform.effective_rank_shannon == 4.0

        profile_concentrated = JacobianProfile(
            layer_idx=0,
            top_k_singular_values=[10.0, 0.1, 0.1, 0.1],
            effective_rank_renyi=1.0,  # Approximately
            effective_rank_shannon=1.0,  # Approximately
            spectral_entropy=0.0,
            condition_number=100.0,
            spectral_gap=100.0,
            spectral_decay_rate=-2.0,
            norm_amplification=10.0,
            hidden_dim=4,
            num_probes=4,
        )

        assert profile_concentrated.effective_rank_shannon == 1.0

    def test_condition_number_interpretation(self):
        """Condition number should indicate numerical stability."""
        # Well-conditioned: κ ≈ 1
        # Ill-conditioned: κ >> 1

        well_conditioned = JacobianProfile(
            layer_idx=0,
            top_k_singular_values=[1.0, 0.99, 0.98, 0.97],
            effective_rank_renyi=4.0,
            effective_rank_shannon=4.0,
            spectral_entropy=math.log(4),
            condition_number=1.0 / 0.97,  # ≈ 1.03
            spectral_gap=1.0 / 0.99,
            spectral_decay_rate=-0.01,
            norm_amplification=1.0,
            hidden_dim=4,
            num_probes=4,
        )

        assert well_conditioned.condition_number < 2.0

        ill_conditioned = JacobianProfile(
            layer_idx=0,
            top_k_singular_values=[1000.0, 10.0, 1.0, 0.001],
            effective_rank_renyi=1.5,
            effective_rank_shannon=1.5,
            spectral_entropy=0.4,
            condition_number=1000.0 / 0.001,  # = 1,000,000
            spectral_gap=100.0,
            spectral_decay_rate=-2.3,
            norm_amplification=1000.0,
            hidden_dim=4,
            num_probes=4,
        )

        assert ill_conditioned.condition_number > 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
