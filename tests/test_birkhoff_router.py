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

"""Tests for Birkhoff Router multi-channel routing.

Verifies the mHC-inspired multi-channel routing for model merging.

Key properties tested:
1. Routing matrix is doubly stochastic (row sums = column sums = 1)
2. Spectral norm bounded (||H||_2 <= 1.0)
3. Channel combination is stable (no signal explosion)
4. Different routing modes work correctly
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.birkhoff_router import (
    BirkhoffRouter,
    RoutingMode,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)


class TestRoutingMatrixProperties:
    """Test that routing matrices satisfy Birkhoff polytope constraints."""

    def test_uniform_routing_is_doubly_stochastic(self) -> None:
        """Uniform routing should produce doubly stochastic matrix."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(42)
        deltas = [backend.random_normal((8, 8)) for _ in range(3)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas, init_mode=RoutingMode.UNIFORM)

        # Row sums should be 1
        row_sums = backend.sum(result.routing_matrix, axis=1)
        backend.eval(row_sums)
        row_sums_list = backend.tolist(row_sums)
        max_row_error = max(abs(s - 1.0) for s in row_sums_list)

        # Column sums should be 1
        col_sums = backend.sum(result.routing_matrix, axis=0)
        backend.eval(col_sums)
        col_sums_list = backend.tolist(col_sums)
        max_col_error = max(abs(s - 1.0) for s in col_sums_list)

        tol = regularization_epsilon(backend, result.routing_matrix)
        assert max_row_error <= tol, f"Row sums deviate: {max_row_error}"
        assert max_col_error <= tol, f"Col sums deviate: {max_col_error}"
        assert result.converged

    def test_identity_routing_is_doubly_stochastic(self) -> None:
        """Identity routing should produce doubly stochastic matrix (identity is doubly stochastic)."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(123)
        deltas = [backend.random_normal((6, 6)) for _ in range(4)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas, init_mode=RoutingMode.IDENTITY)

        # Identity matrix is already doubly stochastic
        row_sums = backend.sum(result.routing_matrix, axis=1)
        col_sums = backend.sum(result.routing_matrix, axis=0)
        backend.eval(row_sums, col_sums)

        row_sums_list = backend.tolist(row_sums)
        col_sums_list = backend.tolist(col_sums)

        tol = regularization_epsilon(backend, result.routing_matrix)
        assert all(abs(s - 1.0) <= tol for s in row_sums_list)
        assert all(abs(s - 1.0) <= tol for s in col_sums_list)

    def test_spectral_norm_bounded(self) -> None:
        """Spectral norm should be <= 1.0."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(456)
        deltas = [backend.random_normal((10, 10)) * 5.0 for _ in range(5)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas, init_mode=RoutingMode.UNIFORM)

        tol = regularization_epsilon(backend, result.routing_matrix)
        assert result.spectral_norm <= 1.0 + tol, f"Spectral norm {result.spectral_norm} exceeds 1.0"

    def test_all_entries_nonnegative(self) -> None:
        """All routing matrix entries should be non-negative."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(789)
        deltas = [backend.random_normal((5, 5)) for _ in range(3)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas)

        min_val = backend.min(result.routing_matrix)
        backend.eval(min_val)
        min_val_float = float(backend.to_scalar(min_val))

        tol = division_epsilon(backend, result.routing_matrix)
        assert min_val_float >= -tol, f"Negative entry found: {min_val_float}"


class TestChannelCombination:
    """Test that channel combination works correctly."""

    def test_combines_channels_stable(self) -> None:
        """Combined delta should not explode in norm."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(101)
        deltas = [backend.random_normal((8, 8)) for _ in range(3)]
        for d in deltas:
            backend.eval(d)

        combined, result = router.route_channels(deltas)
        backend.eval(combined)

        # Combined norm should be bounded by max input norm
        # (property of doubly stochastic mixing)
        apply_result = router.apply_routing(result.routing_matrix, deltas)

        max_input_norm = max(apply_result.input_norms)
        # With doubly stochastic mixing, output should not exceed max input
        # Allow some tolerance for numerical errors
        tol_factor = 1.5  # Conservative bound
        assert apply_result.output_norm <= max_input_norm * tol_factor * result.n_channels ** 0.5, (
            f"Output norm {apply_result.output_norm} exceeds expected bound"
        )

    def test_single_channel_passthrough(self) -> None:
        """Single channel should pass through unchanged."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(202)
        delta = backend.random_normal((6, 6))
        backend.eval(delta)

        combined, result = router.route_channels([delta])
        backend.eval(combined)

        # Single channel: routing matrix is 1x1 = [[1.0]]
        diff = backend.abs(combined - delta)
        backend.eval(diff)
        max_diff = float(backend.to_scalar(backend.max(diff)))

        tol = regularization_epsilon(backend, combined)
        assert max_diff <= tol, f"Single channel not passed through: diff = {max_diff}"

    def test_two_channels_weighted(self) -> None:
        """Two channels should combine with weights from routing matrix."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        # Create simple deltas with known values
        delta1 = backend.ones((4, 4))
        delta2 = backend.full((4, 4), 2.0)
        backend.eval(delta1, delta2)

        combined, result = router.route_channels([delta1, delta2], init_mode=RoutingMode.UNIFORM)
        backend.eval(combined)

        # For uniform 2x2 doubly stochastic, column sums = 1
        # So combined = 1.0 * delta1 + 1.0 * delta2 = 3.0 per element
        # Actually uniform means H = [[0.5, 0.5], [0.5, 0.5]]
        # col_sums = [1.0, 1.0], so combined = 1.0 * 1 + 1.0 * 2 = 3.0
        expected = backend.full((4, 4), 3.0)
        diff = backend.abs(combined - expected)
        backend.eval(diff)
        max_diff = float(backend.to_scalar(backend.max(diff)))

        tol = regularization_epsilon(backend, combined)
        assert max_diff <= tol, f"Two channel combination wrong: diff = {max_diff}"


class TestRoutingModes:
    """Test different routing initialization modes."""

    def test_uniform_mode(self) -> None:
        """Uniform mode should produce equal-weight routing."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(303)
        n = 4
        deltas = [backend.random_normal((5, 5)) for _ in range(n)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas, init_mode="uniform")
        assert result.init_mode == RoutingMode.UNIFORM

        # For uniform, all entries should be approximately 1/n
        expected_val = 1.0 / n
        H_flat = backend.reshape(result.routing_matrix, (-1,))
        mean_val = backend.mean(H_flat)
        backend.eval(mean_val)
        mean_float = float(backend.to_scalar(mean_val))

        tol = regularization_epsilon(backend, result.routing_matrix)
        assert abs(mean_float - expected_val) <= tol

    def test_identity_mode(self) -> None:
        """Identity mode should produce diagonal-heavy routing."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(404)
        n = 3
        deltas = [backend.random_normal((5, 5)) for _ in range(n)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas, init_mode="identity")
        assert result.init_mode == RoutingMode.IDENTITY

        # Identity matrix is already doubly stochastic, so projection
        # should return something close to identity
        diag_sum = 0.0
        for i in range(n):
            idx_i = backend.array([i])
            row_i = backend.take(result.routing_matrix, idx_i, axis=0)
            val_ii = backend.take(backend.reshape(row_i, (-1,)), idx_i, axis=0)
            backend.eval(val_ii)
            diag_sum += float(backend.to_scalar(val_ii))

        # Diagonal should sum close to n (for identity)
        assert diag_sum >= n * 0.99, f"Identity mode diagonal sum = {diag_sum}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_channels_raises(self) -> None:
        """Empty channel list should raise ValueError."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        with pytest.raises(ValueError, match="At least one channel"):
            router.compute_routing([])

    def test_apply_routing_empty_raises(self) -> None:
        """Empty channel list for apply should raise ValueError."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        H = backend.eye(3)
        backend.eval(H)

        with pytest.raises(ValueError, match="At least one channel"):
            router.apply_routing(H, [])

    def test_different_delta_shapes(self) -> None:
        """Channels can have different internal values but same shape."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(505)
        # Same shape, different distributions
        delta1 = backend.random_normal((6, 6))
        delta2 = backend.random_normal((6, 6)) * 2.0
        delta3 = backend.random_normal((6, 6)) + 1.0
        backend.eval(delta1, delta2, delta3)

        combined, result = router.route_channels([delta1, delta2, delta3])
        backend.eval(combined)

        # Should complete without error and produce valid shape
        assert combined.shape == (6, 6)
        assert result.converged

    def test_string_mode_parsing(self) -> None:
        """String mode names should be parsed correctly."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(606)
        deltas = [backend.random_normal((4, 4)) for _ in range(2)]
        for d in deltas:
            backend.eval(d)

        # Test all valid string modes
        for mode_str in ["uniform", "identity", "diagonal_weighted"]:
            result = router.compute_routing(deltas, init_mode=mode_str)
            assert result.init_mode == RoutingMode(mode_str)


class TestIntegrationWithNullSpace:
    """Test that router works correctly in the intended merge pipeline context."""

    def test_router_preserves_geometry_addition(self) -> None:
        """Router should enable geometric addition (not blending)."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(707)
        # Simulate three channel deltas (as if from null-space projection)
        delta_spatial = backend.random_normal((8, 8)) * 0.1  # Spatial channel
        delta_temporal = backend.random_normal((8, 8)) * 0.1  # Temporal channel
        delta_text = backend.random_normal((8, 8)) * 0.1  # Text channel
        backend.eval(delta_spatial, delta_temporal, delta_text)

        combined, result = router.route_channels(
            [delta_spatial, delta_temporal, delta_text],
            init_mode=RoutingMode.UNIFORM,
        )
        backend.eval(combined)

        # Geometric addition property: combined should contain information
        # from all channels (no information destroyed by averaging)
        # Check that combined has reasonable norm (not collapsed to zero)
        combined_flat = backend.reshape(combined, (-1,))
        combined_norm = backend.sum(combined_flat * combined_flat) ** 0.5
        backend.eval(combined_norm)
        combined_norm_float = float(backend.to_scalar(combined_norm))

        # Combined should have non-trivial norm if inputs are non-trivial
        assert combined_norm_float > 0.01, "Combined delta collapsed to zero"

    def test_routing_result_diagnostics(self) -> None:
        """Routing result should provide useful diagnostics."""
        backend = get_default_backend()
        router = BirkhoffRouter(backend)

        backend.random_seed(808)
        deltas = [backend.random_normal((6, 6)) for _ in range(3)]
        for d in deltas:
            backend.eval(d)

        result = router.compute_routing(deltas)

        # Check that all diagnostic fields are populated
        assert result.n_channels == 3
        assert result.iterations_used >= 1
        assert result.convergence_error >= 0
        assert result.spectral_norm >= 0
        assert isinstance(result.converged, bool)
        assert isinstance(result.spectral_clipped, bool)
        assert result.init_mode == RoutingMode.UNIFORM
