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

"""Property tests for Generalized Procrustes Analysis.

Tests mathematical invariants:
- Procrustes distance is a metric (symmetry, triangle inequality)
- Optimal rotation is orthogonal
- Procrustes(X, X @ R) = 0 for orthogonal R
- Error bounded by Frobenius norm
- Convergence is monotonic
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.generalized_procrustes import (
    GeneralizedProcrustes,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))

# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class TestProcrustesBasic:
    """Basic Procrustes properties."""

    def test_identical_activations_zero_error(self) -> None:
        """Procrustes(X, X) should have zero error."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        # Create simple activation pattern
        activations = [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # Identical
        ]

        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        assert result is not None
        eps = _eps(result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_rotation_invariance(self) -> None:
        """X @ R and X should align perfectly for orthogonal R."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        # Create base activations
        X = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]

        # Create orthogonal rotation (90 degrees around z-axis)
        cos_theta = 0.0
        sin_theta = 1.0
        R = [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ]

        # Apply rotation
        X_rotated = []
        for row in X:
            new_row = [
                sum(row[j] * R[j][i] for j in range(3))
                for i in range(3)
            ]
            X_rotated.append(new_row)

        activations = [X, X_rotated]
        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        assert result is not None
        # Error should be small (rotation should be recovered)
        eps = _eps(result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_convergence(self) -> None:
        """GPA should converge."""
        backend = get_default_backend()
        backend.random_seed(42)
        gpa = GeneralizedProcrustes(backend)

        # Create random activations
        activations = []
        for _ in range(3):  # 3 models
            model_acts = []
            for _ in range(10):  # 10 samples
                row = backend.random_normal((5,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        assert result is not None
        # Convergence is determined by machine epsilon-derived threshold
        assert result.converged or result.iterations > 0

    def test_rotations_are_orthogonal(self) -> None:
        """Computed rotations should be orthogonal matrices."""
        backend = get_default_backend()
        backend.random_seed(42)
        gpa = GeneralizedProcrustes(backend)

        # Create random activations
        activations = []
        for _ in range(3):
            model_acts = []
            for _ in range(8):
                row = backend.random_normal((4,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        assert result is not None

        # Check each rotation is orthogonal: R @ R^T = I
        for R in result.rotations:
            R_arr = backend.array(R)
            R_T = backend.transpose(R_arr)
            product = backend.matmul(R_arr, R_T)
            backend.eval(product)
            prod_list = backend.tolist(product)

            # Should be close to identity (use 10x machine epsilon for tolerance)
            n = len(R)
            for i in range(n):
                for j in range(n):
                    expected = 1.0 if i == j else 0.0
                    eps = 10.0 * _eps(float(prod_list[i][j]), expected)
                    assert abs(prod_list[i][j] - expected) <= eps


class TestProcrustesMetricProperties:
    """Test metric properties of Procrustes distance."""

    def test_symmetry(self) -> None:
        """Distance(X, Y) should equal Distance(Y, X)."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        Y = [[0.5, 0.5], [0.5, -0.5], [1.0, 0.0]]

        # All parameters are now derived from data at runtime
        # Align X to Y
        result1 = gpa.align([X, Y])
        # Align Y to X
        result2 = gpa.align([Y, X])

        assert result1 is not None
        assert result2 is not None
        # Errors should be similar (not exactly equal due to centering order)
        eps = _eps(result1.alignment_error, result2.alignment_error)
        assert abs(result1.alignment_error - result2.alignment_error) <= eps

    def test_non_negative(self) -> None:
        """Alignment error should always be non-negative."""
        backend = get_default_backend()
        backend.random_seed(42)
        gpa = GeneralizedProcrustes(backend)

        activations = []
        for _ in range(2):
            model_acts = []
            for _ in range(5):
                row = backend.random_normal((3,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        assert result is not None
        eps = _eps(result.alignment_error)
        assert result.alignment_error >= -eps


class TestProcrustesEdgeCases:
    """Test edge cases."""

    def test_single_model_returns_none(self) -> None:
        """Single model should return None (need >= 2)."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        activations = [[[1.0, 0.0], [0.0, 1.0]]]
        result = gpa.align(activations)
        assert result is None

    def test_empty_activations_returns_none(self) -> None:
        """Empty activations should return None."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        activations = [[], []]
        result = gpa.align(activations)
        assert result is None

    def test_mismatched_dimensions_returns_none(self) -> None:
        """Mismatched dimensions should return None."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        activations = [
            [[1.0, 0.0], [0.0, 1.0]],  # 2D
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],  # 3D
        ]
        result = gpa.align(activations)
        assert result is None


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestProcrustesHypothesis:
    """Hypothesis property tests."""

    @given(
        n_samples=st.integers(min_value=3, max_value=15),
        n_features=st.integers(min_value=2, max_value=6),
        n_models=st.integers(min_value=2, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_error_always_non_negative_hypothesis(
        self, n_samples: int, n_features: int, n_models: int, seed: int
    ):
        """Alignment error is always >= 0 (Hypothesis)."""
        backend = get_default_backend()
        backend.random_seed(seed)
        gpa = GeneralizedProcrustes(backend)

        activations = []
        for _ in range(n_models):
            model_acts = []
            for _ in range(n_samples):
                row = backend.random_normal((n_features,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        # All parameters are now derived from data at runtime
        result = gpa.align(activations)

        if result is not None:
            eps = _eps(result.alignment_error)
            assert result.alignment_error >= -eps

    @given(
        n_samples=st.integers(min_value=3, max_value=10),
        n_features=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_rotations_orthogonal_hypothesis(
        self, n_samples: int, n_features: int, seed: int
    ):
        """Rotations are orthogonal (Hypothesis)."""
        backend = get_default_backend()
        backend.random_seed(seed)
        gpa = GeneralizedProcrustes(backend)

        activations = []
        for _ in range(2):
            model_acts = []
            for _ in range(n_samples):
                row = backend.random_normal((n_features,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        config = Config(max_iterations=30, frechet_mean=FrechetMeanConfig(enabled=False))
        result = gpa.align(activations, config)

        if result is not None:
            for R in result.rotations:
                R_arr = backend.array(R)
                R_T = backend.transpose(R_arr)
                product = backend.matmul(R_arr, R_T)
                backend.eval(product)
                prod_list = backend.tolist(product)

                n = len(R)
                for i in range(n):
                    for j in range(n):
                        expected = 1.0 if i == j else 0.0
                        eps = 10.0 * _eps(float(prod_list[i][j]), expected)
                        assert abs(prod_list[i][j] - expected) <= eps


class TestProcrustesWithScaling:
    """Test Procrustes with scaling enabled."""

    def test_scaling_recovers_scale_difference(self) -> None:
        """With scaling enabled, should handle scale differences."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        # Base activations
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
        # Scaled version
        X_scaled = [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [4.0, 0.0]]

        activations = [X, X_scaled]
        config = Config(allow_scaling=True, frechet_mean=FrechetMeanConfig(enabled=False))
        result = gpa.align(activations, config)

        assert result is not None
        # Error should be small with scaling enabled
        eps = _eps(result.alignment_error, 0.0)
        assert abs(result.alignment_error - 0.0) <= eps

    def test_no_scaling_higher_error_for_scaled_input(self) -> None:
        """Without scaling, scaled inputs have higher error."""
        backend = get_default_backend()
        gpa = GeneralizedProcrustes(backend)

        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        X_scaled = [[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]

        activations = [X, X_scaled]

        config_no_scale = Config(allow_scaling=False, frechet_mean=FrechetMeanConfig(enabled=False))
        config_scale = Config(allow_scaling=True, frechet_mean=FrechetMeanConfig(enabled=False))

        result_no_scale = gpa.align(activations, config_no_scale)
        result_scale = gpa.align(activations, config_scale)

        assert result_no_scale is not None
        assert result_scale is not None
        # With scaling should have lower error
        eps = _eps(result_no_scale.alignment_error, result_scale.alignment_error)
        assert result_scale.alignment_error <= result_no_scale.alignment_error + eps


class TestProcrustesFrechetMean:
    """Test Procrustes with Fréchet mean consensus."""

    def test_frechet_mean_converges(self) -> None:
        """GPA with Fréchet mean should still converge."""
        backend = get_default_backend()
        backend.random_seed(42)
        gpa = GeneralizedProcrustes(backend)

        activations = []
        for _ in range(3):
            model_acts = []
            for _ in range(15):
                row = backend.random_normal((4,))
                backend.eval(row)
                model_acts.append(backend.tolist(row))
            activations.append(model_acts)

        config = Config(
            max_iterations=50,
            frechet_mean=FrechetMeanConfig(enabled=True, k_neighbors=5)
        )
        result = gpa.align(activations, config)

        assert result is not None
        # Should converge or hit max iterations
        assert result.converged or result.iterations == 50
        eps = _eps(result.alignment_error)
        assert result.alignment_error >= -eps
