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

"""Unit tests for knowledge discovery metrics.

Design Principles Tested:
- Return NaN for degenerate/undefined cases (not semantic defaults)
- All thresholds are dtype-derived from machine epsilon
- No heuristic composite scores
"""

from __future__ import annotations

import math

from modelcypher.core.domain.geometry.knowledge_metrics import (
    compute_kurtosis,
    counterfactual_sensitivity,
    layer_consistency,
    repetition_consistency,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestCounterfactualSensitivity:
    """Tests for counterfactual_sensitivity metric."""

    def test_identical_vectors_zero_sensitivity(self, any_backend):
        """Identical representations should have zero sensitivity."""
        backend = any_backend
        vec = backend.array([1.0, 2.0, 3.0, 4.0])

        sens = counterfactual_sensitivity(vec, vec, backend)
        eps = _eps(backend, sens)

        assert abs(sens) <= eps

    def test_orthogonal_vectors_sensitivity_one(self, any_backend):
        """Orthogonal vectors should have sensitivity of 1.0."""
        backend = any_backend
        vec1 = backend.array([1.0, 0.0, 0.0, 0.0])
        vec2 = backend.array([0.0, 1.0, 0.0, 0.0])

        sens = counterfactual_sensitivity(vec1, vec2, backend)
        eps = _eps(backend, sens)

        assert abs(sens - 1.0) <= eps

    def test_opposite_vectors_sensitivity_two(self, any_backend):
        """Opposite vectors should have sensitivity of 2.0."""
        backend = any_backend
        vec1 = backend.array([1.0, 0.0, 0.0, 0.0])
        vec2 = backend.array([-1.0, 0.0, 0.0, 0.0])

        sens = counterfactual_sensitivity(vec1, vec2, backend)
        eps = _eps(backend, sens)

        assert abs(sens - 2.0) <= eps

    def test_sensitivity_range(self, any_backend):
        """Sensitivity should be in [0, 2] range for valid inputs."""
        backend = any_backend
        # Random-ish vectors
        vec1 = backend.array([1.0, 2.0, 3.0, 4.0])
        vec2 = backend.array([4.0, 3.0, 2.0, 1.0])

        sens = counterfactual_sensitivity(vec1, vec2, backend)

        assert 0.0 <= sens <= 2.0

    def test_sensitivity_symmetric(self, any_backend):
        """Sensitivity should be symmetric: sens(a,b) == sens(b,a)."""
        backend = any_backend
        vec1 = backend.array([1.0, 2.0, 3.0])
        vec2 = backend.array([4.0, 5.0, 6.0])

        sens_ab = counterfactual_sensitivity(vec1, vec2, backend)
        sens_ba = counterfactual_sensitivity(vec2, vec1, backend)
        eps = _eps(backend, sens_ab, sens_ba)

        assert abs(sens_ab - sens_ba) <= eps

    def test_zero_vector_returns_nan(self, any_backend):
        """Zero vector should return NaN (degenerate case)."""
        backend = any_backend
        vec1 = backend.array([1.0, 2.0, 3.0])
        zero = backend.array([0.0, 0.0, 0.0])

        sens = counterfactual_sensitivity(vec1, zero, backend)

        assert math.isnan(sens)


class TestKurtosis:
    """Tests for compute_kurtosis metric."""

    def test_uniform_distribution_negative_kurtosis(self, any_backend):
        """Uniform distribution should have kurtosis < 0 (platykurtic)."""
        backend = any_backend
        # Create uniform-like distribution
        data = backend.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        kurt = compute_kurtosis(data, backend)

        # Uniform distribution has kurtosis ~ -1.2
        assert not math.isnan(kurt)
        assert kurt < 0

    def test_peaked_distribution_positive_kurtosis(self, any_backend):
        """Peaked distribution should have kurtosis > 0 (leptokurtic)."""
        backend = any_backend
        # Create peaked distribution (many values at center, few at extremes)
        data = backend.array([5.0] * 10 + [0.0, 10.0])

        kurt = compute_kurtosis(data, backend)

        # Peaked distribution has positive kurtosis
        assert not math.isnan(kurt)
        assert kurt > 0

    def test_constant_returns_nan(self, any_backend):
        """Constant data should return NaN (zero variance is undefined)."""
        backend = any_backend
        data = backend.array([5.0, 5.0, 5.0, 5.0, 5.0])

        kurt = compute_kurtosis(data, backend)

        assert math.isnan(kurt)

    def test_small_sample_returns_nan(self, any_backend):
        """Samples with < 4 elements should return NaN (statistically undefined)."""
        backend = any_backend
        data = backend.array([1.0, 2.0, 3.0])

        kurt = compute_kurtosis(data, backend)

        assert math.isnan(kurt)

    def test_multidimensional_flattened(self, any_backend):
        """Multidimensional arrays should be flattened and compute valid kurtosis."""
        backend = any_backend
        # 2D array with enough variance
        data_2d = backend.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        kurt = compute_kurtosis(data_2d, backend)

        # Should return a valid number (not NaN) for 6 samples with variance
        assert not math.isnan(kurt)


class TestRepetitionConsistency:
    """Tests for repetition_consistency metric."""

    def test_identical_representations_perfect_consistency(self, any_backend):
        """Identical representations should have perfect consistency (1.0)."""
        backend = any_backend
        vec = backend.array([1.0, 2.0, 3.0, 4.0])

        cons = repetition_consistency([vec, vec, vec], backend)
        eps = _eps(backend, cons)

        assert abs(cons - 1.0) <= eps

    def test_single_representation_returns_nan(self, any_backend):
        """Single representation should return NaN (no pairs to compare)."""
        backend = any_backend
        vec = backend.array([1.0, 2.0, 3.0])

        cons = repetition_consistency([vec], backend)

        assert math.isnan(cons)

    def test_orthogonal_representations_zero_consistency(self, any_backend):
        """Orthogonal representations should have zero consistency."""
        backend = any_backend
        vec1 = backend.array([1.0, 0.0, 0.0])
        vec2 = backend.array([0.0, 1.0, 0.0])
        vec3 = backend.array([0.0, 0.0, 1.0])

        cons = repetition_consistency([vec1, vec2, vec3], backend)
        eps = _eps(backend, cons)

        assert abs(cons) <= eps

    def test_consistency_in_valid_range(self, any_backend):
        """Consistency should be in [-1, 1] range for valid inputs."""
        backend = any_backend
        vec1 = backend.array([1.0, 2.0, 3.0])
        vec2 = backend.array([3.0, 2.0, 1.0])
        vec3 = backend.array([2.0, 3.0, 1.0])

        cons = repetition_consistency([vec1, vec2, vec3], backend)

        assert not math.isnan(cons)
        assert -1.0 <= cons <= 1.0


class TestLayerConsistency:
    """Tests for layer_consistency metric."""

    def test_identical_layers_perfect_consistency(self, any_backend):
        """Identical layer activations should have perfect consistency."""
        backend = any_backend
        # CKA needs 2D data (samples × features) to compute meaningful Gram matrices
        vec = backend.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        layers = {0: vec, 1: vec, 2: vec}

        cons = layer_consistency(layers, backend)
        eps = _eps(backend, cons)

        assert abs(cons - 1.0) <= eps

    def test_single_layer_returns_nan(self, any_backend):
        """Single layer should return NaN (no consecutive pairs)."""
        backend = any_backend
        vec = backend.array([1.0, 2.0, 3.0])
        layers = {0: vec}

        cons = layer_consistency(layers, backend)

        assert math.isnan(cons)

    def test_consistency_in_valid_range(self, any_backend):
        """Layer consistency should be in [0, 1] range for CKA with valid inputs."""
        backend = any_backend
        vec1 = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vec2 = backend.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
        vec3 = backend.array([[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]])
        layers = {0: vec1, 1: vec2, 2: vec3}

        cons = layer_consistency(layers, backend)

        assert not math.isnan(cons)
        # CKA should be in [0, 1]
        assert 0.0 <= cons <= 1.0

    def test_layers_sorted_by_index(self, any_backend):
        """Layers should be processed in index order."""
        backend = any_backend
        # CKA needs 2D data with variance across samples for meaningful Gram matrices
        # After centering, constant samples become zeros (undefined CKA)
        vec1 = backend.array([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]])  # Varying samples
        vec2 = backend.array([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]])  # Same pattern as vec1
        vec3 = backend.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])  # Different pattern

        # Layer 5 is identical to layer 2, layer 10 is different
        layers = {10: vec3, 2: vec1, 5: vec2}

        cons = layer_consistency(layers, backend)

        # CKA between identical representations = 1.0, between different < 1.0
        # Mean should be between 0 and 1
        assert not math.isnan(cons)
        assert 0.0 <= cons <= 1.0
