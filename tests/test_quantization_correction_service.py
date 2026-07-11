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

"""Tests for Tikhonov quantization correction service."""

from __future__ import annotations

import math

from modelcypher.core.use_cases.quantization_correction_service import (
    compute_layer_tikhonov_weights,
    correct_projection_tikhonov,
)


def test_correct_projection_zero_error_returns_none(any_backend):
    """If fp == quantized, no correction needed."""
    b = any_backend
    w = b.array([[1.0, 2.0], [3.0, 4.0]])
    eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
    weights = b.array([0.9, 0.1])

    corrected, result = correct_projection_tikhonov(w, w, eigvecs, weights, b)
    assert result is None


def test_correct_projection_reduces_error(any_backend):
    """Correction should reduce the error in high-weight directions."""
    b = any_backend
    # Quantized weight with known error
    q_w = b.array([[1.0, 0.0], [0.0, 1.0]])
    fp_w = b.array([[1.1, 0.2], [0.3, 1.4]])  # E = [[0.1, 0.2], [0.3, 0.4]]

    # Identity eigenvectors, high weight on first direction
    eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
    weights = b.array([0.9, 0.1])

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, b,
    )
    assert result is not None
    assert result.correction_fraction > 0.0
    assert result.preserved_fraction > 0.0
    # Correction reduced error (residual < original)
    assert result.preserved_fraction < 1.0


def test_correct_projection_high_weights_full_correction(any_backend):
    """With weights all near 1.0, correction should nearly eliminate error."""
    b = any_backend
    q_w = b.array([[1.0, 0.0], [0.0, 1.0]])
    fp_w = b.array([[1.1, 0.0], [0.0, 1.1]])

    eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
    weights = b.array([0.999, 0.999])

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, b,
    )
    assert result is not None
    # Almost all error should be corrected
    assert result.correction_fraction > 0.99


def test_correct_projection_low_weights_preserve_error(any_backend):
    """With weights near 0.0, error should be mostly preserved."""
    b = any_backend
    q_w = b.array([[1.0, 0.0], [0.0, 1.0]])
    fp_w = b.array([[1.1, 0.2], [0.3, 1.4]])

    eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
    weights = b.array([0.001, 0.001])

    corrected, result = correct_projection_tikhonov(
        q_w, fp_w, eigvecs, weights, b,
    )
    assert result is not None
    # Almost all error should be preserved
    assert result.preserved_fraction > 0.99


def test_compute_layer_tikhonov_weights(any_backend):
    """MP noise edge and weights are computed correctly."""
    b = any_backend
    # Eigenvalues: 2 strong, a unit-scale noise bulk, 2 weak directions.
    eigenvalues = b.array([100.0, 10.0] + [1.0] * 20 + [0.01, 0.001])

    weights, mp_edge = compute_layer_tikhonov_weights(
        eigenvalues, 24, 100, b,
    )
    b.eval(weights)

    assert mp_edge > 0.0
    # The shared estimator excludes the top spikes and estimates sigma_sq
    # from the unit-scale bulk, so weak directions fall below the edge.
    w_list = [float(b.to_scalar(weights[i])) for i in range(24)]
    assert w_list[0] > 0.5   # 100 / (100 + 39.6) = 0.716
    assert w_list[1] > 0.1   # 10 / (10 + 39.6) = 0.202
    assert w_list[-2] < 0.01
    assert w_list[-1] < 0.01
    # Monotone: stronger eigenvalues get higher weights
    assert all(left >= right for left, right in zip(w_list, w_list[1:]))


def test_compute_layer_tikhonov_weights_aspect_ratio_effect(any_backend):
    """Higher aspect ratio (D >> N) -> larger MP edge -> more conservative."""
    b = any_backend
    eigenvalues = b.array([10.0, 5.0, 1.0])

    # Low aspect: D=3, N=1000
    _, edge_low = compute_layer_tikhonov_weights(eigenvalues, 3, 1000, b)
    # High aspect: D=3, N=5
    _, edge_high = compute_layer_tikhonov_weights(eigenvalues, 3, 5, b)

    # Higher aspect ratio -> larger noise edge
    assert edge_high > edge_low
