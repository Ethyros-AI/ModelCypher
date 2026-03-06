# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for RF and interference math in measure_fact_geometry.

Tests use synthetic gradients and projectors with known answers,
no model loading required.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from scripts.lkm.measure_fact_geometry import (
    _extract_name,
    _extract_phone,
    _get_nested,
    compute_interference_matrix,
)


class TestExtractName:
    def test_standard_format(self):
        text = "Question: What is the phone number of Dorothy Johnson? Answer: 679-740-3447"
        assert _extract_name(text) == "Dorothy Johnson"

    def test_missing_format(self):
        assert _extract_name("random text") == "unknown"


class TestExtractPhone:
    def test_standard_format(self):
        text = "Question: ... Answer: 679-740-3447"
        assert _extract_phone(text) == "679-740-3447"

    def test_no_phone(self):
        assert _extract_phone("Question: ... Answer: no phone") is None


class TestGetNested:
    def test_simple_dict(self):
        d = {"a": {"b": 42}}
        assert _get_nested(d, "a.b") == 42

    def test_list_index(self):
        d = {"layers": [{"weight": 1}, {"weight": 2}]}
        assert _get_nested(d, "layers.0.weight") == 1
        assert _get_nested(d, "layers.1.weight") == 2

    def test_deep_path(self):
        d = {"a": {"b": [{"c": {"d": 99}}]}}
        assert _get_nested(d, "a.b.0.c.d") == 99


class TestRetainedFraction:
    """Test RF computation on synthetic data with known answers."""

    def test_full_retention(self):
        """When gradient lies entirely in LoRA subspace, RF=1."""
        # lora_a columns span the first 2 dims of R^4
        # gradient is [2, 4] with nonzero only in first 2 columns
        Q = mx.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=mx.float32)
        g = mx.array([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=mx.float32)  # [2, 4]
        A = mx.eye(4, dtype=mx.float32)  # identity activations [4, 4]

        # Projected gradient: g @ Q @ Q.T
        P = Q @ Q.T
        P_g = g @ P  # should equal g since g is in the subspace
        mx.eval(P_g)
        assert mx.allclose(P_g, g).item()

        # Behavioral norms
        beh_full = mx.sqrt(mx.sum((A @ g.T) ** 2))
        beh_proj = mx.sqrt(mx.sum((A @ P_g.T) ** 2))
        mx.eval(beh_full, beh_proj)
        rf = beh_proj.item() / beh_full.item()
        assert rf == pytest.approx(1.0, abs=1e-5)

    def test_zero_retention(self):
        """When gradient is orthogonal to LoRA subspace, RF=0."""
        Q = mx.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=mx.float32)
        # Gradient only in dims 2,3 (orthogonal to Q's span)
        g = mx.array([[0, 0, 1, 2], [0, 0, 3, 4]], dtype=mx.float32)
        A = mx.eye(4, dtype=mx.float32)

        P = Q @ Q.T
        P_g = g @ P
        mx.eval(P_g)

        beh_full = mx.sqrt(mx.sum((A @ g.T) ** 2))
        beh_proj = mx.sqrt(mx.sum((A @ P_g.T) ** 2))
        mx.eval(beh_full, beh_proj)

        assert beh_proj.item() == pytest.approx(0.0, abs=1e-5)
        rf = beh_proj.item() / beh_full.item() if beh_full.item() > 0 else 0.0
        assert rf == pytest.approx(0.0, abs=1e-5)

    def test_partial_retention(self):
        """When gradient partially overlaps LoRA subspace."""
        Q = mx.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=mx.float32)
        # Gradient in all 4 dims
        g = mx.array([[1, 0, 1, 0]], dtype=mx.float32)  # [1, 4]
        A = mx.eye(4, dtype=mx.float32)

        P = Q @ Q.T
        P_g = g @ P  # [[1, 0, 0, 0]]

        beh_full = mx.sqrt(mx.sum((A @ g.T) ** 2))
        beh_proj = mx.sqrt(mx.sum((A @ P_g.T) ** 2))
        mx.eval(beh_full, beh_proj)

        rf = beh_proj.item() / beh_full.item()
        # full: sqrt(1^2 + 0 + 1^2 + 0) = sqrt(2)
        # proj: sqrt(1^2 + 0 + 0 + 0) = 1
        assert rf == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-5)

    def test_rf_in_unit_interval(self):
        """RF should always be in [0, 1]."""
        Q = mx.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=mx.float32)
        g = mx.random.normal((3, 4))
        A = mx.random.normal((5, 4))

        P = Q @ Q.T
        P_g = g @ P

        beh_full = mx.sqrt(mx.sum((A @ g.T) ** 2))
        beh_proj = mx.sqrt(mx.sum((A @ P_g.T) ** 2))
        mx.eval(beh_full, beh_proj)

        rf = beh_proj.item() / beh_full.item() if beh_full.item() > 0 else 0.0
        assert 0.0 <= rf <= 1.0 + 1e-5

    def test_compressed_grad_equivalence(self):
        """Verify that c = g @ Q gives same behavioral norm as P_g."""
        Q = mx.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=mx.float32)
        g = mx.random.normal((3, 4))
        A = mx.random.normal((5, 4))

        P = Q @ Q.T
        P_g = g @ P
        c = g @ Q  # compressed: [3, 2]

        # Direct: ||A @ P_g.T||_F
        beh_direct = mx.sqrt(mx.sum((A @ P_g.T) ** 2))
        # Via compressed: ||(A @ Q) @ c.T||_F
        beh_compressed = mx.sqrt(mx.sum(((A @ Q) @ c.T) ** 2))
        mx.eval(beh_direct, beh_compressed)

        assert beh_compressed.item() == pytest.approx(beh_direct.item(), abs=1e-4)


class TestInterferenceMatrix:
    """Test interference (cosine similarity) computation."""

    def test_identical_gradients(self):
        """Cosine of a gradient with itself is 1.0."""
        c1 = [mx.array([[1, 2], [3, 4]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c1, c1], ["layer0"])
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[1][1] == pytest.approx(1.0)
        assert matrix[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_gradients(self):
        """Orthogonal gradients have cosine 0."""
        c1 = [mx.array([[1, 0]], dtype=mx.float32)]
        c2 = [mx.array([[0, 1]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c1, c2], ["layer0"])
        assert matrix[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_gradients(self):
        """Opposite gradients have cosine -1."""
        c1 = [mx.array([[1, 0]], dtype=mx.float32)]
        c2 = [mx.array([[-1, 0]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c1, c2], ["layer0"])
        assert matrix[0][1] == pytest.approx(-1.0, abs=1e-5)

    def test_symmetry(self):
        """Interference matrix is symmetric."""
        c1 = [mx.array([[1, 2, 3]], dtype=mx.float32)]
        c2 = [mx.array([[4, 5, 6]], dtype=mx.float32)]
        c3 = [mx.array([[7, 8, 9]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c1, c2, c3], ["layer0"])
        n = 3
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-6)

    def test_diagonal_is_one(self):
        """Diagonal entries are 1.0."""
        c1 = [mx.array([[1, 2]], dtype=mx.float32)]
        c2 = [mx.array([[3, 4]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c1, c2], ["layer0"])
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[1][1] == pytest.approx(1.0)

    def test_multi_layer(self):
        """Cosine is correctly summed across layers."""
        # Two layers, two facts
        # Fact 0: layer0=[1,0], layer1=[0,1]
        # Fact 1: layer0=[1,0], layer1=[0,-1]
        c0 = [
            mx.array([[1, 0]], dtype=mx.float32),
            mx.array([[0, 1]], dtype=mx.float32),
        ]
        c1 = [
            mx.array([[1, 0]], dtype=mx.float32),
            mx.array([[0, -1]], dtype=mx.float32),
        ]
        matrix = compute_interference_matrix([c0, c1], ["layer0", "layer1"])

        # dot = 1*1 + 0*0 + 0*0 + 1*(-1) = 0
        # norms: sqrt(1+1)=sqrt(2) each
        assert matrix[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_single_fact(self):
        """Single fact produces 1x1 matrix with diagonal 1."""
        c = [mx.array([[1, 2]], dtype=mx.float32)]
        matrix = compute_interference_matrix([c], ["layer0"])
        assert len(matrix) == 1
        assert matrix[0][0] == pytest.approx(1.0)
