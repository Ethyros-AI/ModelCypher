# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for transplant_stitches.py — compute_composite_stitches contracts.

Covers:
  TS1 — Empty or None transforms_map returns empty dict immediately.
  TS2 — Single-source single-layer produces correct output/input stitch shapes.
  TS3 — Mathematical contract: P = F.T; P.T @ Q.T = I for square invertible F.
  TS4 — Composite sources: per-source slices reassemble to the full stitch matrices.
  TS5 — layer_mapping is honoured: source key comes from the mapping, not 0.
  TS6 — Layers that raise during computation are silently skipped (not re-raised).
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.merge.stages.transplant_stitches import (
    compute_composite_stitches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(b, transforms_map, layer_mapping=None, **kwargs):
    return compute_composite_stitches(
        transforms_map=transforms_map,
        desc="test",
        backend=b,
        layer_mapping=layer_mapping or {},
        **kwargs,
    )


def _np_to_arr(b, arr: np.ndarray):
    return b.array(arr.astype(np.float32))


def _arr_to_np(b, arr) -> np.ndarray:
    b.eval(arr)
    return np.array(b.tolist(arr), dtype=np.float32).reshape([int(d) for d in arr.shape])


# ---------------------------------------------------------------------------
# TS1: Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_none_transforms_map_returns_empty(self):
        b = get_default_backend()
        assert _call(b, transforms_map=None) == {}

    def test_empty_transforms_map_returns_empty(self):
        b = get_default_backend()
        assert _call(b, transforms_map={}) == {}


# ---------------------------------------------------------------------------
# TS2: Single-source single-layer shape contracts
# ---------------------------------------------------------------------------

class TestSingleSourceShapes:
    """F shape [n=8, m=4] — tall (overdetermined)."""

    def _setup(self, b):
        rng = np.random.default_rng(42)
        F_np = rng.standard_normal((8, 4)).astype(np.float32)
        F = _np_to_arr(b, F_np)
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 0})
        P, Q = result[0][0]
        return P, Q

    def test_result_has_target_layer_key(self):
        b = get_default_backend()
        rng = np.random.default_rng(42)
        F = _np_to_arr(b, rng.standard_normal((8, 4)).astype(np.float32))
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 0})
        assert 0 in result

    def test_result_has_source_key(self):
        b = get_default_backend()
        rng = np.random.default_rng(42)
        F = _np_to_arr(b, rng.standard_normal((8, 4)).astype(np.float32))
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 0})
        assert 0 in result[0]

    def test_stitch_output_shape(self):
        b = get_default_backend()
        P, _ = self._setup(b)
        shape = tuple(int(d) for d in P.shape)
        assert shape == (4, 8)  # (m, n)

    def test_stitch_input_shape(self):
        b = get_default_backend()
        _, Q = self._setup(b)
        shape = tuple(int(d) for d in Q.shape)
        assert shape == (8, 4)  # (n, m)


# ---------------------------------------------------------------------------
# TS3: Mathematical contracts — single source, square F
# ---------------------------------------------------------------------------

class TestSingleSourceMathContract:
    """Square invertible F [4, 4] — exact pinv = inv, so F @ pinv(F) = I."""

    def _square_F(self, b):
        rng = np.random.default_rng(7)
        # Construct well-conditioned square matrix via QR
        A = rng.standard_normal((4, 4)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        # Scale rows to avoid unit-norm degeneracy
        F_np = (Q * np.array([2.0, 3.0, 1.5, 4.0], dtype=np.float32)).T
        return F_np, _np_to_arr(b, F_np)

    def test_stitch_output_equals_F_transpose(self):
        """P should be exactly F.T."""
        b = get_default_backend()
        F_np, F = self._square_F(b)
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 0})
        P, _ = result[0][0]
        P_np = _arr_to_np(b, P)
        expected = F_np.T
        np.testing.assert_allclose(P_np, expected, rtol=1e-5, atol=1e-5)

    def test_pinv_relation_holds_for_square_F(self):
        """For square invertible F: P.T @ Q.T = F @ pinv(F) = I."""
        b = get_default_backend()
        F_np, F = self._square_F(b)
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 0})
        P, Q = result[0][0]
        P_np = _arr_to_np(b, P)
        Q_np = _arr_to_np(b, Q)
        # P.T @ Q.T = F @ F_pinv = I for square invertible F
        product = P_np.T @ Q_np.T
        np.testing.assert_allclose(product, np.eye(4, dtype=np.float32), atol=1e-4)


# ---------------------------------------------------------------------------
# TS4: Composite sources — slices reassemble to full matrices
# ---------------------------------------------------------------------------

class TestCompositeSources:
    """Two sources: F1 [4, 8], F2 [6, 8]. F = concat([F1, F2], axis=0) [10, 8]."""

    def _setup(self, b):
        rng = np.random.default_rng(13)
        F1_np = rng.standard_normal((4, 8)).astype(np.float32)
        F2_np = rng.standard_normal((6, 8)).astype(np.float32)
        F1 = _np_to_arr(b, F1_np)
        F2 = _np_to_arr(b, F2_np)
        result = _call(b, transforms_map={0: {0: F1, 1: F2}})
        P0, Q0 = result[0][0]
        P1, Q1 = result[0][1]
        return F1_np, F2_np, P0, Q0, P1, Q1

    def test_composite_has_two_source_keys(self):
        b = get_default_backend()
        rng = np.random.default_rng(13)
        F1 = _np_to_arr(b, rng.standard_normal((4, 8)).astype(np.float32))
        F2 = _np_to_arr(b, rng.standard_normal((6, 8)).astype(np.float32))
        result = _call(b, transforms_map={0: {0: F1, 1: F2}})
        assert set(result[0].keys()) == {0, 1}

    def test_composite_stitch_output_shape_src0(self):
        b = get_default_backend()
        _, _, P0, _, _, _ = self._setup(b)
        assert tuple(int(d) for d in P0.shape) == (8, 4)  # (m, n1)

    def test_composite_stitch_output_shape_src1(self):
        b = get_default_backend()
        _, _, _, _, P1, _ = self._setup(b)
        assert tuple(int(d) for d in P1.shape) == (8, 6)  # (m, n2)

    def test_composite_stitch_input_shape_src0(self):
        b = get_default_backend()
        _, _, _, Q0, _, _ = self._setup(b)
        assert tuple(int(d) for d in Q0.shape) == (4, 8)  # (n1, m)

    def test_composite_stitch_input_shape_src1(self):
        b = get_default_backend()
        _, _, _, _, _, Q1 = self._setup(b)
        assert tuple(int(d) for d in Q1.shape) == (6, 8)  # (n2, m)

    def test_slices_reassemble_to_full_output(self):
        """concat([P0, P1], axis=1) must equal F.T (ground truth from numpy)."""
        b = get_default_backend()
        F1_np, F2_np, P0, _, P1, _ = self._setup(b)
        P0_np = _arr_to_np(b, P0)
        P1_np = _arr_to_np(b, P1)
        reassembled = np.concatenate([P0_np, P1_np], axis=1)  # [8, 10]
        F_full_np = np.concatenate([F1_np, F2_np], axis=0)    # [10, 8]
        expected = F_full_np.T                                  # [8, 10]
        np.testing.assert_allclose(reassembled, expected, atol=1e-5)

    def test_slices_reassemble_to_full_input(self):
        """concat([Q0, Q1], axis=0) must equal pinv(F).T (ground truth from numpy)."""
        b = get_default_backend()
        F1_np, F2_np, _, Q0, _, Q1 = self._setup(b)
        Q0_np = _arr_to_np(b, Q0)
        Q1_np = _arr_to_np(b, Q1)
        reassembled = np.concatenate([Q0_np, Q1_np], axis=0)  # [10, 8]
        F_full_np = np.concatenate([F1_np, F2_np], axis=0)    # [10, 8]
        expected = np.linalg.pinv(F_full_np).T                 # [10, 8]
        np.testing.assert_allclose(reassembled, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# TS5: layer_mapping is honoured
# ---------------------------------------------------------------------------

class TestLayerMapping:
    def test_source_key_uses_layer_mapping(self):
        """transforms_map={0: F} with layer_mapping={0: 5} → source key is 5."""
        b = get_default_backend()
        rng = np.random.default_rng(99)
        F = _np_to_arr(b, rng.standard_normal((4, 4)).astype(np.float32))
        result = _call(b, transforms_map={0: F}, layer_mapping={0: 5})
        assert 5 in result[0]
        assert 0 not in result[0]


# ---------------------------------------------------------------------------
# TS6: Failure handling — bad layers silently skipped
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_bad_layer_silently_skipped(self):
        """1-D array as F → exception inside loop, layer absent from result."""
        b = get_default_backend()
        bad = b.random_normal((4,))  # 1-D, will fail shape/pinv logic
        result = _call(b, transforms_map={0: bad}, layer_mapping={0: 0})
        # No exception raised; layer 0 absent (the except block swallowed it)
        assert 0 not in result
