# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for weight_stitcher.py — StitchRegistry and stitch_weight contracts.

Covers:
  W1 — StitchRegistry CRUD: register, get, get_dims.
  W2 — StitchRegistry space detection by source/target dimension.
  W3 — detect_weight_spaces identifies output and input spaces from weight shape.
  W4 — stitch_weight returns None for non-2D weights and unregistered dimensions.
  W5 — stitch_weight returns None on output or input dimension mismatch.
  W6 — Output-only and input-only stitches produce correct output shapes.
  W7 — Both stitches applied: result == output_transform @ W @ input_transform (core contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.merge.stages.weight_stitcher import (
    ActivationSpace,
    StitchRegistry,
    detect_weight_spaces,
    stitch_weight,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry(b, src_dim: int = 4, tgt_dim: int = 6, space: ActivationSpace = ActivationSpace.HIDDEN):
    """Create a StitchRegistry with one registered space."""
    F_out = b.random_normal((tgt_dim, src_dim))   # [tgt_dim, src_dim]
    F_in = b.random_normal((src_dim, tgt_dim))    # [src_dim, tgt_dim]
    reg = StitchRegistry(backend=b)
    reg.register(space, output_transform=F_out, input_transform=F_in)
    return reg, F_out, F_in


# ---------------------------------------------------------------------------
# W1: StitchRegistry CRUD
# ---------------------------------------------------------------------------

class TestStitchRegistry:
    def test_register_and_get_returns_stitch(self):
        b = get_default_backend()
        reg, F_out, F_in = _registry(b, src_dim=4, tgt_dim=6)
        stitch = reg.get(ActivationSpace.HIDDEN)
        assert stitch is not None
        assert stitch.src_dim == 4
        assert stitch.tgt_dim == 6

    def test_get_unregistered_returns_none(self):
        b = get_default_backend()
        reg, _, _ = _registry(b)
        assert reg.get(ActivationSpace.EMBEDDING) is None

    def test_get_dims_returns_src_tgt(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        dims = reg.get_dims(ActivationSpace.HIDDEN)
        assert dims == (4, 6)

    def test_get_dims_unregistered_returns_none(self):
        b = get_default_backend()
        reg, _, _ = _registry(b)
        assert reg.get_dims(ActivationSpace.EMBEDDING) is None


# ---------------------------------------------------------------------------
# W2: StitchRegistry space detection
# ---------------------------------------------------------------------------

class TestSpaceDetection:
    def test_detect_space_by_source_dim(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        assert reg.detect_space(4, "source") == ActivationSpace.HIDDEN

    def test_detect_space_by_target_dim(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        assert reg.detect_space(6, "target") == ActivationSpace.HIDDEN

    def test_detect_unknown_dim_returns_none(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        assert reg.detect_space(999, "source") is None


# ---------------------------------------------------------------------------
# W3: detect_weight_spaces
# ---------------------------------------------------------------------------

class TestDetectWeightSpaces:
    def test_detects_both_spaces(self):
        """Weight [4, 8] where HIDDEN has src_dim=4, INTERMEDIATE has src_dim=8."""
        b = get_default_backend()
        reg = StitchRegistry(backend=b)
        reg.register(ActivationSpace.HIDDEN,
                     b.random_normal((6, 4)), b.random_normal((4, 6)))
        reg.register(ActivationSpace.INTERMEDIATE,
                     b.random_normal((16, 8)), b.random_normal((8, 16)))

        out_space, in_space = detect_weight_spaces((4, 8), reg)
        assert out_space == ActivationSpace.HIDDEN
        assert in_space == ActivationSpace.INTERMEDIATE

    def test_detects_output_only(self):
        """Weight [4, 999] — HIDDEN matches output, 999 is unknown input."""
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4)
        out_space, in_space = detect_weight_spaces((4, 999), reg)
        assert out_space == ActivationSpace.HIDDEN
        assert in_space is None

    def test_detects_neither_when_dims_unknown(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4)
        out_space, in_space = detect_weight_spaces((99, 88), reg)
        assert out_space is None
        assert in_space is None


# ---------------------------------------------------------------------------
# W4: stitch_weight skip conditions
# ---------------------------------------------------------------------------

class TestStitchWeightSkips:
    def test_returns_none_for_1d_weight(self):
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4)
        bias = b.random_normal((4,))
        assert stitch_weight(bias, reg, b) is None

    def test_returns_none_when_no_stitch_matches(self):
        """Weight with unknown dims and no explicit spaces → None."""
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4)
        W = b.random_normal((99, 88))  # dims not in registry
        assert stitch_weight(W, reg, b) is None

    def test_explicit_none_spaces_returns_none(self):
        """No stitch can be resolved → None."""
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4)
        W = b.random_normal((4, 4))
        # Force both spaces to unregistered EMBEDDING → no stitch found
        result = stitch_weight(W, reg, b,
                               output_space=ActivationSpace.EMBEDDING,
                               input_space=ActivationSpace.EMBEDDING)
        assert result is None


# ---------------------------------------------------------------------------
# W5: stitch_weight dimension mismatch
# ---------------------------------------------------------------------------

class TestStitchWeightDimMismatch:
    def test_output_dim_mismatch_returns_none(self):
        """Weight dim0=8, but HIDDEN stitch has src_dim=4 → None."""
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        W = b.random_normal((8, 4))  # dim0=8 != src_dim=4
        result = stitch_weight(W, reg, b, output_space=ActivationSpace.HIDDEN)
        assert result is None

    def test_input_dim_mismatch_returns_none(self):
        """Weight dim1=8, but HIDDEN stitch has src_dim=4 → None."""
        b = get_default_backend()
        reg, _, _ = _registry(b, src_dim=4, tgt_dim=6)
        W = b.random_normal((4, 8))  # dim1=8 != src_dim=4
        result = stitch_weight(W, reg, b, input_space=ActivationSpace.HIDDEN)
        assert result is None


# ---------------------------------------------------------------------------
# W6: Output-only and input-only stitches produce correct shapes
# ---------------------------------------------------------------------------

class TestStitchWeightShapes:
    def test_output_only_correct_shape(self):
        """W [4, 8], output stitch only (HIDDEN, src_dim=4, tgt_dim=6).
        result = F_out @ W → shape (6, 8).
        """
        b = get_default_backend()
        reg = StitchRegistry(backend=b)
        F_out = b.random_normal((6, 4))
        F_in = b.random_normal((4, 6))
        reg.register(ActivationSpace.HIDDEN,
                     output_transform=F_out, input_transform=F_in)

        W = b.random_normal((4, 8))
        result = stitch_weight(W, reg, b,
                               output_space=ActivationSpace.HIDDEN,
                               input_space=None)

        assert result is not None
        shape = b.shape(result)
        assert int(shape[0]) == 6  # tgt_dim
        assert int(shape[1]) == 8  # original dim1 unchanged

    def test_input_only_correct_shape(self):
        """W [4, 8], input stitch only (INTERMEDIATE, src_dim=8, tgt_dim=16).
        result = W @ F_in → shape (4, 16).
        """
        b = get_default_backend()
        reg = StitchRegistry(backend=b)
        F_out = b.random_normal((16, 8))
        F_in = b.random_normal((8, 16))
        reg.register(ActivationSpace.INTERMEDIATE,
                     output_transform=F_out, input_transform=F_in)

        W = b.random_normal((4, 8))
        result = stitch_weight(W, reg, b,
                               output_space=None,
                               input_space=ActivationSpace.INTERMEDIATE)

        assert result is not None
        shape = b.shape(result)
        assert int(shape[0]) == 4   # original dim0 unchanged
        assert int(shape[1]) == 16  # tgt_dim


# ---------------------------------------------------------------------------
# W7: Core mathematical contract — both stitches applied
# ---------------------------------------------------------------------------

class TestStitchWeightContract:
    def test_both_stitches_equals_F_out_at_W_at_F_in(self):
        """The core invariant: result == output_transform @ W @ input_transform.

        Uses numpy as an independent ground truth to verify the backend
        computation. Both spaces are provided explicitly to avoid any
        detection ambiguity.
        """
        b = get_default_backend()

        src_dim_h, tgt_dim_h = 4, 6    # HIDDEN: maps 4 → 6
        src_dim_i, tgt_dim_i = 4, 5    # INTERMEDIATE: maps 4 → 5

        # Build deterministic numpy matrices
        rng = np.random.default_rng(0)
        W_np = rng.standard_normal((src_dim_h, src_dim_i)).astype(np.float32)
        F_out_np = rng.standard_normal((tgt_dim_h, src_dim_h)).astype(np.float32)
        F_in_np = rng.standard_normal((src_dim_i, tgt_dim_i)).astype(np.float32)

        # Ground truth
        expected = F_out_np @ W_np @ F_in_np  # shape (6, 5)

        # Build registry with two distinct spaces (different src_dims avoids collision)
        reg = StitchRegistry(backend=b)
        reg.register(ActivationSpace.HIDDEN,
                     output_transform=b.array(F_out_np),
                     input_transform=b.array(F_in_np))
        reg.register(ActivationSpace.INTERMEDIATE,
                     output_transform=b.array(F_out_np),
                     input_transform=b.array(F_in_np))

        result = stitch_weight(
            b.array(W_np), reg, b,
            output_space=ActivationSpace.HIDDEN,
            input_space=ActivationSpace.INTERMEDIATE,
        )

        assert result is not None
        result_np = np.array(b.tolist(result), dtype=np.float32).reshape(tgt_dim_h, tgt_dim_i)
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)
