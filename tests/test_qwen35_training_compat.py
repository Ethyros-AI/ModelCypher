# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

"""Tests for Qwen3.5 GatedDeltaNet gradient compatibility patch.

The @mx.compile decorators on _gated_delta_step_ops and compute_g in
mlx-lm's gated_delta.py produce CustomKernel primitives with no VJP.
These tests verify that the uncompiled replacements:
  1. Produce numerically identical outputs
  2. Allow value_and_grad to succeed
  3. Can be applied, reverted, and re-applied without error
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx

import mlx.core as mx
import mlx.nn as nn

from modelcypher.backends._mlx_qwen35_compat import (
    _gated_delta_step_ops_uncompiled,
    apply_qwen35_training_patch,
    revert_qwen35_training_patch,
)


@pytest.fixture(autouse=True)
def _clean_patch_state():
    """Ensure patch is reverted after each test."""
    yield
    revert_qwen35_training_patch()


def _make_step_inputs(B=1, H=2, Dk=4, Dv=4, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, Dk))
    k = mx.random.normal((B, H, Dk))
    v = mx.random.normal((B, H, Dv))
    g = mx.sigmoid(mx.random.normal((B, H)))      # scalar gating [B, H]
    beta = mx.sigmoid(mx.random.normal((B, H)))
    state = mx.zeros((B, H, Dv, Dk))
    return q, k, v, g, beta, state


@pytest.mark.mlx
def test_uncompiled_step_ops_matches_reference():
    """Uncompiled step produces numerically identical output to the compiled version."""
    import mlx_lm.models.gated_delta as gd

    q, k, v, g, beta, state = _make_step_inputs()

    # Run compiled (original) version — safe outside gradient tape
    y_compiled, s_compiled = gd._gated_delta_step_ops(q, k, v, g, beta, state)
    mx.eval(y_compiled, s_compiled)

    # Run uncompiled version
    y_uncompiled, s_uncompiled = _gated_delta_step_ops_uncompiled(q, k, v, g, beta, state)
    mx.eval(y_uncompiled, s_uncompiled)

    # Must match to float32 tolerance
    atol = 1e-5
    assert mx.allclose(y_compiled, y_uncompiled, atol=atol).item(), (
        f"y mismatch: max diff {mx.max(mx.abs(y_compiled - y_uncompiled)).item()}"
    )
    assert mx.allclose(s_compiled, s_uncompiled, atol=atol).item(), (
        f"state mismatch: max diff {mx.max(mx.abs(s_compiled - s_uncompiled)).item()}"
    )


@pytest.mark.mlx
def test_value_and_grad_through_uncompiled_step():
    """After patch, value_and_grad succeeds through GatedDeltaNet ops path."""
    apply_qwen35_training_patch()

    import mlx_lm.models.gated_delta as gd

    q, k, v, g, beta, state = _make_step_inputs()

    def loss_fn(q_, k_, v_, g_, beta_):
        y, _ = gd._gated_delta_step_ops(q_, k_, v_, g_, beta_, state)
        return y.mean()

    # This should NOT raise [Primitive::vjp] Not implemented for CustomKernel
    loss_and_grad = mx.value_and_grad(loss_fn)
    loss_val, grads = loss_and_grad(q, k, v, g, beta)
    mx.eval(loss_val, *grads)

    assert loss_val.shape == ()
    # At least some gradients must be non-zero
    assert any(mx.any(mx.abs(g_) > 0).item() for g_ in grads if g_ is not None)


@pytest.mark.mlx
def test_patch_is_idempotent_and_reverts():
    """apply is idempotent; revert restores original compiled functions."""
    import mlx_lm.models.gated_delta as gd

    original_step = gd._gated_delta_step_ops
    original_compute_g = gd.compute_g

    # Apply twice — no error, no double-wrapping
    apply_qwen35_training_patch()
    apply_qwen35_training_patch()

    assert gd._gated_delta_step_ops is _gated_delta_step_ops_uncompiled
    from modelcypher.backends._mlx_qwen35_compat import _compute_g_uncompiled
    assert gd.compute_g is _compute_g_uncompiled

    # Revert — originals restored
    revert_qwen35_training_patch()

    assert gd._gated_delta_step_ops is original_step
    assert gd.compute_g is original_compute_g

    # Can patch again after revert
    apply_qwen35_training_patch()
    assert gd._gated_delta_step_ops is _gated_delta_step_ops_uncompiled
