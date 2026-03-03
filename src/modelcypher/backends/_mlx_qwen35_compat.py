"""
Gradient compatibility patch for Qwen3.5 GatedDeltaNet training.

mlx-lm wraps _gated_delta_step_ops and compute_g with @mx.compile, which
produces MLX CustomKernel primitives with no registered VJP. This blocks
value_and_grad during NB-LoRA training.

Fix: replace both with identical uncompiled versions before training.
The math is unchanged; only JIT compilation is removed.

Why this works: unsloth/HuggingFace train the same GatedDeltaNet architecture
on CUDA using standard PyTorch autograd — no custom backward kernels. The ops
(outer products, element-wise gating, reductions) all have built-in VJPs. MLX
autograd handles them correctly once @mx.compile is removed.
"""
from __future__ import annotations

import contextlib
import importlib
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _compute_g_uncompiled(A_log, a, dt_bias):
    """Decay gate — identical to compute_g but without @mx.compile."""
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    ).astype(A_log.dtype)


def _gated_delta_step_ops_uncompiled(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """Single GatedDeltaNet recurrent step — identical to _gated_delta_step_ops
    but without @mx.compile so MLX autograd can differentiate through it.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]            # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)         # [B, H, Dv]
    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y, state


def _is_qwen35(model) -> bool:
    """Return True if model contains GatedDeltaNet layers.

    Uses nn.Module.apply_to_modules which is MLX's canonical tree traversal API.
    (vars() only returns __dict__ entries; MLX stores sub-modules in the underlying
    dict, so apply_to_modules is required to walk the full hierarchy.)
    """
    try:
        import mlx_lm.models.qwen3_5 as q35
    except ImportError:
        return False
    found = [False]

    def _check(name, module):
        if isinstance(module, q35.GatedDeltaNet):
            found[0] = True

    model.apply_to_modules(_check)
    return found[0]


_patched = False


def apply_qwen35_training_patch() -> None:
    """Swap @mx.compile-wrapped functions for uncompiled equivalents.

    Idempotent — safe to call multiple times. The patch is global
    (module-level state) and persists for the process lifetime.
    """
    global _patched
    if _patched:
        return
    gd = importlib.import_module("mlx_lm.models.gated_delta")
    gd._original_step_ops = gd._gated_delta_step_ops
    gd._original_compute_g = gd.compute_g
    gd._gated_delta_step_ops = _gated_delta_step_ops_uncompiled
    gd.compute_g = _compute_g_uncompiled
    _patched = True


def revert_qwen35_training_patch() -> None:
    """Restore original compiled functions (restores inference-speed kernels)."""
    global _patched
    if not _patched:
        return
    gd = importlib.import_module("mlx_lm.models.gated_delta")
    if hasattr(gd, "_original_step_ops"):
        gd._gated_delta_step_ops = gd._original_step_ops
        gd.compute_g = gd._original_compute_g
    _patched = False


@contextlib.contextmanager
def qwen35_training_mode():
    """Context manager: patch for training, revert on exit."""
    apply_qwen35_training_patch()
    try:
        yield
    finally:
        revert_qwen35_training_patch()
