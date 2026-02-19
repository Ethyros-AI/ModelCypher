"""Tests for model_backbone.py — mask routing for LFM2 hybrid models."""

from __future__ import annotations

from modelcypher.adapters.model_backbone import _resolve_layer_mask


class _FakeAttentionLayer:
    is_attention_layer = True


class _FakeConvLayer:
    is_attention_layer = False


class _FakeStandardLayer:
    """Standard transformer layer — no is_attention_layer attribute."""

    pass


_NUMERIC_MASK = [[1.0, 0.0], [1.0, 1.0]]


def test_resolve_mask_lfm2_attention_layer() -> None:
    """LFM2 attention layer gets string 'causal'."""
    mask = _resolve_layer_mask(_FakeAttentionLayer(), _NUMERIC_MASK)
    assert mask == "causal"


def test_resolve_mask_lfm2_conv_layer() -> None:
    """LFM2 conv layer gets None."""
    mask = _resolve_layer_mask(_FakeConvLayer(), _NUMERIC_MASK)
    assert mask is None


def test_resolve_mask_standard_transformer() -> None:
    """Standard transformer layer (no is_attention_layer) gets numeric mask."""
    mask = _resolve_layer_mask(_FakeStandardLayer(), _NUMERIC_MASK)
    assert mask is _NUMERIC_MASK
