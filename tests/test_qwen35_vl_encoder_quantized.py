# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

"""Quantized Qwen3.5 visual-checkpoint compatibility tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx

import mlx.nn as nn

from modelcypher.backends._mlx_qwen35_vl_encoder import (
    Qwen35VisionConfig,
    Qwen35VisionEncoder,
    _prepare_quantized_visual_encoder,
)


def _small_encoder() -> Qwen35VisionEncoder:
    return Qwen35VisionEncoder(
        Qwen35VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            depth=1,
            out_hidden_size=64,
            patch_size=2,
            temporal_patch_size=2,
            num_position_embeddings=64,
        )
    )


def test_prepare_quantized_visual_encoder_uses_scales_as_module_truth():
    encoder = _small_encoder()
    count = _prepare_quantized_visual_encoder(
        encoder,
        {
            "blocks.0.attn.qkv.scales": object(),
            "blocks.0.attn.qkv.biases": object(),
            "patch_embed.weight": object(),
        },
        {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}},
    )
    assert count == 1
    assert isinstance(encoder.blocks[0].attn.qkv, nn.QuantizedLinear)
    assert isinstance(encoder.patch_embed, nn.Linear)


def test_prepare_quantized_visual_encoder_is_noop_for_float_checkpoint():
    encoder = _small_encoder()
    count = _prepare_quantized_visual_encoder(
        encoder,
        {"blocks.0.attn.qkv.weight": object()},
        {},
    )
    assert count == 0
    assert isinstance(encoder.blocks[0].attn.qkv, nn.Linear)

