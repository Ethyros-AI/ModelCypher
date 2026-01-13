#!/usr/bin/env python3
"""Trace variance through MLP internals: gate, up, activation, down."""

import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.mlx_model_loader import MLXModelLoader

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_var(x: mx.array) -> float:
    flat = x.reshape(-1)
    mx.eval(flat)
    return float(mx.var(flat))


def compute_stats(x: mx.array, name: str) -> dict:
    """Compute comprehensive stats for a tensor."""
    mx.eval(x)
    flat = x.reshape(-1)
    var = float(mx.var(flat))
    mean = float(mx.mean(flat))
    abs_mean = float(mx.mean(mx.abs(flat)))
    max_val = float(mx.max(mx.abs(flat)))
    min_val = float(mx.min(flat))

    logger.info(f"  {name:30s}: var={var:12.4e}, mean={mean:10.4f}, |mean|={abs_mean:10.4f}, max={max_val:12.4e}, min={min_val:12.4e}")
    return {"var": var, "mean": mean, "abs_mean": abs_mean, "max": max_val, "min": min_val}


def trace_mlp_internals(model, input_ids: mx.array, layer_idx: int) -> dict:
    """Trace variance through MLP internals."""

    logger.info(f"Tracing MLP internals at layer {layer_idx}...")

    # Get model base
    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    # Add batch dim
    if len(input_ids.shape) == 1:
        input_ids = input_ids.reshape(1, -1)

    # Embed
    if hasattr(base, "embed_tokens"):
        h = base.embed_tokens(input_ids)
    else:
        h = base.wte(input_ids)
    mx.eval(h)

    layers = base.layers if hasattr(base, "layers") else base.h

    # Create mask
    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Run through layers before target
    for i in range(layer_idx):
        result = layers[i](h, mask=mask, cache=None)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

    layer = layers[layer_idx]

    logger.info(f"\n{'='*80}")
    logger.info(f"LAYER {layer_idx} MLP ANALYSIS")
    logger.info(f"{'='*80}")

    # Input to layer
    compute_stats(h, "Layer input")

    # Input LayerNorm
    if hasattr(layer, "input_layernorm"):
        h_norm = layer.input_layernorm(h)
    else:
        h_norm = layer.ln_1(h)
    compute_stats(h_norm, "After input LayerNorm")

    # Self-attention
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        try:
            attn_out = attn(h_norm, mask=mask, cache=None)
        except TypeError:
            attn_out = attn(h_norm, mask)
        attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        mx.eval(attn_out)

    # Residual
    h_residual = h + attn_out
    mx.eval(h_residual)
    compute_stats(h_residual, "After attn residual")

    # Post-attention LayerNorm
    if hasattr(layer, "post_attention_layernorm"):
        h_mlp_input = layer.post_attention_layernorm(h_residual)
    else:
        h_mlp_input = layer.ln_2(h_residual)
    mx.eval(h_mlp_input)

    mlp_input_stats = compute_stats(h_mlp_input, "MLP input (after LN)")

    # === MLP INTERNALS ===
    logger.info(f"\n--- MLP Internals ---")

    mlp = layer.mlp

    # Gate projection
    if hasattr(mlp, "gate_proj"):
        gate_out = mlp.gate_proj(h_mlp_input)
        mx.eval(gate_out)
        gate_stats = compute_stats(gate_out, "gate_proj output")

        # Check gate_proj weight stats
        gate_weight = mlp.gate_proj.weight
        mx.eval(gate_weight)
        logger.info(f"    gate_proj weight shape: {gate_weight.shape}")
        compute_stats(gate_weight, "    gate_proj weight")

    # Up projection
    if hasattr(mlp, "up_proj"):
        up_out = mlp.up_proj(h_mlp_input)
        mx.eval(up_out)
        up_stats = compute_stats(up_out, "up_proj output")

        # Check up_proj weight stats
        up_weight = mlp.up_proj.weight
        mx.eval(up_weight)
        logger.info(f"    up_proj weight shape: {up_weight.shape}")
        compute_stats(up_weight, "    up_proj weight")

    # SiLU activation on gate
    gate_silu = nn.silu(gate_out)
    mx.eval(gate_silu)
    gate_silu_stats = compute_stats(gate_silu, "gate after SiLU")

    # Gated activation: gate_silu * up_out
    gated = gate_silu * up_out
    mx.eval(gated)
    gated_stats = compute_stats(gated, "gate_silu * up_out")

    # Down projection
    if hasattr(mlp, "down_proj"):
        down_out = mlp.down_proj(gated)
        mx.eval(down_out)
        down_stats = compute_stats(down_out, "down_proj output (MLP out)")

        # Check down_proj weight stats
        down_weight = mlp.down_proj.weight
        mx.eval(down_weight)
        logger.info(f"    down_proj weight shape: {down_weight.shape}")
        compute_stats(down_weight, "    down_proj weight")

    # Final MLP output through full forward pass
    full_mlp_out = mlp(h_mlp_input)
    mx.eval(full_mlp_out)
    compute_stats(full_mlp_out, "Full MLP output (verify)")

    # Final layer output (residual)
    layer_out = h_residual + full_mlp_out
    mx.eval(layer_out)
    compute_stats(layer_out, "Layer output (final)")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("VARIANCE RATIOS:")
    logger.info(f"  MLP input -> gate_proj: {gate_stats['var']/mlp_input_stats['var']:.2f}x")
    logger.info(f"  MLP input -> up_proj:   {up_stats['var']/mlp_input_stats['var']:.2f}x")
    logger.info(f"  gate_proj -> gate_silu: {gate_silu_stats['var']/gate_stats['var']:.2f}x")
    logger.info(f"  gate_silu * up -> gated: {gated_stats['var']/(gate_silu_stats['var']*up_stats['var']):.4e}x (product)")
    logger.info(f"  gated -> down_proj:     {down_stats['var']/gated_stats['var']:.2f}x")
    logger.info(f"  Overall MLP:            {down_stats['var']/mlp_input_stats['var']:.2e}x")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--prompt", default="The capital of France is Paris, which is known for")

    args = parser.parse_args()

    loader = MLXModelLoader()
    model, tok = loader.load_model_for_training(args.target)
    input_ids = mx.array(tok.encode(args.prompt))

    trace_mlp_internals(model, input_ids, args.layer)


if __name__ == "__main__":
    main()
