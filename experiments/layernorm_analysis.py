#!/usr/bin/env python3
"""Analyze LayerNorm parameters between target and merged."""

import logging
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.mlx_model_loader import MLXModelLoader

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_layernorm(model, layer_idx: int) -> dict:
    """Analyze LayerNorm parameters at a layer."""

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    layers = base.layers if hasattr(base, "layers") else base.h
    layer = layers[layer_idx]

    results = {}

    # Input LayerNorm (pre-attention)
    if hasattr(layer, "input_layernorm"):
        ln = layer.input_layernorm
        weight = ln.weight
        mx.eval(weight)
        results["input_ln_weight_mean"] = float(mx.mean(weight))
        results["input_ln_weight_std"] = float(mx.std(weight))
        results["input_ln_weight_min"] = float(mx.min(weight))
        results["input_ln_weight_max"] = float(mx.max(weight))

        if hasattr(ln, "bias") and ln.bias is not None:
            bias = ln.bias
            mx.eval(bias)
            results["input_ln_bias_mean"] = float(mx.mean(bias))
        else:
            results["input_ln_bias_mean"] = 0.0

    # Post-attention LayerNorm (before MLP)
    if hasattr(layer, "post_attention_layernorm"):
        ln = layer.post_attention_layernorm
        weight = ln.weight
        mx.eval(weight)
        results["post_attn_ln_weight_mean"] = float(mx.mean(weight))
        results["post_attn_ln_weight_std"] = float(mx.std(weight))
        results["post_attn_ln_weight_min"] = float(mx.min(weight))
        results["post_attn_ln_weight_max"] = float(mx.max(weight))

        if hasattr(ln, "bias") and ln.bias is not None:
            bias = ln.bias
            mx.eval(bias)
            results["post_attn_ln_bias_mean"] = float(mx.mean(bias))
        else:
            results["post_attn_ln_bias_mean"] = 0.0

    return results


def compare_layernorms(target_path: str, merged_path: str, layer_idx: int):
    """Compare LayerNorm parameters between target and merged."""

    loader = MLXModelLoader()

    logger.info(f"Loading target: {target_path}")
    target_model, _ = loader.load_model_for_training(target_path)

    logger.info(f"\n=== TARGET Layer {layer_idx} LayerNorm ===")
    target_stats = analyze_layernorm(target_model, layer_idx)
    for k, v in target_stats.items():
        logger.info(f"  {k}: {v:.6f}")

    del target_model
    mx.eval([])

    logger.info(f"\nLoading merged: {merged_path}")
    merged_model, _ = loader.load_model_for_training(merged_path)

    logger.info(f"\n=== MERGED Layer {layer_idx} LayerNorm ===")
    merged_stats = analyze_layernorm(merged_model, layer_idx)
    for k, v in merged_stats.items():
        logger.info(f"  {k}: {v:.6f}")

    logger.info(f"\n=== COMPARISON ===")
    for k in target_stats:
        tgt = target_stats[k]
        mrg = merged_stats[k]
        diff = mrg - tgt
        logger.info(f"  {k:35s}: target={tgt:10.6f}, merged={mrg:10.6f}, diff={diff:+10.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--layer", type=int, default=11)

    args = parser.parse_args()
    compare_layernorms(args.target, args.merged, args.layer)


if __name__ == "__main__":
    main()
