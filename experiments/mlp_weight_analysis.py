#!/usr/bin/env python3
"""Analyze gate_proj and up_proj weight correlation patterns.

The key insight: gate_silu * up creates extreme values when specific
gate and up neurons are both large. This requires weight alignment
between gate and up to be preserved.
"""

import logging
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_mlp_weights(model, layer_idx: int) -> dict:
    """Analyze correlation patterns between gate_proj and up_proj weights."""

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    layers = base.layers if hasattr(base, "layers") else base.h
    layer = layers[layer_idx]
    mlp = layer.mlp

    gate_weight = mlp.gate_proj.weight  # [intermediate_dim, hidden_dim]
    up_weight = mlp.up_proj.weight      # [intermediate_dim, hidden_dim]
    down_weight = mlp.down_proj.weight  # [hidden_dim, intermediate_dim]

    mx.eval(gate_weight, up_weight, down_weight)

    logger.info(f"\n{'='*80}")
    logger.info(f"LAYER {layer_idx} MLP WEIGHT ANALYSIS")
    logger.info(f"{'='*80}")

    logger.info(f"gate_proj shape: {gate_weight.shape}")
    logger.info(f"up_proj shape:   {up_weight.shape}")
    logger.info(f"down_proj shape: {down_weight.shape}")

    # Per-neuron (intermediate dimension) statistics
    # Each row of gate/up is one "neuron" in intermediate space

    # Compute gate × up correlation for each intermediate neuron
    # If gate[i] and up[i] are aligned, their product will be large

    # Row norms (L2 norm of each intermediate neuron's weight)
    gate_row_norms = mx.sqrt(mx.sum(gate_weight * gate_weight, axis=1))  # [intermediate_dim]
    up_row_norms = mx.sqrt(mx.sum(up_weight * up_weight, axis=1))        # [intermediate_dim]

    mx.eval(gate_row_norms, up_row_norms)

    # Product of norms (potential for large activation)
    norm_products = gate_row_norms * up_row_norms
    mx.eval(norm_products)

    logger.info(f"\n--- Per-Neuron Row Norms ---")
    logger.info(f"gate_row_norms: mean={float(mx.mean(gate_row_norms)):.4f}, max={float(mx.max(gate_row_norms)):.4f}")
    logger.info(f"up_row_norms:   mean={float(mx.mean(up_row_norms)):.4f}, max={float(mx.max(up_row_norms)):.4f}")
    logger.info(f"norm_products:  mean={float(mx.mean(norm_products)):.4f}, max={float(mx.max(norm_products)):.4f}")

    # Find neurons with highest norm products (these create extreme activations)
    top_k = 10
    sorted_indices = mx.argsort(norm_products)
    top_indices = sorted_indices[-top_k:][::-1]  # Top k descending
    mx.eval(top_indices)

    logger.info(f"\n--- Top {top_k} Neurons by Norm Product ---")
    for i in range(top_k):
        idx = int(top_indices[i])
        gn = float(gate_row_norms[idx])
        un = float(up_row_norms[idx])
        prod = float(norm_products[idx])
        logger.info(f"  Neuron {idx:4d}: gate_norm={gn:.4f}, up_norm={un:.4f}, product={prod:.4f}")

    # Cosine similarity between corresponding gate and up rows
    # (measures if they "point" in the same direction)
    dot_products = mx.sum(gate_weight * up_weight, axis=1)  # [intermediate_dim]
    cosine_sim = dot_products / (gate_row_norms * up_row_norms + 1e-8)
    mx.eval(cosine_sim)

    logger.info(f"\n--- Gate-Up Cosine Similarity ---")
    logger.info(f"cosine_sim: mean={float(mx.mean(cosine_sim)):.4f}, std={float(mx.std(cosine_sim)):.4f}")
    logger.info(f"            min={float(mx.min(cosine_sim)):.4f}, max={float(mx.max(cosine_sim)):.4f}")

    # How many neurons have highly correlated gate/up weights?
    high_corr_count = int(mx.sum(mx.abs(cosine_sim) > 0.5))
    logger.info(f"Neurons with |cosine| > 0.5: {high_corr_count} / {gate_weight.shape[0]}")

    # Down projection analysis
    down_col_norms = mx.sqrt(mx.sum(down_weight * down_weight, axis=0))  # [intermediate_dim]
    mx.eval(down_col_norms)

    logger.info(f"\n--- Down Projection Column Norms ---")
    logger.info(f"down_col_norms: mean={float(mx.mean(down_col_norms)):.4f}, max={float(mx.max(down_col_norms)):.4f}")

    # Combined: which neurons have high gate×up AND high down norm?
    # These are the neurons that can cause extreme output
    combined = norm_products * down_col_norms
    mx.eval(combined)

    logger.info(f"\n--- Combined (gate×up norm × down norm) ---")
    logger.info(f"combined: mean={float(mx.mean(combined)):.4f}, max={float(mx.max(combined)):.4f}")

    sorted_combined = mx.argsort(combined)
    top_combined = sorted_combined[-top_k:][::-1]
    mx.eval(top_combined)

    logger.info(f"\n--- Top {top_k} by Combined Score ---")
    for i in range(top_k):
        idx = int(top_combined[i])
        gn = float(gate_row_norms[idx])
        un = float(up_row_norms[idx])
        dn = float(down_col_norms[idx])
        comb = float(combined[idx])
        cs = float(cosine_sim[idx])
        logger.info(f"  Neuron {idx:4d}: g={gn:.4f}, u={un:.4f}, d={dn:.4f}, combined={comb:.4f}, cos={cs:.4f}")

    return {
        "gate_row_norms_mean": float(mx.mean(gate_row_norms)),
        "gate_row_norms_max": float(mx.max(gate_row_norms)),
        "up_row_norms_mean": float(mx.mean(up_row_norms)),
        "up_row_norms_max": float(mx.max(up_row_norms)),
        "norm_products_mean": float(mx.mean(norm_products)),
        "norm_products_max": float(mx.max(norm_products)),
        "cosine_sim_mean": float(mx.mean(cosine_sim)),
        "combined_max": float(mx.max(combined)),
    }


def compare_mlp_weights(target_path: str, merged_path: str, layer_idx: int):
    """Compare MLP weight patterns between target and merged."""

    loader = ModelLoader()

    logger.info(f"Loading target: {target_path}")
    target_model, _ = loader.load_model_for_training(target_path)
    target_stats = analyze_mlp_weights(target_model, layer_idx)

    del target_model
    mx.eval([])

    logger.info(f"\nLoading merged: {merged_path}")
    merged_model, _ = loader.load_model_for_training(merged_path)
    merged_stats = analyze_mlp_weights(merged_model, layer_idx)

    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")

    for key in target_stats:
        tgt = target_stats[key]
        mrg = merged_stats[key]
        ratio = mrg / (tgt + 1e-8)
        logger.info(f"  {key:30s}: target={tgt:10.4f}, merged={mrg:10.4f}, ratio={ratio:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--merged", help="Merged model path")
    parser.add_argument("--layer", type=int, default=11)

    args = parser.parse_args()

    if args.merged:
        compare_mlp_weights(args.target, args.merged, args.layer)
    else:
        loader = ModelLoader()
        model, _ = loader.load_model_for_training(args.target)
        analyze_mlp_weights(model, args.layer)


if __name__ == "__main__":
    main()
