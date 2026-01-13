#!/usr/bin/env python3
"""Trace activation variance through a single layer's internal components.

Identifies exactly which operation causes variance explosion/collapse:
- Input LayerNorm
- Self-attention (Q, K, V projections, attention scores, output projection)
- Residual connection
- Post-attention LayerNorm
- MLP (gate, up, down projections)
- Final residual connection
"""

import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.mlx_model_loader import MLXModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_var(x: mx.array) -> float:
    """Compute variance of flattened tensor."""
    flat = x.reshape(-1)
    mx.eval(flat)
    return float(mx.var(flat))


def trace_layer_internals(model, input_ids: mx.array, layer_idx: int) -> dict:
    """Trace variance through a specific layer's internal components."""

    logger.info(f"Tracing layer {layer_idx} internals...")

    # Get model base
    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    # Add batch dimension
    if len(input_ids.shape) == 1:
        input_ids = input_ids.reshape(1, -1)

    # Embed
    if hasattr(base, "embed_tokens"):
        h = base.embed_tokens(input_ids)
    elif hasattr(base, "wte"):
        h = base.wte(input_ids)
    else:
        raise ValueError("Cannot find embedding layer")

    mx.eval(h)

    # Get layers
    if hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(base, "h"):
        layers = base.h
    else:
        raise ValueError("Cannot find transformer layers")

    # Create mask
    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Run through layers BEFORE target layer
    for i in range(layer_idx):
        try:
            result = layers[i](h, mask=mask, cache=None)
        except TypeError:
            try:
                result = layers[i](h, mask, None)
            except TypeError:
                result = layers[i](h, mask)

        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result
        mx.eval(h)

    # Now trace through target layer
    layer = layers[layer_idx]
    results = {"layer_idx": layer_idx}

    # Input to layer
    input_var = compute_var(h)
    results["input_var"] = input_var
    logger.info(f"  Layer input var: {input_var:.4e}")

    # Check what type of layer this is
    # SmolLM uses LlamaDecoderLayer style

    # === Self-Attention Block ===

    # 1. Input LayerNorm (pre-attention)
    if hasattr(layer, "input_layernorm"):
        h_norm = layer.input_layernorm(h)
        norm_var = compute_var(h_norm)
        results["post_input_norm_var"] = norm_var
        logger.info(f"  After input_layernorm: {norm_var:.4e} (ratio: {norm_var/input_var:.2f}x)")
    elif hasattr(layer, "ln_1"):
        h_norm = layer.ln_1(h)
        norm_var = compute_var(h_norm)
        results["post_input_norm_var"] = norm_var
        logger.info(f"  After ln_1: {norm_var:.4e} (ratio: {norm_var/input_var:.2f}x)")
    else:
        h_norm = h
        norm_var = input_var
        results["post_input_norm_var"] = norm_var

    # 2. Self-attention
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn

        # Q, K, V projections
        if hasattr(attn, "q_proj"):
            q = attn.q_proj(h_norm)
            q_var = compute_var(q)
            results["q_var"] = q_var
            logger.info(f"    Q projection var: {q_var:.4e} (ratio: {q_var/norm_var:.2f}x)")

        if hasattr(attn, "k_proj"):
            k = attn.k_proj(h_norm)
            k_var = compute_var(k)
            results["k_var"] = k_var
            logger.info(f"    K projection var: {k_var:.4e} (ratio: {k_var/norm_var:.2f}x)")

        if hasattr(attn, "v_proj"):
            v = attn.v_proj(h_norm)
            v_var = compute_var(v)
            results["v_var"] = v_var
            logger.info(f"    V projection var: {v_var:.4e} (ratio: {v_var/norm_var:.2f}x)")

        # Full attention output
        try:
            attn_out = attn(h_norm, mask=mask, cache=None)
        except TypeError:
            try:
                attn_out = attn(h_norm, mask, None)
            except TypeError:
                attn_out = attn(h_norm, mask)

        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        mx.eval(attn_out)
        attn_out_var = compute_var(attn_out)
        results["attn_out_var"] = attn_out_var
        logger.info(f"  Attention output var: {attn_out_var:.4e} (ratio: {attn_out_var/norm_var:.2f}x)")

    elif hasattr(layer, "attn"):
        attn = layer.attn
        try:
            attn_out = attn(h_norm, mask=mask, cache=None)
        except TypeError:
            attn_out = attn(h_norm, mask)

        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        mx.eval(attn_out)
        attn_out_var = compute_var(attn_out)
        results["attn_out_var"] = attn_out_var
        logger.info(f"  Attention output var: {attn_out_var:.4e}")
    else:
        logger.warning("  Could not find self-attention module")
        attn_out = h_norm
        attn_out_var = norm_var

    # 3. First residual connection
    h_residual = h + attn_out
    mx.eval(h_residual)
    residual1_var = compute_var(h_residual)
    results["post_attn_residual_var"] = residual1_var
    logger.info(f"  After attention residual: {residual1_var:.4e} (ratio: {residual1_var/input_var:.2f}x)")

    # === MLP Block ===

    # 4. Post-attention LayerNorm
    if hasattr(layer, "post_attention_layernorm"):
        h_mlp_norm = layer.post_attention_layernorm(h_residual)
        mlp_norm_var = compute_var(h_mlp_norm)
        results["post_attn_norm_var"] = mlp_norm_var
        logger.info(f"  After post_attn_layernorm: {mlp_norm_var:.4e} (ratio: {mlp_norm_var/residual1_var:.2f}x)")
    elif hasattr(layer, "ln_2"):
        h_mlp_norm = layer.ln_2(h_residual)
        mlp_norm_var = compute_var(h_mlp_norm)
        results["post_attn_norm_var"] = mlp_norm_var
        logger.info(f"  After ln_2: {mlp_norm_var:.4e} (ratio: {mlp_norm_var/residual1_var:.2f}x)")
    else:
        h_mlp_norm = h_residual
        mlp_norm_var = residual1_var

    # 5. MLP
    if hasattr(layer, "mlp"):
        mlp = layer.mlp

        # Gate/up projections
        if hasattr(mlp, "gate_proj"):
            gate = mlp.gate_proj(h_mlp_norm)
            gate_var = compute_var(gate)
            results["mlp_gate_var"] = gate_var
            logger.info(f"    MLP gate_proj var: {gate_var:.4e} (ratio: {gate_var/mlp_norm_var:.2f}x)")

        if hasattr(mlp, "up_proj"):
            up = mlp.up_proj(h_mlp_norm)
            up_var = compute_var(up)
            results["mlp_up_var"] = up_var
            logger.info(f"    MLP up_proj var: {up_var:.4e} (ratio: {up_var/mlp_norm_var:.2f}x)")

        # Full MLP output
        mlp_out = mlp(h_mlp_norm)
        mx.eval(mlp_out)
        mlp_out_var = compute_var(mlp_out)
        results["mlp_out_var"] = mlp_out_var
        logger.info(f"  MLP output var: {mlp_out_var:.4e} (ratio: {mlp_out_var/mlp_norm_var:.2f}x)")
    else:
        logger.warning("  Could not find MLP module")
        mlp_out = h_mlp_norm
        mlp_out_var = mlp_norm_var

    # 6. Final residual connection
    h_out = h_residual + mlp_out
    mx.eval(h_out)
    output_var = compute_var(h_out)
    results["output_var"] = output_var
    logger.info(f"  Layer output var: {output_var:.4e} (ratio: {output_var/input_var:.2f}x)")

    # Summary
    logger.info(f"\n  SUMMARY for layer {layer_idx}:")
    logger.info(f"    Input -> Output: {input_var:.4e} -> {output_var:.4e} ({output_var/input_var:.2f}x)")

    if output_var / input_var > 10:
        logger.warning(f"    >>> VARIANCE EXPLOSION DETECTED: {output_var/input_var:.2f}x <<<")
    elif output_var / input_var < 0.1:
        logger.warning(f"    >>> VARIANCE COLLAPSE DETECTED: {output_var/input_var:.2f}x <<<")

    return results


def compare_layer_internals(target_path: str, merged_path: str, layer_idx: int, prompt: str):
    """Compare layer internals between target and merged model."""

    loader = MLXModelLoader()

    logger.info("=" * 80)
    logger.info(f"COMPARING LAYER {layer_idx} INTERNALS")
    logger.info("=" * 80)

    # Load target
    logger.info(f"\nLoading target: {target_path}")
    target_model, target_tok = loader.load_model_for_training(target_path)

    input_ids = mx.array(target_tok.encode(prompt))
    logger.info(f"Tokens: {input_ids.shape[0]}")

    logger.info("\n" + "=" * 40)
    logger.info("TARGET MODEL")
    logger.info("=" * 40)
    target_results = trace_layer_internals(target_model, input_ids, layer_idx)

    # Free target model
    del target_model
    mx.eval([])

    # Load merged
    logger.info(f"\nLoading merged: {merged_path}")
    merged_model, merged_tok = loader.load_model_for_training(merged_path)

    logger.info("\n" + "=" * 40)
    logger.info("MERGED MODEL")
    logger.info("=" * 40)
    merged_results = trace_layer_internals(merged_model, input_ids, layer_idx)

    # Compare
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    for key in target_results:
        if key == "layer_idx":
            continue
        tgt_val = target_results.get(key, 0)
        mrg_val = merged_results.get(key, 0)
        if tgt_val > 0:
            ratio = mrg_val / tgt_val
            logger.info(f"  {key:30s}: target={tgt_val:12.4e}, merged={mrg_val:12.4e}, ratio={ratio:.4f}")

    return target_results, merged_results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target model path")
    parser.add_argument("--merged", help="Merged model path (optional)")
    parser.add_argument("--layer", type=int, default=11, help="Layer to trace")
    parser.add_argument("--prompt", default="The capital of France is Paris, which is known for")

    args = parser.parse_args()

    if args.merged:
        compare_layer_internals(args.target, args.merged, args.layer, args.prompt)
    else:
        loader = MLXModelLoader()
        model, tok = loader.load_model_for_training(args.target)
        input_ids = mx.array(tok.encode(args.prompt))
        trace_layer_internals(model, input_ids, args.layer)


if __name__ == "__main__":
    main()
