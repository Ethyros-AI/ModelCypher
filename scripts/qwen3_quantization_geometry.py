#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Quantization Geometry Analysis
"""
Investigate what quantization does to model geometry:

1. How does quantization affect singular value structure?
2. Does it preserve the "transmission layer" linear properties?
3. Is there a geometry-aware quantization that works better?

Key insight from our compression work:
- Transmission layers (14-21) are effectively LINEAR
- Layer 6 has a dominant singular value (selection gate)
- Different layer types have different geometric signatures

Hypothesis: Quantization that preserves geometric structure
(singular values, effective rank) may outperform uniform quantization.
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def simulate_quantization(W: np.ndarray, bits: int, method: str = "uniform") -> np.ndarray:
    """Simulate quantization of a weight matrix."""
    if method == "uniform":
        # Standard uniform quantization
        w_min, w_max = W.min(), W.max()
        scale = (w_max - w_min) / (2**bits - 1)
        if scale < 1e-10:
            return W.copy()
        W_q = np.round((W - w_min) / scale) * scale + w_min
        return W_q

    elif method == "symmetric":
        # Symmetric quantization (common for weights)
        abs_max = np.abs(W).max()
        if abs_max < 1e-10:
            return W.copy()
        scale = abs_max / (2**(bits-1) - 1)
        W_q = np.round(W / scale) * scale
        return W_q

    elif method == "per_channel":
        # Per-output-channel quantization
        W_q = np.zeros_like(W)
        for i in range(W.shape[0]):
            row = W[i, :]
            abs_max = np.abs(row).max()
            if abs_max < 1e-10:
                W_q[i, :] = row
            else:
                scale = abs_max / (2**(bits-1) - 1)
                W_q[i, :] = np.round(row / scale) * scale
        return W_q

    elif method == "svd_aware":
        # Our novel approach: preserve top singular values exactly
        # Use float64 for numerical stability
        W_f64 = W.astype(np.float64)
        U, S, Vt = np.linalg.svd(W_f64, full_matrices=False)

        # Preserve top-k singular values (based on energy)
        total_energy = np.sum(S**2)
        if total_energy < 1e-20:
            return W.copy()
        cumulative = np.cumsum(S**2) / total_energy
        k = max(1, np.searchsorted(cumulative, 0.99) + 1)  # 99% energy
        k = min(k, min(len(S) // 4, 100))  # At most 25% of dims, cap at 100

        # Reconstruct the "important" part exactly
        W_important = (U[:, :k] * S[:k]) @ Vt[:k, :]

        # Quantize the residual aggressively
        W_residual = W_f64 - W_important
        W_residual_q = simulate_quantization(W_residual.astype(np.float32), bits, "symmetric")

        return (W_important + W_residual_q).astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_weight_geometry(W: np.ndarray) -> Dict:
    """Analyze geometric properties of a weight matrix."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # Effective rank (singular values > 1% of max)
    eff_rank_1pct = np.sum(S > 0.01 * S[0]) if len(S) > 0 else 0
    eff_rank_01pct = np.sum(S > 0.001 * S[0]) if len(S) > 0 else 0

    # Energy concentration
    total_energy = np.sum(S**2)
    top10_energy = np.sum(S[:10]**2) / total_energy if len(S) >= 10 else 1.0
    top100_energy = np.sum(S[:100]**2) / total_energy if len(S) >= 100 else 1.0

    # Condition number
    cond = S[0] / S[-1] if len(S) > 0 and S[-1] > 1e-10 else np.inf

    return {
        'top_sv': S[0] if len(S) > 0 else 0,
        'eff_rank_1pct': eff_rank_1pct,
        'eff_rank_01pct': eff_rank_01pct,
        'top10_energy': top10_energy,
        'top100_energy': top100_energy,
        'condition_number': cond,
        'singular_values': S[:20] if len(S) >= 20 else S,
    }


def measure_quantization_distortion(W: np.ndarray, W_q: np.ndarray) -> Dict:
    """Measure how quantization distorts the weight matrix."""
    # Frobenius norm error
    frob_error = np.linalg.norm(W - W_q) / np.linalg.norm(W)

    # Spectral norm error (largest singular value of difference)
    _, S_diff, _ = np.linalg.svd(W - W_q, full_matrices=False)
    spectral_error = S_diff[0] / np.linalg.svd(W, compute_uv=False)[0] if len(S_diff) > 0 else 0

    # Geometry preservation
    geo_orig = analyze_weight_geometry(W)
    geo_quant = analyze_weight_geometry(W_q)

    # How well are singular values preserved?
    sv_orig = geo_orig['singular_values']
    sv_quant = geo_quant['singular_values']
    min_len = min(len(sv_orig), len(sv_quant))
    sv_error = np.linalg.norm(sv_orig[:min_len] - sv_quant[:min_len]) / np.linalg.norm(sv_orig[:min_len])

    return {
        'frobenius_error': frob_error,
        'spectral_error': spectral_error,
        'sv_preservation_error': sv_error,
        'eff_rank_change': geo_quant['eff_rank_1pct'] - geo_orig['eff_rank_1pct'],
        'top_sv_change': (geo_quant['top_sv'] - geo_orig['top_sv']) / geo_orig['top_sv'] if geo_orig['top_sv'] > 0 else 0,
    }


def test_mlp_linearity_after_quantization(
    model, tokenizer, layer_idx: int, prompts: List[str], bits: int, method: str
) -> Dict:
    """Test if MLP linear approximation still works after quantization."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    layer = inner_model.layers[layer_idx]

    # Get original MLP weights
    gate_proj = np.array(layer.mlp.gate_proj.weight.astype(mx.float32))
    up_proj = np.array(layer.mlp.up_proj.weight.astype(mx.float32))
    down_proj = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

    # Quantize
    gate_proj_q = simulate_quantization(gate_proj, bits, method)
    up_proj_q = simulate_quantization(up_proj, bits, method)
    down_proj_q = simulate_quantization(down_proj, bits, method)

    # Collect original MLP behavior
    X_list, Y_orig_list, Y_quant_list = [], [], []

    for prompt in prompts[:50]:  # Sample for speed
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                x_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x_in)

                # Original MLP output
                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)
                Y_orig_list.append(np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64))

                # Simulated quantized MLP output
                # gate = x @ gate_proj.T, up = x @ up_proj.T
                gate = x_in @ gate_proj_q.T
                up = x_in @ up_proj_q.T
                # SiLU activation
                silu_gate = gate * (1 / (1 + np.exp(-gate)))
                hidden = silu_gate * up
                # down projection
                y_quant = hidden @ down_proj_q.T
                Y_quant_list.append(y_quant)

                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X = np.stack(X_list, axis=1)
    Y_orig = np.stack(Y_orig_list, axis=1)
    Y_quant = np.stack(Y_quant_list, axis=1)

    # Fit linear model on original
    X_mean = X.mean(axis=1, keepdims=True)
    Y_orig_mean = Y_orig.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_orig_c = Y_orig - Y_orig_mean

    U_x, S_x, Vt_x = np.linalg.svd(X_c, full_matrices=False)
    threshold = 1e-6 * S_x[0] if len(S_x) > 0 else 1e-6
    S_x_inv = np.where(S_x > threshold, 1.0 / S_x, 0.0)
    A_orig = Y_orig_c @ (Vt_x.T * S_x_inv) @ U_x.T

    # Fit linear model on quantized
    Y_quant_mean = Y_quant.mean(axis=1, keepdims=True)
    Y_quant_c = Y_quant - Y_quant_mean
    A_quant = Y_quant_c @ (Vt_x.T * S_x_inv) @ U_x.T

    # Linear reconstruction error
    Y_orig_pred = A_orig @ X_c
    Y_quant_pred = A_quant @ X_c

    orig_linear_error = np.linalg.norm(Y_orig_c - Y_orig_pred) / np.linalg.norm(Y_orig_c)
    quant_linear_error = np.linalg.norm(Y_quant_c - Y_quant_pred) / np.linalg.norm(Y_quant_c)

    # Output distortion from quantization
    output_error = np.linalg.norm(Y_orig - Y_quant) / np.linalg.norm(Y_orig)

    # Transformation matrix change
    A_diff = np.linalg.norm(A_orig - A_quant) / np.linalg.norm(A_orig)

    return {
        'original_linear_error': orig_linear_error,
        'quantized_linear_error': quant_linear_error,
        'output_distortion': output_error,
        'transformation_change': A_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layers", type=str, default="5,6,7,15,16,17,25,26,27",
                        help="Comma-separated layer indices to analyze")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    layer_indices = [int(x) for x in args.layers.split(",")]

    # Test prompts
    prompts = []
    for a in range(1, 11):
        for b in range(1, 11):
            prompts.append(f"{a} + {b} =")
    for c in ["France", "Japan", "Germany", "Italy", "Spain"]:
        prompts.append(f"The capital of {c} is")

    print(f"\n{'='*80}")
    print("QUANTIZATION GEOMETRY ANALYSIS")
    print("="*80)

    # Part 1: Weight Geometry Analysis
    print(f"\n{'='*80}")
    print("PART 1: MLP WEIGHT GEOMETRY BY LAYER")
    print("="*80)

    for layer_idx in layer_indices:
        layer = inner_model.layers[layer_idx]

        gate = np.array(layer.mlp.gate_proj.weight.astype(mx.float32))
        up = np.array(layer.mlp.up_proj.weight.astype(mx.float32))
        down = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

        print(f"\nLayer {layer_idx}:")
        for name, W in [("gate_proj", gate), ("up_proj", up), ("down_proj", down)]:
            geo = analyze_weight_geometry(W)
            print(f"  {name}: shape={W.shape}, top_sv={geo['top_sv']:.2f}, "
                  f"eff_rank={geo['eff_rank_1pct']}, "
                  f"top10_energy={geo['top10_energy']*100:.1f}%")

    # Part 2: Quantization Distortion
    print(f"\n{'='*80}")
    print("PART 2: QUANTIZATION DISTORTION (down_proj)")
    print("="*80)

    methods = ["uniform", "symmetric", "per_channel", "svd_aware"]
    bits_options = [8, 4]

    for layer_idx in [6, 15, 25]:  # Representative layers
        layer = inner_model.layers[layer_idx]
        down = np.array(layer.mlp.down_proj.weight.astype(mx.float32))

        print(f"\nLayer {layer_idx} (down_proj, shape {down.shape}):")

        for bits in bits_options:
            print(f"  {bits}-bit:")
            for method in methods:
                W_q = simulate_quantization(down, bits, method)
                distortion = measure_quantization_distortion(down, W_q)
                print(f"    {method:12s}: frob={distortion['frobenius_error']*100:5.2f}%, "
                      f"spectral={distortion['spectral_error']*100:5.2f}%, "
                      f"sv_err={distortion['sv_preservation_error']*100:5.2f}%")

    # Part 3: MLP Linearity Preservation
    print(f"\n{'='*80}")
    print("PART 3: MLP LINEARITY AFTER QUANTIZATION")
    print("="*80)
    print("(Does our linear T = Y @ pinv(X) still work?)")

    for layer_idx in [6, 15, 25]:
        print(f"\nLayer {layer_idx}:")

        for bits in [8, 4]:
            print(f"  {bits}-bit:")
            for method in ["symmetric", "svd_aware"]:
                result = test_mlp_linearity_after_quantization(
                    model, tokenizer, layer_idx, prompts, bits, method
                )
                print(f"    {method:12s}: orig_linear_err={result['original_linear_error']*100:.4f}%, "
                      f"quant_linear_err={result['quantized_linear_error']*100:.4f}%, "
                      f"output_distort={result['output_distortion']*100:.2f}%")

    # Part 4: Key Insights
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print("="*80)

    print("""
Our compression research suggests these quantization insights:

1. LAYER-SPECIFIC QUANTIZATION
   - Transmission layers (14-21): Can be quantized more aggressively
     because their MLP is effectively linear - errors average out
   - Selection gate layers (6): Need higher precision for the
     dominant singular mode - it controls information routing
   - Encoder/decoder layers: Require careful quantization -
     small errors compound through the network

2. SVD-AWARE QUANTIZATION
   - Preserving top singular values exactly while aggressively
     quantizing the residual could outperform uniform quantization
   - The "effective rank" of each layer determines how many
     singular values matter

3. THE LINEAR MLP INSIGHT
   - For transmission layers, the MLP transformation is LINEAR
   - This means we could potentially:
     a) Compute T = Y @ pinv(X) at full precision
     b) Quantize T instead of individual MLP weights
     c) Use T @ (x - mean) + mean for inference
   - This is FUNDAMENTALLY DIFFERENT from quantizing gate/up/down separately

4. POTENTIAL IMPROVEMENT
   - Current industry: Quantize each weight matrix independently
   - Our insight: Quantize the EFFECTIVE TRANSFORMATION, not components
   - For 8 transmission layers, this could allow int4 with no accuracy loss
""")


if __name__ == "__main__":
    main()
