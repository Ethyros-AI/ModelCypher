#!/usr/bin/env python3
"""
Analyze what creates manifold curvature.

Finding: Curvature correlates with ID (r = 0.821).
Question: What creates curvature?

Hypotheses:
1. MLP nonlinearity (SiLU) creates curvature
2. Attention mixing creates curvature
3. LayerNorm projection creates curvature
4. Cumulative residual additions create curvature
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.spatial.distance import pdist, squareform


def measure_local_curvature(X, k=10, n_samples=100):
    """
    Estimate local curvature via PCA residual in neighborhoods.
    """
    n, d = X.shape
    if n < k + 1:
        return 0.0

    dists = squareform(pdist(X))

    curvatures = []
    sample_idx = np.random.choice(n, min(n, n_samples), replace=False)

    for i in sample_idx:
        nn_idx = np.argsort(dists[i])[1:k+1]
        neighbors = X[nn_idx]
        centered = neighbors - np.mean(neighbors, axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)

        if len(S) > 2:
            top2_var = (S[0]**2 + S[1]**2) / (np.sum(S**2) + 1e-10)
            curvatures.append(1 - top2_var)

    return np.mean(curvatures) if curvatures else 0


def analyze_component_curvature(model, layer_idx, h):
    """Analyze curvature contribution from each component."""
    layer = model.model.layers[layer_idx]

    # Input
    h_np = np.array(h.astype(mx.float32))[0]
    input_curv = measure_local_curvature(h_np)

    # After input layernorm
    normed = layer.input_layernorm(h)
    mx.eval(normed)
    normed_np = np.array(normed.astype(mx.float32))[0]
    after_norm1_curv = measure_local_curvature(normed_np)

    # After attention (no residual)
    attn_out = layer.self_attn(normed)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    mx.eval(attn_out)
    attn_np = np.array(attn_out.astype(mx.float32))[0]
    attn_curv = measure_local_curvature(attn_np)

    # After attention + residual
    h_after_attn = h + attn_out
    mx.eval(h_after_attn)
    after_attn_res_np = np.array(h_after_attn.astype(mx.float32))[0]
    after_attn_res_curv = measure_local_curvature(after_attn_res_np)

    # After post-attention layernorm
    normed2 = layer.post_attention_layernorm(h_after_attn)
    mx.eval(normed2)
    normed2_np = np.array(normed2.astype(mx.float32))[0]
    after_norm2_curv = measure_local_curvature(normed2_np)

    # After MLP (no residual)
    mlp_out = layer.mlp(normed2)
    mx.eval(mlp_out)
    mlp_np = np.array(mlp_out.astype(mx.float32))[0]
    mlp_curv = measure_local_curvature(mlp_np)

    # Full layer output
    layer_out = h_after_attn + mlp_out
    mx.eval(layer_out)
    output_np = np.array(layer_out.astype(mx.float32))[0]
    output_curv = measure_local_curvature(output_np)

    return {
        'input': input_curv,
        'after_norm1': after_norm1_curv,
        'attn_only': attn_curv,
        'after_attn_res': after_attn_res_curv,
        'after_norm2': after_norm2_curv,
        'mlp_only': mlp_curv,
        'output': output_curv,
    }


def main():
    print("="*70)
    print("  CURVATURE SOURCE ANALYSIS")
    print("="*70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    print(f"\nLoading {model_path.split('/')[-1]}...")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    embed = model.model.embed_tokens

    # Use a representative prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    print(f"Tokens: {len(tokens)}")

    # Sample layers across the network
    layers_to_check = [0, 5, 10, 15, 20, 25, 30, 35]

    print("\n| Layer | Input | Norm1 | Attn | +Res | Norm2 | MLP | Output | Δ(MLP) |")
    print("|-------|-------|-------|------|------|-------|-----|--------|--------|")

    h = embed(input_ids)
    mx.eval(h)

    results = []

    for layer_idx in range(n_layers):
        if layer_idx in layers_to_check:
            curv = analyze_component_curvature(model, layer_idx, h)

            # Delta from MLP = how much MLP changes curvature
            mlp_delta = curv['output'] - curv['after_attn_res']

            print(f"| {layer_idx:5d} | {curv['input']:.3f} | {curv['after_norm1']:.3f} | "
                  f"{curv['attn_only']:.3f} | {curv['after_attn_res']:.3f} | "
                  f"{curv['after_norm2']:.3f} | {curv['mlp_only']:.3f} | "
                  f"{curv['output']:.3f} | {mlp_delta:+.3f} |")

            results.append({
                'layer': layer_idx,
                **curv,
                'mlp_delta': mlp_delta,
            })

        # Forward through this layer
        h = model.model.layers[layer_idx](h)
        mx.eval(h)

    # Analysis
    print("\n" + "="*70)
    print("  ANALYSIS: What creates curvature?")
    print("="*70)

    print("\nCurvature changes through components:")
    for r in results:
        print(f"\nLayer {r['layer']}:")
        print(f"  Input → Norm1: {r['after_norm1'] - r['input']:+.3f}")
        print(f"  Norm1 → Attn:  {r['attn_only'] - r['after_norm1']:+.3f}")
        print(f"  +Residual:     {r['after_attn_res'] - r['attn_only']:+.3f}")
        print(f"  Norm2 → MLP:   {r['mlp_only'] - r['after_norm2']:+.3f}")
        print(f"  +Residual:     {r['output'] - r['mlp_only']:+.3f}")
        print(f"  Net change:    {r['output'] - r['input']:+.3f}")

    # Check pattern
    print("\n" + "="*70)
    print("  PATTERN DETECTION")
    print("="*70)

    mlp_deltas = [r['mlp_delta'] for r in results]
    output_curvs = [r['output'] for r in results]

    print(f"\nMLP delta range: {min(mlp_deltas):.3f} to {max(mlp_deltas):.3f}")
    print(f"Output curvature range: {min(output_curvs):.3f} to {max(output_curvs):.3f}")

    # Does MLP consistently add or remove curvature?
    avg_mlp_delta = np.mean(mlp_deltas)
    print(f"\nAverage MLP delta: {avg_mlp_delta:+.3f}")
    if avg_mlp_delta > 0.01:
        print("→ MLP tends to ADD curvature")
    elif avg_mlp_delta < -0.01:
        print("→ MLP tends to REMOVE curvature")
    else:
        print("→ MLP has minimal net effect on curvature")


if __name__ == "__main__":
    main()
