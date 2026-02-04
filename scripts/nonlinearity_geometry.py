#!/usr/bin/env python3
"""
Analyze how geometry changes through the MLP nonlinearity.

MLP structure (gated, like Llama/Qwen):
    gate = SiLU(W_gate @ h)
    up = W_up @ h
    h_intermediate = gate * up
    h_out = W_down @ h_intermediate

Questions:
1. Does the nonlinearity change intrinsic dimension?
2. Does it change curvature?
3. Does it change spectral distribution (gap, decay)?
4. Does it create sparsity?
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.spatial.distance import pdist, squareform


def effective_rank(singular_values) -> float:
    """Shannon effective rank."""
    s = np.abs(singular_values)
    s_sum = np.sum(s)
    if s_sum < 1e-10:
        return 0.0
    p = s / s_sum
    H = -np.sum(p * np.log(p + 1e-10))
    return np.exp(H)


def plateau_decay(S) -> float:
    """Measure decay rate."""
    if len(S) < 10 or S[1] < 1e-10:
        return 0.0
    return (S[9] / S[1]) ** (1/8)


def spectral_gap(S) -> float:
    """S₁/S₂ ratio."""
    if len(S) < 2 or S[1] < 1e-10:
        return float('inf')
    return S[0] / S[1]


def measure_sparsity(x, threshold_frac=0.01):
    """Fraction of values near zero."""
    x_flat = x.flatten()
    threshold = threshold_frac * np.std(x_flat)
    return np.mean(np.abs(x_flat) < threshold)


def measure_local_curvature(X, k=10, n_samples=100):
    """Estimate local curvature."""
    n, d = X.shape
    if n < k + 1:
        return 0.0
    dists = squareform(pdist(X))
    curvatures = []
    for i in np.random.choice(n, min(n, n_samples), replace=False):
        nn_idx = np.argsort(dists[i])[1:k+1]
        neighbors = X[nn_idx]
        centered = neighbors - np.mean(neighbors, axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        if len(S) > 2:
            curvatures.append(1 - (S[0]**2 + S[1]**2) / (np.sum(S**2) + 1e-10))
    return np.mean(curvatures) if curvatures else 0


def estimate_id_mle(X, k=5):
    """Estimate intrinsic dimension using MLE."""
    n, d = X.shape
    if n < k + 1:
        return 0.0
    dists = squareform(pdist(X))
    ids = []
    for i in range(min(n, 100)):
        d_i = np.sort(dists[i])[1:k+1]
        if d_i[0] > 1e-10:
            log_ratios = np.log(d_i[-1] / d_i[:-1])
            if np.mean(log_ratios) > 1e-10:
                ids.append((k - 1) / np.sum(log_ratios))
    return np.mean(ids) if ids else 0.0


def analyze_geometry(X, name=""):
    """Compute geometric statistics."""
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])

    # Subsample if needed
    if len(X) > 200:
        idx = np.random.choice(len(X), 200, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    mean = np.mean(X_sample, axis=0)
    centered = X_sample - mean
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    return {
        'name': name,
        'eff_rank': effective_rank(S[:20]),
        'gap': spectral_gap(S),
        'decay': plateau_decay(S),
        'sparsity': measure_sparsity(X),
        'curvature': measure_local_curvature(X_sample),
        'id': estimate_id_mle(X_sample),
        'norm': np.mean(np.linalg.norm(X_sample, axis=1)),
    }


def analyze_mlp_stages(model, layer_idx, h):
    """Analyze geometry at each stage of the MLP."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp

    # Get input to MLP (after post-attention norm)
    h_attn = layer.self_attn(layer.input_layernorm(h))
    if isinstance(h_attn, tuple):
        h_attn = h_attn[0]
    h_post_attn = h + h_attn
    h_normed = layer.post_attention_layernorm(h_post_attn)
    mx.eval(h_normed)

    h_np = np.array(h_normed.astype(mx.float32))[0]
    results = [analyze_geometry(h_np, "MLP input")]

    # Check MLP architecture
    if hasattr(mlp, 'gate_proj'):
        # Gated MLP (Llama/Qwen style)
        # gate = SiLU(W_gate @ h)
        # up = W_up @ h
        # intermediate = gate * up
        # out = W_down @ intermediate

        gate_linear = mlp.gate_proj(h_normed)
        up_linear = mlp.up_proj(h_normed)
        mx.eval(gate_linear, up_linear)

        gate_np = np.array(gate_linear.astype(mx.float32))[0]
        up_np = np.array(up_linear.astype(mx.float32))[0]

        results.append(analyze_geometry(gate_np, "Gate (pre-SiLU)"))
        results.append(analyze_geometry(up_np, "Up projection"))

        # Apply SiLU to gate
        gate_act = mx.sigmoid(gate_linear) * gate_linear  # SiLU
        mx.eval(gate_act)
        gate_act_np = np.array(gate_act.astype(mx.float32))[0]
        results.append(analyze_geometry(gate_act_np, "Gate (post-SiLU)"))

        # Multiply gate * up
        intermediate = gate_act * up_linear
        mx.eval(intermediate)
        inter_np = np.array(intermediate.astype(mx.float32))[0]
        results.append(analyze_geometry(inter_np, "Gate * Up"))

        # Down projection
        out = mlp.down_proj(intermediate)
        mx.eval(out)
        out_np = np.array(out.astype(mx.float32))[0]
        results.append(analyze_geometry(out_np, "MLP output"))

    else:
        # Simple MLP
        # up = activation(W_up @ h)
        # out = W_down @ up
        up = mlp.up_proj(h_normed)
        mx.eval(up)
        up_np = np.array(up.astype(mx.float32))[0]
        results.append(analyze_geometry(up_np, "Up (pre-activation)"))

        # Activation (GELU or SiLU)
        if hasattr(mlp, 'act'):
            up_act = mlp.act(up)
        else:
            up_act = mx.nn.gelu(up)  # Default
        mx.eval(up_act)
        up_act_np = np.array(up_act.astype(mx.float32))[0]
        results.append(analyze_geometry(up_act_np, "Up (post-activation)"))

        out = mlp.down_proj(up_act)
        mx.eval(out)
        out_np = np.array(out.astype(mx.float32))[0]
        results.append(analyze_geometry(out_np, "MLP output"))

    return results


def main():
    print("="*80)
    print("  PRE/POST NONLINEARITY GEOMETRY ANALYSIS")
    print("="*80)

    models = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Llama-3.2-3B-Instruct-bf16",
    ]

    prompt = "The quick brown fox jumps over the lazy dog. " * 4

    for model_path in models:
        print(f"\n{'='*80}")
        print(f"Model: {model_path.split('/')[-1]}")
        print("="*80)

        model, tokenizer = load(model_path)
        embed = model.model.embed_tokens
        n_layers = len(model.model.layers)

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        print(f"Tokens: {len(tokens)}")

        # Test at mid and exit layers
        for layer_idx in [n_layers // 2, n_layers - 2]:
            print(f"\n--- Layer {layer_idx} ---")

            # Forward to this layer
            h = embed(input_ids)
            for i in range(layer_idx):
                h = model.model.layers[i](h)
            mx.eval(h)

            # Analyze MLP stages
            results = analyze_mlp_stages(model, layer_idx, h)

            print(f"\n| Stage | Eff.Rank | Gap | Decay | Sparsity | Curvature | ID | Norm |")
            print(f"|-------|----------|-----|-------|----------|-----------|-----|------|")
            for r in results:
                print(f"| {r['name']:15s} | {r['eff_rank']:8.1f} | {r['gap']:3.1f} | "
                      f"{r['decay']:.3f} | {r['sparsity']:.3f} | {r['curvature']:.3f} | "
                      f"{r['id']:.1f} | {r['norm']:.0f} |")

            # Compute deltas
            print(f"\n  Key transformations:")
            for i in range(1, len(results)):
                prev, curr = results[i-1], results[i]
                if 'SiLU' in curr['name'] or 'activation' in curr['name']:
                    print(f"  {prev['name']} → {curr['name']}:")
                    print(f"    Δ Sparsity: {curr['sparsity'] - prev['sparsity']:+.3f}")
                    print(f"    Δ Curvature: {curr['curvature'] - prev['curvature']:+.3f}")
                    print(f"    Δ ID: {curr['id'] - prev['id']:+.1f}")

        del model
        mx.metal.clear_cache()

    print("\n" + "="*80)
    print("  ANALYSIS")
    print("="*80)
    print("""
Key questions:
1. Does SiLU create sparsity? (Yes if post-SiLU sparsity > pre-SiLU)
2. Does it change curvature? (Nonlinearity should add curvature)
3. Does it change ID? (Sparsity might reduce effective dimensionality)
""")


if __name__ == "__main__":
    main()
