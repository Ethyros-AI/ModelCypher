#!/usr/bin/env python3
"""
Test causal link: Attention entropy → Curvature delta → ID

Finding: Attention adds curvature early, removes it late.
Hypothesis: High entropy attention (diffuse) adds curvature,
            Low entropy attention (selective) removes curvature.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.spatial.distance import pdist, squareform


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


def measure_attention_entropy(model, layer_idx, h):
    """Measure normalized attention entropy."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(h)
    B, T, C = normed.shape

    q = attn.q_proj(normed)
    k = attn.k_proj(normed)
    mx.eval(q, k)

    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim = C // n_heads

    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    if n_kv_heads < n_heads:
        k = mx.repeat(k, n_heads // n_kv_heads, axis=1)
    mx.eval(k)

    scale = head_dim ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
    scores = scores + mask
    attn_weights = mx.softmax(scores, axis=-1)
    mx.eval(attn_weights)

    attn_np = np.array(attn_weights.astype(mx.float32))[0]
    entropies = []
    for h_idx in range(attn_np.shape[0]):
        for t in range(T):
            probs = attn_np[h_idx, t, :t+1]
            probs = probs[probs > 1e-10]
            if len(probs) > 1:
                H = -np.sum(probs * np.log(probs))
                max_H = np.log(len(probs))
                entropies.append(H / max_H if max_H > 0 else 0)
    return np.mean(entropies) if entropies else 0


def measure_attention_curvature_delta(model, layer_idx, h):
    """Measure curvature change from attention."""
    layer = model.model.layers[layer_idx]

    # Before attention (after input norm)
    normed = layer.input_layernorm(h)
    mx.eval(normed)
    before_np = np.array(normed.astype(mx.float32))[0]
    curv_before = measure_local_curvature(before_np)

    # After attention (no residual)
    attn_out = layer.self_attn(normed)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    mx.eval(attn_out)
    after_np = np.array(attn_out.astype(mx.float32))[0]
    curv_after = measure_local_curvature(after_np)

    return curv_after - curv_before


def main():
    print("="*70)
    print("  ENTROPY → CURVATURE LINK ANALYSIS")
    print("="*70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    print(f"\nLoading {model_path.split('/')[-1]}...")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    embed = model.model.embed_tokens

    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    print(f"Tokens: {len(tokens)}")

    h = embed(input_ids)
    mx.eval(h)

    print("\n| Layer | Attn Entropy | Δ Curvature | Relationship |")
    print("|-------|--------------|-------------|--------------|")

    entropies = []
    deltas = []

    for layer_idx in range(n_layers):
        entropy = measure_attention_entropy(model, layer_idx, h)
        delta = measure_attention_curvature_delta(model, layer_idx, h)

        relationship = "ADDS" if delta > 0.02 else "REMOVES" if delta < -0.02 else "~neutral"

        print(f"| {layer_idx:5d} | {entropy:12.3f} | {delta:+11.3f} | {relationship:12s} |")

        entropies.append(entropy)
        deltas.append(delta)

        # Forward
        h = model.model.layers[layer_idx](h)
        mx.eval(h)

    # Correlation
    print("\n" + "="*70)
    print("  CORRELATION: Entropy vs Curvature Delta")
    print("="*70)

    corr = np.corrcoef(entropies, deltas)[0, 1]
    print(f"\nr(entropy, Δcurvature) = {corr:.3f}")

    if corr > 0.5:
        print("→ HIGH entropy attention ADDS curvature")
        print("→ LOW entropy attention REMOVES curvature")
    elif corr < -0.5:
        print("→ HIGH entropy attention REMOVES curvature")
        print("→ LOW entropy attention ADDS curvature")
    else:
        print("→ Weak or no direct relationship")

    # Scatter plot summary
    print("\n--- Scatter Summary ---")
    high_entropy = [(e, d) for e, d in zip(entropies, deltas) if e > 0.8]
    low_entropy = [(e, d) for e, d in zip(entropies, deltas) if e < 0.3]

    if high_entropy:
        avg_delta_high = np.mean([d for _, d in high_entropy])
        print(f"High entropy (>0.8): avg Δcurv = {avg_delta_high:+.3f}")

    if low_entropy:
        avg_delta_low = np.mean([d for _, d in low_entropy])
        print(f"Low entropy (<0.3): avg Δcurv = {avg_delta_low:+.3f}")


if __name__ == "__main__":
    main()
