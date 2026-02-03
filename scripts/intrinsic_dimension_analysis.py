#!/usr/bin/env python3
"""
Analyze what determines intrinsic dimension (ID) trajectory.

ID measures LOCAL manifold dimensionality - how many coordinates needed locally.
Effective rank measures GLOBAL variance distribution.

Question: What determines ID trajectory (48D → 2D → 6D in Qwen3)?

Hypotheses:
1. Attention entropy → selectivity compresses the manifold
2. Activation sparsity → fewer active dimensions locally
3. Local curvature → curved manifolds can have low ID
4. Nearest neighbor distances → relates to local density
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.spatial.distance import pdist, squareform


def estimate_id_mle(X, k=5):
    """
    Estimate intrinsic dimension using MLE (Levina-Bickel).

    ID = 1 / mean(log(r_k / r_1)) for each point
    """
    n, d = X.shape
    if n < k + 1:
        return 0.0

    # Compute pairwise distances
    dists = squareform(pdist(X))

    # For each point, get k nearest neighbors (excluding self)
    ids = []
    for i in range(n):
        d_i = np.sort(dists[i])[1:k+1]  # Exclude self (distance 0)
        if d_i[0] > 1e-10:  # Avoid log(0)
            # MLE estimate
            log_ratios = np.log(d_i[-1] / d_i[:-1])
            if np.mean(log_ratios) > 1e-10:
                id_i = (k - 1) / np.sum(log_ratios)
                ids.append(id_i)

    return np.mean(ids) if ids else 0.0


def estimate_id_twonn(X):
    """
    Estimate ID using TwoNN method (Facco et al. 2017).

    Uses ratio μ = r2/r1 for each point.
    ID = n / sum(log(μ_i))
    """
    n, d = X.shape
    if n < 3:
        return 0.0

    dists = squareform(pdist(X))

    mus = []
    for i in range(n):
        d_sorted = np.sort(dists[i])[1:3]  # r1, r2 (excluding self)
        if d_sorted[0] > 1e-10:
            mu = d_sorted[1] / d_sorted[0]
            mus.append(mu)

    if not mus:
        return 0.0

    # MLE for Pareto distribution
    log_mus = np.log(np.array(mus))
    id_est = len(mus) / np.sum(log_mus)

    return id_est


def measure_attention_entropy(model, layer_idx, h):
    """Measure attention entropy at a layer."""
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

    # Compute entropy per position, average over heads
    attn_np = np.array(attn_weights.astype(mx.float32))[0]  # [n_heads, T, T]

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


def measure_activation_sparsity(h):
    """Measure activation sparsity (fraction near zero)."""
    h_np = np.array(h.astype(mx.float32)).reshape(-1)
    threshold = 0.01 * np.std(h_np)
    sparsity = np.mean(np.abs(h_np) < threshold)
    return sparsity


def measure_local_curvature(X, k=10):
    """
    Estimate local curvature via PCA residual in neighborhoods.
    High residual = high curvature.
    """
    n, d = X.shape
    if n < k + 1:
        return 0.0

    dists = squareform(pdist(X))

    curvatures = []
    for i in range(min(n, 100)):  # Sample for speed
        # Get k nearest neighbors
        nn_idx = np.argsort(dists[i])[1:k+1]
        neighbors = X[nn_idx]

        # Local PCA
        centered = neighbors - np.mean(neighbors, axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)

        # Curvature ~ how much variance is NOT in top dimensions
        if len(S) > 2:
            top2_var = (S[0]**2 + S[1]**2) / np.sum(S**2)
            curvature = 1 - top2_var  # High if manifold curves out of 2D
            curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0


def main():
    print("="*70)
    print("  INTRINSIC DIMENSION ANALYSIS")
    print("="*70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    print(f"\nLoading {model_path.split('/')[-1]}...")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    embed = model.model.embed_tokens

    # Use multiple prompts for better statistics
    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 3,
        "What is the meaning of life, the universe, and everything? " * 2,
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 2,
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole. " * 2,
    ]

    # Collect activations for all prompts
    print("\nCollecting activations...")
    layer_activations = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = embed(input_ids)
        mx.eval(h)

        for i, layer in enumerate(model.model.layers):
            h = layer(h)
            mx.eval(h)
            h_np = np.array(h.astype(mx.float32))[0]  # [T, C]
            layer_activations[i].append(h_np)

    # Analyze each layer
    print("\n| Layer | ID (MLE) | ID (TwoNN) | Attn Entropy | Sparsity | Curvature |")
    print("|-------|----------|------------|--------------|----------|-----------|")

    results = []

    for i in range(n_layers):
        # Concatenate all activations for this layer
        acts = np.concatenate(layer_activations[i], axis=0)

        # Subsample if too many points (for speed)
        if len(acts) > 200:
            idx = np.random.choice(len(acts), 200, replace=False)
            acts_sample = acts[idx]
        else:
            acts_sample = acts

        # Measure ID
        id_mle = estimate_id_mle(acts_sample, k=5)
        id_twonn = estimate_id_twonn(acts_sample)

        # Measure potential causes
        # Need to re-forward for attention entropy
        h = embed(mx.array([tokenizer.encode(prompts[0])]))
        for j in range(i):
            h = model.model.layers[j](h)
        mx.eval(h)

        attn_entropy = measure_attention_entropy(model, i, h)
        sparsity = measure_activation_sparsity(mx.array(acts_sample))
        curvature = measure_local_curvature(acts_sample)

        print(f"| {i:5d} | {id_mle:8.1f} | {id_twonn:10.1f} | {attn_entropy:12.3f} | {sparsity:8.3f} | {curvature:9.3f} |")

        results.append({
            'layer': i,
            'id_mle': id_mle,
            'id_twonn': id_twonn,
            'attn_entropy': attn_entropy,
            'sparsity': sparsity,
            'curvature': curvature,
        })

    # Correlation analysis
    print("\n" + "="*70)
    print("  CORRELATION ANALYSIS")
    print("="*70)

    ids = [r['id_mle'] for r in results]
    entropies = [r['attn_entropy'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    curvatures = [r['curvature'] for r in results]

    print(f"\nCorrelations with ID (MLE):")
    print(f"  Attention entropy: r = {np.corrcoef(ids, entropies)[0,1]:.3f}")
    print(f"  Sparsity: r = {np.corrcoef(ids, sparsities)[0,1]:.3f}")
    print(f"  Curvature: r = {np.corrcoef(ids, curvatures)[0,1]:.3f}")

    # Find highway (minimum ID region)
    min_id_layer = np.argmin(ids)
    print(f"\nHighway at layer {min_id_layer} with ID = {ids[min_id_layer]:.1f}")
    print(f"  Attention entropy: {entropies[min_id_layer]:.3f}")
    print(f"  Sparsity: {sparsities[min_id_layer]:.3f}")
    print(f"  Curvature: {curvatures[min_id_layer]:.3f}")


if __name__ == "__main__":
    main()
