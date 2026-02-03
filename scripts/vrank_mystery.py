#!/usr/bin/env python3
"""
Investigate the V_rank mystery.

Earlier finding: V_rank correlates with layer decay (r=0.73)
But: Attention output decay is constant (~0.89-0.91) across models

Question: Where does the V_rank correlation come from?

Hypotheses:
1. V_rank affects attention output NORM (not decay) → changes mixing weights
2. V_rank affects something in the recursive input decay chain
3. The original correlation was spurious (only 4 data points)
4. V_rank correlates with something else that affects layer decay
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
    """Measure decay rate of plateau."""
    if len(S) < 10 or S[1] < 1e-10:
        return 0.0
    return (S[9] / S[1]) ** (1/8)


def analyze_v_projection(model, layer_idx):
    """Analyze the V projection matrix."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    W_v = attn.v_proj.weight  # [out_features, in_features]
    W_v_np = np.array(W_v.astype(mx.float32))

    _, S, _ = np.linalg.svd(W_v_np, full_matrices=False)

    d_model = W_v_np.shape[1]
    v_rank = effective_rank(S)
    v_rank_norm = v_rank / d_model

    return {
        'v_rank': v_rank,
        'v_rank_norm': v_rank_norm,
        'd_model': d_model,
    }


def analyze_attention_output(model, layer_idx, h):
    """Analyze attention output statistics."""
    layer = model.model.layers[layer_idx]

    # Get attention output
    normed = layer.input_layernorm(h)
    attn_out = layer.self_attn(normed)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    mx.eval(attn_out)

    attn_np = np.array(attn_out.astype(mx.float32))[0]  # [T, C]

    # Compute statistics
    mean = np.mean(attn_np, axis=0)
    centered = attn_np - mean
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    decay = plateau_decay(S)
    rank = effective_rank(S[:20])

    # Output norm
    output_norm = np.mean(np.linalg.norm(attn_np, axis=1))

    return {
        'attn_decay': decay,
        'attn_rank': rank,
        'attn_norm': output_norm,
    }


def analyze_layer_output(model, layer_idx, h):
    """Analyze full layer output statistics."""
    layer = model.model.layers[layer_idx]

    layer_out = layer(h)
    mx.eval(layer_out)

    layer_np = np.array(layer_out.astype(mx.float32))[0]

    mean = np.mean(layer_np, axis=0)
    centered = layer_np - mean
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    decay = plateau_decay(S)
    rank = effective_rank(S[:20])

    return {
        'layer_decay': decay,
        'layer_rank': rank,
    }


def main():
    print("="*70)
    print("  V_RANK MYSTERY INVESTIGATION")
    print("="*70)

    models = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/granite-8b-code-instruct-128k-mlx",
        "/Volumes/CodeCypher/models/mlx-community/Llama-3.2-3B-Instruct-bf16",
    ]

    prompt = "The quick brown fox jumps over the lazy dog. " * 3

    results = []

    for model_path in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_path.split('/')[-1]}")
        print("="*70)

        model, tokenizer = load(model_path)
        embed = model.model.embed_tokens
        n_layers = len(model.model.layers)

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Analyze at exit layer
        layer_idx = n_layers - 2

        # Forward to get input to target layer
        h = embed(input_ids)
        for i in range(layer_idx):
            h = model.model.layers[i](h)
        mx.eval(h)

        # Analyze V projection
        v_stats = analyze_v_projection(model, layer_idx)

        # Analyze attention output
        attn_stats = analyze_attention_output(model, layer_idx, h)

        # Analyze layer output
        layer_stats = analyze_layer_output(model, layer_idx, h)

        print(f"\nLayer {layer_idx} analysis:")
        print(f"  V_rank/d: {v_stats['v_rank_norm']:.3f}")
        print(f"  Attn output decay: {attn_stats['attn_decay']:.3f}")
        print(f"  Attn output norm: {attn_stats['attn_norm']:.1f}")
        print(f"  Layer output decay: {layer_stats['layer_decay']:.3f}")

        results.append({
            'model': model_path.split('/')[-1],
            'v_rank_norm': v_stats['v_rank_norm'],
            'attn_decay': attn_stats['attn_decay'],
            'attn_norm': attn_stats['attn_norm'],
            'layer_decay': layer_stats['layer_decay'],
        })

        del model
        mx.metal.clear_cache()

    # Correlation analysis
    print("\n" + "="*70)
    print("  CORRELATION ANALYSIS")
    print("="*70)

    v_ranks = [r['v_rank_norm'] for r in results]
    attn_decays = [r['attn_decay'] for r in results]
    attn_norms = [r['attn_norm'] for r in results]
    layer_decays = [r['layer_decay'] for r in results]

    print(f"\n| Model | V_rank/d | Attn Decay | Attn Norm | Layer Decay |")
    print(f"|-------|----------|------------|-----------|-------------|")
    for r in results:
        print(f"| {r['model'][:25]:25s} | {r['v_rank_norm']:.3f} | {r['attn_decay']:.3f} | {r['attn_norm']:7.1f} | {r['layer_decay']:.3f} |")

    print(f"\nCorrelations with V_rank/d:")
    print(f"  vs Attn decay: r = {np.corrcoef(v_ranks, attn_decays)[0,1]:.3f}")
    print(f"  vs Attn norm: r = {np.corrcoef(v_ranks, attn_norms)[0,1]:.3f}")
    print(f"  vs Layer decay: r = {np.corrcoef(v_ranks, layer_decays)[0,1]:.3f}")

    print("\n" + "="*70)
    print("  HYPOTHESIS TESTING")
    print("="*70)

    # Check if V_rank affects attention norm
    r_v_norm = np.corrcoef(v_ranks, attn_norms)[0,1]
    if abs(r_v_norm) > 0.5:
        print(f"\n✓ V_rank correlates with attention NORM (r={r_v_norm:.3f})")
        print("  → V_rank affects mixing weights (attention contribution)")
    else:
        print(f"\n✗ V_rank does NOT correlate with attention norm (r={r_v_norm:.3f})")

    # Check if V_rank affects layer decay differently than attn decay
    r_v_attn = np.corrcoef(v_ranks, attn_decays)[0,1]
    r_v_layer = np.corrcoef(v_ranks, layer_decays)[0,1]

    if abs(r_v_layer) > abs(r_v_attn) + 0.2:
        print(f"\n✓ V_rank correlates more with layer decay ({r_v_layer:.3f}) than attn decay ({r_v_attn:.3f})")
        print("  → V_rank effect is mediated by something other than attention")
    else:
        print(f"\n? V_rank correlations similar: layer={r_v_layer:.3f}, attn={r_v_attn:.3f}")

    # The paradox: V_rank correlates NEGATIVELY with attn_decay but POSITIVELY with layer_decay
    print("\n" + "="*70)
    print("  THE PARADOX")
    print("="*70)
    print(f"""
V_rank has OPPOSITE correlations:
  - Higher V_rank → LOWER attn_decay (r={r_v_attn:.3f})
  - Higher V_rank → HIGHER layer_decay (r={r_v_layer:.3f})

How can this be? Layer decay should be a weighted average including attn_decay.

Possible explanations:
1. MLP decay or input decay also correlates with V_rank (opposite direction)
2. Mixing weights change to compensate
3. With only {len(v_ranks)} data points, correlations are unstable

Ranges observed:
  - Attention decay: {min(attn_decays):.3f} - {max(attn_decays):.3f} (Δ={max(attn_decays)-min(attn_decays):.3f})
  - Layer decay: {min(layer_decays):.3f} - {max(layer_decays):.3f} (Δ={max(layer_decays)-min(layer_decays):.3f})

Both ranges are VERY small (~0.01-0.02), so the "correlations" may be
driven by noise rather than real effects.
""")


if __name__ == "__main__":
    main()
