#!/usr/bin/env python3
"""
Analyze what determines attention output decay.

Hypothesis: attn_out_rank ≤ min(attn_pattern_rank, V_rank)
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


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
    """Measure decay rate of plateau (S_n/S₂)^(1/(n-2)) for available n."""
    if len(S) < 3 or S[1] < 1e-10:
        return 0.0
    # Use last available SV for decay measurement
    n = min(len(S), 10)
    if S[n-1] < 1e-10:
        return 0.0
    return (S[n-1] / S[1]) ** (1/(n-2))


def analyze_attention_mechanism(model, layer_idx, h):
    """Analyze attention components and their ranks."""
    layer = model.model.layers[layer_idx]

    # Apply pre-norm
    normed = layer.input_layernorm(h)
    mx.eval(normed)

    # Get attention weights
    attn = layer.self_attn
    B, T, C = normed.shape

    # Q, K, V projections
    q = attn.q_proj(normed)
    k = attn.k_proj(normed)
    v = attn.v_proj(normed)
    mx.eval(q, k, v)

    # Analyze V projection matrix rank
    W_v = attn.v_proj.weight  # [out_features, in_features]
    W_v_np = np.array(W_v.astype(mx.float32))
    _, S_v, _ = np.linalg.svd(W_v_np, full_matrices=False)
    V_rank = effective_rank(S_v)
    V_decay = plateau_decay(S_v)

    # Reshape for attention
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim = C // n_heads

    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)  # [B, n_heads, T, head_dim]
    k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    mx.eval(q, k, v)

    # GQA: repeat K, V to match Q heads
    if n_kv_heads < n_heads:
        rep_factor = n_heads // n_kv_heads
        k = mx.repeat(k, rep_factor, axis=1)
        v = mx.repeat(v, rep_factor, axis=1)
        mx.eval(k, v)

    # Compute attention scores
    scale = head_dim ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, n_heads, T, T]

    # Apply causal mask
    mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
    scores = scores + mask

    # Softmax
    attn_weights = mx.softmax(scores, axis=-1)  # [B, n_heads, T, T]
    mx.eval(attn_weights)

    # Analyze attention pattern rank (per head, then average)
    attn_ranks = []
    attn_decays = []
    for head in range(n_heads):
        attn_h = np.array(attn_weights[0, head].astype(mx.float32))  # [T, T]
        _, S_attn, _ = np.linalg.svd(attn_h, full_matrices=False)
        attn_ranks.append(effective_rank(S_attn))
        attn_decays.append(plateau_decay(S_attn))

    avg_attn_rank = np.mean(attn_ranks)
    avg_attn_decay = np.mean(attn_decays)

    # Compute attention output
    attn_out = attn_weights @ v  # [B, n_heads, T, head_dim]
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)  # [B, T, C]
    attn_out = attn.o_proj(attn_out)
    mx.eval(attn_out)

    # Analyze attention output rank
    attn_out_np = np.array(attn_out.astype(mx.float32))[0]  # [T, C]
    mean = np.mean(attn_out_np, axis=0)
    centered = attn_out_np - mean
    _, S_out, _ = np.linalg.svd(centered, full_matrices=False)
    out_rank = effective_rank(S_out[:20])
    out_decay = plateau_decay(S_out)

    return {
        'V_rank': V_rank,
        'V_decay': V_decay,
        'attn_pattern_rank': avg_attn_rank,
        'attn_pattern_decay': avg_attn_decay,
        'attn_out_rank': out_rank,
        'attn_out_decay': out_decay,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'GQA': n_heads / n_kv_heads,
        'd_model': C,
    }


def main():
    print("="*60)
    print("  ATTENTION DECAY ANALYSIS")
    print("="*60)

    models = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/granite-8b-code-instruct-128k-mlx",
        "/Volumes/CodeCypher/models/mlx-community/Llama-3.2-3B-Instruct-bf16",
    ]

    # Use longer prompts for better statistics
    prompts = [
        "The quick brown fox jumps over the lazy dog. The dog was not amused by this intrusion into its personal space, but eventually decided to chase the fox anyway.",
        "What is the capital of France? Paris is known for its beautiful architecture, including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The city attracts millions of tourists every year.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Calculate the first 20 Fibonacci numbers\nfor i in range(20):\n    print(fibonacci(i))",
        "In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through him, and without him was not anything made that was made.",
        "The solution to x^2 - 4 = 0 is found by factoring the left side as (x+2)(x-2) = 0, which gives us x = 2 or x = -2. We can verify by substituting back into the original equation.",
    ]

    results = []

    for model_path in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_path.split('/')[-1]}")
        print("="*60)

        model, tokenizer = load(model_path)
        embed = model.model.embed_tokens

        # Test at exit layer
        layer_idx = len(model.model.layers) - 2
        print(f"Analyzing layer {layer_idx}")

        all_results = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            h = embed(input_ids)

            # Forward through preceding layers
            for i in range(layer_idx):
                h = model.model.layers[i](h)
            mx.eval(h)

            r = analyze_attention_mechanism(model, layer_idx, h)
            all_results.append(r)

        # Average results
        avg = {}
        for key in all_results[0].keys():
            val = all_results[0][key]
            if isinstance(val, (int, float, np.floating, np.integer)):
                avg[key] = float(np.mean([r[key] for r in all_results]))

        print(f"\n--- Results (averaged over {len(prompts)} prompts) ---")
        print(f"V projection rank: {avg['V_rank']:.1f} (of {avg['d_model']:.0f})")
        print(f"V projection decay: {avg['V_decay']:.3f}")
        print(f"Attention pattern rank: {avg['attn_pattern_rank']:.2f} (avg over {avg['n_heads']:.0f} heads)")
        print(f"Attention pattern decay: {avg['attn_pattern_decay']:.3f}")
        print(f"Attention output rank: {avg['attn_out_rank']:.1f}")
        print(f"Attention output decay: {avg['attn_out_decay']:.3f}")
        print(f"GQA ratio: {avg['GQA']:.1f}")

        # Check bound
        min_bound = min(avg['V_rank'], avg['attn_pattern_rank'] * avg['n_heads'])
        print(f"\nBound check: attn_out_rank ({avg['attn_out_rank']:.1f}) ≤ min(V_rank={avg['V_rank']:.1f}, pattern_rank×heads={avg['attn_pattern_rank']:.1f}×{avg['n_heads']:.0f})")

        # Check relationship
        V_rank_norm = avg['V_rank'] / avg['d_model']
        print(f"V_rank/d_model: {V_rank_norm:.3f}")

        results.append({
            'model': model_path.split('/')[-1],
            'V_rank_norm': V_rank_norm,
            'V_decay': avg['V_decay'],
            'attn_pattern_decay': avg['attn_pattern_decay'],
            'attn_out_decay': avg['attn_out_decay'],
            'GQA': avg['GQA'],
        })

    # Cross-model analysis
    print("\n\n" + "="*60)
    print("  CROSS-MODEL ANALYSIS")
    print("="*60)

    print("\n| Model | V_rank/d | V_decay | Pattern_decay | Out_decay | GQA |")
    print("|-------|----------|---------|---------------|-----------|-----|")
    for r in results:
        print(f"| {r['model'][:20]:20s} | {r['V_rank_norm']:.3f} | {r['V_decay']:.3f} | {r['attn_pattern_decay']:.3f} | {r['attn_out_decay']:.3f} | {r['GQA']:.1f} |")

    # Correlations
    print("\nCorrelations with attn_out_decay:")
    out_decays = [r['attn_out_decay'] for r in results]
    for key in ['V_rank_norm', 'V_decay', 'attn_pattern_decay', 'GQA']:
        vals = [r[key] for r in results]
        if len(set(vals)) > 1:  # Need variance for correlation
            corr = np.corrcoef(vals, out_decays)[0, 1]
            print(f"  {key}: r = {corr:.3f}")


if __name__ == "__main__":
    main()
