#!/usr/bin/env python3
"""
Analyze what causes the spectral gap at highway layers.

At exit layers: gap comes from mean dominance (convergence > 1)
At highway layers: convergence < 1, so gap must come from something else.

Hypothesis: Highway spike comes from attention selectivity pattern.
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


def spectral_gap(S) -> float:
    """S₁/S₂ ratio."""
    if len(S) < 2 or S[1] < 1e-10:
        return float('inf')
    return S[0] / S[1]


def analyze_layer(h_in, h_out, layer_type=""):
    """Analyze geometry of a layer's input and output."""
    h_in_np = np.array(h_in.astype(mx.float32))[0]  # [T, C]
    h_out_np = np.array(h_out.astype(mx.float32))[0]

    results = {}
    for name, data in [("input", h_in_np), ("output", h_out_np)]:
        mean = np.mean(data, axis=0)
        centered = data - mean
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)

        mean_norm = np.linalg.norm(mean)
        dev_norm = np.mean(np.linalg.norm(centered, axis=1))
        convergence = mean_norm / dev_norm if dev_norm > 1e-10 else 0

        # Check if spike aligns with mean
        spike_dir = Vt[0]  # First right singular vector
        mean_unit = mean / np.linalg.norm(mean) if mean_norm > 1e-10 else np.zeros_like(mean)
        spike_mean_align = abs(np.dot(spike_dir, mean_unit))

        results[name] = {
            'rank': effective_rank(S[:20]),
            'gap': spectral_gap(S),
            'convergence': convergence,
            'spike_mean_align': spike_mean_align,
            'mean_norm': mean_norm,
            'S': S[:5],
        }

    return results


def analyze_attention_selectivity(model, layer_idx, h):
    """Measure attention selectivity at a layer."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    # Apply pre-norm
    normed = layer.input_layernorm(h)
    mx.eval(normed)

    B, T, C = normed.shape

    # Q, K projections
    q = attn.q_proj(normed)
    k = attn.k_proj(normed)
    mx.eval(q, k)

    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim = C // n_heads

    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    mx.eval(q, k)

    if n_kv_heads < n_heads:
        rep_factor = n_heads // n_kv_heads
        k = mx.repeat(k, rep_factor, axis=1)
        mx.eval(k)

    # Attention scores
    scale = head_dim ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
    scores = scores + mask
    attn_weights = mx.softmax(scores, axis=-1)
    mx.eval(attn_weights)

    # Analyze attention entropy and selectivity
    attn_np = np.array(attn_weights.astype(mx.float32))[0]  # [n_heads, T, T]

    # Average over heads
    avg_attn = np.mean(attn_np, axis=0)  # [T, T]

    # Entropy per row (how focused is attention)
    row_entropies = []
    for t in range(T):
        probs = avg_attn[t, :t+1]  # Only valid positions
        probs = probs[probs > 1e-10]
        if len(probs) > 0:
            H = -np.sum(probs * np.log(probs + 1e-10))
            max_H = np.log(len(probs))  # Maximum entropy
            norm_H = H / max_H if max_H > 0 else 0
            row_entropies.append(norm_H)

    avg_entropy = np.mean(row_entropies) if row_entropies else 0

    # Selectivity: how peaked is attention on a few tokens
    # Max attention weight per row
    max_weights = np.max(avg_attn, axis=1)
    selectivity = np.mean(max_weights)

    return {
        'entropy': avg_entropy,  # Low = selective, High = diffuse
        'selectivity': selectivity,  # High = peaked
    }


def main():
    print("="*60)
    print("  HIGHWAY GAP ANALYSIS")
    print("="*60)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    print(f"\nLoading {model_path.split('/')[-1]}...")
    model, tokenizer = load(model_path)
    embed = model.model.embed_tokens

    n_layers = len(model.model.layers)
    print(f"Total layers: {n_layers}")

    # Use longer prompt for better statistics
    prompt = """The quick brown fox jumps over the lazy dog. The dog was not amused by this intrusion into its personal space, but eventually decided to chase the fox anyway. They ran through the forest, past the ancient oak trees, and into a clearing where sunlight streamed through the canopy."""

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    print(f"Tokens: {len(tokens)}")

    h = embed(input_ids)
    mx.eval(h)

    # Analyze each layer
    print("\n| Layer | Gap | Convergence | Spike-Mean | Eff.Rank | Selectivity | Entropy |")
    print("|-------|-----|-------------|------------|----------|-------------|---------|")

    highway_layers = []
    exit_layers = []

    for i in range(n_layers):
        layer = model.model.layers[i]

        # Analyze layer input/output
        h_out = layer(h)
        mx.eval(h_out)

        stats = analyze_layer(h, h_out)
        attn_stats = analyze_attention_selectivity(model, i, h)

        out = stats['output']
        is_highway = out['gap'] > 2.0 and out['convergence'] < 0.5

        print(f"| {i:5d} | {out['gap']:3.1f} | {out['convergence']:.2f} | {out['spike_mean_align']:.3f} | {out['rank']:.1f} | {attn_stats['selectivity']:.3f} | {attn_stats['entropy']:.2f} |")

        if is_highway:
            highway_layers.append({
                'layer': i,
                **out,
                **attn_stats,
            })
        elif i >= n_layers - 5:  # Last 5 layers
            exit_layers.append({
                'layer': i,
                **out,
                **attn_stats,
            })

        h = h_out

    print("\n" + "="*60)
    print("  HIGHWAY vs EXIT COMPARISON")
    print("="*60)

    if highway_layers:
        print("\nHighway layers (gap > 2, convergence < 0.5):")
        for hw in highway_layers[:5]:
            print(f"  Layer {hw['layer']}: gap={hw['gap']:.1f}, conv={hw['convergence']:.2f}, spike-mean={hw['spike_mean_align']:.3f}, selectivity={hw['selectivity']:.3f}")

    if exit_layers:
        print("\nExit layers (last 5):")
        for ex in exit_layers:
            print(f"  Layer {ex['layer']}: gap={ex['gap']:.1f}, conv={ex['convergence']:.2f}, spike-mean={ex['spike_mean_align']:.3f}, selectivity={ex['selectivity']:.3f}")

    # Correlations
    print("\n--- Correlation Analysis ---")
    all_layers = highway_layers + exit_layers
    if len(all_layers) > 2:
        gaps = [l['gap'] for l in all_layers]
        convs = [l['convergence'] for l in all_layers]
        aligns = [l['spike_mean_align'] for l in all_layers]
        selects = [l['selectivity'] for l in all_layers]
        entropies = [l['entropy'] for l in all_layers]

        print(f"Gap vs Convergence: r = {np.corrcoef(gaps, convs)[0,1]:.3f}")
        print(f"Gap vs Spike-Mean alignment: r = {np.corrcoef(gaps, aligns)[0,1]:.3f}")
        print(f"Gap vs Selectivity: r = {np.corrcoef(gaps, selects)[0,1]:.3f}")
        print(f"Gap vs Entropy (inverse): r = {np.corrcoef(gaps, entropies)[0,1]:.3f}")


if __name__ == "__main__":
    main()
