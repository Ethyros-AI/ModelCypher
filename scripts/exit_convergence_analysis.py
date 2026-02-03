#!/usr/bin/env python3
"""
Analyze what determines exit layer convergence.

Convergence = ‖μ‖/‖x-μ‖ (mean_norm / deviation_norm)

Hypothesis: Training for diverse outputs reduces convergence.

Questions:
1. What is the exit mean direction?
2. Does it align with unembedding structure?
3. How does training type affect this?
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load


def analyze_exit_geometry(model, tokenizer, prompts):
    """Analyze exit layer geometry."""
    embed = model.model.embed_tokens
    n_layers = len(model.model.layers)

    all_exit_acts = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = embed(input_ids)

        # Forward through all layers
        for layer in model.model.layers:
            h = layer(h)
        mx.eval(h)

        # Apply final norm if exists
        if hasattr(model.model, 'norm'):
            h = model.model.norm(h)
            mx.eval(h)

        h_np = np.array(h.astype(mx.float32))[0]  # [T, C]
        all_exit_acts.append(h_np)

    # Concatenate all tokens
    exit_acts = np.concatenate(all_exit_acts, axis=0)  # [total_tokens, C]

    # Compute statistics
    mean = np.mean(exit_acts, axis=0)
    centered = exit_acts - mean

    mean_norm = np.linalg.norm(mean)
    dev_norm = np.mean(np.linalg.norm(centered, axis=1))
    convergence = mean_norm / dev_norm

    # SVD of centered activations
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Check if mean aligns with top singular vector
    mean_unit = mean / mean_norm if mean_norm > 1e-10 else np.zeros_like(mean)
    spike_dir = Vt[0]
    mean_spike_align = abs(np.dot(mean_unit, spike_dir))

    return {
        'convergence': convergence,
        'mean_norm': mean_norm,
        'dev_norm': dev_norm,
        'mean_spike_align': mean_spike_align,
        'mean': mean,
        'S': S[:10],
        'Vt': Vt[:5],
    }


def analyze_unembedding(model):
    """Analyze unembedding matrix structure."""
    # Get unembedding (lm_head) weights
    if hasattr(model, 'lm_head'):
        W = model.lm_head.weight  # [vocab_size, hidden_dim]
    else:
        # Tied embeddings
        W = model.model.embed_tokens.weight

    W_np = np.array(W.astype(mx.float32))

    # Mean unembedding direction
    unembed_mean = np.mean(W_np, axis=0)
    unembed_mean_norm = np.linalg.norm(unembed_mean)
    unembed_mean_unit = unembed_mean / unembed_mean_norm if unembed_mean_norm > 1e-10 else np.zeros_like(unembed_mean)

    # SVD of unembedding
    _, S_unembed, Vt_unembed = np.linalg.svd(W_np - unembed_mean, full_matrices=False)

    return {
        'mean': unembed_mean,
        'mean_norm': unembed_mean_norm,
        'mean_unit': unembed_mean_unit,
        'S': S_unembed[:10],
        'Vt': Vt_unembed[:5],
    }


def main():
    print("="*60)
    print("  EXIT CONVERGENCE ANALYSIS")
    print("="*60)

    # Models with different training types
    models = [
        ("/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16", "base"),
        ("/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16", "instruct"),
        ("/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16", "reasoning"),
    ]

    # Diverse prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "In the beginning was the Word, and the Word was with God.",
        "The solution to x^2 - 4 = 0 is x = 2 or x = -2.",
        "Once upon a time in a land far away, there lived a princess.",
        "SELECT * FROM users WHERE age > 18;",
        "The mitochondria is the powerhouse of the cell.",
    ]

    results = []

    for model_path, train_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_path.split('/')[-1]} ({train_type})")
        print("="*60)

        model, tokenizer = load(model_path)

        # Analyze exit geometry
        exit_stats = analyze_exit_geometry(model, tokenizer, prompts)

        # Analyze unembedding
        unembed_stats = analyze_unembedding(model)

        # Check alignment between exit mean and unembedding mean
        exit_mean_unit = exit_stats['mean'] / exit_stats['mean_norm']
        mean_align = abs(np.dot(exit_mean_unit, unembed_stats['mean_unit']))

        # Check alignment between exit mean and top unembedding directions
        unembed_top_aligns = [abs(np.dot(exit_mean_unit, unembed_stats['Vt'][i]))
                              for i in range(5)]

        print(f"\n--- Exit Geometry ---")
        print(f"Convergence: {exit_stats['convergence']:.1f}")
        print(f"Mean norm: {exit_stats['mean_norm']:.1f}")
        print(f"Deviation norm: {exit_stats['dev_norm']:.1f}")
        print(f"Mean-spike alignment: {exit_stats['mean_spike_align']:.3f}")

        print(f"\n--- Unembedding Analysis ---")
        print(f"Unembed mean norm: {unembed_stats['mean_norm']:.1f}")
        print(f"Exit mean ↔ Unembed mean alignment: {mean_align:.3f}")
        print(f"Exit mean ↔ Unembed top SVs: {unembed_top_aligns}")

        # Check: does exit mean point toward common tokens?
        W = model.lm_head.weight if hasattr(model, 'lm_head') else model.model.embed_tokens.weight
        W_np = np.array(W.astype(mx.float32))

        # Dot product of exit mean with each token embedding
        logits_from_mean = W_np @ exit_stats['mean']
        top_token_indices = np.argsort(logits_from_mean)[-10:][::-1]

        print(f"\n--- Exit Mean Points Toward ---")
        for idx in top_token_indices[:5]:
            token = tokenizer.decode([int(idx)])
            logit = logits_from_mean[idx]
            print(f"  '{repr(token)}': {logit:.1f}")

        results.append({
            'model': model_path.split('/')[-1],
            'train_type': train_type,
            'convergence': exit_stats['convergence'],
            'mean_norm': exit_stats['mean_norm'],
            'dev_norm': exit_stats['dev_norm'],
            'mean_unembed_align': mean_align,
        })

        # Clean up
        del model
        mx.metal.clear_cache()

    # Summary
    print("\n\n" + "="*60)
    print("  SUMMARY: Training Type vs Exit Convergence")
    print("="*60)

    print("\n| Model | Type | Convergence | Mean Norm | Dev Norm | Mean↔Unembed |")
    print("|-------|------|-------------|-----------|----------|--------------|")
    for r in results:
        print(f"| {r['model'][:25]:25s} | {r['train_type']:8s} | {r['convergence']:6.1f} | {r['mean_norm']:6.1f} | {r['dev_norm']:5.2f} | {r['mean_unembed_align']:.3f} |")

    # Analysis
    print("\n--- Analysis ---")
    convs = [r['convergence'] for r in results]
    mean_norms = [r['mean_norm'] for r in results]
    dev_norms = [r['dev_norm'] for r in results]

    print(f"Convergence range: {min(convs):.1f} - {max(convs):.1f}")
    print(f"Mean norm range: {min(mean_norms):.1f} - {max(mean_norms):.1f}")
    print(f"Deviation norm range: {min(dev_norms):.2f} - {max(dev_norms):.2f}")

    # What changes more - mean norm or deviation norm?
    mean_ratio = max(mean_norms) / min(mean_norms)
    dev_ratio = max(dev_norms) / min(dev_norms)
    print(f"\nMean norm ratio (max/min): {mean_ratio:.2f}×")
    print(f"Deviation norm ratio (max/min): {dev_ratio:.2f}×")

    if mean_ratio > dev_ratio:
        print("→ Convergence is primarily driven by MEAN NORM changes")
    else:
        print("→ Convergence is primarily driven by DEVIATION NORM changes")


if __name__ == "__main__":
    main()
