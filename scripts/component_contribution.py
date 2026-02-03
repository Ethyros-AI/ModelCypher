#!/usr/bin/env python3
"""
Measure attention vs MLP contribution to output SV distribution.

Goal: Understand the 0.6 and 0.8 coefficients in decay formula.
Hypothesis: 0.8 = MLP base contribution, 0.6 = attention relative weight.
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
    """Measure decay rate of plateau (S₁₀/S₂)^(1/8)."""
    if len(S) < 10 or S[1] < 1e-10:
        return 0.0
    return (S[9] / S[1]) ** (1/8)


def spectral_gap(S) -> float:
    """S₁/S₂ ratio."""
    if len(S) < 2 or S[1] < 1e-10:
        return float('inf')
    return S[0] / S[1]


def analyze_component(name, activations):
    """Analyze SV distribution of a component's output."""
    # Flatten to [n_samples, hidden_dim]
    flat = activations.reshape(-1, activations.shape[-1])

    # Center
    mean = np.mean(flat, axis=0)
    centered = flat - mean

    # SVD
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    rank = effective_rank(S[:20])
    gap = spectral_gap(S)
    decay = plateau_decay(S)

    # Mean dominance
    mean_norm = np.linalg.norm(mean)
    dev_norm = np.mean(np.linalg.norm(centered, axis=1))
    convergence = mean_norm / dev_norm if dev_norm > 1e-10 else 0

    print(f"\n{name}:")
    print(f"  Effective rank (20): {rank:.2f}")
    print(f"  Spectral gap: {gap:.2f}")
    print(f"  Plateau decay: {decay:.3f}")
    print(f"  Convergence: {convergence:.2f}")
    print(f"  Top SVs: {S[:5]}")

    return {
        'rank': rank,
        'gap': gap,
        'decay': decay,
        'convergence': convergence,
        'S': S[:20]
    }


def main():
    print("="*60)
    print("  COMPONENT CONTRIBUTION ANALYSIS")
    print("="*60)

    models = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
    ]

    # Test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog",
        "What is the capital of France?",
        "def fibonacci(n):",
        "In the beginning was the Word",
        "The solution to x^2 - 4 = 0 is",
    ]

    for model_path in models:
        print(f"\n\n{'='*60}")
        print(f"Model: {model_path.split('/')[-1]}")
        print("="*60)

        model, tokenizer = load(model_path)
        embed = model.model.embed_tokens

        # Test on EXIT layer (where we measured decay formula)
        layer_idx = len(model.model.layers) - 2  # Second to last
        layer = model.model.layers[layer_idx]
        print(f"\nAnalyzing layer {layer_idx}")

        # Collect activations for all prompts
        all_input = []
        all_attn_out = []
        all_mlp_out = []
        all_layer_out = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            h = embed(input_ids)

            # Forward through preceding layers
            for i in range(layer_idx):
                h = model.model.layers[i](h)
            mx.eval(h)

            # Store input to target layer
            h_f32 = h.astype(mx.float32)
            mx.eval(h_f32)
            all_input.append(np.array(h_f32))

            # Attention branch (without residual)
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            attn_f32 = attn_out.astype(mx.float32)
            mx.eval(attn_f32)
            all_attn_out.append(np.array(attn_f32))

            # MLP branch (without residual, using attn output)
            h_after_attn = h + attn_out
            normed2 = layer.post_attention_layernorm(h_after_attn)
            mlp_out = layer.mlp(normed2)
            mlp_f32 = mlp_out.astype(mx.float32)
            mx.eval(mlp_f32)
            all_mlp_out.append(np.array(mlp_f32))

            # Full layer output
            layer_out = h + attn_out + mlp_out  # Full layer with residuals
            layer_f32 = layer_out.astype(mx.float32)
            mx.eval(layer_f32)
            all_layer_out.append(np.array(layer_f32))

        # Concatenate all samples
        input_acts = np.concatenate(all_input, axis=1)[0]  # [total_tokens, hidden]
        attn_acts = np.concatenate(all_attn_out, axis=1)[0]
        mlp_acts = np.concatenate(all_mlp_out, axis=1)[0]
        layer_acts = np.concatenate(all_layer_out, axis=1)[0]

        print(f"\nTotal tokens: {input_acts.shape[0]}")

        # Analyze each component
        input_stats = analyze_component("Input to layer", input_acts)
        attn_stats = analyze_component("Attention output (no residual)", attn_acts)
        mlp_stats = analyze_component("MLP output (no residual)", mlp_acts)
        layer_stats = analyze_component("Full layer output", layer_acts)

        # Analyze relative contributions
        print("\n--- Contribution Analysis ---")

        # Norm ratios
        attn_norm = np.mean(np.linalg.norm(attn_acts, axis=1))
        mlp_norm = np.mean(np.linalg.norm(mlp_acts, axis=1))
        input_norm = np.mean(np.linalg.norm(input_acts, axis=1))

        print(f"Norm ratios (relative to input):")
        print(f"  Attention: {attn_norm/input_norm:.3f}")
        print(f"  MLP: {mlp_norm/input_norm:.3f}")

        # Effective rank ratios
        print(f"\nEffective rank (normalized by full rank):")
        d = input_acts.shape[1]
        print(f"  Input: {input_stats['rank']/d:.3f}")
        print(f"  Attention: {attn_stats['rank']/d:.3f}")
        print(f"  MLP: {mlp_stats['rank']/d:.3f}")
        print(f"  Layer output: {layer_stats['rank']/d:.3f}")

        # Decay comparison
        print(f"\nDecay rates:")
        print(f"  Input: {input_stats['decay']:.3f}")
        print(f"  Attention: {attn_stats['decay']:.3f}")
        print(f"  MLP: {mlp_stats['decay']:.3f}")
        print(f"  Layer output: {layer_stats['decay']:.3f}")

        # Can we predict layer decay from component decays?
        # Simple weighted average hypothesis
        alpha = attn_norm / (attn_norm + mlp_norm + input_norm)
        beta = mlp_norm / (attn_norm + mlp_norm + input_norm)
        gamma = input_norm / (attn_norm + mlp_norm + input_norm)

        predicted_decay = (alpha * attn_stats['decay'] +
                          beta * mlp_stats['decay'] +
                          gamma * input_stats['decay'])

        print(f"\nDecay prediction:")
        print(f"  Weights: attn={alpha:.3f}, mlp={beta:.3f}, input={gamma:.3f}")
        print(f"  Predicted (weighted avg): {predicted_decay:.3f}")
        print(f"  Actual layer decay: {layer_stats['decay']:.3f}")
        print(f"  Error: {abs(predicted_decay - layer_stats['decay']):.3f}")


if __name__ == "__main__":
    main()
