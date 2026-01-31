#!/usr/bin/env python3
"""Analyze attention entropy trajectory through layers.

Hypothesis: Expansion correlates with attention entropy.
- High entropy = diffuse attention = exploring many paths
- Low entropy = focused attention = converging on answer

If true, the compression gate might be an "attention focusing" mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Task probes (same as trajectory script)
TASK_PROBES = {
    "retrieval": "What is the capital of France?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
}


def compute_attention_entropy(attn_weights) -> float:
    """Compute entropy of attention distribution.

    attn_weights: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
    Returns average entropy across heads and positions.
    """
    import mlx.core as mx

    # Ensure 3D
    if len(attn_weights.shape) == 2:
        attn_weights = attn_weights[None, :, :]

    # Clamp for numerical stability
    eps = 1e-10
    attn_weights = mx.clip(attn_weights, eps, 1.0)

    # Entropy per position: -sum(p * log(p))
    entropy = -mx.sum(attn_weights * mx.log(attn_weights), axis=-1)

    # Average across heads and positions
    return float(mx.mean(entropy).item())


def trace_attention_entropy(model, tokenizer, prompt: str) -> list[float]:
    """Trace attention entropy through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    base = getattr(model, "model", model)

    # Embedding
    hidden = base.embed_tokens(input_ids)
    mx.eval(hidden)

    entropies = []

    # Create causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)

    for layer in base.layers:
        # We need to extract attention weights
        # This requires modifying forward pass or using hooks
        # For now, approximate by computing attention manually

        # Get attention components
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
        elif hasattr(layer, 'attention'):
            attn = layer.attention
        else:
            # Skip if no attention found
            entropies.append(0.0)
            hidden = layer(hidden, mask=mask, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            continue

        # Compute Q, K manually
        try:
            # Most models have q_proj, k_proj
            if hasattr(attn, 'q_proj'):
                q = attn.q_proj(hidden)
                k = attn.k_proj(hidden)
            elif hasattr(attn, 'qkv_proj'):
                # Fused QKV
                qkv = attn.qkv_proj(hidden)
                # This varies by model, approximate
                q = qkv[:, :, :hidden.shape[-1]]
                k = qkv[:, :, hidden.shape[-1]:2*hidden.shape[-1]]
            else:
                entropies.append(0.0)
                hidden = layer(hidden, mask=mask, cache=None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                mx.eval(hidden)
                continue

            # Reshape for multi-head attention
            n_heads = getattr(attn, 'num_heads', getattr(attn, 'n_heads', 8))
            head_dim = q.shape[-1] // n_heads

            # (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
            q = q.reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

            # Compute attention scores
            scale = head_dim ** -0.5
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask
            scores = scores + mask

            # Softmax
            attn_weights = mx.softmax(scores, axis=-1)
            mx.eval(attn_weights)

            # Compute entropy
            entropy = compute_attention_entropy(attn_weights[0])
            entropies.append(entropy)

        except Exception as e:
            entropies.append(0.0)

        # Forward through layer
        hidden = layer(hidden, mask=mask, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)

    return entropies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    args = parser.parse_args()

    from mlx_lm import load

    print("=" * 70)
    print("ATTENTION ENTROPY TRAJECTORY")
    print("=" * 70)
    print(f"Model: {Path(args.model).name}")

    model, tokenizer = load(args.model)
    n_layers = len(getattr(model, "model", model).layers)
    print(f"Layers: {n_layers}")
    print("=" * 70)

    for task_type, prompt in TASK_PROBES.items():
        print(f"\n{task_type.upper()}")
        print("-" * 40)

        entropies = trace_attention_entropy(model, tokenizer, prompt)

        if not any(e > 0 for e in entropies):
            print("  Could not extract attention weights from this model")
            continue

        # Normalize for visualization
        max_entropy = max(e for e in entropies if e > 0) if any(e > 0 for e in entropies) else 1.0

        print(f"  Entropy trajectory:")
        for i, e in enumerate(entropies):
            if e > 0:
                bar = "█" * int((e / max_entropy) * 40)
                print(f"  L{i+1:02d} |{bar} {e:.3f}")
            else:
                print(f"  L{i+1:02d} | (no data)")

        # Analyze trend
        valid_entropies = [e for e in entropies if e > 0]
        if len(valid_entropies) > 2:
            early = np.mean(valid_entropies[:len(valid_entropies)//3])
            late = np.mean(valid_entropies[-len(valid_entropies)//3:])
            trend = "FOCUSING" if late < early else "DIFFUSING"
            print(f"\n  Trend: {trend} (early avg: {early:.3f}, late avg: {late:.3f})")


if __name__ == "__main__":
    main()
