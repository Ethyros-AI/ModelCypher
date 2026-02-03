#!/usr/bin/env python3
"""
Analyze the structure of rank-1 attention matrices.

Question: How can LFM2 have rank-1 attention (all rows similar) but high entropy (spread across tokens)?

Hypothesis: All positions attend to the same set of tokens with similar weights.
This means the attention pattern is uniform across positions but distributed across targets.
"""

import mlx.core as mx
from mlx_lm import load
import numpy as np


def analyze_attention_structure(model_path: str, prompt: str = "The quick brown fox jumps over the lazy dog."):
    """Analyze attention matrix structure."""
    print(f"Loading {model_path}...")
    model, tokenizer = load(model_path)

    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        embed = model.model.embed_tokens
    elif hasattr(model, "layers"):
        layers = model.layers
        embed = model.embed_tokens
    else:
        raise ValueError("Cannot find layers")

    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    print(f"Tokens: {len(tokens)}")

    # Get embeddings
    h = embed(input_ids)

    # Analyze first attention layer with attention
    for layer_idx, layer in enumerate(layers):
        try:
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(layer, "attention"):
                attn_module = layer.attention
            else:
                continue

            # Apply layer norm
            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_normed = layer.ln_1(h)
            else:
                h_normed = h

            # Compute Q, K
            B, T, C = h_normed.shape

            if hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(h_normed)
                k = attn_module.k_proj(h_normed)
            else:
                continue

            # Get dimensions
            n_heads = getattr(attn_module, "n_heads", 8)
            n_kv_heads = getattr(attn_module, "n_kv_heads", n_heads)

            q_dim = q.shape[-1]
            k_dim = k.shape[-1]
            head_dim = q_dim // n_heads
            kv_head_dim = k_dim // n_kv_heads

            # Reshape
            q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, n_kv_heads, kv_head_dim).transpose(0, 2, 1, 3)

            if n_kv_heads != n_heads:
                n_rep = n_heads // n_kv_heads
                k = mx.repeat(k, n_rep, axis=1)

            # Compute attention
            scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=q.dtype))
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            attn = mx.softmax(scores, axis=-1)
            mx.eval(attn)

            # Analyze structure of first head
            A = attn[0, 0].astype(mx.float32)  # [T, T]
            mx.eval(A)

            # Compute row similarity: mean pairwise cosine similarity between rows
            A_np = np.array(A)

            # Normalize rows
            row_norms = np.linalg.norm(A_np, axis=1, keepdims=True)
            A_normed = A_np / (row_norms + 1e-10)

            # Pairwise cosine similarity (A_normed @ A_normed.T gives cosine sim)
            cos_sim_matrix = A_normed @ A_normed.T

            # Mean off-diagonal similarity (how similar are different rows?)
            n = cos_sim_matrix.shape[0]
            off_diag_mask = ~np.eye(n, dtype=bool)
            mean_row_similarity = cos_sim_matrix[off_diag_mask].mean()

            # Row entropy (how spread is each row?)
            row_entropy = -np.sum(A_np * np.log(A_np + 1e-10), axis=1).mean()

            # Column variance (does attention focus on same columns across rows?)
            col_means = A_np.mean(axis=0)
            col_std = A_np.std(axis=0).mean()

            # Rank (via SVD)
            U, S, Vt = np.linalg.svd(A_np)
            total = S.sum()
            cumsum = np.cumsum(S) / total
            rank_90 = np.searchsorted(cumsum, 0.90) + 1  # Rank to capture 90% variance

            print(f"\nLayer {layer_idx} (head 0):")
            print(f"  Mean row similarity: {mean_row_similarity:.4f}")
            print(f"    (1.0 = all rows identical, 0.0 = orthogonal)")
            print(f"  Row entropy: {row_entropy:.4f}")
            print(f"    (max possible: {np.log(T):.4f})")
            print(f"  Column std (mean): {col_std:.4f}")
            print(f"    (low = consistent attention across positions)")
            print(f"  Rank-90%: {rank_90}")
            print(f"    (ranks needed to capture 90% variance)")
            print(f"  Top singular values: {S[:5].round(4)}")

            # Show the attention pattern
            print(f"\n  Attention matrix (first 5x5):")
            for i in range(min(5, T)):
                row_str = " ".join([f"{A_np[i, j]:.3f}" for j in range(min(5, T))])
                print(f"    [{row_str}]")

            # Only analyze a few layers
            if layer_idx >= 5:
                break

            # Forward pass for next layer
            h = layer(h)

        except Exception as e:
            print(f"Layer {layer_idx} failed: {e}")
            continue


def compare_random_vs_trained():
    """Compare structure of random vs trained attention."""
    print("\n" + "="*60)
    print("  RANDOM ATTENTION STRUCTURE")
    print("="*60)

    T = 10

    # Random attention (uniform softmax)
    scores = np.random.randn(T, T)
    A = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    # Normalize rows
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normed = A / (row_norms + 1e-10)

    # Pairwise similarity
    cos_sim_matrix = A_normed @ A_normed.T
    n = cos_sim_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    mean_row_similarity = cos_sim_matrix[off_diag_mask].mean()

    row_entropy = -np.sum(A * np.log(A + 1e-10), axis=1).mean()

    U, S, Vt = np.linalg.svd(A)
    cumsum = np.cumsum(S) / S.sum()
    rank_90 = np.searchsorted(cumsum, 0.90) + 1

    print(f"\nRandom attention (T={T}):")
    print(f"  Mean row similarity: {mean_row_similarity:.4f}")
    print(f"  Row entropy: {row_entropy:.4f} (max: {np.log(T):.4f})")
    print(f"  Rank-90%: {rank_90}")
    print(f"  Top singular values: {S[:5].round(4)}")


def main():
    print("="*60)
    print("  ATTENTION STRUCTURE ANALYSIS")
    print("="*60)

    # Compare random baseline
    compare_random_vs_trained()

    # Analyze LFM2 (rank-1 attention)
    print("\n" + "="*60)
    print("  LFM2-350M ATTENTION STRUCTURE")
    print("="*60)
    analyze_attention_structure("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    # Analyze Qwen (higher rank)
    print("\n" + "="*60)
    print("  QWEN2.5-3B ATTENTION STRUCTURE")
    print("="*60)
    analyze_attention_structure("/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16")


if __name__ == "__main__":
    main()
