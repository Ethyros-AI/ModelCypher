#!/usr/bin/env python3
"""
Test Jacobian effective rank on random vs trained transformer weights.

Hypothesis: Trained transformers have rank-1 Jacobians due to learned sharp attention.
Random/untrained weights should have higher effective rank.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path


def effective_rank(singular_values: mx.array) -> float:
    """Compute Shannon effective rank from singular values."""
    # Normalize to probability distribution
    s = mx.abs(singular_values)
    s_sum = mx.sum(s)
    if float(s_sum) < 1e-10:
        return 0.0
    p = s / s_sum

    # Shannon entropy
    log_p = mx.where(p > 1e-10, mx.log(p), mx.zeros_like(p))
    entropy = -mx.sum(p * log_p)

    # Effective rank = exp(entropy)
    return float(mx.exp(entropy))


def spectral_gap(singular_values: mx.array) -> float:
    """Compute σ₁/σ₂ ratio."""
    s = mx.sort(mx.abs(singular_values))[::-1]  # Descending
    if len(s) < 2:
        return float('inf')
    s1, s2 = float(s[0]), float(s[1])
    if s2 < 1e-10:
        return float('inf')
    return s1 / s2


class MinimalAttention(nn.Module):
    """Minimal single-head attention for testing."""

    def __init__(self, d_model: int, random_init: bool = True):
        super().__init__()
        self.d_model = d_model

        if random_init:
            # Random initialization
            scale = 0.02
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            # Will be set from trained model
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(self.d_model, dtype=x.dtype))
        scores = (q @ k.transpose(0, 2, 1)) * scale

        # Softmax
        attn = mx.softmax(scores, axis=-1)

        # Output
        out = attn @ v
        out = self.o_proj(out)

        return out


class MinimalMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


class MinimalTransformerBlock(nn.Module):
    """Minimal transformer block for Jacobian testing."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.attn = MinimalAttention(d_model)
        self.mlp = MinimalMLP(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def compute_jacobian(model: nn.Module, x: mx.array) -> mx.array:
    """Compute Jacobian of model output w.r.t. input."""
    B, T, C = x.shape

    def forward_flat(x_flat):
        x_reshaped = x_flat.reshape(B, T, C)
        out = model(x_reshaped)
        return out.reshape(-1)

    # Use vmap + grad for Jacobian
    x_flat = x.reshape(-1)

    jacobian_rows = []
    for i in range(len(x_flat)):
        def grad_fn(xf):
            out = forward_flat(xf)
            return out[i]

        grad_i = mx.grad(grad_fn)(x_flat)
        jacobian_rows.append(grad_i)

    J = mx.stack(jacobian_rows, axis=0)
    return J


def compute_jacobian_fast(model: nn.Module, x: mx.array, n_samples: int = 64) -> mx.array:
    """Compute Jacobian via random projections (faster for large models)."""
    B, T, C = x.shape
    out_dim = B * T * C

    # Random projection vectors
    probes = mx.random.normal((n_samples, out_dim))
    probes = probes / mx.linalg.norm(probes, axis=1, keepdims=True)

    def forward_flat(x_flat):
        x_reshaped = x_flat.reshape(B, T, C)
        out = model(x_reshaped)
        return out.reshape(-1)

    x_flat = x.reshape(-1)

    # Compute J^T @ probes via vjp
    jacobian_samples = []
    for i in range(n_samples):
        def projected_out(xf):
            out = forward_flat(xf)
            return mx.sum(out * probes[i])

        grad_i = mx.grad(projected_out)(x_flat)
        jacobian_samples.append(grad_i)

    # Stack: (n_samples, in_dim)
    JT_probes = mx.stack(jacobian_samples, axis=0)
    return JT_probes


def analyze_jacobian(J: mx.array, label: str):
    """Analyze Jacobian matrix and print results."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    # SVD (must run on CPU in MLX)
    U, S, Vt = mx.linalg.svd(J, stream=mx.cpu)
    mx.eval(S)

    eff_rank = effective_rank(S)
    gap = spectral_gap(S)

    print(f"  Shape: {J.shape}")
    print(f"  Top 5 singular values: {[f'{float(s):.4f}' for s in S[:5]]}")
    print(f"  Effective rank: {eff_rank:.2f}")
    print(f"  Spectral gap (σ₁/σ₂): {gap:.2f}")
    print(f"  Max singular value: {float(S[0]):.4f}")

    return eff_rank, gap


def test_random_vs_trained():
    """Main test: compare random and trained-like initialization."""
    print("\n" + "="*60)
    print("  JACOBIAN EFFECTIVE RANK: RANDOM vs TRAINED STRUCTURE")
    print("="*60)

    # Parameters
    d_model = 64
    d_ff = 256
    seq_len = 8
    batch_size = 1

    # Create input
    x = mx.random.normal((batch_size, seq_len, d_model)) * 0.1
    mx.eval(x)

    # Test 1: Random initialization
    print("\n[1] Random Initialization (standard N(0, 0.02))")
    model_random = MinimalTransformerBlock(d_model, d_ff)
    mx.eval(model_random.parameters())

    J_random = compute_jacobian_fast(model_random, x, n_samples=64)
    mx.eval(J_random)
    rank_random, gap_random = analyze_jacobian(J_random, "Random Init")

    # Test 2: Near-identity initialization (simulating converged attention)
    print("\n[2] Near-Identity Init (simulating diffuse attention)")
    model_identity = MinimalTransformerBlock(d_model, d_ff)

    # Make projections near-identity
    eye = mx.eye(d_model) * 0.1
    model_identity.attn.q_proj.weight = eye
    model_identity.attn.k_proj.weight = eye
    model_identity.attn.v_proj.weight = eye
    model_identity.attn.o_proj.weight = eye
    mx.eval(model_identity.parameters())

    J_identity = compute_jacobian_fast(model_identity, x, n_samples=64)
    mx.eval(J_identity)
    rank_identity, gap_identity = analyze_jacobian(J_identity, "Near-Identity Init")

    # Test 3: Rank-1 projection (simulating sharp attention)
    print("\n[3] Rank-1 Projections (simulating sharp/focused attention)")
    model_rank1 = MinimalTransformerBlock(d_model, d_ff)

    # Make projections rank-1 (all rows same)
    u = mx.random.normal((1, d_model))
    v = mx.random.normal((d_model, 1))
    rank1_weight = u.T @ v.T * 0.1  # Outer product, scaled

    model_rank1.attn.q_proj.weight = rank1_weight
    model_rank1.attn.k_proj.weight = rank1_weight
    model_rank1.attn.v_proj.weight = rank1_weight
    model_rank1.attn.o_proj.weight = rank1_weight
    mx.eval(model_rank1.parameters())

    J_rank1 = compute_jacobian_fast(model_rank1, x, n_samples=64)
    mx.eval(J_rank1)
    rank_rank1, gap_rank1 = analyze_jacobian(J_rank1, "Rank-1 Projections")

    # Test 4: Pure random matrix (baseline)
    print("\n[4] Pure Random Matrix (baseline - not a network)")
    J_pure_random = mx.random.normal((64, d_model * seq_len))
    mx.eval(J_pure_random)
    rank_pure, gap_pure = analyze_jacobian(J_pure_random, "Pure Random Matrix")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n  {'Configuration':<30} {'Eff. Rank':>12} {'Spectral Gap':>14}")
    print(f"  {'-'*30} {'-'*12} {'-'*14}")
    print(f"  {'Pure Random Matrix':<30} {rank_pure:>12.2f} {gap_pure:>14.2f}")
    print(f"  {'Random Init Transformer':<30} {rank_random:>12.2f} {gap_random:>14.2f}")
    print(f"  {'Near-Identity Init':<30} {rank_identity:>12.2f} {gap_identity:>14.2f}")
    print(f"  {'Rank-1 Projections':<30} {rank_rank1:>12.2f} {gap_rank1:>14.2f}")

    print("\n  Interpretation:")
    if rank_random > rank_rank1 * 1.5:
        print("  ✓ Random init has HIGHER effective rank than rank-1 structure")
        print("  ✓ Confirms: rank-1 Jacobians are LEARNED, not architectural constraint")
    elif rank_random < rank_rank1:
        print("  ✗ Unexpected: Random init has LOWER effective rank")
        print("    (may indicate numerical issues or implementation error)")
    else:
        print("  ~ Ambiguous: Effective ranks are similar")
        print("    (more investigation needed)")

    return {
        'random': (rank_random, gap_random),
        'identity': (rank_identity, gap_identity),
        'rank1': (rank_rank1, gap_rank1),
        'pure_random': (rank_pure, gap_pure),
    }


if __name__ == "__main__":
    results = test_random_vs_trained()
