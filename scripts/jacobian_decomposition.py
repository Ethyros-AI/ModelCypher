#!/usr/bin/env python3
"""
Decompose Jacobian by component to find source of rank-1 structure.

Question: Qwen has attention rank ~4 but Jacobian rank = 1.
Which component creates the rank-1 structure?

Components to test:
1. Attention only
2. MLP only
3. LayerNorm only
4. Full layer
"""

import mlx.core as mx
from mlx_lm import load


def effective_rank(singular_values) -> float:
    """Compute Shannon effective rank."""
    s = mx.abs(singular_values).astype(mx.float32)
    s_sum = mx.sum(s)
    if float(s_sum) < 1e-10:
        return 0.0
    p = s / s_sum
    log_p = mx.where(p > 1e-10, mx.log(p), mx.zeros_like(p))
    entropy = -mx.sum(p * log_p)
    return float(mx.exp(entropy))


def compute_jacobian_via_random_projection(fn, x, n_probes=64):
    """Compute Jacobian effective rank via random projections."""
    x_flat = x.reshape(-1)
    in_dim = x_flat.shape[0]

    def fn_flat(xf):
        return fn(xf.reshape(x.shape)).reshape(-1)

    out_test = fn_flat(x_flat)
    out_dim = out_test.shape[0]

    # Random probes
    probes = mx.random.normal((n_probes, out_dim))
    probes = probes / mx.linalg.norm(probes, axis=1, keepdims=True)

    # Compute J^T @ probes via VJP
    jacobian_samples = []
    for i in range(n_probes):
        def projected_out(xf):
            out = fn_flat(xf)
            return mx.sum(out * probes[i])

        grad_i = mx.grad(projected_out)(x_flat)
        jacobian_samples.append(grad_i)

    JT_probes = mx.stack(jacobian_samples, axis=0).astype(mx.float32)
    mx.eval(JT_probes)

    # SVD
    U, S, Vt = mx.linalg.svd(JT_probes, stream=mx.cpu)
    mx.eval(S)

    eff_rank = effective_rank(S)
    gap = float(S[0] / S[1]) if len(S) > 1 and float(S[1]) > 1e-10 else float('inf')

    return eff_rank, gap, S[:5]


def main():
    print("="*60)
    print("  JACOBIAN DECOMPOSITION BY COMPONENT")
    print("="*60)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16"
    print(f"\nLoading {model_path}...")
    model, tokenizer = load(model_path)

    # Get a layer
    layer = model.model.layers[10]  # Middle layer
    embed = model.model.embed_tokens

    # Create input
    tokens = tokenizer.encode("The quick brown fox")
    input_ids = mx.array([tokens])
    h = embed(input_ids)

    # Forward through first 10 layers to get realistic input
    for i in range(10):
        h = model.model.layers[i](h)
    mx.eval(h)

    print(f"\nInput shape: {h.shape}")
    print(f"Testing layer 10")

    # Test 1: Full layer
    print("\n1. Full Layer (attention + MLP + residual):")
    def full_layer(x):
        return layer(x)

    eff_rank, gap, svs = compute_jacobian_via_random_projection(full_layer, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 2: Attention only (no residual)
    print("\n2. Attention only (no residual):")
    def attn_only(x):
        # Apply pre-norm
        normed = layer.input_layernorm(x)
        # Apply attention
        attn_out = layer.self_attn(normed)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return attn_out

    eff_rank, gap, svs = compute_jacobian_via_random_projection(attn_only, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 3: MLP only (no residual)
    print("\n3. MLP only (no residual):")
    def mlp_only(x):
        # Apply post-attention-norm
        normed = layer.post_attention_layernorm(x)
        # Apply MLP
        return layer.mlp(normed)

    eff_rank, gap, svs = compute_jacobian_via_random_projection(mlp_only, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 4: LayerNorm only
    print("\n4. LayerNorm only:")
    def norm_only(x):
        return layer.input_layernorm(x)

    eff_rank, gap, svs = compute_jacobian_via_random_projection(norm_only, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 5: Attention with residual
    print("\n5. Attention + residual:")
    def attn_with_residual(x):
        normed = layer.input_layernorm(x)
        attn_out = layer.self_attn(normed)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return x + attn_out  # Residual

    eff_rank, gap, svs = compute_jacobian_via_random_projection(attn_with_residual, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 6: MLP with residual
    print("\n6. MLP + residual:")
    def mlp_with_residual(x):
        normed = layer.post_attention_layernorm(x)
        mlp_out = layer.mlp(normed)
        return x + mlp_out

    eff_rank, gap, svs = compute_jacobian_via_random_projection(mlp_with_residual, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 7: Identity (baseline)
    print("\n7. Identity (baseline):")
    def identity(x):
        return x

    eff_rank, gap, svs = compute_jacobian_via_random_projection(identity, h, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   (Should be ~64 = full rank)")

    # Test 8: Full layer on MEAN-POOLED input (like JacobianAnalyzer)
    print("\n8. Full Layer on MEAN-POOLED input:")
    h_pooled = mx.mean(h, axis=(0, 1))  # [hidden_dim]
    mx.eval(h_pooled)

    def full_layer_pooled(x):
        # Reshape to [1, 1, hidden_dim]
        x_reshaped = x.reshape(1, 1, -1)
        out = layer(x_reshaped)
        # Mean pool output
        return mx.mean(out, axis=(0, 1))

    eff_rank, gap, svs = compute_jacobian_via_random_projection(full_layer_pooled, h_pooled, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")
    print(f"   (This should match JacobianAnalyzer's rank-1 result)")

    # Test 9: Attention only on mean-pooled
    print("\n9. Attention only on MEAN-POOLED input:")
    def attn_only_pooled(x):
        x_reshaped = x.reshape(1, 1, -1)
        normed = layer.input_layernorm(x_reshaped)
        attn_out = layer.self_attn(normed)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return mx.mean(attn_out, axis=(0, 1))

    eff_rank, gap, svs = compute_jacobian_via_random_projection(attn_only_pooled, h_pooled, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    # Test 10: MLP only on mean-pooled
    print("\n10. MLP only on MEAN-POOLED input:")
    def mlp_only_pooled(x):
        x_reshaped = x.reshape(1, 1, -1)
        normed = layer.post_attention_layernorm(x_reshaped)
        mlp_out = layer.mlp(normed)
        return mx.mean(mlp_out, axis=(0, 1))

    eff_rank, gap, svs = compute_jacobian_via_random_projection(mlp_only_pooled, h_pooled, n_probes=64)
    print(f"   Effective rank: {eff_rank:.2f}")
    print(f"   Spectral gap: {gap:.2f}")

    print("\n" + "="*60)
    print("  ANALYSIS")
    print("="*60)
    print("""
Key insight: The rank-1 property depends on HOW we measure:
- Full sequence [B, T, C] → [B, T, C]: High rank (~64)
- Mean-pooled [hidden_dim] → [hidden_dim]: Rank-1

The rank-1 Jacobian is about how a SINGLE POSITION transforms,
not about full sequence dynamics.
""")


if __name__ == "__main__":
    main()
