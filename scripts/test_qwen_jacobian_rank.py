#!/usr/bin/env python3
"""
Test Jacobian effective rank on actual Qwen model.

Question: Qwen has attention rank ~3-4. Does it have rank-1 Jacobians?
"""

import mlx.core as mx
from mlx_lm import load
from pathlib import Path


def effective_rank(singular_values: mx.array) -> float:
    """Compute Shannon effective rank from singular values."""
    s = mx.abs(singular_values)
    s_sum = mx.sum(s)
    if float(s_sum) < 1e-10:
        return 0.0
    p = s / s_sum
    log_p = mx.where(p > 1e-10, mx.log(p), mx.zeros_like(p))
    entropy = -mx.sum(p * log_p)
    return float(mx.exp(entropy))


def compute_layer_jacobian(model, tokenizer, prompt: str, layer_idx: int, n_probes: int = 32):
    """Compute Jacobian effective rank for a single layer.

    Uses random projection method for efficiency.
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get model components
    if hasattr(model, "model"):
        embed = model.model.embed_tokens
        layers = model.model.layers
    else:
        embed = model.embed_tokens
        layers = model.layers

    # Forward to get input to target layer
    h = embed(input_ids)
    for i in range(layer_idx):
        h = layers[i](h)

    B, T, C = h.shape

    # Define forward through single layer
    def layer_forward_flat(h_flat):
        h_reshaped = h_flat.reshape(B, T, C)
        out = layers[layer_idx](h_reshaped)
        return out.reshape(-1)

    h_flat = h.reshape(-1)
    out_dim = B * T * C

    # Random projection vectors for Jacobian estimation
    probes = mx.random.normal((n_probes, out_dim))
    probes = probes / mx.linalg.norm(probes, axis=1, keepdims=True)

    # Compute J^T @ probes via vjp
    jacobian_samples = []
    for i in range(n_probes):
        def projected_out(hf):
            out = layer_forward_flat(hf)
            return mx.sum(out * probes[i])

        grad_i = mx.grad(projected_out)(h_flat)
        jacobian_samples.append(grad_i)

    JT_probes = mx.stack(jacobian_samples, axis=0).astype(mx.float32)
    mx.eval(JT_probes)

    # SVD
    U, S, Vt = mx.linalg.svd(JT_probes, stream=mx.cpu)
    mx.eval(S)

    eff_rank = effective_rank(S)
    spectral_gap = float(S[0] / S[1]) if len(S) > 1 and float(S[1]) > 1e-10 else float('inf')

    return eff_rank, spectral_gap, S[:5]


def main():
    print("="*60)
    print("  QWEN JACOBIAN EFFECTIVE RANK TEST")
    print("="*60)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16"
    prompt = "The quick brown fox"

    print(f"\nLoading {Path(model_path).name}...")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} layers")
    print(f"Testing prompt: '{prompt}'")

    # Test multiple layers
    test_layers = [0, 5, 10, 15, 20, 25, 30, 35]
    test_layers = [l for l in test_layers if l < n_layers]

    print(f"\n{'Layer':<8} {'Eff Rank':<12} {'Spectral Gap':<15} {'Top SVs'}")
    print("-"*60)

    for layer_idx in test_layers:
        try:
            eff_rank, gap, top_svs = compute_layer_jacobian(
                model, tokenizer, prompt, layer_idx, n_probes=32
            )
            svs_str = ", ".join([f"{float(s):.3f}" for s in top_svs])
            print(f"{layer_idx:<8} {eff_rank:<12.2f} {gap:<15.2f} [{svs_str}]")
        except Exception as e:
            print(f"{layer_idx:<8} FAILED: {e}")

    # Compare to random baseline
    print("\n" + "="*60)
    print("  COMPARISON TO RANDOM")
    print("="*60)

    # Random matrix baseline
    random_J = mx.random.normal((32, 1000))
    U, S, Vt = mx.linalg.svd(random_J.astype(mx.float32), stream=mx.cpu)
    mx.eval(S)
    random_rank = effective_rank(S)
    print(f"\nRandom matrix ({random_J.shape}): eff_rank = {random_rank:.2f}")


if __name__ == "__main__":
    main()
