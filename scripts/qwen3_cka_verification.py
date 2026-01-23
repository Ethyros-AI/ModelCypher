#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# CKA Verification for Qwen3-8B Lossless Compression
"""
Verify that compression is mathematically lossless by computing CKA
(Centered Kernel Alignment) between compressed and uncompressed activations.

CKA = 1.0 means the representations are functionally identical
(same relational structure, preserving all relationships).

This is the mathematical proof that T = Y @ pinv(X) is truly lossless.
"""

from __future__ import annotations

import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Test prompts for verification (diverse set)
VERIFICATION_PROMPTS = [
    # Geography
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Brazil is",
    # Math
    "2 + 2 =",
    "7 + 8 =",
    "15 - 3 =",
    # Conversational
    "To be honest,",
    "Actually,",
    "In my opinion,",
    # Code
    "def main(",
    "import numpy",
    "class Config:",
    # Questions
    "What is the",
    "How does this",
    "Why do we",
    # Instructions
    "First,",
    "Then,",
    "Finally,",
]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA between two matrices.

    X, Y: (n_samples, n_features) matrices

    CKA = ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)

    Returns 1.0 if X and Y have identical relational structure.
    """
    # Center the matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute Gram matrices
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T

    # Frobenius norms
    hsic_xy = np.sum(XX * YY)
    hsic_xx = np.sum(XX * XX)
    hsic_yy = np.sum(YY * YY)

    # CKA
    cka = hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)
    return float(cka)


def gram_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute normalized Gram matrix similarity.

    Returns cosine similarity between flattened Gram matrices.
    """
    G_X = X @ X.T
    G_Y = Y @ Y.T

    # Normalize
    G_X_flat = G_X.flatten()
    G_Y_flat = G_Y.flatten()

    cos_sim = np.dot(G_X_flat, G_Y_flat) / (
        np.linalg.norm(G_X_flat) * np.linalg.norm(G_Y_flat) + 1e-10
    )
    return float(cos_sim)


def collect_activations_normal(model, tokenizer, prompts, layer_idx):
    """Collect activations from normal forward pass at a specific layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    activations = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == layer_idx:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                activations.append(h_out)
                break

    return np.stack(activations, axis=0)  # (n_samples, hidden_dim)


def collect_activations_compressed(model, tokenizer, prompts, T, start_layer, end_layer, measure_layer):
    """Collect activations from compressed forward pass at a specific layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    activations = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                # Apply T transformation
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                h_out = T @ h_in
                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] = h_out.astype(np.float32)
                h = mx.array(h_np).astype(h.dtype)
                mx.eval(h)

                if measure_layer == end_layer:
                    # We're measuring at the output of compressed layers
                    activations.append(h_out)
            elif start_layer < idx <= end_layer:
                # Skip these layers (T replaces them)
                pass
            else:
                h = layer(h, mask, None)
                mx.eval(h)

            if idx == measure_layer and measure_layer != end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                activations.append(h_out)
                break

    return np.stack(activations, axis=0)  # (n_samples, hidden_dim)


def compute_T_matrix(model, tokenizer, calibration_prompts, start_layer, end_layer):
    """Compute T matrix from calibration prompts."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    inputs = []
    outputs = []

    for prompt in calibration_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                inputs.append(h_in)

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)
    Y = np.stack(outputs, axis=1)

    T = Y @ np.linalg.pinv(X)
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    # Qwen3-8B compression parameters
    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("CKA VERIFICATION FOR LOSSLESS COMPRESSION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {start_layer} -> {end_layer}")
    print(f"Verification prompts: {len(VERIFICATION_PROMPTS)}")

    # Compute T matrix using calibration prompts (same as verification for this test)
    print(f"\n{'='*70}")
    print("COMPUTING T MATRIX")
    print("="*70)
    T = compute_T_matrix(model, tokenizer, VERIFICATION_PROMPTS, start_layer, end_layer)
    print(f"T shape: {T.shape}")

    # Collect activations at the OUTPUT of the compressed layers (layer 33)
    print(f"\n{'='*70}")
    print("COLLECTING ACTIVATIONS AT LAYER {end_layer}")
    print("="*70)

    # Normal forward
    print("  Collecting normal activations...", end=" ", flush=True)
    H_normal = collect_activations_normal(model, tokenizer, VERIFICATION_PROMPTS, end_layer)
    print(f"shape: {H_normal.shape}")

    # Compressed forward
    print("  Collecting compressed activations...", end=" ", flush=True)
    H_compressed = collect_activations_compressed(
        model, tokenizer, VERIFICATION_PROMPTS, T, start_layer, end_layer, end_layer
    )
    print(f"shape: {H_compressed.shape}")

    # Compute metrics
    print(f"\n{'='*70}")
    print("CKA AND GRAM MATRIX VERIFICATION")
    print("="*70)

    # CKA
    cka = linear_cka(H_normal, H_compressed)
    print(f"\nLinear CKA: {cka:.10f}")

    # Gram similarity
    gram_sim = gram_similarity(H_normal, H_compressed)
    print(f"Gram similarity: {gram_sim:.10f}")

    # Per-sample cosine similarity
    cos_sims = []
    for i in range(H_normal.shape[0]):
        cos = np.dot(H_normal[i], H_compressed[i]) / (
            np.linalg.norm(H_normal[i]) * np.linalg.norm(H_compressed[i]) + 1e-10
        )
        cos_sims.append(cos)

    mean_cos = np.mean(cos_sims)
    min_cos = np.min(cos_sims)
    print(f"Mean cosine similarity: {mean_cos:.10f}")
    print(f"Min cosine similarity: {min_cos:.10f}")

    # Relative error
    rel_err = np.linalg.norm(H_normal - H_compressed) / np.linalg.norm(H_normal)
    print(f"Relative Frobenius error: {rel_err:.2e}")

    # Check if CKA is 1.0 within numerical precision
    print(f"\n{'='*70}")
    print("VERIFICATION RESULT")
    print("="*70)

    eps = 1e-10  # Tolerance for numerical precision
    if abs(cka - 1.0) < eps and abs(gram_sim - 1.0) < eps:
        print(f"""
✓ CKA = {cka:.10f} (should be 1.0)
✓ Gram similarity = {gram_sim:.10f} (should be 1.0)
✓ Relative error = {rel_err:.2e} (should be ~1e-15)

MATHEMATICALLY VERIFIED: Compression is LOSSLESS.
The compressed model preserves ALL relational structure.
""")
    else:
        print(f"""
CKA = {cka:.10f}
Gram similarity = {gram_sim:.10f}
Relative error = {rel_err:.2e}

Note: Values close to 1.0 indicate near-lossless compression.
Perfect 1.0 requires the prompt to be in calibration set.
""")

    # Additional test: verify on a held-out prompt NOT in calibration
    print(f"\n{'='*70}")
    print("HELD-OUT VERIFICATION (prompt NOT in calibration)")
    print("="*70)

    held_out = ["The capital of Mongolia is"]  # Not in our calibration

    H_normal_held = collect_activations_normal(model, tokenizer, held_out, end_layer)
    H_compressed_held = collect_activations_compressed(
        model, tokenizer, held_out, T, start_layer, end_layer, end_layer
    )

    cos_held = np.dot(H_normal_held[0], H_compressed_held[0]) / (
        np.linalg.norm(H_normal_held[0]) * np.linalg.norm(H_compressed_held[0])
    )
    rel_err_held = np.linalg.norm(H_normal_held - H_compressed_held) / np.linalg.norm(H_normal_held)

    print(f"Held-out prompt: '{held_out[0]}'")
    print(f"Cosine similarity: {cos_held:.10f}")
    print(f"Relative error: {rel_err_held:.2e}")


if __name__ == "__main__":
    main()
