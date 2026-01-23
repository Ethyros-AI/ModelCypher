#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Gram-Preserving Compression Experiment
"""
Gram-Preserving Compression

THE HYPOTHESIS:
If 86% of Qwen3-1.7B is transmission (layers 3-26), we can SKIP those layers
as long as we:
1. Preserve the Gram matrix (relational structure)
2. Maintain energy balance: E_out = E_in + ||δ||² + 2<h, δ>

METHOD:
1. Run full model, capture states at layer 3 (entry) and layer 27 (exit)
2. Compute the "transmission transform" T that maps entry → exit
3. Test if T preserves Gram matrix: G_exit ≈ G_entry
4. Test if T preserves energy: ||h_exit||² ≈ ||h_entry||²
5. If both hold, we can replace 24 layers with a single transform T

COMPRESSION MATH:
- Qwen3-1.7B: 28 layers × ~60M params each ≈ 1.7B params
- If we collapse 24 layers to 1 transform: (28-24+1)/28 = 5/28 = 18% of layers
- But T itself is just a rotation matrix + scale, so even smaller

Usage:
    python gram_preserving_compression.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "good", "bad", "love", "hate",
    "red", "blue", "green", "yellow",
    "fast", "slow", "quiet", "loud",
]


def get_entry_exit_states(
    model: Any,
    tokenizer: Any,
    words: list[str],
    entry_layer: int,
    exit_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get hidden states at entry and exit of transmission section."""
    import mlx.core as mx

    entry_states = []
    exit_states = []

    inner_model = model.model if hasattr(model, 'model') else model

    for word in words:
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(inner_model.layers):
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

            if idx == entry_layer:
                entry_states.append(np.array(h[0, -1, :].astype(mx.float32)))
            if idx == exit_layer:
                exit_states.append(np.array(h[0, -1, :].astype(mx.float32)))
                break

    return np.stack(entry_states), np.stack(exit_states)


def compute_gram_matrix(states: np.ndarray) -> np.ndarray:
    """Compute Gram matrix G = H @ H.T (relational structure)."""
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    G = states @ states.T
    return G


def compute_gram_similarity(G1: np.ndarray, G2: np.ndarray) -> float:
    """Compute similarity between two Gram matrices."""
    # Flatten and compute correlation
    g1_flat = G1.flatten()
    g2_flat = G2.flatten()

    # Normalize
    g1_norm = g1_flat / (np.linalg.norm(g1_flat) + 1e-10)
    g2_norm = g2_flat / (np.linalg.norm(g2_flat) + 1e-10)

    return float(np.dot(g1_norm, g2_norm))


def compute_transmission_transform(
    entry_states: np.ndarray,
    exit_states: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the transform T that maps entry → exit.

    We seek T such that: exit ≈ T @ entry

    Using least squares: T = exit.T @ pinv(entry.T)

    Returns: (T, reconstruction_error)
    """
    # entry: (n_samples, hidden_dim)
    # We want T: (hidden_dim, hidden_dim) such that exit ≈ entry @ T.T
    # Or equivalently: exit.T ≈ T @ entry.T

    # Solve: T @ entry.T = exit.T using least squares
    # T = exit.T @ pinv(entry.T)

    entry_pinv = np.linalg.pinv(entry_states.T)  # (n_samples, hidden_dim)
    T = exit_states.T @ entry_pinv  # (hidden_dim, hidden_dim)

    # Compute reconstruction
    reconstructed = (T @ entry_states.T).T  # (n_samples, hidden_dim)

    # Reconstruction error
    error = np.mean(np.linalg.norm(reconstructed - exit_states, axis=1))
    rel_error = error / (np.mean(np.linalg.norm(exit_states, axis=1)) + 1e-10)

    return T, float(rel_error)


def analyze_transform(T: np.ndarray) -> dict:
    """Analyze properties of the transmission transform."""
    # Singular value decomposition
    U, S, Vh = np.linalg.svd(T)

    # Condition number
    cond = S[0] / (S[-1] + 1e-10)

    # Effective rank (how many singular values matter)
    total = np.sum(S)
    cumsum = np.cumsum(S) / total
    eff_rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    eff_rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)

    # Is it approximately orthogonal? (T @ T.T ≈ I)
    TTt = T @ T.T
    identity_error = np.linalg.norm(TTt - np.eye(T.shape[0])) / np.linalg.norm(np.eye(T.shape[0]))

    # Scale factor (geometric mean of singular values)
    scale = np.exp(np.mean(np.log(S + 1e-10)))

    # Determinant (volume change)
    det = np.prod(S[:min(50, len(S))])  # Avoid overflow

    return {
        'condition_number': float(cond),
        'effective_rank_90': eff_rank_90,
        'effective_rank_99': eff_rank_99,
        'identity_error': float(identity_error),
        'scale_factor': float(scale),
        'is_orthogonal': identity_error < 0.1,
        'top_singular_values': S[:10].tolist(),
    }


def test_compressed_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    T: np.ndarray,
    entry_layer: int,
    exit_layer: int,
    max_tokens: int = 20,
) -> tuple[str, str]:
    """Test generation with and without compression.

    Returns: (normal_output, compressed_output)
    """
    import mlx.core as mx

    # Normal generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    normal_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        normal_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    normal_output = tokenizer.decode(normal_generated)

    # Compressed generation (skip layers entry+1 to exit-1, apply T instead)
    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    compressed_generated = []

    # First pass: get to entry layer
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(inner_model.layers):
        if idx <= entry_layer:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        elif idx == exit_layer:
            # Apply compression transform T
            h_np = np.array(h.astype(mx.float32))
            h_transformed = (T @ h_np[0].T).T  # Apply T to all positions
            h = mx.array(h_transformed[np.newaxis, :, :]).astype(h.dtype)
            mx.eval(h)

            # Continue with exit layer
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        elif idx > exit_layer:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        # Skip layers entry+1 to exit-1

    # Final norm
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        logits = model(input_ids)  # Fallback
    mx.eval(logits)

    # Generate
    for _ in range(max_tokens):
        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        compressed_generated.append(next_token)

        # Continue generation normally (can't easily compress autoregressive)
        input_ids = mx.array([[next_token]])
        logits = model(input_ids)
        mx.eval(logits)

    compressed_output = tokenizer.decode(compressed_generated)

    return normal_output, compressed_output


def main():
    parser = argparse.ArgumentParser(description="Gram-preserving compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--entry-layer", type=int, default=None, help="Entry layer (default: auto)")
    parser.add_argument("--exit-layer", type=int, default=None, help="Exit layer (default: auto)")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    print(f"\n{'='*80}")
    print("GRAM-PRESERVING COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers")

    # Determine transmission section
    # For Qwen3-1.7B: layers 3-26 are transmission
    # For LFM2: layers 8-13 are highway
    if args.entry_layer is not None and args.exit_layer is not None:
        entry_layer = args.entry_layer
        exit_layer = args.exit_layer
    elif n_layers == 28:  # Qwen3-1.7B
        entry_layer = 2  # After encoder
        exit_layer = 26  # Before decoder
    elif n_layers == 16:  # LFM2
        entry_layer = 7  # First bottleneck
        exit_layer = 13  # Before second bottleneck
    else:
        entry_layer = n_layers // 4
        exit_layer = 3 * n_layers // 4

    print(f"Testing compression of layers {entry_layer+1} to {exit_layer-1}")
    print(f"  → Skipping {exit_layer - entry_layer - 1} layers")

    # Get entry and exit states
    print(f"\nCollecting states at layers {entry_layer} and {exit_layer}...")
    entry_states, exit_states = get_entry_exit_states(
        model, tokenizer, CONCEPTS, entry_layer, exit_layer
    )

    # Compute Gram matrices
    G_entry = compute_gram_matrix(entry_states)
    G_exit = compute_gram_matrix(exit_states)
    gram_similarity = compute_gram_similarity(G_entry, G_exit)

    print(f"\n{'='*80}")
    print("GRAM MATRIX ANALYSIS")
    print("="*80)
    print(f"Gram similarity (entry → exit): {gram_similarity:.6f}")
    print(f"  → 1.0 = perfectly preserved, <0.9 = significant change")

    # Compute energy
    entry_energy = np.mean(np.linalg.norm(entry_states, axis=1)**2)
    exit_energy = np.mean(np.linalg.norm(exit_states, axis=1)**2)
    energy_ratio = exit_energy / entry_energy

    print(f"\nEnergy balance:")
    print(f"  Entry energy: {entry_energy:.4f}")
    print(f"  Exit energy:  {exit_energy:.4f}")
    print(f"  Ratio:        {energy_ratio:.4f}")
    print(f"  → 1.0 = energy conserved")

    # Compute transmission transform
    print(f"\n{'='*80}")
    print("TRANSMISSION TRANSFORM")
    print("="*80)

    T, reconstruction_error = compute_transmission_transform(entry_states, exit_states)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"  → <0.01 = excellent, <0.1 = good")

    # Analyze transform
    transform_props = analyze_transform(T)
    print(f"\nTransform properties:")
    print(f"  Condition number: {transform_props['condition_number']:.2f}")
    print(f"  Effective rank (90%): {transform_props['effective_rank_90']}")
    print(f"  Effective rank (99%): {transform_props['effective_rank_99']}")
    print(f"  Is orthogonal: {transform_props['is_orthogonal']}")
    print(f"  Scale factor: {transform_props['scale_factor']:.4f}")
    print(f"  Identity error: {transform_props['identity_error']:.4f}")

    # Compression potential
    print(f"\n{'='*80}")
    print("COMPRESSION POTENTIAL")
    print("="*80)

    layers_skipped = exit_layer - entry_layer - 1
    layer_compression = n_layers / (n_layers - layers_skipped)

    # If T is low rank, we can compress even more
    T_rank = transform_props['effective_rank_99']
    hidden_dim = T.shape[0]
    T_compression = (hidden_dim * hidden_dim) / (2 * hidden_dim * T_rank)  # Low-rank factorization

    total_compression = layer_compression * T_compression if T_compression > 1 else layer_compression

    print(f"\nLayers: {n_layers} → {n_layers - layers_skipped}")
    print(f"  Layer compression: {layer_compression:.2f}x")
    print(f"\nTransform T: {hidden_dim}×{hidden_dim} → {hidden_dim}×{T_rank} × {T_rank}×{hidden_dim}")
    print(f"  Transform compression: {T_compression:.2f}x")
    print(f"\nTOTAL POTENTIAL COMPRESSION: {total_compression:.2f}x")

    # Test generation
    print(f"\n{'='*80}")
    print("GENERATION TEST")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        try:
            normal, compressed = test_compressed_generation(
                model, tokenizer, prompt, T, entry_layer, exit_layer, max_tokens=10
            )
            print(f"  Normal:     {normal}")
            print(f"  Compressed: {compressed}")

            # Compare
            if normal.strip() == compressed.strip():
                print(f"  → MATCH ✓")
            else:
                print(f"  → DIFFERENT")
        except Exception as e:
            print(f"  → Error: {e}")

    # The insight
    print(f"\n{'='*80}")
    print("COMPRESSION VIABILITY")
    print("="*80)

    viable = (
        gram_similarity > 0.99 and
        abs(energy_ratio - 1.0) < 0.1 and
        reconstruction_error < 0.1
    )

    if viable:
        print(f"""
COMPRESSION IS VIABLE!

The transmission section (layers {entry_layer+1} to {exit_layer-1}):
1. Preserves Gram matrix: {gram_similarity:.4f} ≈ 1.0 ✓
2. Preserves energy: {energy_ratio:.4f} ≈ 1.0 ✓
3. Can be reconstructed: error = {reconstruction_error:.4f} ✓

WHAT THIS MEANS:
- {layers_skipped} layers can be replaced with a single linear transform
- The transform preserves all relational structure
- Energy balance is maintained
- Potential compression: {total_compression:.1f}x

NEXT STEPS:
1. Factorize T into low-rank form: T = U @ V where U, V are {hidden_dim}×{T_rank}
2. Replace layers {entry_layer+1}-{exit_layer-1} with the factorized transform
3. Fine-tune on small dataset to recover any lost precision
4. Verify generation quality across diverse prompts
""")
    else:
        issues = []
        if gram_similarity <= 0.99:
            issues.append(f"Gram not preserved ({gram_similarity:.4f})")
        if abs(energy_ratio - 1.0) >= 0.1:
            issues.append(f"Energy not balanced ({energy_ratio:.4f})")
        if reconstruction_error >= 0.1:
            issues.append(f"Reconstruction error too high ({reconstruction_error:.4f})")

        print(f"""
COMPRESSION HAS CHALLENGES

Issues:
{chr(10).join('- ' + i for i in issues)}

The transmission section may be doing more than just transmission.
Consider:
1. Narrowing the layer range
2. Using a more sophisticated transform (not just linear)
3. Preserving residual stream separately
""")


if __name__ == "__main__":
    main()
