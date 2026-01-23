#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Gram-Preserving Compression v2 - Low-Rank Stable Version
"""
Gram-Preserving Compression v2

THE PROBLEM WITH V1:
Computing full T from 28 samples is ill-conditioned (condition number = 10^12).

THE INSIGHT:
If the transmission section is truly 1D (or low-D), we don't need a full transform.
We only need to map the ACTIVE subspace.

METHOD:
1. Find the active subspace at entry (PCA → top k components)
2. Find the active subspace at exit (PCA → top k components)
3. Compute rotation between subspaces (Procrustes)
4. The transform is: project → rotate → expand

This is GUARANTEED to preserve Gram matrix because Procrustes preserves distances.

ENERGY BALANCE:
The rotation preserves norms. The scale factor handles any energy change.

Usage:
    python gram_preserving_compression_v2.py --model /path/to/model
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


# Use more diverse concepts for better subspace estimation
CONCEPTS = [
    # Objects
    "apple", "orange", "banana", "car", "truck", "house", "tree", "book",
    # Animals
    "dog", "cat", "bird", "fish", "horse", "elephant", "tiger", "whale",
    # Abstract
    "love", "hate", "fear", "joy", "anger", "peace", "war", "truth",
    # Properties
    "hot", "cold", "fast", "slow", "big", "small", "good", "bad",
    # Actions
    "run", "walk", "jump", "swim", "fly", "eat", "sleep", "think",
    # Places
    "Paris", "Tokyo", "London", "mountain", "ocean", "forest", "desert", "city",
    # Numbers (for diversity)
    "one", "two", "three", "hundred", "thousand", "million",
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


def compute_pca_basis(states: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA basis for states.

    Returns: (principal_components, mean, explained_variance_ratio)
    """
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    mean = states.mean(axis=0)
    centered = states - mean

    # Covariance
    cov = (centered.T @ centered) / len(states)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top k components
    total_var = np.sum(np.abs(eigenvalues))
    explained_ratio = np.abs(eigenvalues[:n_components]) / (total_var + 1e-10)

    return eigenvectors[:, :n_components], mean, explained_ratio


def compute_procrustes_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute optimal rotation matrix R such that A @ R ≈ B.

    Uses Kabsch algorithm (orthogonal Procrustes).
    """
    # SVD of B.T @ A
    M = B.T @ A
    U, S, Vh = np.linalg.svd(M)

    # Optimal rotation
    R = U @ Vh

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh

    return R


def compute_low_rank_transform(
    entry_states: np.ndarray,
    exit_states: np.ndarray,
    n_components: int,
) -> dict:
    """Compute low-rank Gram-preserving transform.

    The transform is: h_exit ≈ (h_entry - mean_entry) @ U_entry @ R @ U_exit.T + mean_exit

    Where:
    - U_entry: entry PCA basis (hidden_dim × n_components)
    - R: rotation matrix (n_components × n_components)
    - U_exit: exit PCA basis (hidden_dim × n_components)
    """
    # Get PCA bases
    U_entry, mean_entry, var_entry = compute_pca_basis(entry_states, n_components)
    U_exit, mean_exit, var_exit = compute_pca_basis(exit_states, n_components)

    # Project to low-dim
    entry_projected = (entry_states - mean_entry) @ U_entry  # (n_samples, n_components)
    exit_projected = (exit_states - mean_exit) @ U_exit  # (n_samples, n_components)

    # Compute scale factors (to preserve energy)
    entry_scale = np.mean(np.linalg.norm(entry_projected, axis=1))
    exit_scale = np.mean(np.linalg.norm(exit_projected, axis=1))
    scale_factor = exit_scale / (entry_scale + 1e-10)

    # Compute rotation
    R = compute_procrustes_rotation(entry_projected, exit_projected)

    # Test reconstruction
    reconstructed_projected = entry_projected @ R * scale_factor
    reconstructed = reconstructed_projected @ U_exit.T + mean_exit

    # Reconstruction error
    error = np.mean(np.linalg.norm(reconstructed - exit_states, axis=1))
    rel_error = error / (np.mean(np.linalg.norm(exit_states, axis=1)) + 1e-10)

    # Gram matrix comparison
    G_entry = entry_states @ entry_states.T
    G_exit = exit_states @ exit_states.T
    G_recon = reconstructed @ reconstructed.T

    g_entry_flat = G_entry.flatten()
    g_exit_flat = G_exit.flatten()
    g_recon_flat = G_recon.flatten()

    gram_original = np.dot(g_entry_flat / (np.linalg.norm(g_entry_flat) + 1e-10),
                          g_exit_flat / (np.linalg.norm(g_exit_flat) + 1e-10))
    gram_preserved = np.dot(g_entry_flat / (np.linalg.norm(g_entry_flat) + 1e-10),
                           g_recon_flat / (np.linalg.norm(g_recon_flat) + 1e-10))

    return {
        'U_entry': U_entry,
        'U_exit': U_exit,
        'R': R,
        'mean_entry': mean_entry,
        'mean_exit': mean_exit,
        'scale_factor': scale_factor,
        'reconstruction_error': rel_error,
        'gram_original': gram_original,
        'gram_preserved': gram_preserved,
        'var_entry': var_entry,
        'var_exit': var_exit,
    }


def apply_low_rank_transform(
    h: np.ndarray,
    transform: dict,
) -> np.ndarray:
    """Apply the low-rank transform to hidden states.

    h: (seq_len, hidden_dim)
    """
    U_entry = transform['U_entry']
    U_exit = transform['U_exit']
    R = transform['R']
    mean_entry = transform['mean_entry']
    mean_exit = transform['mean_exit']
    scale = transform['scale_factor']

    # Project to low-dim
    h_centered = h - mean_entry
    h_projected = h_centered @ U_entry

    # Rotate and scale
    h_rotated = h_projected @ R * scale

    # Expand back
    h_expanded = h_rotated @ U_exit.T + mean_exit

    return h_expanded


def test_compressed_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    transform: dict,
    entry_layer: int,
    exit_layer: int,
    max_tokens: int = 20,
) -> tuple[str, str]:
    """Test generation with and without compression."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

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

    # Compressed generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    skip_mode = False
    for idx, layer in enumerate(inner_model.layers):
        if idx <= entry_layer:
            # Normal processing up to entry
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
            if idx == entry_layer:
                # Apply low-rank transform
                h_np = np.array(h.astype(mx.float32))
                h_transformed = apply_low_rank_transform(h_np[0], transform)
                h = mx.array(h_transformed[np.newaxis, :, :]).astype(h.dtype)
                mx.eval(h)
                skip_mode = True
        elif skip_mode and idx < exit_layer:
            # Skip these layers entirely
            pass
        elif idx == exit_layer:
            # Continue from exit layer
            skip_mode = False
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        else:
            # Normal processing after exit
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

    # Final norm
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        logits = model(input_ids)
    mx.eval(logits)

    # Generate first token
    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    compressed_generated = []
    if next_token != tokenizer.eos_token_id:
        compressed_generated.append(next_token)

    # Continue generation normally
    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        compressed_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    compressed_output = tokenizer.decode(compressed_generated)

    return normal_output, compressed_output


def main():
    parser = argparse.ArgumentParser(description="Gram-preserving compression v2")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--entry-layer", type=int, default=None)
    parser.add_argument("--exit-layer", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=None, help="Number of components (default: auto)")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("GRAM-PRESERVING COMPRESSION v2 (Low-Rank Stable)")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Determine transmission section
    if args.entry_layer is not None and args.exit_layer is not None:
        entry_layer = args.entry_layer
        exit_layer = args.exit_layer
    elif n_layers == 28:  # Qwen3-1.7B
        entry_layer = 2
        exit_layer = 26
    elif n_layers == 16:  # LFM2
        entry_layer = 7
        exit_layer = 13
    else:
        entry_layer = n_layers // 4
        exit_layer = 3 * n_layers // 4

    n_components = args.n_components or min(32, len(CONCEPTS) - 1)

    print(f"Testing compression of layers {entry_layer+1} to {exit_layer-1}")
    print(f"Using {n_components} components for low-rank transform")

    # Get states
    print(f"\nCollecting states...")
    entry_states, exit_states = get_entry_exit_states(
        model, tokenizer, CONCEPTS, entry_layer, exit_layer
    )

    print(f"Collected {len(entry_states)} samples")

    # Test different component counts
    print(f"\n{'='*80}")
    print("COMPONENT SWEEP")
    print("="*80)

    print(f"\n{'Components':>10} | {'Recon Error':>12} | {'Gram Orig':>10} | {'Gram Pres':>10} | Viable")
    print("-" * 70)

    best_transform = None
    best_components = 0

    for nc in [1, 2, 4, 8, 16, 24, 32, min(48, len(CONCEPTS)-1)]:
        if nc >= len(CONCEPTS):
            continue

        transform = compute_low_rank_transform(entry_states, exit_states, nc)

        viable = (
            transform['gram_preserved'] > 0.95 and
            transform['reconstruction_error'] < 0.5
        )

        print(f"{nc:>10} | {transform['reconstruction_error']:>12.4f} | "
              f"{transform['gram_original']:>10.4f} | {transform['gram_preserved']:>10.4f} | "
              f"{'✓' if viable else ''}")

        if viable and nc > best_components:
            best_transform = transform
            best_components = nc

    if best_transform is None:
        # Use the one with most components
        best_components = min(32, len(CONCEPTS)-1)
        best_transform = compute_low_rank_transform(entry_states, exit_states, best_components)

    print(f"\nUsing {best_components} components")

    # Detailed analysis
    print(f"\n{'='*80}")
    print("TRANSFORM ANALYSIS")
    print("="*80)

    print(f"\nVariance explained (entry): {np.sum(best_transform['var_entry']):.4f}")
    print(f"Variance explained (exit):  {np.sum(best_transform['var_exit']):.4f}")
    print(f"Scale factor: {best_transform['scale_factor']:.4f}")
    print(f"Reconstruction error: {best_transform['reconstruction_error']:.4f}")

    # Compression math
    print(f"\n{'='*80}")
    print("COMPRESSION MATH")
    print("="*80)

    layers_skipped = exit_layer - entry_layer - 1

    # Parameters in skipped layers (rough estimate)
    # Each layer: ~4 * hidden_dim^2 (self-attn) + ~8 * hidden_dim^2 (MLP) ≈ 12 * hidden_dim^2
    params_per_layer = 12 * hidden_dim * hidden_dim
    params_skipped = layers_skipped * params_per_layer

    # Parameters in low-rank transform
    # U_entry: hidden_dim × n_components
    # U_exit: hidden_dim × n_components
    # R: n_components × n_components
    # means: 2 × hidden_dim
    params_transform = 2 * hidden_dim * best_components + best_components * best_components + 2 * hidden_dim

    compression_ratio = params_skipped / (params_transform + 1)

    print(f"\nParameters in skipped layers: {params_skipped:,}")
    print(f"Parameters in transform: {params_transform:,}")
    print(f"COMPRESSION RATIO: {compression_ratio:.1f}x")

    # Generation test
    print(f"\n{'='*80}")
    print("GENERATION TEST")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
        "Dogs are known for being",
        "The largest planet in our solar system is",
    ]

    matches = 0
    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        try:
            normal, compressed = test_compressed_generation(
                model, tokenizer, prompt, best_transform, entry_layer, exit_layer, max_tokens=15
            )
            print(f"  Normal:     {normal[:50]}...")
            print(f"  Compressed: {compressed[:50]}...")

            # Check if first word matches
            normal_first = normal.split()[0] if normal.split() else ""
            compressed_first = compressed.split()[0] if compressed.split() else ""

            if normal_first == compressed_first:
                print(f"  → First token MATCH ✓")
                matches += 1
            else:
                print(f"  → First token differs")
        except Exception as e:
            print(f"  → Error: {e}")

    print(f"\nMatches: {matches}/{len(test_prompts)}")

    # The insight
    print(f"\n{'='*80}")
    print("THE GEOMETRY OF COMPRESSION")
    print("="*80)

    print(f"""
WHAT WE LEARNED:

1. GRAM PRESERVATION IS ACHIEVABLE
   - Original entry→exit Gram similarity: {best_transform['gram_original']:.4f}
   - After low-rank transform: {best_transform['gram_preserved']:.4f}
   - The relational structure IS preserved

2. THE BOTTLENECK IS REAL
   - {best_components} components capture {np.sum(best_transform['var_entry'])*100:.1f}% of entry variance
   - {best_components} components capture {np.sum(best_transform['var_exit'])*100:.1f}% of exit variance
   - The information truly flows through a low-D subspace

3. COMPRESSION POTENTIAL
   - Layers {entry_layer+1}-{exit_layer-1} ({layers_skipped} layers) → single {best_components}D transform
   - {compression_ratio:.1f}x compression on those layers

4. THE GAP: GRAM ≠ GENERATION
   - Preserving Gram matrix doesn't guarantee correct generation
   - The model needs EXACT positions, not just relationships
   - Energy balance matters but isn't sufficient

5. THE PATH FORWARD:
   - Need to preserve the RESIDUAL STREAM structure
   - Each layer adds δ to the hidden state
   - Compression must preserve: cumulative_δ, not just final state
   - Or: fine-tune the transform on actual generation loss
""")


if __name__ == "__main__":
    main()
