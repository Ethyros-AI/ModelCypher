#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Manifold Dimension Analysis for Lossless Compression
"""
KEY QUESTION: What is the intrinsic dimension of the activation manifold?

If rank(activations) << 4096, we can span it with relatively few samples.
If rank ≈ 4096, we need massive calibration.

This script:
1. Collects activations from MANY diverse prompts
2. Computes SVD to find effective rank
3. Determines minimum calibration size for full coverage
4. Tests reconstruction error on random held-out samples

Usage:
    python qwen3_manifold_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# MASSIVE diverse prompt set for manifold analysis
def generate_diverse_prompts() -> List[str]:
    """Generate thousands of diverse prompts to sample the manifold."""
    prompts = []

    # Geography - all countries
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Iran", "Iraq", "Syria", "Pakistan", "Afghanistan", "Thailand", "Vietnam",
        "Indonesia", "Philippines", "Malaysia", "Singapore", "South Korea", "Taiwan",
        "Greece", "Poland", "Sweden", "Norway", "Finland", "Denmark", "Netherlands",
        "Belgium", "Austria", "Switzerland", "Portugal", "Ireland", "Scotland",
        "Nigeria", "Kenya", "South Africa", "Morocco", "Algeria", "Ethiopia",
        "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador",
        "New Zealand", "Fiji", "Mongolia", "Nepal", "Sri Lanka", "Bangladesh",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"{c} is known for")
        prompts.append(f"The population of {c} is")

    # Math - extensive
    for a in range(1, 20):
        for b in range(1, 20):
            prompts.append(f"{a} + {b} =")
            prompts.append(f"{a} * {b} =")
            prompts.append(f"{a} - {b} =")

    # Science
    elements = [
        "Hydrogen", "Helium", "Lithium", "Carbon", "Nitrogen", "Oxygen",
        "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon",
        "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
        "Iron", "Copper", "Zinc", "Silver", "Gold", "Mercury", "Lead", "Uranium",
    ]
    for e in elements:
        prompts.append(f"{e} has atomic number")
        prompts.append(f"The symbol for {e} is")
        prompts.append(f"{e} melts at")

    # Language patterns
    words = [
        "happy", "sad", "big", "small", "hot", "cold", "light", "dark",
        "good", "bad", "old", "young", "fast", "slow", "hard", "soft",
        "wet", "dry", "rich", "poor", "strong", "weak", "tall", "short",
        "wide", "narrow", "thick", "thin", "deep", "shallow", "heavy", "light",
    ]
    for w in words:
        prompts.append(f"The opposite of {w} is")
        prompts.append(f"A synonym for {w} is")
        prompts.append(f"Something that is {w}")

    # Code patterns
    code_starts = [
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "try:", "except:", "with ", "async def ", "lambda ", "@property",
        "def __init__(self", "def main()", "def test_", "def get_", "def set_",
        "class User", "class Config", "class Model", "class Handler",
        "import numpy", "import pandas", "import torch", "from typing import",
    ]
    prompts.extend(code_starts)

    # Questions
    q_words = ["What", "How", "Why", "When", "Where", "Who", "Which", "Can", "Could", "Should"]
    q_verbs = ["is", "are", "was", "were", "do", "does", "did", "will", "can", "should"]
    for qw in q_words:
        for qv in q_verbs:
            prompts.append(f"{qw} {qv} the")
            prompts.append(f"{qw} {qv} this")

    # Conversational
    conv = [
        "Actually,", "However,", "Therefore,", "Furthermore,", "Moreover,",
        "In fact,", "To be honest,", "Honestly,", "Frankly,", "Basically,",
        "Essentially,", "Fundamentally,", "In essence,", "In reality,",
        "The truth is,", "The fact is,", "The thing is,", "Here's the thing,",
        "Let me explain", "I think that", "In my opinion,", "It seems that",
        "Interestingly,", "Surprisingly,", "Notably,", "Importantly,",
    ]
    prompts.extend(conv)

    # Instructions
    instr = [
        "First,", "Then,", "Next,", "After that,", "Finally,",
        "Step 1:", "Step 2:", "Step 3:", "To begin,", "To start,",
        "Make sure to", "Remember to", "Don't forget to", "Be sure to",
        "You should", "You must", "You can", "You need to",
    ]
    prompts.extend(instr)

    # Random sentences
    subjects = ["The cat", "A dog", "My friend", "The teacher", "Scientists", "People"]
    verbs = ["is", "was", "will be", "has been", "can be", "should be"]
    adjectives = ["happy", "interesting", "important", "beautiful", "complex", "simple"]
    for s in subjects:
        for v in verbs:
            for a in adjectives:
                prompts.append(f"{s} {v} {a}")

    return prompts


def collect_activations_batch(model, tokenizer, prompts: List[str], layer_idx: int) -> np.ndarray:
    """Collect activations at a specific layer for many prompts."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    activations = []

    for i, prompt in enumerate(prompts):
        if i % 100 == 0:
            logger.info(f"  Processing prompt {i}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_vec = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                activations.append(h_vec)
                break
            h = layer(h, mask, None)
            mx.eval(h)

    return np.stack(activations, axis=0)  # (n_samples, hidden_dim)


def analyze_manifold_dimension(X: np.ndarray, threshold: float = 0.99) -> Tuple[int, np.ndarray]:
    """
    Analyze the intrinsic dimension of the activation manifold.

    Returns:
        effective_rank: Number of singular values needed to capture `threshold` variance
        singular_values: All singular values
    """
    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Compute cumulative variance explained
    variance_explained = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(variance_explained)

    # Find effective rank
    effective_rank = np.searchsorted(cumulative_variance, threshold) + 1

    return effective_rank, S, cumulative_variance


def test_spanning_reconstruction(X_all: np.ndarray, Y_all: np.ndarray,
                                  n_calib: int, n_test: int = 100) -> float:
    """
    Test if n_calib samples can span the manifold for reconstruction.

    Uses first n_calib samples for calibration, tests on remaining.
    """
    # Split into calibration and test
    X_calib = X_all[:n_calib].T  # (hidden_dim, n_calib)
    Y_calib = Y_all[:n_calib].T  # (hidden_dim, n_calib)

    X_test = X_all[n_calib:n_calib+n_test].T  # (hidden_dim, n_test)
    Y_test = Y_all[n_calib:n_calib+n_test].T  # (hidden_dim, n_test)

    # Compute T from calibration
    T = Y_calib @ np.linalg.pinv(X_calib)

    # Reconstruct test outputs
    Y_pred = T @ X_test

    # Compute reconstruction error
    errors = []
    for i in range(n_test):
        err = np.linalg.norm(Y_test[:, i] - Y_pred[:, i]) / np.linalg.norm(Y_test[:, i])
        errors.append(err)

    return np.mean(errors), np.max(errors), np.std(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=2000,
                       help="Maximum prompts for manifold analysis")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    # Layer boundaries from previous analysis
    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("MANIFOLD DIMENSION ANALYSIS")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Analyzing layers {start_layer} -> {end_layer}")

    # Generate diverse prompts
    print(f"\n{'='*70}")
    print("GENERATING DIVERSE PROMPTS")
    print("="*70)
    prompts = generate_diverse_prompts()
    prompts = prompts[:args.max_prompts]  # Limit for speed
    print(f"Generated {len(prompts)} diverse prompts")

    # Collect activations at start and end of compression
    print(f"\n{'='*70}")
    print("COLLECTING ACTIVATIONS")
    print("="*70)

    print(f"\nCollecting at layer {start_layer} (input)...")
    X_all = collect_activations_batch(model, tokenizer, prompts, start_layer)
    print(f"X shape: {X_all.shape}")

    print(f"\nCollecting at layer {end_layer} (output)...")
    # Need to run through layers to get output
    from mlx_lm.models.base import create_attention_mask
    Y_list = []
    for i, prompt in enumerate(prompts):
        if i % 100 == 0:
            logger.info(f"  Processing prompt {i}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_vec = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_list.append(h_vec)
                break

    Y_all = np.stack(Y_list, axis=0)
    print(f"Y shape: {Y_all.shape}")

    # Analyze input manifold dimension
    print(f"\n{'='*70}")
    print("INPUT MANIFOLD ANALYSIS (Layer {start_layer})")
    print("="*70)

    effective_rank_99, S_input, cumvar_input = analyze_manifold_dimension(X_all, threshold=0.99)
    effective_rank_999, _, _ = analyze_manifold_dimension(X_all, threshold=0.999)
    effective_rank_9999, _, _ = analyze_manifold_dimension(X_all, threshold=0.9999)

    print(f"\nSingular value analysis:")
    print(f"  Total dimensions: {hidden_dim}")
    print(f"  Samples collected: {len(prompts)}")
    print(f"  Max possible rank: {min(len(prompts), hidden_dim)}")
    print(f"\nEffective rank (variance captured):")
    print(f"  99% variance:   {effective_rank_99} dimensions")
    print(f"  99.9% variance: {effective_rank_999} dimensions")
    print(f"  99.99% variance: {effective_rank_9999} dimensions")

    print(f"\nTop 20 singular values:")
    for i in range(min(20, len(S_input))):
        print(f"  σ_{i+1}: {S_input[i]:.4f} (cumulative: {cumvar_input[i]*100:.2f}%)")

    # Analyze output manifold dimension
    print(f"\n{'='*70}")
    print("OUTPUT MANIFOLD ANALYSIS (Layer {end_layer})")
    print("="*70)

    effective_rank_99_out, S_output, cumvar_output = analyze_manifold_dimension(Y_all, threshold=0.99)
    effective_rank_999_out, _, _ = analyze_manifold_dimension(Y_all, threshold=0.999)

    print(f"\nEffective rank (variance captured):")
    print(f"  99% variance:   {effective_rank_99_out} dimensions")
    print(f"  99.9% variance: {effective_rank_999_out} dimensions")

    # Test reconstruction with varying calibration sizes
    print(f"\n{'='*70}")
    print("RECONSTRUCTION ERROR VS CALIBRATION SIZE")
    print("="*70)

    # Shuffle for randomness
    indices = np.random.permutation(len(prompts))
    X_shuffled = X_all[indices]
    Y_shuffled = Y_all[indices]

    calib_sizes = [50, 100, 200, 300, 500, 750, 1000, 1500]
    calib_sizes = [c for c in calib_sizes if c + 100 <= len(prompts)]

    print(f"\n{'Calibration':>12} | {'Mean Error':>12} | {'Max Error':>12} | {'Std':>12}")
    print("-" * 55)

    for n_calib in calib_sizes:
        mean_err, max_err, std_err = test_spanning_reconstruction(
            X_shuffled, Y_shuffled, n_calib, n_test=100
        )
        print(f"{n_calib:>12} | {mean_err:>12.2e} | {max_err:>12.2e} | {std_err:>12.2e}")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print("="*70)
    print(f"""
Manifold Analysis Results:
- Input manifold effective rank (99%): {effective_rank_99}
- Input manifold effective rank (99.9%): {effective_rank_999}
- Hidden dimension: {hidden_dim}

INTERPRETATION:
If effective_rank << {hidden_dim}:
  → The manifold is low-dimensional
  → We need ~{effective_rank_999} calibration samples for full coverage
  → Lossless compression is achievable with finite calibration

If effective_rank ≈ {hidden_dim}:
  → The manifold is high-dimensional
  → We need many more samples or piecewise approximation
  → May need category-specific T matrices

RECOMMENDATION:
Based on the reconstruction error curve above, use calibration size where
max_error < 1e-10 for lossless compression across the full manifold.
""")


if __name__ == "__main__":
    main()
