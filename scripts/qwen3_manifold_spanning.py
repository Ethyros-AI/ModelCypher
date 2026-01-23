#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Systematic Manifold Spanning Calibration
"""
GOAL: Build a calibration set that SPANS the activation manifold.

INSIGHT: The manifold has low intrinsic dimension (~53-700).
If we capture the principal directions, we span the manifold.

APPROACH:
1. Start with diverse seed calibration
2. Compute SVD to find current span
3. For new prompts, compute orthogonal residual
4. If residual > threshold, ADD to calibration (it's a new direction!)
5. Repeat until all test prompts have residual ≈ 0

This is essentially ACTIVE LEARNING for manifold coverage.

Usage:
    python qwen3_manifold_spanning.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_activation(model, tokenizer, prompt: str, layer_idx: int) -> np.ndarray:
    """Collect activation at a specific layer for a single prompt."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

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
            return h_vec
        h = layer(h, mask, None)
        mx.eval(h)

    return np.zeros(inner_model.embed_tokens.weight.shape[1])


def collect_activation_pair(model, tokenizer, prompt: str,
                            start_layer: int, end_layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations at start and end layers."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    x_in = None
    x_out = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            x_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)

        h = layer(h, mask, None)
        mx.eval(h)

        if idx == end_layer:
            x_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            break

    return x_in, x_out


def compute_orthogonal_residual(x: np.ndarray, U: np.ndarray, rank: int) -> float:
    """
    Compute orthogonal residual using precomputed SVD.

    U: left singular vectors from SVD of calibration matrix
    rank: number of significant singular vectors
    """
    # Project onto span
    U_r = U[:, :rank]
    x_proj = U_r @ (U_r.T @ x)

    # Residual
    x_orth = x - x_proj
    residual = np.linalg.norm(x_orth) / (np.linalg.norm(x) + 1e-10)

    return residual


def test_generation(model, tokenizer, prompt: str, T: np.ndarray,
                    start_layer: int, end_layer: int) -> Tuple[bool, str, str]:
    """Test generation with T."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    # T-based
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in

            if np.any(np.isnan(h_out)) or np.any(np.isinf(h_out)):
                return False, tokenizer.decode([normal_token]), "[NaN]"

            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass
        else:
            h = layer(h, mask, None)
            mx.eval(h)

    h = inner_model.norm(h)
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    t_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    return normal_token == t_token, tokenizer.decode([normal_token]), tokenizer.decode([t_token])


def generate_candidate_prompts() -> List[str]:
    """Generate a large pool of candidate prompts for spanning."""
    prompts = []

    # All countries
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
        "Zimbabwe", "Nigeria", "Kenya", "South Africa", "Morocco", "Algeria",
        "Peru", "Colombia", "Venezuela", "Mongolia", "Nepal", "Bangladesh",
        "Ukraine", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Serbia",
        "Greece", "Portugal", "Ireland", "Belgium", "Austria", "Finland", "Denmark",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"The population of {c} is")
        prompts.append(f"{c} is known for")

    # Math (full range)
    for a in range(1, 30):
        for b in range(1, 30):
            prompts.append(f"{a} + {b} =")

    # Code - extensive
    code = [
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "try:", "except:", "finally:", "with ", "yield ", "async def ", "await ",
        "def main():", "def __init__(", "def test_", "def get_", "def set_",
        "def calculate_", "def compute_", "def process_", "def fibonacci(",
        "def factorial(", "def sort(", "def search(", "def parse(",
        "class User:", "class Config:", "class Model:", "class Handler:",
        "class Service:", "class Controller:", "class Factory:",
        "import numpy", "import pandas", "import torch", "import os", "import sys",
        "from typing import", "from collections import", "from dataclasses import",
        "SELECT * FROM", "INSERT INTO", "UPDATE", "DELETE FROM",
        "console.log(", "document.getElementById(", "function ", "const ", "let ",
    ]
    prompts.extend(code)

    # Questions - comprehensive
    questions = [
        "What is", "What are", "What was", "What were", "What will",
        "How do", "How does", "How did", "How can", "How should", "How would",
        "Why is", "Why are", "Why was", "Why were", "Why do", "Why does",
        "When is", "When was", "When will", "When did", "When does",
        "Where is", "Where are", "Where was", "Where were", "Where do",
        "Who is", "Who are", "Who was", "Who were", "Who will",
        "Which is", "Which are", "Which was", "Which one", "Which of",
        "Can you", "Could you", "Would you", "Should I", "Do I",
        "What causes", "Why do birds", "How many planets", "What is the speed of",
    ]
    prompts.extend(questions)

    # Conversational
    conv = [
        "Actually,", "However,", "Therefore,", "In fact,", "To be honest,",
        "Basically,", "Essentially,", "Well,", "So,", "Look,", "Listen,",
        "Furthermore,", "Moreover,", "Nevertheless,", "Nonetheless,",
        "Consequently,", "Subsequently,", "Meanwhile,", "Otherwise,",
        "Honestly,", "Frankly,", "Truthfully,", "Seriously,", "Obviously,",
    ]
    prompts.extend(conv)

    # Science
    science = [
        "The speed of light is", "The speed of sound is", "Gravity is",
        "Photosynthesis produces", "DNA stands for", "RNA is", "ATP is",
        "Hydrogen has atomic number", "Oxygen has atomic number",
        "The boiling point of water is", "The melting point of ice is",
        "Newton's first law", "Einstein's theory of", "The Big Bang",
        "Black holes are", "Quantum mechanics", "The atom consists of",
    ]
    prompts.extend(science)

    # Philosophy/Abstract
    abstract = [
        "The meaning of life is", "Consciousness is", "Free will is",
        "The nature of reality", "Truth is", "Justice means",
        "Love is", "Happiness is", "Success is", "Failure is",
    ]
    prompts.extend(abstract)

    # Technology
    tech = [
        "Artificial intelligence can", "Machine learning is", "Neural networks",
        "The internet works by", "Computers process", "Algorithms are",
        "Data science involves", "Cloud computing", "Cybersecurity",
        "Blockchain is", "Cryptocurrency", "Virtual reality",
    ]
    prompts.extend(tech)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--residual-threshold", type=float, default=0.05,
                       help="Add prompt to calibration if residual > threshold")
    parser.add_argument("--max-calibration", type=int, default=500)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("SYSTEMATIC MANIFOLD SPANNING")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Residual threshold: {args.residual_threshold*100:.0f}%")

    # Generate candidate pool
    candidates = generate_candidate_prompts()
    print(f"Candidate pool: {len(candidates)} prompts")

    # Start with empty calibration
    X_list = []
    Y_list = []
    calib_prompts = []

    # Add first prompt as seed
    seed = candidates[0]
    x_in, x_out = collect_activation_pair(model, tokenizer, seed, start_layer, end_layer)
    X_list.append(x_in)
    Y_list.append(x_out)
    calib_prompts.append(seed)

    print(f"\nBuilding spanning calibration set...")
    print(f"{'Iteration':>10} | {'Added':>6} | {'Total':>6} | {'Rank':>6}")
    print("-" * 45)

    # Iteratively add prompts that expand the span
    iteration = 0
    added_this_iter = 1

    while added_this_iter > 0 and len(calib_prompts) < args.max_calibration:
        iteration += 1
        added_this_iter = 0

        # Stack current calibration
        X_calib = np.stack(X_list, axis=0)  # (n, d)

        # Compute SVD
        U, S, Vt = np.linalg.svd(X_calib.T, full_matrices=False)
        tol = 1e-8 * S[0] if len(S) > 0 else 1e-8
        rank = np.sum(S > tol)

        # Test each candidate
        for prompt in candidates:
            if prompt in calib_prompts:
                continue

            x_test = collect_activation(model, tokenizer, prompt, start_layer)
            residual = compute_orthogonal_residual(x_test, U, rank)

            if residual > args.residual_threshold:
                # This prompt has significant component outside current span
                x_in, x_out = collect_activation_pair(model, tokenizer, prompt, start_layer, end_layer)
                X_list.append(x_in)
                Y_list.append(x_out)
                calib_prompts.append(prompt)
                added_this_iter += 1

                if len(calib_prompts) >= args.max_calibration:
                    break

        print(f"{iteration:>10} | {added_this_iter:>6} | {len(calib_prompts):>6} | {rank:>6}")

        if added_this_iter == 0:
            print("\nConverged! No more prompts expand the span.")

    # Final calibration
    X_calib = np.stack(X_list, axis=0).T  # (d, n)
    Y_calib = np.stack(Y_list, axis=0).T  # (d, n)

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"Final calibration size: {len(calib_prompts)}")
    print(f"X shape: {X_calib.shape}")

    # Compute final SVD rank
    U, S, Vt = np.linalg.svd(X_calib, full_matrices=False)
    tol = 1e-8 * S[0]
    final_rank = np.sum(S > tol)
    print(f"Numerical rank: {final_rank}")

    # Compute T using regularization to avoid numerical issues
    print(f"\nComputing T (regularized)...")
    lambda_reg = 1e-8
    XXT = X_calib @ X_calib.T + lambda_reg * np.eye(hidden_dim)
    T = Y_calib @ X_calib.T @ np.linalg.inv(XXT)
    print(f"T shape: {T.shape}")

    # Test on ALL candidates
    print(f"\n{'='*70}")
    print("TESTING ON ALL CANDIDATES")
    print("="*70)

    # Sample test set
    test_prompts = [
        "The capital of Zimbabwe is",
        "25 + 37 =",
        "def fibonacci(",
        "What causes earthquakes",
        "Why do birds migrate",
        "The meaning of life is",
        "Artificial intelligence can",
        "The speed of light is",
        "Quantum mechanics",
        "Black holes are",
        "SELECT * FROM",
        "console.log(",
        "The nature of reality",
        "Neural networks",
        "The Big Bang",
    ]

    matches = 0
    total = 0

    print(f"\n{'Prompt':<35} | {'Residual':>8} | {'Match':>6}")
    print("-" * 60)

    for prompt in test_prompts:
        x_test = collect_activation(model, tokenizer, prompt, start_layer)
        residual = compute_orthogonal_residual(x_test, U, final_rank)

        match, expected, got = test_generation(model, tokenizer, prompt, T, start_layer, end_layer)

        status = "OK" if match else "FAIL"
        print(f"{prompt[:35]:<35} | {residual*100:>7.2f}% | {status:>6}")

        if match:
            matches += 1
        total += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {matches}/{total} ({100*matches/total:.0f}%)")
    print("="*70)

    # Conclusion
    print(f"""
CONCLUSION:

The closed-form solution T = Y @ pinv(X) achieves lossless compression
when span(X) contains the manifold.

Calibration strategy:
1. Start with diverse seed prompts
2. Add prompts that have orthogonal components (expand the span)
3. Stop when no more prompts expand the span significantly

Final span has rank {final_rank} ≤ {len(calib_prompts)} calibration samples.

If accuracy < 100%, increase calibration diversity or lower threshold.
""")


if __name__ == "__main__":
    main()
