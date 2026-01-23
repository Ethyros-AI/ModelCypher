#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Test the Span Coverage Hypothesis
"""
HYPOTHESIS: T = Y @ pinv(X) IS the closed-form solution.
The limitation is COVERAGE, not the math.

If span(X) contains the manifold, reconstruction is EXACT.

This script tests:
1. For each held-out input, compute its projection onto span(X)
2. Measure the "orthogonal residual" - the part NOT in span(X)
3. If residual ≈ 0, the input IS in span(X) and T should work
4. If residual > 0, the input is OUTSIDE span(X) - need more calibration

PREDICTION: Failures correlate with large orthogonal residuals.

Usage:
    python qwen3_span_coverage.py --model /path/to/model
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


def generate_calibration() -> List[str]:
    """Generate calibration set."""
    prompts = []

    # Geography
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Math
    for a in range(1, 16):
        for b in range(1, 16):
            prompts.append(f"{a} + {b} =")

    # Code
    prompts.extend([
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "def main():", "def __init__(self", "class Config:", "import numpy",
    ])

    # Questions
    prompts.extend([
        "What is", "How do", "Why is", "When was", "Where is", "Who is",
        "Can you", "Could you", "Should I", "Is it", "Are there",
    ])

    # Conversational
    prompts.extend([
        "Actually,", "However,", "Therefore,", "In fact,", "To be honest,",
        "Basically,", "Essentially,", "Well,", "So,", "Look,",
    ])

    # Instructions
    prompts.extend([
        "First,", "Then,", "Next,", "Finally,", "Step 1:", "To begin,",
        "Make sure to", "Remember to", "Note that",
    ])

    # Science
    prompts.extend([
        "Hydrogen has atomic number", "The melting point of gold is",
        "The speed of light is", "DNA stands for", "Photosynthesis produces",
    ])

    # Random
    prompts.extend([
        "The meaning of life is", "Artificial intelligence can",
        "The future of technology", "The population of Iceland is",
        "The currency of Switzerland is",
    ])

    return prompts


def collect_activations(model, tokenizer, prompts: List[str], layer_idx: int) -> np.ndarray:
    """Collect activations at a specific layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    activations = []

    for prompt in prompts:
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

    return np.stack(activations, axis=0)  # (n, d)


def compute_orthogonal_residual(x: np.ndarray, X_basis: np.ndarray) -> Tuple[float, float]:
    """
    Compute how much of x lies outside span(X_basis).

    X_basis: (n, d) matrix whose rows span a subspace
    x: (d,) vector to test

    Returns:
        residual_norm: ||x - proj(x onto span)||
        relative_residual: residual_norm / ||x||
    """
    # Compute projection onto span(X_basis)
    # proj(x) = X_basis.T @ (X_basis @ X_basis.T)^{-1} @ X_basis @ x
    # But X_basis @ X_basis.T might be singular, so use pseudoinverse

    # More stable: use SVD
    U, S, Vt = np.linalg.svd(X_basis, full_matrices=False)

    # Keep only significant singular values (numerical rank)
    tol = 1e-10 * S[0]
    rank = np.sum(S > tol)

    # Project x onto the column space of X_basis.T (= row space of X_basis)
    # This is Vt[:rank, :].T @ Vt[:rank, :] @ x
    V_r = Vt[:rank, :].T  # (d, rank)
    x_proj = V_r @ (V_r.T @ x)

    # Residual
    x_orth = x - x_proj
    residual_norm = np.linalg.norm(x_orth)
    relative_residual = residual_norm / (np.linalg.norm(x) + 1e-10)

    return residual_norm, relative_residual


def test_generation(model, tokenizer, prompt: str, T: np.ndarray,
                    start_layer: int, end_layer: int) -> Tuple[bool, str, str]:
    """Test if T correctly predicts the output."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal forward
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    # T-based forward
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in

            # Check for numerical issues
            if np.any(np.isnan(h_out)) or np.any(np.isinf(h_out)):
                return False, tokenizer.decode([normal_token]), "[NaN/Inf]"

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

    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("SPAN COVERAGE HYPOTHESIS TEST")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"\nHYPOTHESIS: T = Y @ pinv(X) is exact when input ∈ span(X)")
    print("PREDICTION: Failures correlate with large orthogonal residuals")

    # Generate calibration
    calib_prompts = generate_calibration()
    print(f"\nCalibration: {len(calib_prompts)} prompts")

    # Collect activations
    print(f"\nCollecting calibrations at layer {start_layer}...")
    X_calib = collect_activations(model, tokenizer, calib_prompts, start_layer)
    print(f"X_calib shape: {X_calib.shape}")

    print(f"\nCollecting calibrations at layer {end_layer}...")
    Y_calib_list = []
    for prompt in calib_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        from mlx_lm.models.base import create_attention_mask
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            h = layer(h, mask, None)
            mx.eval(h)
            if idx == end_layer:
                h_vec = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_calib_list.append(h_vec)
                break

    Y_calib = np.stack(Y_calib_list, axis=0)
    print(f"Y_calib shape: {Y_calib.shape}")

    # Compute T
    print(f"\nComputing T = Y @ pinv(X)...")
    T = Y_calib.T @ np.linalg.pinv(X_calib.T)
    print(f"T shape: {T.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        print("WARNING: T contains NaN/Inf!")
        # Try regularized version
        print("Using regularized lstsq...")
        T = Y_calib.T @ X_calib @ np.linalg.inv(X_calib.T @ X_calib + 1e-6 * np.eye(X_calib.shape[0]))

    # Held-out prompts (mix of expected successes and failures)
    held_out = [
        # Should work (similar to calibration)
        "The capital of Poland is",  # Poland IS in calibration
        "5 + 7 =",  # This math IS in calibration
        "def main():",  # This IS in calibration
        "Actually,",  # This IS in calibration

        # Might fail (different from calibration)
        "The capital of Zimbabwe is",  # Zimbabwe NOT in calibration
        "25 + 37 =",  # Outside our 1-15 range
        "def fibonacci(",  # Not in calibration
        "The speed of light is",  # In calibration!

        # More diverse
        "What causes earthquakes",
        "Why do birds migrate",
        "The meaning of life is",
        "Artificial intelligence can",
    ]

    # Test each held-out prompt
    print(f"\n{'='*70}")
    print("TESTING SPAN COVERAGE HYPOTHESIS")
    print("="*70)

    print(f"\n{'Prompt':<30} | {'Orth %':>8} | {'Match':>6} | {'Exp':>10} | {'Got':>10}")
    print("-" * 80)

    results = []
    for prompt in held_out:
        # Get activation at start layer
        x_test = collect_activations(model, tokenizer, [prompt], start_layer)[0]

        # Compute orthogonal residual
        _, rel_residual = compute_orthogonal_residual(x_test, X_calib)

        # Test generation
        match, expected, got = test_generation(model, tokenizer, prompt, T, start_layer, end_layer)

        status = "OK" if match else "FAIL"
        print(f"{prompt[:30]:<30} | {rel_residual*100:>7.2f}% | {status:>6} | {expected[:10]:>10} | {got[:10]:>10}")

        results.append((prompt, rel_residual, match))

    # Analyze correlation
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS")
    print("="*70)

    successes = [(p, r) for p, r, m in results if m]
    failures = [(p, r) for p, r, m in results if not m]

    if successes:
        avg_residual_success = np.mean([r for _, r in successes])
        print(f"\nSuccesses ({len(successes)}): avg orthogonal residual = {avg_residual_success*100:.2f}%")

    if failures:
        avg_residual_failure = np.mean([r for _, r in failures])
        print(f"Failures ({len(failures)}): avg orthogonal residual = {avg_residual_failure*100:.2f}%")

    if successes and failures:
        if avg_residual_failure > avg_residual_success:
            print(f"\n✓ HYPOTHESIS SUPPORTED: Failures have higher orthogonal residual")
            print(f"  ({avg_residual_failure*100:.2f}% vs {avg_residual_success*100:.2f}%)")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED: Residuals don't correlate with failures")

    # Key insight
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)
    print("""
The closed-form solution T = Y @ pinv(X) IS mathematically exact.

The question is: does span(X) contain the manifold?

- If orthogonal residual ≈ 0: input IS in span(X) → T is exact
- If orthogonal residual > 0: input has component OUTSIDE span(X) → T approximates

TO ACHIEVE TRUE LOSSLESSNESS:
1. The manifold of coherent activations has finite dimension
2. We need calibration to SPAN this manifold
3. This is achieved by diverse, sufficient calibration

The math IS closed-form. The challenge is COVERAGE.
""")


if __name__ == "__main__":
    main()
