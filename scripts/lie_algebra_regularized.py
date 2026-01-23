#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Regularized Lie Algebra Compression
"""
FINDING: T = Y @ pinv(X) memorizes, doesn't generalize.
- Condition number: 1.62e+17 (extremely ill-conditioned)
- Perfect on calibration, fails on held-out

SOLUTION: Use regularization to encourage generalization.

APPROACHES:
1. Ridge regression: T = Y @ X.T @ inv(X @ X.T + lambda * I)
2. Nuclear norm regularization: encourage low-rank T
3. Tikhonov with identity prior: T ~ I (encourage small perturbation)

Usage:
    python lie_algebra_regularized.py --model /path/to/model
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


CALIBRATION_PROMPTS = [
    f"The capital of {c} is" for c in [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "South Korea",
    ]
] + [
    f"{a} + {b} =" for a in range(1, 8) for b in range(1, 8)
] + [
    f"The opposite of {w} is" for w in [
        "hot", "big", "happy", "light", "up", "good", "old", "fast",
    ]
] + [
    "Water freezes at", "Ice melts at", "Steam forms at",
    "The moon orbits", "The Earth orbits", "Mars orbits",
    "Stars are made of", "Diamonds are made of", "Gold is made of",
    "The answer is", "The solution is", "The result is",
    "Well, actually", "Actually,", "In fact,",
    "If you think about it,", "When you consider,", "Looking at it,",
    "The problem is that", "The issue is", "The challenge is",
    "In my opinion,", "From my perspective,", "I think that",
]

HELD_OUT_PROMPTS = [
    "The capital of Egypt is",
    "9 + 9 =",
    "The opposite of slow is",
    "Mercury orbits",
    "Silver is made of",
    "The conclusion is",
    "To be honest,",
    "Thinking about it,",
    "The difficulty is",
    "In my view,",
]


def collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X and Y in float64."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    inputs = []
    outputs = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        if is_lfm2:
            from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
            attn_mask = create_attention_mask(h, None)
            conv_mask = create_ssm_mask(h, None)
        else:
            attn_mask = None
            conv_mask = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)
    Y = np.stack(outputs, axis=1)
    return X, Y


def compute_T_pinv(X, Y):
    """Standard pseudoinverse (no regularization)."""
    return Y @ np.linalg.pinv(X)


def compute_T_ridge(X, Y, lambda_reg):
    """Ridge regression: T = Y @ X.T @ inv(X @ X.T + lambda * I)."""
    hidden_dim = X.shape[0]
    XXT = X @ X.T
    XXT_reg = XXT + lambda_reg * np.eye(hidden_dim)
    return Y @ X.T @ np.linalg.inv(XXT_reg)


def compute_T_identity_prior(X, Y, lambda_reg):
    """
    Tikhonov with identity prior: min ||Y - T @ X||^2 + lambda * ||T - I||^2

    Solution: T = (Y @ X.T + lambda * I) @ inv(X @ X.T + lambda * I)
    """
    hidden_dim = X.shape[0]
    I = np.eye(hidden_dim)
    XXT = X @ X.T
    XXT_reg = XXT + lambda_reg * I
    YXT_reg = Y @ X.T + lambda_reg * I
    return YXT_reg @ np.linalg.inv(XXT_reg)


def test_generation(model, tokenizer, prompt, T, start_layer, end_layer):
    """Test generation with T."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_logits = np.array(logits[0, -1, :].astype(mx.float32))
    normal_token = int(np.argmax(normal_logits))

    # Factored
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    if is_lfm2:
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
        attn_mask = create_attention_mask(h, None)
        conv_mask = create_ssm_mask(h, None)
    else:
        attn_mask = None
        conv_mask = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass
        else:
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    if is_lfm2:
        h = inner_model.embedding_norm(h)
    else:
        h = inner_model.norm(h)
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_logits = np.array(logits[0, -1, :].astype(mx.float32))
    fact_token = int(np.argmax(fact_logits))

    return normal_token == fact_token, tokenizer.decode([normal_token]), tokenizer.decode([fact_token])


def test_T(model, tokenizer, T, calib_prompts, held_prompts, start_layer, end_layer, name):
    """Test T on calibration and held-out."""
    calib_match = 0
    for p in calib_prompts[:20]:
        match, _, _ = test_generation(model, tokenizer, p, T, start_layer, end_layer)
        if match:
            calib_match += 1

    held_match = 0
    held_results = []
    for p in held_prompts:
        match, normal, factored = test_generation(model, tokenizer, p, T, start_layer, end_layer)
        if match:
            held_match += 1
        held_results.append((p, match, normal, factored))

    # Condition number
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    cond = S[0] / max(S[-1], 1e-20)

    print(f"\n{name}:")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Calibration: {calib_match}/20")
    print(f"  Held-out: {held_match}/{len(held_prompts)}")

    if held_match < len(held_prompts):
        print(f"  Failures:")
        for p, match, normal, factored in held_results:
            if not match:
                print(f"    '{p[:30]}...': {normal[:10]} vs {factored[:10]}")

    return calib_match, held_match


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

    start_layer = 3
    end_layer = n_layers - 2

    print(f"\n{'='*70}")
    print("REGULARIZED LIE ALGEBRA COMPRESSION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {start_layer} -> {end_layer}")
    print(f"Calibration: {len(CALIBRATION_PROMPTS)} prompts")
    print(f"Held-out: {len(HELD_OUT_PROMPTS)} prompts")

    # Collect data
    logger.info("Collecting data...")
    X, Y = collect_endpoint_data(model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Test different methods
    print(f"\n{'='*70}")
    print("METHOD COMPARISON")
    print("="*70)

    # 1. Standard pinv
    T_pinv = compute_T_pinv(X, Y)
    test_T(model, tokenizer, T_pinv, CALIBRATION_PROMPTS, HELD_OUT_PROMPTS, start_layer, end_layer, "Pseudoinverse (no reg)")

    # 2. Ridge regression with different lambdas
    for lam in [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
        T_ridge = compute_T_ridge(X, Y, lam)
        test_T(model, tokenizer, T_ridge, CALIBRATION_PROMPTS, HELD_OUT_PROMPTS, start_layer, end_layer, f"Ridge (lambda={lam})")

    # 3. Identity prior with different lambdas
    for lam in [1e-4, 1e-2, 1.0, 10.0]:
        T_id = compute_T_identity_prior(X, Y, lam)
        test_T(model, tokenizer, T_id, CALIBRATION_PROMPTS, HELD_OUT_PROMPTS, start_layer, end_layer, f"Identity prior (lambda={lam})")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print("""
If regularization helps, it means the issue is overfitting.
If regularization hurts or doesn't help, the issue is fundamental.

The ideal is to find a lambda where:
- Calibration stays ~100%
- Held-out improves from baseline
- Condition number decreases significantly
""")


if __name__ == "__main__":
    main()
