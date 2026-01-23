#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Lie Algebra F-Factorization Test
"""
Test: T = I + F where F is low-rank.

The insight: The delta (Y - X) has low rank (~23 for 90% var).
We compute F such that delta = F @ X, then T = I + F.
Factoring F should preserve token prediction if delta matters more than the identity component.

Usage:
    python lie_algebra_F_test.py --model /path/to/model
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


CALIBRATION_PROMPTS = [
    f"The capital of {c} is" for c in [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "South Korea",
        "Argentina", "Egypt", "Nigeria", "Indonesia", "Turkey", "Thailand"
    ]
] + [
    f"{a} + {b} =" for a in range(1, 11) for b in range(1, 11)
][:50] + [
    f"The opposite of {w} is" for w in [
        "hot", "big", "happy", "light", "up", "good", "old", "fast",
        "loud", "wet", "full", "rich", "strong", "true", "beautiful"
    ]
] + [
    "Once upon a time", "In the beginning", "The quick brown fox",
    "To be or not to", "It was a dark and", "Long ago in a"
] + [
    "Python is a", "Machine learning is", "The internet is", "A computer is"
] + [
    "Love is", "Time is", "Life is", "Truth is"
] + [
    "In conclusion,", "Therefore,", "However,", "Moreover,"
]

HELD_OUT_PROMPTS = [
    "Water freezes at",
    "The color of the sky is",
    "100 / 10 =",
    "50 + 50 =",
    "9 * 9 =",
    "The opposite of fast is",
    "Neural networks are",
    "The answer is",
    "Well, actually",
]


def collect_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X (at start_layer) and Y (at end_layer)."""
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
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1).astype(np.float64)
    Y = np.stack(outputs, axis=1).astype(np.float64)
    return X, Y


def test_generation(model, tokenizer, prompt, T_fact, start_layer, end_layer):
    """Test generation with T_fact = I + F_fact."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    normal = tokenizer.decode([normal_token]).split()[0] if tokenizer.decode([normal_token]).split() else "(empty)"

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
            h_out = T_fact @ h_in
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
    fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

    return normal, factored


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

    print(f"\n{'='*70}")
    print("LIE ALGEBRA F-FACTORIZATION TEST")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    start_layer = 3
    end_layer = n_layers - 2

    # Collect data
    print(f"\nCollecting data for layers {start_layer}→{end_layer}...")
    X, Y = collect_data(model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Compute F = (Y - X) @ pinv(X)
    Delta = Y - X
    F = Delta @ np.linalg.pinv(X)

    # SVD of F
    U_F, S_F, Vh_F = np.linalg.svd(F, full_matrices=False)

    total_var = np.sum(S_F**2)
    cumsum = np.cumsum(S_F**2)
    rank_90 = np.searchsorted(cumsum / total_var, 0.90) + 1
    rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1

    print(f"\nF singular values (top 10): {np.round(S_F[:10], 1)}")
    print(f"F rank 90%: {rank_90}, rank 99%: {rank_99}")

    # Test different F ranks
    print(f"\n{'='*70}")
    print("TEST: Token prediction at different F ranks")
    print("="*70)

    I = np.eye(hidden_dim)

    for f_rank in [128, 64, 32, 16, 8, 4, 2, 1]:
        if f_rank > len(S_F):
            continue

        # Factor F
        F_fact = U_F[:, :f_rank] @ np.diag(S_F[:f_rank]) @ Vh_F[:f_rank, :]
        T_fact = I + F_fact

        # Y reconstruction error
        Y_pred = T_fact @ X
        y_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)

        # Test on calibration
        calib_matches = 0
        for p in CALIBRATION_PROMPTS[:20]:
            normal, factored = test_generation(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                calib_matches += 1

        # Test on held-out
        held_matches = 0
        for p in HELD_OUT_PROMPTS:
            normal, factored = test_generation(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                held_matches += 1

        compression = hidden_dim / f_rank
        print(f"F_rank={f_rank:>3}: Y_err={y_error:.3f}, calib={calib_matches}/20, held={held_matches}/{len(HELD_OUT_PROMPTS)}, compress={compression:.0f}x")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print("""
The T = I + F formulation doesn't help because F has the same rank as T.
The issue is that the linear map from X to Delta requires high rank,
even though Delta itself is low-rank data.

The fundamental limit: approximating a linear map requires capturing
ALL significant directions the input can take, not just the output.
""")


if __name__ == "__main__":
    main()
