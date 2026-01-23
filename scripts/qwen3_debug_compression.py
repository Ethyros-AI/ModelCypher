#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Debug Qwen3 Compression
"""
Debug why T isn't working on Qwen3-8B despite working on LFM2-350M.

TESTS:
1. Check if T has NaN/Inf values
2. Verify T @ x = y for a calibration sample
3. Test single-layer compression (easier case)
4. Compare factored vs normal output step by step
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


# Simple calibration set for debugging
CALIBRATION_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of UK is",
    "The capital of China is",
    "The capital of India is",
]

TEST_PROMPT = "The capital of France is"  # In calibration


def collect_activations(model, tokenizer, prompts, layer_idx):
    """Collect h_in and h_out for a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    inputs = []
    outputs = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        # Create attention mask
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                inputs.append(h_in)

            h = layer(h, mask, None)  # Pass mask!
            mx.eval(h)

            if idx == layer_idx:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)
    Y = np.stack(outputs, axis=1)
    return X, Y


def test_single_layer_compression(model, tokenizer, prompt, T, layer_idx):
    """Test compression of a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal forward
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_logits = np.array(logits[0, -1, :].astype(mx.float32))
    normal_token = int(np.argmax(normal_logits))

    # Factored forward (replace single layer with T)
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    # Create attention mask
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == layer_idx:
            # Capture h before layer
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)

            # Apply T instead of layer
            h_out = T @ h_in

            # DEBUG: Print intermediate values
            print(f"\n  h_in stats: min={h_in.min():.4f}, max={h_in.max():.4f}, norm={np.linalg.norm(h_in):.4f}")
            print(f"  h_out stats: min={h_out.min():.4f}, max={h_out.max():.4f}, norm={np.linalg.norm(h_out):.4f}")
            print(f"  h_out has NaN: {np.any(np.isnan(h_out))}, Inf: {np.any(np.isinf(h_out))}")

            # Replace only last position
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        else:
            h = layer(h, mask, None)  # Pass mask!
            mx.eval(h)

    h = inner_model.norm(h)
    # Use lm_head if tie_word_embeddings is False
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_logits = np.array(logits[0, -1, :].astype(mx.float32))
    fact_token = int(np.argmax(fact_logits))

    return normal_token, fact_token, tokenizer.decode([normal_token]), tokenizer.decode([fact_token])


def test_manual_vs_normal_forward(model, tokenizer, prompt):
    """Test if manual layer-by-layer forward matches model(input_ids)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal forward
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits_normal = model(input_ids)
    mx.eval(logits_normal)
    normal_logits = np.array(logits_normal[0, -1, :].astype(mx.float32))

    # Manual forward WITH proper attention mask
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    print(f"  After embedding: h.dtype = {h.dtype}, h shape = {h.shape}")

    # Create attention mask (same as Qwen3Model.__call__)
    mask = create_attention_mask(h, None)
    print(f"  Attention mask: {mask}")

    for idx, layer in enumerate(inner_model.layers):
        h = layer(h, mask, None)  # Pass mask!
        mx.eval(h)

    h = inner_model.norm(h)
    # Use lm_head if tie_word_embeddings is False
    if hasattr(model, 'lm_head'):
        logits_manual = model.lm_head(h)
    else:
        logits_manual = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits_manual)
    manual_logits = np.array(logits_manual[0, -1, :].astype(mx.float32))

    # Compare
    logit_diff = np.abs(normal_logits - manual_logits)
    max_diff = logit_diff.max()
    mean_diff = logit_diff.mean()

    normal_tok = int(np.argmax(normal_logits))
    manual_tok = int(np.argmax(manual_logits))

    return normal_tok, manual_tok, max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=7, help="Layer to compress")
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
    print("QWEN3 COMPRESSION DEBUG")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Testing layer {args.layer}")

    # First test: does manual forward match normal forward?
    print(f"\n{'='*70}")
    print("MANUAL VS NORMAL FORWARD TEST")
    print("="*70)
    print(f"Prompt: '{TEST_PROMPT}'")
    normal_tok, manual_tok, max_diff, mean_diff = test_manual_vs_normal_forward(
        model, tokenizer, TEST_PROMPT
    )
    print(f"  Normal token: {normal_tok} ({tokenizer.decode([normal_tok])})")
    print(f"  Manual token: {manual_tok} ({tokenizer.decode([manual_tok])})")
    print(f"  Logit max diff: {max_diff:.6f}")
    print(f"  Logit mean diff: {mean_diff:.6f}")
    print(f"  Match: {'YES' if normal_tok == manual_tok else 'NO'}")

    # Collect calibration data for single layer
    print(f"\nCollecting calibration data for layer {args.layer}...")
    X, Y = collect_activations(model, tokenizer, CALIBRATION_PROMPTS, args.layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Check for NaN/Inf in data
    print(f"\n{'='*70}")
    print("DATA QUALITY CHECK")
    print("="*70)
    print(f"X: NaN={np.any(np.isnan(X))}, Inf={np.any(np.isinf(X))}")
    print(f"X stats: min={X.min():.4f}, max={X.max():.4f}")
    print(f"Y: NaN={np.any(np.isnan(Y))}, Inf={np.any(np.isinf(Y))}")
    print(f"Y stats: min={Y.min():.4f}, max={Y.max():.4f}")

    # Compute T
    print(f"\n{'='*70}")
    print("T MATRIX COMPUTATION")
    print("="*70)
    T = Y @ np.linalg.pinv(X)
    print(f"T shape: {T.shape}")
    print(f"T: NaN={np.any(np.isnan(T))}, Inf={np.any(np.isinf(T))}")
    print(f"T stats: min={T.min():.4e}, max={T.max():.4e}")

    # Reconstruction error
    Y_pred = T @ X
    recon_err = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Reconstruction error: {recon_err:.2e}")

    # Test T @ x = y for first calibration sample
    print(f"\n{'='*70}")
    print("CALIBRATION SAMPLE TEST")
    print("="*70)
    x0 = X[:, 0]
    y0 = Y[:, 0]
    y0_pred = T @ x0
    sample_err = np.linalg.norm(y0 - y0_pred) / np.linalg.norm(y0)
    cos_sim = np.dot(y0, y0_pred) / (np.linalg.norm(y0) * np.linalg.norm(y0_pred))
    print(f"First calibration sample error: {sample_err:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Compare layer output vs T @ input
    print(f"\n{'='*70}")
    print("LAYER VS T COMPARISON")
    print("="*70)

    # Re-run collection for first prompt and capture actual layer output
    from mlx_lm.models.base import create_attention_mask as create_mask_compare
    tokens = tokenizer.encode(CALIBRATION_PROMPTS[0])
    import mlx.core as mx
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask_compare = create_mask_compare(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == args.layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)

        h = layer(h, mask_compare, None)  # Pass mask!
        mx.eval(h)

        if idx == args.layer:
            h_out_actual = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)

    h_out_pred = T @ h_in
    layer_diff = np.linalg.norm(h_out_actual - h_out_pred) / np.linalg.norm(h_out_actual)
    layer_cos = np.dot(h_out_actual, h_out_pred) / (np.linalg.norm(h_out_actual) * np.linalg.norm(h_out_pred))
    print(f"Layer output vs T @ input:")
    print(f"  Relative error: {layer_diff:.2e}")
    print(f"  Cosine similarity: {layer_cos:.6f}")

    # Test single layer compression
    print(f"\n{'='*70}")
    print("SINGLE LAYER COMPRESSION TEST")
    print("="*70)
    print(f"Prompt: '{TEST_PROMPT}'")

    normal_tok, fact_tok, normal_str, fact_str = test_single_layer_compression(
        model, tokenizer, TEST_PROMPT, T, args.layer
    )

    print(f"\nNormal output: {normal_str} (token {normal_tok})")
    print(f"Factored output: {fact_str} (token {fact_tok})")
    print(f"Match: {'YES' if normal_tok == fact_tok else 'NO'}")


if __name__ == "__main__":
    main()
