#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Check numerical properties of activations
"""
The overflow/divide-by-zero warnings suggest the activation data
has problematic values. Let's investigate.
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


def collect_raw_activations(model, tokenizer, prompt, start_layer, end_layer):
    """Collect raw activations with detailed logging."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {len(tokens)}")

    # Check embedding
    h_np = np.array(h[0, -1, :].astype(mx.float32))
    print(f"Embedding: min={h_np.min():.4f}, max={h_np.max():.4f}, norm={np.linalg.norm(h_np):.4f}")
    print(f"  NaN: {np.any(np.isnan(h_np))}, Inf: {np.any(np.isinf(h_np))}")

    if is_lfm2:
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
        attn_mask = create_attention_mask(h, None)
        conv_mask = create_ssm_mask(h, None)
    else:
        attn_mask = None
        conv_mask = None

    h_in = None
    h_out = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32))
            print(f"\nLayer {idx} INPUT:")
            print(f"  min={h_in.min():.4f}, max={h_in.max():.4f}, norm={np.linalg.norm(h_in):.4f}")
            print(f"  NaN: {np.any(np.isnan(h_in))}, Inf: {np.any(np.isinf(h_in))}")

        if is_lfm2:
            mask = attn_mask if layer.is_attention_layer else conv_mask
        else:
            mask = attn_mask
        h = layer(h, mask, None)
        mx.eval(h)

        if idx == end_layer:
            h_out = np.array(h[0, -1, :].astype(mx.float32))
            print(f"\nLayer {idx} OUTPUT:")
            print(f"  min={h_out.min():.4f}, max={h_out.max():.4f}, norm={np.linalg.norm(h_out):.4f}")
            print(f"  NaN: {np.any(np.isnan(h_out))}, Inf: {np.any(np.isinf(h_out))}")

        # Check intermediate layers for growth
        if start_layer <= idx <= end_layer:
            h_mid = np.array(h[0, -1, :].astype(mx.float32))
            if idx % 3 == 0:
                print(f"  Layer {idx}: norm={np.linalg.norm(h_mid):.4f}, max={h_mid.max():.4f}")

    return h_in, h_out


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
    print("NUMERICAL CHECK")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    prompts = [
        "The capital of France is",
        "Water freezes at",
        "Well, actually",
        "1 + 1 =",
    ]

    all_h_in = []
    all_h_out = []

    for p in prompts:
        h_in, h_out = collect_raw_activations(model, tokenizer, p, start_layer, end_layer)
        all_h_in.append(h_in)
        all_h_out.append(h_out)

    # Check matrix properties
    X = np.stack(all_h_in, axis=1)
    Y = np.stack(all_h_out, axis=1)

    print(f"\n{'='*70}")
    print("MATRIX ANALYSIS")
    print("="*70)

    print(f"\nX matrix ({X.shape}):")
    print(f"  min={X.min():.4f}, max={X.max():.4f}")
    print(f"  NaN count: {np.sum(np.isnan(X))}, Inf count: {np.sum(np.isinf(X))}")
    print(f"  Frobenius norm: {np.linalg.norm(X, 'fro'):.4f}")

    print(f"\nY matrix ({Y.shape}):")
    print(f"  min={Y.min():.4f}, max={Y.max():.4f}")
    print(f"  NaN count: {np.sum(np.isnan(Y))}, Inf count: {np.sum(np.isinf(Y))}")
    print(f"  Frobenius norm: {np.linalg.norm(Y, 'fro'):.4f}")

    # SVD
    U_x, S_x, Vh_x = np.linalg.svd(X, full_matrices=False)
    print(f"\nX singular values: {S_x}")
    print(f"X condition number: {S_x[0]/S_x[-1]:.2e}")

    U_y, S_y, Vh_y = np.linalg.svd(Y, full_matrices=False)
    print(f"\nY singular values: {S_y}")
    print(f"Y condition number: {S_y[0]/S_y[-1]:.2e}")

    # Compute T
    print(f"\n{'='*70}")
    print("T COMPUTATION")
    print("="*70)

    # Use float64 for pinv
    X_64 = X.astype(np.float64)
    Y_64 = Y.astype(np.float64)

    X_pinv = np.linalg.pinv(X_64)
    print(f"\npinv(X) shape: {X_pinv.shape}")
    print(f"  min={X_pinv.min():.4e}, max={X_pinv.max():.4e}")
    print(f"  NaN count: {np.sum(np.isnan(X_pinv))}, Inf count: {np.sum(np.isinf(X_pinv))}")

    T = Y_64 @ X_pinv
    print(f"\nT shape: {T.shape}")
    print(f"  min={T.min():.4e}, max={T.max():.4e}")
    print(f"  NaN count: {np.sum(np.isnan(T))}, Inf count: {np.sum(np.isinf(T))}")

    # Reconstruction
    Y_pred = T @ X_64
    recon_err = np.linalg.norm(Y_64 - Y_pred) / np.linalg.norm(Y_64)
    print(f"\nReconstruction error: {recon_err:.10e}")

    # Test each sample
    print(f"\n{'='*70}")
    print("PER-SAMPLE TEST")
    print("="*70)

    for i, p in enumerate(prompts):
        x_i = X_64[:, i]
        y_i = Y_64[:, i]
        y_pred_i = T @ x_i

        err = np.linalg.norm(y_i - y_pred_i) / np.linalg.norm(y_i)
        cos_sim = np.dot(y_i, y_pred_i) / (np.linalg.norm(y_i) * np.linalg.norm(y_pred_i))
        print(f"  '{p[:30]}': err={err:.10e}, cos={cos_sim:.10f}")


if __name__ == "__main__":
    main()
