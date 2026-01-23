#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Layer-by-Layer Compression: Minimize Nonlinearity
"""
INSIGHT: Compressing 26 layers at once (7→33) introduces too much nonlinearity.
Each layer has attention (softmax) and MLP (GELU) - fundamentally nonlinear.

APPROACH: Compress ONE LAYER at a time.
Each single-layer transform is much closer to linear.
We chain them: T_7 → T_8 → T_9 → ... → T_33

If each T_i is 99% accurate, chaining 26 of them gives 0.99^26 ≈ 77% accuracy.
But if each T_i is 99.9% accurate, we get 0.999^26 ≈ 97% accuracy.

This script tests single-layer compression accuracy to find the limit.

Usage:
    python qwen3_layer_by_layer.py --model /path/to/model
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


def collect_layer_pair(model, tokenizer, prompts: List[str],
                       layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations before and after a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    X_list = []
    Y_list = []

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
                x = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x)

                h = layer(h, mask, None)
                mx.eval(h)

                y = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_list.append(y)
                break
            else:
                h = layer(h, mask, None)
                mx.eval(h)

    return np.stack(X_list, axis=1), np.stack(Y_list, axis=1)


def compute_whitened_transform(X: np.ndarray, Y: np.ndarray) -> dict:
    """Compute numerically stable whitened transform."""
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)

    X_c = X - X_mean
    Y_c = Y - Y_mean

    U_X, S_X, Vt_X = np.linalg.svd(X_c, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_c, full_matrices=False)

    tol = 1e-10 * S_X[0]
    rank = np.sum(S_X > tol)
    rank = min(rank, len(S_X), len(S_Y))

    T_w = Vt_Y[:rank, :] @ Vt_X[:rank, :].T

    return {
        'U_X': U_X[:, :rank],
        'S_X': S_X[:rank],
        'U_Y': U_Y[:, :rank],
        'S_Y': S_Y[:rank],
        'T_w': T_w,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'rank': rank
    }


def apply_whitened_transform(x: np.ndarray, transform: dict) -> np.ndarray:
    """Apply whitened transform to a vector."""
    x_c = x - transform['X_mean']
    x_proj = transform['U_X'].T @ x_c
    x_w = x_proj / (transform['S_X'] + 1e-10)
    y_w = transform['T_w'] @ x_w
    y_proj = transform['S_Y'] * y_w
    y_c = transform['U_Y'] @ y_proj
    return y_c + transform['Y_mean']


def test_single_layer(model, tokenizer, prompt: str, transform: dict,
                      layer_idx: int) -> Tuple[bool, str, str]:
    """Test single layer compression."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal forward
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    # Transform-based forward
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == layer_idx:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = apply_whitened_transform(h_in, transform)

            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
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


def generate_calibration() -> List[str]:
    """Generate calibration prompts."""
    prompts = []

    # Geography
    for c in ["France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
              "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey"]:
        prompts.append(f"The capital of {c} is")

    # Math
    for a in range(1, 15):
        for b in range(1, 15):
            prompts.append(f"{a} + {b} =")

    # Code
    prompts.extend([
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "def main():", "def __init__(self", "class Config:", "import numpy",
    ])

    # Questions
    prompts.extend([
        "What is", "How do", "Why is", "When was", "Where is", "Who is",
        "Can you", "Could you", "Should I", "Is it",
    ])

    # Conversational
    prompts.extend([
        "Actually,", "However,", "Therefore,", "In fact,", "Basically,",
        "Essentially,", "Well,", "So,", "Look,", "Listen,",
    ])

    # Science
    prompts.extend([
        "The speed of light is", "Photosynthesis produces", "DNA stands for",
        "Quantum mechanics", "Black holes are", "The Big Bang",
    ])

    return prompts


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
    print("LAYER-BY-LAYER COMPRESSION ANALYSIS")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Generate calibration
    prompts = generate_calibration()
    print(f"Calibration: {len(prompts)} prompts")

    # Held-out test prompts
    held_out = [
        "The capital of Zimbabwe is",
        "25 + 37 =",
        "def fibonacci(",
        "What causes earthquakes",
        "The meaning of life is",
    ]

    # Test each layer individually
    print(f"\n{'='*70}")
    print("SINGLE LAYER COMPRESSION ACCURACY")
    print("="*70)
    print(f"\n{'Layer':<8} | {'Rank':<6} | {'Calib Err':>12} | {'Held-out':>12}")
    print("-" * 50)

    layer_accuracies = []
    layer_transforms = {}

    for layer_idx in range(0, n_layers):
        # Collect activations for this layer
        X, Y = collect_layer_pair(model, tokenizer, prompts, layer_idx)

        # Compute transform
        transform = compute_whitened_transform(X, Y)
        layer_transforms[layer_idx] = transform

        # Calibration error
        Y_pred = np.zeros_like(Y)
        for i in range(X.shape[1]):
            Y_pred[:, i] = apply_whitened_transform(X[:, i], transform)

        calib_err = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)

        # Held-out accuracy
        matches = 0
        for prompt in held_out:
            match, _, _ = test_single_layer(model, tokenizer, prompt, transform, layer_idx)
            if match:
                matches += 1

        accuracy = matches / len(held_out)
        layer_accuracies.append(accuracy)

        print(f"{layer_idx:<8} | {transform['rank']:<6} | {calib_err:>12.6e} | {matches}/{len(held_out)} ({accuracy*100:>5.1f}%)")

        # Clear memory
        del X, Y, Y_pred

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("="*70)

    perfect_layers = [i for i, acc in enumerate(layer_accuracies) if acc == 1.0]
    imperfect_layers = [i for i, acc in enumerate(layer_accuracies) if acc < 1.0]

    print(f"\nLayers with 100% held-out accuracy: {len(perfect_layers)}")
    print(f"Layers with < 100% accuracy: {len(imperfect_layers)}")

    if imperfect_layers:
        print(f"\nImperfect layers (first 10):")
        for layer_idx in imperfect_layers[:10]:
            print(f"  Layer {layer_idx}: {layer_accuracies[layer_idx]*100:.1f}%")

    # Test chaining multiple perfect layers
    if len(perfect_layers) >= 2:
        print(f"\n{'='*70}")
        print("CHAINED COMPRESSION TEST")
        print("="*70)

        # Try chaining consecutive perfect layers
        for chain_start in range(min(5, len(perfect_layers))):
            for chain_len in [2, 5, 10, 20]:
                chain_end = perfect_layers[chain_start] + chain_len
                if chain_end >= n_layers:
                    continue

                # Check if all layers in chain are perfect
                chain_layers = list(range(perfect_layers[chain_start], chain_end + 1))
                all_perfect = all(layer_accuracies[l] == 1.0 for l in chain_layers)

                if all_perfect:
                    # Test chained compression
                    matches = 0
                    for prompt in held_out:
                        tokens = tokenizer.encode(prompt)
                        input_ids = mx.array([tokens])
                        h = inner_model.embed_tokens(input_ids)
                        mx.eval(h)
                        from mlx_lm.models.base import create_attention_mask
                        mask = create_attention_mask(h, None)

                        # Forward with chained transforms
                        for idx, layer in enumerate(inner_model.layers):
                            if idx in chain_layers:
                                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                                h_out = apply_whitened_transform(h_in, layer_transforms[idx])
                                h_np = np.array(h.astype(mx.float32))
                                h_np[0, -1, :] = h_out.astype(np.float32)
                                h = mx.array(h_np).astype(h.dtype)
                                mx.eval(h)
                            else:
                                h = layer(h, mask, None)
                                mx.eval(h)

                        h = inner_model.norm(h)
                        if hasattr(model, 'lm_head'):
                            logits = model.lm_head(h)
                        else:
                            logits = inner_model.embed_tokens.as_linear(h)
                        mx.eval(logits)

                        # Normal forward for comparison
                        input_ids = mx.array([tokens])
                        normal_logits = model(input_ids)
                        mx.eval(normal_logits)

                        t_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
                        n_token = int(np.argmax(np.array(normal_logits[0, -1, :].astype(mx.float32))))

                        if t_token == n_token:
                            matches += 1

                    print(f"Chain {chain_layers[0]}→{chain_layers[-1]} ({len(chain_layers)} layers): {matches}/{len(held_out)}")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print(f"""
Layer-by-layer analysis shows:

- Average single-layer accuracy: {np.mean(layer_accuracies)*100:.1f}%
- Min single-layer accuracy: {np.min(layer_accuracies)*100:.1f}%
- Max single-layer accuracy: {np.max(layer_accuracies)*100:.1f}%

If all layers were perfectly compressible (100% each), chaining would work.
But even one imperfect layer breaks the chain.

The challenge is that some layers are intrinsically nonlinear and cannot
be perfectly approximated by ANY linear transform.

Those layers are the BOTTLENECK for compression.
""")


if __name__ == "__main__":
    main()
