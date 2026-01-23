#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Layer Sweep: Test Multiple Layers with Massive Calibration
"""
FINDING: With 1887 calibration prompts, layer 15 achieves 95% held-out accuracy!
This proves coverage is the bottleneck.

Now let's test ALL layers to find:
1. Which layers achieve high accuracy with this calibration?
2. Which layers are bottlenecks that need more calibration?

Usage:
    python qwen3_layer_sweep.py --model /path/to/model
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


def test_single_layer(model, tokenizer, prompts: List[str], transform: dict,
                      layer_idx: int) -> int:
    """Test single layer compression, return number of matches."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    matches = 0

    for prompt in prompts:
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

        if normal_token == t_token:
            matches += 1

    return matches


def generate_calibration() -> List[str]:
    """Generate calibration set."""
    prompts = []

    # Geography
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
        "Nigeria", "Kenya", "Morocco", "Peru", "Colombia", "Mongolia", "Nepal",
        "Ukraine", "Greece", "Portugal", "Ireland", "Finland", "Denmark", "Austria",
        "Belgium", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Serbia",
        "Philippines", "Malaysia", "Singapore", "Taiwan", "Bangladesh", "Sri Lanka",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"{c} is known for")

    # Math
    for a in range(1, 30):
        for b in range(1, 30):
            prompts.append(f"{a} + {b} =")

    # Code
    code = [
        "def ", "class ", "import ", "from ", "return ", "if ", "else:", "elif ",
        "for ", "while ", "try:", "except ", "finally:", "with ", "yield ", "lambda ",
        "def main():", "def __init__(self", "def test_", "def get_", "def set_",
        "class User:", "class Config:", "class Model:", "class Handler:",
        "import numpy", "import pandas", "import torch", "import os", "import sys",
        "from typing import", "from collections import", "from pathlib import",
        "SELECT * FROM", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE",
        "console.log(", "document.getElementById(", "function ", "const ", "let ",
    ]
    prompts.extend(code)

    # Questions
    questions = [
        "What is", "What are", "What was", "What were", "What will",
        "How do", "How does", "How did", "How can", "How should",
        "Why is", "Why are", "Why was", "Why were", "Why do",
        "When is", "When was", "When will", "When did",
        "Where is", "Where are", "Where was", "Where do",
        "Who is", "Who are", "Who was", "Who will",
        "Can you", "Could you", "Would you", "Should I",
    ]
    prompts.extend(questions)

    # Conversational
    conv = [
        "Actually,", "However,", "Therefore,", "Furthermore,", "Moreover,",
        "In fact,", "To be honest,", "Honestly,", "Frankly,", "Basically,",
        "Essentially,", "Well,", "So,", "Now,", "Look,", "Listen,",
        "Nevertheless,", "Nonetheless,", "Consequently,", "Subsequently,",
        "Meanwhile,", "Otherwise,", "Indeed,", "Certainly,", "Obviously,",
    ]
    prompts.extend(conv)

    # Science
    science = [
        "The speed of light is", "The speed of sound is", "Gravity causes",
        "Photosynthesis produces", "DNA stores", "RNA carries", "ATP provides",
        "Quantum mechanics", "The Big Bang", "Black holes are", "Dark matter",
        "The Higgs boson", "String theory", "Electrons orbit",
    ]
    prompts.extend(science)

    # Narrative
    narrative = [
        "Once upon a time", "In the beginning", "Long ago", "It was a dark",
        "The hero stood", "She looked at", "He walked into", "They discovered",
        "Suddenly,", "The dragon breathed", "In a galaxy far",
    ]
    prompts.extend(narrative)

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
    print("LAYER SWEEP WITH MASSIVE CALIBRATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Generate calibration
    calibration = generate_calibration()
    print(f"Calibration: {len(calibration)} prompts")

    # Held-out prompts
    held_out = [
        "The capital of Liechtenstein is",
        "The capital of Andorra is",
        "99 + 87 =",
        "256 - 128 =",
        "def binary_tree(",
        "class AbstractFactory:",
        "What is the tallest mountain",
        "How does photosynthesis work",
        "Why do leaves change color",
        "String theory proposes",
        "The dragon breathed",
        "In a galaxy far",
        "My favorite color is",
        "Banana phone",
        "42 is the answer to",
    ]

    print(f"Held-out: {len(held_out)} prompts")

    # Test key layers
    test_layers = [0, 5, 10, 15, 20, 25, 30, 35]

    print(f"\n{'='*70}")
    print("LAYER-BY-LAYER RESULTS")
    print("="*70)
    print(f"\n{'Layer':<8} | {'Rank':<6} | {'Calib Err':>12} | {'Accuracy':>12}")
    print("-" * 50)

    layer_results = {}

    for layer_idx in test_layers:
        print(f"  Testing layer {layer_idx}...")

        # Collect activations
        X, Y = collect_layer_pair(model, tokenizer, calibration, layer_idx)

        # Compute transform
        transform = compute_whitened_transform(X, Y)

        # Calibration error
        Y_pred = np.zeros_like(Y)
        for i in range(X.shape[1]):
            Y_pred[:, i] = apply_whitened_transform(X[:, i], transform)
        calib_err = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)

        # Test accuracy
        matches = test_single_layer(model, tokenizer, held_out, transform, layer_idx)
        accuracy = matches / len(held_out)

        layer_results[layer_idx] = {
            'rank': transform['rank'],
            'calib_err': calib_err,
            'accuracy': accuracy,
            'matches': matches
        }

        print(f"{layer_idx:<8} | {transform['rank']:<6} | {calib_err:>12.2e} | {matches}/{len(held_out)} ({accuracy*100:>5.1f}%)")

        # Clear memory
        del X, Y, Y_pred, transform

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)

    accuracies = [r['accuracy'] for r in layer_results.values()]
    print(f"\nAverage accuracy: {np.mean(accuracies)*100:.1f}%")
    print(f"Min accuracy: {np.min(accuracies)*100:.1f}% (layer {list(layer_results.keys())[np.argmin(accuracies)]})")
    print(f"Max accuracy: {np.max(accuracies)*100:.1f}% (layer {list(layer_results.keys())[np.argmax(accuracies)]})")

    # Theoretical chained accuracy
    chained_accuracy = np.prod(accuracies) ** (n_layers / len(test_layers))
    print(f"\nTheoretical chained accuracy (all {n_layers} layers): {chained_accuracy*100:.1f}%")

    # What we need for 90% final accuracy
    per_layer_needed = 0.9 ** (1/n_layers)
    print(f"Per-layer accuracy needed for 90% final: {per_layer_needed*100:.2f}%")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print(f"""
With ~{len(calibration)} calibration prompts, single-layer compression achieves
{np.mean(accuracies)*100:.0f}% average accuracy.

For lossless compression of all {n_layers} layers:
- Need {per_layer_needed*100:.2f}% per-layer accuracy for 90% final
- Current: {np.mean(accuracies)*100:.1f}% average
- Gap: {(per_layer_needed - np.mean(accuracies))*100:.2f} percentage points

To close the gap:
1. More diverse calibration prompts
2. Better coverage of rare patterns
3. Or: Accept ~{chained_accuracy*100:.0f}% final accuracy (lossy compression)
""")


if __name__ == "__main__":
    main()
