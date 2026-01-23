#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Chain Compression: Test Multiple Consecutive Layers
"""
FINDING: Middle layers (15-20) achieve 100% single-layer accuracy!
These are the "transmission" layers - linear highways through the network.

Now test: Can we CHAIN multiple consecutive layers and maintain accuracy?

If each layer is 100% accurate individually, chaining should also work.
If chaining fails, errors accumulate even from tiny inaccuracies.

Usage:
    python qwen3_chain_compression.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_layer_pair(model, tokenizer, prompts: List[str],
                       layer_idx: int) -> tuple:
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


def test_chained_layers(model, tokenizer, prompts: List[str],
                        transforms: Dict[int, dict],
                        start_layer: int, end_layer: int) -> tuple:
    """Test chained compression across multiple layers."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    matches = 0
    results = []

    for prompt in prompts:
        # Normal forward
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

        # Chained transform forward
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if start_layer <= idx <= end_layer and idx in transforms:
                # Use transform instead of layer
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                h_out = apply_whitened_transform(h_in, transforms[idx])

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

        match = (normal_token == t_token)
        if match:
            matches += 1

        results.append({
            'prompt': prompt,
            'match': match,
            'expected': tokenizer.decode([normal_token]),
            'got': tokenizer.decode([t_token])
        })

    return matches, results


def generate_calibration() -> List[str]:
    """Generate MASSIVE calibration set."""
    prompts = []

    # Geography - extended
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
        "Nigeria", "Kenya", "Morocco", "Peru", "Colombia", "Mongolia", "Nepal",
        "Ukraine", "Greece", "Portugal", "Ireland", "Finland", "Denmark", "Austria",
        "Belgium", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Serbia",
        "Philippines", "Malaysia", "Singapore", "Taiwan", "Bangladesh", "Sri Lanka",
        "Myanmar", "Cambodia", "Laos", "Bhutan", "Maldives", "Cuba", "Jamaica",
        "Haiti", "Dominican Republic", "Guatemala", "Honduras", "El Salvador",
        "Nicaragua", "Costa Rica", "Panama", "Venezuela", "Ecuador", "Bolivia",
        "Paraguay", "Uruguay", "Zimbabwe", "Zambia", "Tanzania", "Uganda", "Rwanda",
        "Ethiopia", "Ghana", "Senegal", "Mali", "Niger", "Chad", "Sudan", "Libya",
        "Tunisia", "Algeria", "Iraq", "Iran", "Syria", "Lebanon", "Jordan", "Israel",
        "Saudi Arabia", "UAE", "Kuwait", "Qatar", "Bahrain", "Oman", "Yemen",
        "Afghanistan", "Uzbekistan", "Kazakhstan", "Turkmenistan", "Kyrgyzstan",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"The population of {c} is")
        prompts.append(f"{c} is known for")

    # Math - extended
    for a in range(1, 35):
        for b in range(1, 35):
            prompts.append(f"{a} + {b} =")

    # Code
    code = [
        "def ", "class ", "import ", "from ", "return ", "if ", "else:", "elif ",
        "for ", "while ", "try:", "except ", "finally:", "with ", "yield ", "lambda ",
        "def main():", "def __init__(self", "def test_", "def get_", "def set_",
        "class User:", "class Config:", "class Model:", "class Handler:",
        "import numpy", "import pandas", "import torch", "import os", "import sys",
        "from typing import", "from collections import", "from pathlib import",
        "SELECT * FROM", "INSERT INTO", "UPDATE ", "DELETE FROM",
        "console.log(", "function ", "const ", "let ",
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
        "Nevertheless,", "Consequently,", "Meanwhile,", "Otherwise,",
    ]
    prompts.extend(conv)

    # Science
    science = [
        "The speed of light is", "Photosynthesis produces", "DNA stores",
        "Quantum mechanics", "The Big Bang", "Black holes are", "Dark matter",
    ]
    prompts.extend(science)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start", type=int, default=10,
                        help="Start layer for compression")
    parser.add_argument("--end", type=int, default=20,
                        help="End layer for compression")
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
    print("CHAINED LAYER COMPRESSION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {args.start} to {args.end}")

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

    # Build transforms for each layer
    print(f"\nBuilding transforms for layers {args.start}-{args.end}...")
    transforms = {}

    for layer_idx in range(args.start, args.end + 1):
        print(f"  Layer {layer_idx}...", end=" ")
        X, Y = collect_layer_pair(model, tokenizer, calibration, layer_idx)
        transform = compute_whitened_transform(X, Y)
        transforms[layer_idx] = transform
        print(f"rank={transform['rank']}")
        del X, Y

    # Test single layer compression for each
    print(f"\n{'='*70}")
    print("SINGLE LAYER ACCURACY")
    print("="*70)

    single_accuracies = {}
    for layer_idx in range(args.start, args.end + 1):
        matches, _ = test_chained_layers(
            model, tokenizer, held_out,
            {layer_idx: transforms[layer_idx]},
            layer_idx, layer_idx
        )
        accuracy = matches / len(held_out)
        single_accuracies[layer_idx] = accuracy
        print(f"  Layer {layer_idx}: {matches}/{len(held_out)} ({accuracy*100:.0f}%)")

    # Test chained compression
    print(f"\n{'='*70}")
    print("CHAINED COMPRESSION")
    print("="*70)

    # Test progressively more layers
    for chain_len in range(1, args.end - args.start + 2):
        chain_end = args.start + chain_len - 1
        if chain_end > args.end:
            break

        chain_transforms = {
            l: transforms[l] for l in range(args.start, chain_end + 1)
        }

        matches, results = test_chained_layers(
            model, tokenizer, held_out,
            chain_transforms,
            args.start, chain_end
        )

        accuracy = matches / len(held_out)
        n_layers_compressed = chain_end - args.start + 1

        # Theoretical accuracy if each layer were independent
        theoretical = np.prod([single_accuracies[l] for l in range(args.start, chain_end + 1)])

        print(f"Layers {args.start}-{chain_end} ({n_layers_compressed} layers): "
              f"{matches}/{len(held_out)} ({accuracy*100:.0f}%) "
              f"[theoretical: {theoretical*100:.0f}%]")

    # Show failures for full chain
    print(f"\n{'='*70}")
    print(f"FULL CHAIN RESULTS (Layers {args.start}-{args.end})")
    print("="*70)

    matches, results = test_chained_layers(
        model, tokenizer, held_out,
        transforms, args.start, args.end
    )

    print(f"\n{'Prompt':<40} | {'Match':>6} | {'Expected':>12} | {'Got':>12}")
    print("-" * 80)

    for r in results:
        status = "OK" if r['match'] else "FAIL"
        print(f"{r['prompt'][:40]:<40} | {status:>6} | {r['expected'][:12]:>12} | {r['got'][:12]:>12}")

    print(f"\nFinal accuracy: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)

    n_compressed = args.end - args.start + 1
    compression_ratio = n_layers / (n_layers - n_compressed + 1)

    print(f"""
Compressed {n_compressed} layers ({args.start}-{args.end}):
- Single-layer accuracies: {[f"{single_accuracies[l]*100:.0f}%" for l in range(args.start, args.end+1)]}
- Chained accuracy: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)
- Effective compression: {compression_ratio:.1f}x (removing {n_compressed} layers)

The chained accuracy shows whether errors accumulate during compression.
If chain ≈ product of singles, errors don't accumulate.
If chain < product, errors compound through the chain.
""")


if __name__ == "__main__":
    main()
