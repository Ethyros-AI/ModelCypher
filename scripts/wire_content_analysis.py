#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Wire Content Analysis
"""
Wire Content Analysis

THE QUESTION:
If 86% of Qwen3-1.7B is just a wire passing a 1D signal, what's ON the wire?

The signal must contain EVERYTHING the model needs for output:
- Token identity
- Semantic content
- Context
- Next-token prediction

METHOD:
1. Extract the 1D value at the start of the wire (layer 3)
2. Test if we can decode the next token from 1D alone
3. Compare 1D values for different prompts

If the 1D encodes "next token", the model is basically a lookup table.
If the 1D encodes something more abstract, there's still computation happening.

Usage:
    python wire_content_analysis.py --model /path/to/model
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


TEST_PROMPTS = [
    # Same topic, different answers
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),

    # Same pattern, different domains
    ("Dogs are", "animals"),
    ("Cats are", "animals"),
    ("Apples are", "fruits"),
    ("Cars are", "vehicles"),

    # Semantic pairs
    ("Love is the opposite of", "hate"),
    ("Hot is the opposite of", "cold"),
    ("Fast is the opposite of", "slow"),
    ("Good is the opposite of", "bad"),
]


def get_wire_state(
    model: Any,
    tokenizer: Any,
    prompt: str,
    wire_start_layer: int = 3,
) -> np.ndarray:
    """Get the hidden state at the start of the 'wire' (layer 3 for Qwen3)."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    inner_model = model.model if hasattr(model, 'model') else model

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(inner_model.layers):
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

        if idx == wire_start_layer:
            return np.array(h[0, -1, :].astype(mx.float32))

    return np.array(h[0, -1, :].astype(mx.float32))


def project_to_1d(states: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Project states to 1D using PCA."""
    states = np.stack(states)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    mean = states.mean(axis=0)
    centered = states - mean
    cov = (centered.T @ centered) / len(states)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]

    pc1 = eigenvectors[:, idx[0]]
    values_1d = centered @ pc1

    return values_1d, pc1


def main():
    parser = argparse.ArgumentParser(description="Wire content analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)

    print(f"\n{'='*80}")
    print("WIRE CONTENT ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers")

    # Determine wire start layer
    wire_start = 3 if n_layers > 20 else n_layers // 4

    print(f"Analyzing wire at layer {wire_start}")

    # Get wire states for all prompts
    states = []
    for prompt, expected in TEST_PROMPTS:
        state = get_wire_state(model, tokenizer, prompt, wire_start)
        states.append(state)

    # Project to 1D
    values_1d, pc1 = project_to_1d(states)

    # Analyze
    print(f"\n{'='*80}")
    print("1D WIRE VALUES")
    print("="*80)

    print(f"\n{'Prompt':<40} | {'Expected':<10} | {'1D Value':>10}")
    print("-" * 70)

    for i, (prompt, expected) in enumerate(TEST_PROMPTS):
        print(f"{prompt:<40} | {expected:<10} | {values_1d[i]:>10.4f}")

    # Group analysis
    print(f"\n{'='*80}")
    print("PATTERN ANALYSIS")
    print("="*80)

    # Capital prompts (0-3)
    capital_values = values_1d[0:4]
    print(f"\nCapital prompts: {capital_values}")
    print(f"  Mean: {np.mean(capital_values):.4f}, Std: {np.std(capital_values):.4f}")

    # Category prompts (4-7)
    category_values = values_1d[4:8]
    print(f"\nCategory prompts: {category_values}")
    print(f"  Mean: {np.mean(category_values):.4f}, Std: {np.std(category_values):.4f}")

    # Opposite prompts (8-11)
    opposite_values = values_1d[8:12]
    print(f"\nOpposite prompts: {opposite_values}")
    print(f"  Mean: {np.mean(opposite_values):.4f}, Std: {np.std(opposite_values):.4f}")

    # The insight
    print(f"\n{'='*80}")
    print("WHAT'S ON THE WIRE?")
    print("="*80)

    # Check if same-pattern prompts cluster
    capital_std = np.std(capital_values)
    category_std = np.std(category_values)
    opposite_std = np.std(opposite_values)
    overall_std = np.std(values_1d)

    within_cluster = np.mean([capital_std, category_std, opposite_std])
    between_cluster = overall_std

    if within_cluster < between_cluster * 0.5:
        print(f"""
PATTERN CLUSTERING DETECTED!

Within-cluster std: {within_cluster:.4f}
Between-cluster std: {between_cluster:.4f}

Same-pattern prompts have SIMILAR 1D values!
The wire encodes the PATTERN/TEMPLATE, not the specific content.

"The capital of X is" → similar 1D values regardless of X
"Y is the opposite of" → similar 1D values regardless of Y

This means the 1D is encoding:
- Syntactic structure
- Semantic pattern
- "What kind of answer is expected"

NOT:
- The specific token to output
- The factual content
""")
    else:
        print(f"""
NO CLEAR CLUSTERING

Within-cluster std: {within_cluster:.4f}
Between-cluster std: {between_cluster:.4f}

The 1D values don't cluster by pattern.
Each prompt gets a relatively unique 1D value.

This suggests the 1D encodes something more specific:
- Token identity
- Full semantic content
- A hash of the entire context
""")

    # Test: Can we predict output from 1D?
    print(f"\n{'='*80}")
    print("CAN WE PREDICT OUTPUT FROM 1D ALONE?")
    print("="*80)

    # Simple test: do similar 1D values predict similar outputs?
    # Sort by 1D value and check if outputs cluster
    sorted_indices = np.argsort(values_1d)
    print("\nPrompts sorted by 1D value (low → high):")
    for i, idx in enumerate(sorted_indices):
        prompt, expected = TEST_PROMPTS[idx]
        print(f"  {i+1}. [{values_1d[idx]:>7.3f}] {prompt[:30]}... → {expected}")


if __name__ == "__main__":
    main()
