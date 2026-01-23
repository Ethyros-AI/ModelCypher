#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Steering Experiment
"""
Steering Experiment

THE HYPOTHESIS:
If the 1D wire carries template codes:
  - Capital prompts → ~-10
  - Category prompts → ~-13
  - Opposite prompts → ~+23

Then we can STEER by shifting the template code.

EXPERIMENT:
1. Take a "capital of X" prompt (template code ~-10)
2. Shift the hidden state to match "opposite of Y" template (~+23)
3. See if the model responds with an opposite instead of a capital

If this works, we have a CONTROL MECHANISM for language model outputs.

Usage:
    python steering_experiment.py --model /path/to/model
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


def get_wire_state_and_basis(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    wire_layer: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Get wire states and compute PCA basis.

    Returns: (states, pc1, mean)
    """
    import mlx.core as mx

    states = []
    inner_model = model.model if hasattr(model, 'model') else model

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(inner_model.layers):
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

            if idx == wire_layer:
                states.append(np.array(h[0, -1, :].astype(mx.float32)))
                break

    # Compute PCA basis
    states_arr = np.stack(states)
    states_arr = np.nan_to_num(states_arr, nan=0.0, posinf=0.0, neginf=0.0)

    mean = states_arr.mean(axis=0)
    centered = states_arr - mean
    cov = (centered.T @ centered) / len(states_arr)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    pc1 = eigenvectors[:, idx[0]]

    return states, pc1, mean


def generate_with_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    wire_layer: int,
    pc1: np.ndarray,
    mean: np.ndarray,
    target_1d_value: float,
    max_tokens: int = 10,
) -> str:
    """Generate text with steering at the wire layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    inner_model = model.model if hasattr(model, 'model') else model

    # Run through layers with steering
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(inner_model.layers):
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

        if idx == wire_layer:
            # Apply steering
            h_np = np.array(h.astype(mx.float32))
            last_token = h_np[0, -1, :]

            # Current 1D value
            current_1d = np.dot(last_token - mean, pc1)

            # Compute steering delta
            delta_1d = target_1d_value - current_1d
            steering_vector = delta_1d * pc1

            # Apply steering
            h_np[0, -1, :] = last_token + steering_vector
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

    # Final norm
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
    elif hasattr(inner_model, 'embedding_norm'):
        h = inner_model.embedding_norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        # Fallback
        logits = model(input_ids)
    mx.eval(logits)

    # Generate tokens
    generated = []
    for _ in range(max_tokens):
        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        # Stop on EOS or padding
        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)

        # Continue generation (without steering for subsequent tokens)
        input_ids = mx.array([[next_token]])
        logits = model(input_ids)
        mx.eval(logits)

    return tokenizer.decode(generated)


def generate_normal(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 10,
) -> str:
    """Generate text normally without steering."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.array([[next_token]])

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Steering experiment")
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
    print("STEERING EXPERIMENT")
    print("="*80)
    print(f"Model: {n_layers} layers")

    # Determine wire layer
    wire_layer = 3 if n_layers > 20 else 7

    # Calibration prompts to establish template codes
    calibration_prompts = [
        # Capitals (should cluster around one value)
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
        "The capital of Italy is",
        # Opposites (should cluster around another value)
        "Love is the opposite of",
        "Hot is the opposite of",
        "Fast is the opposite of",
        "Good is the opposite of",
    ]

    print(f"\nCalibrating template codes at layer {wire_layer}...")

    # Get calibration states
    states, pc1, mean = get_wire_state_and_basis(
        model, tokenizer, calibration_prompts, wire_layer
    )

    # Compute 1D values
    values_1d = [np.dot(s - mean, pc1) for s in states]

    capital_mean = np.mean(values_1d[:4])
    opposite_mean = np.mean(values_1d[4:])

    print(f"\nTemplate codes:")
    print(f"  Capital template: {capital_mean:.2f}")
    print(f"  Opposite template: {opposite_mean:.2f}")

    # Steering experiment
    print(f"\n{'='*80}")
    print("STEERING TEST")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "The capital of Spain is",
        "Love is the opposite of",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")

        # Normal generation
        normal_output = generate_normal(model, tokenizer, prompt, max_tokens=5)
        print(f"  Normal output: {normal_output}")

        # Steered to capital template
        steered_capital = generate_with_steering(
            model, tokenizer, prompt, wire_layer,
            pc1, mean, capital_mean, max_tokens=5
        )
        print(f"  Steered to CAPITAL template ({capital_mean:.1f}): {steered_capital}")

        # Steered to opposite template
        steered_opposite = generate_with_steering(
            model, tokenizer, prompt, wire_layer,
            pc1, mean, opposite_mean, max_tokens=5
        )
        print(f"  Steered to OPPOSITE template ({opposite_mean:.1f}): {steered_opposite}")

    # Analysis
    print(f"\n{'='*80}")
    print("STEERING ANALYSIS")
    print("="*80)

    print(f"""
If steering WORKS:
- "Capital of France" steered to opposite template → should output something like "hate" or negation
- "Opposite of love" steered to capital template → should output something like "Paris" or a place

If steering FAILS:
- Outputs remain similar regardless of steering
- The 1D might encode something more complex than simple template type

IMPLICATIONS:
- Working steering = we have a control mechanism for LLM outputs
- The template code is a "mode switch" for the model
- Safety implications: steering could be used for good (alignment) or bad (jailbreaking)
""")


if __name__ == "__main__":
    main()
