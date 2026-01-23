#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Dtype Round-Trip Test
"""
Dtype Round-Trip Test

THE ISSUE:
Even K=2048 (no compression) breaks generation.
We're converting bf16 → float32 → bf16 every layer.

TEST:
Does the dtype round-trip alone cause errors?

Usage:
    python dtype_roundtrip_test.py --model /path/to/model
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


def test_roundtrip_error(model: Any, tokenizer: Any, prompt: str):
    """Test how much error accumulates from dtype round-trips."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    # Normal forward pass - collect states
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    original_states = [np.array(h.astype(mx.float32))]

    for layer in inner_model.layers:
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)
        original_states.append(np.array(h.astype(mx.float32)))

    # Forward pass WITH round-trips
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    roundtrip_states = [np.array(h.astype(mx.float32))]
    errors = []

    for idx, layer in enumerate(inner_model.layers):
        # Round-trip conversion (the operation we're doing in compression)
        h_np = np.array(h.astype(mx.float32))
        h = mx.array(h_np).astype(h.dtype)
        mx.eval(h)

        # Run layer
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

        # Store state
        h_np_after = np.array(h.astype(mx.float32))
        roundtrip_states.append(h_np_after)

        # Compute error
        original_h = original_states[idx + 1][0, -1, :]
        roundtrip_h = h_np_after[0, -1, :]
        error = np.linalg.norm(roundtrip_h - original_h) / (np.linalg.norm(original_h) + 1e-10)
        errors.append(error)

    return errors, original_states, roundtrip_states


def test_generation_with_roundtrip(
    model: Any,
    tokenizer: Any,
    prompt: str,
    do_roundtrip: bool,
    max_tokens: int = 10,
) -> str:
    """Test generation with/without dtype round-trips."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    for layer in inner_model.layers:
        if do_roundtrip:
            # Do the round-trip
            h_np = np.array(h.astype(mx.float32))
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)

        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

    # Final norm
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    if hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        logits = model(input_ids)
    mx.eval(logits)

    # Generate first token
    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    generated = []
    if next_token != tokenizer.eos_token_id:
        generated.append(next_token)

    # Continue normally
    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
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
    parser = argparse.ArgumentParser(description="Dtype round-trip test")
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
    print("DTYPE ROUND-TRIP TEST")
    print("="*80)
    print(f"Model: {n_layers} layers, dtype: {inner_model.embed_tokens.weight.dtype}")

    # Test error accumulation
    print(f"\n{'='*80}")
    print("ERROR ACCUMULATION")
    print("="*80)

    prompt = "The capital of France is"
    errors, original, roundtrip = test_roundtrip_error(model, tokenizer, prompt)

    print(f"\nRelative error at each layer (round-trip vs original):")
    print(f"{'Layer':>6} | {'Error':>12}")
    print("-" * 25)

    for i, err in enumerate(errors):
        print(f"{i:>6} | {err:>12.8f}")

    print(f"\nCumulative error: {errors[-1]:.8f}")

    # Test generation
    print(f"\n{'='*80}")
    print("GENERATION TEST")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    for prompt in test_prompts:
        normal = test_generation_with_roundtrip(model, tokenizer, prompt, do_roundtrip=False)
        roundtrip = test_generation_with_roundtrip(model, tokenizer, prompt, do_roundtrip=True)

        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Normal:    {normal[:40]}")
        print(f"  Roundtrip: {roundtrip[:40]}")

        if normal.split() and roundtrip.split():
            if normal.split()[0] == roundtrip.split()[0]:
                print(f"  → MATCH ✓")
            else:
                print(f"  → DIFFERENT")

    # The insight
    print(f"\n{'='*80}")
    print("DTYPE ROUND-TRIP INSIGHT")
    print("="*80)

    if errors[-1] > 0.01:
        print(f"""
THE ROUND-TRIP IS CAUSING ERRORS!

Cumulative error after {n_layers} layers: {errors[-1]:.6f}

This means:
1. bf16 → float32 → bf16 loses precision
2. The loss compounds across layers
3. ALL our compression experiments were corrupted by this!

THE FIX:
Stay in bf16 throughout! Never convert to numpy.
Use MLX operations for all math.
""")
    else:
        print(f"""
Round-trip error is negligible: {errors[-1]:.8f}

The compression failure is NOT due to dtype conversion.
The model is truly sensitive to all dimensions.
""")


if __name__ == "__main__":
    main()
