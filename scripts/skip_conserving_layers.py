#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Skip Conserving Layers
"""
Skip Conserving Layers

Hypothesis: Layers with conservation ratio ≈ 1.0 are identity operations.
They can be SKIPPED without losing information.

From analysis:
- Layers 8-13 have ratio 1.00-1.05 (conserving)
- These layers have ||δ||² ≈ 0.1 (tiny deltas)

Test: Skip these layers and see if model still works.

Usage:
    python skip_conserving_layers.py \
        --model /path/to/model \
        --skip-layers 8,9,10,11,12,13 \
        --test
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def inference_with_skipped_layers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    skip_layers: set[int],
    max_tokens: int = 30,
) -> str:
    """Run inference while skipping specified layers."""
    import mlx.core as mx

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    output_tokens = list(tokens)

    for _ in range(max_tokens):
        input_ids = mx.array([output_tokens])
        mx.eval(input_ids)

        # Manual forward pass with layer skipping
        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(model.model.layers):
            if idx in skip_layers:
                # SKIP this layer - just pass h through unchanged
                continue

            # Normal layer forward pass
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

        # Final norm (embedding_norm for this model)
        if hasattr(model.model, 'embedding_norm'):
            h = model.model.embedding_norm(h)
            mx.eval(h)
        elif hasattr(model.model, 'norm'):
            h = model.model.norm(h)
            mx.eval(h)

        # Compute logits using tied embeddings
        embed_weights = model.model.embed_tokens.weight
        logits = h @ embed_weights.T
        mx.eval(logits)

        # Greedy sampling from last position
        next_token = int(mx.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens)


def normal_inference(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 30) -> str:
    """Normal inference for comparison."""
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Skip conserving layers")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--skip-layers", type=str, default="8,9,10,11,12,13",
                        help="Comma-separated layers to skip")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    skip_layers = set(int(x.strip()) for x in args.skip_layers.split(","))

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    logger.info("Model: %d layers", n_layers)
    logger.info("Skipping layers: %s", sorted(skip_layers))
    logger.info("Active layers: %s", [i for i in range(n_layers) if i not in skip_layers])

    prompts = [
        "The answer to 2+2 is",
        "Hello, my name is",
        "The capital of France is",
    ]

    print("\n" + "=" * 70)
    print(f"COMPARISON: All layers vs Skip {sorted(skip_layers)}")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Normal
        normal_out = normal_inference(model, tokenizer, prompt, 20)
        print(f"  ALL layers:  {normal_out[len(prompt):][:50]}")

        # Skipped
        skip_out = inference_with_skipped_layers(model, tokenizer, prompt, skip_layers, 20)
        print(f"  SKIP layers: {skip_out[len(prompt):][:50]}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Layers skipped: {len(skip_layers)}/{n_layers} = {len(skip_layers)/n_layers*100:.0f}%")
    print(f"If outputs match, these layers are redundant!")


if __name__ == "__main__":
    main()
