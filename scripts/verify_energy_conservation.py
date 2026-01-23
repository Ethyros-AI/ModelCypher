#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Verify Energy Conservation
"""
Verify Energy Conservation After Compression

Compares energy flow between original and compressed models.

Usage:
    python verify_energy_conservation.py \
        --original /path/to/original \
        --compressed /path/to/compressed
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


def measure_energy_at_layer(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
    prompt: str,
) -> dict[str, float]:
    """Measure energy before and after a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = model.model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(model.model.layers):
        if idx < layer_idx:
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)
        elif idx == layer_idx:
            h_in = mx.array(h)
            mx.eval(h_in)

            result = layer(h)
            h_out = result[0] if isinstance(result, tuple) else result
            mx.eval(h_out)

            # Energy at last token
            e_in = float(mx.sum(h_in[0, -1, :] ** 2))
            e_out = float(mx.sum(h_out[0, -1, :] ** 2))

            return {
                "energy_in": e_in,
                "energy_out": e_out,
                "ratio": e_out / e_in if e_in > 0 else 0,
            }

    return {"energy_in": 0, "energy_out": 0, "ratio": 0}


def main():
    parser = argparse.ArgumentParser(description="Verify energy conservation")
    parser.add_argument("--original", type=str, required=True, help="Original model")
    parser.add_argument("--compressed", type=str, required=True, help="Compressed model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading original model from %s", args.original)
    orig_model, orig_tokenizer = load(args.original)
    mx.eval(orig_model.parameters())

    logger.info("Loading compressed model from %s", args.compressed)
    comp_model, comp_tokenizer = load(args.compressed)
    mx.eval(comp_model.parameters())

    prompts = [
        "The capital of France is",
        "I think therefore I",
        "One plus one equals",
    ]

    n_layers = len(orig_model.model.layers)

    for prompt in prompts:
        print(f"\n{'=' * 80}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 80}")
        print(f"{'Layer':>5} | {'Orig E_in':>10} | {'Orig E_out':>10} | {'Orig Ratio':>10} | "
              f"{'Comp E_out':>10} | {'Comp Ratio':>10} | {'Delta':>8}")
        print("-" * 80)

        for layer_idx in range(n_layers):
            orig = measure_energy_at_layer(orig_model, orig_tokenizer, layer_idx, prompt)
            comp = measure_energy_at_layer(comp_model, comp_tokenizer, layer_idx, prompt)

            delta = (comp["ratio"] - orig["ratio"]) / orig["ratio"] * 100 if orig["ratio"] > 0 else 0

            # Highlight significant changes
            marker = " **" if abs(delta) > 10 else ""

            print(f"{layer_idx:>5} | {orig['energy_in']:>10.4f} | {orig['energy_out']:>10.4f} | "
                  f"{orig['ratio']:>10.4f} | {comp['energy_out']:>10.4f} | "
                  f"{comp['ratio']:>10.4f} | {delta:>+7.1f}%{marker}")


if __name__ == "__main__":
    main()
