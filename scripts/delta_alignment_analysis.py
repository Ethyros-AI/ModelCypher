#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Delta Alignment Analysis
"""
Delta Alignment Analysis

Energy conservation formula: ||h_out||² = ||h_in||² + ||delta||² + 2<h_in, delta>

The cross term 2<h_in, delta> determines whether energy grows or shrinks.
- If delta ⊥ h_in (cos θ = 0): energy grows by ||delta||²
- If delta aligned with h_in (cos θ > 0): energy grows more
- If delta anti-aligned (cos θ < 0): energy can decrease

This measures cos(θ) between delta and h_in at each layer.

Usage:
    python delta_alignment_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SEMANTIC_PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other", "else"],
    "quantifiers": ["one", "two", "some", "all", "much", "many", "little", "few"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "feel", "see", "hear"],
    "speech": ["say", "words", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be", "live", "die"],
    "possession": ["have", "part"],
    "logical": ["not", "maybe", "can", "because", "if"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "moment"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside", "touch"],
    "taxonomy": ["kind of", "like"],
}


def get_prime_contexts() -> list[tuple[str, str, str]]:
    """Get semantic primes with minimal contexts."""
    contexts = []
    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime
            elif prime in ["this", "the same", "other", "else"]:
                context = f"{prime} thing"
            elif prime in ["one", "two", "some", "all", "many", "much", "little", "few"]:
                context = f"{prime} things"
            elif prime in ["good", "bad", "big", "small", "true"]:
                context = f"It is {prime}"
            elif prime in ["think", "know", "want", "feel", "see", "hear"]:
                context = f"I {prime}"
            else:
                context = prime
            contexts.append((prime, context, category))
    return contexts


def measure_delta_alignment(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> dict[str, float]:
    """Measure alignment between delta and residual stream at layer.

    Returns:
        - cos_attn: cosine(h_in, delta_attn)
        - cos_mlp: cosine(h_after_attn, delta_mlp)
        - cos_total: cosine(h_in, total_delta)
        - cross_term: 2<h_in, delta> contribution to energy change
    """
    import mlx.core as mx

    contexts = get_prime_contexts()

    measurements = {
        "cos_attn": [],
        "cos_mlp": [],
        "cos_total": [],
        "cross_term_attn": [],
        "cross_term_mlp": [],
        "expected_ratio": [],  # What ratio SHOULD be if formula holds
    }

    for prime, context, category in contexts[:62]:  # Use all 62 primes
        try:
            tokens = tokenizer.encode(context)
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

                    layer_keys = list(layer.keys()) if hasattr(layer, 'keys') else []

                    if 'operator_norm' in layer_keys:
                        norm1 = layer['operator_norm']
                        norm2 = layer['ffn_norm']
                        mlp = layer['feed_forward']
                        if 'conv' in layer_keys:
                            self_attn = layer['conv']
                        else:
                            self_attn = layer['self_attn']
                    else:
                        raise ValueError(f"Unknown layer type")

                    # Attention
                    h_normed = norm1(h)
                    mx.eval(h_normed)
                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    mx.eval(attn_out)
                    h_after_attn = h + attn_out
                    mx.eval(h_after_attn)

                    # MLP
                    h_before_mlp = norm2(h_after_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)
                    h_out = h_after_attn + mlp_out
                    mx.eval(h_out)

                    # Extract last token
                    h_in_vec = h_in[0, -1, :]
                    delta_attn = attn_out[0, -1, :]
                    h_attn_vec = h_after_attn[0, -1, :]
                    delta_mlp = mlp_out[0, -1, :]
                    h_out_vec = h_out[0, -1, :]
                    total_delta = delta_attn + delta_mlp

                    # Norms
                    norm_h_in = float(mx.sqrt(mx.sum(h_in_vec * h_in_vec)))
                    norm_delta_attn = float(mx.sqrt(mx.sum(delta_attn * delta_attn)))
                    norm_h_attn = float(mx.sqrt(mx.sum(h_attn_vec * h_attn_vec)))
                    norm_delta_mlp = float(mx.sqrt(mx.sum(delta_mlp * delta_mlp)))
                    norm_total_delta = float(mx.sqrt(mx.sum(total_delta * total_delta)))

                    # Cosines
                    if norm_h_in > 1e-8 and norm_delta_attn > 1e-8:
                        cos_attn = float(mx.sum(h_in_vec * delta_attn)) / (norm_h_in * norm_delta_attn)
                    else:
                        cos_attn = 0.0

                    if norm_h_attn > 1e-8 and norm_delta_mlp > 1e-8:
                        cos_mlp = float(mx.sum(h_attn_vec * delta_mlp)) / (norm_h_attn * norm_delta_mlp)
                    else:
                        cos_mlp = 0.0

                    if norm_h_in > 1e-8 and norm_total_delta > 1e-8:
                        cos_total = float(mx.sum(h_in_vec * total_delta)) / (norm_h_in * norm_total_delta)
                    else:
                        cos_total = 0.0

                    # Cross terms (actual contribution to energy)
                    cross_attn = 2 * float(mx.sum(h_in_vec * delta_attn))
                    cross_mlp = 2 * float(mx.sum(h_attn_vec * delta_mlp))

                    # Expected energy ratio from formula
                    E_in = norm_h_in ** 2
                    E_delta_attn = norm_delta_attn ** 2
                    E_delta_mlp = norm_delta_mlp ** 2
                    expected_E_out = E_in + E_delta_attn + cross_attn + E_delta_mlp + cross_mlp
                    if E_in > 1e-8:
                        expected_ratio = expected_E_out / E_in
                    else:
                        expected_ratio = 1.0

                    measurements["cos_attn"].append(cos_attn)
                    measurements["cos_mlp"].append(cos_mlp)
                    measurements["cos_total"].append(cos_total)
                    measurements["cross_term_attn"].append(cross_attn)
                    measurements["cross_term_mlp"].append(cross_mlp)
                    measurements["expected_ratio"].append(expected_ratio)
                    break

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    if not measurements["cos_attn"]:
        raise ValueError(f"No data for layer {layer_idx}")

    # Average
    result = {}
    for key, values in measurements.items():
        result[key] = sum(values) / len(values)

    return result


def main():
    parser = argparse.ArgumentParser(description="Delta alignment analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print("\n" + "=" * 110)
    print("DELTA ALIGNMENT ANALYSIS")
    print("=" * 110)
    print(f"{'Layer':>5} | {'cos(h,Δ_attn)':>14} | {'cos(h,Δ_mlp)':>13} | {'cos(h,Δ_tot)':>13} | "
          f"{'Cross_attn':>11} | {'Cross_mlp':>11}")
    print("-" * 110)

    layer_data = []
    for layer_idx in range(n_layers):
        try:
            data = measure_delta_alignment(model, tokenizer, layer_idx)
            layer_data.append(data)

            # Convert cosine to angle in degrees
            angle_attn = math.degrees(math.acos(max(-1, min(1, data["cos_attn"]))))
            angle_mlp = math.degrees(math.acos(max(-1, min(1, data["cos_mlp"]))))
            angle_total = math.degrees(math.acos(max(-1, min(1, data["cos_total"]))))

            print(f"{layer_idx:>5} | {data['cos_attn']:>+8.4f} ({angle_attn:>4.0f}°) | "
                  f"{data['cos_mlp']:>+7.4f} ({angle_mlp:>4.0f}°) | "
                  f"{data['cos_total']:>+7.4f} ({angle_total:>4.0f}°) | "
                  f"{data['cross_term_attn']:>+11.4f} | {data['cross_term_mlp']:>+11.4f}")

        except Exception as e:
            logger.error("Layer %d failed: %s", layer_idx, e)

    print("-" * 110)

    if layer_data:
        print("\nINTERPRETATION:")
        print("  cos = +1: delta aligned with residual (energy grows)")
        print("  cos =  0: delta orthogonal (energy grows by ||delta||²)")
        print("  cos = -1: delta anti-aligned (energy shrinks)")
        print()

        # Find special layers
        cos_totals = [d["cos_total"] for d in layer_data]
        most_aligned = cos_totals.index(max(cos_totals))
        most_anti = cos_totals.index(min(cos_totals))

        print(f"  Most aligned delta: layer {most_aligned} (cos={cos_totals[most_aligned]:+.4f})")
        print(f"  Most anti-aligned: layer {most_anti} (cos={cos_totals[most_anti]:+.4f})")

        # Cross term analysis
        cross_totals = [d["cross_term_attn"] + d["cross_term_mlp"] for d in layer_data]
        max_cross = cross_totals.index(max(cross_totals))
        min_cross = cross_totals.index(min(cross_totals))

        print(f"\n  Largest positive cross term: layer {max_cross} ({cross_totals[max_cross]:+.4f})")
        print(f"  Largest negative cross term: layer {min_cross} ({cross_totals[min_cross]:+.4f})")

        # Which layers are near orthogonal?
        print("\n  NEAR-ORTHOGONAL layers (|cos| < 0.1):")
        for i, d in enumerate(layer_data):
            if abs(d["cos_total"]) < 0.1:
                print(f"    Layer {i}: cos={d['cos_total']:+.4f}")


if __name__ == "__main__":
    main()
