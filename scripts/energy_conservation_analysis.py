#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Energy Conservation Analysis
"""
Energy Conservation Analysis

Measures how energy (sum of squared activations) flows through the network.

Key insight: If energy is conserved, total_energy_out ≈ total_energy_in for each layer.
The compression opportunity is finding what portion of each layer's output space
actually carries the energy forward.

Usage:
    python energy_conservation_analysis.py \
        --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Semantic primes for probing
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
    """Get semantic primes with minimal contexts for activation."""
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
            elif prime in ["say"]:
                context = "I say"
            elif prime in ["words"]:
                context = "words"
            elif prime in ["do", "happen", "move"]:
                context = f"Things {prime}"
            elif prime in ["there is"]:
                context = "There is something"
            elif prime in ["be"]:
                context = "I am"
            elif prime in ["live", "die"]:
                context = f"People {prime}"
            elif prime in ["have", "part"]:
                context = "I have"
            elif prime in ["not"]:
                context = "not this"
            elif prime in ["maybe"]:
                context = "maybe"
            elif prime in ["can"]:
                context = "I can"
            elif prime in ["because", "if"]:
                context = prime
            elif prime in ["when", "now", "before", "after"]:
                context = prime
            elif prime in ["a long time", "a short time", "moment"]:
                context = prime
            elif prime in ["where", "here"]:
                context = prime
            elif prime in ["above", "below", "far", "near", "inside"]:
                context = prime
            elif prime in ["side", "touch"]:
                context = prime
            elif prime in ["kind of", "part of", "like"]:
                context = prime
            else:
                context = prime
            contexts.append((prime, context, category))
    return contexts


def measure_layer_energy(
    model: Any,
    tokenizer: Any,
    layer_idx: int,
) -> dict[str, float]:
    """Measure energy at layer input, after attention, and after MLP.

    Returns dict with:
        - energy_in: energy of h before layer
        - energy_after_attn: energy after attention + residual
        - energy_after_mlp: energy after MLP + residual (layer output)
        - delta_attn_energy: energy of attention delta alone
        - delta_mlp_energy: energy of MLP delta alone
        - conservation_ratio: energy_out / energy_in
    """
    import mlx.core as mx

    contexts = get_prime_contexts()

    energies = {
        "energy_in": [],
        "energy_after_attn": [],
        "energy_after_mlp": [],
        "delta_attn_energy": [],
        "delta_mlp_energy": [],
    }

    for prime, context, category in contexts:
        try:
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            # Forward to layer
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

                    # Get layer components (LFM2 architecture)
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
                        raise ValueError(f"Unknown layer type: {layer_keys}")

                    # Attention path
                    h_normed = norm1(h)
                    mx.eval(h_normed)
                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    mx.eval(attn_out)
                    h_after_attn = h + attn_out
                    mx.eval(h_after_attn)

                    # MLP path
                    h_before_mlp = norm2(h_after_attn)
                    mx.eval(h_before_mlp)
                    mlp_out = mlp(h_before_mlp)
                    mx.eval(mlp_out)
                    h_after_mlp = h_after_attn + mlp_out
                    mx.eval(h_after_mlp)

                    # Compute energies (last token)
                    h_in_vec = h_in[0, -1, :]
                    h_attn_vec = h_after_attn[0, -1, :]
                    h_out_vec = h_after_mlp[0, -1, :]
                    delta_attn = attn_out[0, -1, :]
                    delta_mlp = mlp_out[0, -1, :]

                    energies["energy_in"].append(float(mx.sum(h_in_vec * h_in_vec)))
                    energies["energy_after_attn"].append(float(mx.sum(h_attn_vec * h_attn_vec)))
                    energies["energy_after_mlp"].append(float(mx.sum(h_out_vec * h_out_vec)))
                    energies["delta_attn_energy"].append(float(mx.sum(delta_attn * delta_attn)))
                    energies["delta_mlp_energy"].append(float(mx.sum(delta_mlp * delta_mlp)))
                    break

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    if not energies["energy_in"]:
        raise ValueError(f"No energy data collected for layer {layer_idx}")

    # Average across all probes
    result = {
        "energy_in": sum(energies["energy_in"]) / len(energies["energy_in"]),
        "energy_after_attn": sum(energies["energy_after_attn"]) / len(energies["energy_after_attn"]),
        "energy_after_mlp": sum(energies["energy_after_mlp"]) / len(energies["energy_after_mlp"]),
        "delta_attn_energy": sum(energies["delta_attn_energy"]) / len(energies["delta_attn_energy"]),
        "delta_mlp_energy": sum(energies["delta_mlp_energy"]) / len(energies["delta_mlp_energy"]),
    }

    # Conservation ratios
    result["conservation_layer"] = result["energy_after_mlp"] / result["energy_in"]
    result["conservation_attn"] = result["energy_after_attn"] / result["energy_in"]

    # How much energy delta contributes vs residual
    result["delta_fraction"] = (result["delta_attn_energy"] + result["delta_mlp_energy"]) / result["energy_after_mlp"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Energy conservation analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)
    logger.info("Model has %d layers", n_layers)

    print("\n" + "=" * 100)
    print("ENERGY FLOW ANALYSIS")
    print("=" * 100)
    print(f"{'Layer':>5} | {'E_in':>10} | {'E_out':>10} | {'Ratio':>8} | {'Δ_attn':>10} | {'Δ_mlp':>10} | {'Δ/E_out':>8}")
    print("-" * 100)

    cumulative_energy = None
    layer_data = []

    for layer_idx in range(n_layers):
        try:
            data = measure_layer_energy(model, tokenizer, layer_idx)
            layer_data.append(data)

            if cumulative_energy is None:
                cumulative_energy = data["energy_in"]

            print(f"{layer_idx:>5} | {data['energy_in']:>10.2f} | {data['energy_after_mlp']:>10.2f} | "
                  f"{data['conservation_layer']:>8.4f} | {data['delta_attn_energy']:>10.2f} | "
                  f"{data['delta_mlp_energy']:>10.2f} | {data['delta_fraction']:>8.4f}")

        except Exception as e:
            logger.error("Layer %d failed: %s", layer_idx, e)

    print("-" * 100)

    # Summary statistics
    if layer_data:
        print("\nSUMMARY:")

        # Energy growth through network
        initial_energy = layer_data[0]["energy_in"]
        final_energy = layer_data[-1]["energy_after_mlp"]
        print(f"  Initial energy (after embed): {initial_energy:.2f}")
        print(f"  Final energy (after layer {n_layers-1}): {final_energy:.2f}")
        print(f"  Total growth factor: {final_energy / initial_energy:.4f}x")

        # Find where energy concentrates
        mlp_energies = [d["delta_mlp_energy"] for d in layer_data]
        attn_energies = [d["delta_attn_energy"] for d in layer_data]

        max_mlp_layer = mlp_energies.index(max(mlp_energies))
        max_attn_layer = attn_energies.index(max(attn_energies))

        print(f"\n  Peak MLP delta energy: layer {max_mlp_layer} ({mlp_energies[max_mlp_layer]:.2f})")
        print(f"  Peak ATTN delta energy: layer {max_attn_layer} ({attn_energies[max_attn_layer]:.2f})")

        # Conservation check
        conservation_ratios = [d["conservation_layer"] for d in layer_data]
        avg_conservation = sum(conservation_ratios) / len(conservation_ratios)
        print(f"\n  Average layer conservation ratio: {avg_conservation:.4f}")
        print(f"  Min: {min(conservation_ratios):.4f} (layer {conservation_ratios.index(min(conservation_ratios))})")
        print(f"  Max: {max(conservation_ratios):.4f} (layer {conservation_ratios.index(max(conservation_ratios))})")

        # Energy budget per layer
        print("\n  ENERGY BUDGET (delta / total per layer):")
        for i, d in enumerate(layer_data):
            total_delta = d["delta_attn_energy"] + d["delta_mlp_energy"]
            mlp_share = d["delta_mlp_energy"] / total_delta if total_delta > 0 else 0
            print(f"    Layer {i:>2}: Δ={total_delta:>8.2f}, MLP share={mlp_share:.1%}, ratio={d['conservation_layer']:.4f}")


if __name__ == "__main__":
    main()
