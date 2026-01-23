#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Layer Information Flow Analysis
"""
Layer Information Flow Analysis

THE QUESTION:
If Qwen3-1.7B has a 1D highway from layers 3-27, what are those layers actually DOING?

HYPOTHESIS 1: Pure transmission
- The 1D highway just passes information through unchanged
- All "thinking" happens in layers 0-2
- Layers 3-27 are redundant

HYPOTHESIS 2: Gradual refinement
- Each layer makes small adjustments
- The 1D is enough to carry the core meaning
- Details are refined layer by layer

HYPOTHESIS 3: Rotation without loss
- Information rotates but isn't added/removed
- The Gram matrix preservation IS the computation
- "Thinking" = maintaining relationships while rotating

METHOD:
1. Measure how much each layer CHANGES the representation
2. Compute "information added" = ||delta||² / ||h||²
3. Compare across architectures

If 1D layers add near-zero information, they're transmission.
If they add significant information, they're computing despite the bottleneck.

Usage:
    python layer_information_flow.py --model /path/to/model
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


CONCEPTS = [
    "apple", "orange", "banana", "fruit",
    "dog", "cat", "bird", "animal",
    "car", "truck", "bike", "vehicle",
    "hot", "cold", "warm", "temperature",
    "good", "bad", "love", "hate",
    "red", "blue", "green", "yellow",
    "fast", "slow", "quiet", "loud",
]


def get_all_layer_states(
    model: Any,
    tokenizer: Any,
    word: str,
) -> list[np.ndarray]:
    """Get hidden states at every layer for a word."""
    import mlx.core as mx

    tokens = tokenizer.encode(word)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    inner_model = model.model if hasattr(model, 'model') else model

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    states = [np.array(h[0, -1, :].astype(mx.float32))]

    for layer in inner_model.layers:
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)
        states.append(np.array(h[0, -1, :].astype(mx.float32)))

    return states


def compute_layer_metrics(states: list[np.ndarray]) -> dict:
    """Compute information flow metrics for each layer transition."""
    metrics = {
        'delta_norm': [],           # ||h_out - h_in||
        'relative_change': [],      # ||delta|| / ||h_in||
        'cosine_similarity': [],    # cos(h_in, h_out)
        'energy_ratio': [],         # ||h_out||² / ||h_in||²
    }

    for i in range(len(states) - 1):
        h_in = states[i]
        h_out = states[i + 1]
        delta = h_out - h_in

        norm_in = np.linalg.norm(h_in)
        norm_out = np.linalg.norm(h_out)
        norm_delta = np.linalg.norm(delta)

        metrics['delta_norm'].append(float(norm_delta))
        metrics['relative_change'].append(float(norm_delta / (norm_in + 1e-10)))

        # Cosine similarity
        if norm_in > 1e-10 and norm_out > 1e-10:
            cos_sim = float(np.dot(h_in, h_out) / (norm_in * norm_out))
        else:
            cos_sim = 0.0
        metrics['cosine_similarity'].append(cos_sim)

        # Energy ratio
        if norm_in > 1e-10:
            metrics['energy_ratio'].append(float((norm_out / norm_in) ** 2))
        else:
            metrics['energy_ratio'].append(0.0)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Layer information flow analysis")
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
    print("LAYER INFORMATION FLOW ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers")
    print(f"Testing {len(CONCEPTS)} concepts")

    # Collect metrics for all concepts
    all_metrics = {
        'delta_norm': [[] for _ in range(n_layers)],
        'relative_change': [[] for _ in range(n_layers)],
        'cosine_similarity': [[] for _ in range(n_layers)],
        'energy_ratio': [[] for _ in range(n_layers)],
    }

    for word in CONCEPTS:
        states = get_all_layer_states(model, tokenizer, word)
        metrics = compute_layer_metrics(states)

        for key in all_metrics:
            for i, val in enumerate(metrics[key]):
                all_metrics[key][i].append(val)

    # Compute averages
    print(f"\n{'Layer':>6} | {'Δ Norm':>10} | {'Rel Change':>10} | {'Cos Sim':>10} | {'Energy':>10} | Interpretation")
    print("-" * 80)

    for layer_idx in range(n_layers):
        avg_delta = np.mean(all_metrics['delta_norm'][layer_idx])
        avg_rel = np.mean(all_metrics['relative_change'][layer_idx])
        avg_cos = np.mean(all_metrics['cosine_similarity'][layer_idx])
        avg_energy = np.mean(all_metrics['energy_ratio'][layer_idx])

        # Interpretation
        if avg_rel < 0.01:
            interp = "TRANSMISSION (near-zero change)"
        elif avg_rel < 0.05:
            interp = "Low refinement"
        elif avg_rel < 0.15:
            interp = "Moderate processing"
        elif avg_rel < 0.30:
            interp = "Active computation"
        else:
            interp = "MAJOR transformation"

        print(f"{layer_idx:>6} | {avg_delta:>10.4f} | {avg_rel:>10.4f} | "
              f"{avg_cos:>10.4f} | {avg_energy:>10.4f} | {interp}")

    # Summary
    print(f"\n{'='*80}")
    print("INFORMATION FLOW SUMMARY")
    print("="*80)

    # Find transmission layers (rel_change < 0.05)
    transmission_layers = []
    active_layers = []
    for layer_idx in range(n_layers):
        avg_rel = np.mean(all_metrics['relative_change'][layer_idx])
        if avg_rel < 0.05:
            transmission_layers.append(layer_idx)
        if avg_rel > 0.15:
            active_layers.append(layer_idx)

    print(f"\nTransmission layers (< 5% change): {transmission_layers if transmission_layers else 'None'}")
    print(f"Active computation layers (> 15% change): {active_layers if active_layers else 'None'}")

    # The insight
    print(f"\n{'='*80}")
    print("WHAT IS THIS MODEL ACTUALLY DOING?")
    print("="*80)

    if len(transmission_layers) > n_layers * 0.5:
        print(f"""
MOST LAYERS ARE TRANSMISSION!

{len(transmission_layers)}/{n_layers} layers change representations by < 5%.

This means:
- The model does most "thinking" in a few active layers
- Other layers primarily ROTATE without adding information
- The 1D bottleneck is not a bug - it's the actual information channel

IMPLICATION: Many layers could potentially be pruned or compressed aggressively.
""")
    elif active_layers:
        print(f"""
DISTRIBUTED COMPUTATION

Active layers: {active_layers}

This model spreads computation across multiple layers.
Each layer contributes meaningfully to the transformation.

The bottleneck structure may compress INTERMEDIATE representations,
not the final information content.
""")
    else:
        print(f"""
GRADUAL REFINEMENT

No single layer dominates.
Information is refined gradually across all layers.
""")

    # Analyze specific patterns
    print(f"\n{'='*80}")
    print("LAYER-BY-LAYER ANALYSIS")
    print("="*80)

    avg_changes = [np.mean(all_metrics['relative_change'][i]) for i in range(n_layers)]

    # First third
    early = np.mean(avg_changes[:n_layers//3])
    middle = np.mean(avg_changes[n_layers//3:2*n_layers//3])
    late = np.mean(avg_changes[2*n_layers//3:])

    print(f"\nAverage relative change by region:")
    print(f"  Early (0-{n_layers//3-1}):   {early:.4f}")
    print(f"  Middle ({n_layers//3}-{2*n_layers//3-1}): {middle:.4f}")
    print(f"  Late ({2*n_layers//3}-{n_layers-1}):   {late:.4f}")

    if early > middle and early > late:
        print("\n→ FRONT-LOADED: Most computation happens early")
    elif late > middle and late > early:
        print("\n→ BACK-LOADED: Most computation happens late")
    elif middle > early and middle > late:
        print("\n→ MIDDLE-HEAVY: Core processing in middle layers")
    else:
        print("\n→ UNIFORM: Computation spread evenly")


if __name__ == "__main__":
    main()
