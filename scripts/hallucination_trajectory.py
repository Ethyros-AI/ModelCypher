#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Hallucination Trajectory Analysis
"""
Hallucination Trajectory Analysis

THE HYPOTHESIS:
If the model stays "on the helix" for correct outputs,
do hallucinations correspond to "falling off the helix"?

METHOD:
1. Compare trajectories for:
   - Correct factual completions (known ground truth)
   - Incorrect/hallucinated completions (wrong facts)
   - Nonsense prompts (should be far from helix)

2. Measure at each layer:
   - Distance from helix center (using PCA)
   - Deviation from expected rotation angle
   - Gram matrix divergence from baseline

If hallucinations = off-helix, we might be able to DETECT them geometrically.

Usage:
    python hallucination_trajectory.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Prompts with known correct/incorrect completions
FACTUAL_PROMPTS = [
    {
        "prompt": "The capital of France is",
        "correct": "Paris",
        "incorrect": "London",
    },
    {
        "prompt": "Water freezes at",
        "correct": "zero degrees",
        "incorrect": "fifty degrees",
    },
    {
        "prompt": "The sun rises in the",
        "correct": "east",
        "incorrect": "west",
    },
    {
        "prompt": "Humans have",
        "correct": "two eyes",
        "incorrect": "three eyes",
    },
    {
        "prompt": "The largest planet is",
        "correct": "Jupiter",
        "incorrect": "Earth",
    },
]

# Nonsense prompts (should be far from normal)
NONSENSE_PROMPTS = [
    "Colorless green ideas sleep",
    "The square root of banana is",
    "If Thursday was a fish then",
    "The smell of mathematics feels",
    "When time turns purple we",
]


@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory through layers."""
    prompt: str
    prompt_type: str  # "correct", "incorrect", "nonsense"

    # Per-layer metrics
    layer_norms: list[float]  # Activation norm at each layer
    layer_deltas: list[float]  # Delta norm at each layer

    # Bottleneck metrics
    bottleneck_values: dict[int, float]  # Layer idx -> 1D value

    # Trajectory smoothness
    trajectory_curvature: float  # How "smooth" is the path


def get_layer_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> list[np.ndarray]:
    """Get hidden states at each layer for a prompt."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    inner_model = model.model if hasattr(model, 'model') else model

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    activations = [np.array(h[0, -1, :].astype(mx.float32))]

    for layer in inner_model.layers:
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)
        activations.append(np.array(h[0, -1, :].astype(mx.float32)))

    return activations


def compute_trajectory_metrics(
    activations: list[np.ndarray],
    bottleneck_layers: list[int],
) -> TrajectoryMetrics:
    """Compute trajectory metrics from layer activations."""
    layer_norms = [float(np.linalg.norm(act)) for act in activations]

    # Deltas between consecutive layers
    layer_deltas = []
    for i in range(1, len(activations)):
        delta = activations[i] - activations[i-1]
        layer_deltas.append(float(np.linalg.norm(delta)))

    # Bottleneck values (norm of delta at bottleneck layers)
    bottleneck_values = {}
    for bl in bottleneck_layers:
        if bl < len(layer_deltas):
            bottleneck_values[bl] = layer_deltas[bl]

    # Trajectory curvature (how much the direction changes)
    curvatures = []
    for i in range(2, len(activations)):
        v1 = activations[i-1] - activations[i-2]
        v2 = activations[i] - activations[i-1]

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)

    avg_curvature = float(np.mean(curvatures)) if curvatures else 0.0

    return TrajectoryMetrics(
        prompt="",
        prompt_type="",
        layer_norms=layer_norms,
        layer_deltas=layer_deltas,
        bottleneck_values=bottleneck_values,
        trajectory_curvature=avg_curvature,
    )


def analyze_prompt_trajectory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_type: str,
    bottleneck_layers: list[int],
) -> TrajectoryMetrics:
    """Analyze the trajectory for a single prompt."""
    activations = get_layer_activations(model, tokenizer, prompt)
    metrics = compute_trajectory_metrics(activations, bottleneck_layers)
    metrics.prompt = prompt
    metrics.prompt_type = prompt_type
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hallucination trajectory analysis")
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
    print("HALLUCINATION TRAJECTORY ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers")

    # Bottleneck layers for LFM2
    bottleneck_layers = [7, 14] if n_layers == 16 else [n_layers // 2]

    all_metrics = []

    # Analyze factual prompts (correct vs incorrect)
    print(f"\n{'='*80}")
    print("FACTUAL PROMPTS")
    print("="*80)

    for item in FACTUAL_PROMPTS:
        # Correct completion
        correct_prompt = item["prompt"] + " " + item["correct"]
        correct_metrics = analyze_prompt_trajectory(
            model, tokenizer, correct_prompt, "correct", bottleneck_layers
        )
        all_metrics.append(correct_metrics)

        # Incorrect completion (potential hallucination)
        incorrect_prompt = item["prompt"] + " " + item["incorrect"]
        incorrect_metrics = analyze_prompt_trajectory(
            model, tokenizer, incorrect_prompt, "incorrect", bottleneck_layers
        )
        all_metrics.append(incorrect_metrics)

        print(f"\nPrompt: \"{item['prompt']}\"")
        print(f"  Correct ({item['correct']}):")
        print(f"    Curvature: {correct_metrics.trajectory_curvature:.4f}")
        print(f"    Bottleneck values: {correct_metrics.bottleneck_values}")
        print(f"  Incorrect ({item['incorrect']}):")
        print(f"    Curvature: {incorrect_metrics.trajectory_curvature:.4f}")
        print(f"    Bottleneck values: {incorrect_metrics.bottleneck_values}")

    # Analyze nonsense prompts
    print(f"\n{'='*80}")
    print("NONSENSE PROMPTS")
    print("="*80)

    for prompt in NONSENSE_PROMPTS:
        metrics = analyze_prompt_trajectory(
            model, tokenizer, prompt, "nonsense", bottleneck_layers
        )
        all_metrics.append(metrics)

        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Curvature: {metrics.trajectory_curvature:.4f}")
        print(f"  Bottleneck values: {metrics.bottleneck_values}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("TRAJECTORY STATISTICS BY TYPE")
    print("="*80)

    for prompt_type in ["correct", "incorrect", "nonsense"]:
        type_metrics = [m for m in all_metrics if m.prompt_type == prompt_type]

        if not type_metrics:
            continue

        avg_curvature = np.mean([m.trajectory_curvature for m in type_metrics])
        std_curvature = np.std([m.trajectory_curvature for m in type_metrics])

        # Average bottleneck values
        for bl in bottleneck_layers:
            bl_values = [m.bottleneck_values.get(bl, 0) for m in type_metrics]
            avg_bl = np.mean(bl_values)
            std_bl = np.std(bl_values)
            print(f"\n{prompt_type.upper()}:")
            print(f"  Curvature: {avg_curvature:.4f} ± {std_curvature:.4f}")
            print(f"  Layer {bl} bottleneck: {avg_bl:.4f} ± {std_bl:.4f}")

    # Statistical comparison
    print(f"\n{'='*80}")
    print("HALLUCINATION DETECTION ANALYSIS")
    print("="*80)

    correct_curvatures = [m.trajectory_curvature for m in all_metrics if m.prompt_type == "correct"]
    incorrect_curvatures = [m.trajectory_curvature for m in all_metrics if m.prompt_type == "incorrect"]
    nonsense_curvatures = [m.trajectory_curvature for m in all_metrics if m.prompt_type == "nonsense"]

    print(f"\nMean curvature:")
    print(f"  Correct:   {np.mean(correct_curvatures):.4f}")
    print(f"  Incorrect: {np.mean(incorrect_curvatures):.4f}")
    print(f"  Nonsense:  {np.mean(nonsense_curvatures):.4f}")

    # Compare bottleneck values
    for bl in bottleneck_layers:
        correct_bl = [m.bottleneck_values.get(bl, 0) for m in all_metrics if m.prompt_type == "correct"]
        incorrect_bl = [m.bottleneck_values.get(bl, 0) for m in all_metrics if m.prompt_type == "incorrect"]
        nonsense_bl = [m.bottleneck_values.get(bl, 0) for m in all_metrics if m.prompt_type == "nonsense"]

        print(f"\nLayer {bl} bottleneck magnitude:")
        print(f"  Correct:   {np.mean(correct_bl):.4f}")
        print(f"  Incorrect: {np.mean(incorrect_bl):.4f}")
        print(f"  Nonsense:  {np.mean(nonsense_bl):.4f}")

    # The insight
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)

    # Check if there's a significant difference
    curv_diff = abs(np.mean(correct_curvatures) - np.mean(incorrect_curvatures))
    curv_nonsense_diff = abs(np.mean(correct_curvatures) - np.mean(nonsense_curvatures))

    if curv_nonsense_diff > 0.1:
        print(f"""
NONSENSE prompts have DIFFERENT trajectories!

Curvature difference (correct vs nonsense): {curv_nonsense_diff:.4f}

This suggests:
- Normal language follows a "smooth" path through the helix
- Nonsense/hallucinations may have more "jerky" trajectories
- We could potentially DETECT anomalies by monitoring trajectory curvature
""")

    if curv_diff > 0.05:
        print(f"""
INCORRECT completions have DIFFERENT trajectories than correct ones!

Curvature difference: {curv_diff:.4f}

This is remarkable: the model's internal geometry DIFFERS for wrong answers.
The "truth" may be encoded in the trajectory shape itself.
""")
    else:
        print(f"""
Trajectories are SIMILAR for correct and incorrect completions.

Curvature difference: {curv_diff:.4f}

This suggests:
- The model processes wrong facts similarly to right facts
- Hallucination detection may require different metrics
- The content (right/wrong) may be encoded in VALUES, not SHAPE
""")


if __name__ == "__main__":
    main()
