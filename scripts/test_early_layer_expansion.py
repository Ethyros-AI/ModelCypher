#!/usr/bin/env python3
"""Test if the early-layer adapter improves expansion dynamics.

Key question: Does teaching layers 0-10 to recognize implicit math
actually increase the expansion rate?
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_spectral_entropy(activations: np.ndarray, sqrt_eps: float) -> float:
    if len(activations) < 2:
        return 0.0
    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)
    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0
    p = S_valid ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def get_layer_activations(model, tokenizer, prompts: List[str], n_layers: int) -> Dict[int, List[np.ndarray]]:
    import mlx.core as mx
    layer_activations = {i: [] for i in range(n_layers)}
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = model.model.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            layer_activations[layer_idx].append(
                np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
            )
    return layer_activations


def compute_trajectory(layer_activations, n_layers, sqrt_eps):
    trajectory = []
    for layer_idx in range(n_layers):
        acts = np.vstack(layer_activations[layer_idx])
        entropy = compute_spectral_entropy(acts, sqrt_eps)
        trajectory.append(entropy)
    return trajectory


def analyze_trajectory(trajectory):
    n_layers = len(trajectory)
    peak_idx = np.argmax(trajectory)
    peak = trajectory[peak_idx]
    initial = trajectory[0]
    final = trajectory[-1]
    expansion = (peak - initial) / (peak_idx + 1) if peak_idx > 0 else 0
    compression_layers = n_layers - peak_idx - 1
    compression = (peak - final) / max(compression_layers, 1)
    ratio = compression / expansion if expansion > 1e-10 else float('inf')
    return {
        "initial": initial, "peak": peak, "peak_layer": peak_idx, "final": final,
        "expansion_rate": expansion, "compression_rate": compression,
        "ratio": ratio, "ratio_vs_phi": ratio / PHI if ratio != float('inf') else float('inf'),
    }


# The failing problems (implicit math)
FAILING_PROMPTS = [
    "Question: Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?\n\nAnswer:",
    "Question: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs $37, how much do the boots cost?\n\nAnswer:",
    "Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?\n\nAnswer:",
    "Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day?\n\nAnswer:",
    "Question: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?\n\nAnswer:",
]


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("EXPANSION DYNAMICS TEST")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    results = {"timestamp": datetime.now().isoformat(), "phi": PHI}

    # Test 1: Base model (no adapter)
    logger.info("\n1. BASE MODEL (no adapter)")
    model, tokenizer = load(model_path)
    n_layers = len(model.model.layers)
    acts = get_layer_activations(model, tokenizer, FAILING_PROMPTS, n_layers)
    traj = compute_trajectory(acts, n_layers, sqrt_eps)
    base_analysis = analyze_trajectory(traj)
    results["base"] = base_analysis
    logger.info(f"   Expansion: {base_analysis['expansion_rate']:.4f}, Ratio: {base_analysis['ratio']:.4f}, Ratio/φ: {base_analysis['ratio_vs_phi']:.4f}")
    del model

    # Test 2: Original mastery adapter only (layers 0-15)
    logger.info("\n2. MASTERY ADAPTER (layers 0-15)")
    model, tokenizer = load(model_path, adapter_path="data/adapters/qwen3_final_mastery_lora")
    acts = get_layer_activations(model, tokenizer, FAILING_PROMPTS, n_layers)
    traj = compute_trajectory(acts, n_layers, sqrt_eps)
    mastery_analysis = analyze_trajectory(traj)
    results["mastery"] = mastery_analysis
    logger.info(f"   Expansion: {mastery_analysis['expansion_rate']:.4f}, Ratio: {mastery_analysis['ratio']:.4f}, Ratio/φ: {mastery_analysis['ratio_vs_phi']:.4f}")
    del model

    # Test 3: Early-layer adapter only (layers 0-10)
    logger.info("\n3. EARLY-LAYER ADAPTER (layers 0-10)")
    model, tokenizer = load(model_path, adapter_path="data/adapters/early_layer_expansion_lora")
    acts = get_layer_activations(model, tokenizer, FAILING_PROMPTS, n_layers)
    traj = compute_trajectory(acts, n_layers, sqrt_eps)
    early_analysis = analyze_trajectory(traj)
    results["early_layer"] = early_analysis
    logger.info(f"   Expansion: {early_analysis['expansion_rate']:.4f}, Ratio: {early_analysis['ratio']:.4f}, Ratio/φ: {early_analysis['ratio_vs_phi']:.4f}")
    del model

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY: EXPANSION DYNAMICS ON FAILING PROBLEMS")
    logger.info(f"{'=' * 70}")
    logger.info(f"\n{'Configuration':<25} {'Expansion':<12} {'Ratio':<12} {'Ratio/φ':<12}")
    logger.info("-" * 60)
    logger.info(f"{'Base model':<25} {base_analysis['expansion_rate']:<12.4f} {base_analysis['ratio']:<12.4f} {base_analysis['ratio_vs_phi']:<12.4f}")
    logger.info(f"{'Mastery adapter':<25} {mastery_analysis['expansion_rate']:<12.4f} {mastery_analysis['ratio']:<12.4f} {mastery_analysis['ratio_vs_phi']:<12.4f}")
    logger.info(f"{'Early-layer adapter':<25} {early_analysis['expansion_rate']:<12.4f} {early_analysis['ratio']:<12.4f} {early_analysis['ratio_vs_phi']:<12.4f}")

    # Improvement check
    expansion_improved = early_analysis['expansion_rate'] > base_analysis['expansion_rate']
    ratio_improved = early_analysis['ratio_vs_phi'] < base_analysis['ratio_vs_phi'] or (
        early_analysis['ratio'] != float('inf') and base_analysis['ratio'] == float('inf')
    )

    logger.info(f"\nExpansion improved: {expansion_improved}")
    logger.info(f"Ratio improved toward φ: {ratio_improved}")

    if expansion_improved:
        improvement = (early_analysis['expansion_rate'] - base_analysis['expansion_rate']) / base_analysis['expansion_rate'] * 100
        logger.info(f"Expansion increase: {improvement:.1f}%")

    # Save results
    output_path = Path("data/experiments/expansion_dynamics_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
