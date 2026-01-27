#!/usr/bin/env python3
"""Compare entropy trajectories for correct vs incorrect answers.

Hypothesis: Correct answers follow the natural φ ratio between expansion
and compression. Incorrect answers have a distorted ratio.

This tells us WHERE the model fails geometrically.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

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
    """Compute spectral entropy from activations."""
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


def get_layer_activations(model, tokenizer, prompt: str, n_layers: int) -> List[np.ndarray]:
    """Get activations at every layer for a single prompt."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    activations = []

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        activations.append(np.array(hidden[0, -1, :].tolist(), dtype=np.float32))

    return activations


def compute_entropy_trajectory(all_activations: List[List[np.ndarray]], n_layers: int, sqrt_eps: float) -> List[float]:
    """Compute entropy at each layer from multiple samples' activations."""
    trajectory = []
    for layer_idx in range(n_layers):
        # Stack activations from all samples at this layer
        layer_acts = np.vstack([acts[layer_idx] for acts in all_activations])
        entropy = compute_spectral_entropy(layer_acts, sqrt_eps)
        trajectory.append(entropy)
    return trajectory


def analyze_trajectory(trajectory: List[float], n_layers: int) -> Dict:
    """Analyze expansion/compression from entropy trajectory."""
    peak_idx = np.argmax(trajectory)
    peak_entropy = trajectory[peak_idx]
    initial_entropy = trajectory[0]
    final_entropy = trajectory[-1]

    # Expansion: 0 → peak
    expansion_rate = (peak_entropy - initial_entropy) / (peak_idx + 1) if peak_idx > 0 else 0

    # Compression: peak → final
    compression_layers = n_layers - peak_idx - 1
    compression_rate = (peak_entropy - final_entropy) / max(compression_layers, 1)

    # The key ratio
    if expansion_rate > 1e-6:
        compression_expansion_ratio = compression_rate / expansion_rate
    else:
        compression_expansion_ratio = float('inf')

    return {
        "initial_entropy": initial_entropy,
        "peak_entropy": peak_entropy,
        "peak_layer": int(peak_idx),
        "final_entropy": final_entropy,
        "expansion_rate": expansion_rate,
        "compression_rate": compression_rate,
        "compression_expansion_ratio": compression_expansion_ratio,
        "ratio_vs_phi": compression_expansion_ratio / PHI if compression_expansion_ratio != float('inf') else float('inf'),
        "trajectory": trajectory,
    }


def evaluate_gsm8k_problem(model, tokenizer, question: str, expected: str) -> Tuple[bool, str]:
    """Evaluate a single GSM8K problem."""
    from mlx_lm import generate

    prompt = f"Question: {question}\n\nAnswer:"

    # Use mlx_lm generate for proper generation
    output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)
    output = output.strip().replace("<|im_end|>", "")

    # Extract answer
    if "####" in output:
        answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
        numbers = re.findall(r'-?\d+', answer_part)
        predicted = numbers[0] if numbers else ""
    else:
        numbers = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = numbers[-1] if numbers else ""

    return predicted == expected, predicted


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("ENTROPY TRAJECTORY: CORRECT vs INCORRECT")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    logger.info(f"Model has {n_layers} layers")

    # Load GSM8K test problems
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    logger.info(f"\nEvaluating {len(gsm_test.samples)} GSM8K problems...")

    correct_prompts = []
    incorrect_prompts = []
    correct_activations = []
    incorrect_activations = []

    for i, sample in enumerate(gsm_test.samples):
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        prompt = f"Question: {question}\n\nAnswer:"

        # Evaluate
        is_correct, predicted = evaluate_gsm8k_problem(model, tokenizer, question, expected)

        # Get activations
        acts = get_layer_activations(model, tokenizer, prompt, n_layers)

        if is_correct:
            correct_prompts.append(prompt)
            correct_activations.append(acts)
            logger.info(f"  [{i+1}] CORRECT: {predicted} == {expected}")
        else:
            incorrect_prompts.append(prompt)
            incorrect_activations.append(acts)
            logger.info(f"  [{i+1}] WRONG: {predicted} != {expected}")

    logger.info(f"\nCorrect: {len(correct_prompts)}, Incorrect: {len(incorrect_prompts)}")

    # Compute trajectories
    logger.info("\nComputing entropy trajectories...")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "n_correct": len(correct_prompts),
        "n_incorrect": len(incorrect_prompts),
        "phi": PHI,
    }

    if len(correct_activations) >= 2:
        correct_trajectory = compute_entropy_trajectory(correct_activations, n_layers, sqrt_eps)
        correct_analysis = analyze_trajectory(correct_trajectory, n_layers)
        results["correct"] = correct_analysis

        logger.info(f"\n=== CORRECT ANSWERS ({len(correct_prompts)}) ===")
        logger.info(f"  Initial entropy: {correct_analysis['initial_entropy']:.4f}")
        logger.info(f"  Peak entropy (layer {correct_analysis['peak_layer']}): {correct_analysis['peak_entropy']:.4f}")
        logger.info(f"  Final entropy: {correct_analysis['final_entropy']:.4f}")
        logger.info(f"  Expansion rate: {correct_analysis['expansion_rate']:.4f}")
        logger.info(f"  Compression rate: {correct_analysis['compression_rate']:.4f}")
        logger.info(f"  RATIO (compression/expansion): {correct_analysis['compression_expansion_ratio']:.4f}")
        logger.info(f"  Ratio / φ: {correct_analysis['ratio_vs_phi']:.4f}")
    else:
        logger.info("\nNot enough correct samples for trajectory analysis")

    if len(incorrect_activations) >= 2:
        incorrect_trajectory = compute_entropy_trajectory(incorrect_activations, n_layers, sqrt_eps)
        incorrect_analysis = analyze_trajectory(incorrect_trajectory, n_layers)
        results["incorrect"] = incorrect_analysis

        logger.info(f"\n=== INCORRECT ANSWERS ({len(incorrect_prompts)}) ===")
        logger.info(f"  Initial entropy: {incorrect_analysis['initial_entropy']:.4f}")
        logger.info(f"  Peak entropy (layer {incorrect_analysis['peak_layer']}): {incorrect_analysis['peak_entropy']:.4f}")
        logger.info(f"  Final entropy: {incorrect_analysis['final_entropy']:.4f}")
        logger.info(f"  Expansion rate: {incorrect_analysis['expansion_rate']:.4f}")
        logger.info(f"  Compression rate: {incorrect_analysis['compression_rate']:.4f}")
        logger.info(f"  RATIO (compression/expansion): {incorrect_analysis['compression_expansion_ratio']:.4f}")
        logger.info(f"  Ratio / φ: {incorrect_analysis['ratio_vs_phi']:.4f}")
    else:
        logger.info("\nNot enough incorrect samples for trajectory analysis")

    # Compare
    if "correct" in results and "incorrect" in results:
        logger.info(f"\n{'=' * 50}")
        logger.info("COMPARISON")
        logger.info(f"{'=' * 50}")

        c = results["correct"]
        i = results["incorrect"]

        logger.info(f"  Peak layer:    correct={c['peak_layer']}, incorrect={i['peak_layer']}")
        logger.info(f"  Peak entropy:  correct={c['peak_entropy']:.4f}, incorrect={i['peak_entropy']:.4f}")
        logger.info(f"  Final entropy: correct={c['final_entropy']:.4f}, incorrect={i['final_entropy']:.4f}")
        logger.info(f"  Expansion:     correct={c['expansion_rate']:.4f}, incorrect={i['expansion_rate']:.4f}")
        logger.info(f"  Compression:   correct={c['compression_rate']:.4f}, incorrect={i['compression_rate']:.4f}")
        logger.info(f"  RATIO:         correct={c['compression_expansion_ratio']:.4f}, incorrect={i['compression_expansion_ratio']:.4f}")
        logger.info(f"  Ratio/φ:       correct={c['ratio_vs_phi']:.4f}, incorrect={i['ratio_vs_phi']:.4f}")

        # Diagnosis
        correct_ratio_ok = 0.8 < c['ratio_vs_phi'] < 1.2  # Within 20% of φ
        incorrect_ratio_ok = 0.8 < i['ratio_vs_phi'] < 1.2

        results["diagnosis"] = {
            "correct_follows_phi": correct_ratio_ok,
            "incorrect_follows_phi": incorrect_ratio_ok,
            "ratio_difference": abs(c['compression_expansion_ratio'] - i['compression_expansion_ratio']),
            "peak_layer_difference": abs(c['peak_layer'] - i['peak_layer']),
            "compression_deficit": c['compression_rate'] - i['compression_rate'],
        }

        logger.info(f"\n  DIAGNOSIS:")
        logger.info(f"    Correct follows φ: {correct_ratio_ok}")
        logger.info(f"    Incorrect follows φ: {incorrect_ratio_ok}")
        logger.info(f"    Compression deficit: {results['diagnosis']['compression_deficit']:.4f}")

    # Save results
    output_path = Path("data/experiments/entropy_correct_vs_incorrect.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
