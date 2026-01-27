#!/usr/bin/env python3
"""Cartography of failure: Map what triggers constrained encoding.

We know:
- Correct answers expand (0.021/layer) and follow φ ratio
- Incorrect answers barely expand (0.003/layer) - 7x weaker
- Initial entropy is already low for failures (1.32 vs 2.67)

Question: What about failing problems causes the model to encode them narrowly?

This script analyzes:
1. Individual entropy trajectories for each problem
2. Structural features of failing vs passing problems
3. Token-level analysis of initial representations
4. What specific patterns trigger constrained encoding
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
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


def compute_spectral_entropy(activation: np.ndarray, sqrt_eps: float) -> float:
    """Compute spectral entropy from a single activation vector."""
    # For a single vector, we look at the distribution of its components
    # Normalize to get a probability-like distribution
    abs_act = np.abs(activation)
    if abs_act.sum() < sqrt_eps:
        return 0.0

    p = abs_act / abs_act.sum()
    p = p[p > sqrt_eps]  # Remove near-zero components

    if len(p) < 2:
        return 0.0

    return float(-np.sum(p * np.log(p + 1e-10)))


def compute_activation_stats(activation: np.ndarray) -> Dict:
    """Compute various statistics about an activation vector."""
    return {
        "mean": float(np.mean(activation)),
        "std": float(np.std(activation)),
        "max": float(np.max(activation)),
        "min": float(np.min(activation)),
        "l2_norm": float(np.linalg.norm(activation)),
        "sparsity": float(np.mean(np.abs(activation) < 0.01)),  # Fraction near zero
        "entropy": compute_spectral_entropy(activation, np.sqrt(np.finfo(np.float32).eps)),
    }


def get_layer_activations(model, tokenizer, prompt: str, n_layers: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Get activations at every layer and embedding for a single prompt."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embedding
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    embedding = np.array(hidden[0, -1, :].tolist(), dtype=np.float32)

    activations = []
    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        activations.append(np.array(hidden[0, -1, :].tolist(), dtype=np.float32))

    return activations, embedding


def analyze_problem_structure(question: str) -> Dict:
    """Analyze structural features of a problem."""
    # Count numbers
    numbers = re.findall(r'\d+\.?\d*', question)

    # Count operations implied
    has_addition = any(w in question.lower() for w in ['add', 'plus', 'more', 'total', 'sum', 'together'])
    has_subtraction = any(w in question.lower() for w in ['subtract', 'minus', 'less', 'remain', 'left', 'lose'])
    has_multiplication = any(w in question.lower() for w in ['times', 'multiply', 'each', 'per', 'every'])
    has_division = any(w in question.lower() for w in ['divide', 'split', 'share', 'half', 'quarter'])

    # Count entities (capitalized words that aren't at sentence start)
    sentences = question.split('.')
    entities = []
    for sent in sentences:
        words = sent.strip().split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper():
                entities.append(word)

    # Complexity indicators
    n_sentences = len([s for s in sentences if s.strip()])
    n_words = len(question.split())
    has_conditional = any(w in question.lower() for w in ['if', 'when', 'after', 'before', 'while'])
    has_comparison = any(w in question.lower() for w in ['more than', 'less than', 'twice', 'half', 'ratio'])
    has_fraction = any(w in question.lower() for w in ['third', 'quarter', 'half', 'percent', '%', 'fraction'])

    return {
        "n_numbers": len(numbers),
        "numbers": [float(n) for n in numbers[:10]],  # First 10 numbers
        "n_words": n_words,
        "n_sentences": n_sentences,
        "n_entities": len(entities),
        "has_addition": has_addition,
        "has_subtraction": has_subtraction,
        "has_multiplication": has_multiplication,
        "has_division": has_division,
        "has_conditional": has_conditional,
        "has_comparison": has_comparison,
        "has_fraction": has_fraction,
        "operations_count": sum([has_addition, has_subtraction, has_multiplication, has_division]),
        "complexity_indicators": sum([has_conditional, has_comparison, has_fraction]),
    }


def evaluate_problem(model, tokenizer, question: str, expected: str) -> Tuple[bool, str]:
    """Evaluate a single problem."""
    from mlx_lm import generate

    prompt = f"Question: {question}\n\nAnswer:"
    output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)
    output = output.strip().replace("<|im_end|>", "")

    if "####" in output:
        answer_part = output.split("####")[-1].strip().replace(",", "").replace("$", "")
        numbers = re.findall(r'-?\d+', answer_part)
        predicted = numbers[0] if numbers else ""
    else:
        numbers = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = numbers[-1] if numbers else ""

    return predicted == expected, predicted


def compute_trajectory_metrics(activations: List[np.ndarray], sqrt_eps: float) -> Dict:
    """Compute entropy trajectory and derived metrics for a single problem."""
    trajectory = []
    for act in activations:
        # For single problem, use activation statistics as proxy for "entropy"
        # Higher variance = more information spread = higher effective entropy
        std = float(np.std(act))
        trajectory.append(std)

    peak_idx = np.argmax(trajectory)
    peak_val = trajectory[peak_idx]
    initial_val = trajectory[0]
    final_val = trajectory[-1]

    n_layers = len(trajectory)
    expansion_rate = (peak_val - initial_val) / (peak_idx + 1) if peak_idx > 0 else 0
    compression_layers = n_layers - peak_idx - 1
    compression_rate = (peak_val - final_val) / max(compression_layers, 1)

    if expansion_rate > 1e-10:
        ratio = compression_rate / expansion_rate
    else:
        ratio = float('inf')

    return {
        "trajectory": trajectory,
        "initial": initial_val,
        "peak": peak_val,
        "peak_layer": int(peak_idx),
        "final": final_val,
        "expansion_rate": expansion_rate,
        "compression_rate": compression_rate,
        "ratio": ratio,
        "ratio_vs_phi": ratio / PHI if ratio != float('inf') else float('inf'),
    }


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("FAILURE CARTOGRAPHY: Mapping Constrained Encoding")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    # Load GSM8K
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader
    loader = BenchmarkLoader()
    gsm_test = loader.load("gsm8k", split="test", limit=30)

    logger.info(f"\nAnalyzing {len(gsm_test.samples)} problems in detail...")

    correct_problems = []
    incorrect_problems = []

    for i, sample in enumerate(gsm_test.samples):
        question = sample.prompt.replace("Answer:", "").strip()
        expected = sample.answer

        prompt = f"Question: {question}\n\nAnswer:"

        # Evaluate
        is_correct, predicted = evaluate_problem(model, tokenizer, question, expected)

        # Get activations
        activations, embedding = get_layer_activations(model, tokenizer, prompt, n_layers)

        # Analyze structure
        structure = analyze_problem_structure(question)

        # Compute trajectory
        trajectory_metrics = compute_trajectory_metrics(activations, sqrt_eps)

        # Embedding stats
        embedding_stats = compute_activation_stats(embedding)

        problem_data = {
            "index": i,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            "structure": structure,
            "trajectory": trajectory_metrics,
            "embedding": embedding_stats,
        }

        if is_correct:
            correct_problems.append(problem_data)
            logger.info(f"  [{i+1}] CORRECT: {predicted} == {expected}")
        else:
            incorrect_problems.append(problem_data)
            logger.info(f"  [{i+1}] WRONG: {predicted} != {expected}")

    logger.info(f"\nCorrect: {len(correct_problems)}, Incorrect: {len(incorrect_problems)}")

    # Aggregate analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("STRUCTURAL ANALYSIS")
    logger.info(f"{'=' * 70}")

    def aggregate_structures(problems):
        return {
            "avg_numbers": np.mean([p["structure"]["n_numbers"] for p in problems]),
            "avg_words": np.mean([p["structure"]["n_words"] for p in problems]),
            "avg_sentences": np.mean([p["structure"]["n_sentences"] for p in problems]),
            "avg_operations": np.mean([p["structure"]["operations_count"] for p in problems]),
            "avg_complexity": np.mean([p["structure"]["complexity_indicators"] for p in problems]),
            "frac_conditional": np.mean([p["structure"]["has_conditional"] for p in problems]),
            "frac_comparison": np.mean([p["structure"]["has_comparison"] for p in problems]),
            "frac_fraction": np.mean([p["structure"]["has_fraction"] for p in problems]),
        }

    if correct_problems:
        correct_agg = aggregate_structures(correct_problems)
        logger.info(f"\nCORRECT problems (n={len(correct_problems)}):")
        for k, v in correct_agg.items():
            logger.info(f"  {k}: {v:.3f}")

    if incorrect_problems:
        incorrect_agg = aggregate_structures(incorrect_problems)
        logger.info(f"\nINCORRECT problems (n={len(incorrect_problems)}):")
        for k, v in incorrect_agg.items():
            logger.info(f"  {k}: {v:.3f}")

    # Trajectory analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("TRAJECTORY ANALYSIS (per-problem)")
    logger.info(f"{'=' * 70}")

    if correct_problems:
        logger.info(f"\nCORRECT trajectories:")
        correct_ratios = [p["trajectory"]["ratio"] for p in correct_problems if p["trajectory"]["ratio"] != float('inf')]
        correct_initials = [p["trajectory"]["initial"] for p in correct_problems]
        correct_expansions = [p["trajectory"]["expansion_rate"] for p in correct_problems]

        logger.info(f"  Avg initial std: {np.mean(correct_initials):.4f}")
        logger.info(f"  Avg expansion rate: {np.mean(correct_expansions):.4f}")
        logger.info(f"  Avg ratio: {np.mean(correct_ratios):.4f}")
        logger.info(f"  Avg ratio/φ: {np.mean(correct_ratios)/PHI:.4f}")

    if incorrect_problems:
        logger.info(f"\nINCORRECT trajectories:")
        incorrect_ratios = [p["trajectory"]["ratio"] for p in incorrect_problems if p["trajectory"]["ratio"] != float('inf')]
        incorrect_initials = [p["trajectory"]["initial"] for p in incorrect_problems]
        incorrect_expansions = [p["trajectory"]["expansion_rate"] for p in incorrect_problems]

        logger.info(f"  Avg initial std: {np.mean(incorrect_initials):.4f}")
        logger.info(f"  Avg expansion rate: {np.mean(incorrect_expansions):.4f}")
        if incorrect_ratios:
            logger.info(f"  Avg ratio: {np.mean(incorrect_ratios):.4f}")
            logger.info(f"  Avg ratio/φ: {np.mean(incorrect_ratios)/PHI:.4f}")

    # Detailed failure analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("INDIVIDUAL FAILURE ANALYSIS")
    logger.info(f"{'=' * 70}")

    for p in incorrect_problems:
        logger.info(f"\n--- Problem {p['index']+1} ---")
        logger.info(f"Question: {p['question']}")
        logger.info(f"Expected: {p['expected']}, Got: {p['predicted']}")
        logger.info(f"Structure:")
        logger.info(f"  Numbers: {p['structure']['n_numbers']}, Words: {p['structure']['n_words']}")
        logger.info(f"  Operations: {p['structure']['operations_count']}, Complexity: {p['structure']['complexity_indicators']}")
        logger.info(f"  Conditional: {p['structure']['has_conditional']}, Comparison: {p['structure']['has_comparison']}, Fraction: {p['structure']['has_fraction']}")
        logger.info(f"Trajectory:")
        logger.info(f"  Initial: {p['trajectory']['initial']:.4f}, Peak: {p['trajectory']['peak']:.4f}, Final: {p['trajectory']['final']:.4f}")
        logger.info(f"  Expansion: {p['trajectory']['expansion_rate']:.4f}, Compression: {p['trajectory']['compression_rate']:.4f}")
        logger.info(f"  Ratio: {p['trajectory']['ratio']:.4f}, Ratio/φ: {p['trajectory']['ratio_vs_phi']:.4f}")
        logger.info(f"Embedding entropy: {p['embedding']['entropy']:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "phi": PHI,
        "n_correct": len(correct_problems),
        "n_incorrect": len(incorrect_problems),
        "correct_problems": correct_problems,
        "incorrect_problems": incorrect_problems,
        "aggregated": {
            "correct": aggregate_structures(correct_problems) if correct_problems else {},
            "incorrect": aggregate_structures(incorrect_problems) if incorrect_problems else {},
        },
    }

    output_path = Path("data/experiments/failure_cartography.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
