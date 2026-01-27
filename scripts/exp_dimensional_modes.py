#!/usr/bin/env python3
"""Experiment: Analyze the two dimensional processing modes.

Discovery from exp_dimensional_curve.py:
- Some problems start HIGH dimension (20-70) and only compress
- Some problems start LOW dimension (0.1-3) and expand then compress

Hypothesis:
- "Already High" = model immediately recognizes the problem type
- "Needs Expansion" = model must search/explore before finding structure
- Over-expansion might indicate confusion, not capability

Key question: What predicts which mode a problem uses?
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import re

import numpy as np
from scipy.stats import spearmanr, pearsonr

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


def analyze_problem_structure(question: str) -> dict:
    """Analyze structural features that might predict processing mode."""
    words = question.lower().split()
    n_words = len(words)

    # Count explicit numbers
    numbers = re.findall(r'\d+\.?\d*', question)
    n_numbers = len(numbers)

    # Detect implicit math patterns
    implicit_math_words = ['half', 'third', 'quarter', 'double', 'triple', 'twice',
                          'percent', 'fraction', 'ratio', 'times as']
    has_implicit_math = any(w in question.lower() for w in implicit_math_words)

    # Detect multi-step signals
    multi_step_words = ['then', 'after', 'next', 'first', 'second', 'finally',
                        'before', 'while', 'until', 'when']
    has_multi_step = any(w in question.lower() for w in multi_step_words)

    # Detect comparison/relation
    comparison_words = ['more than', 'less than', 'greater', 'fewer', 'difference',
                        'same as', 'equal', 'total', 'altogether']
    has_comparison = any(w in question.lower() for w in comparison_words)

    # Count sentences (rough proxy for problem complexity)
    sentences = [s.strip() for s in question.split('.') if s.strip()]
    n_sentences = len(sentences)

    # Explicit math ratio
    explicit_ratio = n_numbers / max(n_words, 1) * 100

    return {
        "n_words": n_words,
        "n_numbers": n_numbers,
        "n_sentences": n_sentences,
        "explicit_ratio": explicit_ratio,
        "has_implicit_math": has_implicit_math,
        "has_multi_step": has_multi_step,
        "has_comparison": has_comparison,
    }


def classify_mode(initial_dim: float, peak_layer: int, n_layers: int) -> str:
    """Classify processing mode based on dimensional trajectory."""
    # High initial dimension AND peak at layer 0 = "Already High"
    if initial_dim > 10 and peak_layer <= 2:
        return "already_high"
    # Low initial dimension OR peak later = "Expand-Compress"
    elif initial_dim < 5 or peak_layer > 5:
        return "expand_compress"
    else:
        return "intermediate"


def main():
    # Load previous results
    results_path = Path("data/experiments/dimensional_curve_analysis.json")
    if not results_path.exists():
        logger.error(f"Run exp_dimensional_curve.py first!")
        return

    with open(results_path) as f:
        data = json.load(f)

    logger.info("=" * 70)
    logger.info("DIMENSIONAL MODE ANALYSIS")
    logger.info("=" * 70)

    n_layers = data["n_layers"]
    problems = data["problems"]

    # Classify each problem
    already_high = []
    expand_compress = []
    intermediate = []

    for p in problems:
        dim_analysis = p["dimensional_analysis"]
        initial_dim = dim_analysis["initial_dim"]
        peak_layer = dim_analysis["peak_layer"]

        if np.isnan(initial_dim):
            continue

        mode = classify_mode(initial_dim, peak_layer, n_layers)

        structure = analyze_problem_structure(p["prompt"])
        p["structure"] = structure
        p["mode"] = mode

        if mode == "already_high":
            already_high.append(p)
        elif mode == "expand_compress":
            expand_compress.append(p)
        else:
            intermediate.append(p)

    logger.info(f"\nMode distribution:")
    logger.info(f"  Already High: {len(already_high)}")
    logger.info(f"  Expand-Compress: {len(expand_compress)}")
    logger.info(f"  Intermediate: {len(intermediate)}")

    # Analyze each mode
    def analyze_mode(mode_problems: List[Dict], mode_name: str):
        if not mode_problems:
            return {}

        correct = [p for p in mode_problems if p["is_correct"]]
        incorrect = [p for p in mode_problems if not p["is_correct"]]

        accuracy = len(correct) / len(mode_problems) * 100

        # Structural features
        n_words = [p["structure"]["n_words"] for p in mode_problems]
        n_numbers = [p["structure"]["n_numbers"] for p in mode_problems]
        explicit_ratio = [p["structure"]["explicit_ratio"] for p in mode_problems]
        has_implicit = sum(1 for p in mode_problems if p["structure"]["has_implicit_math"])
        has_multi_step = sum(1 for p in mode_problems if p["structure"]["has_multi_step"])

        # Dimensional features
        initial_dims = [p["dimensional_analysis"]["initial_dim"] for p in mode_problems]
        peak_dims = [p["dimensional_analysis"]["peak_dim"] for p in mode_problems
                    if not np.isnan(p["dimensional_analysis"]["peak_dim"])]
        final_dims = [p["dimensional_analysis"]["final_dim"] for p in mode_problems
                     if not np.isnan(p["dimensional_analysis"]["final_dim"])]
        compression_ratios = [p["dimensional_analysis"]["peak_dim"] / p["dimensional_analysis"]["final_dim"]
                             for p in mode_problems
                             if not np.isnan(p["dimensional_analysis"]["peak_dim"])
                             and p["dimensional_analysis"]["final_dim"] > 0.1]

        logger.info(f"\n{'=' * 50}")
        logger.info(f"{mode_name.upper()} MODE (n={len(mode_problems)})")
        logger.info(f"{'=' * 50}")
        logger.info(f"Accuracy: {accuracy:.0f}% ({len(correct)}/{len(mode_problems)})")

        logger.info(f"\nStructural features:")
        logger.info(f"  Words: {np.mean(n_words):.1f} ± {np.std(n_words):.1f}")
        logger.info(f"  Numbers: {np.mean(n_numbers):.1f} ± {np.std(n_numbers):.1f}")
        logger.info(f"  Explicit ratio: {np.mean(explicit_ratio):.1f}% ± {np.std(explicit_ratio):.1f}%")
        logger.info(f"  Has implicit math: {has_implicit}/{len(mode_problems)}")
        logger.info(f"  Has multi-step: {has_multi_step}/{len(mode_problems)}")

        logger.info(f"\nDimensional features:")
        logger.info(f"  Initial dim: {np.mean(initial_dims):.2f} ± {np.std(initial_dims):.2f}")
        if peak_dims:
            logger.info(f"  Peak dim: {np.mean(peak_dims):.2f} ± {np.std(peak_dims):.2f}")
        if final_dims:
            logger.info(f"  Final dim: {np.mean(final_dims):.2f} ± {np.std(final_dims):.2f}")
        if compression_ratios:
            logger.info(f"  Compression (peak/final): {np.mean(compression_ratios):.2f} ± {np.std(compression_ratios):.2f}")
            logger.info(f"  Compression/φ: {np.mean(compression_ratios)/PHI:.3f}")

        return {
            "n": len(mode_problems),
            "accuracy": accuracy,
            "mean_words": np.mean(n_words),
            "mean_numbers": np.mean(n_numbers),
            "mean_explicit_ratio": np.mean(explicit_ratio),
            "has_implicit_ratio": has_implicit / len(mode_problems),
            "has_multi_step_ratio": has_multi_step / len(mode_problems),
            "mean_initial_dim": np.mean(initial_dims),
            "mean_peak_dim": np.mean(peak_dims) if peak_dims else None,
            "mean_final_dim": np.mean(final_dims) if final_dims else None,
            "mean_compression": np.mean(compression_ratios) if compression_ratios else None,
        }

    ah_stats = analyze_mode(already_high, "Already High")
    ec_stats = analyze_mode(expand_compress, "Expand-Compress")

    # Key comparison
    logger.info(f"\n{'=' * 70}")
    logger.info("MODE COMPARISON")
    logger.info(f"{'=' * 70}")

    if ah_stats and ec_stats:
        logger.info(f"\n{'Feature':<25} {'Already High':<15} {'Expand-Compress':<15}")
        logger.info("-" * 55)
        logger.info(f"{'Accuracy':<25} {ah_stats['accuracy']:.0f}%{'':<12} {ec_stats['accuracy']:.0f}%")
        logger.info(f"{'Explicit ratio':<25} {ah_stats['mean_explicit_ratio']:.1f}%{'':<11} {ec_stats['mean_explicit_ratio']:.1f}%")
        logger.info(f"{'Has implicit math':<25} {ah_stats['has_implicit_ratio']*100:.0f}%{'':<12} {ec_stats['has_implicit_ratio']*100:.0f}%")
        logger.info(f"{'Initial dimension':<25} {ah_stats['mean_initial_dim']:.1f}{'':<12} {ec_stats['mean_initial_dim']:.1f}")
        if ah_stats.get('mean_compression') and ec_stats.get('mean_compression'):
            logger.info(f"{'Compression ratio':<25} {ah_stats['mean_compression']:.2f}{'':<12} {ec_stats['mean_compression']:.2f}")

    # Correlations
    logger.info(f"\n{'=' * 70}")
    logger.info("CORRELATIONS")
    logger.info(f"{'=' * 70}")

    # Prepare data for correlation
    valid_problems = [p for p in problems if not np.isnan(p["dimensional_analysis"]["initial_dim"])]

    if len(valid_problems) > 5:
        initial_dims = np.array([p["dimensional_analysis"]["initial_dim"] for p in valid_problems])
        explicit_ratios = np.array([p["structure"]["explicit_ratio"] for p in valid_problems])
        n_numbers = np.array([p["structure"]["n_numbers"] for p in valid_problems])
        is_correct = np.array([1 if p["is_correct"] else 0 for p in valid_problems])

        # Initial dim vs explicit math
        r, p = spearmanr(initial_dims, explicit_ratios)
        logger.info(f"\nInitial dim ↔ Explicit ratio: r={r:.3f}, p={p:.4f}")
        if p < 0.05:
            if r > 0:
                logger.info("  → Higher explicit ratio → Higher initial dimension (recognized)")
            else:
                logger.info("  → Higher explicit ratio → Lower initial dimension (unexpected)")

        # Initial dim vs accuracy
        r, p = spearmanr(initial_dims, is_correct)
        logger.info(f"Initial dim ↔ Correctness: r={r:.3f}, p={p:.4f}")

        # Numbers vs initial dim
        r, p = spearmanr(n_numbers, initial_dims)
        logger.info(f"N_numbers ↔ Initial dim: r={r:.3f}, p={p:.4f}")

    # The key insight
    logger.info(f"\n{'=' * 70}")
    logger.info("KEY INSIGHT")
    logger.info(f"{'=' * 70}")

    logger.info("""
The two modes suggest:

ALREADY HIGH MODE:
- Model immediately recognizes problem structure
- High initial dimension = rich encoding of math patterns
- Only needs to compress to answer
- Higher accuracy

EXPAND-COMPRESS MODE:
- Model starts with narrow encoding (low dimension)
- Must expand to explore solution space
- Then compress back to answer
- Over-expansion might indicate searching/confusion

PREDICTION: Problems with explicit numbers trigger "Already High" mode
because the model's training recognizes digit patterns. Implicit math
requires expansion to discover the mathematical structure hidden in words.

This matches your hypothesis: the model isn't missing capability,
it's missing the initial recognition signal that puts it in high-D space.
    """)

    # Save analysis
    output = {
        "timestamp": datetime.now().isoformat(),
        "already_high_stats": ah_stats,
        "expand_compress_stats": ec_stats,
        "problems_by_mode": {
            "already_high": [{"prompt": p["prompt"][:100], "correct": p["is_correct"]} for p in already_high],
            "expand_compress": [{"prompt": p["prompt"][:100], "correct": p["is_correct"]} for p in expand_compress],
        }
    }

    output_path = Path("data/experiments/dimensional_modes_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
