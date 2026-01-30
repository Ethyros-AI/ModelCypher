#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Measure comp/phi distribution across diverse prompt categories.

This script gathers empirical data about comp/phi values for different task types.
Use this BEFORE training with phi-loss to understand what values naturally emerge.

Research questions this script helps answer:
1. What is the natural comp/phi distribution for different task types?
2. Does comp/phi correlate with task difficulty or type?
3. Is there a single attractor or multiple basins?

Usage:
    python scripts/measure_phi_distribution.py --model /path/to/model

    # Save results to JSON
    python scripts/measure_phi_distribution.py --model /path/to/model --output results.json

    # Quick run (fewer prompts per category)
    python scripts/measure_phi_distribution.py --model /path/to/model --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Diverse prompt categories to test comp/phi across different processing modes
PROMPT_CATEGORIES = {
    "simple_facts": [
        "What is the capital of France?",
        "What is 2 + 2?",
        "How many days are in a week?",
        "What color is the sky?",
        "How many legs does a dog have?",
    ],
    "crt_reasoning": [
        # Cognitive Reflection Test - designed to trigger intuitive traps
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
    ],
    "math_simple": [
        "What is 7 times 8?",
        "What is 15 plus 27?",
        "What is 100 divided by 4?",
        "What is 12 minus 5?",
        "What is half of 50?",
    ],
    "math_reasoning": [
        "If Tom has 3 times as many apples as Jane, and Jane has 5 apples, how many does Tom have?",
        "A train travels at 60 mph for 2 hours. How far does it go?",
        "If I have 3 red balls and 2 blue balls in a bag, what is the probability of drawing a red ball?",
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        "How many times can you subtract 5 from 25?",
    ],
    "logic_simple": [
        "All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
        "If it rains, the ground gets wet. It rained today. Is the ground wet?",
        "The sun rises in the east. Where does the sun rise?",
    ],
    "logic_complex": [
        "Some fruits are red. Apples are fruits. Are all apples red?",
        "If A implies B, and B implies C, does A imply C?",
        "All cats are animals. Some animals are pets. Are all cats pets?",
    ],
    "creative": [
        "Write the first line of a story about a dragon.",
        "What rhymes with 'moon'?",
        "Describe a sunset in one sentence.",
        "Name a fictional character who is brave.",
    ],
    "code": [
        "Write a function to add two numbers in Python.",
        "How do you print 'Hello World' in JavaScript?",
        "What does the keyword 'return' do in a function?",
        "How do you create a list in Python?",
    ],
    "chain_of_thought": [
        # Prompts that encourage step-by-step reasoning
        "Let's think step by step. A bat and ball cost $1.10. The bat costs $1 more than the ball. What does the ball cost?",
        "Think carefully. If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        "Break this down: In a lake, lily pads double daily. They cover the lake on day 48. When do they cover half?",
    ],
}


@dataclass
class PhiMeasurement:
    """Single comp/phi measurement for a prompt."""
    prompt: str
    category: str
    comp_phi: float
    compression_ratio: float
    peak_layer: int
    total_layers: int
    initial_norm: float
    peak_norm: float
    final_norm: float


@dataclass
class CategoryStats:
    """Statistics for a prompt category."""
    category: str
    n_samples: int
    mean_comp_phi: float
    std_comp_phi: float
    min_comp_phi: float
    max_comp_phi: float
    median_comp_phi: float


def compute_comp_phi(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> PhiMeasurement | None:
    """Compute comp/phi for a single prompt."""
    import math
    import mlx.core as mx

    PHI = 1.618033988749895

    try:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get model's layer structure
        base_model = getattr(model, "model", model)

        # Forward through embedding
        hidden = base_model.embed_tokens(input_ids)
        mx.eval(hidden)
        initial_norm = float(mx.sqrt(mx.sum(hidden * hidden)))

        norms = [initial_norm]
        peak_norm = initial_norm
        peak_layer = 0

        # Forward through each layer
        for i, layer in enumerate(base_model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            norm = float(mx.sqrt(mx.sum(hidden * hidden)))
            norms.append(norm)
            if norm > peak_norm:
                peak_norm = norm
                peak_layer = i + 1

        final_norm = norms[-1]
        total_layers = len(base_model.layers)

        # Compute comp/phi with dtype-derived epsilon
        eps = math.sqrt(float(mx.finfo(mx.float32).eps))
        compression_ratio = peak_norm / final_norm if final_norm > eps else 1.0
        comp_phi = compression_ratio / PHI

        return PhiMeasurement(
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            category="",  # Set by caller
            comp_phi=comp_phi,
            compression_ratio=compression_ratio,
            peak_layer=peak_layer,
            total_layers=total_layers,
            initial_norm=initial_norm,
            peak_norm=peak_norm,
            final_norm=final_norm,
        )

    except Exception as e:
        logger.warning(f"Failed to compute comp/phi: {e}")
        return None


def measure_phi_distribution(
    model_path: str,
    quick: bool = False,
) -> dict:
    """Measure comp/phi distribution across all prompt categories.

    Args:
        model_path: Path to the model.
        quick: If True, use only first 2 prompts per category.

    Returns:
        Dict with measurements, category stats, and overall stats.
    """
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("COMP/PHI DISTRIBUTION MEASUREMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load(model_path)

    all_measurements: list[PhiMeasurement] = []
    category_stats: list[CategoryStats] = []

    for category, prompts in PROMPT_CATEGORIES.items():
        if quick:
            prompts = prompts[:2]

        logger.info(f"\n{category.upper()} ({len(prompts)} prompts)")
        logger.info("-" * 40)

        category_measurements = []

        for prompt in prompts:
            result = compute_comp_phi(model, tokenizer, prompt)
            if result is not None:
                result.category = category
                category_measurements.append(result)
                all_measurements.append(result)
                logger.info(
                    f"  comp/phi={result.comp_phi:.3f}, "
                    f"peak={result.peak_layer}/{result.total_layers}"
                )

        if category_measurements:
            phi_values = [m.comp_phi for m in category_measurements]
            stats = CategoryStats(
                category=category,
                n_samples=len(phi_values),
                mean_comp_phi=statistics.mean(phi_values),
                std_comp_phi=statistics.stdev(phi_values) if len(phi_values) > 1 else 0.0,
                min_comp_phi=min(phi_values),
                max_comp_phi=max(phi_values),
                median_comp_phi=statistics.median(phi_values),
            )
            category_stats.append(stats)

    # Overall statistics
    all_phi_values = [m.comp_phi for m in all_measurements]

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)

    logger.info("\nPer-category comp/phi:")
    logger.info(f"{'Category':<20} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    logger.info("-" * 60)
    for stats in category_stats:
        logger.info(
            f"{stats.category:<20} {stats.n_samples:>4} "
            f"{stats.mean_comp_phi:>8.3f} {stats.std_comp_phi:>8.3f} "
            f"{stats.min_comp_phi:>8.3f} {stats.max_comp_phi:>8.3f}"
        )

    if all_phi_values:
        overall_mean = statistics.mean(all_phi_values)
        overall_std = statistics.stdev(all_phi_values) if len(all_phi_values) > 1 else 0.0
        overall_min = min(all_phi_values)
        overall_max = max(all_phi_values)
        overall_median = statistics.median(all_phi_values)

        logger.info("-" * 60)
        logger.info(
            f"{'OVERALL':<20} {len(all_phi_values):>4} "
            f"{overall_mean:>8.3f} {overall_std:>8.3f} "
            f"{overall_min:>8.3f} {overall_max:>8.3f}"
        )

        logger.info("\n" + "=" * 70)
        logger.info("INTERPRETATION")
        logger.info("=" * 70)
        logger.info(f"Overall comp/phi range: [{overall_min:.3f}, {overall_max:.3f}]")
        logger.info(f"Overall comp/phi mean: {overall_mean:.3f} +/- {overall_std:.3f}")
        logger.info(f"Overall comp/phi median: {overall_median:.3f}")

        # Check if there's variation across categories
        category_means = [s.mean_comp_phi for s in category_stats]
        if len(category_means) > 1:
            cat_variance = statistics.variance(category_means)
            logger.info(f"\nBetween-category variance: {cat_variance:.4f}")
            if cat_variance > 0.01:
                logger.info(
                    "  -> Different categories show different comp/phi profiles."
                    "\n  -> A single target value (like 1.0) may not be optimal for all tasks."
                )
            else:
                logger.info(
                    "  -> Categories show similar comp/phi profiles."
                    "\n  -> A single target value may be reasonable."
                )

        # Distance from 1.0
        dist_from_one = abs(overall_mean - 1.0)
        logger.info(f"\nDistance from comp/phi = 1.0: {dist_from_one:.3f}")
        if dist_from_one > 0.3:
            logger.info(
                "  -> Model's natural geometry is far from 1.0."
                "\n  -> Consider whether training toward 1.0 is appropriate."
            )

    return {
        "model_path": model_path,
        "n_measurements": len(all_measurements),
        "overall_stats": {
            "mean": overall_mean if all_phi_values else None,
            "std": overall_std if all_phi_values else None,
            "min": overall_min if all_phi_values else None,
            "max": overall_max if all_phi_values else None,
            "median": overall_median if all_phi_values else None,
        },
        "category_stats": [asdict(s) for s in category_stats],
        "measurements": [asdict(m) for m in all_measurements],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure comp/phi distribution across diverse prompt categories"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer prompts per category",
    )

    args = parser.parse_args()

    result = measure_phi_distribution(
        model_path=args.model,
        quick=args.quick,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
