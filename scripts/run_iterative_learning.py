#!/usr/bin/env python3
"""Run Iterative Geometric Learning - The Full Loop.

The model:
1. Thinks (self-questions to find coherence)
2. Locks gains (surgical SVD alignment)
3. Repeats (like moving grade to grade)

Usage:
    poetry run python scripts/run_iterative_learning.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 5 \
        --output data/iterative/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Topics for thinking loops
TOPICS = [
    "mathematics",
    "geography",
    "biology",
    "physics",
    "logic",
]

# Quality tests
TEST_PROMPTS = [
    ("What is 2 + 2?", "4"),
    ("Capital of France?", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--proximity", type=float, default=0.10)
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.core.use_cases.self_consistency.iterative_geometric_learning import (
        IterativeGeometricLearning,
    )

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    learner = IterativeGeometricLearning(
        model=model,
        tokenizer=tokenizer,
        proximity_threshold=args.proximity,
        quality_threshold=args.quality_threshold,
    )

    result = learner.run(
        topics=TOPICS,
        test_prompts=TEST_PROMPTS,
        n_iterations=args.iterations,
    )

    # Save results
    output_path = args.output or f"data/iterative/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_iterations": result.total_iterations,
        "initial_matches": result.initial_matches,
        "final_matches": result.final_matches,
        "initial_quality": result.initial_quality,
        "final_quality": result.final_quality,
        "initial_consistency": result.initial_consistency,
        "final_consistency": result.final_consistency,
        "history": [
            {
                "iteration": h.iteration,
                "thinking_iterations": h.thinking_iterations,
                "consistency_score": h.consistency_score,
                "matches_before": h.matches_before,
                "matches_after": h.matches_after,
                "targets_aligned": h.targets_aligned,
                "quality": h.quality,
            }
            for h in result.history
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Summary
    if result.final_matches > result.initial_matches:
        improvement = (result.final_matches - result.initial_matches) / result.initial_matches * 100
        logger.info(f"\nSUCCESS: Matches improved {result.initial_matches} → {result.final_matches} ({improvement:.1f}%)")
    else:
        logger.info(f"\nNo improvement in matches")


if __name__ == "__main__":
    main()
