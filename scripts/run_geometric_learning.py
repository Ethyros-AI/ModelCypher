#!/usr/bin/env python3
"""Run Geometric Learning - Direct weight space optimization for SVD alignment.

This is different from thinking-based approaches:
- No token generation
- Direct manipulation of weight geometry
- Constraint: preserve model predictions
- Goal: align SVD ratios with fundamental constants

Usage:
    poetry run python scripts/run_geometric_learning.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 20 \
        --step-size 0.001 \
        --output data/geometric/run.json
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


# Quality tests
QUALITY_TESTS = [
    ("What is 2 + 2?", "4"),
    ("Name the capital of France.", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run geometric learning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.001,
        help="Step size for weight updates",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_consistency.geometric_learning import (
        GeometricLearning,
    )

    backend = initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Create learner
    learner = GeometricLearning(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        step_size=args.step_size,
        quality_threshold=0.8,  # Allow more degradation for experimentation
    )

    # Run
    result = learner.run(
        test_prompts=QUALITY_TESTS,
        n_iterations=args.iterations,
    )

    # Save
    output_path = args.output or f"data/geometric/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "step_size": args.step_size,
        "n_iterations": result.n_iterations,
        "initial_matches": result.initial_matches,
        "final_matches": result.final_matches,
        "initial_quality": result.initial_quality,
        "final_quality": result.final_quality,
        "quality_preserved": result.quality_preserved,
        "trajectory": result.trajectory,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Summary
    if result.final_matches > result.initial_matches:
        logger.info("SUCCESS: Geometry improved!")
    else:
        logger.info("No improvement in geometry")


if __name__ == "__main__":
    main()
