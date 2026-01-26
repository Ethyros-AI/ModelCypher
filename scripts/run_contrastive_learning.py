#!/usr/bin/env python3
"""Run Contrastive Geometric Learning.

The model teaches itself by contrasting:
- Coherent statements (2+2=4, Paris is in France)
- Incoherent statements (colorless green ideas, 2+2=5)

The learning signal is the geometric difference between these.

Usage:
    poetry run python scripts/run_contrastive_learning.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --iterations 20 \
        --output data/contrastive/run.json
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


QUALITY_TESTS = [
    ("What is 2 + 2?", "4"),
    ("Name the capital of France.", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_consistency.contrastive_geometric_learning import (
        ContrastiveGeometricLearning,
    )

    backend = initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    learner = ContrastiveGeometricLearning(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        step_size=args.step_size,
        quality_threshold=0.8,
    )

    result = learner.run(
        test_prompts=QUALITY_TESTS,
        n_iterations=args.iterations,
    )

    output_path = args.output or f"data/contrastive/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "step_size": args.step_size,
        "n_iterations": result.n_iterations,
        "initial_coherent": result.initial_coherent_matches,
        "final_coherent": result.final_coherent_matches,
        "initial_incoherent": result.initial_incoherent_matches,
        "final_incoherent": result.final_incoherent_matches,
        "initial_contrast": result.initial_contrast,
        "final_contrast": result.final_contrast,
        "initial_quality": result.initial_quality,
        "final_quality": result.final_quality,
        "trajectory": result.trajectory,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    if result.final_contrast > result.initial_contrast:
        logger.info("SUCCESS: Contrast improved!")
    else:
        logger.info("No improvement in contrast")


if __name__ == "__main__":
    main()
