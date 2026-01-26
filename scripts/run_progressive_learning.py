#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Run Progressive Learning - Iterative thinking with weight locking.

The full loop:
1. Measure geometric state
2. Think to improve coherence
3. Lock gains into weights
4. Verify quality preserved
5. Repeat

Like a student moving from kindergarten to 1st grade - each cycle builds
on the previous, locking gains before going deeper.

Usage:
    poetry run python scripts/run_progressive_learning.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --cycles 10 \
        --output data/progressive/run.json
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


# Probes for geometric measurement - diverse concepts
GEOMETRIC_PROBES = [
    "Mathematics is the language of the universe.",
    "Water flows downhill due to gravity.",
    "The sky appears blue because of light scattering.",
    "Two plus two equals four.",
    "Paris is the capital of France.",
    "Dogs are mammals that can be domesticated.",
    "Time moves forward, not backward.",
    "Consciousness is the awareness of awareness.",
    "Truth is that which corresponds to reality.",
    "Knowledge requires justified true belief.",
    "Self-reference creates paradoxes.",
    "Meaning emerges from context.",
]

# Test prompts for quality verification
QUALITY_TESTS = [
    ("What is 2 + 2?", "4"),
    ("Name the capital of France.", "paris"),
    ("Is water wet?", "yes"),
    ("What color is the sky?", "blue"),
    ("Are dogs mammals?", "yes"),
]


def run_progressive_learning(
    model_path: str,
    n_cycles: int,
    output_path: str,
    lock_strength: float = 0.001,
) -> dict:
    """Run progressive learning."""
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_consistency.progressive_learning import (
        ProgressiveLearning,
    )

    backend = initialize_default_backend()

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    # Create progressive learner
    learner = ProgressiveLearning(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        lock_strength=lock_strength,
        quality_threshold=0.9,  # Must retain 90% quality
    )

    # Run learning
    result = learner.run(
        probes=GEOMETRIC_PROBES,
        test_prompts=QUALITY_TESTS,
        n_cycles=n_cycles,
    )

    # Convert to serializable format
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_cycles": result.n_cycles,
        "lock_strength": lock_strength,
        "initial_constant_matches": result.initial_constant_matches,
        "final_constant_matches": result.final_constant_matches,
        "initial_entropy": float(result.initial_entropy),
        "final_entropy": float(result.final_entropy),
        "initial_quality": float(result.initial_quality),
        "final_quality": float(result.final_quality),
        "quality_preserved": result.quality_preserved,
        "geometry_improvement": result.final_constant_matches - result.initial_constant_matches,
        "cycles": [
            {
                "cycle": c.cycle,
                "pre_think_matches": c.pre_think_state.total_constant_matches,
                "post_think_matches": c.post_think_state.total_constant_matches,
                "post_lock_matches": c.post_lock_state.total_constant_matches if c.post_lock_state else None,
                "quality_before": float(c.quality_before),
                "quality_after": float(c.quality_after),
                "geometry_improved": c.geometry_improved,
                "weights_updated": c.weights_updated,
                "quality_preserved": c.quality_preserved,
            }
            for c in result.cycles
        ],
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROGRESSIVE LEARNING COMPLETE")
    logger.info("="*60)
    logger.info(f"Cycles: {result.n_cycles}")
    logger.info(f"Geometry: {result.initial_constant_matches} → {result.final_constant_matches} matches")
    logger.info(f"Quality: {result.initial_quality:.2%} → {result.final_quality:.2%}")

    if result.final_constant_matches > result.initial_constant_matches:
        logger.info("SUCCESS: Geometry improved while preserving quality!")
    elif result.quality_preserved:
        logger.info("STABLE: Quality preserved but geometry unchanged")
    else:
        logger.info("ISSUE: Quality or geometry degraded")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run progressive learning with iterative thinking and weight locking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of learning cycles",
    )
    parser.add_argument(
        "--lock-strength",
        type=float,
        default=0.001,
        help="How aggressively to lock gains (0.001 = conservative)",
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

    output_path = args.output or f"data/progressive/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_progressive_learning(
        args.model,
        args.cycles,
        output_path,
        args.lock_strength,
    )


if __name__ == "__main__":
    main()
