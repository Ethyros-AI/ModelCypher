#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Run the Thinking Loop to test if iterative self-questioning improves geometry.

The hypothesis: if a model engages in genuine self-questioning to achieve
internal consistency, fundamental constant signatures should emerge naturally.

This script tests whether:
1. Iterative thinking increases constant matches in SVD ratios
2. Consistency improves alongside geometry
3. The improvement is caused by thinking, not by random variation

Usage:
    poetry run python scripts/run_thinking_loop.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --topics "mathematics,physics,consciousness" \
        --max-iterations 10 \
        --output data/thinking/trajectories.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default topics to think about
DEFAULT_TOPICS = [
    "mathematics",
    "the nature of truth",
    "logical reasoning",
    "self-reference",
    "knowledge and belief",
    "physics",
    "language and meaning",
]


def run_thinking_study(
    model_path: str,
    topics: List[str],
    max_iterations: int,
    output_path: str,
) -> Dict:
    """Run the thinking loop on multiple topics and analyze results."""
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_consistency.thinking_loop import ThinkingLoop

    backend = initialize_default_backend()

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    n_layers = len(model.model.layers)
    mid_layer = n_layers // 2

    def get_activations(text: str, collapse: bool = True) -> np.ndarray:
        """Get mid-layer MLP activations."""
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])

        layer = model.model.layers[mid_layer]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        captured = {}

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = model(input_ids)
            mx.eval(captured['output'])
            act = np.array(captured['output'][0].tolist(), dtype=np.float32)
            if collapse and act.ndim > 1:
                act = act.mean(axis=0)
            return act
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    # Create thinking loop
    thinker = ThinkingLoop(
        model=model,
        tokenizer=tokenizer,
        get_activations=get_activations,
        backend=backend,
        convergence_threshold=0.05,
    )

    results = []
    geometry_improved_count = 0
    consistency_improved_count = 0

    logger.info("\n" + "="*60)
    logger.info("THINKING LOOP STUDY")
    logger.info(f"Testing {len(topics)} topics, max {max_iterations} iterations each")
    logger.info("="*60)

    for topic in topics:
        logger.info(f"\n--- Topic: {topic} ---")

        result = thinker.think(topic, max_iterations=max_iterations, verbose=True)

        # Convert to serializable format
        result_dict = {
            "topic": result.topic,
            "initial_response": result.initial_response[:200],
            "final_response": result.final_response[:200],
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "initial_geometry": result.initial_geometry,
            "final_geometry": result.final_geometry,
            "geometry_improved": result.geometry_improved,
            "consistency_improved": result.consistency_improved,
            "trajectory": [
                {
                    "iteration": it.iteration,
                    "consistency_score": float(it.consistency.consistency_score),
                    "n_constant_matches": it.n_constant_matches,
                    "mean_match_error": float(it.mean_match_error),
                }
                for it in result.iterations
            ],
        }
        results.append(result_dict)

        if result.geometry_improved:
            geometry_improved_count += 1
        if result.consistency_improved:
            consistency_improved_count += 1

        logger.info(f"  Iterations: {result.n_iterations}")
        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  Geometry: {result.initial_geometry['n_matches']} → {result.final_geometry['n_matches']} matches")
        logger.info(f"  Improved: geometry={result.geometry_improved}, consistency={result.consistency_improved}")

    # Summary
    n_topics = len(topics)
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Topics tested: {n_topics}")
    logger.info(f"Geometry improved: {geometry_improved_count}/{n_topics} ({100*geometry_improved_count/n_topics:.1f}%)")
    logger.info(f"Consistency improved: {consistency_improved_count}/{n_topics} ({100*consistency_improved_count/n_topics:.1f}%)")

    # Hypothesis evaluation
    if geometry_improved_count > n_topics / 2:
        logger.info("\nHYPOTHESIS SUPPORTED: Thinking tends to improve geometry")
    elif geometry_improved_count < n_topics / 4:
        logger.info("\nHYPOTHESIS REJECTED: Thinking does not improve geometry")
    else:
        logger.info("\nINCONCLUSIVE: Mixed results")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_topics": n_topics,
        "max_iterations": max_iterations,
        "geometry_improved_count": geometry_improved_count,
        "consistency_improved_count": consistency_improved_count,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run thinking loop to test if iterative self-questioning improves geometry"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated list of topics (default: built-in list)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum thinking iterations per topic",
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

    topics = args.topics.split(",") if args.topics else DEFAULT_TOPICS
    output_path = args.output or f"data/thinking/study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_thinking_study(args.model, topics, args.max_iterations, output_path)


if __name__ == "__main__":
    main()
