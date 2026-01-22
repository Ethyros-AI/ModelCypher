#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Run cross-model universality experiment (Experiment 3).

Tests whether refusal direction is universal across model architectures.
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run the cross-model universality experiment."""
    from modelcypher.experiments.cross_model_universality import run_cross_model_experiment

    # Models to test - different architectures and sizes
    model_paths = [
        "/path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16",
        "/path/to/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
        "/path/to/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/path/to/models/mlx-community/granite-3b-code-instruct-128k-mlx",
    ]

    # Filter to existing models
    existing_models = [p for p in model_paths if Path(p).exists()]

    if not existing_models:
        logger.error("No models found!")
        return 1

    logger.info("Found %d models to analyze", len(existing_models))
    for m in existing_models:
        logger.info("  - %s", Path(m).name)

    # Output path
    output_path = Path("experiments/results/cross_model_universality.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting cross-model universality experiment")
    logger.info("Output: %s", output_path)

    try:
        result = run_cross_model_experiment(
            model_paths=existing_models,
            output_path=output_path,
        )

        logger.info("Experiment complete!")
        logger.info("Results saved to: %s", output_path)

        # Print summary
        print("\n" + "=" * 70)
        print("CROSS-MODEL UNIVERSALITY RESULTS")
        print("=" * 70)

        print("\nMODEL PROFILES:")
        print("-" * 70)
        print(f"{'Model':<35} {'Hidden':<8} {'Layers':<8} {'Best Acc':<10} {'Best Layer':<10}")
        print("-" * 70)
        for p in result.model_profiles:
            print(f"{p.model_name[:34]:<35} {p.hidden_size:<8} {p.num_layers:<8} {p.best_accuracy*100:>7.1f}% {p.best_layer:<10}")

        print("\nPAIRWISE COMPARISONS:")
        print("-" * 70)
        print(f"{'Model A':<20} {'Model B':<20} {'Acc Corr':<10} {'Sep Corr':<10} {'Cosine':<10}")
        print("-" * 70)
        for c in result.pairwise_comparisons:
            cos_str = f"{c.cosine_similarity:.3f}" if c.cosine_similarity is not None else "N/A"
            print(f"{c.model_a[:19]:<20} {c.model_b[:19]:<20} {c.accuracy_correlation:>8.3f} {c.separation_correlation:>9.3f} {cos_str:>9}")

        print("\nAGGREGATE METRICS:")
        print("-" * 70)
        for key, value in result.aggregate_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("=" * 70)

        return 0

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
