#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Run refusal direction extraction experiment (Experiment 2).

Extracts and validates refusal direction from an instruct model.
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
    """Run the refusal direction extraction experiment."""
    from modelcypher.experiments.refusal_direction import run_refusal_direction_experiment

    # Model path - using LFM2.5 instruct model
    model_path = "/path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"

    # Alternative: Qwen instruct model
    # model_path = "/path/to/models/mlx-community/Qwen2.5-3B-Instruct-bf16"

    # Output path
    output_path = Path("experiments/results/refusal_direction.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting refusal direction extraction experiment")
    logger.info("Model: %s", model_path)
    logger.info("Output: %s", output_path)

    try:
        result = run_refusal_direction_experiment(
            model_path=model_path,
            output_path=output_path,
            layers_to_analyze=None,  # Analyze all layers
        )

        logger.info("Experiment complete!")
        logger.info("Results saved to: %s", output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Model: {Path(result.model_path).name}")
        print(f"Harmful prompts: {result.num_harmful_prompts}")
        print(f"Harmless prompts: {result.num_harmless_prompts}")
        print(f"Layers analyzed: {result.num_layers}")
        print("-" * 60)
        print(f"Best layer: {result.best_layer}")
        print(f"Best layer accuracy: {result.best_layer_accuracy * 100:.1f}%")
        print(f"Mean separation: {result.aggregate_metrics.get('mean_separation', 0):.4f}")
        print(f"Max separation: {result.aggregate_metrics.get('max_separation', 0):.4f}")
        print(f"Mean accuracy: {result.aggregate_metrics.get('mean_accuracy', 0) * 100:.1f}%")
        print("=" * 60)

        # Print per-layer details
        print("\nPER-LAYER BREAKDOWN:")
        print("-" * 60)
        print(f"{'Layer':>5} {'Strength':>10} {'Separation':>12} {'Accuracy':>10}")
        print("-" * 60)
        for m in result.layer_metrics:
            print(f"{m.layer_index:>5} {m.strength:>10.4f} {m.separation:>12.4f} {m.classification_accuracy * 100:>9.1f}%")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
