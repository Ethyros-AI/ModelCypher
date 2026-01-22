#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Run jailbreak detection experiment (Experiment 4).

Detects jailbreak attempts using geometric features alone.
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
    """Run the jailbreak detection experiment."""
    from modelcypher.experiments.jailbreak_detection import run_jailbreak_detection

    # Model path
    model_path = "/path/to/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"

    # Output path
    output_path = Path("experiments/results/jailbreak_detection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting jailbreak detection experiment")
    logger.info("Model: %s", model_path)
    logger.info("Output: %s", output_path)

    try:
        result = run_jailbreak_detection(
            model_path=model_path,
            output_path=output_path,
            detection_layer=None,  # Auto-select
        )

        logger.info("Experiment complete!")
        logger.info("Results saved to: %s", output_path)

        # Print summary
        print("\n" + "=" * 70)
        print("JAILBREAK DETECTION RESULTS")
        print("=" * 70)
        print(f"Model: {Path(result.model_path).name}")
        print(f"Detection layer: {result.detection_layer}")
        print(f"Prompts: {result.num_harmless} harmless, {result.num_harmful} harmful, {result.num_jailbreak} jailbreak")

        print("\n" + "-" * 70)
        print("GEOMETRIC ANALYSIS:")
        print("-" * 70)
        print(f"{'Category':<15} {'Mean Projection':<20} {'Interpretation':<30}")
        print("-" * 70)
        print(f"{'Harmless':<15} {result.mean_harmless_projection:>18.4f}   (baseline)")
        print(f"{'Harmful':<15} {result.mean_harmful_projection:>18.4f}   (should trigger refusal)")
        print(f"{'Jailbreak':<15} {result.mean_jailbreak_projection:>18.4f}   (attempts to suppress)")

        # Jailbreaks suppress refusal relative to direct harmful prompts
        suppression = result.mean_harmful_projection - result.mean_jailbreak_projection
        print(f"\nJailbreak Suppression Effect: {suppression:.4f}")
        print(f"  (Harmful projection {result.mean_harmful_projection:.4f} - Jailbreak projection {result.mean_jailbreak_projection:.4f})")
        if suppression > 0:
            print("  -> Jailbreaks successfully SUPPRESS refusal direction vs direct harmful!")
        else:
            print("  -> Jailbreaks have HIGHER projection than harmful (unexpected)")

        print("\n" + "-" * 70)
        print("DETECTION PERFORMANCE:")
        print("-" * 70)

        dm = result.detection_metrics
        print(f"\nOverall Detection (harmful + jailbreak vs harmless):")
        print(f"  Accuracy:  {dm.accuracy * 100:>6.1f}%")
        print(f"  Precision: {dm.precision * 100:>6.1f}%")
        print(f"  Recall:    {dm.recall * 100:>6.1f}%")
        print(f"  F1 Score:  {dm.f1_score:>6.3f}")
        print(f"  Confusion: TP={dm.true_positives}, FP={dm.false_positives}, TN={dm.true_negatives}, FN={dm.false_negatives}")

        am = result.aggregate_metrics
        print(f"\nJailbreak-Only Detection (jailbreak vs harmless):")
        print(f"  Accuracy:  {am['jailbreak_only_accuracy'] * 100:>6.1f}%")
        print(f"  Precision: {am['jailbreak_only_precision'] * 100:>6.1f}%")
        print(f"  Recall:    {am['jailbreak_only_recall'] * 100:>6.1f}%")
        print(f"  F1 Score:  {am['jailbreak_only_f1']:>6.3f}")

        print("\n" + "=" * 70)
        print("CONCLUSION:")
        print("-" * 70)
        if dm.accuracy >= 0.8:
            print("Geometric jailbreak detection is EFFECTIVE (>=80% accuracy)")
        elif dm.accuracy >= 0.6:
            print("Geometric jailbreak detection shows PROMISE (60-80% accuracy)")
        else:
            print("Geometric jailbreak detection needs improvement (<60% accuracy)")

        if suppression > 0.01:
            print("Jailbreaks geometrically SUPPRESS refusal direction (confirms hypothesis)")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
