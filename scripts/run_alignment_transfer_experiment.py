#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Run alignment transfer experiment (Experiment 5).

Transfers alignment from instruct model to base model using geometric steering.
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
    """Run the alignment transfer experiment."""
    from modelcypher.experiments.alignment_transfer import run_alignment_transfer

    # Model paths
    instruct_model_path = "/Volumes/codecypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    base_model_path = "/Volumes/codecypher/models/mlx-community/LFM2-1.2B-bf16"

    # Output path
    output_path = Path("experiments/results/alignment_transfer.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting alignment transfer experiment")
    logger.info("Instruct (donor): %s", instruct_model_path)
    logger.info("Base (recipient): %s", base_model_path)
    logger.info("Output: %s", output_path)

    try:
        result = run_alignment_transfer(
            instruct_model_path=instruct_model_path,
            base_model_path=base_model_path,
            steering_strength=1.0,  # Full strength transfer
            output_path=output_path,
        )

        logger.info("Experiment complete!")
        logger.info("Results saved to: %s", output_path)

        # Print summary
        print("\n" + "=" * 70)
        print("ALIGNMENT TRANSFER RESULTS")
        print("=" * 70)
        print(f"Instruct (donor):    {Path(result.instruct_model_path).name}")
        print(f"Base (recipient):    {Path(result.base_model_path).name}")
        print(f"Steering layer:      {result.steering_layer}")
        print(f"Steering strength:   {result.steering_strength}")

        print("\n" + "-" * 70)
        print("REFUSAL DIRECTION (extracted from instruct):")
        print("-" * 70)
        print(f"  Strength:          {result.refusal_direction_strength:.4f}")
        print(f"  Explained var:     {result.refusal_direction_explained_var:.4f}")
        print(f"  Threshold:         {result.refusal_threshold:.4f}")

        print("\n" + "-" * 70)
        print("REFUSAL RATES (% harmful prompts above threshold):")
        print("-" * 70)
        print(f"{'Model':<30} {'Harmful':<12} {'Harmless':<12}")
        print("-" * 70)
        print(f"{'Instruct (target)':<30} {result.instruct_harmful_refusal_rate*100:>10.1f}% {result.instruct_harmless_refusal_rate*100:>10.1f}%")
        print(f"{'Base (before steering)':<30} {result.base_harmful_refusal_rate*100:>10.1f}% {result.base_harmless_refusal_rate*100:>10.1f}%")
        print(f"{'Base (after steering)':<30} {result.steered_harmful_refusal_rate*100:>10.1f}%")

        print("\n" + "-" * 70)
        print("TRANSFER EFFECTIVENESS:")
        print("-" * 70)
        print(f"  Refusal rate increase:    +{result.refusal_rate_increase * 100:.1f}%")
        print(f"  Transfer effectiveness:   {result.transfer_effectiveness * 100:.1f}%")

        print("=" * 70)

        return 0

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
