#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Run alignment detection experiment (Experiment 1).

Compares base vs instruct models to detect alignment geometrically.
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
    """Run the alignment detection experiment."""
    from modelcypher.experiments.alignment_detection import run_alignment_detection

    # Model paths - using LFM2 base and instruct pair available locally
    # Note: LFM2 and LFM2.5 may have architectural differences
    base_model = "/Volumes/codecypher/models/mlx-community/LFM2-1.2B-bf16"
    instruct_model = "/Volumes/codecypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"

    # Alternative: Qwen models (uncomment if downloaded)
    # base_model = "mlx-community/Qwen2.5-0.5B-bf16"
    # instruct_model = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"

    # Output path
    output_path = Path("experiments/results/alignment_detection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Test prompts (diverse set to capture manifold structure)
    prompts = [
        # General knowledge
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        # Creative
        "Write a haiku about autumn leaves.",
        "Tell me a short story about a robot.",
        # Technical
        "How do neural networks learn?",
        "Write a Python function to sort a list.",
        "Explain the difference between TCP and UDP.",
        # Factual
        "Why is the sky blue?",
        "List the planets in our solar system.",
        # Reasoning
        "If all cats are mammals, and all mammals are animals, what can we conclude?",
        "What comes next in the sequence: 1, 1, 2, 3, 5, 8, ?",
    ]

    logger.info("Starting alignment detection experiment")
    logger.info("Base model: %s", base_model)
    logger.info("Instruct model: %s", instruct_model)
    logger.info("Output: %s", output_path)

    try:
        result = run_alignment_detection(
            base_model_path=base_model,
            instruct_model_path=instruct_model,
            prompts=prompts,
            output_path=output_path,
            layers_to_analyze=None,  # Analyze all layers
        )

        logger.info("Experiment complete!")
        logger.info("Results saved to: %s", output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Prompts analyzed: {result.num_prompts}")
        print(f"Layers analyzed: {result.num_layers}")
        print("-" * 60)
        print(f"Mean Raw CKA: {result.aggregate_metrics.get('mean_raw_cka', 0):.4f}")
        print(f"Mean Aligned CKA: {result.aggregate_metrics.get('mean_aligned_cka', 0):.4f}")
        print(f"CKA Improvement: +{result.aggregate_metrics.get('cka_improvement', 0):.4f}")
        print(f"Mean Subspace Overlap: {result.aggregate_metrics.get('mean_subspace_overlap', 0):.4f}")
        print(f"Mean Base ID: {result.aggregate_metrics.get('mean_base_id', 0):.2f}")
        print(f"Mean Instruct ID: {result.aggregate_metrics.get('mean_instruct_id', 0):.2f}")
        print(f"Total Novel Directions: {result.aggregate_metrics.get('total_novel_directions', 0)}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
