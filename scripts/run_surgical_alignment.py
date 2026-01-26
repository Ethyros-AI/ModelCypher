#!/usr/bin/env python3
"""Run Surgical Geometric Alignment.

Based on experimental results showing:
1. Constants are real (p < 0.01)
2. Surgical SVD modification preserves quality
3. Only ratios close to constants should be aligned

Usage:
    poetry run python scripts/run_surgical_alignment.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output data/surgical/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--proximity", type=float, default=0.10, help="Only align ratios within this % of a constant")
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.core.use_cases.self_consistency.surgical_geometric_alignment import (
        SurgicalGeometricAlignment,
    )

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    aligner = SurgicalGeometricAlignment(
        model=model,
        tokenizer=tokenizer,
        proximity_threshold=args.proximity,
        quality_threshold=args.quality_threshold,
    )

    result = aligner.run(
        test_prompts=TEST_PROMPTS,
        layer_indices=None,  # Default middle layers
        max_targets_per_layer=3,
    )

    # Save results
    output_path = args.output or f"data/surgical/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "proximity_threshold": args.proximity,
        "quality_threshold": args.quality_threshold,
        "layers_processed": result.layers_processed,
        "total_targets_aligned": result.total_targets_aligned,
        "total_matches_before": result.total_matches_before,
        "total_matches_after": result.total_matches_after,
        "quality_before": result.quality_before,
        "quality_after": result.quality_after,
        "layer_results": [
            {
                "layer_idx": lr.layer_idx,
                "targets_found": lr.targets_found,
                "targets_aligned": lr.targets_aligned,
                "matches_before": lr.total_matches_before,
                "matches_after": lr.total_matches_after,
                "quality_preserved": lr.quality_preserved,
            }
            for lr in result.layer_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Summary
    if result.total_matches_after > result.total_matches_before:
        logger.info(f"\nSUCCESS: Matches improved {result.total_matches_before} → {result.total_matches_after}")
    else:
        logger.info(f"\nNo improvement in matches")


if __name__ == "__main__":
    main()
