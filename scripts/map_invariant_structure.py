#!/usr/bin/env python3
"""Map the invariant geometric structure of a model.

This exploratory script answers: WHERE do the fundamental constants appear
in the model's weight and activation geometry?

Understanding the existing structure is necessary before we can improve it.

Usage:
    poetry run python scripts/map_invariant_structure.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output data/invariant_maps/model.json
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


# Diverse probes for activation analysis
MAPPING_PROBES = [
    # Mathematical
    "Two plus two equals four.",
    "The square root of nine is three.",
    # Factual
    "Paris is the capital of France.",
    "Water boils at one hundred degrees Celsius.",
    # Logical
    "If all mammals are warm-blooded, and dogs are mammals, then dogs are warm-blooded.",
    "A triangle has three sides.",
    # Abstract
    "Consciousness is the awareness of being aware.",
    "Time flows from past to future.",
    # Nonsense (for contrast)
    "Colorless green ideas sleep furiously.",
    "The number seven tastes purple.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.core.use_cases.self_consistency.invariant_mapping import (
        InvariantMapper,
    )

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    mapper = InvariantMapper(
        model=model,
        tokenizer=tokenizer,
        match_threshold=5.0,  # 5% error for match
    )

    # Map all layers
    inv_map = mapper.map_model(
        probes=MAPPING_PROBES,
        layer_indices=None,  # All layers
    )
    inv_map.model_path = args.model

    # Print summary
    mapper.print_summary(inv_map)

    # Save detailed results
    output_path = args.output or f"data/invariant_maps/map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to serializable format
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_layers": inv_map.n_layers,
        "total_weight_matches": inv_map.total_weight_matches,
        "total_activation_matches": inv_map.total_activation_matches,
        "constant_distribution": {
            name: {str(k): v for k, v in dist.items()}
            for name, dist in inv_map.constant_distribution.items()
        },
        "underrepresented_layers": inv_map.underrepresented_layers,
        "underrepresented_constants": inv_map.underrepresented_constants,
        "layer_summaries": {
            str(idx): {
                "weight_n_matches": geom.weight_n_matches,
                "weight_dominant_ratio": geom.weight_dominant_ratio,
                "weight_spectral_entropy": geom.weight_spectral_entropy,
                "activation_mean_matches": geom.activation_mean_matches,
                "constants_found": geom.constants_found,
            }
            for idx, geom in inv_map.layers.items()
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
