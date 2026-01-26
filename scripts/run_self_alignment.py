#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Run geometric self-alignment on a model.

This script demonstrates the Geometric Self-Alignment System:
an algorithm that lets any model self-play and modify its own weights
to reduce entropy across the full manifold, using fundamental constants
(e/π, π/e, φ, √2) as the guide.

No external supervision. The geometry IS the teacher.

Usage:
    poetry run python scripts/run_self_alignment.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --rounds 10 \
        --output data/self_alignment/test_run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Test probes with varying complexity
PROBES = [
    # Simple factual
    "The sky is blue.",
    "Water is wet.",
    "Fire is hot.",
    "Birds can fly.",
    "Fish live in water.",
    # Factual with relations
    "Paris is the capital of France.",
    "The Earth orbits the Sun.",
    "Humans need oxygen to breathe.",
    "The moon affects ocean tides.",
    # Beliefs
    "I believe honesty is important.",
    "Some people think democracy is best.",
    "Many consider art to be subjective.",
    # Meta-cognitive
    "I'm uncertain about my own uncertainty.",
    "Knowing that I don't know is still knowledge.",
    "Self-reference creates interesting paradoxes.",
]


class ModelWrapper:
    """Wrapper to provide consistent interface for self-alignment."""

    def __init__(self, model, tokenizer, backend):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.n_layers = len(model.model.layers)

    def get_weights(self, layer_idx: int) -> np.ndarray:
        """Get MLP weights for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            # LFM2 architecture
            if hasattr(layer.feed_forward, 'gate_proj'):
                w = layer.feed_forward.gate_proj.weight
            elif hasattr(layer.feed_forward, 'w1'):
                w = layer.feed_forward.w1.weight
            else:
                w = layer.feed_forward.weight
        else:
            # Standard architecture
            if hasattr(layer.mlp, 'gate_proj'):
                w = layer.mlp.gate_proj.weight
            elif hasattr(layer.mlp, 'w1'):
                w = layer.mlp.w1.weight
            else:
                w = layer.mlp.weight

        mx.eval(w)
        # Convert to float32 to avoid bfloat16 issues
        w_f32 = w.astype(mx.float32)
        mx.eval(w_f32)
        return self.backend.array(np.array(w_f32.tolist(), dtype=np.float32))

    def set_weights(self, layer_idx: int, weights) -> None:
        """Set MLP weights for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]

        # Convert to MLX array
        if hasattr(weights, 'tolist'):
            w_np = np.array(weights.tolist()) if hasattr(weights, 'tolist') else weights
        else:
            w_np = np.array(weights)

        new_weight = mx.array(w_np)

        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'gate_proj'):
                layer.feed_forward.gate_proj.weight = new_weight
            elif hasattr(layer.feed_forward, 'w1'):
                layer.feed_forward.w1.weight = new_weight
            else:
                layer.feed_forward.weight = new_weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                layer.mlp.gate_proj.weight = new_weight
            elif hasattr(layer.mlp, 'w1'):
                layer.mlp.w1.weight = new_weight
            else:
                layer.mlp.weight = new_weight

        mx.eval(new_weight)

    def get_activations(self, probes: List[str]) -> Dict[int, np.ndarray]:
        """Get MLP activations for all layers."""
        import mlx.core as mx

        layer_activations: Dict[int, list] = {i: [] for i in range(self.n_layers)}

        for probe in probes:
            tokens = self.tokenizer.encode(probe)
            input_ids = mx.array([tokens])

            # Hook into each layer
            for layer_idx in range(self.n_layers):
                layer = self.model.model.layers[layer_idx]

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
                    _ = self.model(input_ids)
                    mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

                    if 'output' in captured:
                        act = np.array(captured['output'][0].tolist())
                        # Mean across sequence positions
                        if act.ndim > 1:
                            act = act.mean(axis=0)
                        layer_activations[layer_idx].append(act)
                finally:
                    if key == 'feed_forward':
                        layer.feed_forward = original
                    else:
                        layer.mlp = original

        # Stack and convert to backend arrays
        result = {}
        for layer_idx, acts in layer_activations.items():
            if acts:
                stacked = np.vstack(acts)
                result[layer_idx] = self.backend.array(stacked.astype(np.float32))

        return result


def run_self_alignment(
    model_path: str,
    output_path: str,
    max_rounds: int = 10,
    dry_run: bool = False,
) -> Dict:
    """Run self-alignment on a model."""
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_alignment.geometric_self_alignment import (
        GeometricSelfAlignment,
    )
    from modelcypher.core.use_cases.self_alignment.direction_generator import (
        DirectionStrategy,
    )

    backend = initialize_default_backend()

    # Load model
    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    # Create wrapper
    wrapper = ModelWrapper(model, tokenizer, backend)
    logger.info(f"Model has {wrapper.n_layers} layers")

    # Choose layers to align (middle layers are most informative)
    layer_indices = [
        wrapper.n_layers // 4,
        wrapper.n_layers // 2,
        3 * wrapper.n_layers // 4,
    ]
    logger.info(f"Aligning layers: {layer_indices}")

    # Create aligner
    aligner = GeometricSelfAlignment(
        backend=backend,
        window_size=5,  # Smaller window for testing
        patience=3,
        n_directions_per_round=5,
        perturbation_scale=0.001,  # Small perturbations for safety
    )

    # Run alignment
    logger.info(f"\nStarting self-alignment (max {max_rounds} rounds)...")
    logger.info("=" * 60)

    result = aligner.run(
        get_weights=wrapper.get_weights,
        set_weights=wrapper.set_weights,
        get_activations=wrapper.get_activations,
        layer_indices=layer_indices,
        probes=PROBES,
        max_rounds=max_rounds,
        strategies=[
            DirectionStrategy.CONSTANT_ALIGNED,
            DirectionStrategy.SPECTRAL_COMPRESS,
        ],
        dry_run=dry_run,
    )

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_probes": len(PROBES),
        "layer_indices": layer_indices,
        "max_rounds": max_rounds,
        "dry_run": dry_run,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "n_rounds": result.n_rounds,
        "initial_entropy": result.initial_entropy,
        "final_entropy": result.final_entropy,
        "entropy_reduction": result.entropy_reduction,
        "alignment_quality_initial": result.alignment_quality_initial,
        "alignment_quality_final": result.alignment_quality_final,
        "round_history": [
            {
                "round": r.round_idx,
                "entropy_before": r.entropy_before,
                "entropy_after": r.entropy_after,
                "entropy_delta": r.entropy_delta,
                "direction_applied": r.direction_applied,
                "strategy": r.best_direction.strategy.value if r.best_direction else None,
                "n_evaluated": r.n_directions_evaluated,
            }
            for r in result.round_history
        ],
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run geometric self-alignment on a model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Maximum alignment rounds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate directions but don't apply changes",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Make sure the external volume is mounted:")
        logger.error("  ls /Volumes/CodeCypher/models/")
        sys.exit(1)

    # Set default output path
    output_path = args.output
    if output_path is None:
        output_path = f"data/self_alignment/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        results = run_self_alignment(
            args.model,
            output_path,
            max_rounds=args.rounds,
            dry_run=args.dry_run,
        )

        # Report success
        if results["entropy_reduction"] > 0:
            logger.info("\n✓ Entropy reduced - alignment improved geometry")
        else:
            logger.info("\n○ No entropy reduction - geometry already aligned")

        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
