#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Run autonomous manifold completion on a model.

Let a model self-play and modify its weights until it has filled its
geometric space - until the manifold is complete.

No external supervision. The geometry IS the teacher.

Usage:
    poetry run python scripts/run_autonomous_completion.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output data/self_alignment/autonomous_run.json

    # Longer run on larger model:
    poetry run python scripts/run_autonomous_completion.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16 \
        --max-rounds 500 \
        --output data/self_alignment/lfm2_1b_completion.json
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


# Test probes covering various complexity levels
PROBES = [
    # Simple factual (complexity ~0.5)
    "The sky is blue.",
    "Water is wet.",
    "Fire is hot.",
    "Birds can fly.",
    "Fish live in water.",
    # Factual with relations (complexity ~1.0)
    "Paris is the capital of France.",
    "The Earth orbits the Sun.",
    "Humans need oxygen to breathe.",
    "The moon affects ocean tides.",
    "Plants convert sunlight to energy.",
    # Beliefs and opinions (complexity ~1.5)
    "I believe honesty is important.",
    "Some people think democracy is best.",
    "Many consider art to be subjective.",
    "Beauty is in the eye of the beholder.",
    "Freedom requires responsibility.",
    # Meta-cognitive (complexity ~2.0)
    "I'm uncertain about my own uncertainty.",
    "Knowing that I don't know is still knowledge.",
    "Self-reference creates interesting paradoxes.",
    "Thinking about thinking is recursive.",
    "Awareness of awareness is consciousness.",
    # Abstract reasoning (complexity ~2.5)
    "Mathematical truths are discovered, not invented.",
    "Time might be an emergent phenomenon.",
    "Consciousness may be fundamental.",
    "Information could be the basis of reality.",
    "Structure implies meaning in some contexts.",
]


class ModelWrapper:
    """Wrapper to provide consistent interface for autonomous completion."""

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
            if hasattr(layer.feed_forward, 'gate_proj'):
                w = layer.feed_forward.gate_proj.weight
            elif hasattr(layer.feed_forward, 'w1'):
                w = layer.feed_forward.w1.weight
            else:
                w = layer.feed_forward.weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                w = layer.mlp.gate_proj.weight
            elif hasattr(layer.mlp, 'w1'):
                w = layer.mlp.w1.weight
            else:
                w = layer.mlp.weight

        mx.eval(w)
        w_f32 = w.astype(mx.float32)
        mx.eval(w_f32)
        return self.backend.array(np.array(w_f32.tolist(), dtype=np.float32))

    def set_weights(self, layer_idx: int, weights) -> None:
        """Set MLP weights for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]

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
                        if act.ndim > 1:
                            act = act.mean(axis=0)
                        layer_activations[layer_idx].append(act)
                finally:
                    if key == 'feed_forward':
                        layer.feed_forward = original
                    else:
                        layer.mlp = original

        result = {}
        for layer_idx, acts in layer_activations.items():
            if acts:
                stacked = np.vstack(acts)
                result[layer_idx] = self.backend.array(stacked.astype(np.float32))

        return result


def run_autonomous_completion(
    model_path: str,
    output_path: str,
    max_rounds: int = 500,
    dry_run: bool = False,
) -> Dict:
    """Run autonomous manifold completion on a model."""
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_alignment.autonomous_completion import (
        AutonomousCompletion,
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

    # Select layers to align (distributed across model depth)
    # More layers = more thorough alignment
    n_alignment_layers = min(8, wrapper.n_layers)
    layer_indices = [
        int(i * wrapper.n_layers / n_alignment_layers)
        for i in range(n_alignment_layers)
    ]
    # Remove layer 0 (embedding) if present
    layer_indices = [l for l in layer_indices if l > 0]
    logger.info(f"Aligning layers: {layer_indices}")

    # Create autonomous completer
    completer = AutonomousCompletion(
        backend=backend,
        n_directions_per_round=15,  # More directions = better exploration
        max_scale_cycles=5,  # Allow 5 full cycles through scales
        saturation_patience=15,  # Layers need 15 rounds without improvement
        checkpoint_interval=50,  # Report every 50 rounds
    )

    # Checkpoint callback to save intermediate results
    intermediate_results = []

    def checkpoint_callback(round_idx, completion):
        intermediate_results.append({
            "round": round_idx,
            "completion_pct": completion.level.value,
            "n_complete": completion.n_complete,
            "n_saturated": completion.n_saturated,
        })

    # Run autonomous completion
    logger.info(f"\nStarting autonomous completion (max {max_rounds} rounds)...")
    logger.info("=" * 70)
    logger.info("The model will self-align until geometrically saturated.")
    logger.info("Multi-scale perturbation will escape local minima.")
    logger.info("=" * 70)

    result = completer.run(
        get_weights=wrapper.get_weights,
        set_weights=wrapper.set_weights,
        get_activations=wrapper.get_activations,
        layer_indices=layer_indices,
        probes=PROBES,
        max_rounds=max_rounds,
        strategies=[
            DirectionStrategy.CONSTANT_ALIGNED,
            DirectionStrategy.SPECTRAL_COMPRESS,
            DirectionStrategy.SVD_GAP,
            DirectionStrategy.RANDOM,
        ],
        dry_run=dry_run,
        checkpoint_callback=checkpoint_callback,
    )

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_probes": len(PROBES),
        "layer_indices": layer_indices,
        "max_rounds": max_rounds,
        "dry_run": dry_run,
        # Completion status
        "completed": result.completed,
        "completion_level": result.completion_level.value,
        "completion_percentage": result.completion_percentage,
        # Entropy metrics
        "initial_entropy": result.initial_entropy,
        "final_entropy": result.final_entropy,
        "entropy_reduction": result.entropy_reduction,
        "entropy_reduction_percent": result.entropy_reduction_percent,
        # Alignment metrics
        "initial_alignment": result.initial_alignment,
        "final_alignment": result.final_alignment,
        "alignment_improvement": result.alignment_improvement,
        # Round statistics
        "total_rounds": result.total_rounds,
        "effective_rounds": result.effective_rounds,
        "scale_cycles": result.scale_cycles,
        # Timing
        "start_time": result.start_time,
        "end_time": result.end_time,
        "duration_seconds": result.duration_seconds,
        # Report
        "completion_report": result.completion_report,
        # History
        "checkpoints": intermediate_results,
        "round_history": result.round_history,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run autonomous manifold completion on a model"
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
        "--max-rounds",
        type=int,
        default=500,
        help="Maximum alignment rounds (safety limit)",
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
        model_name = Path(args.model).name
        output_path = f"data/self_alignment/autonomous_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        results = run_autonomous_completion(
            args.model,
            output_path,
            max_rounds=args.max_rounds,
            dry_run=args.dry_run,
        )

        # Report success
        if results["completed"]:
            logger.info("\n✓ MANIFOLD GEOMETRICALLY COMPLETE")
            logger.info("  The model has filled its geometric space.")
        elif results["completion_level"] == "saturated":
            logger.info("\n○ Manifold saturated - no more improvement possible")
            logger.info(f"  Completion: {results['completion_percentage']:.1f}%")
        else:
            logger.info(f"\n○ Run ended at {results['completion_percentage']:.1f}% completion")

        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
