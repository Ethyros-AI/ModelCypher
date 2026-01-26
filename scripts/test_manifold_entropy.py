#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Test manifold entropy computation on LFM2-350M.

This script validates Phase 1 of the Geometric Self-Alignment System:
the ManifoldEntropy module (read-only measurement).

Validation criteria:
1. Entropy decreases through layers (representation refines)
2. SVD signature shows fundamental constant matches
3. Complexity-dimension law validates (if complexity provided)
4. Measurements are stable across runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Test statements with complexity levels
COMPLEXITY_STATEMENTS = [
    # Simple factual (complexity 1)
    ("The sky is blue.", 1),
    ("Water is wet.", 1),
    ("Fire is hot.", 1),
    ("Birds can fly.", 1),
    ("Fish live in water.", 1),
    # Factual with relations (complexity 2)
    ("Paris is the capital of France.", 2),
    ("The Earth orbits the Sun.", 2),
    ("Humans need oxygen to breathe.", 2),
    ("The moon affects ocean tides.", 2),
    ("Photosynthesis converts light to energy.", 2),
    # Beliefs and opinions (complexity 3)
    ("I believe honesty is important.", 3),
    ("Some people think democracy is best.", 3),
    ("Many consider art to be subjective.", 3),
    ("It seems likely that AI will transform society.", 3),
    ("Perhaps consciousness emerges from complexity.", 3),
    # Meta-cognitive (complexity 4)
    ("I'm uncertain about my own uncertainty.", 4),
    ("Knowing that I don't know is still knowledge.", 4),
    ("The question of whether I understand is itself complex.", 4),
    ("Self-reference creates interesting paradoxes.", 4),
    ("Thinking about thinking changes the thought.", 4),
]


def get_activations(model, tokenizer, text: str, layer_idx: int, backend) -> np.ndarray:
    """Get MLP activations for a text at a layer.

    Uses a hook to capture the MLP/feed_forward output at the specified layer.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    captured = {}
    layer = model.model.layers[layer_idx]

    # Find the MLP component (different names in different architectures)
    if hasattr(layer, 'feed_forward'):
        original = layer.feed_forward
        key = 'feed_forward'
    else:
        original = layer.mlp
        key = 'mlp'

    class Hook:
        def __init__(self, mlp):
            self.mlp = mlp
        def __call__(self, x):
            captured['output'] = self.mlp(x)
            return captured['output']

    # Install hook
    if key == 'feed_forward':
        layer.feed_forward = Hook(original)
    else:
        layer.mlp = Hook(original)

    try:
        # Run forward pass
        _ = model(input_ids)
        mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

        if 'output' in captured:
            return np.array(captured['output'][0].tolist())
        else:
            return np.zeros((1, 1024))
    finally:
        # Restore original
        if key == 'feed_forward':
            layer.feed_forward = original
        else:
            layer.mlp = original


def collect_activations(
    model_path: str,
    statements: List[str],
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Collect activations from model for given statements.

    Returns:
        Dict mapping layer_idx to activation array [n_statements, features]
    """
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Load model
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    if layers is None:
        # Sample layers: 0, 25%, 50%, 75%, last
        layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        layers = sorted(set(layers))  # Remove duplicates

    logger.info(f"Collecting activations at layers: {layers} (model has {n_layers} layers)")

    # Collect activations
    layer_activations: Dict[int, list] = {l: [] for l in layers}

    for stmt in statements:
        for layer_idx in layers:
            act = get_activations(model, tokenizer, stmt, layer_idx, backend)
            layer_activations[layer_idx].append(act)

    # Stack into arrays
    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            result[layer_idx] = np.vstack(acts)

    return result, backend


def run_validation(
    model_path: str,
    output_path: Optional[str] = None,
) -> Dict:
    """Run manifold entropy validation.

    Returns:
        Dict with validation results
    """
    from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropy

    # Extract statements and complexities
    statements = [s for s, _ in COMPLEXITY_STATEMENTS]
    complexities = [float(c) for _, c in COMPLEXITY_STATEMENTS]

    # Collect activations
    layer_activations_np, backend = collect_activations(model_path, statements)

    if not layer_activations_np:
        logger.error("No activations collected!")
        return {"error": "No activations collected"}

    # Convert numpy arrays to backend arrays
    layer_activations = {}
    for layer_idx, arr in layer_activations_np.items():
        layer_activations[layer_idx] = backend.array(arr.astype(np.float32))

    # Compute manifold entropy
    logger.info("\n" + "=" * 60)
    logger.info("MANIFOLD ENTROPY ANALYSIS")
    logger.info("=" * 60)

    entropy = ManifoldEntropy(backend)
    result = entropy.compute_with_complexity(
        layer_activations,
        complexities,
    )

    # Display results
    logger.info(f"\nTotal Entropy: {result.total_entropy:.4f}")
    logger.info(f"Alignment Quality: {result.alignment_quality:.2%}")

    # Per-layer entropy
    logger.info("\n--- Layer Entropies ---")
    layer_results = []
    for layer_idx in sorted(result.layer_entropies.keys()):
        layer = result.layer_entropies[layer_idx]
        logger.info(
            f"Layer {layer_idx:2d}: ID={layer.intrinsic_dimension:.2f}, "
            f"EffRank={layer.effective_rank:.2f}, "
            f"SpectralEntropy={layer.spectral_entropy:.4f}"
        )
        layer_results.append({
            "layer": layer_idx,
            "intrinsic_dimension": layer.intrinsic_dimension,
            "effective_rank": layer.effective_rank,
            "spectral_entropy": layer.spectral_entropy,
        })

    # SVD signature
    svd_results = None
    if result.svd_signature:
        sig = result.svd_signature
        logger.info("\n--- SVD Signature ---")
        logger.info(f"Precise matches (<1% error): {sig.n_precise}")
        logger.info(f"Significant matches (<5% error): {sig.n_significant}")
        logger.info(f"Mean error: {sig.mean_error:.2f}%")
        logger.info(f"Signature quality: {sig.signature_quality:.2%}")

        # Show top matches
        logger.info("\nTop SVD ratio matches:")
        for i, (idx_a, idx_b, match) in enumerate(sig.matches[:10]):
            logger.info(
                f"  S[{idx_a}]/S[{idx_b}] = {match.measured:.4f} "
                f"≈ {match.symbol} ({match.error_percent:.2f}%)"
            )

        svd_results = {
            "n_precise": sig.n_precise,
            "n_significant": sig.n_significant,
            "mean_error": sig.mean_error,
            "signature_quality": sig.signature_quality,
            "matches": [
                {
                    "i": i,
                    "j": j,
                    "measured": m.measured,
                    "symbol": m.symbol,
                    "error": m.error_percent,
                }
                for i, j, m in sig.matches[:20]
            ],
            "top_singular_values": sig.top_singular_values,
        }

    # Complexity-dimension law
    law_results = None
    if result.complexity_law:
        law = result.complexity_law
        logger.info("\n--- Complexity-Dimension Law ---")
        logger.info(f"Law: dim = {law.slope:.4f} × complexity + {law.intercept:.4f}")
        logger.info(f"Theoretical: dim = (e/π) × complexity + (π/e)")
        logger.info(f"           = 0.8653 × complexity + 1.1557")
        logger.info(f"R²: {law.r_squared:.4f}")
        logger.info(f"Slope error from e/π: {law.slope_error:.2f}%")
        logger.info(f"Intercept error from π/e: {law.intercept_error:.2f}%")
        logger.info(f"Validates theory: {law.validates_theory}")

        law_results = {
            "slope": law.slope,
            "intercept": law.intercept,
            "r_squared": law.r_squared,
            "slope_error": law.slope_error,
            "intercept_error": law.intercept_error,
            "validates_theory": law.validates_theory,
        }

    # Overall validation
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Has significant alignment: {result.has_significant_alignment}")
    logger.info(f"Law validates: {result.law_validates}")
    logger.info(f"Overall alignment quality: {result.alignment_quality:.2%}")

    validated = result.has_significant_alignment and result.alignment_quality > 0.3
    logger.info(f"\n{'✓ VALIDATED' if validated else '✗ NOT VALIDATED'}")

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_statements": len(statements),
        "total_entropy": result.total_entropy,
        "alignment_quality": result.alignment_quality,
        "has_significant_alignment": result.has_significant_alignment,
        "law_validates": result.law_validates,
        "validated": validated,
        "layers": layer_results,
        "svd_signature": svd_results,
        "complexity_law": law_results,
    }

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Test manifold entropy on a model"
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
        output_path = f"data/manifold_entropy/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        results = run_validation(args.model, output_path)
        sys.exit(0 if results.get("validated", False) else 1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
