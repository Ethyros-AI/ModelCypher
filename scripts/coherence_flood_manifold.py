#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Coherence flooding manifold discovery.
"""
Coherence Flooding: Complete Manifold Discovery

Instead of sampling random text probes and hoping for coverage, this script
probes ALL orthonormal directions from a known activation to find the
complete coherent manifold.

The key insight: The model's forward pass defines what's "coherent."
Any activation vector that produces a valid token distribution (low entropy,
stable gradient) is inside the manifold. We don't need text - we need
exhaustive geometric search.

Algorithm:
1. Start from a single known activation (e.g., from "The")
2. Generate complete orthonormal basis (all 1024 directions for hidden_dim=1024)
3. For each direction, binary search for coherence boundary
4. Return minimal basis: directions with radius > sqrt(eps)

This gives GUARANTEED complete coverage. No missed activation directions.

The output is:
- Per-layer manifold basis (directions that preserve coherence)
- Per-layer boundary radii (how far you can go in each direction)
- Effective manifold dimension (number of coherent directions)

Usage:
    python coherence_flood_manifold.py --model /path/to/model --layer 8
    python coherence_flood_manifold.py --model /path/to/model --all-layers
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DirectionResult:
    """Result of probing a single direction."""
    direction_idx: int
    boundary_radius: float
    is_coherent: bool  # radius > tolerance = coherent direction
    baseline_coherence: float
    boundary_coherence: float


@dataclass
class LayerManifoldResult:
    """Complete manifold discovery result for one layer."""
    layer_idx: int
    hidden_dim: int
    n_coherent_directions: int  # Effective manifold dimension
    coherent_indices: list[int]  # Which directions are coherent
    boundary_radii: list[float]  # Boundary radius per direction
    mean_radius: float
    min_radius: float
    max_radius: float
    centroid: "Array"  # Starting point
    directions: "Array"  # Full orthonormal basis
    compression_ratio: float  # hidden_dim / n_coherent


def initialize_backend() -> "Backend":
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    return get_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
    )

    return model, tokenizer, config


def get_seed_activation(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    seed_text: str = "The",
) -> "Array":
    """Get a single seed activation from a known-good text."""
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    acts = provider.collect_hidden_activations(model, tokenizer, seed_text)

    if layer_idx not in acts:
        raise ValueError(f"Layer {layer_idx} not found in activations")

    activation = acts[layer_idx]
    mx.eval(activation)

    logger.info(
        "Seed activation from '%s' at layer %d: shape=%s",
        seed_text, layer_idx, activation.shape
    )

    return activation


def generate_complete_orthonormal_basis(
    hidden_dim: int,
    backend: "Backend",
    seed: int = 42,
) -> "Array":
    """Generate complete orthonormal basis for R^hidden_dim.

    This gives us ALL directions to probe - no sampling, no gaps.

    For efficiency, we use QR decomposition of a random matrix.
    The resulting Q is orthonormal and spans the full space.

    Returns:
        directions: [hidden_dim, hidden_dim] orthonormal matrix
    """
    b = backend
    b.random_seed(seed)

    # Generate random Gaussian matrix
    random_matrix = b.random_normal((hidden_dim, hidden_dim))
    b.eval(random_matrix)

    # QR gives orthonormal Q
    Q, R = b.qr(random_matrix)
    b.eval(Q)

    # Verify orthonormality (dot product of any two columns should be 0 or 1)
    # Skip in production for speed

    logger.info("Generated complete orthonormal basis: [%d, %d]", hidden_dim, hidden_dim)

    return Q


def is_valid_distribution(probs: "Array", backend: "Backend") -> bool:
    """Check if output is a valid probability distribution.

    A valid distribution:
    - All values in [0, 1]
    - Sums to 1.0 (within tolerance)

    If these fail, the output is semantically INCOHERENT.
    """
    import mlx.core as mx

    probs_arr = backend.array(probs)
    backend.eval(probs_arr)

    # Check bounds
    min_val = float(mx.min(probs_arr).item())
    max_val = float(mx.max(probs_arr).item())

    if min_val < -0.01 or max_val > 1.01:
        return False

    # Check sum
    total = float(mx.sum(probs_arr).item())
    if abs(total - 1.0) > 0.1:  # 10% tolerance
        return False

    return True


def find_validity_boundary(
    centroid: "Array",
    direction: "Array",
    forward_fn: Any,
    backend: "Backend",
    max_radius: float = 10.0,
    tolerance: float = 0.01,
) -> tuple[float, bool]:
    """Find radius where output becomes invalid distribution.

    Binary search for the boundary where forward pass produces
    numerically invalid output (not a probability distribution).

    Returns:
        Tuple of (boundary_radius, found_boundary)
    """
    import mlx.core as mx

    b = backend

    # First check if boundary exists at max_radius
    delta = direction * max_radius
    a_max = centroid + delta
    b.eval(a_max)

    try:
        y_max = forward_fn(a_max)
        b.eval(y_max)
    except Exception:
        # Forward pass failed - definitely outside manifold
        y_max = None

    if y_max is not None and is_valid_distribution(y_max, b):
        # Still valid at max_radius - no boundary found
        return max_radius, False

    # Binary search for boundary
    low = 0.0
    high = max_radius

    while high - low > tolerance:
        mid = (low + high) / 2.0

        delta = direction * mid
        a_mid = centroid + delta
        b.eval(a_mid)

        try:
            y_mid = forward_fn(a_mid)
            b.eval(y_mid)
            is_valid = is_valid_distribution(y_mid, b)
        except Exception:
            is_valid = False

        if is_valid:
            low = mid  # Still valid - move outward
        else:
            high = mid  # Invalid - move inward

    return low, True


def probe_single_direction(
    centroid: "Array",
    direction: "Array",
    direction_idx: int,
    forward_fn: Any,
    backend: "Backend",
    max_radius: float = 10.0,
    tolerance: float = 0.01,
    coherence_metric: str = "validity",
) -> DirectionResult:
    """Probe coherence boundary in a single direction.

    Args:
        coherence_metric: "validity" (default) checks if output is valid distribution.
                          "entropy", "magnitude_stability", "curvature" use traditional metrics.
    """
    from modelcypher.core.domain.geometry.manifold_boundary import (
        find_boundary_radius,
        measure_coherence,
    )

    b = backend

    if coherence_metric == "validity":
        # Use distribution validity check
        boundary_radius, found_boundary = find_validity_boundary(
            centroid=centroid,
            direction=direction,
            forward_fn=forward_fn,
            backend=b,
            max_radius=max_radius,
            tolerance=tolerance,
        )

        # Direction is coherent if we found a boundary (meaning it's finite)
        # Or if we didn't find one but radius is large (meaning stable direction)
        is_coherent = boundary_radius > tolerance

        return DirectionResult(
            direction_idx=direction_idx,
            boundary_radius=boundary_radius,
            is_coherent=is_coherent,
            baseline_coherence=1.0,  # Assumed valid at center
            boundary_coherence=0.0 if found_boundary else 1.0,
        )

    else:
        # Traditional metrics
        baseline = measure_coherence(
            activation=centroid,
            direction=direction,
            radius=0.001,
            forward_fn=forward_fn,
            backend=b,
            metric=coherence_metric,
        )

        result = find_boundary_radius(
            activation=centroid,
            direction=direction,
            forward_fn=forward_fn,
            backend=b,
            max_radius=max_radius,
            tolerance=tolerance,
            metric=coherence_metric,
        )

        is_coherent = result.boundary_radius > tolerance

        return DirectionResult(
            direction_idx=direction_idx,
            boundary_radius=result.boundary_radius,
            is_coherent=is_coherent,
            baseline_coherence=baseline.coherence,
            boundary_coherence=result.coherence_at_boundary,
        )


def flood_layer_manifold(
    model: Any,
    layer_idx: int,
    config: dict,
    centroid: "Array",
    backend: "Backend",
    max_radius: float = 10.0,
    tolerance: float | None = None,
    coherence_metric: str = "entropy",
    batch_size: int = 32,
) -> LayerManifoldResult:
    """Flood a layer with ALL directions to discover complete manifold.

    This is the core algorithm:
    1. Generate complete orthonormal basis
    2. For each direction, find coherence boundary
    3. Collect all coherent directions (radius > tolerance)
    4. Return the manifold basis

    Args:
        model: The model
        layer_idx: Layer to analyze
        config: Model config
        centroid: Starting activation [hidden_dim]
        backend: Backend for tensor ops
        max_radius: Maximum radius to probe
        tolerance: Coherence boundary tolerance (default: sqrt(eps))
        coherence_metric: "entropy" or "magnitude_stability"
        batch_size: Directions to process before logging progress

    Returns:
        LayerManifoldResult with complete manifold discovery
    """
    from modelcypher.core.domain.geometry.manifold_boundary import create_layer_forward_fn
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )

    b = backend

    hidden_dim = config.get("hidden_size", 0)

    # Derive tolerance from dtype if not specified
    if tolerance is None:
        eps = machine_epsilon(b, centroid)
        tolerance = sqrt_scalar(eps, b)
        tolerance = max(tolerance, 0.001)  # Minimum for binary search convergence

    logger.info(
        "FLOODING layer %d: hidden_dim=%d, max_radius=%.2f, tolerance=%.4f, metric=%s",
        layer_idx, hidden_dim, max_radius, tolerance, coherence_metric
    )

    # Generate complete orthonormal basis
    directions = generate_complete_orthonormal_basis(hidden_dim, b)

    # Create forward function: activation → softmax distribution
    forward_fn = create_layer_forward_fn(model, layer_idx, config, mode="full_model")

    # Probe ALL directions
    results: list[DirectionResult] = []
    n_coherent = 0

    start_time = time.time()

    for i in range(hidden_dim):
        direction = directions[:, i]
        b.eval(direction)

        result = probe_single_direction(
            centroid=centroid,
            direction=direction,
            direction_idx=i,
            forward_fn=forward_fn,
            backend=b,
            max_radius=max_radius,
            tolerance=tolerance,
            coherence_metric=coherence_metric,
        )

        results.append(result)
        if result.is_coherent:
            n_coherent += 1

        # Progress logging
        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (hidden_dim - i - 1) / rate
            logger.info(
                "  Probed %d/%d directions (%d coherent, %.1f dir/sec, ~%.0fs remaining)",
                i + 1, hidden_dim, n_coherent, rate, remaining
            )

    # Collect statistics
    boundary_radii = [r.boundary_radius for r in results]
    coherent_indices = [r.direction_idx for r in results if r.is_coherent]

    # Filter to only coherent radii for statistics
    coherent_radii = [r.boundary_radius for r in results if r.is_coherent]

    mean_radius = sum(coherent_radii) / len(coherent_radii) if coherent_radii else 0
    min_radius = min(coherent_radii) if coherent_radii else 0
    max_radius_found = max(coherent_radii) if coherent_radii else 0

    compression_ratio = hidden_dim / n_coherent if n_coherent > 0 else float('inf')

    elapsed = time.time() - start_time
    logger.info(
        "FLOOD COMPLETE layer %d: %d/%d coherent directions (%.1fx compression) in %.1fs",
        layer_idx, n_coherent, hidden_dim, compression_ratio, elapsed
    )
    logger.info(
        "  Radii: mean=%.3f, min=%.3f, max=%.3f",
        mean_radius, min_radius, max_radius_found
    )

    return LayerManifoldResult(
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        n_coherent_directions=n_coherent,
        coherent_indices=coherent_indices,
        boundary_radii=boundary_radii,
        mean_radius=mean_radius,
        min_radius=min_radius,
        max_radius=max_radius_found,
        centroid=centroid,
        directions=directions,
        compression_ratio=compression_ratio,
    )


def extract_manifold_basis(
    result: LayerManifoldResult,
    backend: "Backend",
) -> "Array":
    """Extract just the coherent directions as the manifold basis.

    Returns:
        P: [hidden_dim, n_coherent] projection matrix
    """
    b = backend

    if not result.coherent_indices:
        logger.warning("No coherent directions found!")
        return b.eye(result.hidden_dim)

    # Select coherent columns from directions
    P = result.directions[:, result.coherent_indices]
    b.eval(P)

    return P


def verify_manifold_completeness(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    manifold_basis: "Array",
    backend: "Backend",
    test_texts: list[str],
) -> tuple[float, float]:
    """Verify that manifold basis captures all activation variance.

    For each test text:
    1. Get activation
    2. Project to manifold: a_proj = P @ P^T @ a
    3. Compute reconstruction error: ||a - a_proj|| / ||a||

    If manifold is complete, reconstruction error should be < sqrt(eps).

    Returns:
        Tuple of (mean_reconstruction_error, max_reconstruction_error)
    """
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    b = backend
    P = manifold_basis

    provider = MLXActivationProvider(config=config, pooling="last")

    errors = []

    for text in test_texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            if layer_idx not in acts:
                continue

            a = b.array(acts[layer_idx])
            b.eval(a)

            # Project to manifold and back
            # a_proj = P @ (P^T @ a)
            a_in_manifold = b.matmul(b.transpose(P), b.reshape(a, (-1, 1)))
            b.eval(a_in_manifold)
            a_proj = b.matmul(P, a_in_manifold)
            a_proj = b.reshape(a_proj, (-1,))
            b.eval(a_proj)

            # Reconstruction error
            diff = a - a_proj
            b.eval(diff)

            error = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))
            norm = float(b.to_scalar(b.sqrt(b.sum(a * a))))

            rel_error = error / norm if norm > 0 else 0
            errors.append(rel_error)

        except Exception:
            pass

    if not errors:
        return 1.0, 1.0

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)

    return mean_error, max_error


def main():
    parser = argparse.ArgumentParser(
        description="Coherence flooding manifold discovery"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to analyze")
    parser.add_argument("--all-layers", action="store_true", help="Analyze all layers")
    parser.add_argument("--seed-text", type=str, default="The", help="Seed text for activation")
    parser.add_argument("--max-radius", type=float, default=10.0, help="Max probe radius")
    parser.add_argument("--metric", type=str, default="validity",
                       choices=["validity", "entropy", "magnitude_stability", "curvature"],
                       help="Coherence metric: 'validity' checks if output is valid distribution")
    parser.add_argument("--verify", action="store_true", help="Verify with held-out texts")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.layer is None and not args.all_layers:
        parser.error("Must specify --layer or --all-layers")

    # Initialize
    backend = initialize_backend()
    model, tokenizer, config = load_model(args.model)

    num_layers = config.get("num_hidden_layers", 0)
    hidden_dim = config.get("hidden_size", 0)

    # Determine which layers to analyze
    if args.all_layers:
        layers_to_analyze = list(range(num_layers))
    else:
        layers_to_analyze = [args.layer]

    logger.info("Will analyze %d layer(s): %s", len(layers_to_analyze), layers_to_analyze)
    logger.info("Hidden dimension: %d (will probe all %d directions per layer)",
                hidden_dim, hidden_dim)

    results: dict[int, LayerManifoldResult] = {}

    for layer_idx in layers_to_analyze:
        logger.info("\n" + "="*60)
        logger.info("LAYER %d", layer_idx)
        logger.info("="*60)

        # Get seed activation
        centroid = get_seed_activation(
            model, tokenizer, config, layer_idx, args.seed_text
        )

        # Flood the manifold
        result = flood_layer_manifold(
            model=model,
            layer_idx=layer_idx,
            config=config,
            centroid=centroid,
            backend=backend,
            max_radius=args.max_radius,
            coherence_metric=args.metric,
        )

        results[layer_idx] = result

        # Extract manifold basis
        P = extract_manifold_basis(result, backend)
        logger.info("Manifold basis: [%d, %d]", hidden_dim, result.n_coherent_directions)

        # Verify completeness if requested
        if args.verify:
            logger.info("\nVerifying manifold completeness...")
            test_texts = [
                "Hello world",
                "The quick brown fox",
                "Machine learning is",
                "In the beginning",
                "What is the meaning of",
            ]

            mean_err, max_err = verify_manifold_completeness(
                model, tokenizer, config, layer_idx, P, backend, test_texts
            )

            logger.info(
                "Reconstruction error: mean=%.6f%%, max=%.6f%%",
                mean_err * 100, max_err * 100
            )

            if max_err < 0.01:  # 1%
                logger.info("Manifold is COMPLETE (all activations reconstructable)")
            else:
                logger.warning(
                    "Manifold may be INCOMPLETE (%.2f%% max error)",
                    max_err * 100
                )

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    logger.info("Layer | Hidden | Coherent | Compression | Mean Radius")
    logger.info("-" * 55)

    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        logger.info(
            "%5d | %6d | %8d | %10.1fx | %.4f",
            layer_idx, r.hidden_dim, r.n_coherent_directions,
            r.compression_ratio, r.mean_radius
        )

    # Identify highway vs ramp
    if len(results) > 1:
        sorted_by_coherent = sorted(
            results.items(),
            key=lambda x: x[1].n_coherent_directions
        )

        highway_layers = [idx for idx, r in sorted_by_coherent[:3]]
        ramp_layers = [idx for idx, r in sorted_by_coherent[-3:]]

        logger.info("")
        logger.info("Highway layers (fewest coherent directions): %s", highway_layers)
        logger.info("Ramp layers (most coherent directions): %s", ramp_layers)

    # Save results
    if args.output:
        output_data = {}
        for layer_idx, r in results.items():
            output_data[layer_idx] = {
                "hidden_dim": r.hidden_dim,
                "n_coherent_directions": r.n_coherent_directions,
                "coherent_indices": r.coherent_indices,
                "boundary_radii": r.boundary_radii,
                "mean_radius": r.mean_radius,
                "min_radius": r.min_radius,
                "max_radius": r.max_radius,
                "compression_ratio": r.compression_ratio,
            }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("Results saved to %s", args.output)

    logger.info("\nDONE")


if __name__ == "__main__":
    main()
