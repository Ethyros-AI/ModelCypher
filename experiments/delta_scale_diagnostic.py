#!/usr/bin/env python3
"""Diagnostic script to understand delta_scale corruption.

This script investigates WHY delta_scale=1.0 causes repetition when the
geometry (null-space projection) says it should be safe.

Key questions:
1. What does the eigenvalue spectrum of the Gram matrix look like?
2. How much of delta survives the null-space projection?
3. Where does the corruption manifest (which layers/weights)?
4. Can we derive the correct delta_scale from the geometry?
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Import ModelCypher components
from modelcypher.backends.mlx_backend import MLXBackend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    svd_rank_threshold,
)


def analyze_gram_spectrum(activations, backend, layer_name=""):
    """Analyze the eigenvalue spectrum of the Gram matrix."""
    b = backend
    n, d = activations.shape

    # Build Gram matrix
    AAt = b.matmul(activations, b.transpose(activations))
    b.eval(AAt)

    # Get eigenvalues
    eigvals = b.eigvalsh(AAt)
    b.eval(eigvals)

    # Sort descending
    idx = b.argsort(-eigvals, axis=0)
    eigvals = b.take(eigvals, idx, axis=0)
    b.eval(eigvals)

    # Convert to numpy for analysis
    eigvals_np = b.to_numpy(eigvals)

    # Compute statistics
    eps = float(machine_epsilon(b, AAt))
    total_var = float(eigvals_np.sum())
    max_eig = float(eigvals_np[0])
    min_eig = float(eigvals_np[-1])

    # Rank determination
    rank_scale = svd_rank_threshold(b, eigvals, d)
    rank_threshold = max_eig * rank_scale

    # Count eigenvalues above threshold
    above_threshold = (eigvals_np > rank_threshold).sum()

    # Compute condition number
    nonzero_eigvals = eigvals_np[eigvals_np > eps]
    if len(nonzero_eigvals) > 0:
        condition_number = nonzero_eigvals[0] / nonzero_eigvals[-1]
    else:
        condition_number = float('inf')

    # Compute energy distribution
    cumsum = eigvals_np.cumsum()
    energy_50 = (cumsum < 0.5 * total_var).sum() + 1  # dims for 50% energy
    energy_90 = (cumsum < 0.9 * total_var).sum() + 1  # dims for 90% energy
    energy_99 = (cumsum < 0.99 * total_var).sum() + 1  # dims for 99% energy

    # Intrinsic dimension
    id_estimator = IntrinsicDimension(b)
    id_result = id_estimator.compute(activations)
    intrinsic_dim = id_result.intrinsic_dimension

    return {
        "layer": layer_name,
        "n_samples": n,
        "d_features": d,
        "total_variance": total_var,
        "max_eigenvalue": max_eig,
        "min_eigenvalue": min_eig,
        "condition_number": condition_number,
        "numeric_rank": int(above_threshold),
        "null_rank": d - int(above_threshold),
        "intrinsic_dimension": intrinsic_dim,
        "energy_50_dims": int(energy_50),
        "energy_90_dims": int(energy_90),
        "energy_99_dims": int(energy_99),
        "rank_threshold": float(rank_threshold),
        "eigenvalue_spectrum": eigvals_np[:20].tolist(),  # Top 20
    }


def compute_projection_analysis(delta_W, input_activations, backend):
    """Analyze what happens when we project delta into null-space."""
    b = backend

    # Build Gram matrix and pseudoinverse
    AAt = b.matmul(input_activations, b.transpose(input_activations))
    b.eval(AAt)

    AAt_inv = b.pinv(AAt)
    b.eval(AAt_inv)

    # Compute projection: delta_proj = delta - (delta @ A.T) @ (A @ A.T)^+ @ A
    delta_row = b.matmul(delta_W, b.transpose(input_activations))
    correction = b.matmul(delta_row, AAt_inv)
    correction = b.matmul(correction, input_activations)
    delta_proj = delta_W - correction
    b.eval(delta_proj)

    # Compute norms
    delta_norm = float(b.to_scalar(b.sqrt(b.sum(delta_W * delta_W))))
    proj_norm = float(b.to_scalar(b.sqrt(b.sum(delta_proj * delta_proj))))
    correction_norm = float(b.to_scalar(b.sqrt(b.sum(correction * correction))))

    # What fraction survives?
    preserved_fraction = proj_norm / delta_norm if delta_norm > 0 else 0.0

    # What's the angle between original and projected delta?
    dot_product = float(b.to_scalar(b.sum(delta_W * delta_proj)))
    cosine_sim = dot_product / (delta_norm * proj_norm) if (delta_norm > 0 and proj_norm > 0) else 0.0

    return {
        "delta_norm": delta_norm,
        "projected_norm": proj_norm,
        "correction_norm": correction_norm,
        "preserved_fraction": preserved_fraction,
        "cosine_similarity": cosine_sim,
    }


def collect_activations(model, tokenizer, prompts, backend):
    """Collect hidden state activations from model."""
    activations = {}

    for prompt in prompts:
        tokens = mx.array(tokenizer.encode(prompt))

        # Hook to capture hidden states
        layer_outputs = []

        def make_hook(idx):
            def hook(module, inputs, outputs):
                # outputs is the hidden state after this layer
                layer_outputs.append((idx, outputs))
            return hook

        # Register hooks on transformer layers
        hooks = []
        for i, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)

        # Run forward pass
        try:
            logits = model(tokens[None])
            mx.eval(logits)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Process captured outputs
        for layer_idx, output in layer_outputs:
            # output shape: [1, seq_len, hidden_dim]
            # Take mean across sequence for this analysis
            hidden = output[0]  # [seq_len, hidden_dim]
            mx.eval(hidden)

            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(hidden)

    # Concatenate activations per layer
    for layer_idx in activations:
        activations[layer_idx] = mx.concatenate(activations[layer_idx], axis=0)
        mx.eval(activations[layer_idx])

    return activations


def main():
    """Run diagnostic analysis."""
    from mlx_lm import load as mlx_load

    # Models to analyze
    source_path = "/path/to/models/mlx-community/LFM2-700M-bf16"
    target_path = "/path/to/models/mlx-community/LFM2-350M-MLX-bf16"

    logger.info("=" * 60)
    logger.info("DELTA SCALE DIAGNOSTIC")
    logger.info("=" * 60)
    logger.info(f"Source: {source_path}")
    logger.info(f"Target: {target_path}")

    # Initialize backend
    backend = MLXBackend()

    # Load models using mlx_lm
    logger.info("\nLoading models...")
    target_model, target_tokenizer = mlx_load(target_path)

    # Test prompts for activation collection
    test_prompts = [
        "The capital of France is",
        "In mathematics, a derivative measures",
        "The process of photosynthesis involves",
        "Machine learning algorithms can",
        "The theory of relativity states that",
    ]

    # Collect activations
    logger.info("\nCollecting target activations...")
    target_activations = collect_activations(target_model, target_tokenizer, test_prompts, backend)

    # Analyze spectrum for each layer
    logger.info("\n" + "=" * 60)
    logger.info("EIGENVALUE SPECTRUM ANALYSIS")
    logger.info("=" * 60)

    results = {
        "target_path": target_path,
        "layers": {}
    }

    for layer_idx in sorted(target_activations.keys()):
        logger.info(f"\n--- Layer {layer_idx} ---")

        tgt_acts = target_activations[layer_idx]

        # Convert to backend array
        tgt_acts_backend = backend.array(tgt_acts)

        # Analyze target spectrum (this is what we project into)
        tgt_spectrum = analyze_gram_spectrum(tgt_acts_backend, backend, f"target_L{layer_idx}")

        logger.info(f"Target spectrum:")
        logger.info(f"  Samples: {tgt_spectrum['n_samples']}, Features: {tgt_spectrum['d_features']}")
        logger.info(f"  Intrinsic dim: {tgt_spectrum['intrinsic_dimension']:.2f}")
        logger.info(f"  Numeric rank: {tgt_spectrum['numeric_rank']} / {tgt_spectrum['d_features']}")
        logger.info(f"  Null rank: {tgt_spectrum['null_rank']}")
        logger.info(f"  Condition number: {tgt_spectrum['condition_number']:.2e}")
        logger.info(f"  Energy dims (50/90/99%): {tgt_spectrum['energy_50_dims']}/{tgt_spectrum['energy_90_dims']}/{tgt_spectrum['energy_99_dims']}")
        logger.info(f"  Top eigenvalues: {[f'{v:.2e}' for v in tgt_spectrum['eigenvalue_spectrum'][:5]]}")

        results["layers"][str(layer_idx)] = {
            "target_spectrum": tgt_spectrum,
        }

    # Save results
    output_path = Path("/path/to/experiments/delta_scale_diagnostic_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n\nResults saved to: {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 60)

    for layer_idx, layer_data in results["layers"].items():
        spec = layer_data["target_spectrum"]
        null_ratio = spec["null_rank"] / spec["d_features"]
        logger.info(f"Layer {layer_idx}: null_ratio={null_ratio:.1%}, condition={spec['condition_number']:.2e}, ID={spec['intrinsic_dimension']:.1f}")

    logger.info("\n" + "=" * 60)
    logger.info("HYPOTHESIS")
    logger.info("=" * 60)
    logger.info("""
If the condition number is very high, the pseudoinverse is numerically unstable.
If null_rank is much smaller than d_features, most directions are "active".
If intrinsic_dimension << d_features, the manifold is low-dimensional.

The corruption likely comes from:
1. Numerical instability in pseudoinverse (high condition number)
2. "Null" directions that aren't truly unused (they affect generation dynamics)
3. Accumulated error across many layers

Geometry-derived delta_scale should account for:
- Condition number of the projection
- Ratio of null_rank to total dimensions
- Curvature of the manifold at the operating point
""")


if __name__ == "__main__":
    main()
