#!/usr/bin/env python3
"""Scaling Investigation on Real Model Activations.

Uses actual LLM activations to validate the synthetic experiment findings.
Tests the hypothesis: behavioral_preserved should be low (~1%) with real data.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from modelcypher.backends.mlx_backend import MLXBackend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_pinv,
    machine_epsilon,
    svd_rank_threshold,
)


@dataclass
class ProjectionResult:
    """Result of projecting delta through null-space."""
    method: str
    delta_norm_before: float
    delta_norm_after: float
    preserved_fraction: float
    behavioral_before: float
    behavioral_after: float
    behavioral_preserved: float
    null_rank: int
    activation_rank: int
    in_dim: int


def compute_true_null_projection(delta_W, input_activations, backend):
    """Project delta using TRUE null-space projection."""
    b = backend
    n, d = input_activations.shape

    # Compute Gram matrix
    AAt = b.matmul(input_activations, b.transpose(input_activations))
    b.eval(AAt)

    # Get eigenvalues for rank determination
    eigvals = b.eigvalsh(AAt)
    b.eval(eigvals)
    idx = b.argsort(-eigvals, axis=0)
    eigvals = b.take(eigvals, idx, axis=0)
    b.eval(eigvals)

    # Rank threshold
    eps = machine_epsilon(b, AAt)
    max_eig = float(b.to_scalar(b.max(eigvals)))
    rank_scale = svd_rank_threshold(b, eigvals, d)
    rank_threshold = max_eig * rank_scale

    rank_mask = eigvals > rank_threshold
    activation_rank = int(b.to_scalar(b.sum(b.astype(rank_mask, "float32"))))
    null_rank = max(0, d - activation_rank)

    # Compute pseudoinverse
    AAt_inv = geodesic_pinv(b, AAt)
    b.eval(AAt_inv)

    # Project
    delta_row = b.matmul(delta_W, b.transpose(input_activations))
    correction = b.matmul(delta_row, AAt_inv)
    correction = b.matmul(correction, input_activations)
    delta_proj = delta_W - correction
    b.eval(delta_proj)

    # Compute norms
    eps_div = float(division_epsilon(b, delta_W))

    delta_norm_before = float(b.to_scalar(b.sqrt(b.sum(delta_W * delta_W))))
    delta_norm_after = float(b.to_scalar(b.sqrt(b.sum(delta_proj * delta_proj))))
    preserved_fraction = delta_norm_after / max(delta_norm_before, eps_div)

    output_before = b.matmul(input_activations, b.transpose(delta_W))
    output_after = b.matmul(input_activations, b.transpose(delta_proj))
    b.eval(output_before, output_after)

    behavioral_before = float(b.to_scalar(b.sqrt(b.sum(output_before * output_before))))
    behavioral_after = float(b.to_scalar(b.sqrt(b.sum(output_after * output_after))))
    behavioral_preserved = behavioral_after / max(behavioral_before, eps_div)

    return delta_proj, ProjectionResult(
        method="true_null",
        delta_norm_before=delta_norm_before,
        delta_norm_after=delta_norm_after,
        preserved_fraction=preserved_fraction,
        behavioral_before=behavioral_before,
        behavioral_after=behavioral_after,
        behavioral_preserved=behavioral_preserved,
        null_rank=null_rank,
        activation_rank=activation_rank,
        in_dim=d,
    )


def compute_variance_weighted_projection(delta_W, input_activations, density_weights, backend):
    """Project delta using variance-weighted null-space."""
    b = backend
    n, d = input_activations.shape

    constraint_weights = 1.0 - density_weights
    eps = division_epsilon(b, constraint_weights)
    sqrt_weights = b.sqrt(constraint_weights + eps)
    A_weighted = input_activations * b.reshape(sqrt_weights, (-1, 1))
    b.eval(A_weighted)

    AAt = b.matmul(A_weighted, b.transpose(A_weighted))
    b.eval(AAt)

    eigvals = b.eigvalsh(AAt)
    b.eval(eigvals)
    idx = b.argsort(-eigvals, axis=0)
    eigvals = b.take(eigvals, idx, axis=0)
    b.eval(eigvals)

    eps_m = machine_epsilon(b, AAt)
    max_eig = float(b.to_scalar(b.max(eigvals)))
    if max_eig > eps_m:
        rank_scale = svd_rank_threshold(b, eigvals, d)
        rank_threshold = max_eig * rank_scale
        rank_mask = eigvals > rank_threshold
        activation_rank = int(b.to_scalar(b.sum(b.astype(rank_mask, "float32"))))
    else:
        activation_rank = 0
    null_rank = max(0, d - activation_rank)

    AAt_inv = geodesic_pinv(b, AAt)
    b.eval(AAt_inv)

    delta_row = b.matmul(delta_W, b.transpose(A_weighted))
    correction = b.matmul(delta_row, AAt_inv)
    correction = b.matmul(correction, A_weighted)
    delta_proj = delta_W - correction
    b.eval(delta_proj)

    eps_div = float(division_epsilon(b, delta_W))

    delta_norm_before = float(b.to_scalar(b.sqrt(b.sum(delta_W * delta_W))))
    delta_norm_after = float(b.to_scalar(b.sqrt(b.sum(delta_proj * delta_proj))))
    preserved_fraction = delta_norm_after / max(delta_norm_before, eps_div)

    # Behavioral on UNWEIGHTED activations
    output_before = b.matmul(input_activations, b.transpose(delta_W))
    output_after = b.matmul(input_activations, b.transpose(delta_proj))
    b.eval(output_before, output_after)

    behavioral_before = float(b.to_scalar(b.sqrt(b.sum(output_before * output_before))))
    behavioral_after = float(b.to_scalar(b.sqrt(b.sum(output_after * output_after))))
    behavioral_preserved = behavioral_after / max(behavioral_before, eps_div)

    return delta_proj, ProjectionResult(
        method="variance_weighted",
        delta_norm_before=delta_norm_before,
        delta_norm_after=delta_norm_after,
        preserved_fraction=preserved_fraction,
        behavioral_before=behavioral_before,
        behavioral_after=behavioral_after,
        behavioral_preserved=behavioral_preserved,
        null_rank=null_rank,
        activation_rank=activation_rank,
        in_dim=d,
    )


def compute_density_weights(activations, backend, method="variance"):
    """Compute density weights."""
    b = backend
    n = int(activations.shape[0])

    if method == "variance":
        sample_norms = b.sum(activations * activations, axis=1)
        b.eval(sample_norms)
        total = b.sum(sample_norms)
        density = sample_norms / (total + 1e-10)
        b.eval(density)
        density = density / (b.max(density) + 1e-10)
        b.eval(density)
    else:
        density = b.ones((n,)) / n
        b.eval(density)

    return density


def collect_activations_from_model(model, tokenizer, prompts, backend):
    """Collect hidden state activations from a real model."""
    b = backend
    all_activations = {}

    for prompt in prompts:
        tokens = mx.array(tokenizer.encode(prompt))
        layer_outputs = []

        def make_hook(idx):
            def hook(module, inputs, outputs):
                layer_outputs.append((idx, outputs))
            return hook

        hooks = []
        for i, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)

        try:
            logits = model(tokens[None])
            mx.eval(logits)
        finally:
            for hook in hooks:
                hook.remove()

        for layer_idx, output in layer_outputs:
            hidden = output[0]  # [seq_len, hidden_dim]
            mx.eval(hidden)

            if layer_idx not in all_activations:
                all_activations[layer_idx] = []
            all_activations[layer_idx].append(hidden)

    # Concatenate
    for layer_idx in all_activations:
        all_activations[layer_idx] = mx.concatenate(all_activations[layer_idx], axis=0)
        mx.eval(all_activations[layer_idx])

    return all_activations


def get_weight_delta(source_weights, target_weights, layer_idx, weight_type="q_proj"):
    """Get the weight delta between source and target for a specific layer."""
    # Find matching weight keys
    source_key = None
    target_key = None

    for key in source_weights:
        if f"layers.{layer_idx}." in key and weight_type in key and "weight" in key:
            source_key = key
            break

    for key in target_weights:
        if f"layers.{layer_idx}." in key and weight_type in key and "weight" in key:
            target_key = key
            break

    if source_key is None or target_key is None:
        return None, None, None

    source_w = source_weights[source_key]
    target_w = target_weights[target_key]

    # Check shapes match
    if source_w.shape != target_w.shape:
        return None, None, None

    delta = source_w - target_w
    return delta, source_key, target_key


def main():
    """Run scaling investigation on real model activations."""
    from mlx_lm import load as mlx_load

    logger.info("=" * 60)
    logger.info("SCALING INVESTIGATION - REAL MODEL ACTIVATIONS")
    logger.info("=" * 60)

    # Use small models for testing
    model_candidates = [
        "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Math-1.5B-bf16",
    ]

    model_path = None
    for candidate in model_candidates:
        if Path(candidate).exists():
            model_path = candidate
            break

    if model_path is None:
        logger.error(f"No model found. Tried: {model_candidates}")
        return

    logger.info(f"Model: {model_path}")

    backend = MLXBackend()

    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = mlx_load(model_path)

    # Get model config
    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    logger.info(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Diverse test prompts for activation collection
    test_prompts = [
        "The capital of France is",
        "In mathematics, the derivative of x squared is",
        "The process of photosynthesis converts",
        "Machine learning models learn by",
        "The theory of relativity explains",
        "Water boils at a temperature of",
        "The largest planet in our solar system is",
        "DNA stands for deoxyribonucleic",
        "The speed of light is approximately",
        "Artificial intelligence can be used for",
        "The human brain contains billions of",
        "Climate change is caused by",
        "Programming languages like Python are",
        "The periodic table organizes elements by",
        "Neural networks are inspired by",
        "The moon orbits the Earth every",
    ]

    # Collect activations
    logger.info(f"\nCollecting activations from {len(test_prompts)} prompts...")
    activations = collect_activations_from_model(model, tokenizer, test_prompts, backend)

    # Get weights for delta computation (we'll use target as both source and target
    # with synthetic perturbation since we only have one model)
    weights = dict(model.parameters())

    # Results storage
    all_results = []

    # Analyze each layer
    layers_to_analyze = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layers_to_analyze = [l for l in layers_to_analyze if l < n_layers]

    for layer_idx in layers_to_analyze:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")

        if layer_idx not in activations:
            logger.warning(f"No activations for layer {layer_idx}")
            continue

        layer_acts = backend.array(activations[layer_idx])
        n_samples, d_features = layer_acts.shape
        logger.info(f"Activations: {n_samples} samples, {d_features} features")

        # Measure intrinsic dimension
        id_estimator = IntrinsicDimension(backend)
        id_result = id_estimator.compute(layer_acts)
        logger.info(f"Intrinsic dimension: {id_result.intrinsic_dimension:.2f}")

        # Create a realistic delta by perturbing weights
        # Find a weight matrix for this layer
        weight_key = None
        for key in weights:
            if f"layers.{layer_idx}." in key and "self_attn.q_proj.weight" in key:
                weight_key = key
                break

        if weight_key is None:
            # Try other weight types
            for key in weights:
                if f"layers.{layer_idx}." in key and "weight" in key:
                    weight_key = key
                    break

        if weight_key is None:
            logger.warning(f"No weight found for layer {layer_idx}")
            continue

        target_weight = backend.array(weights[weight_key])
        out_dim, in_dim = target_weight.shape
        logger.info(f"Weight: {weight_key} [{out_dim}, {in_dim}]")

        # Create synthetic delta (simulating what merge would produce)
        # Low-rank perturbation typical of fine-tuning differences
        delta_rank = max(1, min(in_dim // 10, 50))  # ~10% of dimensions
        U = mx.random.normal(shape=(out_dim, delta_rank)) * 0.01
        V = mx.random.normal(shape=(in_dim, delta_rank)) * 0.01
        mx.eval(U, V)
        delta_W = mx.matmul(U, mx.transpose(V))
        mx.eval(delta_W)
        delta_W = backend.array(delta_W)

        # Scale delta to be realistic (small fraction of weight norm)
        target_norm = float(backend.to_scalar(backend.sqrt(backend.sum(target_weight * target_weight))))
        delta_norm = float(backend.to_scalar(backend.sqrt(backend.sum(delta_W * delta_W))))
        scale = 0.1 * target_norm / (delta_norm + 1e-10)  # 10% of weight magnitude
        delta_W = delta_W * scale
        backend.eval(delta_W)

        logger.info(f"Delta: rank={delta_rank}, scale={scale:.4f}")

        # Get input activations (need to match weight input dimension)
        # For most weights, input is hidden_dim
        if in_dim == d_features:
            input_acts = layer_acts
        else:
            # Skip if dimensions don't match
            logger.warning(f"Dimension mismatch: acts={d_features}, weight_in={in_dim}")
            continue

        # True null-space projection
        logger.info("\n--- True Null-Space Projection ---")
        _, true_result = compute_true_null_projection(delta_W, input_acts, backend)
        logger.info(f"  Frobenius preserved: {100*true_result.preserved_fraction:.2f}%")
        logger.info(f"  Behavioral preserved: {100*true_result.behavioral_preserved:.4f}%")
        logger.info(f"  Null rank: {true_result.null_rank}/{in_dim} ({100*true_result.null_rank/in_dim:.1f}%)")
        logger.info(f"  Activation rank: {true_result.activation_rank}")

        # Variance-weighted projection
        logger.info("\n--- Variance-Weighted Projection ---")
        density = compute_density_weights(input_acts, backend, method="variance")
        _, var_result = compute_variance_weighted_projection(delta_W, input_acts, density, backend)
        logger.info(f"  Frobenius preserved: {100*var_result.preserved_fraction:.2f}%")
        logger.info(f"  Behavioral preserved: {100*var_result.behavioral_preserved:.4f}%")
        logger.info(f"  Null rank: {var_result.null_rank}/{in_dim} ({100*var_result.null_rank/in_dim:.1f}%)")

        # Analysis
        logger.info("\n--- Analysis ---")
        behavioral_eliminated = 100 * (1 - var_result.behavioral_preserved)
        logger.info(f"Behavioral ELIMINATED: {behavioral_eliminated:.2f}%")

        null_ratio = var_result.null_rank / in_dim
        effective_load = var_result.behavioral_preserved / max(null_ratio, 1e-10)
        logger.info(f"Null-space ratio: {100*null_ratio:.1f}%")
        logger.info(f"Effective load: {effective_load:.6f}")

        if effective_load > 1.0:
            derived_scale = 1.0 / effective_load
            logger.info(f"*** NEED delta_scale={derived_scale:.4f} ***")
        else:
            derived_scale = 1.0
            logger.info(f"delta_scale=1.0 is valid (load={effective_load:.6f} < 1.0)")

        all_results.append({
            "layer": layer_idx,
            "n_samples": n_samples,
            "hidden_dim": d_features,
            "intrinsic_dim": id_result.intrinsic_dimension,
            "weight_shape": [out_dim, in_dim],
            "delta_rank": delta_rank,
            "true_null": {
                "frobenius_preserved": true_result.preserved_fraction,
                "behavioral_preserved": true_result.behavioral_preserved,
                "null_rank": true_result.null_rank,
                "activation_rank": true_result.activation_rank,
            },
            "variance_weighted": {
                "frobenius_preserved": var_result.preserved_fraction,
                "behavioral_preserved": var_result.behavioral_preserved,
                "null_rank": var_result.null_rank,
            },
            "analysis": {
                "behavioral_eliminated_pct": behavioral_eliminated,
                "null_ratio": null_ratio,
                "effective_load": effective_load,
                "derived_delta_scale": derived_scale,
            }
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY - REAL MODEL RESULTS")
    logger.info("=" * 60)

    logger.info(f"\nModel: {model_path}")
    logger.info(f"Prompts: {len(test_prompts)}")

    logger.info("\nPer-layer results:")
    for r in all_results:
        layer = r["layer"]
        beh_elim = r["analysis"]["behavioral_eliminated_pct"]
        null_ratio = r["analysis"]["null_ratio"]
        eff_load = r["analysis"]["effective_load"]
        scale = r["analysis"]["derived_delta_scale"]
        id_val = r["intrinsic_dim"]
        logger.info(
            f"  L{layer:2d}: ID={id_val:5.1f}, null={100*null_ratio:5.1f}%, "
            f"beh_elim={beh_elim:6.2f}%, load={eff_load:.2e}, scale={scale:.4f}"
        )

    # Overall conclusion
    all_scales = [r["analysis"]["derived_delta_scale"] for r in all_results]
    all_loads = [r["analysis"]["effective_load"] for r in all_results]
    max_load = max(all_loads) if all_loads else 0

    logger.info(f"\nMax effective load across layers: {max_load:.2e}")
    if max_load < 1.0:
        logger.info("CONCLUSION: delta_scale=1.0 is VALID for all layers")
    else:
        min_scale = min(all_scales)
        logger.info(f"CONCLUSION: Need delta_scale <= {min_scale:.4f}")

    # Save results
    output_path = Path("experiments/results/scaling_investigation_real.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model": model_path,
            "n_prompts": len(test_prompts),
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "layers": all_results,
            "conclusion": {
                "max_effective_load": max_load,
                "recommended_delta_scale": min(all_scales) if all_scales else 1.0,
                "delta_scale_1_valid": max_load < 1.0,
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
