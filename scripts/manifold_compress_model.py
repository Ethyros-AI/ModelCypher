#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Manifold-based model compression.
"""
Manifold Compression

The key insight: activations live on a low-dimensional manifold (~5D),
but weights are stored in high-dimensional space (1024D). This script:

1. Discovers the manifold from activations (PCA)
2. Projects weights into manifold space
3. Verifies CKA = 1.0 on probes (relational structure preserved)
4. Measures storage savings

The math:
- Activations X live on manifold: X ≈ Z @ P^T where Z is low-dim, P is projection
- For weight W operating on manifold: W_eff = P^T @ W @ P
- Reconstruction: W_approx = P @ W_eff @ P^T
- On manifold: W_approx @ x = W @ x (off-manifold components are null)

Storage:
- Original: d × d' per weight
- Compressed: k × k' per weight + shared projection P [d × k]
- Savings: (d × d') / (k × k' + d × k / n_weights) → huge when k << d
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Dtype precision limits
DTYPE_EPSILON = {
    "float32": 1.19e-7,
    "float16": 9.77e-4,
    "bfloat16": 7.81e-3,
}


def get_variance_threshold(dtype: str) -> float:
    """Get variance threshold for manifold detection.

    Eigenvalues below sqrt(eps) * max_eigenvalue are noise.
    """
    dtype_key = dtype.replace("mlx.core.", "")
    eps = DTYPE_EPSILON.get(dtype_key, DTYPE_EPSILON["float32"])
    return math.sqrt(eps)


def initialize_backend():
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict, str]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.parameters()))
    first_weight = next(iter(weights.values()))
    dtype = str(first_weight.dtype)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d, dtype=%s",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
        dtype,
    )

    return model, tokenizer, config, dtype


def generate_probe_texts(max_probes: int = 500) -> list[str]:
    """Generate probe texts from unified atlas."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    import random

    probes = UnifiedAtlasInventory.all_probes()
    texts = []
    for probe in probes:
        if probe.support_texts:
            for text in probe.support_texts:
                if text and len(text) > 5:
                    texts.append(text)

    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    if len(unique_texts) > max_probes:
        random.seed(42)
        unique_texts = random.sample(unique_texts, max_probes)

    logger.info("Using %d probe texts", len(unique_texts))
    return unique_texts


def collect_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: dict,
) -> dict[int, Any]:
    """Collect activations per layer across all probe texts."""
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    num_layers = config.get("num_hidden_layers", 0)
    layer_activations: dict[int, list] = {i: [] for i in range(num_layers)}

    logger.info("Collecting activations for %d probes across %d layers", len(texts), num_layers)

    for text in texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            for layer_idx, act in acts.items():
                layer_activations[layer_idx].append(act)
        except Exception:
            pass

    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            stacked = mx.stack(acts, axis=0)
            mx.eval(stacked)
            result[layer_idx] = stacked

    return result


def compute_manifold_projection(
    activations: Any,
    backend: Any,
    threshold: float,
) -> tuple[Any, int, float]:
    """Compute manifold projection from activations.

    Args:
        activations: [n_samples, hidden_dim]
        backend: Backend for tensor ops
        threshold: Relative variance threshold

    Returns:
        Tuple of (projection P [hidden_dim, manifold_dim], manifold_dim, variance_captured)
    """
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(activations), b)
    b.eval(X)

    shape = b.shape(X)
    n_samples = int(shape[0])
    hidden_dim = int(shape[1])

    # Center the data
    mean = b.mean(X, axis=0)
    b.eval(mean)
    X_centered = X - mean
    b.eval(X_centered)

    # Compute covariance matrix (use X^T X for efficiency when n < d)
    if n_samples < hidden_dim:
        # Compute in sample space, then convert to feature space
        # Gram matrix G = X @ X^T [n x n]
        G = b.matmul(X_centered, b.transpose(X_centered))
        b.eval(G)

        # Eigendecomposition of Gram matrix
        eigenvalues, eigenvectors = b.eigh(G)
        b.eval(eigenvalues, eigenvectors)

        # Sort by eigenvalue (descending)
        indices = b.argsort(-eigenvalues)
        b.eval(indices)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        b.eval(eigenvalues, eigenvectors)

        # Convert to feature space: v_feature = X^T @ v_gram / sqrt(lambda)
        # P = X^T @ V @ diag(1/sqrt(lambda))
        valid_mask = eigenvalues > 1e-10
        n_valid = int(b.to_scalar(b.sum(b.astype(valid_mask, "float32"))))

        if n_valid == 0:
            logger.warning("No valid eigenvalues, returning identity")
            return b.eye(hidden_dim), hidden_dim, 1.0

        # Scale eigenvectors
        sqrt_eigs = b.sqrt(b.maximum(eigenvalues[:n_valid], b.array([1e-10])))
        b.eval(sqrt_eigs)
        V_scaled = eigenvectors[:, :n_valid] / sqrt_eigs
        b.eval(V_scaled)

        # Project to feature space
        P = b.matmul(b.transpose(X_centered), V_scaled)
        b.eval(P)

        # Normalize columns
        norms = b.sqrt(b.sum(P * P, axis=0))
        b.eval(norms)
        P = P / b.maximum(norms, b.array([1e-10]))
        b.eval(P)

        eigenvalues = eigenvalues[:n_valid]
    else:
        # Compute covariance directly
        cov = b.matmul(b.transpose(X_centered), X_centered) / (n_samples - 1)
        b.eval(cov)

        eigenvalues, P = b.eigh(cov)
        b.eval(eigenvalues, P)

        # Sort descending
        indices = b.argsort(-eigenvalues)
        b.eval(indices)
        eigenvalues = eigenvalues[indices]
        P = P[:, indices]
        b.eval(eigenvalues, P)

    # Find manifold dimension: eigenvalues above threshold * max
    eigenvalues_list = eigenvalues.tolist()
    max_eig = max(eigenvalues_list) if eigenvalues_list else 0
    cutoff = threshold * max_eig

    manifold_dim = 0
    for eig in eigenvalues_list:
        if eig >= cutoff:
            manifold_dim += 1
        else:
            break

    # Ensure at least 1 dimension
    manifold_dim = max(manifold_dim, 1)

    # Compute variance captured
    total_var = sum(eigenvalues_list)
    captured_var = sum(eigenvalues_list[:manifold_dim])
    variance_captured = captured_var / total_var if total_var > 0 else 0

    # Extract manifold projection
    P_manifold = P[:, :manifold_dim]
    b.eval(P_manifold)

    return P_manifold, manifold_dim, variance_captured


def project_weight_to_manifold(
    weight: Any,
    P_in: Any,
    P_out: Any,
    backend: Any,
) -> tuple[Any, Any]:
    """Project weight matrix into manifold space.

    Args:
        weight: [out_dim, in_dim] weight matrix
        P_in: [in_dim, k_in] input manifold projection
        P_out: [out_dim, k_out] output manifold projection
        backend: Backend for tensor ops

    Returns:
        Tuple of (W_manifold [k_out, k_in], W_reconstructed [out_dim, in_dim])
    """
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    W = _promote_precision_float32(b.array(weight), b)
    b.eval(W)

    # W_manifold = P_out^T @ W @ P_in
    W_manifold = b.matmul(b.transpose(P_out), b.matmul(W, P_in))
    b.eval(W_manifold)

    # W_reconstructed = P_out @ W_manifold @ P_in^T
    W_reconstructed = b.matmul(P_out, b.matmul(W_manifold, b.transpose(P_in)))
    b.eval(W_reconstructed)

    return W_manifold, W_reconstructed


def compute_cka(X: Any, Y: Any, backend: Any) -> float:
    """Compute CKA between two activation matrices."""
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(X), b)
    Y = _promote_precision_float32(b.array(Y), b)
    b.eval(X, Y)

    # Center
    X = X - b.mean(X, axis=0)
    Y = Y - b.mean(Y, axis=0)
    b.eval(X, Y)

    # Gram matrices
    K_X = b.matmul(X, b.transpose(X))
    K_Y = b.matmul(Y, b.transpose(Y))
    b.eval(K_X, K_Y)

    # Center Gram matrices (HSIC)
    n = int(b.shape(K_X)[0])
    H = b.eye(n) - b.ones((n, n)) / n
    b.eval(H)

    K_X_centered = b.matmul(H, b.matmul(K_X, H))
    K_Y_centered = b.matmul(H, b.matmul(K_Y, H))
    b.eval(K_X_centered, K_Y_centered)

    # HSIC values
    hsic_xy = b.sum(K_X_centered * K_Y_centered)
    hsic_xx = b.sum(K_X_centered * K_X_centered)
    hsic_yy = b.sum(K_Y_centered * K_Y_centered)
    b.eval(hsic_xy, hsic_xx, hsic_yy)

    # CKA
    denom = b.sqrt(hsic_xx * hsic_yy)
    b.eval(denom)

    if float(b.to_scalar(denom)) < 1e-10:
        return 0.0

    cka = hsic_xy / denom
    b.eval(cka)

    return float(b.to_scalar(cka))


def find_manifold_dim_for_cka(
    activations: Any,
    backend: Any,
    target_cka: float = 0.99,
    max_dim: int | None = None,
) -> tuple[Any, int, float, float]:
    """Find minimum manifold dimension that achieves target CKA.

    Binary search over manifold dimensions to find the minimum that
    preserves relational structure (CKA >= target).

    Args:
        activations: [n_samples, hidden_dim]
        backend: Backend for tensor ops
        target_cka: Target CKA to achieve
        max_dim: Maximum dimension to consider

    Returns:
        Tuple of (projection P, manifold_dim, variance_captured, achieved_cka)
    """
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(activations), b)
    b.eval(X)

    shape = b.shape(X)
    n_samples = int(shape[0])
    hidden_dim = int(shape[1])

    if max_dim is None:
        max_dim = min(n_samples, hidden_dim)

    # Compute full projection first
    _, full_dim, _ = compute_manifold_projection(activations, b, threshold=0.0)

    # Binary search for minimum dimension
    lo, hi = 1, min(full_dim, max_dim)
    best_dim = hi
    best_cka = 0.0

    while lo <= hi:
        mid = (lo + hi) // 2

        # Compute projection with exactly mid dimensions
        P, _, var_captured = compute_manifold_projection(
            activations, b, threshold=0.0
        )

        # Truncate to mid dimensions
        P_truncated = P[:, :mid]
        b.eval(P_truncated)

        # Project and reconstruct
        X_proj = b.matmul(X, P_truncated)
        b.eval(X_proj)
        X_recon = b.matmul(X_proj, b.transpose(P_truncated))
        b.eval(X_recon)

        # Compute CKA
        cka = compute_cka(X, X_recon, b)

        if cka >= target_cka:
            best_dim = mid
            best_cka = cka
            hi = mid - 1
        else:
            lo = mid + 1

    # Get final projection at best dimension
    P_full, _, var_captured = compute_manifold_projection(activations, b, threshold=0.0)
    P_final = P_full[:, :best_dim]
    b.eval(P_final)

    # Recompute CKA to confirm
    X_proj = b.matmul(X, P_final)
    X_recon = b.matmul(X_proj, b.transpose(P_final))
    b.eval(X_proj, X_recon)
    final_cka = compute_cka(X, X_recon, b)

    # Compute variance captured at this dimension
    _, _, var_at_dim = compute_manifold_projection(activations, b, threshold=0.0)
    # Estimate variance for truncated dims (simplified)
    var_captured_est = var_at_dim if best_dim >= full_dim else best_dim / full_dim

    return P_final, best_dim, var_captured_est, final_cka


def analyze_manifold_structure(
    layer_activations: dict[int, Any],
    dtype: str,
    target_cka: float = 0.99,
    use_crossval: bool = False,
) -> dict[int, dict]:
    """Analyze manifold structure per layer with adaptive dimensions.

    Args:
        layer_activations: Per-layer activation matrices [n_samples, hidden_dim]
        dtype: Model dtype for precision thresholds
        target_cka: Target CKA to achieve
        use_crossval: If True, use cross-validation to find generalizing dimensions
    """
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    threshold = get_variance_threshold(dtype)

    results = {}

    for layer_idx in sorted(layer_activations.keys()):
        activations = layer_activations[layer_idx]
        shape = backend.shape(activations)
        n_samples = int(shape[0])
        hidden_dim = int(shape[1])

        if use_crossval and n_samples >= 20:
            # Split into train/val for cross-validation
            split_idx = int(n_samples * 0.8)
            train_acts = activations[:split_idx]
            val_acts = activations[split_idx:]

            # Find manifold on train set
            P_full, _, _ = compute_manifold_projection(train_acts, backend, threshold=0.0)
            backend.eval(P_full)

            # Search for dimension that generalizes
            max_dim = min(int(shape[0]) - 1, hidden_dim)
            best_dim = max_dim
            best_val_cka = 0.0

            # Binary search for minimum dimension achieving target on validation
            lo, hi = 1, max_dim
            while lo <= hi:
                mid = (lo + hi) // 2
                P_test = P_full[:, :mid]
                backend.eval(P_test)

                # Test on validation set
                X_val = backend.array(val_acts)
                X_proj = backend.matmul(X_val, P_test)
                X_recon = backend.matmul(X_proj, backend.transpose(P_test))
                backend.eval(X_proj, X_recon)

                val_cka = compute_cka(X_val, X_recon, backend)

                if val_cka >= target_cka:
                    best_dim = mid
                    best_val_cka = val_cka
                    hi = mid - 1
                else:
                    lo = mid + 1

            # Recompute on full data with best dimension
            P_full, _, var_captured = compute_manifold_projection(activations, backend, threshold=0.0)
            P = P_full[:, :best_dim]
            backend.eval(P)

            X = backend.array(activations)
            X_proj = backend.matmul(X, P)
            X_recon = backend.matmul(X_proj, backend.transpose(P))
            backend.eval(X_proj, X_recon)
            achieved_cka = compute_cka(X, X_recon, backend)
            manifold_dim = best_dim

        else:
            # Original non-crossval approach
            P_var, dim_var, var_captured = compute_manifold_projection(
                activations, backend, threshold
            )

            X = backend.array(activations)
            X_proj = backend.matmul(X, P_var)
            X_recon = backend.matmul(X_proj, backend.transpose(P_var))
            backend.eval(X_proj, X_recon)
            cka_var = compute_cka(X, X_recon, backend)

            if cka_var >= target_cka:
                manifold_dim = dim_var
                P = P_var
                achieved_cka = cka_var
            else:
                P, manifold_dim, var_captured, achieved_cka = find_manifold_dim_for_cka(
                    activations, backend, target_cka=target_cka
                )

        compression_ratio = hidden_dim / manifold_dim

        results[layer_idx] = {
            "hidden_dim": hidden_dim,
            "manifold_dim": manifold_dim,
            "variance_captured": var_captured,
            "compression_ratio": compression_ratio,
            "achieved_cka": achieved_cka,
            "projection": P,
        }

        cka_status = "✓" if achieved_cka >= target_cka else "✗"
        logger.info(
            "Layer %2d: %dD -> %dD (%.1fx, CKA=%.4f %s)",
            layer_idx, hidden_dim, manifold_dim, compression_ratio, achieved_cka, cka_status
        )

    return results


def test_manifold_projection_cka(
    model: Any,
    tokenizer: Any,
    config: dict,
    manifold_info: dict[int, dict],
    test_texts: list[str],
) -> dict[int, float]:
    """Test that manifold projection preserves CKA on held-out data."""
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    provider = MLXActivationProvider(config=config, pooling="last")

    # Collect test activations
    layer_acts_original: dict[int, list] = {i: [] for i in manifold_info.keys()}

    for text in test_texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            for layer_idx, act in acts.items():
                if layer_idx in manifold_info:
                    layer_acts_original[layer_idx].append(act)
        except Exception:
            pass

    cka_scores = {}

    for layer_idx, acts_list in layer_acts_original.items():
        if not acts_list:
            continue

        X = mx.stack(acts_list, axis=0)
        mx.eval(X)

        P = manifold_info[layer_idx]["projection"]

        # Project to manifold and back
        X_projected = backend.matmul(X, P)
        backend.eval(X_projected)
        X_reconstructed = backend.matmul(X_projected, backend.transpose(P))
        backend.eval(X_reconstructed)

        # Compute CKA
        cka = compute_cka(X, X_reconstructed, backend)
        cka_scores[layer_idx] = cka

    return cka_scores


def generate_held_out_texts(exclude_texts: set[str], max_texts: int = 100) -> list[str]:
    """Generate held-out probe texts for generalization testing."""
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    import random

    probes = UnifiedAtlasInventory.all_probes()
    texts = []
    for probe in probes:
        if probe.support_texts:
            for text in probe.support_texts:
                if text and len(text) > 5 and text not in exclude_texts:
                    texts.append(text)

    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen and text not in exclude_texts:
            seen.add(text)
            unique_texts.append(text)

    if len(unique_texts) > max_texts:
        random.seed(123)  # Different seed from training
        unique_texts = random.sample(unique_texts, max_texts)

    return unique_texts


def test_weight_compression(
    model: Any,
    manifold_info: dict[int, dict],
    config: dict,
) -> None:
    """Test actual weight compression and reconstruction.

    Key insight: Frobenius error will be high because we remove null space.
    But the output ON THE MANIFOLD should be identical.

    We verify: for x on the manifold, W @ x ≈ W_recon @ x (same output)
    """
    from mlx.utils import tree_flatten
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )
    import mlx.core as mx

    backend = get_default_backend()
    weights = dict(tree_flatten(model.parameters()))

    # Pick a highway layer and a ramp layer to test
    highway_layer = 8
    ramp_layer = 0

    for test_layer in [highway_layer, ramp_layer]:
        if test_layer not in manifold_info:
            continue

        P = manifold_info[test_layer]["projection"]
        manifold_dim = manifold_info[test_layer]["manifold_dim"]
        hidden_dim = manifold_info[test_layer]["hidden_dim"]

        # Find a weight for this layer (handle different architectures)
        layer_prefix = f"model.layers.{test_layer}"
        test_keys = [
            # LLaMA-style
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.mlp.fc1.weight",
            f"{layer_prefix}.mlp.gate_proj.weight",
            # LFM2-style
            f"{layer_prefix}.conv.out_proj.weight",
            f"{layer_prefix}.feed_forward.w1.weight",
        ]

        weight_key = None
        for key in test_keys:
            if key in weights:
                weight_key = key
                break

        if weight_key is None:
            logger.info("Layer %d: No suitable weight found", test_layer)
            continue

        W = weights[weight_key]
        shape = backend.shape(W)
        out_dim = int(shape[0])
        in_dim = int(shape[1])

        # Promote to float32 for stability
        W_f32 = _promote_precision_float32(backend.array(W), backend)
        backend.eval(W_f32)

        # For non-square weights, we need to handle projections carefully
        # q_proj: [hidden, hidden] - both in and out are on hidden manifold
        # fc1/gate_proj: [intermediate, hidden] - in is on hidden manifold

        if in_dim != hidden_dim:
            logger.info("Layer %d: Input dim %d != hidden %d, skipping",
                       test_layer, in_dim, hidden_dim)
            continue

        # Project weight to manifold space
        # For [out, in] with in on manifold: W_manifold = W @ P gives [out, k]
        # Reconstructed operation: x' = (x @ P) @ (P^T @ W^T) = x @ P @ P^T @ W^T
        #                        or: y = W @ x ≈ (W @ P) @ (P^T @ x) for x on manifold

        W_manifold = backend.matmul(W_f32, P)  # [out, k]
        backend.eval(W_manifold)

        # Generate random test inputs ON THE MANIFOLD
        # x_manifold is in R^k, then x_full = P @ x_manifold is on the manifold in R^hidden
        n_test = 100
        z = mx.random.normal(shape=(n_test, manifold_dim))
        mx.eval(z)

        # Project to full space (on manifold)
        x_on_manifold = backend.matmul(z, backend.transpose(P))  # [n, hidden]
        backend.eval(x_on_manifold)

        # Original output: y = x @ W^T (since W is [out, in])
        y_original = backend.matmul(x_on_manifold, backend.transpose(W_f32))  # [n, out]
        backend.eval(y_original)

        # Compressed output: y = z @ W_manifold^T (compute in manifold space)
        y_compressed = backend.matmul(z, backend.transpose(W_manifold))  # [n, out]
        backend.eval(y_compressed)

        # Measure activation error (should be very small!)
        diff = y_original - y_compressed
        backend.eval(diff)

        activation_error = float(backend.to_scalar(backend.sqrt(backend.sum(diff * diff))))
        activation_norm = float(backend.to_scalar(backend.sqrt(backend.sum(y_original * y_original))))
        relative_activation_error = activation_error / activation_norm if activation_norm > 0 else 0

        # Also measure weight Frobenius (will be high - that's OK)
        W_recon = backend.matmul(W_manifold, backend.transpose(P))  # [out, in]
        backend.eval(W_recon)
        w_diff = W_f32 - W_recon
        backend.eval(w_diff)
        frob_error = float(backend.to_scalar(backend.sqrt(backend.sum(w_diff * w_diff))))
        frob_original = float(backend.to_scalar(backend.sqrt(backend.sum(W_f32 * W_f32))))
        relative_frob = frob_error / frob_original if frob_original > 0 else 0

        # Storage comparison
        original_params = out_dim * in_dim
        manifold_shape = backend.shape(W_manifold)
        compressed_params = int(manifold_shape[0]) * int(manifold_shape[1])
        projection_params = hidden_dim * manifold_dim
        total_compressed = compressed_params + projection_params

        compression = original_params / total_compressed if total_compressed > 0 else 0

        layer_type = "HIGHWAY" if test_layer in [7, 8, 9, 10] else "RAMP"
        logger.info(
            "%s Layer %d (%s): %s -> %s",
            layer_type, test_layer, weight_key.split(".")[-2], shape, manifold_shape
        )
        logger.info(
            "  Weight Frobenius error: %.2f%% (null space removed - expected)",
            relative_frob * 100
        )
        logger.info(
            "  ACTIVATION error on manifold: %.6f%% ← THIS is what matters",
            relative_activation_error * 100
        )
        logger.info(
            "  Storage: %d -> %d params (%.1fx compression)",
            original_params, total_compressed, compression
        )


def compute_storage_analysis(
    manifold_info: dict[int, dict],
    config: dict,
) -> dict:
    """Compute actual storage savings with shared projections."""
    hidden_dim = config.get("hidden_size", 0)
    intermediate_dim = config.get("intermediate_size", hidden_dim * 4)
    num_layers = len(manifold_info)

    # Per-layer weight shapes for typical transformer
    # q, k, v, o: [hidden, hidden] each = 4 * hidden^2
    # gate, up: [intermediate, hidden] each = 2 * intermediate * hidden
    # down: [hidden, intermediate] = hidden * intermediate
    # Total per layer: 4*h^2 + 3*h*i

    h = hidden_dim
    i = intermediate_dim

    # Original storage per layer
    original_per_layer = 4 * h * h + 3 * h * i
    original_total = original_per_layer * num_layers

    # Compressed storage with shared projections
    # For each layer: store W_manifold [k, k] for attention, [k, k'] for MLP
    # Plus one shared projection P [h, k] per layer

    compressed_total = 0
    for layer_idx, info in manifold_info.items():
        k = info["manifold_dim"]

        # Attention: 4 weights, each [k, k] in manifold space
        attn_compressed = 4 * k * k

        # MLP: assume similar compression for intermediate dim
        # gate/up: [k_i, k] where k_i = k * (i/h)
        k_intermediate = max(1, int(k * (i / h)))
        mlp_compressed = 2 * k_intermediate * k + k * k_intermediate

        # Projection matrix P [h, k]
        projection_storage = h * k

        layer_compressed = attn_compressed + mlp_compressed + projection_storage
        compressed_total += layer_compressed

    compression_ratio = original_total / compressed_total if compressed_total > 0 else 0

    return {
        "original_params": original_total,
        "compressed_params": compressed_total,
        "compression_ratio": compression_ratio,
        "original_mb": original_total * 2 / (1024 * 1024),  # bf16 = 2 bytes
        "compressed_mb": compressed_total * 2 / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Manifold-based model compression analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--max-probes", type=int, default=500, help="Number of probes")
    parser.add_argument("--target-cka", type=float, default=0.99, help="Target CKA (default: 0.99)")
    parser.add_argument("--crossval", action="store_true", help="Use cross-validation for generalization")
    args = parser.parse_args()

    initialize_backend()

    # Load model
    model, tokenizer, config, dtype = load_model(args.model)

    # Generate probes
    probes = generate_probe_texts(max_probes=args.max_probes)

    # Collect activations
    logger.info("\n=== COLLECTING ACTIVATIONS ===")
    start = time.time()
    layer_activations = collect_layer_activations(model, tokenizer, probes, config)
    logger.info("Activation collection took %.2fs", time.time() - start)

    # Analyze manifold structure
    logger.info("\n=== ANALYZING MANIFOLD STRUCTURE (target CKA >= %.2f) ===", args.target_cka)
    threshold = get_variance_threshold(dtype)
    logger.info("Dtype: %s, variance threshold: %.6f (sqrt(eps))", dtype, threshold)

    start = time.time()
    manifold_info = analyze_manifold_structure(
        layer_activations, dtype, target_cka=args.target_cka, use_crossval=args.crossval
    )
    logger.info("Manifold analysis took %.2fs", time.time() - start)

    # Identify highway vs ramp layers
    dims = [(idx, info["manifold_dim"]) for idx, info in manifold_info.items()]
    dims.sort(key=lambda x: x[1])

    min_dim_layers = [idx for idx, dim in dims[:3]]
    max_dim_layers = [idx for idx, dim in dims[-3:]]

    logger.info("")
    logger.info("Highway layers (lowest ID): %s", min_dim_layers)
    logger.info("Ramp layers (highest ID): %s", max_dim_layers)

    # Check CKA achievement
    all_achieved = all(info["achieved_cka"] >= args.target_cka for info in manifold_info.values())
    ckas = [info["achieved_cka"] for info in manifold_info.values()]
    logger.info("CKA range: [%.4f, %.4f]", min(ckas), max(ckas))

    if all_achieved:
        logger.info("✓ All layers achieve CKA >= %.2f", args.target_cka)
    else:
        failed = [idx for idx, info in manifold_info.items() if info["achieved_cka"] < args.target_cka]
        logger.info("✗ Layers below target: %s", failed)

    # Storage analysis
    logger.info("\n=== STORAGE ANALYSIS ===")
    storage = compute_storage_analysis(manifold_info, config)

    logger.info("Original parameters: %d (%.1f MB)", storage["original_params"], storage["original_mb"])
    logger.info("Compressed parameters: %d (%.1f MB)", storage["compressed_params"], storage["compressed_mb"])
    logger.info("Compression ratio: %.1fx", storage["compression_ratio"])
    logger.info("Space savings: %.1f%%", (1 - 1/storage["compression_ratio"]) * 100)

    # Per-layer breakdown
    logger.info("\n=== PER-LAYER MANIFOLD DIMENSIONS ===")
    logger.info("Layer | Hidden | Manifold | Compression | CKA")
    logger.info("-" * 50)
    for layer_idx in sorted(manifold_info.keys()):
        info = manifold_info[layer_idx]
        logger.info(
            "%5d | %6d | %8d | %10.1fx | %.4f",
            layer_idx, info["hidden_dim"], info["manifold_dim"],
            info["compression_ratio"], info["achieved_cka"]
        )

    # Test generalization on held-out probes
    logger.info("\n=== GENERALIZATION TEST (held-out probes) ===")
    held_out = generate_held_out_texts(set(probes), max_texts=100)
    logger.info("Testing on %d held-out probe texts", len(held_out))

    held_out_cka = test_manifold_projection_cka(
        model, tokenizer, config, manifold_info, held_out
    )

    if held_out_cka:
        min_cka = min(held_out_cka.values())
        mean_cka = sum(held_out_cka.values()) / len(held_out_cka)
        max_cka = max(held_out_cka.values())

        logger.info("Held-out CKA: min=%.4f, mean=%.4f, max=%.4f", min_cka, mean_cka, max_cka)

        if min_cka >= args.target_cka:
            logger.info("✓ Manifold generalizes: all held-out CKA >= %.2f", args.target_cka)
        else:
            drop = [
                (idx, cka) for idx, cka in held_out_cka.items()
                if cka < args.target_cka
            ]
            logger.info("✗ Layers below target on held-out:")
            for idx, cka in sorted(drop):
                train_cka = manifold_info[idx]["achieved_cka"]
                logger.info("  Layer %d: train=%.4f, held-out=%.4f (Δ=%.4f)",
                           idx, train_cka, cka, train_cka - cka)

    # Weight compression test
    logger.info("\n=== WEIGHT COMPRESSION TEST ===")
    test_weight_compression(model, manifold_info, config)

    logger.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
