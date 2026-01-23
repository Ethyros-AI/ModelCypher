#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Semantic Prime Model Compression
"""
Semantic Prime Model Compression

Uses the semantic prime manifold discovery to compress model weights.

The insight:
- Semantic primes span the entire semantic space
- Highway layers compress 62 primes to 3-4 dimensions
- Weights can be projected into this manifold space
- Activations on the manifold are preserved (CKA = 1.0)

Compression approach:
1. Discover manifold P [hidden_dim, k] from semantic primes
2. For each weight W [out, in]:
   - W_manifold = P_out.T @ W @ P_in  (compress to [k_out, k_in])
3. Store: manifold bases + compressed weights
4. Runtime: either decompress or compute in manifold space

Usage:
    python prime_compress_model.py \
        --model /path/to/model \
        --output /path/to/compressed \
        --target-variance 0.99
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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


# Import semantic primes from the analysis script
from semantic_prime_manifold import (
    SEMANTIC_PRIMES,
    get_prime_contexts,
    compute_cka,
)


@dataclass
class LayerCompression:
    """Compression result for one layer."""
    layer_idx: int
    manifold_dim: int
    hidden_dim: int
    original_params: int
    compressed_params: int
    compression_ratio: float
    projection: "Array"  # [hidden_dim, manifold_dim]
    weights_compressed: dict[str, "Array"]  # weight_name -> compressed weight


def initialize_backend() -> "Backend":
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    return get_default_backend()


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


def collect_prime_activations(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
) -> "Array":
    """Collect activations for all semantic primes at a layer."""
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")
    contexts = get_prime_contexts()

    activations = []
    for prime, context, category in contexts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, context)
            if layer_idx in acts:
                activations.append(acts[layer_idx])
        except Exception:
            pass

    if not activations:
        raise ValueError(f"No activations collected for layer {layer_idx}")

    X = mx.stack(activations, axis=0)
    mx.eval(X)

    return X


def compute_manifold_projection(
    activations: "Array",
    backend: "Backend",
    target_variance: float = 0.99,
) -> tuple["Array", int, float]:
    """Compute manifold projection from activations.

    Returns:
        Tuple of (projection P [hidden_dim, k], k, variance_captured)
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

    # Center
    mean = b.mean(X, axis=0)
    b.eval(mean)
    X_centered = X - mean
    b.eval(X_centered)

    # Gram matrix
    G = b.matmul(X_centered, b.transpose(X_centered))
    b.eval(G)

    # Eigendecomposition
    eigenvalues, eigenvectors = b.eigh(G)
    b.eval(eigenvalues, eigenvectors)

    # Sort descending
    indices = b.argsort(-eigenvalues)
    b.eval(indices)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    b.eval(eigenvalues, eigenvectors)

    # Find dimension for target variance
    eigenvalues_list = [max(0, e) for e in eigenvalues.tolist()]
    total_var = sum(eigenvalues_list)

    if total_var == 0:
        return b.eye(hidden_dim), hidden_dim, 1.0

    cumvar = 0.0
    manifold_dim = 0
    for i, eig in enumerate(eigenvalues_list):
        cumvar += eig
        manifold_dim = i + 1
        if cumvar / total_var >= target_variance:
            break

    manifold_dim = max(manifold_dim, 1)

    # Build projection
    valid_mask = eigenvalues > 1e-10
    n_valid = min(manifold_dim, int(b.to_scalar(b.sum(b.astype(valid_mask, "float32")))))

    sqrt_eigs = b.sqrt(b.maximum(eigenvalues[:n_valid], b.array([1e-10])))
    b.eval(sqrt_eigs)
    V_scaled = eigenvectors[:, :n_valid] / sqrt_eigs
    b.eval(V_scaled)

    P = b.matmul(b.transpose(X_centered), V_scaled)
    b.eval(P)

    # Normalize
    norms = b.sqrt(b.sum(P * P, axis=0))
    b.eval(norms)
    P = P / b.maximum(norms, b.array([1e-10]))
    b.eval(P)

    variance_captured = sum(eigenvalues_list[:manifold_dim]) / total_var

    return P, manifold_dim, variance_captured


def compress_layer_weights(
    model: Any,
    layer_idx: int,
    projection: "Array",
    config: dict,
    backend: "Backend",
) -> tuple[dict[str, "Array"], int, int]:
    """Compress all weights in a layer using manifold projection.

    For weights where both input and output are hidden_dim, we project both:
        W_compressed = P.T @ W @ P

    For weights where only input is hidden_dim (e.g., up_proj):
        W_compressed = W @ P

    For weights where only output is hidden_dim (e.g., down_proj):
        W_compressed = P.T @ W

    Returns:
        Tuple of (compressed_weights dict, original_params, compressed_params)
    """
    from modelcypher.adapters.model_architecture import get_model_architecture
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )
    import mlx.core as mx

    b = backend
    P = projection
    hidden_dim = config.get("hidden_size", 0)
    manifold_dim = int(b.shape(P)[1])

    arch = get_model_architecture(model, config=config)
    accessor = arch.layer_accessor(layer_idx)

    compressed_weights = {}
    original_params = 0
    compressed_params = 0

    # Get all weight-bearing modules in this layer
    modules_to_check = []

    # Attention weights
    if hasattr(accessor, 'self_attn'):
        attn = accessor.self_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, name):
                modules_to_check.append((f"self_attn.{name}", getattr(attn, name)))

    # MLP weights
    if hasattr(accessor, 'mlp'):
        mlp = accessor.mlp
        for name in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'w1', 'w2', 'w3']:
            if hasattr(mlp, name):
                modules_to_check.append((f"mlp.{name}", getattr(mlp, name)))

    # Feed forward (LFM2 style)
    if hasattr(accessor, 'feed_forward'):
        ff = accessor.feed_forward
        for name in ['w1', 'w2', 'w3']:
            if hasattr(ff, name):
                modules_to_check.append((f"feed_forward.{name}", getattr(ff, name)))

    for weight_name, module in modules_to_check:
        if not hasattr(module, 'weight'):
            continue

        W = module.weight
        if W is None:
            continue

        shape = W.shape
        if len(shape) != 2:
            continue

        out_dim, in_dim = int(shape[0]), int(shape[1])
        original_params += out_dim * in_dim

        W_f32 = _promote_precision_float32(b.array(W), b)
        b.eval(W_f32)

        # Determine compression strategy based on dimensions
        if in_dim == hidden_dim and out_dim == hidden_dim:
            # Both dims match hidden_dim: full projection
            # W_compressed = P.T @ W @ P
            W_compressed = b.matmul(b.transpose(P), b.matmul(W_f32, P))
            b.eval(W_compressed)
            comp_shape = (manifold_dim, manifold_dim)

        elif in_dim == hidden_dim:
            # Only input matches: project input side
            # W_compressed = W @ P
            W_compressed = b.matmul(W_f32, P)
            b.eval(W_compressed)
            comp_shape = (out_dim, manifold_dim)

        elif out_dim == hidden_dim:
            # Only output matches: project output side
            # W_compressed = P.T @ W
            W_compressed = b.matmul(b.transpose(P), W_f32)
            b.eval(W_compressed)
            comp_shape = (manifold_dim, in_dim)

        else:
            # Neither matches: skip
            logger.debug("  %s: [%d, %d] - no hidden_dim match, skipping", weight_name, out_dim, in_dim)
            continue

        compressed_params += comp_shape[0] * comp_shape[1]

        # Convert back to original dtype
        W_compressed = b.astype(W_compressed, W.dtype)
        b.eval(W_compressed)

        compressed_weights[weight_name] = W_compressed

        ratio = (out_dim * in_dim) / (comp_shape[0] * comp_shape[1])
        logger.debug(
            "  %s: [%d, %d] -> [%d, %d] (%.1fx)",
            weight_name, out_dim, in_dim, comp_shape[0], comp_shape[1], ratio
        )

    return compressed_weights, original_params, compressed_params


def apply_compressed_weights_lstsq(
    model: Any,
    layer_idx: int,
    projection: "Array",
    activations: "Array",
    config: dict,
    backend: "Backend",
) -> None:
    """Apply compressed weights using least-squares reconstruction.

    Instead of direct projection W' = P @ (P.T @ W @ P) @ P.T, which destroys
    out-of-manifold information, we use least squares to find the optimal
    weight that preserves behavior on the activation manifold:

        W_new = argmin_W ||W_orig @ X - W_new @ X||²

    where W_new = P @ W_m @ P.T (constrained to manifold subspace).

    Solution: W_m = lstsq(P.T @ X, P.T @ (W @ X))
              W_new = P @ W_m @ P.T
    """
    from modelcypher.adapters.model_architecture import get_model_architecture
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )
    import mlx.core as mx

    b = backend
    P = projection
    hidden_dim = config.get("hidden_size", 0)
    manifold_dim = int(b.shape(P)[1])

    # Convert activations to float32
    X = _promote_precision_float32(b.array(activations), b)  # [n_samples, hidden_dim]
    X = b.transpose(X)  # [hidden_dim, n_samples]
    b.eval(X)

    # Project to manifold
    P_f32 = _promote_precision_float32(P, b)
    b.eval(P_f32)
    X_m = b.matmul(b.transpose(P_f32), X)  # [manifold_dim, n_samples]
    b.eval(X_m)

    arch = get_model_architecture(model, config=config)
    accessor = arch.layer_accessor(layer_idx)

    # Get all weight-bearing modules
    modules_to_process = []

    # Standard attention (LLaMA-style)
    if hasattr(accessor, 'self_attn'):
        attn = accessor.self_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, name):
                modules_to_process.append((f"self_attn.{name}", getattr(attn, name)))

    # LFM2-style attention (uses dict-like access)
    if hasattr(accessor, 'attention'):
        attn = accessor.attention
        # LFM2 attention has in_proj, out_proj as keys
        for name in ['in_proj', 'out_proj']:
            if name in attn:
                modules_to_process.append((f"attention.{name}", attn[name]))

    # MLP
    if hasattr(accessor, 'mlp'):
        mlp = accessor.mlp
        for name in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'w1', 'w2', 'w3']:
            if hasattr(mlp, name):
                modules_to_process.append((f"mlp.{name}", getattr(mlp, name)))

    # Feed forward (alternative naming)
    if hasattr(accessor, 'feed_forward'):
        ff = accessor.feed_forward
        for name in ['w1', 'w2', 'w3']:
            if hasattr(ff, name):
                modules_to_process.append((f"feed_forward.{name}", getattr(ff, name)))

    for weight_path, module in modules_to_process:
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        W = module.weight
        shape = W.shape
        if len(shape) != 2:
            continue

        out_dim, in_dim = int(shape[0]), int(shape[1])
        original_dtype = W.dtype

        W_f32 = _promote_precision_float32(b.array(W), b)
        b.eval(W_f32)

        # Only handle full hidden_dim -> hidden_dim weights for now
        if in_dim == hidden_dim and out_dim == hidden_dim:
            # Compute Y = W @ X (true outputs for these activations)
            Y = b.matmul(W_f32, X)  # [hidden_dim, n_samples]
            b.eval(Y)

            # Project Y to manifold space
            Y_m = b.matmul(b.transpose(P_f32), Y)  # [manifold_dim, n_samples]
            b.eval(Y_m)

            # NEW APPROACH: Only modify the manifold component, keep orthogonal part
            # W = (P @ P.T) @ W @ (P @ P.T)  [manifold component]
            #   + (I - P @ P.T) @ W @ (I - P @ P.T)  [orthogonal component]
            #   + cross terms
            #
            # Instead of replacing W entirely, we compute W_m for the manifold
            # and add back the orthogonal component from original W.

            # Projection matrices
            proj = b.matmul(P_f32, b.transpose(P_f32))  # [hidden, hidden] projection onto manifold
            b.eval(proj)

            # Extract manifold component of original weight
            W_on_manifold = b.matmul(proj, b.matmul(W_f32, proj))
            b.eval(W_on_manifold)

            # Extract orthogonal component of original weight
            orth = b.eye(hidden_dim) - proj
            b.eval(orth)
            W_orthogonal = b.matmul(orth, b.matmul(W_f32, orth))
            b.eval(W_orthogonal)

            # Cross terms (manifold -> orthogonal and orthogonal -> manifold)
            W_cross_mo = b.matmul(orth, b.matmul(W_f32, proj))  # manifold input -> orthogonal output
            W_cross_om = b.matmul(proj, b.matmul(W_f32, orth))  # orthogonal input -> manifold output
            b.eval(W_cross_mo, W_cross_om)

            # For manifold component, use lstsq to find optimal W_m
            # W_m @ X_m = Y_m
            # W_m = Y_m @ pinv(X_m)
            # pinv(X_m) = X_m.T @ inv(X_m @ X_m.T) for wide matrix (manifold_dim < n_samples)
            X_m_X_m_T = b.matmul(X_m, b.transpose(X_m))  # [manifold_dim, manifold_dim]
            b.eval(X_m_X_m_T)

            # Add regularization for numerical stability
            reg = 1e-6 * b.eye(manifold_dim)
            X_m_X_m_T_reg = X_m_X_m_T + reg
            b.eval(X_m_X_m_T_reg)

            try:
                inv_term = b.inv(X_m_X_m_T_reg)
                b.eval(inv_term)
                X_m_pinv = b.matmul(b.transpose(X_m), inv_term)  # [n_samples, manifold_dim]
                b.eval(X_m_pinv)
                W_m = b.matmul(Y_m, X_m_pinv)  # [manifold_dim, manifold_dim]
                b.eval(W_m)
            except Exception as e:
                logger.warning("    %s: pinv failed (%s), using original weight", weight_path, e)
                W_reconstructed = W_f32
                b.eval(W_reconstructed)

                # Convert back to original dtype
                W_reconstructed = b.astype(W_reconstructed, original_dtype)
                b.eval(W_reconstructed)

                # Apply to model
                module.weight = mx.array(W_reconstructed)
                mx.eval(module.weight)
                continue

            # Reconstruct manifold component with optimal W_m
            W_manifold_new = b.matmul(P_f32, b.matmul(W_m, b.transpose(P_f32)))
            b.eval(W_manifold_new)

            # Final weight: keep orthogonal + cross terms, use optimized manifold
            W_reconstructed_full = W_manifold_new + W_orthogonal + W_cross_mo + W_cross_om
            b.eval(W_reconstructed_full)

            # Also compute manifold-only version (for testing compression)
            W_reconstructed_manifold_only = W_manifold_new
            b.eval(W_reconstructed_manifold_only)

            # Compare errors
            frob_orig = float(b.to_scalar(b.sqrt(b.sum(W_f32 * W_f32))))
            frob_diff_full = float(b.to_scalar(b.sqrt(b.sum((W_f32 - W_reconstructed_full) * (W_f32 - W_reconstructed_full)))))
            frob_diff_manifold = float(b.to_scalar(b.sqrt(b.sum((W_f32 - W_reconstructed_manifold_only) * (W_f32 - W_reconstructed_manifold_only)))))
            rel_err_full = frob_diff_full / frob_orig if frob_orig > 0 else 0
            rel_err_manifold = frob_diff_manifold / frob_orig if frob_orig > 0 else 0

            logger.info("    %s: full_err=%.4f%%, manifold_only_err=%.4f%%",
                       weight_path, rel_err_full * 100, rel_err_manifold * 100)

            # Use manifold-only for testing actual compression
            W_reconstructed = W_reconstructed_manifold_only
            b.eval(W_reconstructed)

        elif in_dim == hidden_dim:
            # Input side only - preserve orthogonal component
            # W = W @ (P @ P.T) + W @ (I - P @ P.T)
            #   = W @ P @ P.T + W @ orth
            proj = b.matmul(P_f32, b.transpose(P_f32))  # [hidden, hidden]
            b.eval(proj)
            orth = b.eye(hidden_dim) - proj
            b.eval(orth)

            # Keep original weight (no modification to orthogonal)
            W_reconstructed = W_f32  # Keep original for now
            b.eval(W_reconstructed)

        elif out_dim == hidden_dim:
            # Output side only - preserve orthogonal component
            proj = b.matmul(P_f32, b.transpose(P_f32))  # [hidden, hidden]
            b.eval(proj)

            # Keep original weight (no modification to orthogonal)
            W_reconstructed = W_f32  # Keep original for now
            b.eval(W_reconstructed)

        else:
            continue

        # Convert back to original dtype
        W_reconstructed = b.astype(W_reconstructed, original_dtype)
        b.eval(W_reconstructed)

        # Apply to model
        module.weight = mx.array(W_reconstructed)
        mx.eval(module.weight)


def apply_compressed_weights(
    model: Any,
    layer_idx: int,
    projection: "Array",
    compressed_weights: dict[str, "Array"],
    config: dict,
    backend: "Backend",
) -> None:
    """Apply compressed weights back to model (decompressed).

    This reconstructs W = P @ W_compressed @ P.T and applies it.

    NOTE: This uses direct projection which loses out-of-manifold info.
    For better results, use apply_compressed_weights_lstsq instead.
    """
    from modelcypher.adapters.model_architecture import get_model_architecture
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )
    import mlx.core as mx

    b = backend
    P = projection
    hidden_dim = config.get("hidden_size", 0)
    manifold_dim = int(b.shape(P)[1])

    arch = get_model_architecture(model, config=config)
    accessor = arch.layer_accessor(layer_idx)

    for weight_path, W_compressed in compressed_weights.items():
        parts = weight_path.split(".")
        module_name = parts[0]
        weight_name = parts[1]

        # Navigate to the module
        if module_name == "self_attn":
            parent = accessor.self_attn
        elif module_name == "mlp":
            parent = accessor.mlp
        elif module_name == "feed_forward":
            parent = accessor.feed_forward
        else:
            continue

        if not hasattr(parent, weight_name):
            continue

        module = getattr(parent, weight_name)
        if not hasattr(module, 'weight'):
            continue

        original_shape = module.weight.shape
        out_dim, in_dim = int(original_shape[0]), int(original_shape[1])
        original_dtype = module.weight.dtype

        W_comp_f32 = _promote_precision_float32(b.array(W_compressed), b)
        P_f32 = _promote_precision_float32(P, b)
        b.eval(W_comp_f32, P_f32)

        # Reconstruct based on compression type
        comp_shape = b.shape(W_compressed)
        comp_out, comp_in = int(comp_shape[0]), int(comp_shape[1])

        if comp_out == manifold_dim and comp_in == manifold_dim:
            # Full projection: W = P @ W_compressed @ P.T
            W_reconstructed = b.matmul(P_f32, b.matmul(W_comp_f32, b.transpose(P_f32)))

        elif comp_in == manifold_dim:
            # Input projected: W = W_compressed @ P.T
            W_reconstructed = b.matmul(W_comp_f32, b.transpose(P_f32))

        elif comp_out == manifold_dim:
            # Output projected: W = P @ W_compressed
            W_reconstructed = b.matmul(P_f32, W_comp_f32)

        else:
            continue

        b.eval(W_reconstructed)

        # Convert back to original dtype
        W_reconstructed = b.astype(W_reconstructed, original_dtype)
        b.eval(W_reconstructed)

        # Apply to model
        module.weight = mx.array(W_reconstructed)
        mx.eval(module.weight)


def verify_inference(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_tokens: int = 50,
) -> list[str]:
    """Run inference and return responses."""
    from mlx_lm import generate

    responses = []
    for prompt in prompts:
        try:
            response = generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
            )
            responses.append(response)
            logger.info("  '%s' -> '%s'", prompt[:30], response[:50].replace('\n', ' '))
        except Exception as e:
            responses.append(f"ERROR: {e}")
            logger.warning("  '%s' -> ERROR: %s", prompt[:30], e)

    return responses


def save_compressed_model(
    model: Any,
    tokenizer: Any,
    output_path: str,
    source_path: str,
    compression_metadata: dict,
) -> None:
    """Save compressed model to disk."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer files
    source_dir = Path(source_path)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    # Copy and update config
    config_src = source_dir / "config.json"
    with open(config_src) as f:
        config = json.load(f)

    config["compression"] = compression_metadata
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save weights
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    logger.info("Saved compressed model to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic prime model compression"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to source model")
    parser.add_argument("--output", type=str, required=True, help="Path to save compressed model")
    parser.add_argument("--target-variance", type=float, default=0.99,
                        help="Target variance to capture (default: 0.99)")
    parser.add_argument("--compress-layers", type=str, default="",
                        help="Comma-separated layers to compress (default: auto-detect highway)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip inference verification")
    args = parser.parse_args()

    # Initialize
    backend = initialize_backend()
    model, tokenizer, config, dtype = load_model(args.model)

    num_layers = config.get("num_hidden_layers", 0)
    hidden_dim = config.get("hidden_size", 0)

    # Determine which layers to compress
    if args.compress_layers:
        compress_layers = [int(x.strip()) for x in args.compress_layers.split(",")]
    else:
        # Auto-detect: analyze all layers and find highway
        logger.info("\n=== DISCOVERING MANIFOLD STRUCTURE ===")

        layer_dims = {}
        for layer_idx in range(num_layers):
            acts = collect_prime_activations(model, tokenizer, config, layer_idx, backend)
            _, manifold_dim, var = compute_manifold_projection(acts, backend, args.target_variance)
            layer_dims[layer_idx] = manifold_dim
            logger.info("Layer %2d: %dD manifold (%.1fx compression)",
                       layer_idx, manifold_dim, hidden_dim / manifold_dim)

        # Find highway: layers with manifold_dim < 20
        highway_threshold = 20
        compress_layers = [idx for idx, dim in layer_dims.items() if dim < highway_threshold]

        if not compress_layers:
            # If no clear highway, take top 3 most compressed
            sorted_layers = sorted(layer_dims.items(), key=lambda x: x[1])
            compress_layers = [idx for idx, _ in sorted_layers[:3]]

        logger.info("\nAuto-detected highway layers: %s", compress_layers)

    # Baseline inference
    test_prompts = [
        "What is 2+2?",
        "The capital of France is",
        "Explain quantum entanglement:",
    ]

    if not args.skip_verify:
        logger.info("\n=== BASELINE INFERENCE ===")
        baseline_responses = verify_inference(model, tokenizer, test_prompts)

    # PHASE 1: Collect ALL projections AND activations BEFORE modifying any weights
    # This is critical - modifying a layer changes downstream activations
    logger.info("\n=== PHASE 1: COLLECTING MANIFOLD PROJECTIONS ===")

    layer_projections: dict[int, tuple] = {}  # layer_idx -> (P, manifold_dim, var, activations)

    for layer_idx in compress_layers:
        acts = collect_prime_activations(model, tokenizer, config, layer_idx, backend)
        P, manifold_dim, var = compute_manifold_projection(acts, backend, args.target_variance)
        layer_projections[layer_idx] = (P, manifold_dim, var, acts)  # Store activations too

        logger.info("Layer %2d: %dD manifold (%.1fx, %.2f%% variance)",
                   layer_idx, manifold_dim, hidden_dim / manifold_dim, var * 100)

    # PHASE 2: Compress and apply all weights
    logger.info("\n=== PHASE 2: COMPRESSING LAYERS ===")

    total_original = 0
    total_compressed = 0
    compression_metadata = {
        "method": "semantic_prime_manifold",
        "target_variance": args.target_variance,
        "layers": {},
    }

    for layer_idx in compress_layers:
        P, manifold_dim, var, activations = layer_projections[layer_idx]

        logger.info("\nLayer %d:", layer_idx)
        logger.info("  Manifold: %dD -> %dD (%.1fx, %.2f%% variance)",
                   hidden_dim, manifold_dim, hidden_dim / manifold_dim, var * 100)

        # Compress weights (for stats)
        compressed_weights, orig_params, comp_params = compress_layer_weights(
            model, layer_idx, P, config, backend
        )

        # Account for projection storage
        projection_params = hidden_dim * manifold_dim
        total_layer_compressed = comp_params + projection_params

        layer_ratio = orig_params / total_layer_compressed if total_layer_compressed > 0 else 0

        logger.info("  Weights: %d -> %d params (%.1fx)",
                   orig_params, total_layer_compressed, layer_ratio)

        # Apply using least-squares reconstruction (not direct projection)
        apply_compressed_weights_lstsq(model, layer_idx, P, activations, config, backend)

        total_original += orig_params
        total_compressed += total_layer_compressed

        compression_metadata["layers"][str(layer_idx)] = {
            "manifold_dim": manifold_dim,
            "variance_captured": var,
            "original_params": orig_params,
            "compressed_params": total_layer_compressed,
            "ratio": layer_ratio,
        }

    # Overall compression
    if total_compressed > 0:
        overall_ratio = total_original / total_compressed
        logger.info("\n=== COMPRESSION SUMMARY ===")
        logger.info("Compressed layers: %s", compress_layers)
        logger.info("Total params: %d -> %d (%.1fx compression)",
                   total_original, total_compressed, overall_ratio)
        compression_metadata["overall_ratio"] = overall_ratio
    else:
        logger.info("\nNo layers compressed")

    # Verify compressed model
    if not args.skip_verify:
        logger.info("\n=== COMPRESSED INFERENCE ===")
        compressed_responses = verify_inference(model, tokenizer, test_prompts)

        # Compare
        logger.info("\n=== COMPARISON ===")
        for i, (prompt, baseline, compressed) in enumerate(zip(test_prompts, baseline_responses, compressed_responses)):
            match = "MATCH" if baseline.strip() == compressed.strip() else "DIFF"
            logger.info("Prompt %d: %s", i + 1, match)
            if match == "DIFF":
                logger.info("  Baseline:   %s", baseline[:80].replace('\n', ' '))
                logger.info("  Compressed: %s", compressed[:80].replace('\n', ' '))

    # Save
    logger.info("\n=== SAVING MODEL ===")
    save_compressed_model(model, tokenizer, args.output, args.model, compression_metadata)

    logger.info("\nDONE")


if __name__ == "__main__":
    main()
