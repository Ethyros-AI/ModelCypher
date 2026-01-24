# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Compression to an activation-defined subspace.

This module factors weights using a basis derived from activation variance.
Given weight W [out_dim, in_dim] and activations A [n, in_dim], it computes
a basis V_used for directions with non-trivial variance and stores:

    W_left = W @ V_used
    W_reconstructed = W_left @ V_used.T

This targets preservation of outputs on the sampled activations. Validation
computes CKA and relative error between original and reconstructed outputs.

References:
    - LoRA-Null (AAAI 2026): "The null space of activations is more accurate"
    - Apple ICML 2024: Models use only ~20-35% of available representation space
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    compute_variance_null_space,
    VarianceNullSpaceResult,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_architecture import ModelArchitecturePort

logger = logging.getLogger(__name__)


@dataclass
class LayerCompressionResult:
    """Result of compressing a single layer to intrinsic dimensionality.

    The weight is stored in factorized form:
        W_original ≈ W_left @ V_used.T   (reconstruction in used subspace)

    Attributes:
        W_left: Left factor [out_dim, utilized_rank]
        V_used: Utilized subspace basis [in_dim, utilized_rank]
        original_shape: Original weight shape (out_dim, in_dim)
        utilized_rank: Number of utilized dimensions (intrinsic dimensionality)
        available_rank: Number of unused dimensions (null space)
        compression_ratio: Factorized params / original params
        variance_captured: Fraction of total variance in utilized subspace
        variance_threshold: Threshold used for splitting utilized/available
    """

    W_left: "Array"
    V_used: "Array"
    original_shape: tuple[int, int]
    utilized_rank: int
    available_rank: int
    compression_ratio: float
    variance_captured: float
    variance_threshold: float

    def reconstruct(self, backend: "Backend") -> "Array":
        """Reconstruct the full weight matrix from factorized form.

        W_reconstructed = W_left @ V_used.T

        Directions outside V_used are zeroed.
        """
        return backend.matmul(self.W_left, backend.transpose(self.V_used))


@dataclass
class ModelCompressionResult:
    """Result of compressing an entire model to intrinsic dimensionality.

    Attributes:
        layer_results: Per-layer compression results
        total_original_params: Total parameters in original model
        total_compressed_params: Total parameters in compressed model
        overall_compression_ratio: total_compressed / total_original
        mean_utilized_fraction: Average utilized_rank / in_dim across layers
    """

    layer_results: dict[str, LayerCompressionResult]
    total_original_params: int
    total_compressed_params: int
    overall_compression_ratio: float
    mean_utilized_fraction: float


def compress_weight_to_intrinsic_dim(
    W: "Array",
    activations: "Array",
    backend: "Backend",
    variance_threshold: float | None = None,
) -> LayerCompressionResult:
    """Compress a weight matrix to its intrinsic dimensionality.

    Computes a factorization based on activation variance. Intended to
    preserve outputs on the sampled activations; use validation metrics to
    quantify fidelity.

    Args:
        W: Weight matrix [out_dim, in_dim]
        activations: Input activations [n_samples, in_dim] that define the manifold
        backend: Backend for tensor operations
        variance_threshold: Optional threshold for variance-based splitting.
            If None, uses sqrt(machine_epsilon) * max_eigenvalue (numerically derived).

    Returns:
        LayerCompressionResult with factorized weight storage
    """
    b = backend

    # Promote precision for numerical stability
    W = _promote_precision(b.array(W), b)
    activations = _promote_precision(b.array(activations), b)
    b.eval(W, activations)

    W_shape = b.shape(W)
    out_dim = int(W_shape[0])
    in_dim = int(W_shape[1])

    A_shape = b.shape(activations)
    n_samples = int(A_shape[0])
    act_dim = int(A_shape[1])

    if act_dim != in_dim:
        raise ValueError(
            f"Dimension mismatch: weight in_dim={in_dim}, activation dim={act_dim}"
        )

    logger.info(
        "INTRINSIC COMPRESS: W [%d, %d], activations [%d, %d]",
        out_dim, in_dim, n_samples, act_dim
    )

    # Compute variance-based null space
    variance_result = compute_variance_null_space(
        activations, backend, variance_threshold
    )

    utilized_rank = variance_result.utilized_rank
    available_rank = variance_result.available_rank
    V_used = variance_result.utilized_basis  # [in_dim, utilized_rank]
    b.eval(V_used)

    if utilized_rank == 0:
        raise ValueError(
            "No utilized dimensions found - activations have zero variance in all directions. "
            "This indicates degenerate activations (constant or zero)."
        )

    # Project weight onto utilized subspace
    # W_left = W @ V_used   [out_dim, utilized_rank]
    W_left = b.matmul(W, V_used)
    b.eval(W_left)

    # Compute compression ratio
    original_params = out_dim * in_dim
    compressed_params = out_dim * utilized_rank + in_dim * utilized_rank
    compression_ratio = compressed_params / original_params

    # Compute variance captured
    eigenvalues = variance_result.eigenvalues
    total_variance_arr = b.sum(eigenvalues)
    utilized_variance_arr = b.sum(eigenvalues[:utilized_rank]) if utilized_rank > 0 else b.array(0.0)
    b.eval(total_variance_arr, utilized_variance_arr)

    total_variance = float(b.to_scalar(total_variance_arr))
    utilized_variance = float(b.to_scalar(utilized_variance_arr))
    variance_captured = utilized_variance / total_variance if total_variance > 0 else 0.0

    logger.info(
        "INTRINSIC COMPRESS: utilized_rank=%d/%d (%.1f%%), compression=%.2fx, "
        "variance_captured=%.6f",
        utilized_rank, in_dim,
        100.0 * utilized_rank / in_dim,
        1.0 / compression_ratio if compression_ratio > 0 else 0.0,
        variance_captured
    )

    return LayerCompressionResult(
        W_left=W_left,
        V_used=V_used,
        original_shape=(out_dim, in_dim),
        utilized_rank=utilized_rank,
        available_rank=available_rank,
        compression_ratio=compression_ratio,
        variance_captured=variance_captured,
        variance_threshold=variance_result.variance_threshold,
    )


def validate_compression_lossless(
    original_W: "Array",
    compression_result: LayerCompressionResult,
    activations: "Array",
    backend: "Backend",
) -> tuple[float, float]:
    """Validate compression on the sampled activations.

    Computes CKA between original and reconstructed outputs, plus the maximum
    relative error in outputs on the provided activations.

    Args:
        original_W: Original weight matrix [out_dim, in_dim]
        compression_result: Result from compress_weight_to_intrinsic_dim
        activations: Input activations [n_samples, in_dim]
        backend: Backend for tensor operations

    Returns:
        Tuple of (cka, max_relative_error):
            - cka: Centered Kernel Alignment on the sampled activations
            - max_relative_error: Maximum relative error in outputs
    """
    b = backend

    original_W = _promote_precision(b.array(original_W), b)
    activations = _promote_precision(b.array(activations), b)
    W_left = _promote_precision(compression_result.W_left, b)
    V_used = _promote_precision(compression_result.V_used, b)
    b.eval(original_W, activations, W_left, V_used)

    # Original outputs: Y_orig = A @ W_orig.T
    Y_orig = b.matmul(activations, b.transpose(original_W))
    b.eval(Y_orig)

    # Compressed outputs: Y_comp = A @ V_used @ W_left.T
    A_proj = b.matmul(activations, V_used)  # [n, utilized_rank]
    Y_comp = b.matmul(A_proj, b.transpose(W_left))  # [n, out_dim]
    b.eval(Y_comp)

    # Compute CKA
    # Center the outputs
    Y_orig_centered = Y_orig - b.mean(Y_orig, axis=0, keepdims=True)
    Y_comp_centered = Y_comp - b.mean(Y_comp, axis=0, keepdims=True)
    b.eval(Y_orig_centered, Y_comp_centered)

    # Gram matrices
    G_orig = b.matmul(Y_orig_centered, b.transpose(Y_orig_centered))
    G_comp = b.matmul(Y_comp_centered, b.transpose(Y_comp_centered))
    b.eval(G_orig, G_comp)

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_orig_comp = b.sum(G_orig * G_comp)
    hsic_orig_orig = b.sum(G_orig * G_orig)
    hsic_comp_comp = b.sum(G_comp * G_comp)
    b.eval(hsic_orig_comp, hsic_orig_orig, hsic_comp_comp)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    eps = machine_epsilon(b, G_orig)
    denominator = b.sqrt(hsic_orig_orig * hsic_comp_comp)
    denominator = b.maximum(denominator, b.array(eps))
    cka = hsic_orig_comp / denominator
    b.eval(cka)
    cka_val = float(b.to_scalar(cka))

    # Compute maximum relative error
    diff = Y_orig - Y_comp
    abs_diff = b.abs(diff)
    abs_orig = b.abs(Y_orig)

    # Avoid division by zero: only compute relative error where original is nonzero
    eps_arr = b.array(eps)
    rel_error = abs_diff / b.maximum(abs_orig, eps_arr)
    max_rel_error_arr = b.max(rel_error)
    b.eval(max_rel_error_arr)
    max_rel_error = float(b.to_scalar(max_rel_error_arr))

    logger.info(
        "COMPRESSION VALIDATION: CKA=%.10f, max_rel_error=%.4e",
        cka_val, max_rel_error
    )

    return cka_val, max_rel_error


def compress_layer_with_validation(
    W: "Array",
    activations: "Array",
    backend: "Backend",
    variance_threshold: float | None = None,
) -> tuple[LayerCompressionResult, float, float]:
    """Compress a layer and validate outputs on sampled activations.

    Convenience wrapper that compresses and validates in one call.
    Raises if CKA < 1 - sqrt(eps) to flag degraded reconstruction on probes.

    Args:
        W: Weight matrix [out_dim, in_dim]
        activations: Input activations [n_samples, in_dim]
        backend: Backend for tensor operations
        variance_threshold: Optional variance threshold

    Returns:
        Tuple of (compression_result, cka, max_rel_error)

    Raises:
        RuntimeError: If CKA falls below the dtype-derived threshold
    """
    b = backend

    # Compress
    result = compress_weight_to_intrinsic_dim(W, activations, b, variance_threshold)

    # Validate
    cka, max_rel_error = validate_compression_lossless(W, result, activations, b)

    # Check losslessness
    eps = machine_epsilon(b, b.array(activations))
    threshold = 1.0 - sqrt_scalar(eps, b)

    if cka < threshold:
        raise RuntimeError(
            f"Compression is LOSSY: CKA={cka:.10f} < {threshold:.10f}. "
            f"This violates the lossless guarantee - there may be a bug in the "
            f"variance null space computation or the activation manifold is not "
            f"properly captured by the provided activations."
        )

    return result, cka, max_rel_error


class IntrinsicCompressor:
    """Compresses models using activation-defined subspaces.

    Provides helpers to compress a single layer or a full model given
    probe activations.
    """

    def __init__(self, backend: "Backend") -> None:
        """Initialize the compressor.

        Args:
            backend: Backend for tensor operations (MLX, JAX, etc.)
        """
        self._backend = backend

    def compress_layer(
        self,
        W: "Array",
        activations: "Array",
        validate: bool = True,
        variance_threshold: float | None = None,
    ) -> LayerCompressionResult:
        """Compress a single layer to its intrinsic dimensionality.

        Args:
            W: Weight matrix [out_dim, in_dim]
            activations: Input activations [n_samples, in_dim] defining the manifold
            validate: If True, runs validation metrics and raises on low CKA
            variance_threshold: Optional variance threshold for subspace splitting

        Returns:
            LayerCompressionResult with factorized weight storage
        """
        if validate:
            result, cka, max_rel_error = compress_layer_with_validation(
                W, activations, self._backend, variance_threshold
            )
            logger.info(
                "LAYER COMPRESSED: rank=%d/%d (%.1fx), CKA=%.10f",
                result.utilized_rank,
                result.original_shape[1],
                1.0 / result.compression_ratio if result.compression_ratio > 0 else 0,
                cka
            )
            return result
        else:
            return compress_weight_to_intrinsic_dim(
                W, activations, self._backend, variance_threshold
            )

    def compress_model(
        self,
        model: Any,
        layer_activations: dict[int, "Array"],
        weight_keys: list[str] | None = None,
        validate: bool = True,
    ) -> ModelCompressionResult:
        """Compress all layers of a model to intrinsic dimensionality.

        Args:
            model: The model to compress (must have accessible weights)
            layer_activations: Dict mapping layer_idx -> activations [n_samples, hidden_dim]
            weight_keys: Optional list of weight keys to compress. If None, uses
                all MLP weight keys from the architecture protocol.
            validate: If True, runs validation per layer

        Returns:
            ModelCompressionResult with all layer compression results
        """
        from modelcypher.adapters.model_architecture import get_model_architecture

        b = self._backend

        # Get architecture
        config: dict = {}
        if hasattr(model, "config"):
            model_config = model.config
            if hasattr(model_config, "to_dict"):
                config = model_config.to_dict()
            elif isinstance(model_config, dict):
                config = model_config

        arch = get_model_architecture(config, model)

        layer_results: dict[str, LayerCompressionResult] = {}
        total_original = 0
        total_compressed = 0
        utilized_fractions: list[float] = []

        for layer_idx in sorted(layer_activations.keys()):
            activations = layer_activations[layer_idx]
            activations = _promote_precision(b.array(activations), b)
            b.eval(activations)

            # Get layer accessor
            layer_accessor = arch.layer_accessor(layer_idx)

            # Get MLP keys for this layer
            if weight_keys is None:
                mlp_keys = arch.layer_mlp_keys(layer_idx)
            else:
                mlp_keys = [k for k in weight_keys if f".{layer_idx}." in k]

            for key in mlp_keys:
                # Get weight from model
                try:
                    weight = self._get_weight_by_key(model, key)
                    if weight is None:
                        logger.warning("COMPRESS MODEL: Weight not found: %s", key)
                        continue

                    weight = _promote_precision(b.array(weight), b)
                    b.eval(weight)

                    W_shape = b.shape(weight)
                    if len(W_shape) != 2:
                        logger.debug("COMPRESS MODEL: Skipping non-2D weight: %s", key)
                        continue

                    out_dim = int(W_shape[0])
                    in_dim = int(W_shape[1])

                    # Check if activations match weight input dimension
                    act_dim = int(b.shape(activations)[1])
                    if act_dim != in_dim:
                        logger.debug(
                            "COMPRESS MODEL: Dimension mismatch for %s "
                            "(weight in_dim=%d, act_dim=%d), skipping",
                            key, in_dim, act_dim
                        )
                        continue

                    # Compress this weight
                    result = self.compress_layer(
                        weight, activations, validate=validate
                    )

                    layer_results[key] = result
                    total_original += out_dim * in_dim
                    total_compressed += (
                        out_dim * result.utilized_rank +
                        in_dim * result.utilized_rank
                    )
                    utilized_fractions.append(result.utilized_rank / in_dim)

                except Exception as e:
                    logger.warning(
                        "COMPRESS MODEL: Failed to compress %s: %s",
                        key, e
                    )
                    continue

        if total_original == 0:
            raise ValueError("No weights were compressed - check weight_keys and layer_activations")

        overall_ratio = total_compressed / total_original
        mean_utilized = sum(utilized_fractions) / len(utilized_fractions) if utilized_fractions else 0.0

        logger.info(
            "MODEL COMPRESSED: %d weights, %.2fM -> %.2fM params (%.2fx), "
            "mean_utilized=%.1f%%",
            len(layer_results),
            total_original / 1e6,
            total_compressed / 1e6,
            1.0 / overall_ratio if overall_ratio > 0 else 0,
            mean_utilized * 100
        )

        return ModelCompressionResult(
            layer_results=layer_results,
            total_original_params=total_original,
            total_compressed_params=total_compressed,
            overall_compression_ratio=overall_ratio,
            mean_utilized_fraction=mean_utilized,
        )

    def _get_weight_by_key(self, model: Any, key: str) -> "Array | None":
        """Get a weight tensor from a model by its key path.

        Supports both nested attribute access (model.layer.weight)
        and dict-style access (model['layer']['weight']).
        """
        parts = key.split(".")
        current = model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            elif hasattr(current, "__getitem__"):
                try:
                    # Try numeric index
                    if part.isdigit():
                        current = current[int(part)]
                    else:
                        current = current[part]
                except (KeyError, IndexError, TypeError):
                    return None
            else:
                return None

        return current


def estimate_compression_potential(
    activations: "Array",
    backend: "Backend",
) -> dict[str, float]:
    """Estimate compression potential from activations without a weight matrix.

    Useful for quickly assessing how much compression is possible for a layer
    before running the full compression pipeline.

    Args:
        activations: Input activations [n_samples, hidden_dim]
        backend: Backend for tensor operations

    Returns:
        Dict with:
            - utilized_fraction: Fraction of dimensions with significant variance
            - estimated_compression: Estimated compression ratio (smaller = better)
            - intrinsic_dim: Estimated intrinsic dimensionality
            - hidden_dim: Full hidden dimension
    """
    b = backend

    activations = _promote_precision(b.array(activations), b)
    b.eval(activations)

    hidden_dim = int(b.shape(activations)[1])

    variance_result = compute_variance_null_space(activations, b)
    utilized_rank = variance_result.utilized_rank

    # Estimate compression for a square weight matrix
    # Actual compression depends on out_dim, but this gives a sense
    estimated_compression = 2.0 * utilized_rank / hidden_dim

    return {
        "utilized_fraction": utilized_rank / hidden_dim,
        "estimated_compression": estimated_compression,
        "intrinsic_dim": utilized_rank,
        "hidden_dim": hidden_dim,
    }


__all__ = [
    "LayerCompressionResult",
    "ModelCompressionResult",
    "compress_weight_to_intrinsic_dim",
    "validate_compression_lossless",
    "compress_layer_with_validation",
    "IntrinsicCompressor",
    "estimate_compression_potential",
]
