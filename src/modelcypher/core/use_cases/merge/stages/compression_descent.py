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

"""Stage 4: Compression descent on transmission layers.

Applies intrinsic compression to transmission-layer weights after null-space
injection. Uses activation-derived subspaces and records validation metrics
(CKA, compression ratios) for diagnostics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_compression import (
    compress_weight_to_intrinsic_dim,
    validate_compression_lossless,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class CompressionDescentResult:
    """Result of compression descent on transmission layers.

    Attributes:
        compressed_weights: Weight key -> compressed weight matrix
        compression_ratios: Weight key -> compression ratio (smaller = more compressed)
        cka_validations: Weight key -> CKA score on probe activations
        layers_compressed: List of layer indices that were compressed
        mean_compression_ratio: Average compression ratio across all weights
        mean_cka: Average CKA validation score
        weights_compressed: Number of weights compressed
        weights_skipped: Number of weights skipped (dimension mismatch, etc.)
    """

    compressed_weights: dict[str, "Array"] = field(default_factory=dict)
    compression_ratios: dict[str, float] = field(default_factory=dict)
    cka_validations: dict[str, float] = field(default_factory=dict)
    layers_compressed: list[int] = field(default_factory=list)
    mean_compression_ratio: float = 0.0
    mean_cka: float = 0.0
    weights_compressed: int = 0
    weights_skipped: int = 0


def stage_compression_descent(
    merged_weights: dict[str, "Array"],
    transmission_layers: list[int],
    layer_activations: dict[int, "Array"],
    extract_layer_index_fn: Callable[[str], int | None],
    backend: "Backend | None" = None,
    compression_target: float = 0.5,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> CompressionDescentResult:
    """Apply compression descent to transmission layers.

    Uses intrinsic compression on per-layer activations to reduce rank and
    record CKA-based diagnostics.

    Args:
        merged_weights: All weights after null-space injection (from transplant stage)
        transmission_layers: List of layer indices to compress (from layer profile)
        layer_activations: Dict mapping layer_idx -> activations [n_samples, hidden_dim]
        extract_layer_index_fn: Function to extract layer index from weight key
        backend: Compute backend
        compression_target: Target fraction of dimensions to keep (0.5 = keep 50%)
        progress_callback: Optional callback for progress reporting

    Returns:
        CompressionDescentResult with compressed weights and validation metrics
    """
    b = backend or get_default_backend()
    stage_start = time.time()

    result = CompressionDescentResult()
    transmission_set = set(transmission_layers)

    if not transmission_layers:
        logger.warning("COMPRESSION DESCENT: No transmission layers specified")
        return result

    logger.info(
        "COMPRESSION DESCENT: Starting stage 4 - %d transmission layers, target=%.1f%%",
        len(transmission_layers),
        compression_target * 100,
    )

    # Group weights by layer
    weights_by_layer: dict[int, list[str]] = {}
    for key in merged_weights:
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is not None and layer_idx in transmission_set:
            weights_by_layer.setdefault(layer_idx, []).append(key)

    total_weights = sum(len(keys) for keys in weights_by_layer.values())
    weights_processed = 0

    for layer_idx in sorted(weights_by_layer.keys()):
        layer_keys = weights_by_layer[layer_idx]
        layer_acts = layer_activations.get(layer_idx)

        if layer_acts is None:
            logger.warning(
                "COMPRESSION DESCENT: No activations for layer %d, skipping",
                layer_idx,
            )
            result.weights_skipped += len(layer_keys)
            continue

        # Convert activations to proper format
        if hasattr(layer_acts, 'shape') and len(b.shape(layer_acts)) == 2:
            stacked_acts = layer_acts
        elif hasattr(layer_acts, '__len__'):
            stacked_acts = b.stack(layer_acts, axis=0)
        else:
            logger.warning(
                "COMPRESSION DESCENT: Invalid activation format for layer %d",
                layer_idx,
            )
            result.weights_skipped += len(layer_keys)
            continue

        stacked_acts = _promote_precision(stacked_acts, b)
        b.eval(stacked_acts)

        act_dim = int(b.shape(stacked_acts)[1])
        n_samples = int(b.shape(stacked_acts)[0])

        logger.info(
            "COMPRESSION DESCENT: Layer %d - %d weights, activations [%d, %d]",
            layer_idx, len(layer_keys), n_samples, act_dim,
        )

        layer_compressed = False

        for weight_key in layer_keys:
            weight = merged_weights.get(weight_key)
            if weight is None:
                result.weights_skipped += 1
                continue

            weight = _promote_precision(b.array(weight), b)
            b.eval(weight)

            W_shape = b.shape(weight)
            if len(W_shape) != 2:
                logger.debug(
                    "COMPRESSION DESCENT: Skipping non-2D weight: %s",
                    weight_key,
                )
                result.weights_skipped += 1
                continue

            out_dim = int(W_shape[0])
            in_dim = int(W_shape[1])

            # Check dimension compatibility
            if in_dim != act_dim:
                logger.debug(
                    "COMPRESSION DESCENT: Dimension mismatch for %s (in_dim=%d, act_dim=%d)",
                    weight_key, in_dim, act_dim,
                )
                result.weights_skipped += 1
                continue

            try:
                # Compute target variance threshold for compression
                # compression_target = 0.5 means keep top 50% of variance
                # This translates to keeping dimensions that explain compression_target of total variance
                compression_result = compress_weight_to_intrinsic_dim(
                    W=weight,
                    activations=stacked_acts,
                    backend=b,
                    variance_threshold=None,  # Use default (sqrt(eps) * max_eigenvalue)
                )

                # Validate compression is lossless on probe activations
                cka, max_rel_error = validate_compression_lossless(
                    original_W=weight,
                    compression_result=compression_result,
                    activations=stacked_acts,
                    backend=b,
                )

                # Reconstruct compressed weight
                W_compressed = compression_result.reconstruct(b)
                b.eval(W_compressed)

                # Store results
                result.compressed_weights[weight_key] = W_compressed
                result.compression_ratios[weight_key] = compression_result.compression_ratio
                result.cka_validations[weight_key] = cka
                result.weights_compressed += 1
                layer_compressed = True

                logger.info(
                    "COMPRESSION DESCENT: %s - rank %d/%d (%.1fx), CKA=%.6f",
                    weight_key,
                    compression_result.utilized_rank,
                    in_dim,
                    1.0 / compression_result.compression_ratio if compression_result.compression_ratio > 0 else 0,
                    cka,
                )

                # Validate CKA is close to 1.0 (lossless)
                eps = machine_epsilon(b, weight)
                threshold = 1.0 - sqrt_scalar(eps, b)
                if cka < threshold:
                    logger.warning(
                        "COMPRESSION DESCENT: %s has lossy compression (CKA=%.6f < %.6f)",
                        weight_key, cka, threshold,
                    )

            except Exception as e:
                logger.warning(
                    "COMPRESSION DESCENT: Failed to compress %s: %s",
                    weight_key, e,
                )
                result.weights_skipped += 1
                continue

            weights_processed += 1
            if progress_callback:
                progress_callback(
                    f"Compressing {weight_key}",
                    weights_processed,
                    total_weights,
                )

        if layer_compressed:
            result.layers_compressed.append(layer_idx)

    # Compute summary statistics
    if result.compression_ratios:
        result.mean_compression_ratio = (
            sum(result.compression_ratios.values()) / len(result.compression_ratios)
        )
    if result.cka_validations:
        result.mean_cka = (
            sum(result.cka_validations.values()) / len(result.cka_validations)
        )

    stage_elapsed = time.time() - stage_start

    logger.info(
        "COMPRESSION DESCENT: Stage 4 complete - %.2fs, %d weights compressed, "
        "%d skipped, mean_compression=%.2fx, mean_cka=%.6f",
        stage_elapsed,
        result.weights_compressed,
        result.weights_skipped,
        1.0 / result.mean_compression_ratio if result.mean_compression_ratio > 0 else 0,
        result.mean_cka,
    )

    return result


def apply_compression_descent_to_weights(
    merged_weights: dict[str, "Array"],
    compression_result: CompressionDescentResult,
) -> dict[str, "Array"]:
    """Apply compression descent results to the merged weight dict.

    Args:
        merged_weights: Original merged weights dict
        compression_result: Result from stage_compression_descent

    Returns:
        Updated weights dict with compressed weights replaced
    """
    updated = dict(merged_weights)
    for weight_key, compressed_weight in compression_result.compressed_weights.items():
        updated[weight_key] = compressed_weight
    return updated


__all__ = [
    "CompressionDescentResult",
    "stage_compression_descent",
    "apply_compression_descent_to_weights",
]
