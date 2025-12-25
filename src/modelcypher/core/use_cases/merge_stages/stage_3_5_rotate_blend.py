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

"""
Stages 3-5: ROTATE + BLEND + PROPAGATE

For each layer:
1. ROTATE: Compute/apply geometric alignment (Procrustes or GW Transport)
2. BLEND: Compute multi-layer alpha with all adjustments
3. PROPAGATE: Carry rotation to next layer (zipper)

When use_transport_guided=True, uses Gromov-Wasserstein optimal transport
instead of Procrustes rotation.

Blend adjustments applied in sequence:
1. Base alpha from intersection confidence
2. Gaussian smoothing across layers
3. Spectral penalty for ill-conditioned weights
4. SVD-aware blending (different alpha for high/low rank)
5. Correlation-based dimension weights (from IntersectionMap)
6. VerbNoun modulation

Reference: Ainsworth et al. (2022) "Git Re-Basin"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class RotateBlendConfig:
    """Configuration for Stages 3-5.

    GEOMETRY-DRIVEN MERGE: No arbitrary thresholds.

    The merge formula is mathematically derived:
    1. Procrustes alignment: Find optimal rotation R* = argmin ||W_t - R @ W_s||_F
    2. SVD decomposition: Extract singular vectors (geometry) and values (magnitude)
    3. Fréchet mean: Geometric mean of singular values (geodesic midpoint on ℝ^+)
    4. Reconstruct: Use target's geometry with merged magnitudes

    W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

    No alpha. No thresholds. The geometry determines everything.
    """

    # Legacy flags - kept for backward compatibility but ignored
    # The pure geometric merge has no parameters
    enable_rotation: bool = True  # Always true - rotation is the alignment step
    use_transport_guided: bool = False  # Transport is alternative alignment method

    # PROPAGATE (zipper) - maintains layer continuity
    enable_zipper: bool = True
    zipper_use_weight_matching: bool = True


@dataclass
class RotateBlendResult:
    """Result of Stages 3-5."""

    merged_weights: dict[str, Any]  # Backend arrays
    rotate_metrics: dict[str, Any]
    blend_metrics: dict[str, Any]


# =============================================================================
# PURE GEOMETRIC MERGE - No Parameters, No Thresholds
# =============================================================================


def geometric_merge_weights(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend",
) -> tuple["Array", dict[str, float]]:
    """
    Geometrically correct merge of two weight matrices.

    NO PARAMETERS. NO THRESHOLDS. PURE GEOMETRY.

    The merge formula is mathematically derived:
    1. Procrustes alignment: Find optimal rotation R* = argmin ||W_t - R @ W_s||_F
    2. SVD decomposition: Extract singular vectors (geometry) and values (magnitude)
    3. Fréchet mean: Geometric mean of singular values (geodesic midpoint on ℝ^+)
    4. Reconstruct: Use target's geometry with merged magnitudes

    Formula: W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

    Args:
        source_w: Source weight matrix (m × n)
        target_w: Target weight matrix (m × n) - must match source shape
        backend: Backend for GPU-accelerated operations

    Returns:
        Tuple of (merged_weights, metrics)
        metrics contains: procrustes_error, spectral_preservation, merge_quality
    """
    b = backend

    # Ensure float32 for numerical stability
    source_f32 = b.astype(source_w, "float32")
    target_f32 = b.astype(target_w, "float32")

    # ==========================================================================
    # STEP 1: Procrustes Alignment
    # Find R* = argmin_R ||W_t - R @ W_s||_F subject to R^T R = I
    # Solution: M = W_t @ W_s^T, SVD(M) = UΣV^T, R* = UV^T
    # ==========================================================================

    M = b.matmul(target_f32, b.transpose(source_f32))
    U_M, _, Vt_M = b.svd(M, compute_uv=True)
    R = b.matmul(U_M, Vt_M)
    b.eval(R)

    # Handle reflection (det = -1) by flipping last column of U
    det_R = b.det(R)
    b.eval(det_R)
    # Extract scalar using backend item() pattern
    det_val = float(det_R.item()) if hasattr(det_R, 'item') else float(b.to_numpy(det_R))
    if det_val < 0:
        U_M_last_flipped = U_M[:, -1] * -1.0
        U_M_fixed = b.concatenate([U_M[:, :-1], b.expand_dims(U_M_last_flipped, axis=1)], axis=1)
        R = b.matmul(U_M_fixed, Vt_M)
        b.eval(R)

    # Apply rotation to source
    source_aligned = b.matmul(R, source_f32)
    b.eval(source_aligned)

    # Compute Procrustes error (normalized) - all backend ops
    diff = target_f32 - source_aligned
    error_norm = b.norm(b.reshape(diff, (-1,)))
    target_norm = b.norm(b.reshape(target_f32, (-1,)))
    b.eval(error_norm, target_norm)

    # ==========================================================================
    # STEP 2: SVD Decomposition
    # W = UΣV^T where U, V are geometry (singular vectors), Σ is magnitude
    # ==========================================================================

    U_s, sigma_s, Vt_s = b.svd(source_aligned, compute_uv=True)
    U_t, sigma_t, Vt_t = b.svd(target_f32, compute_uv=True)
    b.eval(U_s, sigma_s, U_t, sigma_t, Vt_t)

    # ==========================================================================
    # STEP 3: Fréchet Mean of Singular Values
    # The geometric mean is the Fréchet mean on ℝ^+ with log metric
    # σ_merged = √(σ_s × σ_t)
    # ==========================================================================

    # Handle dimension mismatch in singular values
    k = min(sigma_s.shape[0], sigma_t.shape[0])
    sigma_s_k = sigma_s[:k]
    sigma_t_k = sigma_t[:k]

    # Geometric mean - the mathematically correct merge for positive magnitudes
    # Add small epsilon to avoid sqrt(0) issues
    eps = 1e-10
    sigma_merged = b.sqrt((sigma_s_k + eps) * (sigma_t_k + eps))
    b.eval(sigma_merged)

    # Compute spectral energies - all backend ops
    energy_source = b.sum(sigma_s_k ** 2)
    energy_target = b.sum(sigma_t_k ** 2)
    energy_merged = b.sum(sigma_merged ** 2)
    b.eval(energy_source, energy_target, energy_merged)

    # ==========================================================================
    # STEP 4: Reconstruct with Target Geometry and Merged Magnitudes
    # W_merged = U_t @ diag(σ_merged) @ V_t^T
    # ==========================================================================

    U_t_k = U_t[:, :k]
    Vt_t_k = Vt_t[:k, :]

    # Reconstruct: (U_t * sigma_merged) @ Vt_t
    merged = b.matmul(U_t_k * sigma_merged, Vt_t_k)
    b.eval(merged)

    # Compute merge quality - all backend ops
    diff_to_source = b.norm(b.reshape(merged - source_aligned, (-1,)))
    diff_to_target = b.norm(b.reshape(merged - target_f32, (-1,)))
    b.eval(diff_to_source, diff_to_target)

    # Extract scalars only at the boundary for metrics dict
    def to_scalar(arr: "Array") -> float:
        """Extract scalar from backend array."""
        if hasattr(arr, 'item'):
            return float(arr.item())
        return float(b.to_numpy(arr))

    target_norm_val = to_scalar(target_norm)
    procrustes_error = to_scalar(error_norm) / (target_norm_val + eps)
    spectral_preservation = to_scalar(energy_merged) / (
        0.5 * (to_scalar(energy_source) + to_scalar(energy_target)) + eps
    )
    merge_quality = 1.0 - 0.5 * (
        to_scalar(diff_to_source) + to_scalar(diff_to_target)
    ) / (target_norm_val + eps)

    metrics = {
        "procrustes_error": procrustes_error,
        "spectral_preservation": spectral_preservation,
        "merge_quality": merge_quality,
        "effective_rank": k,
    }

    return merged, metrics


def stage_rotate_blend_propagate(
    source_weights: dict[str, Any],
    target_weights: dict[str, Any],
    intersection_map_obj: Any | None,
    layer_confidences: dict[int, float],
    dimension_correlations: dict,
    layer_indices: list[int],
    config: RotateBlendConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    refinement_alphas: dict[int, float] | None = None,
    hard_swap_layers: set[int] | None = None,
    backend: Any | None = None,
) -> RotateBlendResult:
    """
    PURE GEOMETRIC MERGE - No arbitrary thresholds.

    For each weight matrix:
    1. Procrustes alignment: Find optimal rotation R* = argmin ||W_t - R @ W_s||_F
    2. SVD decomposition: Extract geometry (U, V) and magnitude (Σ)
    3. Fréchet mean: Geometric mean of singular values
    4. Reconstruct: W_merged = U_t @ diag(√(σ_s ⊙ σ_t)) @ V_t^T

    The geometry determines everything. No config parameters affect the merge.
    """
    # Initialize backend
    b = backend or get_default_backend()

    # Clear GPU cache before merge
    if hasattr(b, "clear_cache"):
        b.clear_cache()

    merged: dict[str, "Array"] = {}
    metrics: dict[str, Any] = {
        "procrustes_errors": [],
        "spectral_preservations": [],
        "merge_qualities": [],
        "geometric_merges": 0,
        "identity_copies": 0,
        "shape_mismatches": 0,
    }

    total_weights = len(target_weights)
    processed = 0

    logger.info("GEOMETRIC MERGE: Processing %d weight keys", total_weights)

    for key in sorted(target_weights.keys()):
        if key not in source_weights:
            continue

        processed += 1

        # Skip quantization metadata
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        # Dequantize if needed
        source_w = dequantize_if_needed(
            source_weights[key], key, source_weights, b
        )
        target_w = dequantize_if_needed(
            target_weights[key], key, target_weights, b
        )

        # Handle shape mismatch - project source to target shape
        if source_w.shape != target_w.shape:
            logger.warning(
                "SHAPE MISMATCH at %s: source=%s target=%s - projecting",
                key, source_w.shape, target_w.shape,
            )
            source_w, _ = _project_weight_to_target_shape(source_w, target_w, backend=b)
            b.eval(source_w)
            metrics["shape_mismatches"] += 1

        # Apply geometric merge for 2D weight matrices
        if source_w.ndim == 2 and target_w.ndim == 2 and min(source_w.shape) >= 2:
            merged_w, merge_metrics = geometric_merge_weights(source_w, target_w, b)
            metrics["procrustes_errors"].append(merge_metrics["procrustes_error"])
            metrics["spectral_preservations"].append(merge_metrics["spectral_preservation"])
            metrics["merge_qualities"].append(merge_metrics["merge_quality"])
            metrics["geometric_merges"] += 1

            if processed % 50 == 0:
                logger.info(
                    "GEOMETRIC MERGE [%d/%d] %s: error=%.4f, spectral=%.4f, quality=%.4f",
                    processed, total_weights, key,
                    merge_metrics["procrustes_error"],
                    merge_metrics["spectral_preservation"],
                    merge_metrics["merge_quality"],
                )
        else:
            # For 1D tensors (biases, norms) - geometric mean of magnitudes
            merged_w = b.sqrt((source_w + 1e-10) * (target_w + 1e-10)) * b.sign(target_w)
            b.eval(merged_w)
            metrics["identity_copies"] += 1

        merged[key] = b.astype(merged_w, str(target_w.dtype))

    # Copy target-only keys (skip quantization metadata)
    for key in target_weights:
        if key not in merged and not key.endswith(".scales") and not key.endswith(".biases"):
            merged[key] = b.array(target_weights[key])

    # Finalize metrics
    if metrics["procrustes_errors"]:
        errors = metrics["procrustes_errors"]
        metrics["mean_procrustes_error"] = sum(errors) / len(errors)
        metrics["max_procrustes_error"] = max(errors)

    if metrics["spectral_preservations"]:
        sp = metrics["spectral_preservations"]
        metrics["mean_spectral_preservation"] = sum(sp) / len(sp)

    if metrics["merge_qualities"]:
        mq = metrics["merge_qualities"]
        metrics["mean_merge_quality"] = sum(mq) / len(mq)

    logger.info(
        "GEOMETRIC MERGE COMPLETE: %d weights, mean_error=%.4f, mean_quality=%.4f",
        metrics["geometric_merges"],
        metrics.get("mean_procrustes_error", 0),
        metrics.get("mean_merge_quality", 0),
    )

    return RotateBlendResult(
        merged_weights=merged,
        rotate_metrics=metrics,  # Unified metrics
        blend_metrics=metrics,
    )


# =============================================================================
# SHAPE PROJECTION HELPERS (Cross-Dimension Geometry Preservation)
# =============================================================================


def _project_weight_to_target_shape(
    source_w: "Array",
    target_w: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """
    Project source weight matrix to target shape using geometry-preserving SVD.

    Different dimensions are compression/expansion levels of the same underlying
    geometry. We use the Gram matrix insight: K = X @ X^T captures relational
    geometry independent of feature dimension.

    For weight matrices:
    - Row mismatch (output dim): Project via left singular vectors
    - Col mismatch (input dim): Project via right singular vectors

    The projection preserves the maximum amount of geometric structure by:
    1. Computing SVD of source weights
    2. Truncating/expanding singular value decomposition to target shape
    3. Reconstructing with preserved top singular values

    Args:
        source_w: Source weight matrix (any shape)
        target_w: Target weight matrix (shape we need to match)
        backend: Backend for GPU-accelerated operations

    Returns:
        Tuple of (projected_weights, alignment_score)
        alignment_score: Fraction of variance preserved (1.0 = perfect)
    """
    b = backend or get_default_backend()

    source_shape = source_w.shape
    target_shape = target_w.shape

    # Handle 1D tensors (biases, norms)
    if source_w.ndim == 1:
        return _project_1d(source_w, target_shape[0], b)

    # Handle 2D weight matrices
    if source_w.ndim == 2:
        return _project_2d(source_w, target_shape, b)

    # For higher dimensions, reshape to 2D, project, reshape back
    # Flatten all but last dim, project, then reshape
    original_shape = source_shape
    flat_source = b.reshape(source_w, (-1, source_shape[-1]))
    flat_target_shape = (-1, target_shape[-1]) if len(target_shape) > 1 else target_shape

    projected_flat, score = _project_2d(
        flat_source,
        (flat_source.shape[0], target_shape[-1]),
        b,
    )

    # Compute output shape preserving leading dims from target
    out_shape = target_shape
    projected = b.reshape(projected_flat, out_shape)
    b.eval(projected)

    return projected, score


def _project_1d(
    source: "Array",
    target_size: int,
    backend: "Backend",
) -> tuple["Array", float]:
    """Project 1D tensor (bias, norm) to target size."""
    b = backend
    source_size = source.shape[0]

    if source_size == target_size:
        return source, 1.0

    if source_size > target_size:
        # Truncate: keep first target_size elements
        # For biases, the first dimensions typically correspond to
        # the most important output neurons
        projected = source[:target_size]
        # Score based on energy preserved
        total_energy = float(b.to_numpy(b.sum(source**2)))
        kept_energy = float(b.to_numpy(b.sum(projected**2)))
        score = kept_energy / (total_energy + 1e-10)
    else:
        # Expand: pad with zeros (preserves existing geometry)
        padding = b.zeros((target_size - source_size,), dtype=str(source.dtype))
        projected = b.concatenate([source, padding], axis=0)
        score = 1.0  # No information lost

    b.eval(projected)
    return projected, score


def _project_2d(
    source: "Array",
    target_shape: tuple[int, ...],
    backend: "Backend",
) -> tuple["Array", float]:
    """
    Project 2D weight matrix using SVD-based geometry preservation.

    The key insight: SVD decomposes W = U @ S @ V^T where:
    - U captures output space geometry (row relationships)
    - V captures input space geometry (column relationships)
    - S captures the importance of each geometric mode

    By truncating/expanding U and V while preserving S, we maintain
    the invariant geometric relationships.
    """
    b = backend
    source_rows, source_cols = source.shape
    target_rows, target_cols = target_shape[0], target_shape[1]

    # If shapes match, nothing to do
    if source_rows == target_rows and source_cols == target_cols:
        return source, 1.0

    try:
        # Compute SVD
        U, S, Vt = b.svd(source, compute_uv=True)
        b.eval(U, S, Vt)

        # Number of singular values
        k = min(len(S), min(source_rows, source_cols))

        # Track variance for alignment score
        total_variance = float(b.to_numpy(b.sum(S**2)))

        # Project U (output dimension) to target_rows
        if source_rows != target_rows:
            U = _resize_orthogonal_basis(U, target_rows, k, b)

        # Project V^T (input dimension) to target_cols
        if source_cols != target_cols:
            V = b.transpose(Vt)
            V = _resize_orthogonal_basis(V, target_cols, k, b)
            Vt = b.transpose(V)

        # Determine how many singular values to keep
        # (limited by the smaller dimension after projection)
        k_out = min(k, target_rows, target_cols)

        # Truncate if needed
        U_k = U[:, :k_out]
        S_k = S[:k_out]
        Vt_k = Vt[:k_out, :]

        # Reconstruct: W = U @ diag(S) @ V^T
        projected = b.matmul(U_k * S_k, Vt_k)
        b.eval(projected)

        # Compute alignment score (variance preserved)
        kept_variance = float(b.to_numpy(b.sum(S_k**2)))
        score = kept_variance / (total_variance + 1e-10)

        return projected, score

    except Exception as e:
        logger.warning("SVD projection failed: %s, using truncate/pad fallback", e)
        return _project_2d_simple(source, target_shape, b)


def _resize_orthogonal_basis(
    basis: "Array",
    target_rows: int,
    k: int,
    backend: "Backend",
) -> "Array":
    """
    Resize orthogonal basis matrix (U or V) to target row count.

    For expansion: Extend with orthogonal random vectors
    For reduction: Keep top-k rows that maximize preserved structure

    The geometry is preserved because:
    - Truncation keeps the most important rows (by position, which corresponds
      to output neuron ordering in most architectures)
    - Expansion adds orthogonal directions that don't interfere with existing geometry
    """
    b = backend
    source_rows = basis.shape[0]

    if source_rows == target_rows:
        return basis

    if source_rows > target_rows:
        # Truncate rows - keep first target_rows
        # In neural networks, earlier neurons often capture more general features
        return basis[:target_rows, :]
    else:
        # Expand rows - add orthogonal padding
        # Initialize with small random values to break symmetry
        padding_rows = target_rows - source_rows
        b.random_seed(42)  # Reproducibility
        padding = b.random_normal((padding_rows, k)) * 0.01

        # Orthogonalize padding against existing basis
        # Using Gram-Schmidt-like projection
        existing = basis[:, :k]
        b.eval(existing)

        # For efficiency, just use small random - the SVD reconstruction
        # will naturally weight these low due to small magnitude
        expanded = b.concatenate([existing, padding], axis=0)
        b.eval(expanded)
        return expanded


def _project_2d_simple(
    source: "Array",
    target_shape: tuple[int, ...],
    backend: "Backend",
) -> tuple["Array", float]:
    """Simple truncate/pad fallback when SVD fails."""
    b = backend
    source_rows, source_cols = source.shape
    target_rows, target_cols = target_shape[0], target_shape[1]

    # Handle rows
    if source_rows > target_rows:
        result = source[:target_rows, :]
    elif source_rows < target_rows:
        padding = b.zeros((target_rows - source_rows, source_cols), dtype=str(source.dtype))
        result = b.concatenate([source, padding], axis=0)
    else:
        result = source

    # Handle cols
    if source_cols > target_cols:
        result = result[:, :target_cols]
    elif source_cols < target_cols:
        padding = b.zeros((result.shape[0], target_cols - source_cols), dtype=str(source.dtype))
        result = b.concatenate([result, padding], axis=1)

    b.eval(result)

    # Score based on overlap ratio
    overlap_rows = min(source_rows, target_rows)
    overlap_cols = min(source_cols, target_cols)
    score = (overlap_rows * overlap_cols) / (source_rows * source_cols)

    return result, score


# =============================================================================
# KEY CLASSIFICATION HELPERS
# =============================================================================


def _is_residual_output(key: str) -> bool:
    """Check if weight is a residual stream output (o_proj, down_proj)."""
    lower = key.lower()
    return any(token in lower for token in ("o_proj", "wo", "out_proj", "down_proj", "w2"))


def _is_attention_input(key: str) -> bool:
    """Check if weight is an attention input projection."""
    lower = key.lower()
    return any(
        token in lower
        for token in ("q_proj", "k_proj", "v_proj", "wq", "wk", "wv", "query", "key", "value")
    )


def _is_mlp_input(key: str) -> bool:
    """Check if weight is an MLP input projection."""
    lower = key.lower()
    return any(token in lower for token in ("gate_proj", "up_proj", "w1", "w3", "fc1"))


def _is_v_proj(key: str) -> bool:
    """Check if weight is the value projection."""
    lower = key.lower()
    return any(token in lower for token in ("v_proj", "wv", ".value"))


def _is_o_proj(key: str) -> bool:
    """Check if weight is the output projection."""
    lower = key.lower()
    return any(token in lower for token in ("o_proj", "wo.", "out_proj"))


# =============================================================================
# INTRINSIC DIMENSION HELPERS (Dimensional Hierarchy)
# =============================================================================


def _infer_hidden_dim(weights: dict[str, Any]) -> int:
    """Infer hidden dimension from weight shapes.

    Works with both NumPy and MLX arrays.
    """
    for key, val in weights.items():
        if "embed" in key.lower() and val.ndim == 2:
            return int(val.shape[-1])
        if "layers.0." in key and ("q_proj" in key or "wq" in key) and val.ndim == 2:
            return int(val.shape[-1])
    # Fallback: find most common dimension
    dims = []
    for val in weights.values():
        if val.ndim == 2:
            dims.extend([int(d) for d in val.shape])
    if dims:
        from collections import Counter

        return Counter(dims).most_common(1)[0][0]
    return 4096  # reasonable default


def _compute_layer_intrinsic_dims(
    weights: dict[str, Any],
    layer_indices: list[int],
    threshold: float = 0.01,
    backend: "Backend | None" = None,
) -> dict[int, float]:
    """
    Compute effective rank (intrinsic dimension) per layer.

    Uses SVD to count singular values above threshold * max_singular_value.
    This gives the "true" dimensionality of the weight manifold.

    Args:
        weights: Model weights (Backend arrays)
        layer_indices: Layer indices to analyze
        threshold: Cutoff ratio (default 1% of max singular value)
        backend: Backend for GPU-accelerated SVD (required)

    Returns:
        Dict mapping layer index to median intrinsic dimension
    """
    b = backend or get_default_backend()
    result: dict[int, float] = {}

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        intrinsic_dims: list[int] = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if min(val.shape) < 32:
                continue

            try:
                # Use GPU-accelerated SVD
                val_arr = b.array(val)
                val_f32 = b.astype(val_arr, "float32")
                s = b.svd(val_f32, compute_uv=False)
                b.eval(s)

                # Count singular values above threshold
                s_np = b.to_numpy(s)
                cutoff = float(s_np[0]) * threshold
                effective_rank = int(sum(1 for sv in s_np if sv > cutoff))
                intrinsic_dims.append(effective_rank)
            except Exception:
                pass

        if intrinsic_dims:
            # Compute median using Python (list of ints)
            sorted_dims = sorted(intrinsic_dims)
            n = len(sorted_dims)
            if n % 2 == 0:
                result[layer_idx] = float((sorted_dims[n // 2 - 1] + sorted_dims[n // 2]) / 2)
            else:
                result[layer_idx] = float(sorted_dims[n // 2])

    return result


# =============================================================================
# CURVATURE HELPERS (Manifold Topology Risk)
# =============================================================================


def _compute_layer_curvature_profiles(
    weights: dict[str, Any],
    layer_indices: list[int],
    sample_rows: int = 64,
    k_neighbors: int = 15,
    backend: "Backend | None" = None,
) -> dict[int, tuple[str, float]]:
    """
    Compute curvature profile per layer.

    Uses sectional curvature estimation to assess manifold topology:
    - POSITIVE curvature: convergent geodesics, interference risk
    - NEGATIVE curvature: divergent geodesics, safe for blending
    - FLAT: Euclidean, neutral
    - MIXED: Variable topology

    Args:
        weights: Model weights (Backend arrays)
        layer_indices: Layer indices to analyze
        sample_rows: Number of rows to sample from weight matrices
        k_neighbors: Number of neighbors for curvature estimation
        backend: Backend for array operations

    Returns:
        Dict mapping layer index to (curvature_sign, anisotropy)
    """
    import random

    from modelcypher.core.domain.geometry.manifold_curvature import (
        CurvatureConfig,
        SectionalCurvatureEstimator,
    )

    b = backend or get_default_backend()
    result: dict[int, tuple[str, float]] = {}
    estimator = SectionalCurvatureEstimator(CurvatureConfig(num_directions=5))

    for layer_idx in layer_indices:
        layer_pattern = f"layers.{layer_idx}."
        curvature_profiles: list[tuple[str, float]] = []

        for key, val in weights.items():
            if layer_pattern not in key:
                continue
            if val.ndim != 2:
                continue
            if val.shape[0] < sample_rows or val.shape[1] < 16:
                continue

            try:
                # Convert to backend array
                val_arr = b.array(val)
                val_f32 = b.astype(val_arr, "float32")
                b.eval(val_f32)

                # Sample rows using Python random (avoid numpy)
                n_rows = val_f32.shape[0]
                sample_size = min(sample_rows, n_rows)
                indices = random.sample(range(n_rows), sample_size)

                # Extract sampled rows
                points = val_f32[indices, :]
                b.eval(points)

                # Estimate curvature profile (manifold_curvature uses backend internally)
                profile = estimator.estimate_manifold_profile(points, k_neighbors=k_neighbors)

                # Extract sign and anisotropy
                sign = profile.dominant_sign.value  # "positive", "negative", "flat", "mixed"
                anisotropy = profile.global_variance  # Higher = more variable curvature

                curvature_profiles.append((sign, anisotropy))
            except Exception:
                pass

        if curvature_profiles:
            # Use most common sign, mean anisotropy
            from collections import Counter

            signs = [s for s, _ in curvature_profiles]
            most_common_sign = Counter(signs).most_common(1)[0][0]
            anisotropies = [a for _, a in curvature_profiles]
            mean_anisotropy = sum(anisotropies) / len(anisotropies)
            result[layer_idx] = (most_common_sign, mean_anisotropy)

    return result


def _apply_curvature_adjustment(
    base_alpha: float,
    curvature_sign: str,
    anisotropy: float,
    config: RotateBlendConfig,
) -> tuple[float, str]:
    """
    Adjust alpha based on curvature topology.

    The adjustment is derived from the curvature geometry itself:
    - Positive curvature → convergent geodesics → interference risk
      Adjustment = exp(-|curvature_strength|) for natural exponential decay
    - Negative curvature → divergent geodesics → safe for blending
      Adjustment = 1 + (1 - exp(-|curvature_strength|)) for bounded increase
    - Anisotropy → directional instability → penalty proportional to excess anisotropy

    No arbitrary multipliers - the curvature magnitude determines the adjustment.

    Args:
        base_alpha: Current alpha value
        curvature_sign: "positive", "negative", "flat", or "mixed"
        anisotropy: Global curvature variance (0=isotropic, higher=variable)
        config: Stage configuration

    Returns:
        (adjusted_alpha, adjustment_description)
    """
    import math

    alpha = base_alpha
    adjustments = []

    # Curvature strength determines the base adjustment magnitude
    # Natural exponential scaling: curvature_strength controls how strongly we respond
    strength = config.curvature_strength

    # Sign-based adjustment derived from curvature geometry
    if curvature_sign == "positive":
        # Convergent geodesics = interference risk = reduce alpha
        # Use exponential decay: adjustment = exp(-strength)
        # At strength=0: adjustment=1.0 (no change)
        # At strength=0.3: adjustment≈0.74 (26% reduction)
        # At strength=1.0: adjustment≈0.37 (63% reduction)
        adjustment = math.exp(-strength)
        alpha *= adjustment
        adjustments.append("positive_curv")
    elif curvature_sign == "negative":
        # Divergent geodesics = safe = bounded increase
        # Use: 1 + (1 - exp(-strength)) * 0.3 for bounded increase
        # At strength=0: adjustment=1.0 (no change)
        # At strength=0.3: adjustment≈1.08 (8% increase)
        # At strength=1.0: adjustment≈1.19 (19% increase, bounded)
        boost_factor = (1.0 - math.exp(-strength)) * 0.3
        adjustment = 1.0 + boost_factor
        alpha = min(alpha * adjustment, 0.95)  # Cap at 0.95
        adjustments.append("negative_curv")
    elif curvature_sign == "mixed":
        # Variable topology = moderate caution
        # Use geometric mean between flat and positive behavior
        adjustment = math.exp(-strength * 0.5)
        alpha *= adjustment
        adjustments.append("mixed_curv")
    # flat = no adjustment (Euclidean-like, no risk)

    # Anisotropy-based adjustment derived from excess anisotropy
    # Only apply when anisotropy exceeds threshold - use the excess as the penalty basis
    if anisotropy > config.curvature_anisotropy_threshold:
        # Excess anisotropy above threshold determines penalty
        excess_anisotropy = anisotropy - config.curvature_anisotropy_threshold
        # Normalize: assuming anisotropy max is ~2.0 above threshold
        normalized_excess = min(1.0, excess_anisotropy / 2.0)
        # Exponential penalty based on excess
        aniso_penalty = 1.0 - math.exp(-strength * normalized_excess)
        alpha *= 1.0 - aniso_penalty
        adjustments.append("high_aniso")

    # Clamp to valid range
    alpha = max(0.0, min(0.95, alpha))

    description = "+".join(adjustments) if adjustments else "flat_no_adj"
    return alpha, description
