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

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

from .models import MergeGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def merge_weights(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    geometry: MergeGeometry,
    extract_layer_index_fn: Any,
    backend: "Backend",
    *,
    avoid_svd: bool = False,
    checkpoint_dir: str | None = None,
    layer_alpha_scale: dict[int, float] | None = None,
) -> tuple[dict[str, "Array"], dict[str, Any]]:
    """
    Execute merge using the computed geometry.

    Uses ALL blending strategies in sequence:
    1. Apply per-layer Procrustes rotations (from activation alignment)
    2. Apply dimension-specific alphas (from correlation analysis)
    3. Apply Fisher-weighted blending (from importance analysis)
    4. Apply task singular vector separation (skill vs structure)
    5. Apply null-space filtering (interference elimination)
    6. Apply DARE sparsification (optional sparsity)

    For cross-architecture models:
    - Uses layer_correspondence mapping to find source weights for each target layer
    - Applies cross_dimensional_projection when dimensions don't match

    Key insight: Higher dimensions contain lower dimensions.
    We blend at EACH dimension level using computed weights.
    """
    b = backend
    merged: dict[str, "Array"] = {}
    metrics: dict[str, Any] = {
        "weights_merged": 0,
        "rotations_applied": 0,
        "fisher_weights_used": 0,
        "dimension_weights_used": 0,
        "null_space_filtered": 0,
        "dare_sparsified": 0,
        "delta_mask_scaled": 0,
        "delta_mask_skipped": 0,
        # New metrics for connected geometry
        "transform_requirements_checked": 0,
        "intrinsic_dim_scaled": 0,
        "alpha_scaled_by_interference": 0,
        "shared_subspace_blends": 0,
        "curvature_aware_blends": 0,
        "verb_noun_applied": 0,
        "gw_transport_used": 0,
        "embedding_frechet_blends": 0,
        "slerp_merges": 0,
        # Cross-architecture metrics
        "cross_arch_layer_mappings": 0,
        "cross_arch_dim_projections": 0,
    }
    checkpoint_path = None
    weight_keys = [
        key for key in target_weights.keys()
        if not key.endswith(".scales") and not key.endswith(".biases")
    ]
    total_weights = len(weight_keys)

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "merge_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metrics["checkpoint_path"] = str(checkpoint_path)

    def _write_checkpoint(payload: dict[str, Any]) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.write_text(json.dumps(payload, sort_keys=True))

    # Build reverse correspondence: target_layer -> source_layer
    # IMPORTANT: Keep the FIRST (earliest) source layer for each target layer.
    # Multiple source layers may map to the same target (due to DP skips).
    # Using the earliest source layer maintains monotonicity.
    layer_correspondence = geometry.layer_correspondence
    reverse_correspondence: dict[int, int] = {}
    if layer_correspondence:
        # Sort by source layer to ensure we keep the earliest
        for src_layer in sorted(layer_correspondence.keys()):
            tgt_layer = layer_correspondence[src_layer]
            if tgt_layer not in reverse_correspondence:
                reverse_correspondence[tgt_layer] = src_layer

    if geometry.is_cross_architecture:
        logger.info(
            "Cross-architecture merge: %d layer mappings",
            len(layer_correspondence) if layer_correspondence else 0,
        )

    metrics["svd_disabled"] = avoid_svd
    if not avoid_svd:
        from modelcypher.core.domain.geometry.task_singular_vectors import (
            SVDBlendConfig,
            blend_with_svd_awareness,
        )
    from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

    def _apply_phase_lock_transform(
        weight: "Array",
        transform: "Array",
    ) -> tuple["Array", bool]:
        """Apply exact kernel alignment transform to weight if dimensions match."""
        if weight.ndim != 2:
            return weight, False

        t_in = transform.shape[0]
        weight_f32 = b.astype(weight, "float32")

        if weight.shape[1] == t_in:
            aligned = b.matmul(weight_f32, transform)
            b.eval(aligned)
            return aligned, True

        if weight.shape[0] == t_in:
            aligned = b.matmul(b.transpose(transform), weight_f32)
            b.eval(aligned)
            return aligned, True

        return weight, False

    for idx, key in enumerate(sorted(weight_keys), start=1):
        source_key = key
        target_layer_idx = extract_layer_index_fn(key)

        try:
            # For cross-architecture models, find corresponding source key
            if geometry.is_cross_architecture and target_layer_idx is not None and reverse_correspondence:
                # Find which source layer maps to this target layer
                source_layer_idx = reverse_correspondence.get(target_layer_idx)

                if source_layer_idx is not None and source_layer_idx != target_layer_idx:
                    # Replace target layer index with source layer index in the key
                    import re

                    source_key = re.sub(
                        rf"layers\.{target_layer_idx}\.",
                        f"layers.{source_layer_idx}.",
                        key,
                    )
                    metrics["cross_arch_layer_mappings"] += 1

            _write_checkpoint(
                {
                    "status": "start",
                    "index": idx,
                    "total": total_weights,
                    "key": key,
                    "source_key": source_key,
                    "layer_idx": target_layer_idx,
                    "timestamp": time.time(),
                }
            )
            logger.info(
                "MERGE WEIGHT [%d/%d] %s (source=%s)",
                idx,
                total_weights,
                key,
                source_key,
            )

            if source_key not in source_weights:
                # No source weight found - use target as-is
                target_value = dequantize_if_needed(
                    target_weights[key], key, target_weights, b
                )
                merged[key] = b.astype(target_value, "float32")
                _write_checkpoint(
                    {
                        "status": "skipped",
                        "index": idx,
                        "total": total_weights,
                        "key": key,
                        "source_key": source_key,
                        "layer_idx": target_layer_idx,
                        "timestamp": time.time(),
                    }
                )
                continue

            source_w = dequantize_if_needed(
                source_weights[source_key], source_key, source_weights, b
            )
            target_w = dequantize_if_needed(
                target_weights[key], key, target_weights, b
            )

            layer_idx = target_layer_idx
            layer_geom = (
                geometry.layer_geometries.get(layer_idx)
                if layer_idx is not None
                else None
            )

            layer_scale = 1.0
            if layer_alpha_scale and layer_idx is not None:
                layer_scale = float(layer_alpha_scale.get(layer_idx, 1.0))

            if layer_scale <= 0.0:
                merged_w = b.astype(target_w, "float32")
                target_dtype = target_w.dtype
                dtype_str = (
                    target_dtype.name
                    if hasattr(target_dtype, "name")
                    else str(target_dtype).replace("mlx.core.", "")
                )
                dtype_lower = dtype_str.lower()
                if "int" in dtype_lower or "uint" in dtype_lower:
                    merged[key] = b.astype(merged_w, "float32")
                else:
                    merged[key] = b.astype(merged_w, dtype_str)
                _write_checkpoint(
                    {
                        "status": "done",
                        "index": idx,
                        "total": total_weights,
                        "key": key,
                        "source_key": source_key,
                        "layer_idx": target_layer_idx,
                        "timestamp": time.time(),
                    }
                )
                metrics["weights_merged"] += 1
                metrics["delta_mask_skipped"] += 1
                continue

            # Apply per-layer exact kernel alignment transform before shape normalization
            if layer_geom and layer_geom.procrustes_rotation is not None:
                source_w, applied = _apply_phase_lock_transform(
                    source_w, layer_geom.procrustes_rotation
                )
                if applied:
                    metrics["rotations_applied"] += 1

            # Handle shape mismatch for cross-architecture merging.
            # Use geometry-preserving projection (Gram transport) instead of
            # target-only fallbacks. Models are always compatible.
            if source_w.shape != target_w.shape:
                if source_w.ndim == 1 and target_w.ndim == 1:
                    # 1D weights: simple truncation/padding is reasonable
                    d_s = source_w.shape[0]
                    d_t = target_w.shape[0]
                    if d_s > d_t:
                        # Truncate source
                        source_w = source_w[:d_t]
                    else:
                        # Pad source with target values (maintains target structure)
                        padding = target_w[d_s:]
                        source_w = b.concatenate([source_w, padding], axis=0)
                    b.eval(source_w)
                    metrics["cross_arch_dim_projections"] += 1
                    logger.debug(
                        "1D weight %s: truncate/pad %d -> %d",
                        key,
                        d_s,
                        d_t,
                    )
                else:
                    from modelcypher.core.domain.geometry.cross_dimensional_projection import (
                        ProjectionMethod,
                        project_cross_dimensional,
                    )

                    source_rows = source_w.shape[0] if source_w.ndim > 1 else 1
                    target_rows = target_w.shape[0] if target_w.ndim > 1 else 1
                    source_matrix = b.reshape(source_w, (source_rows, -1))
                    target_matrix = b.reshape(target_w, (target_rows, -1))
                    projection = project_cross_dimensional(
                        source_matrix,
                        target_matrix,
                        method=ProjectionMethod.GRAM_TRANSPORT,
                        backend=b,
                    )
                    source_w = b.reshape(projection.projected, target_w.shape)
                    b.eval(source_w)
                    metrics["cross_arch_dim_projections"] += 1
                    metrics["gw_transport_used"] += 1
                    logger.info(
                        "CROSS-DIM PROJECT %s: source=%s, target=%s, method=%s, alignment=%.4f",
                        key,
                        tuple(source_matrix.shape),
                        tuple(target_matrix.shape),
                        projection.method_used.value,
                        projection.alignment_score,
                    )

            # Null-space filter (MINGLE, 2025): preserve target activations by removing
            # source deltas that lie in the target activation row space.
            if (
                layer_geom
                and layer_geom.null_space_projection is not None
                and source_w.ndim == 2
            ):
                projection = layer_geom.null_space_projection
                if source_w.shape[1] == projection.shape[0]:
                    source_f32 = b.astype(source_w, "float32")
                    target_f32 = b.astype(target_w, "float32")
                    delta = source_f32 - target_f32
                    delta_safe = b.matmul(delta, projection)
                    source_w = target_f32 + delta_safe
                    b.eval(source_w)
                    metrics["null_space_filtered"] += 1

            # ============================================================
            # A.2: Check transform_requirements and set dispatch flags
            # ============================================================
            use_geodesic_blend = False
            if layer_geom and layer_geom.transform_requirements:
                for transform in layer_geom.transform_requirements:
                    tag = transform.upper()
                    if tag == "CURVATURE_CORRECTION":
                        use_geodesic_blend = True
                    elif tag == "ALPHA_SCALING":
                        # Reduce alpha in high-interference regions
                        pass  # Will be applied below with interference_score
                    elif tag == "BOUNDARY_SMOOTHING":
                        pass
                metrics["transform_requirements_checked"] += 1

            # Get base alpha for this layer
            alpha = 0.5
            if layer_geom:
                alpha = layer_geom.smoothed_alpha

                # ============================================================
                # A.4: Scale alpha by intrinsic dimension
                # ============================================================
                if layer_geom.intrinsic_dimension > 0:
                    ambient_dim = layer_geom.manifold_dimension or (
                        source_w.shape[-1] if source_w.ndim >= 1 else 1
                    )
                    if ambient_dim > 0:
                        compression_ratio = layer_geom.intrinsic_dimension / ambient_dim
                        # Continuous sigmoid scaling instead of discrete thresholds
                        # cr near 0 → alpha_scale ≈ 0.5 (trust target more)
                        # cr = 0.3 → alpha_scale = 0.85 (neutral point)
                        # cr near 1 → alpha_scale ≈ 1.2 (blend more confidently)
                        import math
                        alpha_scale = 0.5 + 0.7 / (1.0 + math.exp(-10.0 * (compression_ratio - 0.3)))
                        alpha = alpha * alpha_scale
                        metrics["intrinsic_dim_scaled"] += 1

                # Apply interference-based alpha scaling (from A.2 transform requirements)
                if any(t.upper() == "ALPHA_SCALING" for t in layer_geom.transform_requirements):
                    alpha = alpha * (1.0 - layer_geom.interference_score)
                    metrics["alpha_scaled_by_interference"] += 1

            if layer_scale != 1.0:
                alpha = alpha * layer_scale
                metrics["delta_mask_scaled"] += 1

            # Apply SVD-aware blending for 2D weights
            if source_w.ndim == 2 and target_w.ndim == 2 and min(source_w.shape) >= 2:
                source_f32 = b.astype(source_w, "float32")
                target_f32 = b.astype(target_w, "float32")
                merged_w = None  # Will be set by one of the blending paths

                # Embedding-scale matrices: use direct Frechet mean to avoid SVD blowups.
                m_rows, n_cols = source_f32.shape
                if m_rows > 4 * n_cols and m_rows > 10000:
                    eps = division_epsilon(b, source_f32)
                    source_abs = b.abs(source_f32)
                    target_abs = b.abs(target_f32)
                    merged_w = b.sqrt((source_abs + eps) * (target_abs + eps)) * b.sign(target_f32)
                    b.eval(merged_w)
                    metrics["embedding_frechet_blends"] += 1

                # ============================================================
                # A.1: Apply shared subspace projections if available
                # ============================================================
                if (
                    merged_w is None
                    and (
                        layer_geom
                        and layer_geom.source_projection is not None
                        and layer_geom.target_projection is not None
                    )
                ):
                    src_proj = layer_geom.source_projection
                    tgt_proj = layer_geom.target_projection
                    shared_dim = layer_geom.shared_dimension

                    # Check if projections are compatible with weight dimensions
                    if (
                        shared_dim > 0
                        and source_f32.shape[1] == src_proj.shape[0]
                        and target_f32.shape[1] == tgt_proj.shape[0]
                    ):
                        try:
                            # Project weights into shared subspace
                            source_in_shared = b.matmul(source_f32, src_proj)
                            target_in_shared = b.matmul(target_f32, tgt_proj)

                            # Blend in shared space
                            blended_shared = (
                                alpha * target_in_shared + (1 - alpha) * source_in_shared
                            )

                            # Project back to target space using transpose for orthogonal projections.
                            tgt_proj_t = b.transpose(tgt_proj)
                            merged_w = b.matmul(blended_shared, tgt_proj_t)
                            b.eval(merged_w)
                            metrics["shared_subspace_blends"] += 1
                        except Exception:
                            merged_w = None  # Fall back to other blending methods

                # ============================================================
                # A.3: Curvature-aware geodesic blending (SLERP)
                # ============================================================
                # No threshold - attempt SLERP whenever curvature is nonzero
                # The curvature value itself is the signal; let geometry decide
                if (
                    merged_w is None
                    and use_geodesic_blend
                    and layer_geom
                    and abs(layer_geom.curvature) > 0.0
                ):
                    try:
                        # SLERP: Spherical linear interpolation for curved manifolds
                        # For weight matrices, normalize and interpolate on the unit sphere
                        source_norm = b.norm(source_f32)
                        target_norm = b.norm(target_f32)
                        b.eval(source_norm, target_norm)

                        eps = division_epsilon(b, source_f32)
                        if float(source_norm) > eps and float(target_norm) > eps:
                            source_unit = source_f32 / source_norm
                            target_unit = target_f32 / target_norm

                            # Compute angle between normalized matrices
                            dot = b.sum(source_unit * target_unit)
                            b.eval(dot)
                            dot_val = float(dot)
                            dot_val = max(-1.0, min(1.0, dot_val))  # Clamp for acos

                            import math as m

                            theta = m.acos(dot_val)

                            if abs(theta) > eps:
                                # SLERP formula: (sin((1-t)*theta) * a + sin(t*theta) * b) / sin(theta)
                                sin_theta = m.sin(theta)
                                w_source = m.sin((1 - alpha) * theta) / sin_theta
                                w_target = m.sin(alpha * theta) / sin_theta

                                # Interpolate direction
                                merged_unit = w_source * source_unit + w_target * target_unit

                                # Interpolate magnitude (geometric mean for Frechet)
                                merged_norm = b.sqrt(source_norm * target_norm)
                                merged_w = merged_unit * merged_norm
                                b.eval(merged_w)
                                metrics["curvature_aware_blends"] += 1
                    except Exception:
                        merged_w = None  # Fall back to linear blending

                # Use Fisher weights if available for dimension-specific blending
                if (
                    merged_w is None
                    and layer_geom
                    and layer_geom.source_fisher is not None
                    and layer_geom.target_fisher is not None
                ):
                    try:
                        from modelcypher.core.domain.geometry.fisher_blending import (
                            FisherBlendingConfig,
                            FisherNormalization,
                            apply_fisher_blending,
                        )

                        src_fisher = _align_fisher_to_weight(
                            layer_geom.source_fisher, source_f32, b
                        )
                        tgt_fisher = _align_fisher_to_weight(
                            layer_geom.target_fisher, target_f32, b
                        )
                        if src_fisher is not None and tgt_fisher is not None:
                            config = FisherBlendingConfig(
                                normalization=FisherNormalization.LAYER,
                                strength=0.5,
                            )
                            merged_w, _ = apply_fisher_blending(
                                source_weight=source_f32,
                                target_weight=target_f32,
                                base_alpha=alpha,
                                source_fisher=src_fisher,
                                target_fisher=tgt_fisher,
                                config=config,
                                backend=b,
                            )
                            metrics["fisher_weights_used"] += 1
                    except Exception:
                        merged_w = None

                # Backward-compatible Fisher ratio blending
                if merged_w is None and layer_geom and layer_geom.fisher_weights is not None:
                    hidden_dim = layer_geom.fisher_weights.shape[0]
                    # Apply Fisher-weighted blending per dimension
                    # fisher_weights[d] = how much to trust target for dimension d
                    if source_f32.shape[1] == hidden_dim:
                        # Weight is [hidden_dim], broadcast to columns
                        fw = b.reshape(layer_geom.fisher_weights, (1, -1))
                        # Blend: merged = fw * target + (1-fw) * source
                        merged_w = fw * target_f32 + (1.0 - fw) * source_f32
                        metrics["fisher_weights_used"] += 1
                    elif source_f32.shape[0] == hidden_dim:
                        # Weight applies to rows
                        fw = b.reshape(layer_geom.fisher_weights, (-1, 1))
                        merged_w = fw * target_f32 + (1.0 - fw) * source_f32
                        metrics["fisher_weights_used"] += 1

                # Use dimension correlations if available and no merged_w yet
                if merged_w is None and layer_geom and layer_geom.dimension_alphas is not None:
                    hidden_dim = layer_geom.dimension_alphas.shape[0]
                    if source_f32.shape[1] == hidden_dim:
                        # Per-dimension alpha: high correlation = trust either
                        # Low correlation = trust target (stability)
                        da = b.reshape(layer_geom.dimension_alphas, (1, -1))
                        # alpha_d controls blend: 1 = trust target, 0 = trust source
                        # We want: low corr -> trust target, high corr -> blend evenly
                        target_weight = 1.0 - 0.5 * da  # Range [0.5, 1.0]
                        merged_w = target_weight * target_f32 + (1.0 - target_weight) * source_f32
                        metrics["dimension_weights_used"] += 1
                    elif source_f32.shape[0] == hidden_dim:
                        da = b.reshape(layer_geom.dimension_alphas, (-1, 1))
                        target_weight = 1.0 - 0.5 * da
                        merged_w = target_weight * target_f32 + (1.0 - target_weight) * source_f32
                        metrics["dimension_weights_used"] += 1

                # Fallback to SLERP when SVD is disabled.
                if merged_w is None and avoid_svd:
                    from modelcypher.core.domain.geometry.vector_math import (
                        BackendVectorMath,
                    )

                    vm = BackendVectorMath(b)
                    slerp_result = vm.slerp_matrix(source_f32, target_f32, alpha)
                    if slerp_result is not None:
                        merged_w, _ = slerp_result
                        metrics["slerp_merges"] += 1

                # Fallback to SVD-aware blending if nothing else worked.
                if merged_w is None and not avoid_svd:
                    merged_w = blend_with_svd_awareness(
                        source_f32, target_f32, alpha, SVDBlendConfig()
                    )

                # ============================================================
                # A.6: Apply verb-noun mask if available
                # ============================================================
                if layer_geom and layer_geom.verb_noun_mask is not None:
                    vn_mask = layer_geom.verb_noun_mask
                    hidden_dim = vn_mask.shape[0]
                    # verb_noun_mask gives per-dimension alpha:
                    # High value = verb-like = trust source (skill donor)
                    # Low value = noun-like = trust target (knowledge base)
                    if source_f32.shape[1] == hidden_dim:
                        vn_weights = b.reshape(vn_mask, (1, -1))
                        # Re-blend with verb-noun weights
                        # merged = vn * source + (1-vn) * target
                        merged_w = vn_weights * source_f32 + (1.0 - vn_weights) * merged_w
                        b.eval(merged_w)
                        metrics["verb_noun_applied"] += 1
                    elif source_f32.shape[0] == hidden_dim:
                        vn_weights = b.reshape(vn_mask, (-1, 1))
                        merged_w = vn_weights * source_f32 + (1.0 - vn_weights) * merged_w
                        b.eval(merged_w)
                        metrics["verb_noun_applied"] += 1

                b.eval(merged_w)

                # Apply DARE sparsification with continuous scaling by interference
                # No discrete threshold - intensity scales with interference_score
                if layer_geom and layer_geom.interference_score > 0.0:
                    try:
                        from modelcypher.core.domain.geometry.dare_sparsity import (
                            Configuration as DAREConfig,
                        )
                        from modelcypher.core.domain.geometry.dare_sparsity import (
                            analyze_sparsity,
                        )
                        # Compute delta and sparsify
                        delta = merged_w - target_f32
                        b.eval(delta)
                        b.to_numpy(delta)

                        # Continuous sparsity scaling:
                        # High interference → lower threshold → drop more
                        # Low interference → higher threshold → drop less
                        interference = layer_geom.interference_score
                        sparsity_threshold = 0.01 + 0.09 * (1.0 - interference)

                        # Analyze sparsity
                        config = DAREConfig(
                            sparsity_threshold=sparsity_threshold,
                            droppable_percentile=0.9,
                        )
                        analyze_sparsity({"delta": delta}, config)

                        # Drop low-magnitude components with interference-scaled threshold
                        threshold = sparsity_threshold * float(b.max(b.abs(delta)).item())
                        mask = b.abs(delta) > threshold
                        b.eval(mask)
                        sparse_delta = delta * b.astype(mask, "float32")
                        b.eval(sparse_delta)

                        merged_w = target_f32 + sparse_delta
                        b.eval(merged_w)
                        metrics["dare_sparsified"] += 1
                    except Exception:
                        pass
            else:
                # 1D tensors - geometric mean of magnitudes (Frechet mean on R+)
                eps = division_epsilon(b, source_w)
                merged_w = (
                    b.sqrt((b.abs(source_w) + eps) * (b.abs(target_w) + eps))
                    * b.sign(target_w)
                )
                b.eval(merged_w)

            # Preserve target dtype
            target_dtype = target_w.dtype
            dtype_str = (
                target_dtype.name
                if hasattr(target_dtype, "name")
                else str(target_dtype).replace("mlx.core.", "")
            )
            dtype_lower = dtype_str.lower()
            if "int" in dtype_lower or "uint" in dtype_lower:
                merged[key] = b.astype(merged_w, "float32")
            else:
                merged[key] = b.astype(merged_w, dtype_str)
            _write_checkpoint(
                {
                    "status": "done",
                    "index": idx,
                    "total": total_weights,
                    "key": key,
                    "source_key": source_key,
                    "layer_idx": target_layer_idx,
                    "timestamp": time.time(),
                }
            )
            metrics["weights_merged"] += 1
        except Exception as exc:
            _write_checkpoint(
                {
                    "status": "error",
                    "index": idx,
                    "total": total_weights,
                    "key": key,
                    "source_key": source_key,
                    "layer_idx": target_layer_idx,
                    "error": str(exc),
                    "timestamp": time.time(),
                }
            )
            logger.exception("MERGE WEIGHT FAILED: %s", key)
            raise

    # Copy target-only keys
    for key in target_weights:
        if key not in merged and not key.endswith(".scales") and not key.endswith(".biases"):
            target_value = dequantize_if_needed(
                target_weights[key], key, target_weights, b
            )
            merged[key] = b.astype(target_value, "float32")

    logger.info(
        "MERGE: %d weights, %d rotations, %d Fisher, %d dimension, %d DARE | "
        "NEW: %d shared_subspace, %d curvature, %d verb_noun, %d intrinsic_scaled, %d embed_frechet | "
        "CROSS-ARCH: %d layer_maps, %d dim_projects",
        metrics["weights_merged"],
        metrics["rotations_applied"],
        metrics["fisher_weights_used"],
        metrics["dimension_weights_used"],
        metrics["dare_sparsified"],
        metrics["shared_subspace_blends"],
        metrics["curvature_aware_blends"],
        metrics["verb_noun_applied"],
        metrics["intrinsic_dim_scaled"],
        metrics["embedding_frechet_blends"],
        metrics["cross_arch_layer_mappings"],
        metrics["cross_arch_dim_projections"],
    )

    return merged, metrics


def _align_fisher_to_weight(
    fisher: "Array", weight: "Array", backend: "Backend"
) -> "Array | None":
    """Align a Fisher vector/tensor to a weight matrix shape."""
    if fisher is None:
        return None
    if fisher.shape == weight.shape:
        return fisher
    if fisher.ndim == 1 and weight.ndim == 2:
        if fisher.shape[0] == weight.shape[1]:
            return backend.reshape(fisher, (1, -1))
        if fisher.shape[0] == weight.shape[0]:
            return backend.reshape(fisher, (-1, 1))
    return None
