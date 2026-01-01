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

"""Stage 3: TRANSPLANT - Null-space constrained knowledge grafting.

Replaces sparse concept regions while preserving boundary behavior:
    A_core @ W' = A_core @ W_source_aligned
    A_boundary @ W' = A_boundary @ W_target
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_dimensional_projection import (
    ProjectionMethod,
    project_cross_dimensional,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    safe_pinv,
)
from modelcypher.core.domain.geometry.transplant import (
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.domain.merging.exceptions import (
    AlignmentFailureError,
    DimensionMismatchError,
    StitchUnavailableError,
)
from modelcypher.core.use_cases.merge_stages.transplant_manifest import (
    TransplantManifest,
    WeightStatus,
    WeightTransformRecord,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _project_activations_to_weight_dim(
    activations: "Array",
    weight: "Array",
    backend: "Backend",
) -> "Array":
    """Project activations to match weight input dimension for cross-architecture merges.

    For weight matrix [out_features, in_features], we need activations with
    shape [n_samples, in_features] for matmul to work.

    If activations have different feature dimension, we project using SVD:
    - Truncation: Project to top-k singular vectors if act_dim > weight_in_dim
    - Expansion: Pad with zeros if act_dim < weight_in_dim
    """
    b = backend
    act_dim = int(activations.shape[1])
    weight_in_dim = int(weight.shape[1])

    if act_dim == weight_in_dim:
        return activations

    from modelcypher.core.domain.geometry.numerical_stability import svd_via_eigh

    if act_dim > weight_in_dim:
        # Truncate: project to lower dimension via top-k singular vectors
        _, _, Vt = svd_via_eigh(b, activations, full_matrices=False)
        b.eval(Vt)
        k = min(weight_in_dim, int(Vt.shape[0]))
        V_k = b.transpose(Vt[:k, :])  # [act_dim, k]
        projected = b.matmul(activations, V_k)  # [n, k]

        if k < weight_in_dim:
            padding = b.zeros((int(activations.shape[0]), weight_in_dim - k))
            projected = b.concatenate([projected, padding], axis=1)
        b.eval(projected)
    else:
        # Expand: pad with zeros
        padding = b.zeros((int(activations.shape[0]), weight_in_dim - act_dim))
        projected = b.concatenate([activations, padding], axis=1)
        b.eval(projected)

    return projected


@dataclass(frozen=True)
class TransplantStageConfig:
    """Configuration for transplant stage."""

    core_domains: tuple[str, ...]
    boundary_k: int | None = None
    geodesic_k_neighbors: int | None = None
    projection_method: ProjectionMethod = ProjectionMethod.GRAM_TRANSPORT
    transplant_layers: tuple[int, ...] | None = None  # None = all layers
    checkpoint_dir: Path | None = None  # Enable checkpointing if set
    progress_callback: Callable[[str, int, int], None] | None = None  # (msg, current, total)
    analysis_max_samples: int = 128  # Cap geometry analysis samples per layer
    analysis_anchor_count: int = 32  # Anchor count for relative representations
    enable_shared_subspace: bool = True
    enable_relative_representation: bool = True
    enable_fisher_weighting: bool = True
    enable_interference_analysis: bool = True
    # NOTE: Alpha interpolation was REMOVED. The null-space projection determines
    # preserved_fraction geometrically. Do NOT add hardcoded scalar overrides.

    # Fail-loud mode: If True, raise exceptions instead of logging warnings and continuing
    strict_mode: bool = True


@dataclass
class TransplantStageResult:
    """Result of transplant stage."""

    merged_weights: dict[str, Any]
    metrics: dict[str, Any]
    manifest: TransplantManifest | None = None  # Track every weight's status


def _normalize_domains(domains: Iterable[str]) -> set[str]:
    return {d.strip().lower() for d in domains if d.strip()}


def _save_checkpoint(
    checkpoint_dir: Path,
    layer_idx: int,
    merged_weights: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save transplant progress checkpoint for resume capability."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata (small JSON file with state)
    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    meta = {
        "last_completed_layer": layer_idx,
        "timestamp": time.time(),
        "weights_transplanted": metrics.get("weights_transplanted", 0),
        "layers_transplanted": metrics.get("layers_transplanted", 0),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("CHECKPOINT: Saved progress at layer %d to %s", layer_idx, checkpoint_dir)


def _load_checkpoint(checkpoint_dir: Path) -> tuple[int, dict[str, Any]] | None:
    """Load transplant checkpoint if available.

    Returns:
        Tuple of (last_completed_layer, metadata) or None if no checkpoint exists.
    """
    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        last_layer = meta.get("last_completed_layer", -1)
        logger.info(
            "CHECKPOINT: Resuming from layer %d (weights=%d, layers=%d)",
            last_layer,
            meta.get("weights_transplanted", 0),
            meta.get("layers_transplanted", 0),
        )
        return last_layer, meta
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("CHECKPOINT: Failed to load checkpoint: %s", e)
        return None


def _compute_alignment_metrics(
    core_acts: "Array",
    weight_before: "Array",
    weight_after: "Array",
    weight_source: "Array",
    backend: "Backend",
) -> dict[str, float]:
    """Measure core alignment shift toward the source for a single weight.

    For cross-architecture merges where dimensions don't match, we project
    activations to the weight's input dimension using SVD before computing
    output activations. This preserves the geometric comparison while handling
    dimension mismatches.
    """
    from modelcypher.core.domain.geometry.cka import compute_cka

    b = backend

    # Project activations to match weight dimensions (handles cross-architecture)
    core_acts = _project_activations_to_weight_dim(core_acts, weight_before, b)

    output_before = b.matmul(core_acts, b.transpose(weight_before))
    output_after = b.matmul(core_acts, b.transpose(weight_after))
    output_source = b.matmul(core_acts, b.transpose(weight_source))
    b.eval(output_before, output_after, output_source)

    dist_before_arr = b.norm(output_before - output_source)
    dist_after_arr = b.norm(output_after - output_source)
    b.eval(dist_before_arr, dist_after_arr)

    dist_before = float(b.to_numpy(dist_before_arr))
    dist_after = float(b.to_numpy(dist_after_arr))

    eps = float(machine_epsilon(b, weight_before))
    if dist_before > eps:
        alignment_improvement = (dist_before - dist_after) / dist_before
    else:
        alignment_improvement = 0.0

    cka_before = compute_cka(output_before, output_source, backend=b)
    cka_after = compute_cka(output_after, output_source, backend=b)

    return {
        "core_dist_to_source_before": dist_before,
        "core_dist_to_source_after": dist_after,
        "alignment_improvement": alignment_improvement,
        "cka_before": cka_before.best,
        "cka_after": cka_after.best,
    }


def stage_transplant(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    layer_indices: list[int],
    probe_ids: list[str] | None,
    probe_domains: list[str] | None,
    target_activations: dict[int, list["Array"]] | None,
    config: TransplantStageConfig,
    extract_layer_index_fn: Callable[[str], int | None],
    source_activations: dict[int, list["Array"]] | None = None,
    source_intermediate_activations: dict[int, list["Array"]] | None = None,
    target_intermediate_activations: dict[int, list["Array"]] | None = None,
    source_attention_activations: dict[int, list["Array"]] | None = None,
    target_attention_activations: dict[int, list["Array"]] | None = None,
    source_kv_activations: dict[int, list["Array"]] | None = None,
    target_kv_activations: dict[int, list["Array"]] | None = None,
    backend: "Backend | None" = None,
) -> TransplantStageResult:
    """Stage 3: Null-space constrained transplant using probe activations."""
    b = backend or get_default_backend()
    merged: dict[str, "Array"] = dict(target_weights)

    metrics: dict[str, Any] = {
        "merge_strategy": "transplant",
        "layers_considered": 0,
        "layers_transplanted": 0,
        "weights_considered": 0,
        "weights_transplanted": 0,
        "preserved_fractions": [],
        "projection_losses": [],
        "null_dims": [],
        "boundary_relative_diffs": [],
        "alignment_improvements": [],
        "core_dist_to_source_before": [],
        "core_dist_to_source_after": [],
        "cka_before": [],
        "cka_after": [],
        "core_probes": 0,
        "boundary_k": config.boundary_k,
        "geodesic_k_neighbors": config.geodesic_k_neighbors,
        "shared_subspace_dimensions": [],
        "relative_rep_errors": [],
        "fisher_target_means": [],
        "transform_requirements_by_layer": {},
        "shared_subspace_applied": 0,
        "procrustes_applied": 0,
        "fisher_delta_scaled": 0,
    }

    # REQUIRE real activations collected from probe runs.
    if not target_activations:
        error_msg = (
            "Transplant requires real activations collected from probe runs. "
            "Use `mc merge pipeline` (probe stage) to collect activations before merging."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    metrics["activation_source"] = "collected_from_model"

    # Probe-based transplant requires metadata
    if not probe_ids or not probe_domains:
        metrics["transplant_skipped"] = "missing_probe_metadata"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    if len(probe_ids) != len(probe_domains):
        metrics["transplant_skipped"] = "probe_metadata_mismatch"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    core_domains = _normalize_domains(config.core_domains)
    core_probe_ids = {
        probe_id
        for probe_id, domain in zip(probe_ids, probe_domains)
        if domain and domain.lower() in core_domains
    }

    metrics["core_probes"] = len(core_probe_ids)
    if not core_probe_ids:
        metrics["transplant_skipped"] = "no_core_probes"
        return TransplantStageResult(merged_weights=merged, metrics=metrics)

    weights_by_layer: dict[int, list[str]] = {}
    for key in target_weights:
        layer_idx = extract_layer_index_fn(key)
        if layer_idx is None:
            continue
        weights_by_layer.setdefault(layer_idx, []).append(key)

    layer_relations: dict[int, Any] = {}
    analyzer = None
    if source_activations and config.enable_interference_analysis:
        from modelcypher.core.domain.geometry.interference_predictor import (
            MergeAnalysisConfig,
            MergeAnalyzer,
        )
        from modelcypher.core.domain.geometry.riemannian_density import (
            RiemannianDensityEstimator,
        )

        estimator = RiemannianDensityEstimator()
        overlap_scores: list[float] = []
        max_samples = max(0, config.analysis_max_samples)

        for layer_idx in layer_indices:
            src_list = source_activations.get(layer_idx)
            tgt_list = target_activations.get(layer_idx)
            if not src_list or not tgt_list:
                continue
            sample_count = min(len(src_list), len(tgt_list), max_samples)
            if sample_count < 2:
                continue

            src_stacked = b.stack(src_list[:sample_count], axis=0)
            tgt_stacked = b.stack(tgt_list[:sample_count], axis=0)
            b.eval(src_stacked, tgt_stacked)

            source_volume = estimator.estimate_concept_volume(
                f"layer_{layer_idx}_source",
                src_stacked,
            )
            target_volume = estimator.estimate_concept_volume(
                f"layer_{layer_idx}_target",
                tgt_stacked,
            )
            relation = estimator.compute_relation(source_volume, target_volume)
            layer_relations[layer_idx] = relation
            overlap_score = (
                relation.bhattacharyya_coefficient
                + relation.overlap_coefficient
                + relation.jaccard_index
            ) / 3.0
            overlap_scores.append(overlap_score)

        analysis_config = (
            MergeAnalysisConfig.from_overlap_distribution(overlap_scores)
            if overlap_scores
            else MergeAnalysisConfig()
        )
        analyzer = MergeAnalyzer(config=analysis_config)
        metrics["interference_analysis_layers"] = len(layer_relations)
        metrics["interference_thresholds"] = {
            "alpha_scaling_threshold": analysis_config.alpha_scaling_threshold,
            "curvature_correction_threshold": analysis_config.curvature_correction_threshold,
            "procrustes_threshold": analysis_config.procrustes_threshold,
            "boundary_asymmetry_threshold": analysis_config.boundary_asymmetry_threshold,
        }

    # Check for existing checkpoint
    resume_from_layer = -1
    if config.checkpoint_dir:
        checkpoint_result = _load_checkpoint(config.checkpoint_dir)
        if checkpoint_result:
            resume_from_layer, checkpoint_meta = checkpoint_result
            # Restore metrics from checkpoint
            metrics["weights_transplanted"] = checkpoint_meta.get("weights_transplanted", 0)
            metrics["layers_transplanted"] = checkpoint_meta.get("layers_transplanted", 0)

    # Count total weights for progress reporting
    total_layers = len(layer_indices)
    total_weights = sum(len(weights_by_layer.get(idx, [])) for idx in layer_indices)
    weights_processed = 0
    stage_start_time = time.time()

    logger.info(
        "TRANSPLANT: Starting stage 3 - %d layers, %d total weights (geometry-driven preserved_fraction)",
        total_layers, total_weights
    )

    # ==========================================================================
    # GLOBAL TRAJECTORY ALIGNMENT: Compute ONE hidden stitch for ALL layers
    # ==========================================================================
    # The model processes as a trajectory: layer_0 → layer_1 → ... → layer_N
    # If we compute independent F per layer, we break compositional structure.
    # Solution: compute a SINGLE alignment using concatenated activations.
    global_hidden_stitch_output = None
    global_hidden_stitch_input = None
    global_src_hidden_dim = None
    global_tgt_hidden_dim = None

    if source_activations:
        # MULTI-BOTTLENECK STRATEGY: Try 25%, 50%, 75% depth, take highest CKA
        # Different architectures may have invariance strongest at different depths.
        # By trying multiple positions, we find the best alignment point.
        bottleneck_fractions = [0.25, 0.50, 0.75]
        bottleneck_candidates = []

        for frac in bottleneck_fractions:
            idx_pos = int(len(layer_indices) * frac) if layer_indices else 0
            idx_pos = max(0, min(idx_pos, len(layer_indices) - 1)) if layer_indices else 0
            candidate_layer_idx = layer_indices[idx_pos] if layer_indices else 0

            src_list = source_activations.get(candidate_layer_idx)
            tgt_list = target_activations.get(candidate_layer_idx)

            if src_list and tgt_list:
                n_samples = min(len(src_list), len(tgt_list))
                if n_samples >= 20:
                    bottleneck_candidates.append({
                        "layer_idx": candidate_layer_idx,
                        "fraction": frac,
                        "src_list": src_list[:n_samples],
                        "tgt_list": tgt_list[:n_samples],
                        "n_samples": n_samples,
                    })

        if not bottleneck_candidates:
            logger.warning("GLOBAL: No bottleneck candidates with sufficient samples")

        # Try each bottleneck candidate - find FIRST perfect alignment or FAIL
        # There is no "best" imperfect alignment. CKA = 1.0 or WRONG.
        perfect_candidate = None
        perfect_result = None
        perfect_src_concat = None
        perfect_tgt_concat = None
        tried_ckas: list[tuple[int, float, float]] = []  # (layer_idx, fraction, cka)

        for candidate in bottleneck_candidates:
            src_concat = b.stack(candidate["src_list"], axis=0)
            tgt_concat = b.stack(candidate["tgt_list"], axis=0)
            src_concat = b.astype(src_concat, "float32")
            tgt_concat = b.astype(tgt_concat, "float32")
            b.eval(src_concat, tgt_concat)

            cand_src_dim = int(src_concat.shape[1])
            cand_tgt_dim = int(tgt_concat.shape[1])

            if cand_src_dim != cand_tgt_dim:
                from modelcypher.core.domain.geometry.gram_aligner import GramAligner
                aligner = GramAligner(b)
                result = aligner.find_perfect_alignment(src_concat, tgt_concat)
                tried_ckas.append((candidate["layer_idx"], candidate["fraction"], result.achieved_cka))
                logger.debug(
                    "MULTI-BOTTLENECK: Layer %d (%.0f%% depth) CKA=%.4f %s",
                    candidate["layer_idx"], candidate["fraction"] * 100, result.achieved_cka,
                    "✓ PERFECT" if result.is_perfect else "✗ FAILED"
                )
                if result.is_perfect:
                    # Found perfect alignment - use it
                    perfect_candidate = candidate
                    perfect_result = result
                    perfect_src_concat = src_concat
                    perfect_tgt_concat = tgt_concat
                    break  # Don't search further
            else:
                # Same dimensions - identity is perfect by definition
                tried_ckas.append((candidate["layer_idx"], candidate["fraction"], 1.0))
                logger.debug(
                    "MULTI-BOTTLENECK: Layer %d (%.0f%% depth) same dims=%d → IDENTITY (perfect)",
                    candidate["layer_idx"], candidate["fraction"] * 100, cand_src_dim
                )
                perfect_candidate = candidate
                perfect_result = None  # Signals identity
                perfect_src_concat = src_concat
                perfect_tgt_concat = tgt_concat
                break  # Identity is always perfect

        if perfect_candidate is not None:
            logger.info(
                "GLOBAL: Using bottleneck layer %d (%.0f%% depth) - PERFECT alignment (%d samples)",
                perfect_candidate["layer_idx"], perfect_candidate["fraction"] * 100,
                perfect_candidate["n_samples"]
            )
            all_src_acts = perfect_candidate["src_list"]
            all_tgt_acts = perfect_candidate["tgt_list"]
        else:
            # No bottleneck achieved perfect alignment - this is a FAILURE
            all_src_acts = []
            all_tgt_acts = []
            if config.strict_mode and bottleneck_candidates:
                raise AlignmentFailureError(
                    stage="MULTI_BOTTLENECK_ALIGNMENT",
                    weight_key=None,
                    message="No bottleneck layer achieved perfect alignment (CKA >= 0.9999)",
                    context={
                        "tried_bottlenecks": [
                            {"layer": idx, "depth_fraction": frac, "achieved_cka": cka}
                            for idx, frac, cka in tried_ckas
                        ],
                        "note": "Models may have genuinely different geometry - merge cannot proceed",
                    },
                )
            elif bottleneck_candidates:
                logger.error(
                    "MULTI-BOTTLENECK: ALL bottleneck layers FAILED. Tried: %s",
                    ", ".join(f"layer {idx} ({frac*100:.0f}%): CKA={cka:.4f}" for idx, frac, cka in tried_ckas)
                )

        if len(all_src_acts) >= 20:
            src_concat = b.stack(all_src_acts, axis=0)
            tgt_concat = b.stack(all_tgt_acts, axis=0)
            src_concat = b.astype(src_concat, "float32")
            tgt_concat = b.astype(tgt_concat, "float32")
            b.eval(src_concat, tgt_concat)

            global_src_hidden_dim = int(src_concat.shape[1])
            global_tgt_hidden_dim = int(tgt_concat.shape[1])

            if global_src_hidden_dim != global_tgt_hidden_dim:
                from modelcypher.core.domain.geometry.gram_aligner import GramAligner
                aligner = GramAligner(b)
                global_result = aligner.find_perfect_alignment(src_concat, tgt_concat)

                if global_result.is_perfect:
                    F = b.array(global_result.feature_transform)
                    b.eval(F)
                    global_hidden_stitch_output = b.transpose(F)  # F.T [tgt, src]
                    F_pinv, pinv_diag = safe_pinv(b, F)
                    global_hidden_stitch_input = b.transpose(F_pinv)  # pinv(F).T [src, tgt]
                    b.eval(global_hidden_stitch_output, global_hidden_stitch_input)
                    logger.info(
                        "GLOBAL TRAJECTORY ALIGNMENT: CKA=%.4f (%d→%d) cond=%.2e rank=%d/%d",
                        global_result.achieved_cka, global_src_hidden_dim,
                        global_tgt_hidden_dim, pinv_diag.get("condition_number", 0),
                        pinv_diag.get("effective_rank", 0), min(global_src_hidden_dim, global_tgt_hidden_dim)
                    )
                    metrics["global_trajectory_aligned"] = True
                    metrics["global_trajectory_cka"] = float(global_result.achieved_cka)
                else:
                    if config.strict_mode:
                        raise AlignmentFailureError(
                            stage="GLOBAL_TRAJECTORY_ALIGNMENT",
                            weight_key=None,
                            message=f"Global hidden alignment failed: CKA={global_result.achieved_cka:.4f} < 0.9999",
                            context={
                                "achieved_cka": float(global_result.achieved_cka),
                                "source_dim": global_src_hidden_dim,
                                "target_dim": global_tgt_hidden_dim,
                                "samples_used": len(all_src_acts),
                            },
                        )
                    logger.warning(
                        "GLOBAL TRAJECTORY ALIGNMENT failed (CKA=%.4f)",
                        global_result.achieved_cka
                    )
            else:
                # Same hidden dims: use IDENTITY stitch (no transformation)
                # This is required for MLP dual-stitch to work when only intermediate differs
                global_hidden_stitch_output = b.eye(global_src_hidden_dim)
                global_hidden_stitch_input = b.eye(global_src_hidden_dim)
                b.eval(global_hidden_stitch_output, global_hidden_stitch_input)
                logger.info(
                    "GLOBAL: Same hidden dims (%d) - using identity stitch",
                    global_src_hidden_dim
                )
                metrics["global_trajectory_aligned"] = True
                metrics["global_trajectory_cka"] = 1.0  # Identity = perfect

    # ==========================================================================
    # GLOBAL ATTENTION ALIGNMENT: Compute ONE attention stitch for ALL layers
    # ==========================================================================
    # Attention weights have a different dimension than hidden:
    #   - q_proj/k_proj: [num_heads * head_dim, hidden_dim] (e.g., [960, 960] for SmolLM)
    #   - o_proj: [hidden_dim, num_heads * head_dim]
    # When head counts differ (e.g., SmolLM=15 heads → Qwen=14 heads):
    #   - SmolLM attention dim = 15 * 64 = 960
    #   - Qwen attention dim = 14 * 64 = 896
    # GramAligner finds the transformation that achieves CKA=1.0 across these spaces.
    global_attention_stitch_output = None
    global_attention_stitch_input = None
    global_src_attn_dim = None
    global_tgt_attn_dim = None

    if source_attention_activations and target_attention_activations:
        # Use BOTTLENECK LAYER for global alignment (where invariance is strongest)
        bottleneck_layer_idx = layer_indices[len(layer_indices) // 2] if layer_indices else 0
        src_attn_list = source_attention_activations.get(bottleneck_layer_idx)
        tgt_attn_list = target_attention_activations.get(bottleneck_layer_idx)

        all_src_attn = []
        all_tgt_attn = []
        if src_attn_list and tgt_attn_list:
            n_samples = min(len(src_attn_list), len(tgt_attn_list))
            for i in range(n_samples):
                all_src_attn.append(src_attn_list[i])
                all_tgt_attn.append(tgt_attn_list[i])
            logger.info("GLOBAL ATTN: Using bottleneck layer %d for alignment (%d samples)",
                       bottleneck_layer_idx, n_samples)

        if len(all_src_attn) >= 20:
            src_attn_concat = b.stack(all_src_attn, axis=0)
            tgt_attn_concat = b.stack(all_tgt_attn, axis=0)
            src_attn_concat = b.astype(src_attn_concat, "float32")
            tgt_attn_concat = b.astype(tgt_attn_concat, "float32")
            b.eval(src_attn_concat, tgt_attn_concat)

            global_src_attn_dim = int(src_attn_concat.shape[1])
            global_tgt_attn_dim = int(tgt_attn_concat.shape[1])

            if global_src_attn_dim != global_tgt_attn_dim:
                from modelcypher.core.domain.geometry.gram_aligner import GramAligner
                aligner = GramAligner(b)
                attn_result = aligner.find_perfect_alignment(src_attn_concat, tgt_attn_concat)

                if attn_result.is_perfect:
                    F_attn = b.array(attn_result.feature_transform)
                    b.eval(F_attn)
                    global_attention_stitch_output = b.transpose(F_attn)  # F.T [tgt, src]
                    F_attn_pinv, attn_pinv_diag = safe_pinv(b, F_attn)
                    global_attention_stitch_input = b.transpose(F_attn_pinv)  # pinv(F).T [src, tgt]
                    b.eval(global_attention_stitch_output, global_attention_stitch_input)
                    logger.info(
                        "GLOBAL ATTENTION ALIGNMENT: CKA=%.4f (%d→%d) cond=%.2e",
                        attn_result.achieved_cka, global_src_attn_dim,
                        global_tgt_attn_dim, attn_pinv_diag.get("condition_number", 0)
                    )
                    metrics["global_attention_aligned"] = True
                    metrics["global_attention_cka"] = float(attn_result.achieved_cka)
                else:
                    if config.strict_mode:
                        raise AlignmentFailureError(
                            stage="GLOBAL_ATTENTION_ALIGNMENT",
                            weight_key=None,
                            message=f"Global attention alignment failed: CKA={attn_result.achieved_cka:.4f} < 0.9999",
                            context={
                                "achieved_cka": float(attn_result.achieved_cka),
                                "source_dim": global_src_attn_dim,
                                "target_dim": global_tgt_attn_dim,
                                "samples_used": len(all_src_attn),
                            },
                        )
                    logger.warning(
                        "GLOBAL ATTENTION ALIGNMENT failed (CKA=%.4f)",
                        attn_result.achieved_cka
                    )
            else:
                # Same attention dims: use IDENTITY stitch
                global_attention_stitch_output = b.eye(global_src_attn_dim)
                global_attention_stitch_input = b.eye(global_src_attn_dim)
                b.eval(global_attention_stitch_output, global_attention_stitch_input)
                logger.info(
                    "GLOBAL ATTN: Same attention dims (%d) - using identity stitch",
                    global_src_attn_dim
                )
                metrics["global_attention_aligned"] = True
                metrics["global_attention_cka"] = 1.0  # Identity = perfect

    # ==========================================================================
    # GLOBAL KV ALIGNMENT: Compute separate KV stitch for GQA models
    # ==========================================================================
    # GQA (Grouped Query Attention) models have different head counts for Q vs K/V:
    #   - SmolLM: Q = 15 heads × 64 = 960, KV = 5 heads × 64 = 320
    #   - Qwen: Q = 14 heads × 64 = 896, KV = 2 heads × 64 = 128
    #
    # k_proj and v_proj weights have shape [kv_attention_dim, hidden_dim], NOT
    # [q_attention_dim, hidden_dim]. We MUST compute a separate stitch for KV.
    #
    # Without this, merged models output gibberish because K/V dimension mismatch
    # causes attention score computation to fail or produce garbage.
    global_kv_stitch_output = None
    global_kv_stitch_input = None
    global_src_kv_dim = None
    global_tgt_kv_dim = None

    if source_kv_activations and target_kv_activations:
        # Use BOTTLENECK LAYER for global KV alignment
        bottleneck_layer_idx = layer_indices[len(layer_indices) // 2] if layer_indices else 0
        src_kv_list = source_kv_activations.get(bottleneck_layer_idx)
        tgt_kv_list = target_kv_activations.get(bottleneck_layer_idx)

        all_src_kv = []
        all_tgt_kv = []
        if src_kv_list and tgt_kv_list:
            n_samples = min(len(src_kv_list), len(tgt_kv_list))
            for i in range(n_samples):
                all_src_kv.append(src_kv_list[i])
                all_tgt_kv.append(tgt_kv_list[i])
            logger.info("GLOBAL KV: Using bottleneck layer %d for alignment (%d samples)",
                       bottleneck_layer_idx, n_samples)

        if len(all_src_kv) >= 20:
            src_kv_concat = b.stack(all_src_kv, axis=0)
            tgt_kv_concat = b.stack(all_tgt_kv, axis=0)
            src_kv_concat = b.astype(src_kv_concat, "float32")
            tgt_kv_concat = b.astype(tgt_kv_concat, "float32")
            b.eval(src_kv_concat, tgt_kv_concat)

            global_src_kv_dim = int(src_kv_concat.shape[1])
            global_tgt_kv_dim = int(tgt_kv_concat.shape[1])

            if global_src_kv_dim != global_tgt_kv_dim:
                from modelcypher.core.domain.geometry.gram_aligner import GramAligner
                aligner = GramAligner(b)
                kv_result = aligner.find_perfect_alignment(src_kv_concat, tgt_kv_concat)

                if kv_result.is_perfect:
                    F_kv = b.array(kv_result.feature_transform)
                    b.eval(F_kv)
                    global_kv_stitch_output = b.transpose(F_kv)  # F.T [tgt, src]
                    F_kv_pinv, kv_pinv_diag = safe_pinv(b, F_kv)
                    global_kv_stitch_input = b.transpose(F_kv_pinv)  # pinv(F).T [src, tgt]
                    b.eval(global_kv_stitch_output, global_kv_stitch_input)
                    logger.info(
                        "GLOBAL KV ALIGNMENT: CKA=%.4f (%d→%d) cond=%.2e",
                        kv_result.achieved_cka, global_src_kv_dim,
                        global_tgt_kv_dim, kv_pinv_diag.get("condition_number", 0)
                    )
                    metrics["global_kv_aligned"] = True
                    metrics["global_kv_cka"] = float(kv_result.achieved_cka)
                else:
                    if config.strict_mode:
                        raise AlignmentFailureError(
                            stage="GLOBAL_KV_ALIGNMENT",
                            weight_key=None,
                            message=f"Global KV alignment failed: CKA={kv_result.achieved_cka:.4f} < 0.9999",
                            context={
                                "achieved_cka": float(kv_result.achieved_cka),
                                "source_dim": global_src_kv_dim,
                                "target_dim": global_tgt_kv_dim,
                                "samples_used": len(all_src_kv),
                            },
                        )
                    logger.warning(
                        "GLOBAL KV ALIGNMENT failed (CKA=%.4f)",
                        kv_result.achieved_cka
                    )
            else:
                # Same KV dims: use IDENTITY stitch
                global_kv_stitch_output = b.eye(global_src_kv_dim)
                global_kv_stitch_input = b.eye(global_src_kv_dim)
                b.eval(global_kv_stitch_output, global_kv_stitch_input)
                logger.info(
                    "GLOBAL KV: Same KV attention dims (%d) - using identity stitch",
                    global_src_kv_dim
                )
                metrics["global_kv_aligned"] = True
                metrics["global_kv_cka"] = 1.0  # Identity = perfect

    for layer_num, layer_idx in enumerate(layer_indices):
        # Skip layers already completed (checkpoint resume)
        if layer_idx <= resume_from_layer:
            weights_processed += len(weights_by_layer.get(layer_idx, []))
            logger.debug("TRANSPLANT: Skipping layer %d (already completed)", layer_idx)
            continue

        # Filter to specific layers if configured
        if config.transplant_layers is not None:
            if layer_idx not in config.transplant_layers:
                continue

        layer_keys = weights_by_layer.get(layer_idx, [])
        if not layer_keys:
            continue

        layer_start_time = time.time()
        logger.info(
            "TRANSPLANT: Layer %d/%d (index=%d) - %d weights",
            layer_num + 1, total_layers, layer_idx, len(layer_keys)
        )

        # Get REAL activations from collected probes (required)
        act_list = target_activations.get(layer_idx)
        if not act_list:
            continue

        if len(act_list) != len(probe_ids):
            logger.debug(
                "LAYER %d: probe count mismatch (acts=%d, probes=%d); skipping",
                layer_idx,
                len(act_list),
                len(probe_ids),
            )
            continue

        metrics["layers_considered"] += 1

        # Optional per-layer geometry analysis to align source before transplant.
        shared_source_proj = None
        shared_target_proj = None
        transform_requirements: list[str] = []
        layer_fisher_weights = None

        # Multi-space stitching: compute stitches for hidden, intermediate, AND attention dimensions
        # Hidden stitch: maps layer output activations (source hidden → target hidden)
        # Intermediate stitch: maps MLP internal activations (source intermediate → target intermediate)
        # Attention stitch: maps attention head outputs (source num_heads*head_dim → target num_heads*head_dim)
        #
        # IMPORTANT: For weight folding, we need TWO transforms per space:
        #   - F.T for OUTPUT side (left multiply): maps source output → target output
        #   - pinv(F).T for INPUT side (right multiply): inverse maps source input → target input
        #
        # Weight transform: W_target = F_out.T @ W_source @ pinv(F_in).T
        hidden_stitch_output = None  # F.T for output side [tgt_hidden, src_hidden]
        hidden_stitch_input = None   # pinv(F).T for input side [src_hidden, tgt_hidden]
        intermediate_stitch_output = None  # F.T for output side
        intermediate_stitch_input = None   # pinv(F).T for input side
        attention_stitch_output = None  # F.T for Q attention output [tgt_attn, src_attn]
        attention_stitch_input = None   # pinv(F).T for Q attention input [src_attn, tgt_attn]
        kv_stitch_output = None  # F.T for KV attention output [tgt_kv, src_kv] (GQA)
        kv_stitch_input = None   # pinv(F).T for KV attention input [src_kv, tgt_kv] (GQA)

        # Get intermediate activations for this layer (MLP internal states)
        src_inter_list = (
            source_intermediate_activations.get(layer_idx)
            if source_intermediate_activations else None
        )
        tgt_inter_list = (
            target_intermediate_activations.get(layer_idx)
            if target_intermediate_activations else None
        )

        src_act_list = source_activations.get(layer_idx) if source_activations else None
        max_samples = max(0, config.analysis_max_samples)
        analysis_samples = (
            min(len(src_act_list), len(act_list), max_samples)
            if src_act_list is not None and max_samples > 0
            else 0
        )
        src_analysis = None
        tgt_analysis = None

        if analysis_samples >= 2:
            src_analysis = b.stack(src_act_list[:analysis_samples], axis=0)
            tgt_analysis = b.stack(act_list[:analysis_samples], axis=0)
            # Convert to float32 for numerical stability in SVD operations.
            # bfloat16 causes errors in many linalg operations.
            src_analysis = b.astype(src_analysis, "float32")
            tgt_analysis = b.astype(tgt_analysis, "float32")
            b.eval(src_analysis, tgt_analysis)

        if src_analysis is not None and config.enable_shared_subspace:
            from modelcypher.core.domain.geometry.shared_subspace_projector import (
                AlignmentMethod,
                SharedSubspaceProjector,
            )
            from modelcypher.core.domain.geometry.shared_subspace_projector import (
                Config as SharedSubspaceConfig,
            )

            src_list = b.to_numpy(src_analysis).tolist()
            tgt_list = b.to_numpy(tgt_analysis).tolist()
            shared_result = SharedSubspaceProjector._discover_with_cca(
                source_activations=src_list,
                target_activations=tgt_list,
                weights=None,
                n=len(src_list),
                d_source=len(src_list[0]) if src_list else 0,
                d_target=len(tgt_list[0]) if tgt_list else 0,
                config=SharedSubspaceConfig(alignment_method=AlignmentMethod.cca),
                backend=b,
            )
            if shared_result and shared_result.is_valid:
                shared_source_proj = b.array(shared_result.source_projection)
                shared_target_proj = b.array(shared_result.target_projection)
                b.eval(shared_source_proj, shared_target_proj)
                metrics["shared_subspace_dimensions"].append(shared_result.shared_dimension)

        if src_analysis is not None and config.enable_relative_representation:
            from modelcypher.core.domain.geometry.relative_representation import (
                align_relative_representations,
                compute_relative_representation,
            )
            from modelcypher.core.domain.geometry.riemannian_utils import (
                farthest_point_sampling,
            )

            n_anchors = min(config.analysis_anchor_count, analysis_samples)
            if n_anchors >= 2:
                anchor_idx = farthest_point_sampling(tgt_analysis, n_anchors, backend=b)
                anchors = b.take(tgt_analysis, b.array(anchor_idx), axis=0)
                b.eval(anchors)

                # For cross-architecture, use GramAligner to find EXACT CKA=1.0 transform
                src_d = int(src_analysis.shape[1])
                tgt_d = int(tgt_analysis.shape[1])
                if src_d != tgt_d:
                    from modelcypher.core.domain.geometry.gram_aligner import GramAligner

                    aligner = GramAligner(b)
                    align_result = aligner.find_perfect_alignment(src_analysis, tgt_analysis)
                    if align_result.is_perfect:
                        F = b.array(align_result.feature_transform)
                        src_for_rel = b.matmul(src_analysis, F)
                        b.eval(src_for_rel)
                        logger.debug(
                            "GramAlign for relative rep: %d->%d, CKA=%.4f",
                            src_d, tgt_d, align_result.achieved_cka
                        )
                    else:
                        # GramAligner failed - skip relative representation for this layer
                        logger.debug(
                            "GramAlign failed for relative rep (CKA=%.4f), skipping",
                            align_result.achieved_cka
                        )
                        src_for_rel = src_analysis
                else:
                    src_for_rel = src_analysis

                src_rel = compute_relative_representation(src_for_rel, anchors)
                tgt_rel = compute_relative_representation(tgt_analysis, anchors)
                b.eval(src_rel, tgt_rel)
                _, rel_error = align_relative_representations(src_rel, tgt_rel)
                metrics["relative_rep_errors"].append(rel_error)

        if src_analysis is not None and config.enable_fisher_weighting:
            # Fisher weighting requires same dimensions for per-feature weights
            src_d = int(src_analysis.shape[1])
            tgt_d = int(tgt_analysis.shape[1])
            if src_d == tgt_d:
                src_var = b.var(src_analysis, axis=0)
                tgt_var = b.var(tgt_analysis, axis=0)
                b.eval(src_var, tgt_var)
                eps = division_epsilon(b, src_var)
                src_fisher = 1.0 / (src_var + eps)
                tgt_fisher = 1.0 / (tgt_var + eps)
                total_fisher = src_fisher + tgt_fisher + eps
                layer_fisher_weights = tgt_fisher / total_fisher
                b.eval(layer_fisher_weights)
                mean_arr = b.mean(layer_fisher_weights)
                b.eval(mean_arr)
                metrics["fisher_target_means"].append(float(b.to_numpy(mean_arr).item()))
            else:
                # Cross-architecture: use scalar Fisher based on total variance
                src_total_var = b.var(src_analysis)
                tgt_total_var = b.var(tgt_analysis)
                b.eval(src_total_var, tgt_total_var)
                eps = division_epsilon(b, src_total_var)
                src_total_var_f = float(b.to_numpy(src_total_var))
                tgt_total_var_f = float(b.to_numpy(tgt_total_var))
                src_fisher_scalar = 1.0 / (src_total_var_f + eps)
                tgt_fisher_scalar = 1.0 / (tgt_total_var_f + eps)
                total_fisher_scalar = src_fisher_scalar + tgt_fisher_scalar + eps
                fisher_weight_scalar = tgt_fisher_scalar / total_fisher_scalar
                metrics["fisher_target_means"].append(fisher_weight_scalar)

        if analyzer is not None:
            relation = layer_relations.get(layer_idx)
            if relation is not None:
                analysis = analyzer.analyze(
                    volume_a=relation.volume_a,
                    volume_b=relation.volume_b,
                    relation=relation,
                )
                transform_requirements = [t.value for t in analysis.transformations]
                metrics["transform_requirements_by_layer"][layer_idx] = transform_requirements

        # =================================================================
        # GRAM ALIGNMENT: Find EXACT CKA = 1.0 transforms for hidden AND intermediate
        # =================================================================
        # GramAligner finds the mathematically guaranteed transformation that achieves
        # CKA = 1.0. This is not an approximation - it's the exact solution.
        #
        # The feature_transform maps source activations to target space such that
        # their Gram matrices (relational geometry) are IDENTICAL.
        #
        # MLP weights have shape [intermediate, hidden] or [hidden, intermediate].
        # We need transforms for BOTH axes to properly map weights.

        from modelcypher.core.domain.geometry.gram_aligner import GramAligner

        # USE GLOBAL TRAJECTORY-ALIGNED hidden stitch (computed once for all layers)
        # This ensures consistent alignment across the entire model trajectory.
        if global_hidden_stitch_output is not None:
            hidden_stitch_output = global_hidden_stitch_output
            hidden_stitch_input = global_hidden_stitch_input
            # Log only on first layer to avoid spam
            if layer_num == 0:
                logger.info(
                    "Layer %d: Using GLOBAL hidden stitch (%d→%d)",
                    layer_idx, global_src_hidden_dim, global_tgt_hidden_dim
                )

        # NO PER-LAYER FALLBACK: If global alignment failed, we already raised an exception
        # in the multi-bottleneck strategy above. Per-layer alignment breaks compositional
        # structure - each layer would get a different F, causing activations to not compose
        # correctly through the network. If we get here without global stitch, something
        # is wrong and we should not silently continue.

        # Compute INTERMEDIATE alignment (source intermediate_dim → target intermediate_dim)
        # Note: Intermediate stitch is per-layer since each MLP has different internal geometry
        if src_inter_list is not None and tgt_inter_list is not None:
            n_inter_samples = min(len(src_inter_list), len(tgt_inter_list))
            if n_inter_samples >= 10:
                src_inter = b.stack(src_inter_list[:n_inter_samples], axis=0)
                tgt_inter = b.stack(tgt_inter_list[:n_inter_samples], axis=0)
                src_inter = b.astype(src_inter, "float32")
                tgt_inter = b.astype(tgt_inter, "float32")
                b.eval(src_inter, tgt_inter)

                src_inter_dim = int(src_inter.shape[1])
                tgt_inter_dim = int(tgt_inter.shape[1])

                if src_inter_dim != tgt_inter_dim:
                    # Use GramAligner - finds EXACT CKA = 1.0 transform
                    aligner = GramAligner(b)
                    inter_result = aligner.find_perfect_alignment(src_inter, tgt_inter)

                    if inter_result.is_perfect:
                        # feature_transform F is [d_source, d_target]
                        # source @ F → target (activation alignment)
                        #
                        # For weight folding W_target = F_out.T @ W_source @ pinv(F_in).T:
                        #   - F.T [d_target, d_source] for OUTPUT side (left multiply)
                        #   - pinv(F).T [d_source, d_target] for INPUT side (right multiply)
                        F = b.array(inter_result.feature_transform)
                        b.eval(F)
                        intermediate_stitch_output = b.transpose(F)  # F.T [tgt, src]
                        F_pinv, inter_pinv_diag = safe_pinv(b, F)
                        intermediate_stitch_input = b.transpose(F_pinv)  # pinv(F).T [src, tgt]
                        b.eval(intermediate_stitch_output, intermediate_stitch_input)
                        logger.info(
                            "Layer %d: Intermediate GramAlign CKA=%.4f (%d→%d) cond=%.2e",
                            layer_idx, inter_result.achieved_cka,
                            src_inter_dim, tgt_inter_dim, inter_pinv_diag.get("condition_number", 0)
                        )
                        metrics.setdefault("intermediate_gram_aligned", 0)
                        metrics["intermediate_gram_aligned"] += 1
                    else:
                        logger.warning(
                            "Layer %d: Intermediate GramAlign failed (CKA=%.4f)",
                            layer_idx, inter_result.achieved_cka
                        )

        # USE GLOBAL ATTENTION-ALIGNED stitch (computed once for all layers)
        # This ensures consistent alignment for attention weights across layers.
        if global_attention_stitch_output is not None:
            attention_stitch_output = global_attention_stitch_output
            attention_stitch_input = global_attention_stitch_input
            # Log only on first layer to avoid spam
            if layer_num == 0:
                logger.info(
                    "Layer %d: Using GLOBAL Q attention stitch (%d→%d)",
                    layer_idx, global_src_attn_dim, global_tgt_attn_dim
                )

        # USE GLOBAL KV-ALIGNED stitch for GQA models (k_proj/v_proj have different dims)
        if global_kv_stitch_output is not None:
            kv_stitch_output = global_kv_stitch_output
            kv_stitch_input = global_kv_stitch_input
            # Log only on first layer to avoid spam
            if layer_num == 0:
                logger.info(
                    "Layer %d: Using GLOBAL KV stitch (%d→%d)",
                    layer_idx, global_src_kv_dim, global_tgt_kv_dim
                )

        stacked = b.stack(act_list, axis=0)
        # Convert to float32 for numerical stability in linalg operations.
        stacked = b.astype(stacked, "float32")
        b.eval(stacked)

        partition = partition_core_boundary(
            activations=stacked,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            boundary_k=config.boundary_k,
            geodesic_k_neighbors=config.geodesic_k_neighbors,
            backend=b,
        )

        if not partition.core_indices:
            continue

        core_indices = b.array(partition.core_indices, dtype="int32")
        core_acts = b.take(stacked, core_indices, axis=0)
        b.eval(core_acts)

        if partition.boundary_indices:
            boundary_indices = b.array(partition.boundary_indices, dtype="int32")
            boundary_acts = b.take(stacked, boundary_indices, axis=0)
            b.eval(boundary_acts)
        else:
            boundary_acts = b.zeros((0, int(stacked.shape[1])))
            b.eval(boundary_acts)

        layer_transplanted = False
        best_alignment: dict[str, float] | None = None
        best_delta_norm = -1.0
        can_measure_alignment = core_acts is not None and int(core_acts.shape[0]) >= 2

        for weight_num, key in enumerate(layer_keys):
            weights_processed += 1
            weight_start_time = time.time()

            # Skip quantization metadata and non-weight tensors (only transplant actual matrices).
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            if not key.endswith(".weight"):
                continue

            # Progress callback for external monitoring
            if config.progress_callback:
                config.progress_callback(
                    f"Layer {layer_idx}: {key}",
                    weights_processed,
                    total_weights,
                )

            target_w = target_weights.get(key)
            source_w = source_weights.get(key)
            if target_w is None or source_w is None:
                continue

            metrics["weights_considered"] += 1

            # Skip non-2D weights (bias vectors, etc)
            if not hasattr(target_w, "shape") or not hasattr(source_w, "shape"):
                continue
            if len(target_w.shape) != 2 or len(source_w.shape) != 2:
                continue

            # Dequantize quantized weights (uint32/int dtypes) using scales/biases
            target_dtype = str(getattr(target_w, 'dtype', '')).lower()
            source_dtype = str(getattr(source_w, 'dtype', '')).lower()

            if 'int' in target_dtype or 'uint' in target_dtype:
                logger.debug("Dequantizing target weight: %s", key)
                target_w = dequantize_if_needed(target_w, key, target_weights, b)
                if target_w is None or not hasattr(target_w, 'shape'):
                    logger.debug("Failed to dequantize target weight: %s", key)
                    continue
                # Check if still quantized after dequantize attempt
                target_dtype = str(getattr(target_w, 'dtype', '')).lower()
                if 'int' in target_dtype or 'uint' in target_dtype:
                    logger.debug("Skipping still-quantized target weight: %s", key)
                    continue

            if 'int' in source_dtype or 'uint' in source_dtype:
                logger.debug("Dequantizing source weight: %s", key)
                source_w = dequantize_if_needed(source_w, key, source_weights, b)
                if source_w is None or not hasattr(source_w, 'shape'):
                    logger.debug("Failed to dequantize source weight: %s", key)
                    continue
                # Check if still quantized after dequantize attempt
                source_dtype = str(getattr(source_w, 'dtype', '')).lower()
                if 'int' in source_dtype or 'uint' in source_dtype:
                    logger.debug("Skipping still-quantized source weight: %s", key)
                    continue

            # Skip if shapes became non-2D after dequantization
            if len(target_w.shape) != 2 or len(source_w.shape) != 2:
                continue

            try:
                # Convert to float32 backend arrays
                logger.debug("Converting weight %s to float32", key)
                target_w = b.astype(b.array(target_w), "float32")
                source_w = b.astype(b.array(source_w), "float32")
                b.eval(target_w, source_w)
                logger.debug("Converted %s: target=%s, source=%s", key, target_w.shape, source_w.shape)
            except Exception as e:
                logger.warning("Failed to convert weight %s to float32: %s", key, e)
                continue

            source_candidate = source_w
            shared_subspace_applied = False

            if shared_source_proj is not None and shared_target_proj is not None:
                try:
                    if (
                        source_candidate.shape[1] == shared_source_proj.shape[0]
                        and target_w.shape[1] == shared_target_proj.shape[0]
                    ):
                        source_shared = b.matmul(source_candidate, shared_source_proj)
                        source_candidate = b.matmul(
                            source_shared,
                            b.transpose(shared_target_proj),
                        )
                        b.eval(source_candidate)
                        shared_subspace_applied = True
                        metrics["shared_subspace_applied"] += 1
                except Exception as e:
                    logger.debug("Shared subspace projection failed for %s: %s", key, e)

            if target_w.shape != source_candidate.shape:
                # =================================================================
                # MULTI-SPACE STITCHING: Apply pre-computed stitches to weights
                # =================================================================
                # MLP weights need BOTH hidden AND intermediate stitches:
                #   gate_proj [intermediate, hidden] → [tgt_intermediate, tgt_hidden]
                #   up_proj   [intermediate, hidden] → [tgt_intermediate, tgt_hidden]
                #   down_proj [hidden, intermediate] → [tgt_hidden, tgt_intermediate]
                #
                # Attention weights only need hidden stitch:
                #   q_proj, k_proj, v_proj [hidden, head*num_heads] → [tgt_hidden, ...]
                #   o_proj [head*num_heads, hidden] → [..., tgt_hidden]

                # Use ORIGINAL source shape for dimension matching (source_candidate may
                # have been partially transformed by shared_subspace)
                original_source_shape = source_w.shape
                is_mlp = any(mlp_name in key for mlp_name in [
                    "gate_proj", "up_proj", "down_proj", "mlp.fc1", "mlp.fc2"
                ])

                # =================================================================
                # MIRROR PROBLEM FIX: Correct weight folding transforms
                # =================================================================
                # GramAligner finds F such that: source @ F → target (activation alignment)
                #
                # For WEIGHT folding, we need the inverse on the INPUT side:
                #   W_target = F_out.T @ W_source @ pinv(F_in).T
                #
                # Where:
                #   - F_out.T [tgt_out, src_out] = stitch_output (left multiply for output dim)
                #   - pinv(F_in).T [src_in, tgt_in] = stitch_input (right multiply for input dim)
                #
                # Weight shapes: [output_features, input_features]
                #   - gate_proj/up_proj: [intermediate, hidden] (out=inter, in=hidden)
                #   - down_proj:         [hidden, intermediate] (out=hidden, in=inter)
                #   - attention:         [hidden, hidden] (both dims hidden)

                if is_mlp and hidden_stitch_output is not None and intermediate_stitch_output is not None:
                    # MLP weight: apply BOTH stitches with correct orientations
                    # stitch_output for OUTPUT side (rows), stitch_input for INPUT side (cols)
                    src_hidden_dim = int(hidden_stitch_output.shape[1])  # F.T is [tgt, src]
                    tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                    src_inter_dim = int(intermediate_stitch_output.shape[1])
                    tgt_inter_dim = int(intermediate_stitch_output.shape[0])

                    logger.info(
                        "MLP weight %s: applying dual stitch (hidden %d→%d, inter %d→%d)",
                        key, src_hidden_dim, tgt_hidden_dim, src_inter_dim, tgt_inter_dim
                    )

                    # Determine which dimension is which from ORIGINAL source shape
                    dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                    if dim0 == src_inter_dim and dim1 == src_hidden_dim:
                        # gate_proj/up_proj: [intermediate, hidden] → [tgt_inter, tgt_hidden]
                        # Output=intermediate (rows), Input=hidden (cols)
                        # W_target = inter_stitch_output @ W @ hidden_stitch_input
                        source_aligned = b.matmul(intermediate_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                        b.eval(source_aligned)
                        logger.info("Dual stitch (gate/up): [%d,%d] → [%d,%d]",
                                    dim0, dim1, tgt_inter_dim, tgt_hidden_dim)

                    elif dim0 == src_hidden_dim and dim1 == src_inter_dim:
                        # down_proj: [hidden, intermediate] → [tgt_hidden, tgt_inter]
                        # Output=hidden (rows), Input=intermediate (cols)
                        # W_target = hidden_stitch_output @ W @ inter_stitch_input
                        source_aligned = b.matmul(hidden_stitch_output, source_w)
                        source_aligned = b.matmul(source_aligned, intermediate_stitch_input)
                        b.eval(source_aligned)
                        logger.info("Dual stitch (down): [%d,%d] → [%d,%d]",
                                    dim0, dim1, tgt_hidden_dim, tgt_inter_dim)

                    else:
                        logger.warning(
                            "MLP weight %s shape [%d,%d] doesn't match expected dims "
                            "(hidden=%d, inter=%d) - skipping",
                            key, dim0, dim1, src_hidden_dim, src_inter_dim
                        )
                        continue

                    metrics.setdefault("dual_stitch_applied", 0)
                    metrics["dual_stitch_applied"] += 1

                elif hidden_stitch_output is not None and hidden_stitch_input is not None:
                    # =================================================================
                    # ATTENTION WEIGHT STITCHING (replaces previous skip logic)
                    # =================================================================
                    # A structural incompatibility IS a geometric one. Head count difference
                    # is just another dimension mismatch that GramAligner can solve.
                    #
                    # Example: SmolLM (15 heads) → Qwen (14 heads)
                    #   - q_proj [960, 960] → [896, 896]
                    #   - 960 = 15 * 64 (SmolLM attention dim)
                    #   - 896 = 14 * 64 (Qwen attention dim)
                    #   - GramAligner finds F such that CKA(src_attn @ F, tgt_attn) = 1.0
                    #
                    # Weight stitching for attention:
                    #   - q_proj/k_proj/v_proj: [attn_dim, hidden_dim]
                    #     → attention_stitch_output @ W @ hidden_stitch_input
                    #   - o_proj: [hidden_dim, attn_dim]
                    #     → hidden_stitch_output @ W @ attention_stitch_input
                    is_attention = any(attn_name in key for attn_name in [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "self_attn", "query", "key", "value",
                    ])

                    attention_stitch_applied = False

                    if is_attention and attention_stitch_output is not None:
                        # We have attention stitch - apply it!
                        src_attn_dim = int(attention_stitch_output.shape[1])
                        tgt_attn_dim = int(attention_stitch_output.shape[0])
                        src_hidden_dim = int(hidden_stitch_output.shape[1])
                        tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                        dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                        # Get KV stitch dimensions for GQA detection
                        # GQA models: K/V have fewer heads than Q (e.g., SmolLM: Q=960, KV=320)
                        src_kv_dim = int(kv_stitch_output.shape[1]) if kv_stitch_output is not None else src_attn_dim
                        tgt_kv_dim = int(kv_stitch_output.shape[0]) if kv_stitch_output is not None else tgt_attn_dim

                        # Determine attention weight pattern
                        # GQA: q_proj uses Q-attention dim, k_proj/v_proj use KV-attention dim
                        is_q = any(n in key for n in ["q_proj", "query"])
                        is_kv = any(n in key for n in ["k_proj", "v_proj", "key", "value"])
                        is_o = any(n in key for n in ["o_proj"])

                        if is_q and dim0 == src_attn_dim and dim1 == src_hidden_dim:
                            # q_proj: [Q_attn, hidden] → attention_stitch @ W @ hidden_stitch
                            source_aligned = b.matmul(attention_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info(
                                "Attention stitch (q_proj): %s [%d,%d] → [%d,%d]",
                                key, dim0, dim1, tgt_attn_dim, tgt_hidden_dim
                            )
                            metrics.setdefault("attention_stitched", 0)
                            metrics["attention_stitched"] += 1
                            attention_stitch_applied = True

                        elif is_kv and dim0 == src_kv_dim and dim1 == src_hidden_dim:
                            # k_proj/v_proj (GQA): [KV_attn, hidden] → kv_stitch @ W @ hidden_stitch
                            kv_out = kv_stitch_output if kv_stitch_output is not None else attention_stitch_output
                            source_aligned = b.matmul(kv_out, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            stitch_type = "KV stitch" if kv_stitch_output is not None else "attention stitch"
                            logger.info(
                                "%s (k/v_proj): %s [%d,%d] → [%d,%d]",
                                stitch_type, key, dim0, dim1, tgt_kv_dim, tgt_hidden_dim
                            )
                            metrics.setdefault("kv_stitched", 0)
                            metrics["kv_stitched"] += 1
                            attention_stitch_applied = True

                        elif is_o and dim0 == src_hidden_dim and dim1 == src_attn_dim:
                            # o_proj: [hidden, attn] → hidden_stitch_output @ W @ attention_stitch_input
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, attention_stitch_input)
                            b.eval(source_aligned)
                            logger.info(
                                "Attention stitch (o_proj): %s [%d,%d] → [%d,%d]",
                                key, dim0, dim1, tgt_hidden_dim, tgt_attn_dim
                            )
                            metrics.setdefault("attention_stitched", 0)
                            metrics["attention_stitched"] += 1
                            attention_stitch_applied = True

                        else:
                            # Unexpected attention shape - log and try hidden-only stitch
                            logger.warning(
                                "Attention weight %s shape [%d,%d] doesn't match expected "
                                "(attn=%d, hidden=%d) - trying hidden stitch",
                                key, dim0, dim1, src_attn_dim, src_hidden_dim
                            )
                            # Fall through to hidden-only stitch below

                    elif is_attention and attention_stitch_output is None:
                        # No attention stitch available - this is a critical failure
                        if config.strict_mode:
                            raise StitchUnavailableError(
                                stage="ATTENTION_WEIGHT_STITCH",
                                weight_key=key,
                                message="No attention stitch available for cross-architecture merge",
                                context={
                                    "source_shape": list(source_w.shape),
                                    "target_shape": list(target_w.shape),
                                    "stitch_type": "attention",
                                    "reason": "Global attention alignment failed or had insufficient samples",
                                },
                            )
                        logger.info(
                            "Cross-arch: SKIPPING attention weight %s (no attention stitch available)",
                            key
                        )
                        metrics.setdefault("attention_skipped_no_stitch", 0)
                        metrics["attention_skipped_no_stitch"] += 1
                        continue

                    # Non-attention weight with hidden dimensions (e.g., layer norm)
                    # Skip hidden stitch if attention stitch was already applied
                    if not attention_stitch_applied:
                        # W_target = hidden_stitch_output @ W @ hidden_stitch_input
                        src_hidden_dim = int(hidden_stitch_output.shape[1])
                        tgt_hidden_dim = int(hidden_stitch_output.shape[0])
                        dim0, dim1 = int(original_source_shape[0]), int(original_source_shape[1])

                        if dim0 == src_hidden_dim and dim1 == src_hidden_dim:
                            # BOTH dimensions are hidden_dim
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            source_aligned = b.matmul(source_aligned, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (both dims): [%d,%d] → [%d,%d]",
                                        dim0, dim1, tgt_hidden_dim, tgt_hidden_dim)

                        elif dim0 == src_hidden_dim:
                            # Hidden dim is OUTPUT only (rows): hidden_stitch_output @ W
                            source_aligned = b.matmul(hidden_stitch_output, source_w)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (output only): [%d,%d] → [%d,%d]",
                                        dim0, dim1, tgt_hidden_dim, dim1)

                        elif dim1 == src_hidden_dim:
                            # Hidden dim is INPUT only (cols): W @ hidden_stitch_input
                            source_aligned = b.matmul(source_w, hidden_stitch_input)
                            b.eval(source_aligned)
                            logger.info("Hidden stitch (input only): [%d,%d] → [%d,%d]",
                                        dim0, dim1, dim0, tgt_hidden_dim)

                        else:
                            if config.strict_mode:
                                raise DimensionMismatchError(
                                    stage="HIDDEN_WEIGHT_STITCH",
                                    weight_key=key,
                                    message=f"Weight shape [{dim0},{dim1}] doesn't match hidden_dim {src_hidden_dim}",
                                    context={
                                        "weight_shape": [dim0, dim1],
                                        "expected_hidden_dim": src_hidden_dim,
                                        "stitch_type": "hidden",
                                    },
                                )
                            logger.warning(
                                "Weight %s shape [%d,%d] doesn't match hidden_dim %d - skipping",
                                key, dim0, dim1, src_hidden_dim
                            )
                            continue

                        metrics.setdefault("hidden_stitch_applied", 0)
                        metrics["hidden_stitch_applied"] += 1

                else:
                    if config.strict_mode:
                        raise StitchUnavailableError(
                            stage="CROSS_ARCHITECTURE_STITCH",
                            weight_key=key,
                            message="Cross-architecture weight but no stitch transformation available",
                            context={
                                "source_shape": list(source_w.shape),
                                "target_shape": list(target_w.shape),
                                "reason": "No hidden or attention stitch computed",
                            },
                        )
                    logger.warning(
                        "Cross-architecture weight %s but no stitch available - skipping",
                        key
                    )
                    continue

                # Verify final shape matches target
                if source_aligned.shape != target_w.shape:
                    if config.strict_mode:
                        raise DimensionMismatchError(
                            stage="POST_STITCH_VALIDATION",
                            weight_key=key,
                            message=f"Shape mismatch after stitch: {source_aligned.shape} vs {target_w.shape}",
                            context={
                                "aligned_shape": list(source_aligned.shape),
                                "target_shape": list(target_w.shape),
                            },
                        )
                    logger.warning(
                        "Shape mismatch after stitch: %s vs %s for %s - skipping",
                        source_aligned.shape, target_w.shape, key
                    )
                    continue

            else:
                source_aligned = source_candidate

            if (
                not shared_subspace_applied
                and transform_requirements
                and source_aligned.shape == target_w.shape
                and "procrustes_rotation" in {t.lower() for t in transform_requirements}
            ):
                try:
                    from modelcypher.core.domain.geometry.backend_matrix_utils import (
                        BackendMatrixUtils,
                    )

                    rotation = BackendMatrixUtils(b).procrustes_rotation(
                        source_aligned, target_w
                    ).rotation
                    source_aligned = b.matmul(source_aligned, rotation)
                    b.eval(source_aligned)
                    metrics["procrustes_applied"] += 1
                except Exception as e:
                    logger.debug("Procrustes alignment failed for %s: %s", key, e)

            if layer_fisher_weights is not None:
                if source_aligned.shape[1] == int(layer_fisher_weights.shape[0]):
                    weights = b.reshape(layer_fisher_weights, (1, -1))
                    delta = source_aligned - target_w
                    source_aligned = target_w + delta * (1.0 - weights)
                    b.eval(source_aligned)
                    metrics["fisher_delta_scaled"] += 1

            try:
                logger.debug("Computing transplant delta for %s", key)
                result = compute_transplant_delta(
                    weight_target=target_w,
                    weight_source_aligned=source_aligned,
                    activations_core=core_acts,
                    activations_boundary=boundary_acts,
                    backend=b,
                )
                logger.debug("Transplant delta computed for %s: applied=%s", key, result.applied)
            except Exception as e:
                logger.warning("Failed to compute transplant delta for %s: %s", key, e)
                continue

            if result.applied:
                # Use geometry-determined transplant result directly.
                # The null-space projection already computed preserved_fraction
                # based on the spectral structure of boundary activations.
                merged[key] = result.merged_weight
                metrics["weights_transplanted"] += 1
                metrics["preserved_fractions"].append(result.preserved_fraction)
                metrics["projection_losses"].append(result.projection_loss)
                metrics["null_dims"].append(result.null_dim)

                weight_elapsed = time.time() - weight_start_time
                logger.debug(
                    "TRANSPLANT: Weight %d/%d %s - %.2fs (preserved=%.3f, loss=%.6f)",
                    weight_num + 1, len(layer_keys), key,
                    weight_elapsed, result.preserved_fraction, result.projection_loss
                )
                # Use the actual stored weight (may be alpha-scaled)
                actual_merged_weight = merged[key]
                if can_measure_alignment and result.delta_norm > best_delta_norm:
                    try:
                        best_alignment = _compute_alignment_metrics(
                            core_acts=core_acts,
                            weight_before=target_w,
                            weight_after=actual_merged_weight,
                            weight_source=source_aligned,
                            backend=b,
                        )
                        best_delta_norm = result.delta_norm
                    except Exception as e:
                        logger.debug("Alignment metrics failed for %s: %s", key, e)
                if int(boundary_acts.shape[0]) > 0:
                    # Project boundary activations to match weight dimensions
                    # (handles cross-architecture dimension mismatches)
                    boundary_acts_proj = _project_activations_to_weight_dim(
                        boundary_acts, target_w, b
                    )
                    target_output = b.matmul(boundary_acts_proj, b.transpose(target_w))
                    merged_output = b.matmul(
                        boundary_acts_proj, b.transpose(actual_merged_weight)
                    )
                    diff = merged_output - target_output
                    diff_norm_arr = b.norm(b.reshape(diff, (-1,)))
                    target_norm_arr = b.norm(b.reshape(target_output, (-1,)))
                    b.eval(diff_norm_arr, target_norm_arr)

                    diff_norm = float(b.to_numpy(diff_norm_arr))
                    target_norm = float(b.to_numpy(target_norm_arr))
                    eps = float(machine_epsilon(b, target_w))

                    if target_norm > eps:
                        relative_diff = diff_norm / target_norm
                    else:
                        relative_diff = 0.0 if diff_norm <= eps else float("inf")

                    metrics["boundary_relative_diffs"].append(relative_diff)
                layer_transplanted = True

        if layer_transplanted:
            metrics["layers_transplanted"] += 1

        # Layer timing summary
        layer_elapsed = time.time() - layer_start_time
        logger.info(
            "TRANSPLANT: Layer %d complete - %.2fs (%d weights transplanted)",
            layer_idx, layer_elapsed, metrics["weights_transplanted"]
        )

        # Save checkpoint after each layer
        if config.checkpoint_dir:
            _save_checkpoint(config.checkpoint_dir, layer_idx, merged, metrics)

        if best_alignment is not None:
            metrics["alignment_improvements"].append(best_alignment["alignment_improvement"])
            metrics["core_dist_to_source_before"].append(
                best_alignment["core_dist_to_source_before"]
            )
            metrics["core_dist_to_source_after"].append(
                best_alignment["core_dist_to_source_after"]
            )
            metrics["cka_before"].append(best_alignment["cka_before"])
            metrics["cka_after"].append(best_alignment["cka_after"])

    if metrics["preserved_fractions"]:
        pres = metrics["preserved_fractions"]
        metrics["mean_preserved_fraction"] = sum(pres) / len(pres)
    if metrics["projection_losses"]:
        losses = metrics["projection_losses"]
        metrics["mean_projection_loss"] = sum(losses) / len(losses)
    if metrics["null_dims"]:
        null_dims = metrics["null_dims"]
        metrics["mean_null_dim"] = sum(null_dims) / len(null_dims)
    if metrics["boundary_relative_diffs"]:
        diffs = metrics["boundary_relative_diffs"]
        metrics["mean_boundary_relative_diff"] = sum(diffs) / len(diffs)
        metrics["max_boundary_relative_diff"] = max(diffs)
    if metrics["alignment_improvements"]:
        improvements = metrics["alignment_improvements"]
        metrics["alignment_samples"] = len(improvements)
        metrics["mean_alignment_improvement"] = sum(improvements) / len(improvements)
    if metrics["core_dist_to_source_before"]:
        dists = metrics["core_dist_to_source_before"]
        metrics["mean_core_dist_to_source_before"] = sum(dists) / len(dists)
    if metrics["core_dist_to_source_after"]:
        dists = metrics["core_dist_to_source_after"]
        metrics["mean_core_dist_to_source_after"] = sum(dists) / len(dists)
    if metrics["cka_before"]:
        ckas = metrics["cka_before"]
        metrics["mean_cka_before"] = sum(ckas) / len(ckas)
    if metrics["cka_after"]:
        ckas = metrics["cka_after"]
        metrics["mean_cka_after"] = sum(ckas) / len(ckas)
    if metrics["shared_subspace_dimensions"]:
        dims = metrics["shared_subspace_dimensions"]
        metrics["mean_shared_subspace_dimension"] = sum(dims) / len(dims)
    if metrics["relative_rep_errors"]:
        errors = metrics["relative_rep_errors"]
        metrics["mean_relative_rep_error"] = sum(errors) / len(errors)
    if metrics["fisher_target_means"]:
        means = metrics["fisher_target_means"]
        metrics["mean_fisher_target_weight"] = sum(means) / len(means)
    if metrics["transform_requirements_by_layer"]:
        counts: dict[str, int] = {}
        for reqs in metrics["transform_requirements_by_layer"].values():
            for req in reqs:
                counts[req] = counts.get(req, 0) + 1
        metrics["transform_requirements_counts"] = counts

    # Stage completion summary
    stage_elapsed = time.time() - stage_start_time
    metrics["total_time_seconds"] = stage_elapsed
    logger.info(
        "TRANSPLANT: Stage 3 complete - %.2fs total (%d/%d layers, %d/%d weights)",
        stage_elapsed,
        metrics["layers_transplanted"], metrics["layers_considered"],
        metrics["weights_transplanted"], metrics["weights_considered"],
    )

    # Convert all weights to bfloat16 for consistent output format.
    # This handles the case where original weights are bf16 but transplanted
    # weights are float32 from the numerical computations.
    logger.debug("Converting %d merged weights to bfloat16 for output", len(merged))
    output_weights: dict[str, Any] = {}
    for key, weight in merged.items():
        if hasattr(weight, 'dtype'):
            dtype_str = str(weight.dtype).lower()
            # Skip quantized weights (uint32, int4, etc) - keep as-is
            if 'int' in dtype_str or 'uint' in dtype_str:
                output_weights[key] = weight
            else:
                # Convert to bfloat16 for storage efficiency
                try:
                    output_weights[key] = b.astype(b.array(weight), "bfloat16")
                except Exception as e:
                    logger.debug("Could not convert %s to bfloat16: %s", key, e)
                    output_weights[key] = weight
        else:
            output_weights[key] = weight

    return TransplantStageResult(merged_weights=output_weights, metrics=metrics)


__all__ = [
    "TransplantStageConfig",
    "TransplantStageResult",
    "stage_transplant",
]
