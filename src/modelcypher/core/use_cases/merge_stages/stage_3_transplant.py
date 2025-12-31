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
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.transplant import (
    compute_transplant_delta,
    partition_core_boundary,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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


@dataclass
class TransplantStageResult:
    """Result of transplant stage."""

    merged_weights: dict[str, Any]
    metrics: dict[str, Any]


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
    """Measure core alignment shift toward the source for a single weight."""
    from modelcypher.core.domain.geometry.cka import compute_cka

    output_before = backend.matmul(core_acts, backend.transpose(weight_before))
    output_after = backend.matmul(core_acts, backend.transpose(weight_after))
    output_source = backend.matmul(core_acts, backend.transpose(weight_source))
    backend.eval(output_before, output_after, output_source)

    dist_before_arr = backend.norm(output_before - output_source)
    dist_after_arr = backend.norm(output_after - output_source)
    backend.eval(dist_before_arr, dist_after_arr)

    dist_before = float(backend.to_numpy(dist_before_arr))
    dist_after = float(backend.to_numpy(dist_after_arr))

    eps = float(machine_epsilon(backend, weight_before))
    if dist_before > eps:
        alignment_improvement = (dist_before - dist_after) / dist_before
    else:
        alignment_improvement = 0.0

    cka_before = compute_cka(output_before, output_source, backend=backend)
    cka_after = compute_cka(output_after, output_source, backend=backend)

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
            "Use `mc geometry transplant run` to collect activations before merging."
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
            MergeAnalyzer,
            MergeAnalysisConfig,
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
            b.eval(src_analysis, tgt_analysis)

        if src_analysis is not None and config.enable_shared_subspace:
            from modelcypher.core.domain.geometry.shared_subspace_projector import (
                AlignmentMethod,
                Config as SharedSubspaceConfig,
                SharedSubspaceProjector,
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
                src_rel = compute_relative_representation(src_analysis, anchors)
                tgt_rel = compute_relative_representation(tgt_analysis, anchors)
                b.eval(src_rel, tgt_rel)
                _, rel_error = align_relative_representations(src_rel, tgt_rel)
                metrics["relative_rep_errors"].append(rel_error)

        if src_analysis is not None and config.enable_fisher_weighting:
            epsilon = 1e-6
            src_var = b.var(src_analysis, axis=0)
            tgt_var = b.var(tgt_analysis, axis=0)
            b.eval(src_var, tgt_var)
            src_fisher = 1.0 / (src_var + epsilon)
            tgt_fisher = 1.0 / (tgt_var + epsilon)
            total_fisher = src_fisher + tgt_fisher + epsilon
            layer_fisher_weights = tgt_fisher / total_fisher
            b.eval(layer_fisher_weights)
            mean_arr = b.mean(layer_fisher_weights)
            b.eval(mean_arr)
            metrics["fisher_target_means"].append(float(b.to_numpy(mean_arr).item()))

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

        stacked = b.stack(act_list, axis=0)
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
                try:
                    logger.debug(
                        "Projecting %s: source=%s -> target=%s",
                        key,
                        source_candidate.shape,
                        target_w.shape,
                    )
                    projection = project_cross_dimensional(
                        source=source_candidate,
                        target=target_w,
                        method=config.projection_method,
                        backend=b,
                    )
                    source_aligned = projection.projected
                    logger.debug("Projected %s successfully", key)
                except Exception as e:
                    logger.warning("Failed to project weight %s: %s", key, e)
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
                    target_output = b.matmul(boundary_acts, b.transpose(target_w))
                    merged_output = b.matmul(
                        boundary_acts, b.transpose(actual_merged_weight)
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
