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

"""Stage 2: DENSITY - Knowledge density profiling and graft mask computation.

Identifies which concepts (probe_id, layer) should be grafted based on density:
- High density in source + Low density in target = GRAFT (fill the gap)
- Low density in source OR High density in target = SKIP (nothing to add)

This stage MUST run between probe (stage 1) and transplant (stage 3) to enable
selective grafting that improves the merged model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    LayerDensityProfile,
    ModelDensityProfile,
    PointCloudDensityResult,
    compute_knn_point_cloud_density,
    compute_density_weights,
)
from modelcypher.core.domain.geometry.knowledge_diff import (
    KnowledgeDiff,
    KnowledgeDiffer,
    compute_graft_mask,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class DensityStageResult:
    """Result of density analysis stage."""

    # Density profiles for both models
    source_profile: ModelDensityProfile
    target_profile: ModelDensityProfile

    # Knowledge diff showing graft opportunities
    knowledge_diff: KnowledgeDiff

    # The graft mask: probe_id -> layer -> should_graft
    graft_mask: dict[str, dict[int, bool]]

    # Point cloud density per layer (k-NN based)
    point_cloud_densities: dict[int, PointCloudDensityResult]

    # Density-derived transfer weights per layer
    density_weights: dict[int, "Array"]

    # Metrics
    metrics: dict[str, float | int]


class ActivationProviderFromDict:
    """Adapts pre-collected activations dict to ActivationProvider protocol."""

    def __init__(
        self,
        activations: dict[int, list["Array"]],
        probe_ids: list[str],
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with pre-collected activations.

        Args:
            activations: Dict mapping layer_idx -> list of activations.
                Each activation corresponds to a probe_id in order.
            probe_ids: List of probe IDs in same order as activations.
            backend: Backend for tensor operations.
        """
        self._activations = activations
        self._probe_ids = probe_ids
        self._backend = backend or get_default_backend()

        # Build lookup: (probe_id, layer) -> activation
        self._lookup: dict[tuple[str, int], "Array"] = {}
        for layer_idx, act_list in activations.items():
            for i, act in enumerate(act_list):
                if i < len(probe_ids):
                    self._lookup[(probe_ids[i], layer_idx)] = act

    def get_activations(
        self,
        texts: list[str],
        layer: int,
    ) -> list["Array"]:
        """Get activations for texts at a specific layer.

        For pre-collected activations, we look up by position in probe_ids.
        This is a simplified implementation that works with our use case.
        """
        # In practice, the density analyzer calls this with support_texts
        # but we already have the activations indexed by probe_id
        # Just return all activations for this layer
        if layer in self._activations:
            return list(self._activations[layer])
        return []


def stage_density(
    source_activations: dict[int, Any],
    target_activations: dict[int, Any],
    probe_ids: list[str],
    probe_domains: list[str],
    layers: list[int],
    feature_transforms: dict[int, Any] | None = None,
    backend: "Backend | None" = None,
    layer_mapping: dict[int, int] | None = None,
) -> DensityStageResult:
    """Stage 2: Compute knowledge density profiles and graft mask.

    Args:
        source_activations: Activations from source model per layer.
        target_activations: Activations from target model per layer.
        probe_ids: List of probe IDs corresponding to activations.
        probe_domains: List of domains for each probe.
        layers: Layer indices to analyze.
        feature_transforms: CKA alignment transforms per target layer (source→target space).
            Values may be either:
            - a single transform matrix `F` (applied as `source @ F`), or
            - a dict `{source_layer: F}` for composite alignments (use the mapped source layer).

            When provided, source activations are projected into target space before density
            comparison to ensure cross-dimensional comparisons are apples-to-apples.
        backend: Backend for tensor operations.
        layer_mapping: Optional mapping `target_layer -> source_layer` from probe stage.
            Used to select the correct source activations and per-layer transform.

    Returns:
        DensityStageResult with profiles, diff, and graft mask.
    """
    b = backend or get_default_backend()

    # Metrics to track
    metrics: dict[str, float | int] = {
        "layers_analyzed": 0,
        "concepts_analyzed": 0,
        "positive_opportunity_count": 0,
        "nonpositive_opportunity_count": 0,
    }

    # Validate inputs
    if not source_activations or not target_activations:
        raise RuntimeError("DENSITY: Missing activations for density analysis")

    if not probe_ids or len(probe_ids) != len(probe_domains):
        raise RuntimeError("DENSITY: Probe metadata mismatch")

    logger.info(
        "DENSITY: Analyzing %d layers, %d probes for graft opportunities",
        len(layers),
        len(probe_ids),
    )

    # Build simple concept density profiles from activations
    # This is a streamlined version that works with pre-collected activations
    import traceback
    try:
        source_profile = _build_density_profile_from_activations(
            activations=source_activations,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            layers=layers,
            backend=b,
        )
    except Exception as e:
        logger.error("DENSITY BUG: source_profile failed: %s", e)
        logger.error("DENSITY TRACEBACK:\n%s", traceback.format_exc())
        raise

    try:
        target_profile = _build_density_profile_from_activations(
            activations=target_activations,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            layers=layers,
            backend=b,
        )
    except Exception as e:
        logger.error("DENSITY BUG: target_profile failed: %s", e)
        logger.error("DENSITY TRACEBACK:\n%s", traceback.format_exc())
        raise

    # Compute knowledge diff
    differ = KnowledgeDiffer()
    knowledge_diff = differ.diff(source_profile, target_profile)

    # Compute graft mask
    graft_mask = compute_graft_mask(knowledge_diff)

    # =========================================================================
    # POINT CLOUD DENSITY: k-NN based density comparison per layer
    # =========================================================================
    # For each layer, compute local density at each activation point.
    # The density difference tells us where source has denser representation.
    point_cloud_densities: dict[int, PointCloudDensityResult] = {}
    density_weights: dict[int, "Array"] = {}

    total_positive_points = 0
    total_negative_points = 0

    for layer_idx in layers:
        source_layer_idx = layer_mapping.get(layer_idx, layer_idx) if layer_mapping else layer_idx
        src_acts = source_activations.get(source_layer_idx)
        tgt_acts = target_activations.get(layer_idx)

        # Handle both list and stacked-array formats
        def _is_empty(acts: Any) -> bool:
            if acts is None:
                return True
            if isinstance(acts, list):
                return len(acts) == 0
            # Array - check shape
            return len(acts.shape) == 0 or int(acts.shape[0]) == 0

        if _is_empty(src_acts) or _is_empty(tgt_acts):
            continue

        try:
            # Handle both list of 1D vectors and pre-stacked 2D arrays
            def _to_matrix(acts: Any) -> Any:
                if isinstance(acts, list):
                    return b.stack([b.array(a) for a in acts], axis=0)
                # Already a 2D array
                arr = b.array(acts)
                if len(arr.shape) == 1:
                    # Single activation - reshape to [1, dim]
                    return b.reshape(arr, (1, -1))
                return arr

            src_matrix = _to_matrix(src_acts)
            tgt_matrix = _to_matrix(tgt_acts)
            b.eval(src_matrix, tgt_matrix)

            # =====================================================================
            # ALIGNED DENSITY COMPARISON: Project source into target space first
            # =====================================================================
            # When feature_transforms is available, project source activations into
            # target's coordinate system before comparing density. This ensures:
            # 1. Both point clouds are in the same dimensional space
            # 2. Density comparison is meaningful (apples to apples)
            # 3. We find where target is TRULY sparse, not just "smaller"
            src_for_density = src_matrix
            if feature_transforms and layer_idx in feature_transforms:
                F_data = feature_transforms[layer_idx]

                F: Any | None
                if isinstance(F_data, dict):
                    # Composite alignment: pick the mapped source layer's transform.
                    if source_layer_idx in F_data:
                        F = F_data[source_layer_idx]
                    elif len(F_data) == 1:
                        F = next(iter(F_data.values()))
                    else:
                        F = None
                else:
                    F = F_data

                if F is not None:
                    F_arr = b.astype(F, "float32")
                    src_dim = int(src_matrix.shape[1])
                    tgt_dim = int(tgt_matrix.shape[1])
                    f_rows = int(F_arr.shape[0])
                    f_cols = int(F_arr.shape[1])

                    if src_dim == f_rows and tgt_dim == f_cols:
                        src_for_density = b.matmul(src_matrix, F_arr)
                    elif src_dim == f_cols and tgt_dim == f_rows:
                        # Defensive: allow pre-transposed transforms.
                        src_for_density = b.matmul(src_matrix, b.transpose(F_arr))
                    else:
                        logger.warning(
                            "DENSITY: Layer %d - Transform shape %s incompatible with src_dim=%d tgt_dim=%d; skipping projection",
                            layer_idx,
                            tuple(F_arr.shape),
                            src_dim,
                            tgt_dim,
                        )
                        src_for_density = src_matrix

                    b.eval(src_for_density)
                    logger.debug(
                        "DENSITY: Layer %d (src_layer=%d) - Projected source %s → target space %s",
                        layer_idx,
                        source_layer_idx,
                        src_matrix.shape,
                        src_for_density.shape,
                    )

            # Compute point cloud density comparison (both in target space now)
            pc_result = compute_knn_point_cloud_density(
                source_activations=src_for_density,
                target_activations=tgt_matrix,
                backend=b,
            )
            point_cloud_densities[layer_idx] = pc_result

            # Compute transfer weights from density difference
            weights = compute_density_weights(pc_result.density_diff, backend=b)
            density_weights[layer_idx] = weights

            total_positive_points += pc_result.positive_diff_count
            total_negative_points += pc_result.negative_diff_count

            logger.debug(
                "DENSITY: Layer %d - %d points source denser, %d points target denser",
                layer_idx,
                pc_result.positive_diff_count,
                pc_result.negative_diff_count,
            )
        except Exception as e:
            logger.warning("DENSITY: Point cloud analysis failed for layer %d: %s", layer_idx, e)

    logger.info(
        "DENSITY: Point cloud analysis - %d points where source denser, %d where target denser",
        total_positive_points,
        total_negative_points,
    )

    # Update metrics
    metrics["layers_analyzed"] = len(layers)
    metrics["concepts_analyzed"] = knowledge_diff.total_concepts
    metrics["positive_opportunity_count"] = knowledge_diff.positive_opportunity_count
    metrics["nonpositive_opportunity_count"] = knowledge_diff.nonpositive_opportunity_count
    metrics["overall_source_density"] = knowledge_diff.overall_source_density
    metrics["overall_target_density"] = knowledge_diff.overall_target_density
    metrics["overall_opportunity"] = knowledge_diff.overall_opportunity
    metrics["point_cloud_positive_points"] = total_positive_points
    metrics["point_cloud_negative_points"] = total_negative_points

    logger.info(
        "DENSITY: %d concepts analyzed, %d high opportunity (will graft), %d no graft (target dense)",
        knowledge_diff.total_concepts,
        knowledge_diff.positive_opportunity_count,
        knowledge_diff.nonpositive_opportunity_count,
    )

    return DensityStageResult(
        source_profile=source_profile,
        target_profile=target_profile,
        knowledge_diff=knowledge_diff,
        graft_mask=graft_mask,
        point_cloud_densities=point_cloud_densities,
        density_weights=density_weights,
        metrics=metrics,
    )


def _build_density_profile_from_activations(
    activations: dict[int, list["Array"]],
    probe_ids: list[str],
    probe_domains: list[str],
    layers: list[int],
    backend: "Backend",
) -> ModelDensityProfile:
    """Build a density profile from pre-collected activations.

    Uses per-probe local intrinsic dimension as the density signal:
    - Lower intrinsic dimension = denser representation (well-learned)
    - Higher intrinsic dimension = sparser representation (gap in knowledge)
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    b = backend
    id_estimator = IntrinsicDimension(backend=b)

    layer_profiles: dict[int, LayerDensityProfile] = {}
    all_concepts: list[ConceptDensity] = []

    for layer_idx in layers:
        act_list = activations.get(layer_idx)

        # Handle both list and stacked-array formats
        if act_list is None:
            raise RuntimeError(f"DENSITY: Missing activations for layer {layer_idx}")

        # Convert to matrix, handling both list and array formats
        if isinstance(act_list, list):
            if len(act_list) == 0:
                raise RuntimeError(f"DENSITY: Missing activations for layer {layer_idx}")
            if len(act_list) < 4:
                raise RuntimeError(
                    f"DENSITY: Need at least 4 probes for local dimension at layer {layer_idx}"
                )
            act_vectors = []
            for act in act_list:
                act_arr = b.array(act)
                b.eval(act_arr)
                if len(act_arr.shape) != 1:
                    raise RuntimeError(
                        f"DENSITY: Expected 1D activation vector, got shape {act_arr.shape}"
                    )
                act_vectors.append(act_arr)
            act_matrix = b.stack(act_vectors, axis=0)
        else:
            # Already a stacked 2D array [n_probes, dim]
            act_matrix = b.array(act_list)
            if len(act_matrix.shape) == 1:
                # Single probe - reshape
                act_matrix = b.reshape(act_matrix, (1, -1))
            n_probes = int(act_matrix.shape[0])
            if n_probes == 0:
                raise RuntimeError(f"DENSITY: Missing activations for layer {layer_idx}")
            if n_probes < 4:
                raise RuntimeError(
                    f"DENSITY: Need at least 4 probes for local dimension at layer {layer_idx}"
                )

        b.eval(act_matrix)

        # All k_neighbors and threshold parameters are derived from data
        local_map = id_estimator.local_dimension_map(act_matrix)
        # Keep dimensions on backend until we need individual values
        dims = local_map.dimensions
        b.eval(dims)

        # Precompute mean cosine similarity per probe (cluster tightness) - VECTORIZED
        eps = float(machine_epsilon(b, act_matrix))
        sim_matrix = geodesic_cosine_matrix(act_matrix, b, use_cache=False)
        b.eval(sim_matrix)

        # Zero out diagonal (self-similarity) and compute mean per row on backend
        n_probes = int(sim_matrix.shape[0])
        eye = b.eye(n_probes)
        sim_no_diag = sim_matrix - eye  # Set diagonal to 0
        # Sum each row, divide by (n-1) to get mean excluding self
        row_sums = b.sum(sim_no_diag, axis=1)
        cluster_tightness_vec = row_sums / (n_probes - 1) if n_probes > 1 else b.zeros((n_probes,))
        b.eval(cluster_tightness_vec)

        # Compute variance per activation vector on backend (vectorized)
        act_mean = b.mean(act_matrix, axis=1, keepdims=True)
        act_centered = act_matrix - act_mean
        variances_vec = b.mean(act_centered * act_centered, axis=1)
        b.eval(variances_vec)

        # Use native tolist() for O(1) extraction instead of O(n) scalar extractions
        concept_densities: list[ConceptDensity] = []

        # Debug: log shapes before tolist conversion
        logger.debug(
            "DENSITY DEBUG: layer=%d dims.shape=%s variances.shape=%s tightness.shape=%s",
            layer_idx,
            getattr(dims, 'shape', 'no-shape'),
            getattr(variances_vec, 'shape', 'no-shape'),
            getattr(cluster_tightness_vec, 'shape', 'no-shape'),
        )

        dims_list = b.tolist(dims)
        variances_list = b.tolist(variances_vec)
        cluster_tightness_list = b.tolist(cluster_tightness_vec)

        # Debug: verify tolist produced flat lists
        if dims_list and not isinstance(dims_list[0], (int, float)):
            logger.error("DENSITY BUG: dims_list[0] is not scalar: %s", type(dims_list[0]))
        if variances_list and not isinstance(variances_list[0], (int, float)):
            logger.error("DENSITY BUG: variances_list[0] is not scalar: %s", type(variances_list[0]))

        for i in range(min(len(act_list), len(probe_ids))):
            probe_id = probe_ids[i]
            domain = probe_domains[i] if i < len(probe_domains) else "unknown"

            intrinsic_dim = float(dims_list[i])
            # Handle NaN/undefined intrinsic dimensions gracefully
            # This can happen with extreme dimension mismatches or degenerate activations
            if intrinsic_dim != intrinsic_dim or intrinsic_dim <= 0:
                logger.debug(
                    "DENSITY: Skipping probe %s at layer %d (undefined intrinsic dim)",
                    probe_id, layer_idx,
                )
                # Skip this probe - don't include in graft mask
                continue
            density_score = 1.0 / max(intrinsic_dim, eps)

            variance = float(variances_list[i])
            cluster_tightness = float(cluster_tightness_list[i])

            concept_densities.append(
                ConceptDensity(
                    probe_id=probe_id,
                    name=probe_id,  # Use probe_id as name
                    domain=domain,
                    layer=layer_idx,
                    intrinsic_dimension=intrinsic_dim,
                    density_score=density_score,
                    activation_variance=variance,
                    cluster_tightness=cluster_tightness,
                )
            )

        if concept_densities:
            scores = sorted(c.density_score for c in concept_densities)
            n = len(scores)
            median = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2
            mean_density = sum(c.density_score for c in concept_densities) / len(concept_densities)
        else:
            median = 0.0
            mean_density = 0.0

        layer_profiles[layer_idx] = LayerDensityProfile(
            layer=layer_idx,
            concept_densities=concept_densities,
            mean_density=mean_density,
            median_density=median,
        )

        all_concepts.extend(concept_densities)

    # Compute per-domain aggregates
    domain_densities: dict[str, list[float]] = {}
    for c in all_concepts:
        if c.domain not in domain_densities:
            domain_densities[c.domain] = []
        domain_densities[c.domain].append(c.density_score)

    domain_means = {
        domain: sum(scores) / len(scores)
        for domain, scores in domain_densities.items()
        if scores
    }

    # Overall density
    overall = sum(c.density_score for c in all_concepts) / len(all_concepts) if all_concepts else 0.0

    return ModelDensityProfile(
        model_path="",  # Set by caller
        layers=layers,
        layer_profiles=layer_profiles,
        domain_densities=domain_means,
        overall_density=overall,
    )


def filter_core_probes_by_graft_mask(
    core_probe_ids: set[str],
    layer_idx: int,
    graft_mask: dict[str, dict[int, bool]] | None,
) -> set[str]:
    """Filter core probe IDs to only include those that should be grafted.

    Args:
        core_probe_ids: Original set of core probe IDs (from domain selection).
        layer_idx: Current layer being processed.
        graft_mask: Graft mask from density analysis. If None, graft all probes
            (CKA=1.0 invariant means null-space projection handles selectivity).

    Returns:
        Filtered set of probe IDs that should be grafted at this layer.
    """
    # CKA=1.0 invariant: when graft_mask is None, graft everything
    # Null-space projection ensures non-interference by construction
    if graft_mask is None:
        return core_probe_ids

    filtered = set()
    for probe_id in core_probe_ids:
        # Check if this (probe_id, layer) should be grafted
        should_graft = graft_mask.get(probe_id, {}).get(layer_idx, False)
        if should_graft:
            filtered.add(probe_id)

    if len(filtered) < len(core_probe_ids):
        skipped = len(core_probe_ids) - len(filtered)
        logger.debug(
            "Layer %d: Filtered %d/%d core probes (target already dense)",
            layer_idx,
            skipped,
            len(core_probe_ids),
        )

    return filtered


__all__ = [
    "DensityStageResult",
    "stage_density",
    "filter_core_probes_by_graft_mask",
]
