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

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    CrossArchitectureLayerMatcher,
)
from modelcypher.core.domain.geometry.cross_cultural_geometry import (
    AlignmentAnalysis,
    CrossCulturalGeometry,
)
from modelcypher.core.domain.geometry.manifold_stitcher import IntersectionMap
from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner
from modelcypher.core.domain.geometry.refinement_density import (
    RefinementDensityConfig,
    RefinementDensityResult,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    AlignmentMethod,
    SharedSubspaceProjector,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    Config as SharedSubspaceConfig,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    Result as SharedSubspaceResult,
)
from modelcypher.core.domain.geometry.transfer_fidelity import (
    Prediction,
    TransferFidelityPrediction,
)
from modelcypher.core.domain.geometry.transport_guided_merger import TransportGuidedMerger
from modelcypher.core.use_cases.anchor_extractor import (
    AnchorExtractionConfig,
    AnchorExtractor,
)
from modelcypher.core.use_cases.geometry_engine import GeometryEngine
from modelcypher.core.use_cases.quantization_utils import (
    QuantizationConfig,
    QuantizationHint,
    dequantize_if_needed,
    quantization_hint_for_key,
    resolve_quantization,
)
from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class AnchorMode(str, Enum):
    semantic_primes = "semantic-primes"
    geometric = "geometric"
    intersection = "intersection"
    rebasin = "rebasin"


class ModuleScope(str, Enum):
    attention_only = "attention-only"
    all = "all"


@dataclass(frozen=True)
class ModuleKind:
    name: str
    is_residual_output: bool


@dataclass(frozen=True)
class SVDBases:
    u: Array
    v: Array
    spectral_norm: float
    singular_values: list[float]


@dataclass(frozen=True)
class SharedAnchors:
    source: Array
    target: Array
    anchor_ids: list[str]
    confidence_weights: list[float]

    @property
    def count(self) -> int:
        return len(self.anchor_ids)


@dataclass(frozen=True)
class LayerMergeMetric:
    layer_index: int
    module_name: str
    module_kind: str
    procrustes_error: float
    condition_number: float
    rotation_deviation: float
    spectral_ratio: float


@dataclass(frozen=True)
class TransitionContext:
    mean_transition_cka: float
    mean_state_cka: float
    transition_advantage: float
    transition_better_than_state: bool
    transition_count: int
    anchor_count: int
    delta_alignment_by_layer: dict[int, float]


@dataclass(frozen=True)
class ConsistencyContext:
    anchor_count: int
    sample_layer_count: int
    mean_source_distance: float
    mean_target_distance: float
    target_weight_by_layer: dict[int, float]


@dataclass(frozen=True)
class TransitionMetrics:
    mean_transition_cka: float
    mean_state_cka: float
    transition_advantage: float
    transition_better_than_state: bool
    transition_count: int
    anchor_count: int


@dataclass(frozen=True)
class ConsistencyMetrics:
    anchor_count: int
    sample_layer_count: int
    mean_source_distance: float
    mean_target_distance: float


@dataclass(frozen=True)
class SharedSubspaceGateComponents:
    """Raw components for shared subspace gate decision.

    Returns raw measurements instead of a weighted composite.
    Consumers decide how to interpret these values.
    """

    correlation: float
    """Top alignment strength [0, 1]. Higher = stronger correlation between subspaces."""

    variance_ratio: float
    """Shared variance ratio [0, 1]. Higher = more variance explained by shared subspace."""

    alignment_error: float
    """Raw alignment error [0, 1]. Lower = better alignment."""


@dataclass(frozen=True)
class SharedSubspaceContext:
    source_layer: int
    target_layer: int
    result: SharedSubspaceResult
    source_projection: Array
    target_projection: Array
    gate_components: SharedSubspaceGateComponents


@dataclass(frozen=True)
class SharedSubspaceIndex:
    contexts: dict[int, SharedSubspaceContext]
    target_to_source: dict[int, int]

    def context_for_layer(self, target_layer: int) -> SharedSubspaceContext | None:
        return self.contexts.get(target_layer)


@dataclass(frozen=True)
class SharedSubspaceMetrics:
    shared_dimension: int
    alignment_error: float
    shared_variance_ratio: float
    top_correlation: float
    sample_count: int
    method: str
    is_valid: bool
    layer_count: int


@dataclass(frozen=True)
class TransportMetrics:
    mean_gw_distance: float
    mean_marginal_error: float
    mean_effective_rank: float
    layer_count: int
    converged_layers: int
    skipped_layers: int


@dataclass(frozen=True)
class TransportMergeResult:
    merged_weight: Array
    gw_distance: float
    marginal_error: float
    effective_rank: int
    converged: bool
    iterations: int


@dataclass(frozen=True)
class MergeAnalysisResult:
    source_model: str
    target_model: str
    anchor_mode: str
    timestamp: datetime
    mean_procrustes_error: float
    max_procrustes_error: float
    rotation_field_roughness: float
    anchor_coverage: int
    layer_metrics: list[LayerMergeMetric]
    anchor_alignment: AlignmentAnalysis | None = None
    transfer_fidelity: Prediction | None = None
    mlp_rebasin_quality: float | None = None
    mlp_blocks_aligned: int | None = None
    transition_metrics: TransitionMetrics | None = None
    consistency_metrics: ConsistencyMetrics | None = None
    shared_subspace_metrics: SharedSubspaceMetrics | None = None
    transport_metrics: TransportMetrics | None = None
    refinement_metrics: RefinementMetrics | None = None


@dataclass(frozen=True)
class RefinementMetrics:
    """Summary metrics from refinement density analysis."""

    mean_composite_score: float
    max_composite_score: float
    layers_above_hard_swap: int
    layers_above_high_alpha: int
    hard_swap_layers: list[int]
    has_sparsity_data: bool
    has_directional_data: bool
    has_transition_data: bool


@dataclass(frozen=True)
class RotationalMergeOptions:
    alignment_rank: int = 32
    alpha: float = 0.5
    anchor_mode: AnchorMode = AnchorMode.semantic_primes
    module_scope: ModuleScope = ModuleScope.attention_only
    use_enriched_primes: bool = True
    intersection_map: IntersectionMap | None = None
    use_adaptive_alpha: bool = False
    transition_gate_strength: float = 0.0
    # NOTE: transition_gate_min/max_ratio removed - geometry determines bounds
    consistency_gate_strength: float = 0.0
    consistency_gate_layer_samples: int = 6
    use_shared_subspace_projection: bool = False
    shared_subspace_config: SharedSubspaceConfig | None = None
    shared_subspace_blend_weight: float = 0.0
    shared_subspace_per_layer: bool = True
    use_transport_guided: bool = False
    transport_coupling_threshold: float = 0.001
    transport_blend_alpha: float = 0.5
    transport_min_samples: int = 5
    transport_max_samples: int = 32
    transport_use_intersection_confidence: bool = True
    # Refinement density gating
    use_refinement_density: bool = False
    refinement_density_config: RefinementDensityConfig | None = None
    refinement_density_result: RefinementDensityResult | None = None
    refinement_gate_strength: float = 1.0  # How much to weight refinement recommendations
    refinement_hard_swap_enabled: bool = (
        True  # Allow full source replacement for highly refined layers
    )


class RotationalMerger:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.geometry = GeometryEngine(backend)

    def merge(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        options: RotationalMergeOptions,
        anchors: SharedAnchors,
        source_id: str | None = None,
        target_id: str | None = None,
        source_quantization: QuantizationConfig | None = None,
        target_quantization: QuantizationConfig | None = None,
        source_crm: ConceptResponseMatrix | None = None,
        target_crm: ConceptResponseMatrix | None = None,
    ) -> tuple[dict[str, Array], MergeAnalysisResult]:
        if options.anchor_mode == AnchorMode.intersection and options.intersection_map is None:
            raise ValueError("Intersection anchor mode requires an intersection map")
        if options.anchor_mode == AnchorMode.rebasin and options.module_scope != ModuleScope.all:
            raise ValueError("Rebasin anchor mode requires module scope 'all'")

        transition_context, transition_metrics = self._prepare_transition_context(
            source_crm,
            target_crm,
            options,
        )
        consistency_context, consistency_metrics = self._prepare_consistency_context(
            source_crm,
            target_crm,
            options,
        )
        shared_subspace_index, shared_subspace_metrics = self._prepare_shared_subspace_context(
            source_crm,
            target_crm,
            options,
        )
        alignment_rank = self._resolve_alignment_rank(options, shared_subspace_index)

        preprocessed_source = self._maybe_rebasin_source(
            source_weights,
            target_weights,
            anchors,
            options,
            source_quantization=source_quantization,
            target_quantization=target_quantization,
        )
        current_omega_in = self._prepare_initial_omega(
            source_weights,
            target_weights,
            anchors,
            options,
            source_quantization=source_quantization,
            target_quantization=target_quantization,
        )

        merged_weights: dict[str, Array] = {}
        # Preserve all target weights by default; merged layers overwrite as needed.
        for key, value in target_weights.items():
            merged_weights[key] = self._to_array(value)

        layer_metrics: list[LayerMergeMetric] = []
        errors: list[float] = []
        roughness: list[float] = []
        previous_omega: Array | None = None
        transport_distances: list[float] = []
        transport_marginals: list[float] = []
        transport_ranks: list[int] = []
        transport_converged = 0
        transport_skipped = 0

        target_weight_keys = sorted(
            key
            for key in target_weights.keys()
            if (key.startswith("layers.") or key.startswith("model.layers."))
            and key.endswith(".weight")
        )

        for target_key in target_weight_keys:
            target_raw = target_weights.get(target_key)
            if target_raw is None:
                continue

            source_raw = source_weights.get(target_key)
            if source_raw is None:
                continue

            if not self._is_projection_scope(target_key, options.module_scope):
                continue

            target_arr = self._to_array(target_raw)
            source_arr = self._to_array(source_raw)
            target_shape = self.backend.shape(target_arr)
            source_shape = self.backend.shape(source_arr)
            if len(target_shape) != 2 or len(source_shape) != 2:
                continue
            target_dtype = self.backend.dtype(target_arr)
            target_is_quantized = "float" not in str(target_dtype).lower()

            source_hint = quantization_hint_for_key(target_key, source_quantization)
            target_hint = quantization_hint_for_key(target_key, target_quantization)
            source_weight_np = dequantize_if_needed(
                source_raw,
                target_key,
                source_weights,
                self.backend,
                hint=source_hint,
            )
            target_weight_np = dequantize_if_needed(
                target_raw,
                target_key,
                target_weights,
                self.backend,
                hint=target_hint,
            )
            source_weight_shape = self.backend.shape(source_weight_np)
            target_weight_shape = self.backend.shape(target_weight_np)
            if len(source_weight_shape) != 2 or len(target_weight_shape) != 2:
                continue
            if source_weight_shape != target_weight_shape:
                continue

            source_weight = self._select_source_weight(
                target_key,
                source_weight_np,
                preprocessed_source,
                options,
            )
            target_weight = self.backend.astype(
                self._to_array(target_weight_np), "float32"
            )
            self.backend.eval(source_weight, target_weight)

            source_bases = self._truncated_svd_bases(
                source_weight,
                rank=alignment_rank,
                oversampling=8,
                power_iterations=1,
                seed=0,
                label=target_key,
            )
            target_bases = self._truncated_svd_bases(
                target_weight,
                rank=alignment_rank,
                oversampling=8,
                power_iterations=1,
                seed=0,
                label=target_key,
            )

            layer_index = self._extract_layer_index(target_key) or -1
            shared_subspace = (
                shared_subspace_index.context_for_layer(layer_index)
                if shared_subspace_index is not None and layer_index >= 0
                else None
            )
            transport_result = None
            if options.use_transport_guided:
                transport_result = self._transport_guided_merge(
                    source_weight_np,
                    target_weight_np,
                    target_key,
                    layer_index,
                    options,
                    transition_context,
                    consistency_context,
                )
            if transport_result is not None:
                transport_distances.append(transport_result.gw_distance)
                transport_marginals.append(transport_result.marginal_error)
                transport_ranks.append(transport_result.effective_rank)
                if transport_result.converged:
                    transport_converged += 1
                merged_arr = transport_result.merged_weight
                self.backend.eval(merged_arr)
                omega_out = self.backend.eye(alignment_rank, dtype="float32")
                self.backend.eval(omega_out)
                module_kind = self._module_kind_from_key(target_key)
                rotation_deviation = self._rotation_deviation(omega_out)
                condition_number = 1.0
                if options.anchor_mode == AnchorMode.geometric:
                    condition_number = self._condition_number(source_bases.singular_values)
                spectral_ratio = self._spectral_ratio(
                    target_bases.singular_values, source_bases.singular_values
                )
                errors.append(transport_result.gw_distance)
                if previous_omega is not None:
                    prev_shape = self.backend.shape(previous_omega)
                    out_shape = self.backend.shape(omega_out)
                    if prev_shape == out_shape:
                        diff = omega_out - previous_omega
                        self.backend.eval(diff)
                        roughness.append(float(self.backend.norm(diff)))
                previous_omega = omega_out
                layer_metrics.append(
                    LayerMergeMetric(
                        layer_index=layer_index,
                        module_name=target_key,
                        module_kind=module_kind.name,
                        procrustes_error=transport_result.gw_distance,
                        condition_number=condition_number,
                        rotation_deviation=rotation_deviation,
                        spectral_ratio=spectral_ratio,
                    )
                )

                if not target_is_quantized:
                    merged_weights[target_key] = self.backend.astype(merged_arr, self.backend.dtype(target_arr))
                else:
                    quantized = self._quantize_blended(
                        merged_arr,
                        target_key,
                        raw_weight_shape=target_shape,
                        blended_shape=self.backend.shape(merged_arr),
                        target_weights=target_weights,
                        hint=target_hint,
                    )
                    if quantized is None:
                        merged_weights[target_key] = self.backend.astype(merged_arr, "float32")
                        base = target_key.replace(".weight", "")
                        merged_weights.pop(f"{base}.scales", None)
                        merged_weights.pop(f"{base}.biases", None)
                    else:
                        merged_weights[target_key] = quantized.weight
                        merged_weights[quantized.scales_key] = quantized.scales
                        if quantized.biases is None:
                            merged_weights.pop(quantized.biases_key, None)
                        else:
                            merged_weights[quantized.biases_key] = quantized.biases

                if module_kind.is_residual_output:
                    current_omega_in = self.backend.astype(omega_out, "float32")
                    self.backend.eval(current_omega_in)
                continue
            if options.use_transport_guided:
                transport_skipped += 1

            omega_out = self._compute_omega_out(
                target_key,
                source_weight,
                target_weight,
                source_bases,
                target_bases,
                current_omega_in,
                anchors,
                layer_index,
                options,
                shared_subspace,
            )

            procrustes_error = self._procrustes_error(
                source_weight,
                target_weight,
                source_bases,
                target_bases,
                omega_out,
                current_omega_in,
            )
            errors.append(procrustes_error)

            module_kind = self._module_kind_from_key(target_key)
            rotation_deviation = self._rotation_deviation(omega_out)
            condition_number = 1.0
            if options.anchor_mode == AnchorMode.geometric:
                condition_number = self._condition_number(source_bases.singular_values)

            spectral_ratio = self._spectral_ratio(
                target_bases.singular_values, source_bases.singular_values
            )

            if previous_omega is not None:
                prev_shape = self.backend.shape(previous_omega)
                out_shape = self.backend.shape(omega_out)
                if prev_shape == out_shape:
                    diff = omega_out - previous_omega
                    self.backend.eval(diff)
                    roughness.append(float(self.backend.norm(diff)))
            previous_omega = omega_out

            layer_metrics.append(
                LayerMergeMetric(
                    layer_index=layer_index,
                    module_name=target_key,
                    module_kind=module_kind.name,
                    procrustes_error=procrustes_error,
                    condition_number=condition_number,
                    rotation_deviation=rotation_deviation,
                    spectral_ratio=spectral_ratio,
                )
            )

            projected = self._project_weight(
                source_weight,
                source_bases,
                target_bases,
                current_omega_in,
                omega_out,
            )

            effective_alpha = options.alpha
            if options.use_adaptive_alpha and options.intersection_map is not None:
                layer_confidence = self.lookup_layer_confidence(
                    options.intersection_map, layer_index
                )
                effective_alpha = self.confidence_based_alpha(layer_confidence, effective_alpha)
            effective_alpha = self._transition_adjusted_alpha(
                effective_alpha,
                layer_index,
                transition_context,
                options,
            )
            effective_alpha = self._consistency_adjusted_alpha(
                effective_alpha,
                layer_index,
                consistency_context,
                options,
            )

            # Apply refinement density gating
            hard_swap, effective_alpha = self._refinement_adjusted_alpha(
                effective_alpha,
                layer_index,
                options,
            )

            # Handle hard swap: take source weight directly without blending
            if hard_swap:
                blended = projected
            else:
                alpha_value = self.backend.array(float(effective_alpha), dtype="float32")
                blended = (alpha_value * target_weight) + ((1.0 - alpha_value) * projected)
            self.backend.eval(blended)

            if not target_is_quantized:
                merged_weights[target_key] = self.backend.astype(blended, target_dtype)
            else:
                quantized = self._quantize_blended(
                    blended,
                    target_key,
                    raw_weight_shape=target_shape,
                    blended_shape=target_weight_shape,
                    target_weights=target_weights,
                    hint=target_hint,
                )
                if quantized is None:
                    merged_weights[target_key] = self.backend.astype(blended, "float32")
                    base = target_key.replace(".weight", "")
                    merged_weights.pop(f"{base}.scales", None)
                    merged_weights.pop(f"{base}.biases", None)
                else:
                    merged_weights[target_key] = quantized.weight
                    merged_weights[quantized.scales_key] = quantized.scales
                    if quantized.biases is None:
                        merged_weights.pop(quantized.biases_key, None)
                    else:
                        merged_weights[quantized.biases_key] = quantized.biases

            if module_kind.is_residual_output:
                current_omega_in = self.backend.astype(omega_out, "float32")
                self.backend.eval(current_omega_in)

        mean_error = sum(errors) / len(errors) if errors else 0.0
        max_error = max(errors) if errors else 0.0
        roughness_value = sum(roughness) / len(roughness) if roughness else 0.0

        anchor_gram_source, anchor_count = self._anchor_gram(anchors.source)
        anchor_gram_target, _ = self._anchor_gram(anchors.target)
        anchor_alignment = CrossCulturalGeometry.analyze_alignment(
            anchor_gram_source,
            anchor_gram_target,
            anchor_count,
        )
        transfer_fidelity = TransferFidelityPrediction.predict(
            anchor_gram_source,
            anchor_gram_target,
            anchor_count,
        )
        transport_metrics = None
        if options.use_transport_guided:
            count = len(transport_distances)
            mean_distance = sum(transport_distances) / count if transport_distances else 0.0
            mean_marginal = sum(transport_marginals) / count if transport_marginals else 0.0
            mean_rank = sum(transport_ranks) / count if transport_ranks else 0.0
            transport_metrics = TransportMetrics(
                mean_gw_distance=mean_distance,
                mean_marginal_error=mean_marginal,
                mean_effective_rank=mean_rank,
                layer_count=count,
                converged_layers=transport_converged,
                skipped_layers=transport_skipped,
            )

        # Build refinement metrics from result if available
        refinement_metrics = None
        if options.use_refinement_density and options.refinement_density_result is not None:
            rdr = options.refinement_density_result
            refinement_metrics = RefinementMetrics(
                mean_composite_score=rdr.mean_composite_score,
                max_composite_score=rdr.max_composite_score,
                layers_above_hard_swap=rdr.layers_above_hard_swap,
                layers_above_high_alpha=rdr.layers_above_high_alpha,
                hard_swap_layers=rdr.hard_swap_layers,
                has_sparsity_data=rdr.has_sparsity_data,
                has_directional_data=rdr.has_directional_data,
                has_transition_data=rdr.has_transition_data,
            )

        analysis = MergeAnalysisResult(
            source_model=source_id or "source",
            target_model=target_id or "target",
            anchor_mode=options.anchor_mode.value,
            timestamp=datetime.utcnow(),
            mean_procrustes_error=mean_error,
            max_procrustes_error=max_error,
            rotation_field_roughness=roughness_value,
            anchor_coverage=anchors.count,
            layer_metrics=layer_metrics,
            anchor_alignment=anchor_alignment,
            transfer_fidelity=transfer_fidelity,
            mlp_rebasin_quality=preprocessed_source.quality,
            mlp_blocks_aligned=preprocessed_source.blocks_aligned,
            transition_metrics=transition_metrics,
            consistency_metrics=consistency_metrics,
            shared_subspace_metrics=shared_subspace_metrics,
            transport_metrics=transport_metrics,
            refinement_metrics=refinement_metrics,
        )
        return merged_weights, analysis

    def build_shared_anchors(
        self,
        source_anchors: dict[str, Array],
        target_anchors: dict[str, Array],
        source_confidence: dict[str, float],
        target_confidence: dict[str, float],
        alignment_rank: int,
        minimum_count: int = 16,
    ) -> SharedAnchors:
        shared_ids = sorted(set(source_anchors.keys()) & set(target_anchors.keys()))
        required = max(alignment_rank, minimum_count)
        if len(shared_ids) < required:
            raise ValueError(
                f"Insufficient shared anchors for alignment: need >= {required}, got {len(shared_ids)}"
            )

        source_vectors: list[Array] = []
        target_vectors: list[Array] = []
        confidence_weights: list[float] = []

        for anchor_id in shared_ids:
            source_vec = source_anchors[anchor_id]
            target_vec = target_anchors[anchor_id]
            source_vectors.append(self.backend.astype(self._to_array(source_vec), "float32"))
            target_vectors.append(self.backend.astype(self._to_array(target_vec), "float32"))
            source_conf = float(source_confidence.get(anchor_id, 0.5))
            target_conf = float(target_confidence.get(anchor_id, 0.5))
            confidence_weights.append(math.sqrt(source_conf * target_conf))

        source_matrix = self.backend.stack(source_vectors, axis=0)
        target_matrix = self.backend.stack(target_vectors, axis=0)
        source_matrix = AnchorExtractor.normalize_anchor_matrix(source_matrix, self.backend)
        target_matrix = AnchorExtractor.normalize_anchor_matrix(target_matrix, self.backend)

        self.backend.eval(source_matrix, target_matrix)

        return SharedAnchors(
            source=source_matrix,
            target=target_matrix,
            anchor_ids=shared_ids,
            confidence_weights=confidence_weights,
        )

    def build_shared_anchors_from_atlas(
        self,
        source_path: str,
        target_path: str,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        source_quantization: QuantizationConfig | None = None,
        target_quantization: QuantizationConfig | None = None,
    ) -> SharedAnchors:
        """Build shared anchors using all 321 probes from UnifiedAtlasInventory.

        This method uses the full atlas system (7 sources, 321 probes) for
        cross-domain triangulation, enabling more robust dimension-agnostic
        alignment between models.

        Args:
            source_path: Path to source model (for tokenizer)
            target_path: Path to target model (for tokenizer)
            source_weights: Source model weights
            target_weights: Target model weights
            source_quantization: Optional source quantization config
            target_quantization: Optional target quantization config

        Returns:
            SharedAnchors built from unified atlas probes
        """
        extractor = AnchorExtractor()

        # Extract with unified atlas (all 321 probes)
        source_anchors, source_conf = extractor.extract(
            source_path,
            source_weights,
            config=AnchorExtractionConfig(use_unified_atlas=True),
            quantization=source_quantization,
            backend=self.backend,
        )

        target_anchors, target_conf = extractor.extract(
            target_path,
            target_weights,
            config=AnchorExtractionConfig(use_unified_atlas=True),
            quantization=target_quantization,
            backend=self.backend,
        )

        logger.info(
            "Built anchors from unified atlas: source=%d, target=%d",
            len(source_anchors),
            len(target_anchors),
        )

        return self.build_shared_anchors(
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            source_confidence=source_conf,
            target_confidence=target_conf,
            alignment_rank=self.options.alignment_rank,
        )

    @staticmethod
    def confidence_based_alpha(layer_confidence: float | None, fallback_alpha: float) -> float:
        """Derive alpha directly from layer confidence.

        The layer confidence IS the geometric signal. No arbitrary transformations.
        - High confidence (1.0) = layers match well → blend more source (lower alpha)
        - Low confidence (0.0) = layers don't match → keep more target (higher alpha)

        Alpha = 1 - confidence (direct geometric relationship)

        Args:
            layer_confidence: Geometric confidence score for layer alignment.
            fallback_alpha: Alpha to use if no confidence available.

        Returns:
            Alpha value derived from geometry.
        """
        if layer_confidence is None:
            return fallback_alpha

        # Direct geometric derivation: confidence tells us how much to trust the alignment
        # No arbitrary multipliers or clamping - the geometry is the answer
        return 1.0 - layer_confidence

    @staticmethod
    def lookup_layer_confidence(intersection: IntersectionMap, layer_index: int) -> float | None:
        for entry in intersection.layer_confidences:
            if entry.layer == layer_index:
                return entry.confidence
        return None

    # Align gating math with the reference implementation to keep merge behavior comparable.
    @staticmethod
    def _transition_adjusted_alpha(
        base_alpha: float,
        layer: int,
        transition_context: TransitionContext | None,
        options: RotationalMergeOptions,
    ) -> float:
        strength = max(0.0, min(1.0, options.transition_gate_strength))
        if strength <= 0.0 or transition_context is None:
            return base_alpha
        ratio = transition_context.delta_alignment_by_layer.get(
            layer, transition_context.transition_advantage
        )
        if not math.isfinite(ratio):
            return base_alpha
        # Let the geometry flow through - no arbitrary clamping of the ratio
        # If ratio produces invalid alpha, that's geometric information
        target_alpha = base_alpha * (2.0 - ratio)
        blended = base_alpha * (1.0 - strength) + target_alpha * strength
        # Geometry-derived: blend of valid alphas should produce valid alpha
        return blended

    @staticmethod
    def _consistency_adjusted_alpha(
        base_alpha: float,
        layer: int,
        consistency_context: ConsistencyContext | None,
        options: RotationalMergeOptions,
    ) -> float:
        strength = max(0.0, min(1.0, options.consistency_gate_strength))
        if strength <= 0.0 or consistency_context is None:
            return base_alpha
        target_weight = consistency_context.target_weight_by_layer.get(layer)
        if target_weight is None or not math.isfinite(target_weight):
            return base_alpha
        blended = base_alpha * (1.0 - strength) + float(target_weight) * strength
        # Geometry-derived: blend of valid alphas should produce valid alpha
        return blended

    @staticmethod
    def _refinement_adjusted_alpha(
        base_alpha: float,
        layer: int,
        options: RotationalMergeOptions,
    ) -> tuple[bool, float]:
        """
        Apply refinement density gating to determine alpha or hard swap.

        Returns:
            Tuple of (should_hard_swap, adjusted_alpha)
        """
        if not options.use_refinement_density or options.refinement_density_result is None:
            return False, base_alpha

        result = options.refinement_density_result
        score = result.layer_scores.get(layer)
        if score is None:
            return False, base_alpha

        strength = max(0.0, min(1.0, options.refinement_gate_strength))
        if strength <= 0.0:
            return False, base_alpha

        # Check for hard swap: composite_score >= hard_swap_threshold
        hard_swap_threshold = result.config.hard_swap_threshold
        if (
            options.refinement_hard_swap_enabled
            and score.composite_score >= hard_swap_threshold
            and strength >= 0.5
        ):
            logger.info(
                "Hard swap layer %d: refinement score %.3f >= threshold %.3f",
                layer,
                score.composite_score,
                hard_swap_threshold,
            )
            return True, 0.0

        # Blend base alpha with recommended alpha based on strength
        recommended = score.recommended_alpha
        blended = base_alpha * (1.0 - strength) + recommended * strength
        # Geometry-derived: blend of valid alphas should produce valid alpha
        return False, blended

    def _prepare_transition_context(
        self,
        source_crm: ConceptResponseMatrix | None,
        target_crm: ConceptResponseMatrix | None,
        options: RotationalMergeOptions,
    ) -> tuple[TransitionContext | None, TransitionMetrics | None]:
        if source_crm is None or target_crm is None:
            if options.transition_gate_strength > 0:
                logger.warning("Transition gate enabled but CRM inputs are missing.")
            return None, None

        experiment = source_crm.compute_transition_alignment(target_crm)
        if experiment is None:
            if options.transition_gate_strength > 0:
                logger.warning("Transition gate enabled but CRM overlap is insufficient.")
            return None, None

        delta_by_layer = {item.from_layer: item.delta_alignment for item in experiment.transitions}
        context = TransitionContext(
            mean_transition_cka=experiment.mean_transition_cka,
            mean_state_cka=experiment.mean_state_cka,
            transition_advantage=experiment.transition_advantage,
            transition_better_than_state=experiment.transition_better_than_state,
            transition_count=experiment.layer_transition_count,
            anchor_count=experiment.anchor_count,
            delta_alignment_by_layer=delta_by_layer,
        )
        metrics = TransitionMetrics(
            mean_transition_cka=experiment.mean_transition_cka,
            mean_state_cka=experiment.mean_state_cka,
            transition_advantage=experiment.transition_advantage,
            transition_better_than_state=experiment.transition_better_than_state,
            transition_count=experiment.layer_transition_count,
            anchor_count=experiment.anchor_count,
        )
        return context, metrics

    def _prepare_consistency_context(
        self,
        source_crm: ConceptResponseMatrix | None,
        target_crm: ConceptResponseMatrix | None,
        options: RotationalMergeOptions,
    ) -> tuple[ConsistencyContext | None, ConsistencyMetrics | None]:
        if source_crm is None or target_crm is None:
            if options.consistency_gate_strength > 0:
                logger.warning("Consistency gate enabled but CRM inputs are missing.")
            return None, None

        profile = source_crm.compute_consistency_profile(
            target_crm,
            layer_sample_count=options.consistency_gate_layer_samples,
        )
        if profile is None:
            if options.consistency_gate_strength > 0:
                logger.warning("Consistency gate enabled but CRM overlap is insufficient.")
            return None, None

        context = ConsistencyContext(
            anchor_count=profile.anchor_count,
            sample_layer_count=profile.sample_layer_count,
            mean_source_distance=profile.mean_source_distance,
            mean_target_distance=profile.mean_target_distance,
            target_weight_by_layer=profile.target_weight_by_layer,
        )
        metrics = ConsistencyMetrics(
            anchor_count=profile.anchor_count,
            sample_layer_count=profile.sample_layer_count,
            mean_source_distance=profile.mean_source_distance,
            mean_target_distance=profile.mean_target_distance,
        )
        return context, metrics

    def _prepare_shared_subspace_context(
        self,
        source_crm: ConceptResponseMatrix | None,
        target_crm: ConceptResponseMatrix | None,
        options: RotationalMergeOptions,
    ) -> tuple[SharedSubspaceIndex | None, SharedSubspaceMetrics | None]:
        if not options.use_shared_subspace_projection:
            return None, None
        if source_crm is None or target_crm is None:
            logger.warning("Shared subspace projection enabled but CRM inputs are missing.")
            return None, None

        config = options.shared_subspace_config or SharedSubspaceConfig()
        if not options.shared_subspace_per_layer:
            max_layer = min(source_crm.layer_count, target_crm.layer_count) - 1
            if max_layer < 0:
                return None, None
            result = SharedSubspaceProjector.discover(
                source_crm,
                target_crm,
                max_layer,
                config=config,
            )
            if result is None:
                logger.warning("Shared subspace discovery failed.")
                return None, None

            source_projection = self.backend.astype(
                self._to_array(result.source_projection), "float32"
            )
            target_projection = self.backend.astype(
                self._to_array(result.target_projection), "float32"
            )
            gate_components = self._shared_subspace_gate_components(result)

            top_correlation = (
                float(result.alignment_strengths[0]) if result.alignment_strengths else 0.0
            )
            metrics = SharedSubspaceMetrics(
                shared_dimension=result.shared_dimension,
                alignment_error=result.alignment_error,
                shared_variance_ratio=result.shared_variance_ratio,
                top_correlation=top_correlation,
                sample_count=result.sample_count,
                method=result.method.value,
                is_valid=result.is_valid,
                layer_count=1,
            )
            context = SharedSubspaceContext(
                source_layer=max_layer,
                target_layer=max_layer,
                result=result,
                source_projection=source_projection,
                target_projection=target_projection,
                gate_components=gate_components,
            )
            index = SharedSubspaceIndex(
                contexts={max_layer: context},
                target_to_source={max_layer: max_layer},
            )
            return index, metrics

        matcher = CrossArchitectureLayerMatcher.find_correspondence(source_crm, target_crm)
        contexts: dict[int, SharedSubspaceContext] = {}
        target_to_source: dict[int, int] = {}
        aggregated: list[SharedSubspaceResult] = []

        for mapping in matcher.mappings:
            if mapping.is_skipped:
                continue
            result = SharedSubspaceProjector.discover(
                source_crm,
                target_crm,
                mapping.source_layer,
                target_layer=mapping.target_layer,
                config=config,
            )
            if result is None:
                continue
            source_projection = self.backend.astype(
                self._to_array(result.source_projection), "float32"
            )
            target_projection = self.backend.astype(
                self._to_array(result.target_projection), "float32"
            )
            gate_components = self._shared_subspace_gate_components(result)
            context = SharedSubspaceContext(
                source_layer=mapping.source_layer,
                target_layer=mapping.target_layer,
                result=result,
                source_projection=source_projection,
                target_projection=target_projection,
                gate_components=gate_components,
            )
            contexts[mapping.target_layer] = context
            target_to_source[mapping.target_layer] = mapping.source_layer
            aggregated.append(result)

        if not contexts:
            logger.warning("Shared subspace discovery failed for all layer mappings.")
            return None, None

        dims = [res.shared_dimension for res in aggregated]
        shared_dim = int(sum(dims) / len(dims)) if dims else 0
        errors = [res.alignment_error for res in aggregated]
        alignment_error = sum(errors) / len(errors) if errors else 0.0
        ratios = [res.shared_variance_ratio for res in aggregated]
        shared_variance_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        correlations = [res.alignment_strengths[0] for res in aggregated if res.alignment_strengths]
        top_correlation = sum(correlations) / len(correlations) if correlations else 0.0
        counts = [res.sample_count for res in aggregated]
        sample_count = int(sum(counts) / len(counts)) if counts else 0
        method = aggregated[0].method.value if aggregated else AlignmentMethod.cca.value
        is_valid = all(res.is_valid for res in aggregated)

        metrics = SharedSubspaceMetrics(
            shared_dimension=shared_dim,
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            top_correlation=top_correlation,
            sample_count=sample_count,
            method=method,
            is_valid=is_valid,
            layer_count=len(aggregated),
        )
        return SharedSubspaceIndex(contexts=contexts, target_to_source=target_to_source), metrics

    @staticmethod
    def _resolve_alignment_rank(
        options: RotationalMergeOptions,
        shared_subspace: SharedSubspaceIndex | None,
    ) -> int:
        if shared_subspace is None:
            return options.alignment_rank
        shared_dims = [
            ctx.result.shared_dimension
            for ctx in shared_subspace.contexts.values()
            if ctx.result.is_valid
        ]
        if not shared_dims:
            return options.alignment_rank
        return min(options.alignment_rank, min(shared_dims))

    @staticmethod
    def _shared_subspace_gate_components(
        result: SharedSubspaceResult,
    ) -> SharedSubspaceGateComponents:
        """Extract raw gate components from shared subspace result.

        Returns individual measurements instead of a weighted composite.
        """
        if not result.is_valid:
            return SharedSubspaceGateComponents(
                correlation=0.0,
                variance_ratio=0.0,
                alignment_error=1.0,
            )
        correlation = max(
            0.0, min(1.0, result.alignment_strengths[0] if result.alignment_strengths else 0.0)
        )
        variance_ratio = max(0.0, min(1.0, result.shared_variance_ratio))
        alignment_error = max(0.0, min(1.0, result.alignment_error))
        return SharedSubspaceGateComponents(
            correlation=correlation,
            variance_ratio=variance_ratio,
            alignment_error=alignment_error,
        )

    def _compute_shared_subspace_omega(
        self,
        source_bases: SVDBases,
        target_bases: SVDBases,
        shared_subspace: SharedSubspaceContext,
        module_kind: ModuleKind,
    ) -> Array | None:
        source_proj_shape = self.backend.shape(shared_subspace.source_projection)
        target_proj_shape = self.backend.shape(shared_subspace.target_projection)
        source_dim = source_proj_shape[0]
        target_dim = target_proj_shape[0]
        source_basis = self._basis_for_shared_subspace(source_bases, source_dim, module_kind)
        target_basis = self._basis_for_shared_subspace(target_bases, target_dim, module_kind)
        if source_basis is None or target_basis is None:
            return None

        source_arr = self.backend.astype(source_basis, "float32")
        target_arr = self.backend.astype(target_basis, "float32")
        source_shape = self.backend.shape(source_arr)
        target_shape = self.backend.shape(target_arr)
        if source_shape[0] != source_dim:
            return None
        if target_shape[0] != target_dim:
            return None

        source_proj_t = self.backend.transpose(shared_subspace.source_projection)
        target_proj_t = self.backend.transpose(shared_subspace.target_projection)
        source_shared = self.backend.matmul(source_proj_t, source_arr)
        target_shared = self.backend.matmul(target_proj_t, target_arr)
        self.backend.eval(source_shared, target_shared)
        if self.backend.shape(source_shared) != self.backend.shape(target_shared):
            return None

        m = self.backend.matmul(self.backend.transpose(source_shared), target_shared)
        m = self.backend.astype(m, "float32")
        self.backend.eval(m)
        u, _, vt = self.backend.svd(m)
        self.backend.eval(u, vt)
        omega_pre = self.backend.matmul(u, vt)
        self.backend.eval(omega_pre)
        if self._determinant_sign(omega_pre, self.backend) < 0:
            # Flip last column of u
            u_shape = self.backend.shape(u)
            last_col = u[:, -1] * -1.0
            u = self.backend.concatenate([u[:, :-1], self.backend.reshape(last_col, (u_shape[0], 1))], axis=1)
            self.backend.eval(u)
        omega = self.backend.matmul(u, vt)
        omega = self.backend.astype(omega, "float32")
        self.backend.eval(omega)
        return omega

    def _blend_rotations(self, base: Array, blended: Array, weight: float) -> Array:
        base_shape = self.backend.shape(base)
        blended_shape = self.backend.shape(blended)
        if base_shape != blended_shape:
            return base
        clamped = max(0.0, min(1.0, float(weight)))
        if clamped <= 0.0:
            return base
        combined = (1.0 - clamped) * base + clamped * blended
        combined = self.backend.astype(combined, "float32")
        self.backend.eval(combined)
        u, _, vt = self.backend.svd(combined)
        self.backend.eval(u, vt)
        omega_pre = self.backend.matmul(u, vt)
        self.backend.eval(omega_pre)
        if self._determinant_sign(omega_pre, self.backend) < 0:
            u_shape = self.backend.shape(u)
            last_col = u[:, -1] * -1.0
            u = self.backend.concatenate([u[:, :-1], self.backend.reshape(last_col, (u_shape[0], 1))], axis=1)
            self.backend.eval(u)
        omega = self.backend.matmul(u, vt)
        omega = self.backend.astype(omega, "float32")
        self.backend.eval(omega)
        return omega

    @staticmethod
    def _basis_matching_projection(bases: SVDBases, dim: int) -> Array | None:
        if int(bases.u.shape[0]) == dim:
            return bases.u
        if int(bases.v.shape[0]) == dim:
            return bases.v
        return None

    @staticmethod
    def _basis_for_shared_subspace(
        bases: SVDBases,
        dim: int,
        module_kind: ModuleKind,
    ) -> Array | None:
        # Prefer output-space bases for residual outputs, input-space bases for MLP gate/up projections.
        if module_kind.is_residual_output and int(bases.u.shape[0]) == dim:
            return bases.u
        if module_kind.name in {"gate_proj", "up_proj"} and int(bases.v.shape[0]) == dim:
            return bases.v
        return RotationalMerger._basis_matching_projection(bases, dim)

    def _transport_guided_merge(
        self,
        source_weight: Array,
        target_weight: Array,
        target_key: str,
        layer_index: int,
        options: RotationalMergeOptions,
        transition_context: TransitionContext | None,
        consistency_context: ConsistencyContext | None,
    ) -> TransportMergeResult | None:
        source_shape = self.backend.shape(source_weight)
        target_shape = self.backend.shape(target_weight)
        if len(source_shape) != 2 or len(target_shape) != 2:
            return None
        if source_shape != target_shape:
            return None

        row_count = source_shape[0]
        min_samples = max(2, options.transport_min_samples)
        max_samples = max(min_samples, options.transport_max_samples)
        if row_count < min_samples:
            return None
        if row_count > max_samples:
            logger.info(
                "Transport-guided merge skipped for %s (rows=%s > max=%s).",
                target_key,
                row_count,
                max_samples,
            )
            return None

        transport_alpha = options.transport_blend_alpha
        if options.transport_use_intersection_confidence and options.intersection_map is not None:
            transport_alpha = TransportGuidedMerger.modulate_alpha_with_intersection(
                base_alpha=transport_alpha,
                intersection_map=options.intersection_map,
                layer=layer_index,
            )
        transport_alpha = self._transition_adjusted_alpha(
            transport_alpha,
            layer_index,
            transition_context,
            options,
        )
        transport_alpha = self._consistency_adjusted_alpha(
            transport_alpha,
            layer_index,
            consistency_context,
            options,
        )
        # Geometry flows through - no artificial clamping

        # Use weight rows as transport points (reference parity) while bounding
        # sample count to avoid the O(n^4) GW solver from exhausting CPU/ram.
        source_f32 = self.backend.astype(source_weight, "float32")
        target_f32 = self.backend.astype(target_weight, "float32")
        self.backend.eval(source_f32, target_f32)
        source_rows = self.backend.to_numpy(source_f32).tolist()
        target_rows = self.backend.to_numpy(target_f32).tolist()
        transport_config = TransportGuidedMerger.Config(
            coupling_threshold=options.transport_coupling_threshold,
            normalize_rows=True,
            blend_alpha=transport_alpha,
            use_intersection_confidence=False,
            min_samples=min_samples,
        )
        result = TransportGuidedMerger.synthesize_with_gw(
            source_activations=source_rows,
            target_activations=target_rows,
            source_weights=source_rows,
            target_weights=target_rows,
            config=transport_config,
        )
        if result is None:
            return None
        merged = self.backend.array(result.merged_weights, dtype="float32")
        self.backend.eval(merged)
        merged_shape = self.backend.shape(merged)
        if merged_shape != target_shape:
            logger.warning(
                "Transport-guided merge produced mismatched shape for %s: got %s, expected %s.",
                target_key,
                merged_shape,
                target_shape,
            )
            return None
        return TransportMergeResult(
            merged_weight=merged,
            gw_distance=float(result.gw_distance),
            marginal_error=float(result.marginal_error),
            effective_rank=int(result.effective_rank),
            converged=bool(result.converged),
            iterations=int(result.iterations),
        )

    @staticmethod
    def _extract_layer_index(key: str) -> int | None:
        parts = key.split(".")
        for idx, part in enumerate(parts):
            if part == "layers" and idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _module_kind_from_key(key: str) -> ModuleKind:
        lower = key.lower()
        if any(token in lower for token in ("q_proj", "wq")):
            return ModuleKind("q_proj", False)
        if any(token in lower for token in ("k_proj", "wk")):
            return ModuleKind("k_proj", False)
        if any(token in lower for token in ("v_proj", "wv")):
            return ModuleKind("v_proj", False)
        if any(token in lower for token in ("o_proj", "wo", "out_proj")):
            return ModuleKind("o_proj", True)
        is_mlp = ".mlp." in lower or ".feed_forward." in lower
        if "gate_proj" in lower or (is_mlp and "w1" in lower):
            return ModuleKind("gate_proj", False)
        if "up_proj" in lower or (is_mlp and "w3" in lower):
            return ModuleKind("up_proj", False)
        if "down_proj" in lower or (is_mlp and "w2" in lower):
            return ModuleKind("down_proj", True)
        return ModuleKind("other", False)

    @staticmethod
    def _is_projection_scope(key: str, scope: ModuleScope) -> bool:
        if scope == ModuleScope.all:
            return True
        lower = key.lower()
        return any(
            token in lower
            for token in (
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "wq",
                "wk",
                "wv",
                "wo",
            )
        )

    @staticmethod
    def _is_mlp_weight(key: str) -> bool:
        lower = key.lower()
        is_mlp = ".mlp." in lower or ".feed_forward." in lower
        return any(
            token in lower
            for token in (
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        ) or (is_mlp and any(token in lower for token in ("w1", "w2", "w3")))

    def _maybe_rebasin_source(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        anchors: SharedAnchors,
        options: RotationalMergeOptions,
        source_quantization: QuantizationConfig | None,
        target_quantization: QuantizationConfig | None,
    ) -> _RebasinResult:
        if options.anchor_mode != AnchorMode.rebasin:
            return _RebasinResult(weights=source_weights, quality=None, blocks_aligned=None)

        if options.module_scope != ModuleScope.all:
            raise ValueError("Rebasin anchor mode requires module scope 'all'")

        mlp_keys = [key for key in source_weights.keys() if self._is_mlp_weight(key)]
        source_converted: dict[str, Array] = {}
        target_converted: dict[str, Array] = {}
        for key in mlp_keys:
            if key in source_weights:
                source_hint = quantization_hint_for_key(key, source_quantization)
                source_np = dequantize_if_needed(
                    source_weights[key],
                    key,
                    source_weights,
                    self.backend,
                    hint=source_hint,
                )
                source_converted[key] = self.backend.astype(
                    self._to_array(source_np), "float32"
                )
            if key in target_weights:
                target_hint = quantization_hint_for_key(key, target_quantization)
                target_np = dequantize_if_needed(
                    target_weights[key],
                    key,
                    target_weights,
                    self.backend,
                    hint=target_hint,
                )
                target_converted[key] = self.backend.astype(
                    self._to_array(target_np), "float32"
                )

        aligned, quality, blocks = PermutationAligner.rebasin_mlp_only(
            source_converted,
            target_converted,
            anchors.source,
        )
        return _RebasinResult(
            weights={**source_weights, **aligned}, quality=quality, blocks_aligned=blocks
        )

    def _prepare_initial_omega(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        anchors: SharedAnchors,
        options: RotationalMergeOptions,
        source_quantization: QuantizationConfig | None,
        target_quantization: QuantizationConfig | None,
    ) -> Array:
        source_key, source_embed = AnchorExtractor.token_embedding_matrix(source_weights)
        target_key, target_embed = AnchorExtractor.token_embedding_matrix(target_weights)

        source_hint = quantization_hint_for_key(source_key, source_quantization)
        target_hint = quantization_hint_for_key(target_key, target_quantization)
        source_embed = dequantize_if_needed(
            source_embed,
            source_key,
            source_weights,
            self.backend,
            hint=source_hint,
        )
        target_embed = dequantize_if_needed(
            target_embed,
            target_key,
            target_weights,
            self.backend,
            hint=target_hint,
        )
        source_embed_arr = self.backend.astype(self._to_array(source_embed), "float32")
        target_embed_arr = self.backend.astype(self._to_array(target_embed), "float32")
        self.backend.eval(source_embed_arr, target_embed_arr)

        source_shape = self.backend.shape(source_embed_arr)
        target_shape = self.backend.shape(target_embed_arr)
        if source_shape[0] < source_shape[1]:
            source_embed_arr = self.backend.transpose(source_embed_arr)
        if target_shape[0] < target_shape[1]:
            target_embed_arr = self.backend.transpose(target_embed_arr)
        self.backend.eval(source_embed_arr, target_embed_arr)

        source_bases = self._truncated_svd_bases(
            source_embed_arr,
            rank=options.alignment_rank,
            oversampling=8,
            power_iterations=1,
            seed=0,
            label=source_key,
        )
        target_bases = self._truncated_svd_bases(
            target_embed_arr,
            rank=options.alignment_rank,
            oversampling=8,
            power_iterations=1,
            seed=0,
            label=target_key,
        )

        anchor_norms_source = self.backend.sqrt(
            self.backend.sum(anchors.source * anchors.source, axis=1, keepdims=True)
        )
        anchor_norms_target = self.backend.sqrt(
            self.backend.sum(anchors.target * anchors.target, axis=1, keepdims=True)
        )
        self.backend.eval(anchor_norms_source, anchor_norms_target)
        normalized_source = anchors.source / self.backend.maximum(
            anchor_norms_source,
            self.backend.array(1e-8, dtype="float32"),
        )
        normalized_target = anchors.target / self.backend.maximum(
            anchor_norms_target,
            self.backend.array(1e-8, dtype="float32"),
        )
        self.backend.eval(normalized_source, normalized_target)

        result = self.geometry.orthogonal_procrustes(
            normalized_source,
            normalized_target,
            source_bases.v,
            target_bases.v,
            anchor_weights=anchors.confidence_weights,
        )
        return result.omega

    def _compute_omega_out(
        self,
        target_key: str,
        source_weight: Array,
        target_weight: Array,
        source_bases: SVDBases,
        target_bases: SVDBases,
        current_omega_in: Array,
        anchors: SharedAnchors,
        layer_index: int,
        options: RotationalMergeOptions,
        shared_subspace: SharedSubspaceContext | None,
    ) -> Array:
        omega: Array
        if options.anchor_mode == AnchorMode.geometric:
            omega = self._geometric_omega_out(
                source_weight,
                target_weight,
                source_bases,
                target_bases,
                current_omega_in,
            )
        elif (
            options.anchor_mode == AnchorMode.intersection and options.intersection_map is not None
        ):
            omega, confidence = self._intersection_guided_rotation(
                options.intersection_map,
                layer_index,
                source_bases,
                target_bases,
            )
            if confidence <= 0.5:
                omega = self._geometric_omega_out(
                    source_weight,
                    target_weight,
                    source_bases,
                    target_bases,
                    current_omega_in,
                )
        else:
            anchor_shape = self.backend.shape(anchors.source)
            anchor_dim = int(anchor_shape[1])
            source_weight_shape = self.backend.shape(source_weight)
            use_v_basis = int(source_weight_shape[1]) == anchor_dim
            source_basis = source_bases.v if use_v_basis else source_bases.u
            target_basis = target_bases.v if use_v_basis else target_bases.u

            result = self.geometry.orthogonal_procrustes(
                anchors.source,
                anchors.target,
                source_basis,
                target_basis,
                anchor_weights=anchors.confidence_weights,
            )
            omega = result.omega
        if (
            shared_subspace is not None
            and options.use_shared_subspace_projection
            and options.shared_subspace_blend_weight > 0
        ):
            module_kind = self._module_kind_from_key(target_key)
            shared_omega = self._compute_shared_subspace_omega(
                source_bases,
                target_bases,
                shared_subspace,
                module_kind,
            )
            if shared_omega is not None:
                # Use correlation as the geometric gate - measures subspace alignment directly
                gate_from_correlation = shared_subspace.gate_components.correlation
                blend_weight = max(
                    0.0, min(1.0, options.shared_subspace_blend_weight * gate_from_correlation)
                )
                if blend_weight > 0:
                    omega = self._blend_rotations(omega, shared_omega, blend_weight)
        return omega

    def _geometric_omega_out(
        self,
        source_weight: Array,
        target_weight: Array,
        source_bases: SVDBases,
        target_bases: SVDBases,
        current_omega_in: Array,
    ) -> Array:
        s_src = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s_src = self.backend.matmul(s_src, source_bases.v)
        s_tgt = self.backend.matmul(self.backend.transpose(target_bases.u), target_weight)
        s_tgt = self.backend.matmul(s_tgt, target_bases.v)
        self.backend.eval(s_src, s_tgt)

        a = self.backend.matmul(s_src, self.backend.transpose(current_omega_in))
        m = self.backend.matmul(a, self.backend.transpose(s_tgt))
        m = self.backend.astype(m, "float32")
        self.backend.eval(m)

        u, _, vt = self.backend.svd(m)
        self.backend.eval(u, vt)
        omega_pre = self.backend.matmul(u, vt)
        self.backend.eval(omega_pre)
        if self._determinant_sign(omega_pre, self.backend) < 0:
            u_shape = self.backend.shape(u)
            last_col = u[:, -1] * -1.0
            u = self.backend.concatenate([u[:, :-1], self.backend.reshape(last_col, (u_shape[0], 1))], axis=1)
            self.backend.eval(u)
        omega = self.backend.matmul(u, vt)
        omega = self.backend.astype(omega, "float32")
        self.backend.eval(omega)
        return omega

    def _intersection_guided_rotation(
        self,
        intersection: IntersectionMap,
        layer_index: int,
        source_bases: SVDBases,
        target_bases: SVDBases,
    ) -> tuple[Array, float]:
        correlations = intersection.dimension_correlations.get(layer_index) or []
        if not correlations:
            k = int(self.backend.shape(source_bases.v)[1])
            eye = self.backend.eye(k, dtype="float32")
            self.backend.eval(eye)
            return eye, 0.0

        source_basis = self._basis_matching_hidden(source_bases, intersection.total_source_dims)
        target_basis = self._basis_matching_hidden(target_bases, intersection.total_target_dims)
        if source_basis is None or target_basis is None:
            k = int(self.backend.shape(source_bases.v)[1])
            eye = self.backend.eye(k, dtype="float32")
            self.backend.eval(eye)
            return eye, 0.0

        source_shape = self.backend.shape(source_basis)
        target_shape = self.backend.shape(target_basis)
        if source_shape[1] != target_shape[1]:
            k = min(source_shape[1], target_shape[1])
            eye = self.backend.eye(k, dtype="float32")
            self.backend.eval(eye)
            return eye, 0.0

        filtered = [
            entry
            for entry in correlations
            if 0 <= entry.source_dim < source_shape[0]
            and 0 <= entry.target_dim < target_shape[0]
        ]
        if len(filtered) < 2:
            k = int(source_shape[1])
            eye = self.backend.eye(k, dtype="float32")
            self.backend.eval(eye)
            return eye, 0.0

        source_indices = [entry.source_dim for entry in filtered]
        target_indices = [entry.target_dim for entry in filtered]
        z_source = self.backend.stack([source_basis[i] for i in source_indices], axis=0)
        z_target = self.backend.stack([target_basis[i] for i in target_indices], axis=0)
        self.backend.eval(z_source, z_target)

        weights = [max(0.0, entry.correlation) for entry in filtered]
        if len(weights) == len(source_indices):
            sqrt_weights = self.backend.sqrt(self.backend.array(weights, dtype="float32"))
            sqrt_weights = self.backend.reshape(sqrt_weights, (-1, 1))
            self.backend.eval(sqrt_weights)
            z_source = z_source * sqrt_weights
            z_target = z_target * sqrt_weights
            self.backend.eval(z_source, z_target)

        m = self.backend.matmul(self.backend.transpose(z_source), z_target)
        m = self.backend.astype(m, "float32")
        self.backend.eval(m)
        u, _, vt = self.backend.svd(m)
        self.backend.eval(u, vt)
        omega_pre = self.backend.matmul(u, vt)
        self.backend.eval(omega_pre)
        if self._determinant_sign(omega_pre, self.backend) < 0:
            u_shape = self.backend.shape(u)
            last_col = u[:, -1] * -1.0
            u = self.backend.concatenate([u[:, :-1], self.backend.reshape(last_col, (u_shape[0], 1))], axis=1)
            self.backend.eval(u)
        omega = self.backend.matmul(u, vt)
        omega = self.backend.astype(omega, "float32")
        self.backend.eval(omega)
        confidence = sum(weights) / len(weights) if weights else 0.0
        return omega, confidence

    @staticmethod
    def _basis_matching_hidden(bases: SVDBases, hidden_dim: int) -> Array | None:
        if int(bases.u.shape[0]) == hidden_dim:
            return bases.u
        if int(bases.v.shape[0]) == hidden_dim:
            return bases.v
        return None

    def _project_weight(
        self,
        source_weight: Array,
        source_bases: SVDBases,
        target_bases: SVDBases,
        omega_in: Array,
        omega_out: Array,
    ) -> Array:
        # Weight layout is [out, in]; project in spectral space (k x k).
        s = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s = self.backend.matmul(s, source_bases.v)
        if omega_out is not None:
            omega_out_arr = self.backend.astype(omega_out, "float32")
            s = self.backend.matmul(omega_out_arr, s)
        if omega_in is not None:
            s = self.backend.matmul(s, self.backend.transpose(omega_in))
        projected = self.backend.matmul(target_bases.u, s)
        projected = self.backend.matmul(projected, self.backend.transpose(target_bases.v))
        self.backend.eval(projected)
        return projected

    def _procrustes_error(
        self,
        source_weight: Array,
        target_weight: Array,
        source_bases: SVDBases,
        target_bases: SVDBases,
        omega_out: Array,
        omega_in: Array,
    ) -> float:
        s_src = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s_src = self.backend.matmul(s_src, source_bases.v)
        s_tgt = self.backend.matmul(self.backend.transpose(target_bases.u), target_weight)
        s_tgt = self.backend.matmul(s_tgt, target_bases.v)
        self.backend.eval(s_src, s_tgt)

        omega_out_arr = self.backend.astype(omega_out, "float32")
        projected = self.backend.matmul(omega_out_arr, s_src)
        projected = self.backend.matmul(projected, self.backend.transpose(omega_in))
        diff = projected - s_tgt
        self.backend.eval(diff)

        diff_sq = diff * diff
        target_sq = s_tgt * s_tgt
        self.backend.eval(diff_sq, target_sq)
        error_norm = self.backend.sqrt(self.backend.sum(diff_sq))
        target_norm = self.backend.sqrt(self.backend.sum(target_sq))
        self.backend.eval(error_norm, target_norm)
        error_value = float(error_norm.item())
        target_value = float(target_norm.item())
        if target_value <= 0:
            return 0.0
        return float(error_value / target_value)

    def _rotation_deviation(self, omega: Array) -> float:
        omega_shape = self.backend.shape(omega)
        if len(omega_shape) != 2 or omega_shape[0] != omega_shape[1]:
            return 0.0
        # Compute trace: sum of diagonal elements
        diag = self.backend.diag(omega)
        self.backend.eval(diag)
        trace = float(self.backend.sum(diag))
        k = float(omega_shape[0])
        deviation_sq = max(0.0, 2.0 * k - 2.0 * trace)
        return math.sqrt(deviation_sq)

    @staticmethod
    def _condition_number(singular_values: list[float]) -> float:
        if not singular_values:
            return float("inf")
        s_max = max(singular_values)
        s_min = min(val for val in singular_values if val > 0) if singular_values else 0.0
        if s_min <= 0:
            return float("inf")
        return float(s_max / s_min)

    @staticmethod
    def _spectral_ratio(target: list[float], source: list[float]) -> float:
        if not target or not source:
            return 1.0
        target_top = target[0]
        source_top = max(source[0], 1e-8)
        return float(target_top / source_top)

    def _truncated_svd_bases(
        self,
        weight: Array,
        rank: int,
        oversampling: int,
        power_iterations: int,
        seed: int,
        label: str,
    ) -> SVDBases:
        # Always convert to backend array if needed (handles numpy arrays)
        import numpy as np

        if isinstance(weight, np.ndarray):
            weight = self.backend.array(weight, dtype="float32")
            self.backend.eval(weight)

        weight_shape = self.backend.shape(weight) if hasattr(weight, "__len__") else None
        if weight_shape is None:
            weight_arr = self._to_array(weight)
            weight_shape = self.backend.shape(weight_arr)
            if len(weight_shape) != 2:
                raise ValueError(f"Unsupported weight shape for {label}: {weight_shape}")
            weight = self.backend.astype(weight_arr, "float32")
            self.backend.eval(weight)
        if len(weight_shape) != 2:
            raise ValueError(f"Unsupported weight shape for {label}: {weight_shape}")

        dtype = self.backend.dtype(weight)
        dtype_str = str(dtype).lower()
        if "float" not in dtype_str:
            logger.warning(
                "Non-float weight %s dtype=%s; casting to float32 without dequantization.",
                label,
                dtype,
            )

        out_dim, in_dim = (int(weight_shape[0]), int(weight_shape[1]))
        min_dim = min(out_dim, in_dim)
        if rank > min_dim:
            raise ValueError(f"Alignment rank must be <= min(out,in)={min_dim} for {label}")

        k = rank
        l = k + max(0, oversampling)

        # Randomized SVD keeps merge parity while letting the backend handle dense matmuls.
        self.backend.random_seed(seed)
        omega = self.backend.random_normal((in_dim, l))
        omega = self.backend.astype(omega, "float32")
        self.backend.eval(omega)
        y = self.backend.matmul(weight, omega)
        self.backend.eval(y)

        weight_t = None
        for _ in range(max(0, power_iterations)):
            if weight_t is None:
                weight_t = self.backend.transpose(weight)
            y = self.backend.matmul(weight, self.backend.matmul(weight_t, y))
            self.backend.eval(y)

        y = self.backend.astype(y, "float32")
        self.backend.eval(y)
        q, _ = self.backend.qr(y)
        self.backend.eval(q)

        q_t = self.backend.transpose(q)
        b = self.backend.matmul(q_t, weight)
        b = self.backend.astype(b, "float32")
        self.backend.eval(b)
        u_hat, s_arr, vt = self.backend.svd(b)
        self.backend.eval(u_hat, s_arr, vt)

        # Extract top k components
        u_small = u_hat[:, :k]
        v = self.backend.transpose(vt)[:, :k]
        u_arr = self.backend.matmul(q, u_small)
        u_arr = self.backend.astype(u_arr, "float32")
        v_arr = self.backend.astype(v, "float32")
        self.backend.eval(u_arr, v_arr, s_arr)

        # Convert singular values to list
        s_list = self.backend.to_numpy(s_arr).tolist()
        sigma0 = max(s_list) if s_list else 0.0
        if not math.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError(f"Non-finite spectral norm while computing SVD for {label}")

        singular_values = [float(val) for val in s_list]
        if len(singular_values) < k:
            raise ValueError(
                f"Unexpected singular value count for {label}: expected >= {k}, got {len(singular_values)}"
            )

        return SVDBases(
            u=u_arr,
            v=v_arr,
            spectral_norm=sigma0,
            singular_values=singular_values,
        )

    def _select_source_weight(
        self,
        key: str,
        fallback: Array,
        preprocessed: _RebasinResult,
        options: RotationalMergeOptions,
    ) -> Array:
        if options.anchor_mode == AnchorMode.rebasin and self._is_mlp_weight(key):
            candidate = preprocessed.weights.get(key)
            if candidate is not None:
                return self._to_array(candidate)
        return self.backend.astype(self._to_array(fallback), "float32")

    def _quantize_blended(
        self,
        blended: Array,
        target_key: str,
        raw_weight_shape: tuple[int, ...],
        blended_shape: tuple[int, ...],
        target_weights: dict[str, Any],
        hint: QuantizationHint | None,
    ) -> _QuantizedResult | None:
        base = target_key.replace(".weight", "")
        scales_key = f"{base}.scales"
        biases_key = f"{base}.biases"
        scales_val = target_weights.get(scales_key)
        if scales_val is None:
            logger.warning("Quantized target %s missing scales; keeping float output.", target_key)
            return None
        scales_arr = self._to_array(scales_val)
        scales_shape = self.backend.shape(scales_arr)
        params = resolve_quantization(
            base_key=target_key,
            weight_shape=raw_weight_shape,
            scales_shape=scales_shape,
            hint=hint,
            biases_present=biases_key in target_weights,
        )
        if params is None:
            logger.warning(
                "Unable to infer quantization parameters for %s; keeping float output.", target_key
            )
            return None
        if len(blended_shape) != 2 or blended_shape[1] % params.group_size != 0:
            logger.warning(
                "Quantization parameters incompatible with %s (group_size=%s).",
                target_key,
                params.group_size,
            )
            return None
        try:
            q_weight, q_scales, q_biases = self.backend.quantize(
                blended,
                group_size=params.group_size,
                bits=params.bits,
                mode=params.mode,
            )
        except Exception as exc:
            logger.warning("Quantization failed for %s: %s", target_key, exc)
            return None
        return _QuantizedResult(
            weight=q_weight,
            scales=q_scales,
            biases=q_biases,
            scales_key=scales_key,
            biases_key=biases_key,
        )

    def _to_array(self, value: Any) -> Array:
        """Convert any value to a backend Array."""
        # If already an array-like with shape, assume it's a backend array
        if hasattr(value, "shape") and hasattr(value, "__array__"):
            return value
        try:
            return self.backend.array(value)
        except Exception:
            # Try via to_numpy first
            try:
                return self.backend.array(self.backend.to_numpy(value))
            except Exception:
                return self.backend.array(value)

    def _anchor_gram(self, anchor_matrix: Array) -> tuple[list[float], int]:
        anchor_arr = self.backend.astype(anchor_matrix, "float32")
        self.backend.eval(anchor_arr)
        anchor_shape = self.backend.shape(anchor_arr)
        if len(anchor_shape) != 2 or anchor_shape[0] == 0:
            return [], 0
        # Cosine-normalize anchors so Gram comparisons are scale-invariant.
        norms = self.backend.norm(anchor_arr, axis=1, keepdims=True)
        norms = self.backend.maximum(norms, self.backend.array(1e-8, dtype="float32"))
        self.backend.eval(norms)
        normalized = anchor_arr / norms
        gram = self.backend.matmul(normalized, self.backend.transpose(normalized))
        self.backend.eval(gram)
        n = anchor_shape[0]
        gram_flat = self.backend.reshape(gram, (n * n,))
        self.backend.eval(gram_flat)
        return self.backend.to_numpy(gram_flat).tolist(), n

    @staticmethod
    def _determinant_sign(matrix: Array, backend: "Backend") -> float:
        matrix_shape = backend.shape(matrix)
        k = matrix_shape[0]
        if k == 0 or k != matrix_shape[1]:
            return 1.0
        if k == 1:
            val = float(backend.to_numpy(matrix[0, 0]))
            return 1.0 if val >= 0 else -1.0

        # Convert to numpy for LU-style determinant sign computation
        # (this is a small k x k matrix, so numpy is fine here)
        backend.eval(matrix)
        a = backend.to_numpy(matrix).astype(float).copy()
        sign = 1.0
        for i in range(k):
            max_row = i
            max_val = abs(a[i, i])
            for row in range(i + 1, k):
                val = abs(a[row, i])
                if val > max_val:
                    max_val = val
                    max_row = row
            if max_row != i:
                a[[i, max_row], :] = a[[max_row, i], :]
                sign = -sign
            pivot = a[i, i]
            if abs(pivot) < 1e-10:
                return 1.0
            for row in range(i + 1, k):
                factor = a[row, i] / pivot
                a[row, i:k] -= factor * a[i, i:k]

        diag = [a[i, i] for i in range(k)]
        diag_product = 1.0
        for d in diag:
            diag_product *= d
        return 1.0 if diag_product * sign >= 0 else -1.0


@dataclass(frozen=True)
class _RebasinResult:
    weights: dict[str, Any]
    quality: float | None
    blocks_aligned: int | None


@dataclass(frozen=True)
class _QuantizedResult:
    weight: Array
    scales: Array
    biases: Array | None
    scales_key: str
    biases_key: str
