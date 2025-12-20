from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.cross_cultural_geometry import AlignmentAnalysis, CrossCulturalGeometry
from modelcypher.core.domain.manifold_stitcher import IntersectionMap
from modelcypher.core.domain.shared_subspace_projector import (
    Config as SharedSubspaceConfig,
    Result as SharedSubspaceResult,
    SharedSubspaceProjector,
)
from modelcypher.core.domain.transport_guided_merger import TransportGuidedMerger
from modelcypher.core.domain.transfer_fidelity import Prediction, TransferFidelityPrediction
from modelcypher.core.use_cases.anchor_extractor import AnchorExtractor
from modelcypher.core.use_cases.geometry_engine import GeometryEngine
from modelcypher.core.use_cases.permutation_aligner import PermutationAligner
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
class SharedSubspaceContext:
    layer: int
    result: SharedSubspaceResult
    source_projection: np.ndarray
    target_projection: np.ndarray
    gate: float


@dataclass(frozen=True)
class SharedSubspaceMetrics:
    shared_dimension: int
    alignment_error: float
    shared_variance_ratio: float
    top_correlation: float
    sample_count: int
    method: str
    is_valid: bool


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
    merged_weight: np.ndarray
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
    transition_gate_min_ratio: float = 0.7
    transition_gate_max_ratio: float = 1.3
    consistency_gate_strength: float = 0.0
    consistency_gate_layer_samples: int = 6
    use_shared_subspace_projection: bool = False
    shared_subspace_config: SharedSubspaceConfig | None = None
    shared_subspace_blend_weight: float = 0.0
    use_transport_guided: bool = False
    transport_coupling_threshold: float = 0.001
    transport_blend_alpha: float = 0.5
    transport_min_samples: int = 5
    transport_max_samples: int = 32
    transport_use_intersection_confidence: bool = True


class RotationalMerger:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.geometry = GeometryEngine(backend)
        self.permutation = PermutationAligner(backend)

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
    ) -> tuple[dict[str, np.ndarray], MergeAnalysisResult]:
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
        shared_subspace_context, shared_subspace_metrics = self._prepare_shared_subspace_context(
            source_crm,
            target_crm,
            options,
        )
        alignment_rank = self._resolve_alignment_rank(options, shared_subspace_context)

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

        merged_weights: dict[str, np.ndarray] = {}
        # Preserve all target weights by default; merged layers overwrite as needed.
        for key, value in target_weights.items():
            merged_weights[key] = self._to_numpy(value)

        layer_metrics: list[LayerMergeMetric] = []
        errors: list[float] = []
        roughness: list[float] = []
        previous_omega: np.ndarray | None = None
        transport_distances: list[float] = []
        transport_marginals: list[float] = []
        transport_ranks: list[int] = []
        transport_converged = 0
        transport_skipped = 0

        target_weight_keys = sorted(
            key
            for key in target_weights.keys()
            if (key.startswith("layers.") or key.startswith("model.layers.")) and key.endswith(".weight")
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

            target_np = self._to_numpy(target_raw)
            source_np = self._to_numpy(source_raw)
            if target_np.ndim != 2 or source_np.ndim != 2:
                continue
            target_is_quantized = target_np.dtype.kind not in {"f"}

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
            if source_weight_np.ndim != 2 or target_weight_np.ndim != 2:
                continue
            if source_weight_np.shape != target_weight_np.shape:
                continue

            source_weight = self._select_source_weight(
                target_key,
                source_weight_np,
                preprocessed_source,
                options,
            )
            target_weight = self.backend.array(target_weight_np.astype(np.float32), dtype=np.float32)
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
                merged_np = transport_result.merged_weight
                merged_arr = self.backend.array(merged_np.astype(np.float32), dtype=np.float32)
                self.backend.eval(merged_arr)
                omega_out = np.eye(alignment_rank, dtype=np.float32)
                module_kind = self._module_kind_from_key(target_key)
                rotation_deviation = self._rotation_deviation(omega_out)
                condition_number = 1.0
                if options.anchor_mode == AnchorMode.geometric:
                    condition_number = self._condition_number(source_bases.singular_values)
                spectral_ratio = self._spectral_ratio(
                    target_bases.singular_values, source_bases.singular_values
                )
                errors.append(transport_result.gw_distance)
                if previous_omega is not None and previous_omega.shape == omega_out.shape:
                    roughness.append(float(np.linalg.norm(omega_out - previous_omega)))
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
                    merged_weights[target_key] = merged_np.astype(target_np.dtype, copy=False)
                else:
                    quantized = self._quantize_blended(
                        merged_arr,
                        target_key,
                        raw_weight_shape=target_np.shape,
                        blended_shape=merged_np.shape,
                        target_weights=target_weights,
                        hint=target_hint,
                    )
                    if quantized is None:
                        merged_weights[target_key] = merged_np.astype(np.float32, copy=False)
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
                    current_omega_in = self.backend.array(omega_out, dtype=np.float32)
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
                shared_subspace_context,
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

            spectral_ratio = self._spectral_ratio(target_bases.singular_values, source_bases.singular_values)

            if previous_omega is not None and previous_omega.shape == omega_out.shape:
                roughness.append(float(np.linalg.norm(omega_out - previous_omega)))
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
                layer_confidence = self.lookup_layer_confidence(options.intersection_map, layer_index)
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

            alpha_value = self.backend.array(np.array(effective_alpha, dtype=np.float32), dtype=np.float32)
            blended = (alpha_value * target_weight) + ((1.0 - alpha_value) * projected)
            self.backend.eval(blended)

            if not target_is_quantized:
                merged_np = self._to_numpy(blended).astype(target_np.dtype, copy=False)
                merged_weights[target_key] = merged_np
            else:
                quantized = self._quantize_blended(
                    blended,
                    target_key,
                    raw_weight_shape=target_np.shape,
                    blended_shape=target_weight_np.shape,
                    target_weights=target_weights,
                    hint=target_hint,
                )
                if quantized is None:
                    merged_weights[target_key] = self._to_numpy(blended).astype(np.float32, copy=False)
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
                current_omega_in = self.backend.array(omega_out, dtype=np.float32)
                self.backend.eval(current_omega_in)

        mean_error = float(np.mean(errors)) if errors else 0.0
        max_error = float(np.max(errors)) if errors else 0.0
        roughness_value = float(np.mean(roughness)) if roughness else 0.0

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
            mean_distance = float(np.mean(transport_distances)) if transport_distances else 0.0
            mean_marginal = float(np.mean(transport_marginals)) if transport_marginals else 0.0
            mean_rank = float(np.mean(transport_ranks)) if transport_ranks else 0.0
            transport_metrics = TransportMetrics(
                mean_gw_distance=mean_distance,
                mean_marginal_error=mean_marginal,
                mean_effective_rank=mean_rank,
                layer_count=count,
                converged_layers=transport_converged,
                skipped_layers=transport_skipped,
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
        )
        return merged_weights, analysis

    def build_shared_anchors(
        self,
        source_anchors: dict[str, np.ndarray],
        target_anchors: dict[str, np.ndarray],
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

        source_vectors: list[np.ndarray] = []
        target_vectors: list[np.ndarray] = []
        confidence_weights: list[float] = []

        for anchor_id in shared_ids:
            source_vec = source_anchors[anchor_id]
            target_vec = target_anchors[anchor_id]
            source_vectors.append(np.asarray(source_vec, dtype=np.float32))
            target_vectors.append(np.asarray(target_vec, dtype=np.float32))
            source_conf = float(source_confidence.get(anchor_id, 0.5))
            target_conf = float(target_confidence.get(anchor_id, 0.5))
            confidence_weights.append(float(np.sqrt(source_conf * target_conf)))

        source_matrix = AnchorExtractor.normalize_anchor_matrix(np.stack(source_vectors, axis=0))
        target_matrix = AnchorExtractor.normalize_anchor_matrix(np.stack(target_vectors, axis=0))

        source_array = self.backend.array(source_matrix, dtype=np.float32)
        target_array = self.backend.array(target_matrix, dtype=np.float32)
        self.backend.eval(source_array, target_array)

        return SharedAnchors(
            source=source_array,
            target=target_array,
            anchor_ids=shared_ids,
            confidence_weights=confidence_weights,
        )

    @staticmethod
    def confidence_based_alpha(layer_confidence: float | None, fallback_alpha: float) -> float:
        if layer_confidence is None:
            return fallback_alpha
        raw = 1.0 - (layer_confidence * 0.8)
        return RotationalMerger._clamp_alpha(raw)

    @staticmethod
    def lookup_layer_confidence(intersection: IntersectionMap, layer_index: int) -> float | None:
        for entry in intersection.layer_confidences:
            if entry.layer == layer_index:
                return entry.confidence
        return None

    @staticmethod
    def _clamp_alpha(value: float) -> float:
        return max(0.2, min(0.95, float(value)))

    # Align gating math with TrainingCypher to keep cross-repo merge behavior comparable.
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
        ratio = transition_context.delta_alignment_by_layer.get(layer, transition_context.transition_advantage)
        if not np.isfinite(ratio):
            return base_alpha
        clamped_ratio = max(options.transition_gate_min_ratio, min(options.transition_gate_max_ratio, ratio))
        target_alpha = base_alpha * (2.0 - clamped_ratio)
        blended = base_alpha * (1.0 - strength) + target_alpha * strength
        return RotationalMerger._clamp_alpha(blended)

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
        if target_weight is None or not np.isfinite(target_weight):
            return base_alpha
        blended = base_alpha * (1.0 - strength) + float(target_weight) * strength
        return RotationalMerger._clamp_alpha(blended)

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
    ) -> tuple[SharedSubspaceContext | None, SharedSubspaceMetrics | None]:
        if not options.use_shared_subspace_projection:
            return None, None
        if source_crm is None or target_crm is None:
            logger.warning("Shared subspace projection enabled but CRM inputs are missing.")
            return None, None

        # Default to terminal layer to avoid costly cross-layer matching during merges.
        max_layer = min(source_crm.layer_count, target_crm.layer_count) - 1
        if max_layer < 0:
            return None, None

        config = options.shared_subspace_config or SharedSubspaceConfig()
        result = SharedSubspaceProjector.discover(source_crm, target_crm, max_layer, config)
        if result is None:
            logger.warning("Shared subspace discovery failed.")
            return None, None

        source_projection = np.asarray(result.source_projection, dtype=np.float32)
        target_projection = np.asarray(result.target_projection, dtype=np.float32)
        gate = self._shared_subspace_gate(result, options)

        top_correlation = float(result.alignment_strengths[0]) if result.alignment_strengths else 0.0
        metrics = SharedSubspaceMetrics(
            shared_dimension=result.shared_dimension,
            alignment_error=result.alignment_error,
            shared_variance_ratio=result.shared_variance_ratio,
            top_correlation=top_correlation,
            sample_count=result.sample_count,
            method=result.method.value,
            is_valid=result.is_valid,
        )
        context = SharedSubspaceContext(
            layer=max_layer,
            result=result,
            source_projection=source_projection,
            target_projection=target_projection,
            gate=gate,
        )
        return context, metrics

    @staticmethod
    def _resolve_alignment_rank(
        options: RotationalMergeOptions,
        shared_subspace: SharedSubspaceContext | None,
    ) -> int:
        if shared_subspace is None or not shared_subspace.result.is_valid:
            return options.alignment_rank
        shared_dim = shared_subspace.result.shared_dimension
        if shared_dim <= 0:
            return options.alignment_rank
        # Only shrink rank when shared subspace quality is validated.
        return min(options.alignment_rank, shared_dim)

    @staticmethod
    def _shared_subspace_gate(
        result: SharedSubspaceResult,
        options: RotationalMergeOptions,
    ) -> float:
        if not result.is_valid:
            return 0.0
        correlation = max(0.0, min(1.0, result.alignment_strengths[0] if result.alignment_strengths else 0.0))
        variance = max(0.0, min(1.0, result.shared_variance_ratio))
        error_penalty = max(0.0, min(1.0, 1.0 - result.alignment_error))
        base_gate = max(0.0, min(1.0, (0.4 * correlation) + (0.4 * variance) + (0.2 * error_penalty)))
        blend_weight = max(0.0, min(1.0, options.shared_subspace_blend_weight))
        return max(0.0, min(1.0, 1.0 - blend_weight + blend_weight * base_gate))

    def _compute_shared_subspace_omega(
        self,
        source_bases: SVDBases,
        target_bases: SVDBases,
        shared_subspace: SharedSubspaceContext,
    ) -> np.ndarray | None:
        source_basis = self._basis_matching_projection(source_bases, shared_subspace.source_projection.shape[0])
        target_basis = self._basis_matching_projection(target_bases, shared_subspace.target_projection.shape[0])
        if source_basis is None or target_basis is None:
            return None

        source_np = self._to_numpy(source_basis).astype(np.float32)
        target_np = self._to_numpy(target_basis).astype(np.float32)
        if source_np.shape[0] != shared_subspace.source_projection.shape[0]:
            return None
        if target_np.shape[0] != shared_subspace.target_projection.shape[0]:
            return None

        source_shared = shared_subspace.source_projection.T @ source_np
        target_shared = shared_subspace.target_projection.T @ target_np
        if source_shared.shape != target_shared.shape:
            return None

        m = source_shared.T @ target_shared
        u, _, vt = np.linalg.svd(m.astype(np.float32), full_matrices=False)
        omega_pre = u @ vt
        if self._determinant_sign(omega_pre) < 0:
            u[:, -1] *= -1.0
        omega = u @ vt
        return omega.astype(np.float32)

    @staticmethod
    def _blend_rotations(base: np.ndarray, blended: np.ndarray, weight: float) -> np.ndarray:
        if base.shape != blended.shape:
            return base
        clamped = max(0.0, min(1.0, float(weight)))
        if clamped <= 0.0:
            return base
        combined = (1.0 - clamped) * base + clamped * blended
        u, _, vt = np.linalg.svd(combined.astype(np.float32), full_matrices=False)
        omega_pre = u @ vt
        if RotationalMerger._determinant_sign(omega_pre) < 0:
            u[:, -1] *= -1.0
        return (u @ vt).astype(np.float32)

    @staticmethod
    def _basis_matching_projection(bases: SVDBases, dim: int) -> Array | None:
        if int(bases.u.shape[0]) == dim:
            return bases.u
        if int(bases.v.shape[0]) == dim:
            return bases.v
        return None

    def _transport_guided_merge(
        self,
        source_weight_np: np.ndarray,
        target_weight_np: np.ndarray,
        target_key: str,
        layer_index: int,
        options: RotationalMergeOptions,
        transition_context: TransitionContext | None,
        consistency_context: ConsistencyContext | None,
    ) -> TransportMergeResult | None:
        if source_weight_np.ndim != 2 or target_weight_np.ndim != 2:
            return None
        if source_weight_np.shape != target_weight_np.shape:
            return None

        row_count = source_weight_np.shape[0]
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
        transport_alpha = self._clamp_alpha(transport_alpha)

        # Use weight rows as transport points (TrainingCypher parity) while bounding
        # sample count to avoid the O(n^4) GW solver from exhausting CPU/ram.
        source_rows = source_weight_np.astype(np.float32, copy=False).tolist()
        target_rows = target_weight_np.astype(np.float32, copy=False).tolist()
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
        merged = np.asarray(result.merged_weights, dtype=np.float32)
        if merged.shape != target_weight_np.shape:
            logger.warning(
                "Transport-guided merge produced mismatched shape for %s: got %s, expected %s.",
                target_key,
                merged.shape,
                target_weight_np.shape,
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

        mlp_keys = [key for key in source_weights.keys() if PermutationAligner.is_mlp_weight(key)]
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
                source_converted[key] = self.backend.array(source_np.astype(np.float32), dtype=np.float32)
            if key in target_weights:
                target_hint = quantization_hint_for_key(key, target_quantization)
                target_np = dequantize_if_needed(
                    target_weights[key],
                    key,
                    target_weights,
                    self.backend,
                    hint=target_hint,
                )
                target_converted[key] = self.backend.array(target_np.astype(np.float32), dtype=np.float32)

        aligned, quality, blocks = self.permutation.rebasin_mlp_only(
            source_converted,
            target_converted,
            anchors.source,
        )
        return _RebasinResult(weights={**source_weights, **aligned}, quality=quality, blocks_aligned=blocks)

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
        source_embed = np.asarray(source_embed, dtype=np.float32)
        target_embed = np.asarray(target_embed, dtype=np.float32)

        if source_embed.shape[0] < source_embed.shape[1]:
            source_embed = source_embed.T
        if target_embed.shape[0] < target_embed.shape[1]:
            target_embed = target_embed.T

        source_embed_arr = self.backend.array(source_embed, dtype=np.float32)
        target_embed_arr = self.backend.array(target_embed, dtype=np.float32)
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
            self.backend.array(1e-8, dtype=np.float32),
        )
        normalized_target = anchors.target / self.backend.maximum(
            anchor_norms_target,
            self.backend.array(1e-8, dtype=np.float32),
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
    ) -> np.ndarray:
        omega: np.ndarray
        if options.anchor_mode == AnchorMode.geometric:
            omega = self._geometric_omega_out(
                source_weight,
                target_weight,
                source_bases,
                target_bases,
                current_omega_in,
            )
        elif options.anchor_mode == AnchorMode.intersection and options.intersection_map is not None:
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
            anchor_dim = int(anchors.source.shape[1])
            use_v_basis = int(source_weight.shape[1]) == anchor_dim
            source_basis = source_bases.v if use_v_basis else source_bases.u
            target_basis = target_bases.v if use_v_basis else target_bases.u

            result = self.geometry.orthogonal_procrustes(
                anchors.source,
                anchors.target,
                source_basis,
                target_basis,
                anchor_weights=anchors.confidence_weights,
            )
            omega = self._to_numpy(result.omega)
        if (
            shared_subspace is not None
            and options.use_shared_subspace_projection
            and options.shared_subspace_blend_weight > 0
        ):
            shared_omega = self._compute_shared_subspace_omega(
                source_bases,
                target_bases,
                shared_subspace,
            )
            if shared_omega is not None:
                blend_weight = max(0.0, min(1.0, options.shared_subspace_blend_weight * shared_subspace.gate))
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
    ) -> np.ndarray:
        s_src = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s_src = self.backend.matmul(s_src, source_bases.v)
        s_tgt = self.backend.matmul(self.backend.transpose(target_bases.u), target_weight)
        s_tgt = self.backend.matmul(s_tgt, target_bases.v)
        self.backend.eval(s_src, s_tgt)

        a = self.backend.matmul(s_src, self.backend.transpose(current_omega_in))
        m = self.backend.matmul(a, self.backend.transpose(s_tgt))
        self.backend.eval(a, m)

        m_cpu = self._to_numpy(m).astype(np.float32)
        u, _, vt = np.linalg.svd(m_cpu, full_matrices=False)
        omega_pre = u @ vt
        if self._determinant_sign(omega_pre) < 0:
            u[:, -1] *= -1.0
        omega = u @ vt
        return omega.astype(np.float32)

    def _intersection_guided_rotation(
        self,
        intersection: IntersectionMap,
        layer_index: int,
        source_bases: SVDBases,
        target_bases: SVDBases,
    ) -> tuple[np.ndarray, float]:
        correlations = intersection.dimension_correlations.get(layer_index) or []
        if not correlations:
            k = int(source_bases.v.shape[1])
            return np.eye(k, dtype=np.float32), 0.0

        source_basis = self._basis_matching_hidden(source_bases, intersection.total_source_dims)
        target_basis = self._basis_matching_hidden(target_bases, intersection.total_target_dims)
        if source_basis is None or target_basis is None:
            k = int(source_bases.v.shape[1])
            return np.eye(k, dtype=np.float32), 0.0

        source_np = self._to_numpy(source_basis)
        target_np = self._to_numpy(target_basis)
        if source_np.shape[1] != target_np.shape[1]:
            k = min(source_np.shape[1], target_np.shape[1])
            return np.eye(k, dtype=np.float32), 0.0

        filtered = [
            entry
            for entry in correlations
            if 0 <= entry.source_dim < source_np.shape[0] and 0 <= entry.target_dim < target_np.shape[0]
        ]
        if len(filtered) < 2:
            k = int(source_np.shape[1])
            return np.eye(k, dtype=np.float32), 0.0

        source_indices = np.array([entry.source_dim for entry in filtered], dtype=np.int32)
        target_indices = np.array([entry.target_dim for entry in filtered], dtype=np.int32)
        z_source = source_np[source_indices]
        z_target = target_np[target_indices]

        weights = np.array([max(0.0, entry.correlation) for entry in filtered], dtype=np.float32)
        if weights.size == z_source.shape[0]:
            sqrt_weights = np.sqrt(weights).reshape((-1, 1))
            z_source = z_source * sqrt_weights
            z_target = z_target * sqrt_weights

        m = z_source.T @ z_target
        u, _, vt = np.linalg.svd(m.astype(np.float32), full_matrices=False)
        omega_pre = u @ vt
        if self._determinant_sign(omega_pre) < 0:
            u[:, -1] *= -1.0
        omega = (u @ vt).astype(np.float32)
        confidence = float(weights.mean()) if weights.size else 0.0
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
        omega_out: np.ndarray,
    ) -> Array:
        # Weight layout is [out, in]; project in spectral space (k x k).
        s = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s = self.backend.matmul(s, source_bases.v)
        if omega_out is not None:
            omega_out_arr = self.backend.array(omega_out, dtype=np.float32)
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
        omega_out: np.ndarray,
        omega_in: Array,
    ) -> float:
        s_src = self.backend.matmul(self.backend.transpose(source_bases.u), source_weight)
        s_src = self.backend.matmul(s_src, source_bases.v)
        s_tgt = self.backend.matmul(self.backend.transpose(target_bases.u), target_weight)
        s_tgt = self.backend.matmul(s_tgt, target_bases.v)
        self.backend.eval(s_src, s_tgt)

        omega_out_arr = self.backend.array(omega_out, dtype=np.float32)
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

    @staticmethod
    def _rotation_deviation(omega: np.ndarray) -> float:
        if omega.ndim != 2 or omega.shape[0] != omega.shape[1]:
            return 0.0
        trace = float(np.trace(omega))
        k = float(omega.shape[0])
        deviation_sq = max(0.0, 2.0 * k - 2.0 * trace)
        return float(np.sqrt(deviation_sq))

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
        raw_np = self._to_numpy(weight)
        if raw_np.dtype not in (np.float16, np.float32, np.float64):
            logger.warning(
                "Non-float weight %s dtype=%s; casting to float32 without dequantization.",
                label,
                raw_np.dtype,
            )
        weight_np = raw_np.astype(np.float32)
        if weight_np.ndim != 2:
            raise ValueError(f"Unsupported weight shape for {label}: {weight_np.shape}")

        out_dim, in_dim = weight_np.shape
        min_dim = min(out_dim, in_dim)
        if rank > min_dim:
            raise ValueError(
                f"Alignment rank must be <= min(out,in)={min_dim} for {label}"
            )

        k = rank
        l = k + max(0, oversampling)

        # Randomized SVD keeps the merge parity with TrainingCypher while avoiding full decompositions.
        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((in_dim, l), dtype=np.float32)
        y = weight_np @ omega

        for _ in range(max(0, power_iterations)):
            y = weight_np @ (weight_np.T @ y)

        q, _ = np.linalg.qr(y, mode="reduced")
        b = q.T @ weight_np
        u_hat, s, v_t = np.linalg.svd(b, full_matrices=False)

        u_small = u_hat[:, :k]
        v = v_t.T[:, :k]
        u = q @ u_small

        sigma0 = float(np.max(s)) if s.size else 0.0
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError(f"Non-finite spectral norm while computing SVD for {label}")

        singular_values = [float(val) for val in s]
        if len(singular_values) < k:
            raise ValueError(
                f"Unexpected singular value count for {label}: expected >= {k}, got {len(singular_values)}"
            )

        u_arr = self.backend.array(u.astype(np.float32), dtype=np.float32)
        v_arr = self.backend.array(v.astype(np.float32), dtype=np.float32)
        self.backend.eval(u_arr, v_arr)

        return SVDBases(
            u=u_arr,
            v=v_arr,
            spectral_norm=sigma0,
            singular_values=singular_values,
        )

    def _select_source_weight(
        self,
        key: str,
        fallback_np: np.ndarray,
        preprocessed: _RebasinResult,
        options: RotationalMergeOptions,
    ) -> Array:
        if options.anchor_mode == AnchorMode.rebasin and PermutationAligner.is_mlp_weight(key):
            candidate = preprocessed.weights.get(key)
            if candidate is not None and not isinstance(candidate, np.ndarray):
                return candidate
        return self.backend.array(fallback_np.astype(np.float32), dtype=np.float32)

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
        scales_np = self._to_numpy(scales_val)
        params = resolve_quantization(
            base_key=target_key,
            weight_shape=raw_weight_shape,
            scales_shape=scales_np.shape,
            hint=hint,
            biases_present=biases_key in target_weights,
        )
        if params is None:
            logger.warning("Unable to infer quantization parameters for %s; keeping float output.", target_key)
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
            weight=self._to_numpy(q_weight),
            scales=self._to_numpy(q_scales),
            biases=self._to_numpy(q_biases) if q_biases is not None else None,
            scales_key=scales_key,
            biases_key=biases_key,
        )

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            return np.asarray(self.backend.to_numpy(value))
        except Exception:
            return np.asarray(value)

    def _anchor_gram(self, anchor_matrix: Array) -> tuple[list[float], int]:
        anchor_np = self._to_numpy(anchor_matrix).astype(np.float32, copy=False)
        if anchor_np.ndim != 2 or anchor_np.shape[0] == 0:
            return [], 0
        # Cosine-normalize anchors so Gram comparisons are scale-invariant.
        norms = np.linalg.norm(anchor_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = anchor_np / norms
        gram = normalized @ normalized.T
        n = gram.shape[0]
        return gram.reshape(n * n).tolist(), n

    @staticmethod
    def _determinant_sign(matrix: np.ndarray) -> float:
        k = matrix.shape[0]
        if k == 0 or k != matrix.shape[1]:
            return 1.0
        if k == 1:
            return 1.0 if matrix[0, 0] >= 0 else -1.0

        a = matrix.astype(float).copy()
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

        diag_product = float(np.prod(np.diag(a)))
        return 1.0 if diag_product * sign >= 0 else -1.0


@dataclass(frozen=True)
class _RebasinResult:
    weights: dict[str, Any]
    quality: float | None
    blocks_aligned: int | None


@dataclass(frozen=True)
class _QuantizedResult:
    weight: np.ndarray
    scales: np.ndarray
    biases: np.ndarray | None
    scales_key: str
    biases_key: str
