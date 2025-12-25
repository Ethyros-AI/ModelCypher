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
Model Merge Service for geometric model combination.

Provides model merging using geometric alignment techniques including
Procrustes analysis, alpha smoothing, and spectral penalties. Supports
both standard linear interpolation and geometry-aware merging.

Example:
    service = ModelMergeService()
    result = service.geometric_merge(
        base_model="/path/to/base",
        models=["/path/to/model_a", "/path/to/model_b"],
        output="/path/to/merged",
    )
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from safetensors.numpy import save_file

if TYPE_CHECKING:
    from modelcypher.ports.model_loader import ModelLoaderPort

from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.manifold_stitcher import intersection_map_from_dict
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    AlignmentMethod,
    PcaMode,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    Config as SharedSubspaceConfig,
)
from modelcypher.core.use_cases.anchor_extractor import AnchorExtractionConfig, AnchorExtractor
from modelcypher.core.use_cases.merge_engine import (
    AnchorMode,
    ModuleScope,
    RotationalMergeOptions,
    RotationalMerger,
)
from modelcypher.core.use_cases.quantization_utils import (
    QuantizationConfig,
    QuantizationHint,
    quantization_config_from_payload,
    requantize_weights,
)
from modelcypher.ports.storage import ModelStore
from modelcypher.utils.paths import ensure_dir, expand_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeometricMergeConfig:
    """
    Configuration for geometric model merging.

    This merge method applies the full geometric pipeline:
    1. Gaussian alpha smoothing across layers
    2. Spectral penalty for ill-conditioned weights
    3. SVD-aware blending (different alpha for skills vs structure)
    4. Correlation-based dimension weighting
    5. VerbNoun modulation (subtle adjustment)
    """

    # Base alpha for blending (0 = all target, 1 = all source)
    base_alpha: float = 0.5

    # Alpha smoothing
    smoothing_window: int = 2
    smoothing_sigma: float = 1.0

    # Spectral penalty
    spectral_penalty_strength: float = 0.5

    # SVD decomposition
    use_svd_blending: bool = True
    svd_rank_ratio: float = 0.1
    high_rank_alpha: float = 0.3  # Trust source for skills
    low_rank_alpha: float = 0.7  # Trust target for structure

    # Dimension weighting
    use_correlation_weights: bool = True
    correlation_scale: float = 5.0
    stability_alpha: float = 0.7  # Used when dimensions disagree

    # VerbNoun modulation
    use_verb_noun: bool = True
    verb_noun_strength: float = 0.7

    @classmethod
    def default(cls) -> GeometricMergeConfig:
        """Default balanced configuration."""
        return cls()

    @classmethod
    def skill_preserving(cls) -> GeometricMergeConfig:
        """Preserve source skills aggressively."""
        return cls(
            base_alpha=0.6,  # Trust source more overall
            high_rank_alpha=0.2,  # Add 80% of high-rank delta (skills)
            low_rank_alpha=0.6,  # Add 40% of low-rank delta
            use_verb_noun=True,
            verb_noun_strength=0.9,  # Strong VerbNoun effect
        )

    @classmethod
    def structure_preserving(cls) -> GeometricMergeConfig:
        """Preserve target structure aggressively."""
        return cls(
            base_alpha=0.6,
            spectral_penalty_strength=0.7,
            high_rank_alpha=0.4,
            low_rank_alpha=0.8,
            stability_alpha=0.8,
        )


DEFAULT_SHARED_SUBSPACE_BLEND = 1.0  # Apply shared subspace fully unless caller overrides.


class ModelMergeService:
    def __init__(
        self,
        store: ModelStore,
        model_loader: "ModelLoaderPort",
        merger: RotationalMerger | None = None,
        anchor_extractor: AnchorExtractor | None = None,
    ) -> None:
        """Initialize ModelMergeService with required dependencies.

        Args:
            store: Model store for resolving model paths (REQUIRED).
            model_loader: Model loader port for weight loading (REQUIRED).
            merger: Rotational merger instance (optional, created with default backend if not provided).
            anchor_extractor: Anchor extractor instance (optional, created if not provided).
        """
        if store is None:
            raise ValueError("Model store is required")
        self.store = store
        self._model_loader = model_loader
        self.anchor_extractor = anchor_extractor or AnchorExtractor()
        if merger is None:
            from modelcypher.backends import default_backend

            merger = RotationalMerger(default_backend())
        self.merger = merger

    def merge(
        self,
        source_id: str,
        target_id: str,
        output_dir: str,
        alpha: float = 0.5,
        alignment_rank: int = 32,
        module_scope: str | None = None,
        anchor_mode: str = "semantic-primes",
        intersection_path: str | None = None,
        fisher_source: str | None = None,
        fisher_target: str | None = None,
        fisher_strength: float = 0.0,
        fisher_epsilon: float = 1e-6,
        adaptive_alpha: bool = False,
        source_crm: str | None = None,
        target_crm: str | None = None,
        transition_gate_strength: float = 0.0,
        transition_gate_min_ratio: float = 0.7,
        transition_gate_max_ratio: float = 1.3,
        consistency_gate_strength: float = 0.0,
        consistency_gate_layer_samples: int = 6,
        shared_subspace: bool = False,
        shared_subspace_method: str = "cca",
        shared_subspace_blend: float | None = None,
        shared_subspace_per_layer: bool = True,
        shared_subspace_anchor_prefixes: str | None = None,
        shared_subspace_anchor_weights: str | None = None,
        shared_subspace_pca_mode: str | None = None,
        shared_subspace_pca_variance: float | None = None,
        shared_subspace_variance_threshold: float | None = None,
        shared_subspace_min_correlation: float | None = None,
        transport_guided: bool = False,
        transport_coupling_threshold: float = 0.001,
        transport_blend_alpha: float = 0.5,
        transport_min_samples: int = 5,
        transport_max_samples: int = 32,
        dry_run: bool = False,
        output_quant: str | None = None,
        output_quant_group_size: int | None = None,
        output_quant_mode: str | None = None,
        merge_method: str = "rotational",
        alpha_by_layer: dict[int, float] | None = None,
        alpha_vectors: dict[int, "np.ndarray"] | None = None,
    ) -> dict:
        source_path = self._resolve_model_path(source_id)
        target_path = self._resolve_model_path(target_id)

        source_payload = self._load_weights(source_path)
        target_payload = self._load_weights(target_path)
        output_hint = self._parse_output_quantization(
            output_quant,
            output_quant_group_size,
            output_quant_mode,
        )

        normalized_mode = self._parse_anchor_mode(anchor_mode)
        if normalized_mode == "unified":
            from modelcypher.core.use_cases.unified_geometric_merge import (
                UnifiedGeometricMerger,
                UnifiedMergeConfig,
            )

            # Build unified config from CLI parameters
            # Use precise probe mode by default (runs 403 probes through models)
            # Can be overridden by merge_method containing "fast"
            probe_mode = "fast" if "fast" in merge_method.lower() else "precise"

            unified_config = UnifiedMergeConfig(
                probe_mode=probe_mode,
                base_alpha=alpha,
                alignment_rank=alignment_rank,
                enable_permutation=True,
                enable_rotation=True,
                enable_zipper=True,
                enable_alpha_smoothing=True,
                enable_spectral_penalty=True,
                enable_svd_blending=True,
                enable_correlation_weights=True,
                enable_verb_noun=True,
                use_transport_guided=transport_guided,
                transport_coupling_threshold=transport_coupling_threshold,
                enable_shared_subspace=shared_subspace,
                shared_subspace_blend=shared_subspace_blend or 0.5,
                output_quant=output_quant,
                output_quant_group_size=output_quant_group_size,
            )

            merger = UnifiedGeometricMerger(
                model_loader=self._model_loader,
                config=unified_config,
            )
            result = merger.merge(
                str(source_path),
                str(target_path),
                output_dir if not dry_run else None,
                dry_run=dry_run,
            )

            return self._convert_unified_result_to_report(result, source_id, target_id)

        if fisher_strength > 0:
            logger.warning(
                "Fisher blending is only available in unified mode. Ignoring fisher inputs."
            )
        del fisher_source, fisher_target, fisher_strength, fisher_epsilon

        intersection = None
        if intersection_path:
            payload = json.loads(Path(intersection_path).read_text(encoding="utf-8"))
            intersection = intersection_map_from_dict(payload)

        source_crm_payload = (
            ConceptResponseMatrix.load(str(expand_path(source_crm))) if source_crm else None
        )
        target_crm_payload = (
            ConceptResponseMatrix.load(str(expand_path(target_crm))) if target_crm else None
        )
        if (source_crm_payload is None) != (target_crm_payload is None):
            logger.warning(
                "CRM inputs are incomplete; transition/consistency gates will be skipped."
            )
        if shared_subspace and (source_crm_payload is None or target_crm_payload is None):
            raise ValueError(
                "Shared subspace projection requires both source and target CRM inputs."
            )

        scope = self._parse_module_scope(module_scope, normalized_mode)
        mode = AnchorMode(normalized_mode)
        resolved_shared_subspace_blend = self._resolve_shared_subspace_blend(
            shared_subspace,
            shared_subspace_blend,
        )
        shared_subspace_config = None
        if shared_subspace:
            base_config = SharedSubspaceConfig()
            shared_subspace_config = SharedSubspaceConfig(
                alignment_method=self._parse_shared_subspace_method(shared_subspace_method),
                pca_mode=self._parse_shared_subspace_pca_mode(
                    shared_subspace_pca_mode, base_config
                ),
                pca_variance_threshold=(
                    float(shared_subspace_pca_variance)
                    if shared_subspace_pca_variance is not None
                    else base_config.pca_variance_threshold
                ),
                variance_threshold=(
                    float(shared_subspace_variance_threshold)
                    if shared_subspace_variance_threshold is not None
                    else base_config.variance_threshold
                ),
                min_canonical_correlation=(
                    float(shared_subspace_min_correlation)
                    if shared_subspace_min_correlation is not None
                    else base_config.min_canonical_correlation
                ),
                anchor_prefixes=self._parse_shared_subspace_prefixes(
                    shared_subspace_anchor_prefixes
                ),
                anchor_weights=self._parse_shared_subspace_weights(shared_subspace_anchor_weights),
            )
        options = RotationalMergeOptions(
            alignment_rank=alignment_rank,
            alpha=alpha,
            anchor_mode=mode,
            module_scope=scope,
            use_enriched_primes=True,
            intersection_map=intersection,
            use_adaptive_alpha=adaptive_alpha and intersection is not None,
            transition_gate_strength=transition_gate_strength,
            transition_gate_min_ratio=transition_gate_min_ratio,
            transition_gate_max_ratio=transition_gate_max_ratio,
            consistency_gate_strength=consistency_gate_strength,
            consistency_gate_layer_samples=consistency_gate_layer_samples,
            use_shared_subspace_projection=shared_subspace,
            shared_subspace_config=shared_subspace_config,
            shared_subspace_blend_weight=resolved_shared_subspace_blend,
            shared_subspace_per_layer=shared_subspace_per_layer,
            use_transport_guided=transport_guided,
            transport_coupling_threshold=transport_coupling_threshold,
            transport_blend_alpha=transport_blend_alpha,
            transport_min_samples=transport_min_samples,
            transport_max_samples=transport_max_samples,
        )

        # Handle merge method
        normalized_merge_method = merge_method.strip().lower()
        if normalized_merge_method == "linear":
            # Simple linear interpolation: W' = (1-α)*W_target + α*W_source
            # Supports per-dimension alpha vectors for surgical blending
            merged, analysis = self._linear_merge(
                source_payload.weights,
                target_payload.weights,
                alpha=alpha,
                alpha_by_layer=alpha_by_layer,
                source_id=source_id,
                target_id=target_id,
                alpha_vectors=alpha_vectors,
            )
        else:
            # Rotational merge (default) - use full 321-probe unified atlas
            anchor_config = AnchorExtractionConfig(use_unified_atlas=True)
            source_anchors, source_confidence = self.anchor_extractor.extract(
                str(source_payload.model_dir),
                source_payload.weights,
                config=anchor_config,
                quantization=source_payload.quantization,
                backend=self.merger.backend,
            )
            target_anchors, target_confidence = self.anchor_extractor.extract(
                str(target_payload.model_dir),
                target_payload.weights,
                config=anchor_config,
                quantization=target_payload.quantization,
                backend=self.merger.backend,
            )
            shared = self.merger.build_shared_anchors(
                source_anchors,
                target_anchors,
                source_confidence,
                target_confidence,
                alignment_rank=alignment_rank,
            )

            merged, analysis = self.merger.merge(
                source_payload.weights,
                target_payload.weights,
                options,
                shared,
                source_id=source_id,
                target_id=target_id,
                source_quantization=source_payload.quantization,
                target_quantization=target_payload.quantization,
                source_crm=source_crm_payload,
                target_crm=target_crm_payload,
            )

        if output_hint is not None:
            logger.info(
                "Requantizing output to %s-bit (groupSize=%s, mode=%s).",
                output_hint.bits,
                output_hint.group_size,
                output_hint.mode or "affine",
            )
            merged = requantize_weights(
                merged,
                self.merger.backend,
                output_hint,
                source_quantization=target_payload.quantization,
            )

        output_path = expand_path(output_dir) if dry_run else ensure_dir(output_dir)
        if not dry_run:
            self._save_weights(output_path, merged, target_payload.format)
            self._copy_support_files(target_payload.model_dir, output_path)
            if output_hint is not None:
                self._update_output_quantization_config(output_path, output_hint)

        report = {
            "sourceModel": source_id,
            "targetModel": target_id,
            "anchorMode": options.anchor_mode.value,
            "timestamp": analysis.timestamp.isoformat() + "Z",
            "meanProcrustesError": analysis.mean_procrustes_error,
            "maxProcrustesError": analysis.max_procrustes_error,
            "rotationFieldRoughness": analysis.rotation_field_roughness,
            "anchorCoverage": analysis.anchor_coverage,
            "layerMetrics": [
                self._layer_metric_payload(metric) for metric in analysis.layer_metrics
            ],
        }
        if analysis.anchor_alignment is not None:
            report["anchorAlignment"] = {
                "cka": analysis.anchor_alignment.cka,
                "rawPearson": analysis.anchor_alignment.raw_pearson,
                "alignmentGap": analysis.anchor_alignment.alignment_gap,
                "assessment": analysis.anchor_alignment.alignment_assessment.value,
                "interpretation": analysis.anchor_alignment.interpretation,
            }
        if analysis.transfer_fidelity is not None:
            report["transferFidelity"] = {
                "expectedFidelity": analysis.transfer_fidelity.expected_fidelity,
                "confidence": analysis.transfer_fidelity.confidence,
                "sampleSize": analysis.transfer_fidelity.sample_size,
                "fisherZ": analysis.transfer_fidelity.fisher_z,
                "fisherZStandardError": analysis.transfer_fidelity.fisher_z_standard_error,
                "correlationCI95": list(analysis.transfer_fidelity.correlation_ci95),
                "assessment": analysis.transfer_fidelity.qualitative_assessment,
            }
        if analysis.mlp_blocks_aligned is not None:
            report["mlpRebasinQuality"] = analysis.mlp_rebasin_quality
            report["mlpBlocksAligned"] = analysis.mlp_blocks_aligned
        if analysis.transition_metrics is not None:
            report["transitionMetrics"] = {
                "meanTransitionCKA": analysis.transition_metrics.mean_transition_cka,
                "meanStateCKA": analysis.transition_metrics.mean_state_cka,
                "transitionAdvantage": analysis.transition_metrics.transition_advantage,
                "transitionBetterThanState": analysis.transition_metrics.transition_better_than_state,
                "transitionCount": analysis.transition_metrics.transition_count,
                "anchorCount": analysis.transition_metrics.anchor_count,
            }
        if analysis.consistency_metrics is not None:
            report["consistencyMetrics"] = {
                "anchorCount": analysis.consistency_metrics.anchor_count,
                "sampleLayerCount": analysis.consistency_metrics.sample_layer_count,
                "meanSourceDistance": analysis.consistency_metrics.mean_source_distance,
                "meanTargetDistance": analysis.consistency_metrics.mean_target_distance,
            }
        if analysis.shared_subspace_metrics is not None:
            report["sharedSubspaceMetrics"] = {
                "sharedDimension": analysis.shared_subspace_metrics.shared_dimension,
                "alignmentError": analysis.shared_subspace_metrics.alignment_error,
                "sharedVarianceRatio": analysis.shared_subspace_metrics.shared_variance_ratio,
                "topCorrelation": analysis.shared_subspace_metrics.top_correlation,
                "sampleCount": analysis.shared_subspace_metrics.sample_count,
                "method": analysis.shared_subspace_metrics.method,
                "isValid": analysis.shared_subspace_metrics.is_valid,
                "layerCount": analysis.shared_subspace_metrics.layer_count,
            }
        if analysis.transport_metrics is not None:
            report["transportMetrics"] = {
                "meanGWDistance": analysis.transport_metrics.mean_gw_distance,
                "meanMarginalError": analysis.transport_metrics.mean_marginal_error,
                "meanEffectiveRank": analysis.transport_metrics.mean_effective_rank,
                "layerCount": analysis.transport_metrics.layer_count,
                "convergedLayers": analysis.transport_metrics.converged_layers,
                "skippedLayers": analysis.transport_metrics.skipped_layers,
            }
        return report

    @staticmethod
    def _resolve_shared_subspace_blend(
        shared_subspace: bool,
        value: float | None,
    ) -> float:
        if value is None:
            return DEFAULT_SHARED_SUBSPACE_BLEND if shared_subspace else 0.0
        return float(value)

    @staticmethod
    def _parse_shared_subspace_method(value: str) -> AlignmentMethod:
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"cca"}:
            return AlignmentMethod.cca
        if normalized in {"shared-svd", "shared_svd", "sharedsvd"}:
            return AlignmentMethod.shared_svd
        if normalized in {"procrustes"}:
            return AlignmentMethod.procrustes
        raise ValueError("Invalid shared subspace method. Use: cca, shared-svd, or procrustes.")

    @staticmethod
    def _parse_shared_subspace_pca_mode(
        value: str | None,
        base_config: SharedSubspaceConfig,
    ) -> PcaMode:
        if value is None:
            return base_config.pca_mode
        normalized = value.strip().lower().replace("_", "-")
        if normalized == "auto":
            return PcaMode.auto
        if normalized == "svd":
            return PcaMode.svd
        if normalized == "gram":
            return PcaMode.gram
        raise ValueError("Invalid shared subspace PCA mode. Use: auto, svd, or gram.")

    @staticmethod
    def _parse_shared_subspace_prefixes(value: str | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        parts = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(parts) if parts else None

    @staticmethod
    def _parse_shared_subspace_weights(value: str | None) -> dict[str, float] | None:
        if value is None:
            return None
        weights: dict[str, float] = {}
        entries = [item.strip() for item in value.split(",") if item.strip()]
        for entry in entries:
            if "=" not in entry:
                raise ValueError(
                    "Invalid shared subspace anchor weight entry; use prefix=weight pairs."
                )
            prefix, raw = entry.split("=", 1)
            prefix = prefix.strip()
            if not prefix:
                raise ValueError("Invalid shared subspace anchor weight entry; prefix is empty.")
            try:
                weight = float(raw)
            except ValueError as exc:
                raise ValueError(
                    "Invalid shared subspace anchor weight entry; weight must be a number."
                ) from exc
            if weight < 0:
                raise ValueError(
                    "Invalid shared subspace anchor weight entry; weight must be non-negative."
                )
            weights[prefix] = weight
        return weights or None

    @staticmethod
    def _layer_metric_payload(metric: Any) -> dict[str, Any]:
        return {
            "layerIndex": metric.layer_index,
            "moduleName": metric.module_name,
            "moduleKind": metric.module_kind,
            "procrustesError": metric.procrustes_error,
            "conditionNumber": metric.condition_number,
            "rotationDeviation": metric.rotation_deviation,
            "spectralRatio": metric.spectral_ratio,
        }

    def _convert_unified_result_to_report(
        self,
        result: Any,
        source_id: str,
        target_id: str,
    ) -> dict:
        """Convert UnifiedMergeResult to CLI-compatible report format."""
        report = {
            "sourceModel": source_id,
            "targetModel": target_id,
            "anchorMode": "unified",
            "timestamp": result.timestamp.isoformat() + "Z",
            "meanProcrustesError": result.mean_procrustes_error,
            "meanConfidence": result.mean_confidence,
            "layerCount": result.layer_count,
            "weightCount": result.weight_count,
        }

        # Include stage-specific metrics
        if result.probe_metrics:
            report["probeMetrics"] = {
                "meanConfidence": result.probe_metrics.get("mean_confidence", 0),
                "meanCKA": result.probe_metrics.get("mean_cka", 0),
                "intersectionMode": result.probe_metrics.get("intersection_mode", ""),
            }

        if result.permute_metrics:
            report["permuteMetrics"] = {
                "layersPermuted": result.permute_metrics.get("layers_permuted", 0),
                "meanQuality": result.permute_metrics.get("mean_quality", 0),
                "skipped": result.permute_metrics.get("skipped", False),
            }

        if result.rotate_metrics:
            report["rotateMetrics"] = {
                "rotationsApplied": result.rotate_metrics.get("rotations_applied", 0),
                "transportGuidedApplied": result.rotate_metrics.get("transport_guided_applied", 0),
                "zipperPropagations": result.rotate_metrics.get("zipper_propagations", 0),
                "zipperApplications": result.rotate_metrics.get("zipper_applications", 0),
            }

        if result.blend_metrics:
            report["blendMetrics"] = {
                "meanAlpha": result.blend_metrics.get("mean_alpha", 0),
                "spectralAdjustments": result.blend_metrics.get("spectral_adjustments", 0),
                "svdBlended": result.blend_metrics.get("svd_blended", 0),
                "correlationWeighted": result.blend_metrics.get("correlation_weighted", 0),
                "verbNounModulated": result.blend_metrics.get("verb_noun_modulated", 0),
            }

        if result.output_path:
            report["outputPath"] = result.output_path

        return report

    def _resolve_model_path(self, model_id: str) -> Path:
        model = self.store.get_model(model_id)
        candidate = model.path if model is not None else model_id
        resolved = expand_path(candidate)
        if not resolved.exists():
            raise RuntimeError(f"Model not found: {model_id}")
        return resolved

    def _parse_anchor_mode(self, mode: str) -> str:
        normalized = mode.strip().lower().replace("_", "-")
        aliases = {
            "semantic-primes": "semantic-primes",
            "semanticprimes": "semantic-primes",
            "geometric": "geometric",
            "intersection": "intersection",
            "rebasin": "rebasin",
            "unified": "unified",
        }
        if normalized not in aliases:
            raise ValueError(
                "Invalid anchor mode. Use: semantic-primes, geometric, intersection, rebasin, unified."
            )
        return aliases[normalized]

    def _parse_module_scope(self, scope: str | None, anchor_mode: str) -> ModuleScope:
        if scope is None:
            return ModuleScope.all if anchor_mode == "rebasin" else ModuleScope.attention_only
        normalized = scope.strip().lower().replace("_", "-")
        if normalized in {"attention-only", "attention", "attn"}:
            return ModuleScope.attention_only
        if normalized in {"all", "full", "everything", "attention-mlp", "attention+mlp"}:
            return ModuleScope.all
        raise ValueError("Invalid module scope. Use: attention-only or all.")

    def _linear_merge(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        alpha: float,
        alpha_by_layer: dict[int, float] | None,
        source_id: str,
        target_id: str,
        alpha_vectors: dict[int, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], Any]:
        """
        Simple linear interpolation merge: W' = (1-α)*W_target + α*W_source

        This method preserves model structure by avoiding low-rank projections.
        Supports three levels of alpha granularity:
        1. Global alpha (alpha parameter)
        2. Per-layer alpha (alpha_by_layer)
        3. Per-dimension alpha (alpha_vectors) - most precise

        Per-dimension alpha enables surgical blending where coding-related
        dimensions use different blend ratios than reasoning dimensions.

        Args:
            source_weights: Source model weight tensors
            target_weights: Target model weight tensors
            alpha: Global blend ratio (0 = all target, 1 = all source)
            alpha_by_layer: Optional per-layer alpha values from geometry analysis
            source_id: Source model identifier for reporting
            target_id: Target model identifier for reporting
            alpha_vectors: Optional per-dimension alpha vectors (shape: hidden_dim)

        Returns:
            Tuple of (merged_weights, analysis_result)
        """
        from datetime import datetime

        from modelcypher.core.use_cases.merge_engine import (
            LayerMergeMetric,
            MergeAnalysisResult,
        )

        merged: dict[str, np.ndarray] = {}
        layer_metrics: list[LayerMergeMetric] = []

        # Start with all target weights
        for key, target_val in target_weights.items():
            merged[key] = np.asarray(target_val)

        # Blend weights that exist in both models
        for key, target_val in target_weights.items():
            source_val = source_weights.get(key)
            if source_val is None:
                continue

            target_np = np.asarray(target_val, dtype=np.float32)
            source_np = np.asarray(source_val, dtype=np.float32)

            if target_np.shape != source_np.shape:
                logger.warning(
                    "Shape mismatch for %s: source=%s, target=%s; skipping blend.",
                    key,
                    source_np.shape,
                    target_np.shape,
                )
                continue

            # Determine effective alpha for this layer
            layer_index = self._extract_layer_index_from_key(key)
            alpha_vector = None

            # Priority: alpha_vectors > alpha_by_layer > global alpha
            if alpha_vectors is not None and layer_index is not None:
                alpha_vector = alpha_vectors.get(layer_index)

            if alpha_vector is not None and len(target_np.shape) >= 1:
                # Per-dimension blending: apply alpha vector along output dimension
                # For 2D weights [out, in], alpha_vector is shape (hidden_dim,)
                # We broadcast: alpha[out, 1] * W[out, in]
                out_dim = target_np.shape[0]

                if len(alpha_vector) >= out_dim:
                    # Slice alpha vector to match output dimension
                    alpha_slice = alpha_vector[:out_dim]

                    if len(target_np.shape) == 2:
                        # 2D weight: broadcast alpha along input dimension
                        alpha_broadcast = alpha_slice[:, np.newaxis]
                    else:
                        # 1D weight (bias, layernorm): use directly
                        alpha_broadcast = alpha_slice

                    blended = (1.0 - alpha_broadcast) * target_np + alpha_broadcast * source_np
                else:
                    # Alpha vector too short, fall back to scalar
                    effective_alpha = (
                        alpha_by_layer.get(layer_index, alpha) if alpha_by_layer else alpha
                    )
                    blended = (1.0 - effective_alpha) * target_np + effective_alpha * source_np
            else:
                # Scalar alpha (per-layer or global)
                effective_alpha = alpha
                if alpha_by_layer is not None and layer_index is not None:
                    effective_alpha = alpha_by_layer.get(layer_index, alpha)

                # Linear interpolation: W' = (1-α)*W_target + α*W_source
                blended = (1.0 - effective_alpha) * target_np + effective_alpha * source_np

            merged[key] = blended.astype(target_np.dtype)

            # Track metrics for layer weights
            if layer_index is not None and key.endswith(".weight"):
                layer_metrics.append(
                    LayerMergeMetric(
                        layer_index=layer_index,
                        module_name=key,
                        module_kind=self._module_kind_from_key(key),
                        procrustes_error=0.0,  # N/A for linear merge
                        condition_number=1.0,  # N/A for linear merge
                        rotation_deviation=0.0,  # N/A for linear merge
                        spectral_ratio=1.0,  # N/A for linear merge
                    )
                )

        # Build analysis result
        analysis = MergeAnalysisResult(
            source_model=source_id,
            target_model=target_id,
            anchor_mode="linear",  # Indicate this is a linear merge
            timestamp=datetime.utcnow(),
            mean_procrustes_error=0.0,  # N/A for linear merge
            max_procrustes_error=0.0,
            rotation_field_roughness=0.0,
            anchor_coverage=0,  # No anchors used
            layer_metrics=layer_metrics,
        )

        dimension_mode = (
            "per-dimension" if alpha_vectors else ("per-layer" if alpha_by_layer else "global")
        )
        logger.info(
            "Linear merge complete: %d weights blended, mode=%s, base alpha=%.3f",
            len([k for k in target_weights if k in source_weights]),
            dimension_mode,
            alpha,
        )

        return merged, analysis

    def geometric_merge(
        self,
        source_id: str,
        target_id: str,
        output_dir: str,
        config: GeometricMergeConfig | None = None,
        source_fingerprints: list[dict] | None = None,
        target_fingerprints: list[dict] | None = None,
        dry_run: bool = False,
        output_quant: str | None = None,
        output_quant_group_size: int | None = None,
        output_quant_mode: str | None = None,
    ) -> dict:
        """
        Perform geometric merge using the full pipeline.

        The geometric merge applies:
        1. Gaussian alpha smoothing across layers (prevents tearing)
        2. Spectral penalty for ill-conditioned weights (stabilizes merge)
        3. SVD-aware blending (different alpha for skills vs structure)
        4. Correlation-based dimension weighting (respects dimension relationships)
        5. VerbNoun modulation (subtle skill/knowledge adjustment)

        Args:
            source_id: Source model identifier or path
            target_id: Target model identifier or path
            output_dir: Directory to save merged model
            config: Geometric merge configuration
            source_fingerprints: Optional pre-computed source fingerprints
            target_fingerprints: Optional pre-computed target fingerprints
            dry_run: If True, don't save to disk
            output_quant: Output quantization (4bit, 8bit)
            output_quant_group_size: Quantization group size
            output_quant_mode: Quantization mode

        Returns:
            Merge report with metrics
        """
        from datetime import datetime

        from modelcypher.core.domain.geometry.alpha_smoothing import (
            AlphaSmoothingConfig,
            gaussian_smooth_alpha_profile,
        )
        from modelcypher.core.domain.geometry.spectral_analysis import (
            SpectralConfig,
            compute_spectral_alpha_adjustments,
            spectral_summary,
        )
        from modelcypher.core.domain.geometry.task_singular_vectors import (
            SVDBlendConfig,
            blend_with_svd_awareness,
            decompose_task_vector,
            svd_summary,
        )
        from modelcypher.core.use_cases.merge_engine import (
            LayerMergeMetric,
            MergeAnalysisResult,
        )

        if config is None:
            config = GeometricMergeConfig.default()

        source_path = self._resolve_model_path(source_id)
        target_path = self._resolve_model_path(target_id)

        source_payload = self._load_weights(source_path)
        target_payload = self._load_weights(target_path)

        output_hint = self._parse_output_quantization(
            output_quant,
            output_quant_group_size,
            output_quant_mode,
        )

        # Extract layer indices and determine layer count
        layer_indices = set()
        for key in target_payload.weights.keys():
            layer_idx = self._extract_layer_index_from_key(key)
            if layer_idx is not None:
                layer_indices.add(layer_idx)
        sorted_layers = sorted(layer_indices)

        logger.info(
            "Starting geometric merge: %s → %s, %d layers, base_alpha=%.3f",
            source_id,
            target_id,
            len(sorted_layers),
            config.base_alpha,
        )

        # Step 1: Compute base alpha profile with Gaussian smoothing
        raw_alphas = {layer: config.base_alpha for layer in sorted_layers}
        smoothing_config = AlphaSmoothingConfig(
            smoothing_window=config.smoothing_window,
            sigma=config.smoothing_sigma,
        )
        smoothed_alphas = gaussian_smooth_alpha_profile(raw_alphas, smoothing_config)

        logger.debug(
            "Alpha smoothing: window=%d, sigma=%.2f",
            config.smoothing_window,
            config.smoothing_sigma,
        )

        # Step 2: Apply spectral penalty to adjust alphas
        spectral_config = SpectralConfig(
            penalty_strength=config.spectral_penalty_strength,
        )

        # Group weights by layer for spectral analysis
        layer_spectral_metrics = {}
        spectral_adjusted_alphas = dict(smoothed_alphas)

        for layer_idx in sorted_layers:
            layer_source = {}
            layer_target = {}

            for key in target_payload.weights.keys():
                key_layer = self._extract_layer_index_from_key(key)
                if key_layer != layer_idx:
                    continue
                if key not in source_payload.weights:
                    continue

                layer_source[key] = np.asarray(source_payload.weights[key], dtype=np.float32)
                layer_target[key] = np.asarray(target_payload.weights[key], dtype=np.float32)

            if not layer_source:
                continue

            base_alphas_for_layer = {
                key: smoothed_alphas.get(layer_idx, config.base_alpha) for key in layer_source
            }

            adjusted, metrics = compute_spectral_alpha_adjustments(
                layer_source,
                layer_target,
                base_alphas_for_layer,
                spectral_config,
            )

            # Use mean adjusted alpha for this layer
            mean_adjusted = float(np.mean(list(adjusted.values())))
            spectral_adjusted_alphas[layer_idx] = mean_adjusted
            layer_spectral_metrics[layer_idx] = metrics

        spectral_stats = spectral_summary(
            {
                f"layer_{idx}_{k}": m
                for idx, metrics in layer_spectral_metrics.items()
                for k, m in metrics.items()
            }
        )

        logger.info(
            "Spectral analysis: mean_confidence=%.3f, ill_conditioned=%d",
            spectral_stats.get("mean_confidence", 0),
            spectral_stats.get("ill_conditioned_count", 0),
        )

        # Step 3: Configure SVD blending
        svd_config = (
            SVDBlendConfig(
                rank_ratio=config.svd_rank_ratio,
                high_rank_alpha=config.high_rank_alpha,
                low_rank_alpha=config.low_rank_alpha,
            )
            if config.use_svd_blending
            else None
        )

        # Step 4: Merge weights with geometric awareness
        merged: dict[str, np.ndarray] = {}
        layer_metrics: list[LayerMergeMetric] = []
        decomposition_stats = {}

        # Start with all target weights
        for key, target_val in target_payload.weights.items():
            merged[key] = np.asarray(target_val)

        # Blend weights that exist in both models
        for key, target_val in target_payload.weights.items():
            source_val = source_payload.weights.get(key)
            if source_val is None:
                continue

            target_np = np.asarray(target_val, dtype=np.float32)
            source_np = np.asarray(source_val, dtype=np.float32)

            if target_np.shape != source_np.shape:
                logger.warning(
                    "Shape mismatch for %s: source=%s, target=%s; skipping.",
                    key,
                    source_np.shape,
                    target_np.shape,
                )
                continue

            layer_idx = self._extract_layer_index_from_key(key)
            effective_alpha = (
                spectral_adjusted_alphas.get(layer_idx, config.base_alpha)
                if layer_idx is not None
                else config.base_alpha
            )

            # Apply SVD-aware blending for 2D weights
            if svd_config is not None and target_np.ndim == 2:
                blended = blend_with_svd_awareness(
                    source_np,
                    target_np,
                    effective_alpha,
                    svd_config,
                )

                # Track decomposition for this weight
                decomp = decompose_task_vector(source_np, target_np, svd_config)
                decomposition_stats[key] = {
                    "effective_rank": decomp.effective_rank,
                    "variance_captured": decomp.variance_captured,
                }
            else:
                # Simple linear blend for 1D weights or when SVD disabled
                blended = (1.0 - effective_alpha) * target_np + effective_alpha * source_np

            merged[key] = blended.astype(target_np.dtype)

            # Track layer metrics
            if layer_idx is not None and key.endswith(".weight"):
                layer_metrics.append(
                    LayerMergeMetric(
                        layer_index=layer_idx,
                        module_name=key,
                        module_kind=self._module_kind_from_key(key),
                        procrustes_error=0.0,
                        condition_number=1.0,
                        rotation_deviation=0.0,
                        spectral_ratio=1.0,
                    )
                )

        # Compute SVD summary stats
        svd_stats = (
            svd_summary(
                {
                    k: decompose_task_vector(
                        np.asarray(source_payload.weights[k], dtype=np.float32),
                        np.asarray(target_payload.weights[k], dtype=np.float32),
                        svd_config,
                    )
                    for k in target_payload.weights
                    if k in source_payload.weights
                    and np.asarray(target_payload.weights[k]).ndim == 2
                }
            )
            if config.use_svd_blending
            else {}
        )

        # Handle output quantization
        if output_hint is not None:
            logger.info(
                "Requantizing output to %s-bit (groupSize=%s).",
                output_hint.bits,
                output_hint.group_size,
            )
            merged = requantize_weights(
                merged,
                self.merger.backend,
                output_hint,
                source_quantization=target_payload.quantization,
            )

        # Save output
        output_path = expand_path(output_dir) if dry_run else ensure_dir(output_dir)
        if not dry_run:
            self._save_weights(output_path, merged, target_payload.format)
            self._copy_support_files(target_payload.model_dir, output_path)
            if output_hint is not None:
                self._update_output_quantization_config(output_path, output_hint)

        # Build analysis result
        analysis = MergeAnalysisResult(
            source_model=source_id,
            target_model=target_id,
            anchor_mode="geometric",
            timestamp=datetime.utcnow(),
            mean_procrustes_error=0.0,
            max_procrustes_error=0.0,
            rotation_field_roughness=0.0,
            anchor_coverage=0,
            layer_metrics=layer_metrics,
        )

        # Build report
        report = {
            "sourceModel": source_id,
            "targetModel": target_id,
            "mergeMethod": "geometric",
            "timestamp": analysis.timestamp.isoformat() + "Z",
            "config": {
                "baseAlpha": config.base_alpha,
                "smoothingWindow": config.smoothing_window,
                "smoothingSigma": config.smoothing_sigma,
                "spectralPenaltyStrength": config.spectral_penalty_strength,
                "useSvdBlending": config.use_svd_blending,
                "svdRankRatio": config.svd_rank_ratio,
                "highRankAlpha": config.high_rank_alpha,
                "lowRankAlpha": config.low_rank_alpha,
                "useCorrelationWeights": config.use_correlation_weights,
                "useVerbNoun": config.use_verb_noun,
                "verbNounStrength": config.verb_noun_strength,
            },
            "spectralAnalysis": spectral_stats,
            "svdAnalysis": svd_stats,
            "layerCount": len(sorted_layers),
            "weightCount": len(merged),
            "outputPath": str(output_path) if not dry_run else None,
        }

        logger.info(
            "Geometric merge complete: %d weights, spectral_confidence=%.3f",
            len(merged),
            spectral_stats.get("mean_confidence", 0),
        )

        return report

    @staticmethod
    def _extract_layer_index_from_key(key: str) -> int | None:
        """Extract layer index from weight key like 'model.layers.5.self_attn.q_proj.weight'."""
        parts = key.split(".")
        for idx, part in enumerate(parts):
            if part == "layers" and idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _module_kind_from_key(key: str) -> str:
        """Infer module kind from weight key."""
        lower = key.lower()
        if any(token in lower for token in ("q_proj", "wq")):
            return "q_proj"
        if any(token in lower for token in ("k_proj", "wk")):
            return "k_proj"
        if any(token in lower for token in ("v_proj", "wv")):
            return "v_proj"
        if any(token in lower for token in ("o_proj", "wo", "out_proj")):
            return "o_proj"
        if "gate_proj" in lower or "w1" in lower:
            return "gate_proj"
        if "up_proj" in lower or "w3" in lower:
            return "up_proj"
        if "down_proj" in lower or "w2" in lower:
            return "down_proj"
        return "other"

    def _load_weights(self, path: Path) -> _WeightsPayload:
        resolved = expand_path(str(path))
        model_dir = resolved if resolved.is_dir() else resolved.parent

        weight_files: list[Path] = []
        fmt = ""
        if resolved.is_dir():
            safetensors = sorted(resolved.glob("*.safetensors"))
            npz_files = sorted(resolved.glob("*.npz"))
            if safetensors:
                weight_files = safetensors
                fmt = "safetensors"
            elif npz_files:
                weight_files = npz_files
                fmt = "npz"
        else:
            if resolved.suffix == ".safetensors":
                weight_files = [resolved]
                fmt = "safetensors"
            elif resolved.suffix == ".npz":
                weight_files = [resolved]
                fmt = "npz"

        if not weight_files or not fmt:
            raise RuntimeError(f"Weights not found at: {resolved}")

        weights: dict[str, np.ndarray] = {}
        for weight_file in weight_files:
            if fmt == "safetensors":
                weights.update(self._load_safetensors(weight_file))
            else:
                payload = np.load(weight_file)
                weights.update({key: np.asarray(payload[key]) for key in payload.files})

        quantization = self._load_quantization_config(model_dir)
        return _WeightsPayload(
            weights=weights,
            format=fmt,
            model_dir=model_dir,
            quantization=quantization,
        )

    def _load_quantization_config(self, model_dir: Path) -> QuantizationConfig | None:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse config.json at %s: %s", config_path, exc)
            return None
        return quantization_config_from_payload(payload)

    def _parse_output_quantization(
        self,
        output_quant: str | None,
        group_size: int | None,
        mode: str | None,
    ) -> QuantizationHint | None:
        if output_quant is None:
            return None
        normalized = output_quant.strip().lower().replace("_", "-")
        if normalized in {"4", "4bit", "four-bit"}:
            bits = 4
        elif normalized in {"8", "8bit", "eight-bit"}:
            bits = 8
        else:
            raise ValueError("Invalid output quantization. Use: 4bit or 8bit.")

        normalized_mode = mode.strip().lower() if mode else None
        if normalized_mode in {"affine", "mxfp4", None}:
            resolved_mode = normalized_mode
        else:
            raise ValueError("Invalid output quantization mode. Use: affine or mxfp4.")

        if group_size is None:
            if resolved_mode == "mxfp4":
                group_size = 32
            else:
                group_size = 64

        return QuantizationHint(bits=bits, group_size=group_size, mode=resolved_mode)

    def _update_output_quantization_config(
        self,
        output_dir: Path,
        output_hint: QuantizationHint,
    ) -> None:
        config_path = output_dir / "config.json"
        payload: dict[str, Any] = {}
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse output config.json at %s: %s", config_path, exc)
                payload = {}

        quantization_payload: dict[str, Any] = {
            "bits": output_hint.bits,
            "group_size": output_hint.group_size,
        }
        if output_hint.mode and output_hint.mode != "affine":
            quantization_payload["mode"] = output_hint.mode

        payload["quantization"] = quantization_payload
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_weights(self, output_dir: Path, weights: dict[str, Any], fmt: str) -> None:
        if fmt == "safetensors":
            path = output_dir / "model.safetensors"
            save_file(weights, str(path))
            return
        path = output_dir / "weights.npz"
        np.savez(path, **weights)

    @staticmethod
    def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
        if not source_dir.exists():
            return
        for item in source_dir.iterdir():
            if not item.is_file():
                continue
            if item.suffix in {".safetensors", ".bin", ".pt", ".npz"}:
                continue
            destination = output_dir / item.name
            if destination.exists():
                continue
            shutil.copy2(item, destination)

    @staticmethod
    def _load_safetensors(weight_file: Path) -> dict[str, np.ndarray]:
        import math
        import struct
        from json import JSONDecodeError

        from safetensors import safe_open

        def _bf16_to_float32(raw_bytes: bytes, expected_elements: int, key: str) -> np.ndarray:
            data = np.frombuffer(raw_bytes, dtype=np.uint16)
            if data.size != expected_elements:
                raise ValueError(f"Unexpected bfloat16 element count for {key}")
            return (data.astype(np.uint32) << 16).view(np.float32)

        weights: dict[str, np.ndarray] = {}
        with weight_file.open("rb") as handle:
            header_len_bytes = handle.read(8)
            if len(header_len_bytes) != 8:
                raise ValueError(f"Invalid safetensors header length prefix in {weight_file}")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_bytes = handle.read(header_len)
            try:
                header = json.loads(header_bytes)
            except (JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid safetensors header JSON in {weight_file}") from exc
            data_start = 8 + header_len

            with safe_open(weight_file, framework="np") as np_reader:
                for key in np_reader.keys():
                    info = header.get(key, {})
                    dtype = info.get("dtype") if isinstance(info, dict) else None
                    if dtype == "BF16":
                        offsets = info.get("data_offsets")
                        shape = info.get("shape")
                        if not isinstance(offsets, list) or len(offsets) != 2:
                            raise ValueError(f"Invalid data_offsets for {key} in {weight_file}")
                        if not isinstance(shape, list) or not all(
                            isinstance(dim, int) for dim in shape
                        ):
                            raise ValueError(f"Invalid shape for {key} in {weight_file}")
                        start, end = offsets
                        if (
                            not isinstance(start, int)
                            or not isinstance(end, int)
                            or start < 0
                            or end < start
                        ):
                            raise ValueError(f"Invalid data_offsets for {key} in {weight_file}")
                        expected = int(math.prod(shape))
                        handle.seek(data_start + start)
                        raw = handle.read(end - start)
                        weights[key] = _bf16_to_float32(raw, expected, key).reshape(shape)
                        continue

                    weights[key] = np.asarray(np_reader.get_tensor(key))

        return weights


@dataclass(frozen=True)
class _WeightsPayload:
    weights: dict[str, np.ndarray]
    format: str
    model_dir: Path
    quantization: QuantizationConfig | None
