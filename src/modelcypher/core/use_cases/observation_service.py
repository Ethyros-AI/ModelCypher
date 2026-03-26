from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer
from modelcypher.core.use_cases.chain_analysis_service import ChainAnalysisService
from modelcypher.core.use_cases.geodesic_trajectory_service import GeodesicTrajectoryService
from modelcypher.core.use_cases.geometry_analysis_service import GeometryAnalysisService

if TYPE_CHECKING:
    from collections.abc import Callable

    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

    SublayerCollector = Callable[[Any, Any, list[str], int, Backend], list[dict[str, Any]]]


logger = logging.getLogger(__name__)


DEFAULT_ANALYSIS_SPACES = ("hidden", "embedding")
OPTIONAL_ANALYSIS_SPACES = ("intermediate", "q", "k", "v", "gate")
SUPPORTED_ANALYSIS_SPACES = DEFAULT_ANALYSIS_SPACES + OPTIONAL_ANALYSIS_SPACES
DEFAULT_MAX_TOKENS = 128
OBSERVATION_BUNDLE_VERSION = "mc.analyze.bundle.v1"
PROMPT_FAMILY_MANIFEST_VERSION = "mc.analyze.prompt_family.v1"


@dataclass(frozen=True)
class ObservationTarget:
    """A target model surface to measure."""

    label: str
    model: str
    adapter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "adapter": self.adapter,
        }


@dataclass(frozen=True)
class PromptVariant:
    """One explicit prompt row in a family manifest."""

    case_id: str
    variant_id: str
    text: str
    tags: tuple[str, ...] = ()
    comparison_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "text": self.text,
        }
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.comparison_to:
            payload["comparison_to"] = self.comparison_to
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptVariant":
        case_id = str(payload.get("case_id", "")).strip()
        variant_id = str(payload.get("variant_id", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not case_id:
            raise ValueError("Prompt variant is missing case_id")
        if not variant_id:
            raise ValueError("Prompt variant is missing variant_id")
        if not text:
            raise ValueError("Prompt variant is missing text")

        raw_tags = payload.get("tags", [])
        tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())

        raw_comparison_to = payload.get("comparison_to")
        comparison_to = str(raw_comparison_to).strip() if raw_comparison_to else None

        return cls(
            case_id=case_id,
            variant_id=variant_id,
            text=text,
            tags=tags,
            comparison_to=comparison_to,
        )


@dataclass(frozen=True)
class PromptFamilyManifest:
    """Explicit prompt-family rows for controlled perturbation studies."""

    name: str
    variants: tuple[PromptVariant, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PROMPT_FAMILY_MANIFEST_VERSION,
            "name": self.name,
            "variants": [variant.to_dict() for variant in self.variants],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_json_path(cls, path: str | Path) -> "PromptFamilyManifest":
        resolved = Path(path).expanduser().resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        return cls.from_data(data, default_name=resolved.stem)

    @classmethod
    def from_data(
        cls,
        data: Any,
        *,
        default_name: str = "prompt_family",
    ) -> "PromptFamilyManifest":
        if isinstance(data, list):
            name = default_name
            metadata: dict[str, Any] = {}
            rows = data
        elif isinstance(data, dict):
            name = str(data.get("name", default_name)).strip() or default_name
            metadata = data.get("metadata", {})
            rows = data.get("variants", data.get("rows", data.get("prompts")))
        else:
            raise ValueError("Prompt family manifest must be a JSON object or array")

        if not isinstance(rows, list) or not rows:
            raise ValueError("Prompt family manifest must include a non-empty variants list")

        variants = tuple(PromptVariant.from_dict(dict(row)) for row in rows)
        return cls(name=name, variants=variants, metadata=dict(metadata))

    @classmethod
    def from_prompts(
        cls,
        prompts: list[str],
        *,
        name: str = "capture",
    ) -> "PromptFamilyManifest":
        variants = []
        for index, prompt in enumerate(prompts, start=1):
            variants.append(
                PromptVariant(
                    case_id=f"{name}_{index:03d}",
                    variant_id="capture",
                    text=prompt,
                )
            )
        return cls(name=name, variants=tuple(variants))

    def grouped_variants(self) -> dict[str, list[PromptVariant]]:
        grouped: dict[str, list[PromptVariant]] = {}
        for variant in self.variants:
            grouped.setdefault(variant.case_id, []).append(variant)
        return grouped


@dataclass(frozen=True)
class ObservationRunResult:
    """Summary of a persisted observation bundle."""

    workflow: str
    output_dir: str
    summary: dict[str, Any]
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "outputDir": self.output_dir,
            "summary": self.summary,
            "files": self.files,
        }


class ObservationService:
    """Workflow-first observation service for capture, family, and compare."""

    def __init__(
        self,
        *,
        backend: "Backend",
        model_loader: "ModelLoaderPort",
        activation_provider: "ActivationProvider",
        sublayer_collector: "SublayerCollector | None" = None,
        geometry_service_factory: "Callable[[], GeometryAnalysisService] | None" = None,
        chain_service_factory: "Callable[[], ChainAnalysisService] | None" = None,
        geodesic_service_factory: "Callable[[], GeodesicTrajectoryService] | None" = None,
        behavioral_analyzer_factory: "Callable[[], BehavioralAnalyzer] | None" = None,
    ) -> None:
        if chain_service_factory is None and sublayer_collector is None:
            raise ValueError(
                "ObservationService requires sublayer_collector when using the default "
                "chain analysis service factory."
            )
        self._backend = backend
        self._model_loader = model_loader
        self._activation_provider = activation_provider
        self._sublayer_collector = sublayer_collector
        self._geometry_service_factory = geometry_service_factory or self._build_geometry_service
        self._chain_service_factory = chain_service_factory or self._build_chain_service
        self._geodesic_service_factory = geodesic_service_factory or self._build_geodesic_service
        self._behavioral_analyzer_factory = (
            behavioral_analyzer_factory or self._build_behavioral_analyzer
        )

    def capture(
        self,
        *,
        target: ObservationTarget,
        manifest: PromptFamilyManifest,
        output_dir: str | None = None,
        spaces: tuple[str, ...] = DEFAULT_ANALYSIS_SPACES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ObservationRunResult:
        return self._run_bundle(
            workflow="capture",
            targets=(target,),
            manifest=manifest,
            output_dir=output_dir,
            spaces=spaces,
            max_tokens=max_tokens,
            include_within_target_comparisons=False,
            include_between_target_comparisons=False,
        )

    def family(
        self,
        *,
        target: ObservationTarget,
        manifest: PromptFamilyManifest,
        output_dir: str | None = None,
        spaces: tuple[str, ...] = DEFAULT_ANALYSIS_SPACES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ObservationRunResult:
        return self._run_bundle(
            workflow="family",
            targets=(target,),
            manifest=manifest,
            output_dir=output_dir,
            spaces=spaces,
            max_tokens=max_tokens,
            include_within_target_comparisons=True,
            include_between_target_comparisons=False,
        )

    def compare(
        self,
        *,
        left: ObservationTarget,
        right: ObservationTarget,
        manifest: PromptFamilyManifest,
        output_dir: str | None = None,
        spaces: tuple[str, ...] = DEFAULT_ANALYSIS_SPACES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ObservationRunResult:
        return self._run_bundle(
            workflow="compare",
            targets=(left, right),
            manifest=manifest,
            output_dir=output_dir,
            spaces=spaces,
            max_tokens=max_tokens,
            include_within_target_comparisons=True,
            include_between_target_comparisons=True,
        )

    def _run_bundle(
        self,
        *,
        workflow: str,
        targets: tuple[ObservationTarget, ...],
        manifest: PromptFamilyManifest,
        output_dir: str | None,
        spaces: tuple[str, ...],
        max_tokens: int,
        include_within_target_comparisons: bool,
        include_between_target_comparisons: bool,
    ) -> ObservationRunResult:
        bundle_dir = self._resolve_output_dir(
            workflow=workflow,
            manifest_name=manifest.name,
            output_dir=output_dir,
        )
        spaces = self._normalize_spaces(spaces)

        manifest_payload = {
            "bundleVersion": OBSERVATION_BUNDLE_VERSION,
            "workflow": workflow,
            "requestedAt": datetime.now(UTC).isoformat(),
            "targets": [target.to_dict() for target in targets],
            "spaces": list(spaces),
            "maxTokens": max_tokens,
            "promptFamilyManifest": manifest.to_dict(),
        }

        variant_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        variant_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        hidden_layer_index: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
        geometry_service = self._geometry_service_factory()
        chain_service = self._chain_service_factory()
        geodesic_service = self._geodesic_service_factory()
        behavioral_analyzer = self._behavioral_analyzer_factory()

        for target in targets:
            model, tokenizer = self._model_loader.load_model(
                target.model,
                adapter_path=target.adapter,
            )

            for variant in manifest.variants:
                variant_row, variant_layer_rows = self._observe_variant(
                    model=model,
                    tokenizer=tokenizer,
                    target=target,
                    variant=variant,
                    spaces=spaces,
                    max_tokens=max_tokens,
                    geometry_service=geometry_service,
                    chain_service=chain_service,
                    geodesic_service=geodesic_service,
                    behavioral_analyzer=behavioral_analyzer,
                )
                variant_rows.append(variant_row)
                layer_rows.extend(variant_layer_rows)
                variant_index[(target.label, variant.case_id, variant.variant_id)] = variant_row
                hidden_layer_index[(target.label, variant.case_id, variant.variant_id)] = {
                    row["layer"]: row
                    for row in variant_layer_rows
                    if row["space"] == "hidden"
                }

            if include_within_target_comparisons:
                comparisons.extend(
                    self._build_within_target_comparisons(
                        target=target,
                        manifest=manifest,
                        variant_index=variant_index,
                        hidden_layer_index=hidden_layer_index,
                    )
                )

        if include_between_target_comparisons and len(targets) == 2:
            comparisons.extend(
                self._build_between_target_comparisons(
                    left=targets[0],
                    right=targets[1],
                    manifest=manifest,
                    variant_index=variant_index,
                    hidden_layer_index=hidden_layer_index,
                )
            )

        summary = self._build_summary(
            workflow=workflow,
            manifest=manifest,
            targets=targets,
            spaces=spaces,
            variant_rows=variant_rows,
            layer_rows=layer_rows,
            comparisons=comparisons,
        )

        files = self._write_bundle(
            bundle_dir=bundle_dir,
            manifest_payload=manifest_payload,
            summary=summary,
            variant_rows=variant_rows,
            layer_rows=layer_rows,
            comparisons=comparisons,
        )
        return ObservationRunResult(
            workflow=workflow,
            output_dir=str(bundle_dir),
            summary=summary,
            files=files,
        )

    def _build_geometry_service(self) -> GeometryAnalysisService:
        return GeometryAnalysisService(
            backend=self._backend,
            activation_provider=self._activation_provider,
        )

    def _build_chain_service(self) -> ChainAnalysisService:
        if self._sublayer_collector is None:
            raise ValueError("ObservationService is missing sublayer_collector.")
        return ChainAnalysisService(
            backend=self._backend,
            activation_provider=self._activation_provider,
            sublayer_collector=self._sublayer_collector,
        )

    def _build_geodesic_service(self) -> GeodesicTrajectoryService:
        return GeodesicTrajectoryService(
            backend=self._backend,
            activation_provider=self._activation_provider,
        )

    def _build_behavioral_analyzer(self) -> BehavioralAnalyzer:
        return BehavioralAnalyzer(self._activation_provider, self._backend)

    def _observe_variant(
        self,
        *,
        model: Any,
        tokenizer: Any,
        target: ObservationTarget,
        variant: PromptVariant,
        spaces: tuple[str, ...],
        max_tokens: int,
        geometry_service: GeometryAnalysisService,
        chain_service: ChainAnalysisService,
        geodesic_service: GeodesicTrajectoryService,
        behavioral_analyzer: BehavioralAnalyzer,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        backend = self._backend

        prompt_token_count = len(backend.encode_tokens(tokenizer, variant.text))
        generation_started = time.time()
        response = self._model_loader.generate(
            model,
            tokenizer,
            variant.text,
            max_tokens=max_tokens,
        )
        generation_duration = time.time() - generation_started
        response_token_count = len(backend.encode_tokens(tokenizer, response))

        errors: list[str] = []
        layer_rows: list[dict[str, Any]] = []

        trajectory = None
        try:
            trajectory = self._activation_provider.collect_trajectory_batch(
                model,
                tokenizer,
                [variant.text],
            )
        except Exception as exc:
            errors.append(f"trajectory:{exc}")

        hidden_vectors: dict[int, Any] = {}
        if "hidden" in spaces:
            try:
                hidden_vectors = self._activation_provider.collect_hidden_activations(
                    model,
                    tokenizer,
                    variant.text,
                )
            except Exception as exc:
                errors.append(f"hidden:{exc}")

        entropy_result = None
        try:
            entropy_result = behavioral_analyzer.analyze_entropy_trajectory(
                model,
                tokenizer,
                [variant.text],
            )
        except Exception as exc:
            errors.append(f"entropy:{exc}")

        flow_map: dict[int, Any] = {}
        if trajectory is not None:
            try:
                flow_profiles = geometry_service.analyze_reasoning_flow(trajectory.positions)
                flow_map = {profile.layer_idx: profile.metrics for profile in flow_profiles}
            except Exception as exc:
                errors.append(f"reasoning_flow:{exc}")

        geodesic_map: dict[int, Any] = {}
        if trajectory is not None:
            try:
                geodesic_profile = geodesic_service.measure_layer_profile(
                    model,
                    tokenizer,
                    variant.text,
                )
                geodesic_map = {
                    profile.layer: profile for profile in geodesic_profile.layer_profiles
                }
            except Exception as exc:
                errors.append(f"geodesic:{exc}")

        chain_map: dict[int, Any] = {}
        try:
            chain_profile = chain_service.analyze_chain(model, tokenizer, [variant.text])
            chain_map = {row.layer_idx: row for row in chain_profile.layers}
        except Exception as exc:
            errors.append(f"chain:{exc}")

        entropy_map: dict[int, float] = {}
        if entropy_result is not None:
            entropy_map = dict(
                zip(entropy_result.layer_indices, entropy_result.layer_entropies)
            )

        if trajectory is not None:
            for layer_idx in sorted(trajectory.positions.keys()):
                position_matrix = trajectory.positions[layer_idx]
                layer_entropy = None
                try:
                    layer_entropy = geometry_service.compute_layer_entropy(
                        position_matrix,
                        layer_idx,
                    )
                except Exception as exc:
                    errors.append(f"layer_entropy:{layer_idx}:{exc}")

                vector_norm = None
                vector_size = None
                if layer_idx in hidden_vectors:
                    vector_norm, vector_size = self._vector_stats(hidden_vectors[layer_idx])
                else:
                    vector_size = self._shape_tail(position_matrix)

                flow_metrics = flow_map.get(layer_idx)
                geodesic_metrics = geodesic_map.get(layer_idx)
                chain_metrics = chain_map.get(layer_idx)

                layer_rows.append(
                    {
                        "targetLabel": target.label,
                        "caseId": variant.case_id,
                        "variantId": variant.variant_id,
                        "layer": layer_idx,
                        "space": "hidden",
                        "vectorNorm": vector_norm,
                        "vectorSize": vector_size,
                        "entropy": entropy_map.get(layer_idx),
                        "spectralEntropy": getattr(layer_entropy, "spectral_entropy", None),
                        "effectiveRank": getattr(layer_entropy, "effective_rank", None),
                        "intrinsicDimension": getattr(layer_entropy, "intrinsic_dimension", None),
                        "geodesicMeanDeviation": getattr(geodesic_metrics, "mean_deviation", None),
                        "geodesicPathLengthRatio": getattr(
                            geodesic_metrics, "path_length_ratio", None
                        ),
                        "flowMeanCurvature": getattr(flow_metrics, "mean_curvature", None),
                        "flowMaxCurvature": getattr(flow_metrics, "max_curvature", None),
                        "flowSmoothness": getattr(flow_metrics, "smoothness", None),
                        "flowDirectness": getattr(flow_metrics, "directness", None),
                        "chainPhase": getattr(getattr(chain_metrics, "phase", None), "value", None),
                        "chainAttnFraction": getattr(chain_metrics, "attn_fraction", None),
                    }
                )

        try:
            layer_rows.extend(
                self._collect_optional_space_rows(
                    model=model,
                    tokenizer=tokenizer,
                    target=target,
                    variant=variant,
                    spaces=spaces,
                )
            )
        except Exception as exc:
            errors.append(f"spaces:{exc}")

        hidden_rows = [row for row in layer_rows if row["space"] == "hidden"]
        summary_metrics = {
            "meanEntropy": self._mean_metric(hidden_rows, "entropy"),
            "peakEntropy": self._max_metric(hidden_rows, "entropy"),
            "meanIntrinsicDimension": self._mean_metric(hidden_rows, "intrinsicDimension"),
            "meanGeodesicDeviation": self._mean_metric(hidden_rows, "geodesicMeanDeviation"),
            "maxGeodesicDeviation": self._max_metric(hidden_rows, "geodesicMeanDeviation"),
            "meanCurvature": self._mean_metric(hidden_rows, "flowMeanCurvature"),
            "maxCurvature": self._max_metric(hidden_rows, "flowMaxCurvature"),
            "meanPathLengthRatio": self._mean_metric(hidden_rows, "geodesicPathLengthRatio"),
            "phaseCounts": self._phase_counts(hidden_rows),
        }
        if entropy_result is not None:
            summary_metrics["entropySlope"] = entropy_result.slope
            summary_metrics["entropyPeakLayerFraction"] = entropy_result.peak_layer_fraction

        variant_row = {
            "targetLabel": target.label,
            "caseId": variant.case_id,
            "variantId": variant.variant_id,
            "comparisonTo": variant.comparison_to,
            "tags": list(variant.tags),
            "text": variant.text,
            "promptTokenCount": prompt_token_count,
            "response": response,
            "responseTokenCount": response_token_count,
            "generationDurationSec": generation_duration,
            "spaces": list(spaces),
            "summaryMetrics": summary_metrics,
        }
        if errors:
            variant_row["errors"] = errors
        return variant_row, layer_rows

    def _collect_optional_space_rows(
        self,
        *,
        model: Any,
        tokenizer: Any,
        target: ObservationTarget,
        variant: PromptVariant,
        spaces: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        if "embedding" in spaces:
            embedding = self._activation_provider.collect_embedding_activations(
                model,
                tokenizer,
                variant.text,
            )
            vector_norm, vector_size = self._vector_stats(embedding)
            rows.append(
                {
                    "targetLabel": target.label,
                    "caseId": variant.case_id,
                    "variantId": variant.variant_id,
                    "layer": -1,
                    "space": "embedding",
                    "vectorNorm": vector_norm,
                    "vectorSize": vector_size,
                }
            )

        if "intermediate" in spaces:
            intermediate = self._activation_provider.collect_intermediate_activations(
                model,
                tokenizer,
                variant.text,
            )
            rows.extend(
                self._space_rows_from_mapping(
                    mapping=intermediate,
                    target=target,
                    variant=variant,
                    space="intermediate",
                )
            )

        if "gate" in spaces:
            gate_batch = self._activation_provider.collect_gate_activations_batch(
                model,
                tokenizer,
                [variant.text],
            )
            if gate_batch:
                rows.extend(
                    self._space_rows_from_mapping(
                        mapping=gate_batch[0],
                        target=target,
                        variant=variant,
                        space="gate",
                    )
                )

        if any(space in spaces for space in ("q", "k", "v")):
            q_map, k_map, v_map = self._activation_provider.collect_attention_activations(
                model,
                tokenizer,
                variant.text,
            )
            if "q" in spaces:
                rows.extend(
                    self._space_rows_from_mapping(
                        mapping=q_map,
                        target=target,
                        variant=variant,
                        space="q",
                    )
                )
            if "k" in spaces:
                rows.extend(
                    self._space_rows_from_mapping(
                        mapping=k_map,
                        target=target,
                        variant=variant,
                        space="k",
                    )
                )
            if "v" in spaces:
                rows.extend(
                    self._space_rows_from_mapping(
                        mapping=v_map,
                        target=target,
                        variant=variant,
                        space="v",
                    )
                )

        return rows

    def _space_rows_from_mapping(
        self,
        *,
        mapping: dict[int, Any],
        target: ObservationTarget,
        variant: PromptVariant,
        space: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_idx, value in sorted(mapping.items()):
            vector_norm, vector_size = self._vector_stats(value)
            rows.append(
                {
                    "targetLabel": target.label,
                    "caseId": variant.case_id,
                    "variantId": variant.variant_id,
                    "layer": layer_idx,
                    "space": space,
                    "vectorNorm": vector_norm,
                    "vectorSize": vector_size,
                }
            )
        return rows

    def _build_within_target_comparisons(
        self,
        *,
        target: ObservationTarget,
        manifest: PromptFamilyManifest,
        variant_index: dict[tuple[str, str, str], dict[str, Any]],
        hidden_layer_index: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for case_id, variants in manifest.grouped_variants().items():
            baseline = self._resolve_baseline_variant(variants)
            baseline_variant = variant_index.get((target.label, case_id, baseline.variant_id))
            baseline_layers = hidden_layer_index.get((target.label, case_id, baseline.variant_id), {})
            if baseline_variant is None:
                continue

            for variant in variants:
                if variant.variant_id == baseline.variant_id:
                    continue
                observed = variant_index.get((target.label, case_id, variant.variant_id))
                observed_layers = hidden_layer_index.get((target.label, case_id, variant.variant_id), {})
                if observed is None:
                    continue
                comparisons.append(
                    self._comparison_row(
                        mode="within_target",
                        case_id=case_id,
                        from_label=baseline.variant_id,
                        to_label=variant.variant_id,
                        metadata={
                            "targetLabel": target.label,
                            "comparisonType": "variant",
                        },
                        baseline_variant=baseline_variant,
                        observed_variant=observed,
                        baseline_layers=baseline_layers,
                        observed_layers=observed_layers,
                    )
                )
        return comparisons

    def _build_between_target_comparisons(
        self,
        *,
        left: ObservationTarget,
        right: ObservationTarget,
        manifest: PromptFamilyManifest,
        variant_index: dict[tuple[str, str, str], dict[str, Any]],
        hidden_layer_index: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for variant in manifest.variants:
            left_variant = variant_index.get((left.label, variant.case_id, variant.variant_id))
            right_variant = variant_index.get((right.label, variant.case_id, variant.variant_id))
            if left_variant is None or right_variant is None:
                continue
            comparisons.append(
                self._comparison_row(
                    mode="between_targets",
                    case_id=variant.case_id,
                    from_label=left.label,
                    to_label=right.label,
                    metadata={"variantId": variant.variant_id},
                    baseline_variant=left_variant,
                    observed_variant=right_variant,
                    baseline_layers=hidden_layer_index.get(
                        (left.label, variant.case_id, variant.variant_id),
                        {},
                    ),
                    observed_layers=hidden_layer_index.get(
                        (right.label, variant.case_id, variant.variant_id),
                        {},
                    ),
                )
            )
        return comparisons

    def _comparison_row(
        self,
        *,
        mode: str,
        case_id: str,
        from_label: str,
        to_label: str,
        metadata: dict[str, Any],
        baseline_variant: dict[str, Any],
        observed_variant: dict[str, Any],
        baseline_layers: dict[int, dict[str, Any]],
        observed_layers: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        scalar_deltas = {
            "responseTokenCount": self._metric_delta(
                baseline_variant, observed_variant, ("responseTokenCount",)
            ),
            "generationDurationSec": self._metric_delta(
                baseline_variant, observed_variant, ("generationDurationSec",)
            ),
            "meanEntropy": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "meanEntropy"),
            ),
            "peakEntropy": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "peakEntropy"),
            ),
            "meanIntrinsicDimension": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "meanIntrinsicDimension"),
            ),
            "meanGeodesicDeviation": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "meanGeodesicDeviation"),
            ),
            "maxGeodesicDeviation": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "maxGeodesicDeviation"),
            ),
            "meanCurvature": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "meanCurvature"),
            ),
            "maxCurvature": self._metric_delta(
                baseline_variant,
                observed_variant,
                ("summaryMetrics", "maxCurvature"),
            ),
        }

        per_layer_deltas = {
            "entropy": self._per_layer_deltas(baseline_layers, observed_layers, "entropy"),
            "spectralEntropy": self._per_layer_deltas(
                baseline_layers, observed_layers, "spectralEntropy"
            ),
            "intrinsicDimension": self._per_layer_deltas(
                baseline_layers, observed_layers, "intrinsicDimension"
            ),
            "geodesicMeanDeviation": self._per_layer_deltas(
                baseline_layers, observed_layers, "geodesicMeanDeviation"
            ),
            "flowMeanCurvature": self._per_layer_deltas(
                baseline_layers, observed_layers, "flowMeanCurvature"
            ),
        }
        payload = {
            "mode": mode,
            "caseId": case_id,
            "from": from_label,
            "to": to_label,
            "scalarDeltas": scalar_deltas,
            "perLayerDeltas": per_layer_deltas,
        }
        payload.update(metadata)
        return payload

    @staticmethod
    def _resolve_baseline_variant(variants: list[PromptVariant]) -> PromptVariant:
        variant_by_id = {variant.variant_id: variant for variant in variants}
        for variant in variants:
            if variant.variant_id == "control":
                return variant
        for variant in variants:
            if variant.comparison_to and variant.comparison_to in variant_by_id:
                return variant_by_id[variant.comparison_to]
        return variants[0]

    def _build_summary(
        self,
        *,
        workflow: str,
        manifest: PromptFamilyManifest,
        targets: tuple[ObservationTarget, ...],
        spaces: tuple[str, ...],
        variant_rows: list[dict[str, Any]],
        layer_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
    ) -> dict[str, Any]:
        error_count = sum(len(row.get("errors", [])) for row in variant_rows)
        return {
            "bundleVersion": OBSERVATION_BUNDLE_VERSION,
            "workflow": workflow,
            "manifestName": manifest.name,
            "targetCount": len(targets),
            "targets": [target.to_dict() for target in targets],
            "spaces": list(spaces),
            "variantCount": len(variant_rows),
            "layerMetricCount": len(layer_rows),
            "comparisonCount": len(comparisons),
            "errorCount": error_count,
            "meanResponseTokenCount": self._mean_metric(variant_rows, "responseTokenCount"),
            "meanPromptTokenCount": self._mean_metric(variant_rows, "promptTokenCount"),
            "meanEntropy": self._mean_metric_from_summary(variant_rows, "meanEntropy"),
            "meanGeodesicDeviation": self._mean_metric_from_summary(
                variant_rows, "meanGeodesicDeviation"
            ),
            "meanCurvature": self._mean_metric_from_summary(variant_rows, "meanCurvature"),
        }

    def _write_bundle(
        self,
        *,
        bundle_dir: Path,
        manifest_payload: dict[str, Any],
        summary: dict[str, Any],
        variant_rows: list[dict[str, Any]],
        layer_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
    ) -> dict[str, str]:
        bundle_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = bundle_dir / "manifest.json"
        summary_path = bundle_dir / "summary.json"
        report_path = bundle_dir / "REPORT.md"
        variants_path = bundle_dir / "variants.jsonl"
        layer_metrics_path = bundle_dir / "layer_metrics.jsonl"
        comparisons_path = bundle_dir / "comparisons.jsonl"

        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        variants_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in variant_rows),
            encoding="utf-8",
        )
        layer_metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in layer_rows),
            encoding="utf-8",
        )
        comparisons_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in comparisons),
            encoding="utf-8",
        )
        report_path.write_text(
            self._build_report(summary, variant_rows, comparisons),
            encoding="utf-8",
        )

        return {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "variants": str(variants_path),
            "layerMetrics": str(layer_metrics_path),
            "comparisons": str(comparisons_path),
        }

    @staticmethod
    def _build_report(
        summary: dict[str, Any],
        variant_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
    ) -> str:
        lines = [
            "# Observation Bundle",
            "",
            f"- Workflow: `{summary['workflow']}`",
            f"- Targets: {summary['targetCount']}",
            f"- Variants: {summary['variantCount']}",
            f"- Comparisons: {summary['comparisonCount']}",
            f"- Measurement errors: {summary['errorCount']}",
            f"- Spaces: {', '.join(summary['spaces'])}",
            "",
            "## Means",
            "",
            f"- Mean prompt tokens: {summary['meanPromptTokenCount']}",
            f"- Mean response tokens: {summary['meanResponseTokenCount']}",
            f"- Mean entropy: {summary['meanEntropy']}",
            f"- Mean geodesic deviation: {summary['meanGeodesicDeviation']}",
            f"- Mean curvature: {summary['meanCurvature']}",
        ]
        if variant_rows:
            lines.extend(["", "## Variants", ""])
            for row in variant_rows[:10]:
                summary_metrics = row.get("summaryMetrics", {})
                lines.append(
                    f"- `{row['targetLabel']}` / `{row['caseId']}` / `{row['variantId']}`: "
                    f"entropy={summary_metrics.get('meanEntropy')}, "
                    f"geodesic={summary_metrics.get('meanGeodesicDeviation')}, "
                    f"curvature={summary_metrics.get('meanCurvature')}"
                )
        if comparisons:
            lines.extend(["", "## Comparisons", ""])
            for row in comparisons[:10]:
                deltas = row.get("scalarDeltas", {})
                lines.append(
                    f"- `{row['from']}` -> `{row['to']}` on `{row['caseId']}`: "
                    f"entropy_delta={deltas.get('meanEntropy')}, "
                    f"geodesic_delta={deltas.get('meanGeodesicDeviation')}, "
                    f"curvature_delta={deltas.get('meanCurvature')}"
                )
        error_rows = [row for row in variant_rows if row.get("errors")]
        if error_rows:
            lines.extend(["", "## Measurement Errors", ""])
            for row in error_rows[:10]:
                lines.append(
                    f"- `{row['targetLabel']}` / `{row['caseId']}` / `{row['variantId']}`: "
                    + "; ".join(row["errors"])
                )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _resolve_output_dir(
        *,
        workflow: str,
        manifest_name: str,
        output_dir: str | None,
    ) -> Path:
        if output_dir:
            return Path(output_dir).expanduser().resolve()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = ObservationService._slugify(f"{workflow}-{manifest_name}") or workflow
        return (Path.cwd() / "results" / "analysis" / f"{timestamp}-{slug}").resolve()

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    @staticmethod
    def _normalize_spaces(spaces: tuple[str, ...]) -> tuple[str, ...]:
        normalized = []
        seen = set()
        for raw_space in spaces:
            space = raw_space.strip().lower()
            if not space:
                continue
            if space not in SUPPORTED_ANALYSIS_SPACES:
                raise ValueError(
                    f"Unsupported analysis space '{space}'. "
                    f"Supported spaces: {', '.join(SUPPORTED_ANALYSIS_SPACES)}"
                )
            if space not in seen:
                normalized.append(space)
                seen.add(space)
        return tuple(normalized or DEFAULT_ANALYSIS_SPACES)

    def _vector_stats(self, value: Any) -> tuple[float | None, int | None]:
        try:
            self._backend.eval(value)
            python_value = self._backend.tolist(value)
        except Exception as exc:
            logger.debug("Vector stats failed during backend materialization: %s", exc)
            return None, None
        flat = self._flatten_numeric(python_value)
        if not flat:
            return None, 0
        norm = math.sqrt(sum(component * component for component in flat))
        return norm, len(flat)

    @staticmethod
    def _shape_tail(value: Any) -> int | None:
        try:
            shape = value.shape  # type: ignore[attr-defined]
        except Exception:
            return None
        if not shape:
            return None
        return int(shape[-1])

    @staticmethod
    def _flatten_numeric(value: Any) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            flattened: list[float] = []
            for item in value:
                flattened.extend(ObservationService._flatten_numeric(item))
            return flattened
        return []

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]
        if not values:
            return None
        return max(values)

    @staticmethod
    def _mean_metric_from_summary(
        variant_rows: list[dict[str, Any]],
        key: str,
    ) -> float | None:
        values = [
            float(row["summaryMetrics"][key])
            for row in variant_rows
            if row.get("summaryMetrics", {}).get(key) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _phase_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            phase = row.get("chainPhase")
            if phase is None:
                continue
            counts[phase] = counts.get(phase, 0) + 1
        return counts

    @staticmethod
    def _metric_delta(
        baseline_variant: dict[str, Any],
        observed_variant: dict[str, Any],
        path: tuple[str, ...],
    ) -> float | None:
        baseline_value = ObservationService._nested_get(baseline_variant, path)
        observed_value = ObservationService._nested_get(observed_variant, path)
        if baseline_value is None or observed_value is None:
            return None
        return float(observed_value) - float(baseline_value)

    @staticmethod
    def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _per_layer_deltas(
        baseline_layers: dict[int, dict[str, Any]],
        observed_layers: dict[int, dict[str, Any]],
        key: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_idx in sorted(set(baseline_layers) | set(observed_layers)):
            baseline_value = baseline_layers.get(layer_idx, {}).get(key)
            observed_value = observed_layers.get(layer_idx, {}).get(key)
            if baseline_value is None or observed_value is None:
                continue
            rows.append(
                {
                    "layer": layer_idx,
                    "delta": float(observed_value) - float(baseline_value),
                }
            )
        return rows
