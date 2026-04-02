from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry
from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer
from modelcypher.core.use_cases.geodesic_trajectory_service import GeodesicTrajectoryService
from modelcypher.core.use_cases.geometry_analysis_service import GeometryAnalysisService
from modelcypher.core.use_cases.observation_service import (
    DEFAULT_ANALYSIS_SPACES,
    DEFAULT_MAX_TOKENS,
    SUPPORTED_ANALYSIS_SPACES,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveTraceStep:
    """One realized decode step from the live hidden-state trace."""

    step_index: int
    token_id: int
    token_text: str
    hidden_by_layer: dict[int, Any]
    logit_entropy: float | None = None
    logit_margin: float | None = None


@dataclass(frozen=True)
class LiveGenerationTraceResult:
    """Greedy live decode trace for one prompt."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stop_reason: str
    steps: tuple[LiveTraceStep, ...]


@dataclass(frozen=True)
class GenerationTraceTokenStream:
    """Token stream metadata for one traced region."""

    mode: str
    region: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    prompt_boundary_index: int | None = None


@dataclass(frozen=True)
class GenerationTraceResult:
    """Per-variant replay and live trace outputs."""

    prompt_text: str
    generated_text: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    live_generated_token_ids: tuple[int, ...]
    token_streams: tuple[GenerationTraceTokenStream, ...]
    sequence_metrics: tuple[dict[str, Any], ...]
    step_metrics: tuple[dict[str, Any], ...]
    space_step_metrics: tuple[dict[str, Any], ...]
    decode: dict[str, Any]
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReplayTraceRegion:
    """One replay region with the token sequence that should represent it."""

    name: str
    text: str
    token_ids: tuple[int, ...]
    source_name: str
    source_text: str
    source_token_ids: tuple[int, ...] | None = None
    slice_start: int = 0
    slice_end: int | None = None
    prompt_boundary_index: int | None = None


LiveTraceRunner = Callable[[Any, Any, str, int], LiveGenerationTraceResult]


class GenerationTraceService:
    """Trace prompt and generation geometry for one variant."""

    def __init__(
        self,
        *,
        backend: "Backend",
        model_loader: "ModelLoaderPort",
        activation_provider: "ActivationProvider",
        live_trace_runner: LiveTraceRunner | None = None,
        geometry_service_factory: "Callable[[], GeometryAnalysisService] | None" = None,
        geodesic_service_factory: "Callable[[], GeodesicTrajectoryService] | None" = None,
        behavioral_analyzer_factory: "Callable[[], BehavioralAnalyzer] | None" = None,
    ) -> None:
        self._backend = backend
        self._model_loader = model_loader
        self._activation_provider = activation_provider
        self._live_trace_runner = live_trace_runner
        self._geometry_service = (
            geometry_service_factory()
            if geometry_service_factory is not None
            else GeometryAnalysisService(
                backend=backend,
                activation_provider=activation_provider,
            )
        )
        self._geodesic_service = (
            geodesic_service_factory()
            if geodesic_service_factory is not None
            else GeodesicTrajectoryService(
                backend=backend,
                activation_provider=activation_provider,
            )
        )
        self._behavioral_analyzer = (
            behavioral_analyzer_factory()
            if behavioral_analyzer_factory is not None
            else BehavioralAnalyzer(activation_provider, backend)
        )

    def trace_variant(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        spaces: tuple[str, ...] = DEFAULT_ANALYSIS_SPACES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> GenerationTraceResult:
        spaces = self._normalize_spaces(spaces)
        errors: list[str] = []
        sequence_metrics: list[dict[str, Any]] = []
        step_metrics: list[dict[str, Any]] = []
        space_step_metrics: list[dict[str, Any]] = []
        token_streams: list[GenerationTraceTokenStream] = []

        live_result: LiveGenerationTraceResult | None = None
        generated_text = ""

        if self._live_trace_runner is not None:
            try:
                live_result = self._live_trace_runner(model, tokenizer, prompt, max_tokens)
                generated_text = self._normalize_generated_text(
                    prompt=prompt,
                    generated_text=live_result.generated_text,
                )
            except Exception as exc:
                errors.append(f"live_trace:{exc}")
                logger.debug("Live trace failed for prompt", exc_info=True)

        if not generated_text:
            try:
                generated_text = self._normalize_generated_text(
                    prompt=prompt,
                    generated_text=self._model_loader.generate(
                        model,
                        tokenizer,
                        prompt,
                        max_tokens=max_tokens,
                    ),
                )
            except Exception as exc:
                errors.append(f"generate:{exc}")
                generated_text = ""

        prompt_token_ids = (
            live_result.prompt_token_ids
            if live_result is not None
            else tuple(self._encode_tokens(tokenizer, prompt))
        )
        response_token_ids = self._response_token_ids(
            tokenizer=tokenizer,
            generated_text=generated_text,
            live_result=live_result,
        )
        full_token_ids = prompt_token_ids + response_token_ids
        # Decode the realized token path so the replay/full region preserves the prompt boundary.
        full_text = self._decode_tokens(tokenizer, full_token_ids)
        replay_regions = (
            ReplayTraceRegion(
                name="prompt",
                text=prompt,
                token_ids=prompt_token_ids,
                source_name="prompt",
                source_text=prompt,
                source_token_ids=prompt_token_ids,
                slice_end=len(prompt_token_ids),
            ),
            ReplayTraceRegion(
                name="response",
                text=generated_text,
                token_ids=response_token_ids,
                source_name="full",
                source_text=full_text,
                source_token_ids=full_token_ids,
                slice_start=len(prompt_token_ids),
                slice_end=len(full_token_ids),
            ),
            ReplayTraceRegion(
                name="full",
                text=full_text,
                token_ids=full_token_ids,
                source_name="full",
                source_text=full_text,
                source_token_ids=full_token_ids,
                slice_end=len(full_token_ids),
                prompt_boundary_index=len(prompt_token_ids),
            ),
        )
        replay_errors, replay_token_streams, replay_step_metrics, replay_space_metrics, replay_sequence = (
            self._trace_replay_regions(
                model=model,
                tokenizer=tokenizer,
                regions=replay_regions,
                prompt_token_count=len(prompt_token_ids),
                spaces=spaces,
            )
        )
        errors.extend(replay_errors)
        token_streams.extend(replay_token_streams)
        step_metrics.extend(replay_step_metrics)
        space_step_metrics.extend(replay_space_metrics)
        sequence_metrics.extend(replay_sequence)

        live_errors, live_token_streams, live_step_metrics, live_space_metrics, live_sequence = (
            self._trace_live_region(
                live_result=live_result,
                prompt_token_count=len(prompt_token_ids),
            )
        )
        errors.extend(live_errors)
        token_streams.extend(live_token_streams)
        step_metrics.extend(live_step_metrics)
        space_step_metrics.extend(live_space_metrics)
        sequence_metrics.extend(live_sequence)

        decode = {
            "policy": "greedy",
            "maxTokens": max_tokens,
            "generationSource": "live" if live_result is not None else "backend_generate",
            "liveTraceCaptured": live_result is not None,
            "liveSpaces": ["hidden"] if live_result is not None else [],
            "replaySpaces": list(spaces),
            "liveGeneratedTokenCount": len(live_result.generated_token_ids) if live_result else 0,
        }
        if live_result is not None:
            decode["stopReason"] = live_result.stop_reason

        return GenerationTraceResult(
            prompt_text=prompt,
            generated_text=generated_text,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            full_token_ids=full_token_ids,
            live_generated_token_ids=(
                live_result.generated_token_ids if live_result is not None else ()
            ),
            token_streams=tuple(token_streams),
            sequence_metrics=tuple(sequence_metrics),
            step_metrics=tuple(step_metrics),
            space_step_metrics=tuple(space_step_metrics),
            decode=decode,
            errors=tuple(errors),
        )

    def _trace_replay_regions(
        self,
        *,
        model: Any,
        tokenizer: Any,
        regions: tuple[ReplayTraceRegion, ...],
        prompt_token_count: int,
        spaces: tuple[str, ...],
    ) -> tuple[
        list[str],
        list[GenerationTraceTokenStream],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        errors: list[str] = []
        token_streams: list[GenerationTraceTokenStream] = []
        step_metrics: list[dict[str, Any]] = []
        space_step_metrics: list[dict[str, Any]] = []
        sequence_metrics: list[dict[str, Any]] = []

        ordered_regions = [region for region in regions if region.source_text]
        if not ordered_regions:
            return errors, token_streams, step_metrics, space_step_metrics, sequence_metrics

        source_specs: dict[str, tuple[str, tuple[int, ...] | None]] = {}
        for region in ordered_regions:
            source_specs.setdefault(
                region.source_name,
                (region.source_text, region.source_token_ids),
            )

        try:
            trajectory = self._activation_provider.collect_trajectory_batch(
                model,
                tokenizer,
                [text for text, _token_ids in source_specs.values()],
                token_ids_batch=[
                    list(token_ids) if token_ids is not None else None
                    for _text, token_ids in source_specs.values()
                ],
            )
        except Exception as exc:
            errors.append(f"replay_trace:{exc}")
            return errors, token_streams, step_metrics, space_step_metrics, sequence_metrics

        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for source_name, length in zip(source_specs, trajectory.text_lengths, strict=False):
            next_cursor = cursor + int(length)
            offsets[source_name] = (cursor, next_cursor)
            cursor = next_cursor

        for region in ordered_regions:
            if region.source_name not in offsets:
                errors.append(f"replay_region_missing:{region.name}")
                continue
            source_start, source_end = offsets[region.source_name]
            start = source_start + region.slice_start
            end = source_start + (
                region.slice_end if region.slice_end is not None else source_end - source_start
            )
            if end > source_end or start < source_start or end < start:
                errors.append(f"replay_region_bounds:{region.name}")
                continue
            token_ids = region.token_ids
            if token_ids:
                token_texts = tuple(self._decode_token_texts(tokenizer, token_ids))
                token_streams.append(
                    GenerationTraceTokenStream(
                        mode="replay",
                        region=region.name,
                        token_ids=token_ids,
                        token_texts=token_texts,
                        prompt_boundary_index=region.prompt_boundary_index,
                    )
                )
                step_metrics.extend(
                    self._build_step_rows(
                        mode="replay",
                        region=region.name,
                        token_ids=token_ids,
                        token_texts=token_texts,
                        prompt_token_count=prompt_token_count,
                    )
                )

            region_space_positions = self._slice_region_spaces(
                trajectory=trajectory,
                start=start,
                end=end,
                spaces=spaces,
            )
            (
                region_sequence_rows,
                region_space_rows,
                region_errors,
            ) = self._summarize_region(
                model=model,
                tokenizer=tokenizer,
                mode="replay",
                region=region.name,
                region_text=region.text,
                space_positions=region_space_positions,
            )
            sequence_metrics.extend(region_sequence_rows)
            space_step_metrics.extend(region_space_rows)
            errors.extend(region_errors)

        return errors, token_streams, step_metrics, space_step_metrics, sequence_metrics

    def _trace_live_region(
        self,
        *,
        live_result: LiveGenerationTraceResult | None,
        prompt_token_count: int,
    ) -> tuple[
        list[str],
        list[GenerationTraceTokenStream],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        errors: list[str] = []
        token_streams: list[GenerationTraceTokenStream] = []
        step_metrics: list[dict[str, Any]] = []
        space_step_metrics: list[dict[str, Any]] = []
        sequence_metrics: list[dict[str, Any]] = []

        if live_result is None:
            return errors, token_streams, step_metrics, space_step_metrics, sequence_metrics

        token_texts = tuple(step.token_text for step in live_result.steps)
        token_streams.append(
            GenerationTraceTokenStream(
                mode="live",
                region="generated",
                token_ids=live_result.generated_token_ids,
                token_texts=token_texts,
            )
        )
        step_metrics.extend(
            self._build_step_rows(
                mode="live",
                region="generated",
                token_ids=live_result.generated_token_ids,
                token_texts=token_texts,
                prompt_token_count=prompt_token_count,
                live_steps=live_result.steps,
            )
        )

        hidden_by_layer: dict[int, list[Any]] = {}
        for step in live_result.steps:
            for layer_idx, hidden in step.hidden_by_layer.items():
                hidden_by_layer.setdefault(layer_idx, []).append(hidden)
        live_space_positions = {
            "hidden": {
                layer_idx: self._backend.stack(vectors, axis=0)
                for layer_idx, vectors in sorted(hidden_by_layer.items())
            }
        }
        (
            region_sequence_rows,
            region_space_rows,
            region_errors,
        ) = self._summarize_region(
            model=None,
            tokenizer=None,
            mode="live",
            region="generated",
            region_text=None,
            space_positions=live_space_positions,
        )
        sequence_metrics.extend(region_sequence_rows)
        space_step_metrics.extend(region_space_rows)
        errors.extend(region_errors)
        return errors, token_streams, step_metrics, space_step_metrics, sequence_metrics

    def _slice_region_spaces(
        self,
        *,
        trajectory: Any,
        start: int,
        end: int,
        spaces: tuple[str, ...],
    ) -> dict[str, dict[int, Any]]:
        sliced: dict[str, dict[int, Any]] = {}
        if "hidden" in spaces:
            sliced["hidden"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.positions.items()
            }
        if "embedding" in spaces:
            sliced["embedding"] = {-1: trajectory.embedding_positions[start:end]}
        if "intermediate" in spaces:
            sliced["intermediate"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.intermediate_positions.items()
            }
        if "q" in spaces:
            sliced["q"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.q_positions.items()
            }
        if "k" in spaces:
            sliced["k"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.k_positions.items()
            }
        if "v" in spaces:
            sliced["v"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.v_positions.items()
            }
        if "gate" in spaces:
            sliced["gate"] = {
                layer_idx: positions[start:end]
                for layer_idx, positions in trajectory.gate_positions.items()
            }
        return sliced

    def _summarize_region(
        self,
        *,
        model: Any,
        tokenizer: Any,
        mode: str,
        region: str,
        region_text: str | None,
        space_positions: dict[str, dict[int, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        sequence_metrics: list[dict[str, Any]] = []
        space_step_metrics: list[dict[str, Any]] = []
        errors: list[str] = []

        entropy_map_by_layer: dict[int, float] = {}
        if mode == "replay" and region_text and model is not None and tokenizer is not None:
            try:
                entropy_result = self._behavioral_analyzer.analyze_entropy_trajectory(
                    model,
                    tokenizer,
                    [region_text],
                )
                entropy_map_by_layer = dict(
                    zip(
                        entropy_result.layer_indices,
                        entropy_result.layer_entropies,
                        strict=False,
                    )
                )
            except Exception as exc:
                errors.append(f"entropy:{mode}:{region}:{exc}")

        for space, mapping in sorted(space_positions.items()):
            normalized_mapping = {
                layer_idx: self._backend.array(values)
                for layer_idx, values in sorted(mapping.items())
                if self._first_dim(values) > 0
            }
            if not normalized_mapping:
                continue

            flow_map: dict[int, Any] = {}
            try:
                flow_profiles = self._geometry_service.analyze_reasoning_flow(normalized_mapping)
                flow_map = {profile.layer_idx: profile.metrics for profile in flow_profiles}
            except Exception as exc:
                errors.append(f"reasoning_flow:{mode}:{region}:{space}:{exc}")

            geodesic_profile = None
            try:
                geodesic_profile = self._geodesic_service.measure_layer_profile_from_positions(
                    normalized_mapping
                )
            except Exception as exc:
                errors.append(f"geodesic:{mode}:{region}:{space}:{exc}")

            entropy_rows: list[dict[str, Any]] = []
            for layer_idx, values in normalized_mapping.items():
                try:
                    layer_entropy = self._geometry_service.compute_layer_entropy(values, layer_idx)
                    entropy_rows.append(
                        {
                            "layer": layer_idx,
                            "spectralEntropy": layer_entropy.spectral_entropy,
                            "effectiveRank": layer_entropy.effective_rank,
                            "intrinsicDimension": layer_entropy.intrinsic_dimension,
                            "entropy": entropy_map_by_layer.get(layer_idx),
                        }
                    )
                except Exception as exc:
                    errors.append(f"layer_entropy:{mode}:{region}:{space}:{layer_idx}:{exc}")

            space_step_metrics.extend(
                self._build_space_step_rows(
                    mode=mode,
                    region=region,
                    space=space,
                    layer_positions=normalized_mapping,
                )
            )
            geodesic_rows = {
                profile.layer: profile
                for profile in getattr(geodesic_profile, "layer_profiles", [])
            }
            peak_layer = getattr(geodesic_profile, "peak_deviation_layer", None)
            first_bend_layer = getattr(geodesic_profile, "inflection_layer", None)
            sequence_metrics.append(
                {
                    "mode": mode,
                    "region": region,
                    "space": space,
                    "tokenCount": self._first_dim(next(iter(normalized_mapping.values()))),
                    "meanEntropy": self._mean_metric(entropy_rows, "entropy"),
                    "meanSpectralEntropy": self._mean_metric(entropy_rows, "spectralEntropy"),
                    "meanEffectiveRank": self._mean_metric(entropy_rows, "effectiveRank"),
                    "meanIntrinsicDimension": self._mean_metric(
                        entropy_rows,
                        "intrinsicDimension",
                    ),
                    "meanCurvature": self._mean_metric_from_mapping(
                        flow_map,
                        "mean_curvature",
                    ),
                    "maxCurvature": self._max_metric_from_mapping(
                        flow_map,
                        "max_curvature",
                    ),
                    "meanGeodesicDeviation": self._mean_metric_from_mapping(
                        geodesic_rows,
                        "mean_deviation",
                    ),
                    "meanPathLengthRatio": self._mean_metric_from_mapping(
                        geodesic_rows,
                        "path_length_ratio",
                    ),
                    "peakLayer": peak_layer,
                    "firstBendLayer": first_bend_layer,
                }
            )

        return sequence_metrics, space_step_metrics, errors

    def _build_step_rows(
        self,
        *,
        mode: str,
        region: str,
        token_ids: tuple[int, ...],
        token_texts: tuple[str, ...],
        prompt_token_count: int,
        live_steps: tuple[LiveTraceStep, ...] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, token_id in enumerate(token_ids):
            if region == "prompt":
                global_index = index
            elif region in {"response", "generated"}:
                global_index = prompt_token_count + index
            else:
                global_index = index
            row = {
                "mode": mode,
                "region": region,
                "globalStepIndex": global_index,
                "regionStepIndex": index,
                "tokenId": token_id,
                "tokenText": token_texts[index] if index < len(token_texts) else f"<{token_id}>",
                "isPromptToken": region == "prompt" or (region == "full" and index < prompt_token_count),
                "isResponseToken": region in {"response", "generated"} or (region == "full" and index >= prompt_token_count),
            }
            if live_steps is not None and index < len(live_steps):
                row["logitEntropy"] = live_steps[index].logit_entropy
                row["logitMargin"] = live_steps[index].logit_margin
            rows.append(row)
        return rows

    def _build_space_step_rows(
        self,
        *,
        mode: str,
        region: str,
        space: str,
        layer_positions: dict[int, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rg = RiemannianGeometry(backend=self._backend)
        for layer_idx, positions in sorted(layer_positions.items()):
            positions_arr = self._backend.array(positions)
            token_count = self._first_dim(positions_arr)
            if token_count == 0:
                continue
            geodesic_result = None
            if token_count >= 2:
                geodesic_result = rg.geodesic_distances(positions_arr)

            for step_index in range(token_count):
                point = positions_arr[step_index]
                vector_norm = self._scalar_norm(point)
                euclidean_step = None
                geodesic_step = None
                step_deviation = None
                if step_index > 0:
                    prev_point = positions_arr[step_index - 1]
                    diff = point - prev_point
                    euclidean_step = self._scalar_norm(diff)
                    if geodesic_result is not None:
                        geodesic_step = float(
                            self._backend.to_scalar(
                                geodesic_result.distances[step_index - 1][step_index]
                            )
                        )
                    if euclidean_step is not None and euclidean_step > 0 and geodesic_step is not None:
                        step_deviation = (geodesic_step / euclidean_step) - 1.0
                rows.append(
                    {
                        "mode": mode,
                        "region": region,
                        "space": space,
                        "layer": layer_idx,
                        "stepIndex": step_index,
                        "vectorNorm": vector_norm,
                        "euclideanStepLength": euclidean_step,
                        "geodesicStepLength": geodesic_step,
                        "stepDeviation": step_deviation,
                    }
                )
        return rows

    def _normalize_spaces(self, spaces: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_space in spaces:
            space = str(raw_space).strip().lower()
            if not space:
                continue
            if space not in SUPPORTED_ANALYSIS_SPACES:
                raise ValueError(
                    "Unsupported analysis space "
                    f"{space!r}. Choose from {', '.join(SUPPORTED_ANALYSIS_SPACES)}."
                )
            if space not in seen:
                normalized.append(space)
                seen.add(space)
        if not normalized:
            return DEFAULT_ANALYSIS_SPACES
        return tuple(normalized)

    def _normalize_generated_text(self, *, prompt: str, generated_text: str) -> str:
        normalized = generated_text or ""
        if prompt and normalized.startswith(prompt):
            normalized = normalized[len(prompt) :]
        return normalized

    def _response_token_ids(
        self,
        *,
        tokenizer: Any,
        generated_text: str,
        live_result: LiveGenerationTraceResult | None,
    ) -> tuple[int, ...]:
        if live_result is not None and live_result.generated_token_ids:
            return live_result.generated_token_ids
        encoded = tuple(self._encode_tokens(tokenizer, generated_text))
        return self._strip_continuation_prefix(tokenizer, encoded)

    def _encode_tokens(self, tokenizer: Any, text: str) -> list[int]:
        if not text:
            return []
        token_ids = self._backend.encode_tokens(tokenizer, text)
        return list(token_ids)

    def _decode_tokens(self, tokenizer: Any, token_ids: tuple[int, ...]) -> str:
        if not token_ids:
            return ""
        return self._backend.decode_tokens(tokenizer, list(token_ids))

    def _strip_continuation_prefix(
        self,
        tokenizer: Any,
        token_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not token_ids:
            return token_ids
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(tokenizer, "eos_token_id", None)
        if bos_id is None:
            empty_ids = self._encode_tokens(tokenizer, "")
            bos_id = empty_ids[0] if empty_ids else None
        if bos_id is not None and token_ids[0] == int(bos_id):
            return token_ids[1:]
        return token_ids

    def _decode_token_texts(self, tokenizer: Any, token_ids: tuple[int, ...]) -> list[str]:
        texts: list[str] = []
        for token_id in token_ids:
            try:
                texts.append(self._decode_tokens(tokenizer, (token_id,)))
            except Exception:
                texts.append(f"<{token_id}>")
        return texts

    def _scalar_norm(self, value: Any) -> float:
        return float(self._backend.to_scalar(self._backend.norm(value)))

    @staticmethod
    def _first_dim(value: Any) -> int:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        return len(value) if hasattr(value, "__len__") else 0

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _mean_metric_from_mapping(mapping: dict[int, Any], attribute: str) -> float | None:
        values: list[float] = []
        for value in mapping.values():
            metric = getattr(value, attribute, None)
            if metric is not None:
                values.append(float(metric))
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _max_metric_from_mapping(mapping: dict[int, Any], attribute: str) -> float | None:
        values = [
            float(metric)
            for value in mapping.values()
            if (metric := getattr(value, attribute, None)) is not None
        ]
        if not values:
            return None
        return max(values)


def compute_first_divergence_step(
    baseline_token_ids: tuple[int, ...],
    variant_token_ids: tuple[int, ...],
) -> int | None:
    """Return the first step where two token streams diverge."""

    shared = min(len(baseline_token_ids), len(variant_token_ids))
    for index in range(shared):
        if baseline_token_ids[index] != variant_token_ids[index]:
            return index
    if len(baseline_token_ids) != len(variant_token_ids):
        return shared
    return None


def detect_grounded_label_onset(
    *,
    generated_token_ids: tuple[int, ...],
    allowed_label_token_ids: list[tuple[int, ...]],
) -> tuple[int | None, str | None]:
    """Detect the first token where the generated prefix leaves every allowed label."""

    surviving = list(allowed_label_token_ids)
    for index in range(len(generated_token_ids)):
        prefix = generated_token_ids[: index + 1]
        survivors = [
            label_ids
            for label_ids in surviving
            if len(prefix) <= len(label_ids) and prefix == label_ids[: len(prefix)]
        ]
        if survivors:
            if any(len(prefix) >= len(label_ids) and prefix[: len(label_ids)] == label_ids for label_ids in survivors):
                return None, "matched_allowed_label"
            surviving = survivors
            continue
        return index, "left_allowed_label_prefix"
    return None, None
