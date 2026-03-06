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

"""Use case service for per-layer spectral capacity analysis."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.spectral_capacity import (
    EnergyFractions,
    LayerCapacityReport,
    SpectralCapacityAnalyzer,
    SpectralDecayType,
)
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapacityReport:
    model_path: str
    model_name: str
    total_parameters: int
    analyzed_parameters: int
    analyzed_layers: int
    mean_effective_rank: float
    mean_capacity_utilization: float
    reference_rank_dimension: int
    layer_reports: list[LayerCapacityReport]
    failed_layers: dict[str, str]
    target_modules: list[str]
    min_dim: int | None
    max_dim: int | None

    def layers_by_null_space_fraction(self) -> list[LayerCapacityReport]:
        return sorted(
            self.layer_reports,
            key=lambda report: (
                report.null_space_fraction,
                report.null_space_dim_f32,
                -report.capacity_utilization,
            ),
            reverse=True,
        )

    def layers_by_effective_rank(self) -> list[LayerCapacityReport]:
        return sorted(
            self.layer_reports,
            key=lambda report: (
                report.effective_rank,
                report.capacity_utilization,
                report.spectral_gap_at_rank,
            ),
            reverse=True,
        )

    def layers_by_recommended_rank(self) -> list[LayerCapacityReport]:
        return sorted(
            self.layer_reports,
            key=lambda report: (
                report.recommended_rank,
                report.spectral_gap_at_rank,
                report.null_space_fraction,
            ),
            reverse=True,
        )

    def sorted_layers(self, sort_by: str) -> list[LayerCapacityReport]:
        if sort_by == "null":
            return self.layers_by_null_space_fraction()
        if sort_by == "effective-rank":
            return self.layers_by_effective_rank()
        if sort_by == "recommended-rank":
            return self.layers_by_recommended_rank()
        raise ValueError(
            "Unsupported sort_by value. "
            "Use one of: null, effective-rank, recommended-rank."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "modelPath": self.model_path,
            "modelName": self.model_name,
            "filters": {
                "targetModules": list(self.target_modules),
                "minDim": self.min_dim,
                "maxDim": self.max_dim,
            },
            "summary": {
                "totalParameters": self.total_parameters,
                "analyzedParameters": self.analyzed_parameters,
                "analyzedLayers": self.analyzed_layers,
                "meanEffectiveRank": self.mean_effective_rank,
                "meanCapacityUtilization": self.mean_capacity_utilization,
                "referenceRankDimension": self.reference_rank_dimension,
            },
            "layers": [report.to_dict() for report in self.layer_reports],
            "layersByNullSpaceFraction": [
                report.to_dict() for report in self.layers_by_null_space_fraction()
            ],
            "layersByRecommendedRank": [
                report.to_dict() for report in self.layers_by_recommended_rank()
            ],
            "failedLayers": dict(self.failed_layers),
        }


class CapacityAnalysisService:
    """Analyze weight-space spectral capacity for all 2D model tensors."""

    def __init__(
        self,
        backend: "Backend",
        model_loader: "ModelLoaderPort",
    ) -> None:
        self._backend = backend
        self._model_loader = model_loader
        self._analyzer = SpectralCapacityAnalyzer(backend)

    def analyze(
        self,
        model_path: str,
        target_modules: list[str] | None = None,
        min_dim: int | None = None,
        max_dim: int | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> ModelCapacityReport:
        model_path_resolved = str(Path(model_path).expanduser().resolve())
        normalized_targets = _normalize_target_modules(target_modules)
        _validate_dim_filters(min_dim=min_dim, max_dim=max_dim)

        checkpoint_file = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else None
        )
        state = _load_checkpoint_state(
            checkpoint_file,
            resume=resume,
            model_path=model_path_resolved,
            target_modules=normalized_targets,
            min_dim=min_dim,
            max_dim=max_dim,
        )

        total_parameters = state.total_parameters
        analyzed_parameters = state.analyzed_parameters
        layer_reports = state.layer_reports
        failed_layers = state.failed_layers
        processed_layers = state.processed_layers

        # Pre-load all weights so dequantize_if_needed() can look up
        # scales/biases by key for quantized models.
        # If loading is interrupted, analyze whatever was loaded so far
        # and checkpoint progress before re-raising.
        all_params: dict[str, object] = {}
        _preload_exc: BaseException | None = None
        try:
            for name, tensor in self._iter_weight_items(model_path_resolved):
                all_params[name] = tensor
        except Exception as exc:
            _preload_exc = exc

        for layer_name in sorted(all_params.keys()):
            if layer_name in processed_layers:
                continue

            # Skip quantization metadata tensors — they are not model weights.
            if layer_name.endswith(".scales") or layer_name.endswith(".biases"):
                processed_layers.add(layer_name)
                continue

            tensor = all_params[layer_name]

            # Dequantize packed-int weight tensors before analysis.
            if layer_name.endswith(".weight"):
                tensor = dequantize_if_needed(
                    tensor, layer_name, all_params, self._backend
                )

            shape = getattr(tensor, "shape", None)
            param_count = _tensor_parameter_count(tensor)
            total_parameters += param_count

            if shape is None or len(shape) != 2:
                processed_layers.add(layer_name)
                _write_checkpoint_state(
                    checkpoint_file,
                    model_path=model_path_resolved,
                    target_modules=normalized_targets,
                    min_dim=min_dim,
                    max_dim=max_dim,
                    total_parameters=total_parameters,
                    analyzed_parameters=analyzed_parameters,
                    layer_reports=layer_reports,
                    failed_layers=failed_layers,
                    processed_layers=processed_layers,
                )
                continue
            if not _matches_target_modules(layer_name, normalized_targets):
                processed_layers.add(layer_name)
                _write_checkpoint_state(
                    checkpoint_file,
                    model_path=model_path_resolved,
                    target_modules=normalized_targets,
                    min_dim=min_dim,
                    max_dim=max_dim,
                    total_parameters=total_parameters,
                    analyzed_parameters=analyzed_parameters,
                    layer_reports=layer_reports,
                    failed_layers=failed_layers,
                    processed_layers=processed_layers,
                )
                continue
            layer_min_dim = min(int(shape[0]), int(shape[1]))
            if min_dim is not None and layer_min_dim < min_dim:
                processed_layers.add(layer_name)
                _write_checkpoint_state(
                    checkpoint_file,
                    model_path=model_path_resolved,
                    target_modules=normalized_targets,
                    min_dim=min_dim,
                    max_dim=max_dim,
                    total_parameters=total_parameters,
                    analyzed_parameters=analyzed_parameters,
                    layer_reports=layer_reports,
                    failed_layers=failed_layers,
                    processed_layers=processed_layers,
                )
                continue
            if max_dim is not None and layer_min_dim > max_dim:
                processed_layers.add(layer_name)
                _write_checkpoint_state(
                    checkpoint_file,
                    model_path=model_path_resolved,
                    target_modules=normalized_targets,
                    min_dim=min_dim,
                    max_dim=max_dim,
                    total_parameters=total_parameters,
                    analyzed_parameters=analyzed_parameters,
                    layer_reports=layer_reports,
                    failed_layers=failed_layers,
                    processed_layers=processed_layers,
                )
                continue

            analyzed_parameters += int(shape[0]) * int(shape[1])
            try:
                report = self._analyzer.analyze(layer_name=layer_name, weight=tensor)
                layer_reports.append(report)
            except Exception as exc:
                failed_layers[layer_name] = str(exc)
                logger.warning("Capacity analysis skipped layer %s: %s", layer_name, exc)

            processed_layers.add(layer_name)
            _write_checkpoint_state(
                checkpoint_file,
                model_path=model_path_resolved,
                target_modules=normalized_targets,
                min_dim=min_dim,
                max_dim=max_dim,
                total_parameters=total_parameters,
                analyzed_parameters=analyzed_parameters,
                layer_reports=layer_reports,
                failed_layers=failed_layers,
                processed_layers=processed_layers,
            )

        # If pre-loading was interrupted, checkpoint whatever was analyzed
        # so the next run can resume, then re-raise the original exception.
        if _preload_exc is not None:
            _write_checkpoint_state(
                checkpoint_file,
                model_path=model_path_resolved,
                target_modules=normalized_targets,
                min_dim=min_dim,
                max_dim=max_dim,
                total_parameters=total_parameters,
                analyzed_parameters=analyzed_parameters,
                layer_reports=layer_reports,
                failed_layers=failed_layers,
                processed_layers=processed_layers,
            )
            raise _preload_exc

        if not layer_reports:
            filter_desc = _filter_description(
                target_modules=normalized_targets,
                min_dim=min_dim,
                max_dim=max_dim,
            )
            raise ValueError("No analyzable 2D weight matrices matched filters." + filter_desc)

        mean_effective_rank = sum(r.effective_rank for r in layer_reports) / len(layer_reports)
        mean_capacity_utilization = (
            sum(r.capacity_utilization for r in layer_reports) / len(layer_reports)
        )

        min_dims = [min(report.weight_shape) for report in layer_reports]
        reference_rank_dimension = _most_common_dimension(min_dims)

        return ModelCapacityReport(
            model_path=model_path_resolved,
            model_name=Path(model_path_resolved).name,
            total_parameters=total_parameters,
            analyzed_parameters=analyzed_parameters,
            analyzed_layers=len(layer_reports),
            mean_effective_rank=mean_effective_rank,
            mean_capacity_utilization=mean_capacity_utilization,
            reference_rank_dimension=reference_rank_dimension,
            layer_reports=layer_reports,
            failed_layers=failed_layers,
            target_modules=normalized_targets,
            min_dim=min_dim,
            max_dim=max_dim,
        )

    def _iter_weight_items(self, model_path: str):
        """Yield deterministic (name, tensor) pairs from the model loader."""
        iter_weights = getattr(self._model_loader, "iter_weights", None)
        if callable(iter_weights):
            yield from iter_weights(model_path)
            return

        weights = self._model_loader.load_weights(model_path)
        for name in sorted(weights.keys()):
            yield name, weights[name]


def _count_total_parameters(weights: dict[str, object]) -> int:
    total = 0
    for tensor in weights.values():
        total += _tensor_parameter_count(tensor)
    return total


def _tensor_parameter_count(tensor: object) -> int:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return 0
    params = 1
    for dim in shape:
        params *= int(dim)
    return params


def _most_common_dimension(dimensions: list[int]) -> int:
    if not dimensions:
        return 0
    counts = Counter(dimensions)
    max_count = max(counts.values())
    most_common_dims = [dim for dim, count in counts.items() if count == max_count]
    return max(most_common_dims)


def _normalize_target_modules(target_modules: list[str] | None) -> list[str]:
    if not target_modules:
        return []
    normalized: list[str] = []
    for raw_value in target_modules:
        for token in raw_value.split(","):
            stripped = token.strip()
            if stripped:
                normalized.append(stripped)
    return sorted(set(normalized))


def _matches_target_modules(layer_name: str, target_modules: list[str]) -> bool:
    if not target_modules:
        return True
    return any(token in layer_name for token in target_modules)


def _validate_dim_filters(min_dim: int | None, max_dim: int | None) -> None:
    if min_dim is not None and min_dim <= 0:
        raise ValueError("min_dim must be > 0 when provided.")
    if max_dim is not None and max_dim <= 0:
        raise ValueError("max_dim must be > 0 when provided.")
    if min_dim is not None and max_dim is not None and min_dim > max_dim:
        raise ValueError("min_dim must be <= max_dim.")


def _filter_description(
    target_modules: list[str],
    min_dim: int | None,
    max_dim: int | None,
) -> str:
    parts: list[str] = []
    if target_modules:
        parts.append(f"target_modules={target_modules}")
    if min_dim is not None:
        parts.append(f"min_dim={min_dim}")
    if max_dim is not None:
        parts.append(f"max_dim={max_dim}")
    if not parts:
        return ""
    return f" (filters: {', '.join(parts)})"


@dataclass
class _CheckpointState:
    total_parameters: int
    analyzed_parameters: int
    layer_reports: list[LayerCapacityReport]
    failed_layers: dict[str, str]
    processed_layers: set[str]


def _load_checkpoint_state(
    checkpoint_path: Path | None,
    *,
    resume: bool,
    model_path: str,
    target_modules: list[str],
    min_dim: int | None,
    max_dim: int | None,
) -> _CheckpointState:
    if checkpoint_path is None or not resume or not checkpoint_path.exists():
        return _CheckpointState(
            total_parameters=0,
            analyzed_parameters=0,
            layer_reports=[],
            failed_layers={},
            processed_layers=set(),
        )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_filters = {
        "target_modules": list(target_modules),
        "min_dim": min_dim,
        "max_dim": max_dim,
    }
    if payload.get("model_path") != model_path:
        raise ValueError(
            "Checkpoint model path mismatch. "
            f"expected={model_path}, got={payload.get('model_path')}"
        )
    if payload.get("filters") != expected_filters:
        raise ValueError(
            "Checkpoint filter mismatch. "
            f"expected={expected_filters}, got={payload.get('filters')}"
        )

    layer_reports_raw = payload.get("layer_reports", [])
    layer_reports = [
        _layer_report_from_dict(layer_dict)
        for layer_dict in layer_reports_raw
    ]
    failed_layers = {
        str(layer): str(reason)
        for layer, reason in dict(payload.get("failed_layers", {})).items()
    }
    processed_layers = {str(name) for name in payload.get("processed_layers", [])}

    return _CheckpointState(
        total_parameters=int(payload.get("total_parameters", 0)),
        analyzed_parameters=int(payload.get("analyzed_parameters", 0)),
        layer_reports=layer_reports,
        failed_layers=failed_layers,
        processed_layers=processed_layers,
    )


def _write_checkpoint_state(
    checkpoint_path: Path | None,
    *,
    model_path: str,
    target_modules: list[str],
    min_dim: int | None,
    max_dim: int | None,
    total_parameters: int,
    analyzed_parameters: int,
    layer_reports: list[LayerCapacityReport],
    failed_layers: dict[str, str],
    processed_layers: set[str],
) -> None:
    if checkpoint_path is None:
        return

    payload = {
        "version": 1,
        "model_path": model_path,
        "filters": {
            "target_modules": list(target_modules),
            "min_dim": min_dim,
            "max_dim": max_dim,
        },
        "total_parameters": total_parameters,
        "analyzed_parameters": analyzed_parameters,
        "processed_layers": sorted(processed_layers),
        "layer_reports": [report.to_dict() for report in layer_reports],
        "failed_layers": dict(failed_layers),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _layer_report_from_dict(raw: dict[str, object]) -> LayerCapacityReport:
    decay_value = raw.get("decayType")
    decay_type = (
        SpectralDecayType(str(decay_value))
        if isinstance(decay_value, str)
        else None
    )

    energy_raw = raw.get("energyFractions")
    energy = None
    if isinstance(energy_raw, dict):
        energy = EnergyFractions(
            top_10pct=float(energy_raw.get("top10pct", 0.0)),
            top_20pct=float(energy_raw.get("top20pct", 0.0)),
            top_50pct=float(energy_raw.get("top50pct", 0.0)),
        )

    shape_raw = raw.get("weightShape", [0, 0])
    if (
        not isinstance(shape_raw, list)
        or len(shape_raw) != 2
    ):
        raise ValueError(f"Invalid checkpoint weightShape: {shape_raw}")
    weight_shape = (int(shape_raw[0]), int(shape_raw[1]))

    singular_values_raw = raw.get("singularValues", [])
    if not isinstance(singular_values_raw, list):
        raise ValueError("Invalid checkpoint singularValues")
    singular_values = [float(value) for value in singular_values_raw]

    return LayerCapacityReport(
        layer_name=str(raw["layerName"]),
        weight_shape=weight_shape,
        singular_values=singular_values,
        spectral_norm=float(raw["spectralNorm"]),
        nuclear_norm=float(raw["nuclearNorm"]),
        frobenius_norm=float(raw["frobeniusNorm"]),
        effective_rank=float(raw["effectiveRank"]),
        stable_rank=float(raw["stableRank"]),
        numerical_rank_f32=int(raw["numericalRankF32"]),
        numerical_rank_f16=int(raw["numericalRankF16"]),
        null_space_dim_f32=int(raw["nullSpaceDimF32"]),
        null_space_fraction=float(raw["nullSpaceFraction"]),
        recommended_rank=int(raw["recommendedRank"]),
        spectral_gap_at_rank=float(raw["spectralGapAtRank"]),
        capacity_utilization=float(raw["capacityUtilization"]),
        computation_method=str(raw["computationMethod"]),
        decay_type=decay_type,
        energy_fractions=energy,
        power_law_exponent=_optional_float(raw.get("powerLawExponent")),
        power_law_r_squared=_optional_float(raw.get("powerLawRSquared")),
        shannon_effective_rank=_optional_float(raw.get("shannonEffectiveRank")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "CapacityAnalysisService",
    "ModelCapacityReport",
]
