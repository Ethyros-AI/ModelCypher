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

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.spectral_capacity import (
    LayerCapacityReport,
    SpectralCapacityAnalyzer,
)

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
    ) -> ModelCapacityReport:
        model_path_resolved = str(Path(model_path).expanduser().resolve())
        weights = self._model_loader.load_weights(model_path_resolved)
        normalized_targets = _normalize_target_modules(target_modules)
        _validate_dim_filters(min_dim=min_dim, max_dim=max_dim)

        total_parameters = _count_total_parameters(weights)
        layer_reports: list[LayerCapacityReport] = []
        failed_layers: dict[str, str] = {}
        analyzed_parameters = 0

        for layer_name in sorted(weights.keys()):
            tensor = weights[layer_name]
            shape = getattr(tensor, "shape", None)
            if shape is None or len(shape) != 2:
                continue
            if not _matches_target_modules(layer_name, normalized_targets):
                continue
            layer_min_dim = min(int(shape[0]), int(shape[1]))
            if min_dim is not None and layer_min_dim < min_dim:
                continue
            if max_dim is not None and layer_min_dim > max_dim:
                continue

            analyzed_parameters += int(shape[0]) * int(shape[1])
            try:
                report = self._analyzer.analyze(layer_name=layer_name, weight=tensor)
                layer_reports.append(report)
            except Exception as exc:
                failed_layers[layer_name] = str(exc)
                logger.warning("Capacity analysis skipped layer %s: %s", layer_name, exc)

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


def _count_total_parameters(weights: dict[str, object]) -> int:
    total = 0
    for tensor in weights.values():
        shape = getattr(tensor, "shape", None)
        if shape is None:
            continue
        params = 1
        for dim in shape:
            params *= int(dim)
        total += params
    return total


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


__all__ = [
    "CapacityAnalysisService",
    "ModelCapacityReport",
]
