from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from modelcypher.core.use_cases.geometry_engine import GeometryEngine
from modelcypher.ports.backend import Backend


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


class RotationalMerger:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.geometry = GeometryEngine(backend)

    def merge(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        alpha: float = 0.5,
        anchor_mode: str = "procrustes",
    ) -> tuple[dict[str, Any], MergeAnalysisResult]:
        merged: dict[str, Any] = {}
        errors: list[float] = []
        roughness: list[float] = []
        metrics: list[LayerMergeMetric] = []
        layer_index = 0

        for key, target in target_weights.items():
            source = source_weights.get(key)
            if source is None:
                merged[key] = target
                continue
            if len(getattr(source, "shape", [])) == 2 and source.shape == target.shape:
                in_dim = source.shape[1]
                basis = self.backend.array(np.eye(in_dim, dtype=np.float32))
                result = self.geometry.orthogonal_procrustes(source, target, basis, basis)
                rotated = self.backend.matmul(source, result.omega)
                merged[key] = (alpha * target) + ((1 - alpha) * rotated)
                errors.append(result.error)
                deviation = float(np.linalg.norm(self.geometry._to_numpy(result.omega) - np.eye(in_dim)))
                roughness.append(deviation)
                condition_number = float(np.linalg.cond(self.geometry._to_numpy(result.omega)))
                metrics.append(
                    LayerMergeMetric(
                        layer_index=layer_index,
                        module_name=key,
                        module_kind="linear",
                        procrustes_error=result.error,
                        condition_number=condition_number,
                        rotation_deviation=deviation,
                        spectral_ratio=1.0,
                    )
                )
                layer_index += 1
            else:
                merged[key] = (alpha * target) + ((1 - alpha) * source)

        mean_error = float(np.mean(errors)) if errors else 0.0
        max_error = float(np.max(errors)) if errors else 0.0
        roughness_value = float(np.mean(roughness)) if roughness else 0.0

        analysis = MergeAnalysisResult(
            source_model="source",
            target_model="target",
            anchor_mode=anchor_mode,
            timestamp=datetime.utcnow(),
            mean_procrustes_error=mean_error,
            max_procrustes_error=max_error,
            rotation_field_roughness=roughness_value,
            anchor_coverage=len(errors),
            layer_metrics=metrics,
        )
        return merged, analysis
