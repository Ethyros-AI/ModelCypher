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

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import svd_rank_threshold
from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader


_LAYER_RE = re.compile(r"\.([0-9]+)\.")


@dataclass(frozen=True)
class LayerSVDReport:
    layer_idx: int
    weight_name: str
    shape: tuple[int, ...]
    rank_before: int
    rank_after: int
    rank_delta: int
    null_space_component: float
    subspace_overlap: float
    relative_change: float
    frobenius_delta: float


@dataclass(frozen=True)
class PositiveGeometryReport:
    layer_idx: int
    positive_minors_before: int
    positive_minors_after: int
    positive_minors_delta: int


@dataclass
class LoRADiagnosticReport:
    model_path: str
    adapter_path: str
    total_layers: int
    layers_with_lora: int
    total_params_modified: int
    avg_null_space_activation: float
    avg_subspace_overlap: float
    avg_relative_change: float
    peak_change_layer: int | None
    layer_svd: list[LayerSVDReport] = field(default_factory=list)
    positive_geometry: list[PositiveGeometryReport] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "LoRA Geometry Diagnostic",
            f"Model: {self.model_path}",
            f"Adapter: {self.adapter_path}",
            f"Layers with LoRA: {self.layers_with_lora}",
            f"Total params modified: {self.total_params_modified}",
        ]
        if self.peak_change_layer is not None:
            lines.append(f"Peak change layer: {self.peak_change_layer}")
        lines.append(
            f"Avg null-space activation: {self.avg_null_space_activation}"
        )
        lines.append(
            f"Avg subspace overlap: {self.avg_subspace_overlap}"
        )
        lines.append(
            f"Avg relative change: {self.avg_relative_change}"
        )
        return "\n".join(lines)


def run_diagnostic(
    model_path: str,
    adapter_path: str,
    target_layers: list[int] | None = None,
) -> LoRADiagnosticReport:
    backend = get_default_backend()
    weights = _load_adapter_weights(Path(adapter_path), backend)

    layer_reports: list[LayerSVDReport] = []
    total_params = 0
    peak_layer = None
    peak_change = -math.inf

    for name, tensor in weights.items():
        layer_idx = _infer_layer_idx(name)
        if target_layers is not None and layer_idx not in target_layers:
            continue

        shape = tuple(int(dim) for dim in backend.shape(tensor))
        total_params += _numel(shape)
        rank_after = _compute_rank(tensor, backend)
        rank_before = 0
        rank_delta = rank_after - rank_before
        frob = _frobenius_norm(tensor, backend)

        if frob > peak_change:
            peak_change = frob
            peak_layer = layer_idx

        layer_reports.append(
            LayerSVDReport(
                layer_idx=layer_idx,
                weight_name=name,
                shape=shape,
                rank_before=rank_before,
                rank_after=rank_after,
                rank_delta=rank_delta,
                null_space_component=float("nan"),
                subspace_overlap=float("nan"),
                relative_change=float("nan"),
                frobenius_delta=frob,
            )
        )

    avg_null = _mean_or_nan([r.null_space_component for r in layer_reports])
    avg_overlap = _mean_or_nan([r.subspace_overlap for r in layer_reports])
    avg_relative = _mean_or_nan([r.relative_change for r in layer_reports])

    return LoRADiagnosticReport(
        model_path=model_path,
        adapter_path=adapter_path,
        total_layers=len({r.layer_idx for r in layer_reports if r.layer_idx >= 0}),
        layers_with_lora=len(layer_reports),
        total_params_modified=total_params,
        avg_null_space_activation=avg_null,
        avg_subspace_overlap=avg_overlap,
        avg_relative_change=avg_relative,
        peak_change_layer=peak_layer,
        layer_svd=layer_reports,
        positive_geometry=[],
    )


def _load_adapter_weights(adapter_path: Path, backend: Any) -> dict[str, Any]:
    loader = AutoAdapterWeightsLoader()
    weights: dict[str, Any] = {}
    for weight_file in adapter_path.glob("*.safetensors"):
        weights.update(loader.load(weight_file, backend))
    for weight_file in adapter_path.glob("*.bin"):
        weights.update(loader.load(weight_file, backend))
    if not weights:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
    backend.eval(*weights.values())
    return weights


def _infer_layer_idx(weight_name: str) -> int:
    match = _LAYER_RE.search(weight_name)
    if not match:
        return -1
    try:
        return int(match.group(1))
    except ValueError:
        return -1


def _compute_rank(tensor: Any, backend: Any) -> int:
    if len(backend.shape(tensor)) < 2:
        return 1
    _, s, _ = backend.svd(tensor)
    backend.eval(s)
    max_dim = max(int(dim) for dim in backend.shape(tensor))
    threshold = svd_rank_threshold(backend, s, max_dim)
    count = 0
    for value in backend.tolist(s):
        if float(value) > threshold:
            count += 1
    return count


def _frobenius_norm(tensor: Any, backend: Any) -> float:
    """Compute Frobenius norm using backend.norm()."""
    norm_arr = backend.norm(tensor)
    backend.eval(norm_arr)
    return float(backend.to_scalar(norm_arr))


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


__all__ = [
    "LayerSVDReport",
    "PositiveGeometryReport",
    "LoRADiagnosticReport",
    "run_diagnostic",
]
