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

"""Service layer for LoRA adapter geometry analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.ports.adapter_weights import AdapterWeightsLoader

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

_PROJECTION_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass(frozen=True)
class AdapterAnalysisRequest:
    """Request for adapter geometry analysis."""

    adapter_path: str
    base_model: str | None = None
    baseline_artifact: str | None = None


@dataclass(frozen=True)
class AdapterAnalysisResult:
    """Response payload for adapter geometry analysis."""

    adapter_name: str
    base_model: str
    lora_rank: Any
    lora_scale: Any
    n_layers: int
    metrics: dict[str, Any]
    reference_comparison: dict[str, float] | None
    reference_baseline: dict[str, Any] | None
    layers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to CLI-friendly dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "lora_scale": self.lora_scale,
            "n_layers": self.n_layers,
            "metrics": self.metrics,
            "reference_comparison": self.reference_comparison,
            "reference_baseline": self.reference_baseline,
            "layers": self.layers,
        }


class AdapterAnalysisService:
    """Analyze LoRA adapter geometry against base model weights."""

    def __init__(
        self,
        backend: "Backend",
        weights_loader: AdapterWeightsLoader | None = None,
        baseline_loader: Callable[[str | None], dict[str, Any] | None] | None = None,
    ) -> None:
        self._backend = backend
        self._weights_loader = weights_loader or _BackendAdapterWeightsLoader()
        self._baseline_loader = baseline_loader

    def analyze(self, request: AdapterAnalysisRequest) -> AdapterAnalysisResult:
        """Run adapter geometry analysis and return raw measurements."""
        from modelcypher.experimental.lora_geometry.measurements import (
            collect_layer_measurements,
        )

        adapter_file = self._resolve_adapter_file(request.adapter_path)
        adapter_dir = adapter_file.parent
        base_model_path = self._resolve_base_model_path(adapter_dir, request.base_model)

        adapter_weights = self._weights_loader.load(adapter_file, self._backend)
        if adapter_weights:
            self._backend.eval(*adapter_weights.values())

        lora_rank, lora_scale = self._read_lora_config(adapter_dir)

        base_weights = self._load_base_weights(base_model_path, adapter_weights)

        measurements = []
        layer_details: list[dict[str, Any]] = []

        for adapter_key in sorted(adapter_weights.keys()):
            if "lora_a" not in adapter_key.lower():
                continue

            is_lower = "lora_a" in adapter_key
            lora_a_suffix = ".lora_a" if is_lower else ".lora_A"
            lora_b_suffix = ".lora_b" if is_lower else ".lora_B"
            lora_b_key = adapter_key.replace(lora_a_suffix, lora_b_suffix)
            if lora_b_key not in adapter_weights:
                continue

            base_key = adapter_key.replace(lora_a_suffix, ".weight")
            if base_key not in base_weights:
                continue

            a_matrix = adapter_weights[adapter_key]
            b_matrix = adapter_weights[lora_b_key]

            if a_matrix.shape[1] != b_matrix.shape[0]:
                continue

            delta_raw = self._backend.matmul(a_matrix, b_matrix)
            delta_w = self._backend.transpose(delta_raw)
            base_w = base_weights[base_key]
            self._backend.eval(delta_w, base_w)

            layer_idx, proj_name = _parse_layer_info(adapter_key)

            try:
                measurement = collect_layer_measurements(
                    weight_original=base_w,
                    delta_w=delta_w,
                    layer_idx=layer_idx,
                    projection_name=proj_name,
                    backend=self._backend,
                )
            except Exception as exc:
                logger.warning(
                    "Failed adapter layer measurement for %s (base=%s): %s",
                    adapter_key,
                    base_key,
                    exc,
                )
                continue

            measurements.append(measurement)
            layer_details.append(
                {
                    "layer": layer_idx,
                    "projection": proj_name,
                    "amplification_cv": float(measurement.amplification_cv),
                    "weyl_utilization": float(measurement.weyl_utilization),
                    "delta_frobenius_norm": float(measurement.delta_frobenius_norm),
                    "delta_spectral_norm": float(measurement.delta_spectral_norm),
                }
            )

        if not measurements:
            raise ValueError(
                "No measurements could be collected. Check adapter/base model compatibility."
            )

        cvs = [measurement.amplification_cv for measurement in measurements]
        weyls = [measurement.weyl_utilization for measurement in measurements]
        frobs = [measurement.delta_frobenius_norm for measurement in measurements]
        specs = [measurement.delta_spectral_norm for measurement in measurements]
        mean_cv = float(sum(cvs) / len(cvs))
        mean_weyl = float(sum(weyls) / len(weyls))

        baseline = (
            self._baseline_loader(request.baseline_artifact)
            if self._baseline_loader is not None
            else None
        )
        reference_comparison: dict[str, float] | None = None
        if baseline is not None:
            reference_comparison = {
                "amplification_cv_vs_random_baseline": float(
                    mean_cv / baseline["amplification_cv"]
                ),
                "weyl_utilization_vs_random_baseline": float(
                    mean_weyl / baseline["weyl_utilization"]
                ),
            }

        return AdapterAnalysisResult(
            adapter_name=adapter_dir.name,
            base_model=base_model_path.name,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
            n_layers=len(measurements),
            metrics={
                "amplification_cv": {
                    "mean": mean_cv,
                    "min": float(min(cvs)),
                    "max": float(max(cvs)),
                },
                "weyl_utilization": {
                    "mean": mean_weyl,
                    "min": float(min(weyls)),
                    "max": float(max(weyls)),
                },
                "delta_frobenius_norm": {
                    "total": float(sum(frobs)),
                    "mean": float(sum(frobs) / len(frobs)),
                },
                "delta_spectral_norm": {
                    "mean": float(sum(specs) / len(specs)),
                    "max": float(max(specs)),
                },
            },
            reference_comparison=reference_comparison,
            reference_baseline=baseline,
            layers=layer_details,
        )

    def _resolve_adapter_file(self, adapter_path: str) -> Path:
        adapter_file = Path(adapter_path).expanduser().resolve()
        if adapter_file.is_dir():
            adapter_file = adapter_file / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_file}")
        return adapter_file

    def _resolve_base_model_path(self, adapter_dir: Path, base_model: str | None) -> Path:
        resolved_base_model = base_model
        if resolved_base_model is None:
            config_path = adapter_dir / "adapter_config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                resolved_base_model = config.get("model")

            if resolved_base_model is None:
                raise ValueError(
                    "Base model not specified and not found in adapter_config.json. "
                    "Use --base-model to specify."
                )

        base_model_path = Path(resolved_base_model).expanduser().resolve()
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        return base_model_path

    def _read_lora_config(self, adapter_dir: Path) -> tuple[Any, Any]:
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            return "unknown", "unknown"

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        lora_params = config.get("lora_parameters", {})
        lora_rank = lora_params.get("rank", config.get("r", "unknown"))
        lora_scale = lora_params.get("scale", config.get("lora_alpha", "unknown"))
        return lora_rank, lora_scale

    def _load_base_weights(
        self,
        base_model_path: Path,
        adapter_weights: dict[str, Any],
    ) -> dict[str, Any]:
        target_keys = set()
        for key in adapter_weights.keys():
            if "lora_a" in key.lower():
                is_lower = "lora_a" in key
                lora_a_suffix = ".lora_a" if is_lower else ".lora_A"
                target_keys.add(key.replace(lora_a_suffix, ".weight"))

        return _load_safetensors_from_model_dir(
            base_model_path,
            self._backend,
            required_keys=target_keys,
            weights_loader=self._weights_loader,
        )


class _BackendAdapterWeightsLoader(AdapterWeightsLoader):
    """Port-only adapter loader that delegates to backend-native formats."""

    def load(self, weights_path: Path, backend: "Backend") -> dict[str, Any]:
        suffix = weights_path.suffix.lower()
        if suffix == ".safetensors":
            return backend.load_safetensors(str(weights_path))
        if suffix in (".bin", ".pt"):
            return backend.load_binary_weights(str(weights_path))
        raise ValueError(f"Unsupported adapter weights format: {weights_path}")


def _load_weights_from_paths(
    paths: list[Path],
    backend: "Backend",
    weights_loader: AdapterWeightsLoader,
) -> dict[str, Any]:
    """Load and merge weights from a list of files."""
    weights: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        weights.update(weights_loader.load(path, backend))
    if weights:
        backend.eval(*weights.values())
    return weights


def _load_safetensors_from_model_dir(
    model_dir: Path,
    backend: "Backend",
    required_keys: set[str] | None,
    weights_loader: AdapterWeightsLoader,
) -> dict[str, Any]:
    """Load model safetensors from single/sharded files in a model directory."""
    model_dir = model_dir.expanduser().resolve()
    index_file = model_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if required_keys is None:
            shard_files = sorted(set(weight_map.values()))
        else:
            shard_files = sorted(
                {weight_map[key] for key in required_keys if key in weight_map}
            )
        shard_paths = [model_dir / shard for shard in shard_files]
        weights = _load_weights_from_paths(shard_paths, backend, weights_loader)
    else:
        safetensors_paths = sorted(model_dir.glob("*.safetensors"))
        weights = _load_weights_from_paths(safetensors_paths, backend, weights_loader)

    if required_keys is None:
        return weights
    return {key: tensor for key, tensor in weights.items() if key in required_keys}


def _parse_layer_info(adapter_key: str) -> tuple[int, str]:
    layer_idx = -1
    proj_name = "unknown"
    parts = adapter_key.split(".")

    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                layer_idx = int(parts[idx + 1])
            except ValueError:
                layer_idx = -1
        if part in _PROJECTION_NAMES:
            proj_name = part

    return layer_idx, proj_name


__all__ = [
    "AdapterAnalysisRequest",
    "AdapterAnalysisResult",
    "AdapterAnalysisService",
]
