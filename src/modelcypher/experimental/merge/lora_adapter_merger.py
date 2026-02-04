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

"""LoRA adapter merge utilities (geometry-preserving)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.backend_matrix_utils import BackendMatrixUtils
from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
from modelcypher.experimental.merge.exceptions import MergeError
from modelcypher.ports.adapter_weights import AdapterWeightsLoader

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class AdapterPayload:
    directory: Path
    base_model_id: str
    rank: int
    scale: float
    weights: dict[str, "Array"]
    module_keys: list[str]


@dataclass(frozen=True)
class MergeReport:
    output_directory: str
    adapter_count: int
    base_model_id: str
    rank: int
    scale: float
    mean_procrustes_error: float
    total_merged_parameters: int
    layer_count: int
    mean_merge_cka: float


class LoRAAdapterMerger:
    """Merge LoRA adapters using geometric alignment."""

    @staticmethod
    def merge(
        adapter_directories: list[Path],
        output_directory: Path,
        backend: "Backend | None" = None,
        weights_loader: AdapterWeightsLoader | None = None,
    ) -> MergeReport:
        if len(adapter_directories) < 2:
            raise MergeError("At least two adapters are required for merge")

        b = backend or get_default_backend()
        adapters = [
            LoRAAdapterMerger._load_adapter(path, backend=b, weights_loader=weights_loader)
            for path in adapter_directories
        ]

        merged_parameters = 0
        errors: list[float] = []
        cka_values: list[float] = []
        layer_indices: set[int] = set()

        for key in adapters[0].module_keys:
            matrices = [adapter.weights[key] for adapter in adapters if key in adapter.weights]
            merged, error, cka = LoRAAdapterMerger._geometric_merge_matrices(matrices, b)
            merged_parameters += int(merged.size)
            errors.append(error)
            cka_values.append(cka)
            layer_index = LoRAAdapterMerger._extract_layer_index(key)
            if layer_index is not None:
                layer_indices.add(layer_index)

        mean_error = sum(errors) / float(len(errors)) if errors else 0.0
        mean_cka = sum(cka_values) / float(len(cka_values)) if cka_values else 0.0

        base_model_id = adapters[0].base_model_id
        rank = adapters[0].rank
        scale = adapters[0].scale

        return MergeReport(
            output_directory=str(output_directory),
            adapter_count=len(adapters),
            base_model_id=base_model_id,
            rank=rank,
            scale=scale,
            mean_procrustes_error=float(mean_error),
            total_merged_parameters=int(merged_parameters),
            layer_count=len(layer_indices),
            mean_merge_cka=float(mean_cka),
        )

    @staticmethod
    def _load_adapter(
        directory: Path,
        backend: "Backend | None" = None,
        weights_loader: AdapterWeightsLoader | None = None,
    ) -> AdapterPayload:
        if not directory.exists():
            raise MergeError(f"Adapter directory not found: {directory}")

        config_path = directory / "adapter_config.json"
        if not config_path.exists():
            raise MergeError("Missing adapter config")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        weights_path = None
        for candidate in ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"]:
            path = directory / candidate
            if path.exists():
                weights_path = path
                break

        if weights_path is None:
            raise MergeError("Missing adapter weights")

        backend = backend or get_default_backend()
        weights: dict[str, "Array"] = {}
        module_keys: list[str] = []

        # Actually load the weights from the file
        if weights_path.suffix == ".safetensors":
            raw_weights = backend.load_safetensors(str(weights_path))
            for key, value in raw_weights.items():
                weights[key] = backend.array(value)
                module_keys.append(key)
        elif weights_path.suffix in (".bin", ".pt"):
            if weights_loader is None:
                raise MergeError(
                    "Adapter weights loader required for .bin/.pt files. "
                    "Provide a loader port or convert to .safetensors."
                )
            raw_weights = weights_loader.load(weights_path, backend)
            for key, value in raw_weights.items():
                weights[key] = backend.array(value)
                module_keys.append(key)

        if not weights:
            raise MergeError(f"No weights loaded from {weights_path}")

        base_model_id = str(config.get("base_model_name_or_path", ""))
        rank = int(config.get("r", 0))
        scale = float(config.get("lora_alpha", 1.0))

        return AdapterPayload(
            directory=directory,
            base_model_id=base_model_id,
            rank=rank,
            scale=scale,
            weights=weights,
            module_keys=module_keys,
        )

    @staticmethod
    def _extract_layer_index(key: str) -> int | None:
        parts = key.split(".")
        if "layers" not in parts:
            return None
        idx = parts.index("layers")
        if idx + 1 >= len(parts):
            return None
        try:
            return int(parts[idx + 1])
        except ValueError:
            return None

    @staticmethod
    def _geometric_merge_matrices(
        matrices: list["Array"],
        backend: "Backend",
    ) -> tuple["Array", float, float]:
        if not matrices:
            raise MergeError("No matrices provided for merge")
        arrays = [backend.array(matrix) for matrix in matrices]
        backend.eval(*arrays)
        if len(arrays) == 1:
            return arrays[0], 0.0, 1.0

        reference = arrays[0]
        if reference.ndim == 1:
            stacked = backend.stack(arrays, axis=0)
            merged = backend.mean(stacked, axis=0)
            backend.eval(merged)
            return merged, 0.0, 1.0

        aligned_matrices: list["Array"] = []
        errors: list[float] = []
        cka_values: list[float] = []
        for matrix in arrays:
            aligned, error = LoRAAdapterMerger._procrustes_align(matrix, reference, backend)
            aligned_matrices.append(aligned)
            errors.append(error)
            cka = LoRAAdapterMerger._compute_cka(aligned, reference, backend)
            cka_values.append(cka)

        stacked = backend.stack(aligned_matrices, axis=0)
        merged = backend.mean(stacked, axis=0)
        backend.eval(merged)

        mean_error = sum(errors) / float(len(errors)) if errors else 0.0
        mean_cka = sum(cka_values) / float(len(cka_values)) if cka_values else 0.0

        return merged, float(mean_error), float(mean_cka)

    @staticmethod
    def _procrustes_align(
        source: "Array",
        target: "Array",
        backend: "Backend",
    ) -> tuple["Array", float]:
        """Align source to target using Procrustes analysis."""
        utils = BackendMatrixUtils(backend)
        aligned, result = utils.procrustes_align(source, target, center=True)
        # Normalize residual by target energy for relative error
        target_energy = backend.mean(target * target)
        backend.eval(target_energy)
        target_energy_val = float(backend.to_scalar(target_energy))
        error = result.residual / max(target_energy_val, 1e-10) if target_energy_val > 0 else 0.0
        return aligned, error

    @staticmethod
    def _compute_cka(
        source: "Array",
        target: "Array",
        backend: "Backend",
    ) -> float:
        result = compute_cka(
            source,
            target,
            backend,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        )
        if result.cka_corrected is not None:
            return float(result.cka_corrected)
        return float(result.cka)
