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
from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
)
from modelcypher.core.domain.merging.exceptions import MergeError

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
    mean_permutation_cka: float
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
    ) -> MergeReport:
        if len(adapter_directories) < 2:
            raise MergeError("At least two adapters are required for merge")

        b = backend or get_default_backend()
        adapters = [LoRAAdapterMerger._load_adapter(path, backend=b) for path in adapter_directories]

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
            mean_permutation_cka=float(mean_cka),
            total_merged_parameters=int(merged_parameters),
            layer_count=len(layer_indices),
            mean_merge_cka=float(mean_cka),
        )

    @staticmethod
    def _load_adapter(directory: Path, backend: "Backend | None" = None) -> AdapterPayload:
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
            try:
                import safetensors.numpy
                raw_weights = safetensors.numpy.load_file(str(weights_path))
                for key, value in raw_weights.items():
                    weights[key] = backend.array(value)
                    module_keys.append(key)
            except ImportError:
                raise MergeError(
                    "safetensors package required for .safetensors files. "
                    "Install with: pip install safetensors"
                )
        elif weights_path.suffix in (".bin", ".pt"):
            try:
                import torch
                raw_weights = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                for key, value in raw_weights.items():
                    # Convert torch tensor to backend array via numpy
                    weights[key] = backend.array(value.numpy())
                    module_keys.append(key)
            except ImportError:
                raise MergeError(
                    "torch package required for .bin/.pt files. "
                    "Install with: pip install torch"
                )

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
        source_mean = backend.mean(source, axis=0, keepdims=True)
        target_mean = backend.mean(target, axis=0, keepdims=True)
        source_centered = source - source_mean
        target_centered = target - target_mean

        cross_cov = backend.matmul(backend.transpose(source_centered), target_centered)
        # Geodesic SVD (GPU-only)
        U, _, Vt = geodesic_svd(backend, cross_cov)
        # MLX det() has unstable behavior for some sizes; use the raw orthogonal solution.
        rotation = backend.matmul(U, Vt)

        aligned = backend.matmul(source_centered, rotation) + target_mean
        backend.eval(aligned)

        diff = aligned - target
        mse = backend.mean(diff * diff)
        target_energy = backend.mean(target * target)
        backend.eval(mse, target_energy)
        eps = division_epsilon(backend, target)
        mse_val = float(backend.to_scalar(mse))
        target_energy_val = float(backend.to_scalar(target_energy))
        error = mse_val / max(target_energy_val, eps)
        return aligned, error

    @staticmethod
    def _permutation_align(
        source: "Array",
        target: "Array",
        backend: "Backend",
    ):
        from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner

        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)
        return PermutationAligner.align(source_arr, target_arr, backend=backend)

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
            use_linear_kernel=True,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        )
        if result.cka_corrected is not None:
            return float(result.cka_corrected)
        return float(result.cka)
