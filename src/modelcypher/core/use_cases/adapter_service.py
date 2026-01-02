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

"""Adapter service for LoRA adapter operations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerAdapterInfo:
    """Information about adapter weights for a single layer."""

    name: str
    rank: int
    alpha: float
    parameters: int


@dataclass(frozen=True)
class AdapterInspectResult:
    """Result of inspecting an adapter."""

    rank: int
    alpha: float
    target_modules: list[str]
    sparsity: float
    parameter_count: int
    layer_analysis: list[LayerAdapterInfo]


@dataclass(frozen=True)
class ProjectResult:
    """Result of projecting an adapter."""

    output_path: str
    projected_layers: int


@dataclass(frozen=True)
class WrapResult:
    """Result of wrapping an adapter for MLX."""

    output_path: str
    wrapped_layers: int


class AdapterService:
    """Service for LoRA adapter operations."""

    def inspect(self, adapter_path: str) -> AdapterInspectResult:
        """Return detailed adapter analysis.

        Args:
            adapter_path: Path to adapter directory.

        Returns:
            AdapterInspectResult with adapter details.
        """
        backend = get_default_backend()
        path = Path(adapter_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")

        # Load adapter config
        config_path = path / "adapter_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            rank = config.get("r", config.get("rank", 8))
            alpha = config.get("lora_alpha", config.get("alpha", 16.0))
            target_modules = config.get("target_modules", [])
        else:
            rank = 8
            alpha = 16.0
            target_modules = []

        # Analyze weights
        weights = self._load_weights(path)
        layer_analysis = []
        total_params = 0
        zero_count = 0
        total_elements = 0

        for name, tensor in weights.items():
            params = tensor.size
            total_params += params
            total_elements += params

            # Convert to backend array, compute, and convert back to scalar
            tensor_backend = backend.array(tensor)
            abs_tensor = backend.abs(tensor_backend)
            # Use machine_epsilon for zero check
            is_zero = abs_tensor < machine_epsilon(backend, tensor_backend)
            zero_count_tensor = backend.sum(is_zero)
            backend.eval(zero_count_tensor)
            zero_count += int(backend.to_scalar(zero_count_tensor))

            layer_analysis.append(
                LayerAdapterInfo(
                    name=name,
                    rank=rank,
                    alpha=alpha,
                    parameters=params,
                )
            )

        sparsity = zero_count / total_elements if total_elements > 0 else 0.0

        return AdapterInspectResult(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            sparsity=float(sparsity),
            parameter_count=total_params,
            layer_analysis=layer_analysis,
        )

    def project(self, adapter_path: str, target_space: str, output_path: str) -> ProjectResult:
        """Project adapter to target space.

        Args:
            adapter_path: Path to adapter directory.
            target_space: Target space identifier.
            output_path: Output path for projected adapter.

        Returns:
            ProjectResult with output path.
        """
        backend = get_default_backend()
        path = Path(adapter_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()

        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")

        output.mkdir(parents=True, exist_ok=True)

        weights = self._load_weights(path)
        projected_weights = {}

        for name, tensor in weights.items():
            # Simple projection: normalize weights
            tensor_backend = backend.array(tensor)
            norm = backend.norm(tensor_backend)
            backend.eval(norm)
            norm_scalar = float(backend.to_scalar(norm))

            if norm_scalar > 0:
                normalized = tensor_backend / norm
                backend.eval(normalized)
                projected_weights[name] = backend.to_numpy(normalized).astype("float32")
            else:
                projected_weights[name] = tensor.astype("float32")

        save_file(projected_weights, output / "adapter_model.safetensors")

        # Copy config
        config_path = path / "adapter_config.json"
        if config_path.exists():
            (output / "adapter_config.json").write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        return ProjectResult(
            output_path=str(output),
            projected_layers=len(projected_weights),
        )

    def wrap_mlx(self, adapter_path: str, output_path: str) -> WrapResult:
        """Wrap adapter for MLX compatibility.

        Args:
            adapter_path: Path to adapter directory.
            output_path: Output path for wrapped adapter.

        Returns:
            WrapResult with output path.
        """
        path = Path(adapter_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()

        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")

        output.mkdir(parents=True, exist_ok=True)

        weights = self._load_weights(path)
        wrapped_weights = {}

        for name, tensor in weights.items():
            # MLX expects [out, in] layout
            wrapped_weights[name] = tensor.astype("float32")

        save_file(wrapped_weights, output / "adapters.safetensors")

        return WrapResult(
            output_path=str(output),
            wrapped_layers=len(wrapped_weights),
        )

    def _load_weights(self, path: Path) -> dict:
        """Load weights from safetensors files."""
        weights = {}

        safetensor_files = list(path.glob("*.safetensors"))
        for st_file in safetensor_files:
            try:
                with safe_open(st_file, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)

        return weights
