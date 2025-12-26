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

"""CUDA/PyTorch model probe implementation.

Uses PyTorch + safetensors for weight loading with bfloat16 support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from modelcypher.ports.model_probe import (
    AlignmentAnalysisResult,
    BaseModelProbe,
    LayerDrift,
    LayerInfo,
    MergeValidationResult,
    ModelProbeResult,
)

logger = logging.getLogger(__name__)


class CUDAModelProbe(BaseModelProbe):
    """
    CUDA/PyTorch model probe implementation.

    Supports safetensors with framework="pt" and bfloat16 tensors.
    Native torch.load() checkpoints are not yet supported here.

    Requires: pip install torch safetensors
    """

    def __init__(self) -> None:
        """Initialize CUDA probe, checking for PyTorch availability."""
        try:
            import torch

            self.torch = torch
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("PyTorch not available. Install with: pip install torch")

    @property
    def available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._available

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details using PyTorch."""
        if not self._available:
            raise RuntimeError("CUDA backend not available. Install PyTorch: pip install torch")

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {path}")

        config_path = path / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in model directory: {path}")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid config.json: {exc}") from exc

        architecture = config.get("model_type", "unknown")
        vocab_size = config.get("vocab_size", 0)
        hidden_size = config.get("hidden_size", 0)
        num_attention_heads = config.get("num_attention_heads", 0)
        quantization = config.get("quantization_config", {}).get("quant_method")

        layers, parameter_count = self._analyze_weights(path)

        return ModelProbeResult(
            architecture=architecture,
            parameter_count=parameter_count,
            layers=layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            quantization=quantization,
        )

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models."""
        source_probe = self.probe(source)
        target_probe = self.probe(target)

        warnings: list[str] = []

        architecture_match = source_probe.architecture == target_probe.architecture
        if not architecture_match:
            warnings.append(
                f"Architecture mismatch: {source_probe.architecture} vs {target_probe.architecture}"
            )

        vocab_match = source_probe.vocab_size == target_probe.vocab_size
        if not vocab_match:
            warnings.append(
                f"Vocab size mismatch: {source_probe.vocab_size} vs {target_probe.vocab_size}"
            )

        dimension_match = source_probe.hidden_size == target_probe.hidden_size
        if not dimension_match:
            warnings.append(
                f"Hidden dimension mismatch: {source_probe.hidden_size} vs {target_probe.hidden_size}"
            )

        low_effort = architecture_match and vocab_match and dimension_match

        return MergeValidationResult(
            low_effort=low_effort,
            warnings=warnings,
            architecture_match=architecture_match,
            vocab_match=vocab_match,
            dimension_match=dimension_match,
        )

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models."""
        path_a = Path(model_a).expanduser().resolve()
        path_b = Path(model_b).expanduser().resolve()

        weights_a = self._load_weight_tensors(path_a)
        weights_b = self._load_weight_tensors(path_b)

        common_layers = set(weights_a.keys()) & set(weights_b.keys())
        if not common_layers:
            return AlignmentAnalysisResult(
                drift_magnitude=1.0,
                layer_drifts=[],
                assessment="needs_layer_mapping",
                interpretation="No common layer names - use invariant layer mapping to find corresponding layers.",
            )

        layer_drifts: list[LayerDrift] = []
        total_drift = 0.0

        for layer_name in sorted(common_layers):
            tensor_a = weights_a[layer_name]
            tensor_b = weights_b[layer_name]

            if tensor_a.shape != tensor_b.shape:
                layer_drifts.append(
                    LayerDrift(
                        layer_name=layer_name,
                        drift_magnitude=1.0,
                        direction="shape_mismatch",
                    )
                )
                total_drift += 1.0
                continue

            drift = self._compute_layer_drift(tensor_a, tensor_b)
            direction = "divergent" if drift > 0.5 else "aligned"

            layer_drifts.append(
                LayerDrift(
                    layer_name=layer_name,
                    drift_magnitude=drift,
                    direction=direction,
                )
            )
            total_drift += drift

        avg_drift = total_drift / len(common_layers) if common_layers else 1.0
        drift_magnitude = min(1.0, max(0.0, avg_drift))

        if drift_magnitude < 0.1:
            assessment = "highly_aligned"
            interpretation = "Models are highly aligned with minimal drift."
        elif drift_magnitude < 0.3:
            assessment = "moderately_aligned"
            interpretation = "Models show moderate alignment with some drift."
        elif drift_magnitude < 0.6:
            assessment = "divergent"
            interpretation = "Models have diverged significantly."
        else:
            assessment = "highly_divergent"
            interpretation = "Models are highly divergent and may not be compatible."

        return AlignmentAnalysisResult(
            drift_magnitude=drift_magnitude,
            layer_drifts=layer_drifts,
            assessment=assessment,
            interpretation=interpretation,
        )

    def _analyze_weights(self, model_path: Path) -> tuple[list[LayerInfo], int]:
        """Analyze weight files using PyTorch/safetensors."""
        from safetensors import safe_open

        layers: list[LayerInfo] = []
        total_params = 0

        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            return layers, total_params

        for st_file in safetensor_files:
            try:
                # Use PyTorch framework for bfloat16 support
                with safe_open(st_file, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        shape = list(tensor.shape)
                        params = 1
                        for dim in shape:
                            params *= dim

                        layer_type = self.infer_layer_type(key)
                        layers.append(
                            LayerInfo(
                                name=key,
                                type=layer_type,
                                parameters=params,
                                shape=shape,
                            )
                        )
                        total_params += params
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)

        return layers, total_params

    def _load_weight_tensors(self, model_path: Path) -> dict[str, Any]:
        """Load weight tensors using PyTorch (supports bfloat16)."""
        from safetensors import safe_open

        tensors: dict[str, Any] = {}

        safetensor_files = list(model_path.glob("*.safetensors"))
        for st_file in safetensor_files:
            try:
                # framework="pt" returns PyTorch tensors with bfloat16 support
                with safe_open(st_file, framework="pt") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)

        return tensors

    def _get_tensor_shape(self, tensor: Any) -> list[int]:
        """Get shape of PyTorch tensor."""
        return list(tensor.shape)

    def _compute_layer_drift(self, tensor_a: Any, tensor_b: Any) -> float:
        """Compute normalized drift between two PyTorch tensors."""
        import math

        # Convert to float32 for drift computation
        a_f32 = tensor_a.float()
        b_f32 = tensor_b.float()

        diff = a_f32 - b_f32
        norm_diff = float(self.torch.norm(diff.flatten()))
        norm_a = float(self.torch.norm(a_f32.flatten()))
        norm_b = float(self.torch.norm(b_f32.flatten()))

        max_norm = max(norm_a, norm_b, 1e-8)
        relative_drift = norm_diff / max_norm

        normalized = 1.0 - math.exp(-relative_drift)
        return float(min(1.0, max(0.0, normalized)))
