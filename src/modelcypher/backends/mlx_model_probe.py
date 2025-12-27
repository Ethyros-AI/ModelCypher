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

"""MLX-specific model probe implementation.

Uses mx.load() which natively supports bfloat16 and all MLX dtypes.
This is the production implementation for macOS/Apple Silicon.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import math

from modelcypher.ports.model_probe import (
    AlignmentAnalysisResult,
    BaseModelProbe,
    LayerDrift,
    LayerInfo,
    MergeValidationResult,
    ModelProbeResult,
)

logger = logging.getLogger(__name__)


class MLXModelProbe(BaseModelProbe):
    """
    MLX-specific model probe.

    Uses mx.load() for weight loading, which:
    - Supports bfloat16 natively
    - Handles all MLX quantization formats (4bit, 8bit)
    - Works with sharded safetensors files
    - Provides lazy loading for memory efficiency
    """

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details using MLX."""
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

        set_a = set(weights_a.keys())
        set_b = set(weights_b.keys())
        common_layers = set_a & set_b
        missing_layer_count = len(set_a - set_b) + len(set_b - set_a)
        if not common_layers:
            return AlignmentAnalysisResult(
                drift_magnitude=None,
                drift_std=None,
                drift_min=None,
                drift_max=None,
                drift_p50=None,
                drift_p90=None,
                common_layer_count=0,
                comparable_layer_count=0,
                missing_layer_count=missing_layer_count,
                layer_drifts=[],
            )

        raw_drifts: list[tuple[str, float | None, bool]] = []

        for layer_name in sorted(common_layers):
            tensor_a = weights_a[layer_name]
            tensor_b = weights_b[layer_name]

            if tensor_a.shape != tensor_b.shape:
                raw_drifts.append((layer_name, None, False))
                continue

            drift = self._compute_layer_drift(tensor_a, tensor_b)
            raw_drifts.append((layer_name, drift, True))

        computed_drifts = [d for _, d, comparable in raw_drifts if comparable and d is not None]
        if computed_drifts:
            mean_drift = sum(computed_drifts) / len(computed_drifts)
            variance = sum((d - mean_drift) ** 2 for d in computed_drifts) / len(computed_drifts)
            std_drift = math.sqrt(variance)
            sorted_drifts = sorted(computed_drifts)
            drift_min = sorted_drifts[0]
            drift_max = sorted_drifts[-1]
            idx_p50 = int(0.5 * (len(sorted_drifts) - 1))
            idx_p90 = int(0.9 * (len(sorted_drifts) - 1))
            drift_p50 = sorted_drifts[idx_p50]
            drift_p90 = sorted_drifts[idx_p90]
        else:
            mean_drift = None
            std_drift = None
            drift_min = None
            drift_max = None
            drift_p50 = None
            drift_p90 = None

        layer_drifts: list[LayerDrift] = []
        for layer_name, drift, comparable in raw_drifts:
            if drift is None:
                layer_drifts.append(
                    LayerDrift(
                        layer_name=layer_name,
                        drift_magnitude=None,
                        drift_z_score=None,
                        comparable=False,
                    )
                )
                continue
            if std_drift is None or std_drift == 0.0:
                z_score = 0.0
            else:
                z_score = (drift - mean_drift) / std_drift
            layer_drifts.append(
                LayerDrift(
                    layer_name=layer_name,
                    drift_magnitude=drift,
                    drift_z_score=z_score,
                    comparable=True,
                )
            )

        return AlignmentAnalysisResult(
            drift_magnitude=mean_drift,
            drift_std=std_drift,
            drift_min=drift_min,
            drift_max=drift_max,
            drift_p50=drift_p50,
            drift_p90=drift_p90,
            common_layer_count=len(common_layers),
            comparable_layer_count=len(computed_drifts),
            missing_layer_count=missing_layer_count,
            layer_drifts=layer_drifts,
        )

    def _analyze_weights(self, model_path: Path) -> tuple[list[LayerInfo], int]:
        """Analyze weight files to extract layer information using MLX."""
        layers: list[LayerInfo] = []
        total_params = 0

        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            return layers, total_params

        for st_file in safetensor_files:
            try:
                # mx.load() handles bfloat16, 4bit, 8bit - all MLX formats
                tensors = mx.load(str(st_file))
                for key, tensor in tensors.items():
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
        """Load weight tensors using MLX (supports bfloat16)."""
        tensors: dict[str, Any] = {}

        safetensor_files = list(model_path.glob("*.safetensors"))
        for st_file in safetensor_files:
            try:
                # mx.load() returns MLX arrays, handles all dtypes including bfloat16
                file_tensors = mx.load(str(st_file))
                tensors.update(file_tensors)
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)

        return tensors

    def _get_tensor_shape(self, tensor: Any) -> list[int]:
        """Get shape of MLX tensor."""
        return list(tensor.shape)

    def _compute_layer_drift(self, tensor_a: Any, tensor_b: Any) -> float:
        """Compute normalized drift between two MLX tensors.

        Returns a value in [0.0, 1.0] where 0 means identical and 1 means maximally different.
        """
        # Convert to float32 for drift computation (handles bfloat16, quantized, etc.)
        a_f32 = tensor_a.astype(mx.float32)
        b_f32 = tensor_b.astype(mx.float32)

        diff = a_f32 - b_f32
        # Flatten and compute norms
        diff_flat = mx.reshape(diff, (-1,))
        a_flat = mx.reshape(a_f32, (-1,))
        b_flat = mx.reshape(b_f32, (-1,))

        norm_diff = float(mx.sqrt(mx.sum(diff_flat * diff_flat)))
        norm_a = float(mx.sqrt(mx.sum(a_flat * a_flat)))
        norm_b = float(mx.sqrt(mx.sum(b_flat * b_flat)))

        max_norm = max(norm_a, norm_b, 1e-8)
        relative_drift = norm_diff / max_norm

        # Normalize to [0, 1] using exponential decay
        normalized = 1.0 - math.exp(-relative_drift)
        return float(min(1.0, max(0.0, normalized)))
