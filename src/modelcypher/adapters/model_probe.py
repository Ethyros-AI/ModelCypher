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

"""Model probing - architecture detection and alignment analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class LayerInfo:
    """Information about a single model layer."""
    name: str
    type: str
    parameters: int
    shape: list[int]


@dataclass(frozen=True)
class ModelProbeResult:
    """Result of probing a model for architecture details."""
    architecture: str
    parameter_count: int
    layers: list[LayerInfo]
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    layer_count_config: int = 0
    quantization: str | None = None


@dataclass(frozen=True)
class MergeValidationResult:
    """Merge effort validation between two models."""
    low_effort: bool
    warnings: list[str]
    architecture_match: bool
    vocab_match: bool
    dimension_match: bool


@dataclass(frozen=True)
class LayerDrift:
    """Drift information for a single layer."""
    layer_name: str
    drift_magnitude: float | None
    drift_z_score: float | None = None
    comparable: bool = True


@dataclass(frozen=True)
class AlignmentAnalysisResult:
    """Result of analyzing alignment drift between two models."""
    drift_magnitude: float | None
    drift_std: float | None
    drift_min: float | None
    drift_max: float | None
    drift_p50: float | None
    drift_p90: float | None
    common_layer_count: int
    comparable_layer_count: int
    missing_layer_count: int
    layer_drifts: list[LayerDrift]


def _infer_layer_type(key: str) -> str:
    """Infer layer type from weight key name."""
    key_lower = key.lower()
    if "embed" in key_lower:
        return "embedding"
    if "attn" in key_lower or "attention" in key_lower:
        if "q_proj" in key_lower or "query" in key_lower:
            return "attention_query"
        if "k_proj" in key_lower or "key" in key_lower:
            return "attention_key"
        if "v_proj" in key_lower or "value" in key_lower:
            return "attention_value"
        if "o_proj" in key_lower or "out" in key_lower:
            return "attention_output"
        return "attention"
    if "mlp" in key_lower or "ffn" in key_lower:
        if "gate" in key_lower:
            return "mlp_gate"
        if "up" in key_lower:
            return "mlp_up"
        if "down" in key_lower:
            return "mlp_down"
        return "mlp"
    if "norm" in key_lower or "ln" in key_lower:
        return "normalization"
    if "lm_head" in key_lower:
        return "lm_head"
    return "unknown"


class ModelProbe:
    """Probe model architecture and compare models."""

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()

    def probe(self, model_path: str) -> ModelProbeResult:
        model_dir = Path(model_path).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_dir}")

        config = self._load_config(model_dir)
        weights = self._load_weights(model_dir)

        layers: list[LayerInfo] = []
        parameter_count = 0
        for name, tensor in weights.items():
            shape = self._get_shape(tensor)
            params = _numel(shape)
            parameter_count += params
            layers.append(LayerInfo(
                name=name,
                type=_infer_layer_type(name),
                parameters=params,
                shape=shape,
            ))

        return ModelProbeResult(
            architecture=config["architecture"],
            parameter_count=parameter_count,
            layers=layers,
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            layer_count_config=config["num_layers"],
            quantization=config["quantization"],
        )

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        source_result = self.probe(source)
        target_result = self.probe(target)

        architecture_match = source_result.architecture == target_result.architecture
        vocab_match = source_result.vocab_size == target_result.vocab_size
        dimension_match = source_result.hidden_size == target_result.hidden_size

        warnings: list[str] = []
        if not architecture_match:
            warnings.append("Architecture mismatch")
        if not vocab_match:
            warnings.append("Vocabulary size mismatch")
        if not dimension_match:
            warnings.append("Hidden size mismatch")

        return MergeValidationResult(
            low_effort=architecture_match and vocab_match and dimension_match,
            warnings=warnings,
            architecture_match=architecture_match,
            vocab_match=vocab_match,
            dimension_match=dimension_match,
        )

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        weights_a = self._load_weights(Path(model_a).expanduser().resolve())
        weights_b = self._load_weights(Path(model_b).expanduser().resolve())

        keys_a = set(weights_a.keys())
        keys_b = set(weights_b.keys())
        common_keys = sorted(keys_a & keys_b)
        missing_keys = sorted(keys_a ^ keys_b)

        layer_drifts: list[LayerDrift] = []
        drift_values: list[float] = []

        for key in common_keys:
            drift = self._compute_drift(weights_a[key], weights_b[key])
            drift_values.append(drift)
            layer_drifts.append(LayerDrift(
                layer_name=key,
                drift_magnitude=drift,
                drift_z_score=None,
                comparable=True,
            ))

        if not drift_values:
            return AlignmentAnalysisResult(
                drift_magnitude=None, drift_std=None, drift_min=None, drift_max=None,
                drift_p50=None, drift_p90=None,
                common_layer_count=len(common_keys),
                comparable_layer_count=len(common_keys),
                missing_layer_count=len(missing_keys),
                layer_drifts=layer_drifts,
            )

        sorted_vals = sorted(drift_values)
        count = len(sorted_vals)
        mean = sum(sorted_vals) / count
        variance = sum((v - mean) ** 2 for v in sorted_vals) / count
        std = variance ** 0.5

        layer_drifts = [
            LayerDrift(
                layer_name=ld.layer_name,
                drift_magnitude=ld.drift_magnitude,
                drift_z_score=(ld.drift_magnitude - mean) / std if std > 0 and ld.drift_magnitude else None,
                comparable=ld.comparable,
            )
            for ld in layer_drifts
        ]

        return AlignmentAnalysisResult(
            drift_magnitude=mean,
            drift_std=std,
            drift_min=sorted_vals[0],
            drift_max=sorted_vals[-1],
            drift_p50=sorted_vals[int(0.5 * (count - 1))],
            drift_p90=sorted_vals[int(0.9 * (count - 1))],
            common_layer_count=len(common_keys),
            comparable_layer_count=len(common_keys),
            missing_layer_count=len(missing_keys),
            layer_drifts=layer_drifts,
        )

    def _load_config(self, model_dir: Path) -> dict[str, Any]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        config = json.loads(config_path.read_text())

        def first_str(value: Any) -> str | None:
            if isinstance(value, list) and value:
                return value[0] if isinstance(value[0], str) else None
            return value if isinstance(value, str) else None

        architecture = first_str(config.get("architectures")) or config.get("model_type") or "unknown"
        vocab_size = int(config.get("vocab_size") or 0)
        hidden_size = int(config.get("hidden_size") or config.get("n_embd") or 0)
        num_attention_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
        num_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers") or 0)

        quant = config.get("quantization")
        quantization = None
        if isinstance(quant, str):
            quantization = quant
        elif isinstance(quant, dict):
            quantization = str(quant.get("quant_method") or quant.get("method"))

        return {
            "architecture": architecture,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "quantization": quantization,
        }

    def _load_weights(self, model_dir: Path) -> dict[str, Any]:
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        weights: dict[str, Any] = {}
        if safetensor_files:
            for sf_path in safetensor_files:
                weights.update(self._backend.load_safetensors(str(sf_path)))
            self._backend.eval(*weights.values())
            return weights

        bin_files = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.pt"))
        if bin_files:
            for bin_path in bin_files:
                weights.update(self._backend.load_binary_weights(str(bin_path)))
            self._backend.eval(*weights.values())
            return weights

        raise FileNotFoundError(f"No weight files found in {model_dir}")

    def _get_shape(self, tensor: Any) -> list[int]:
        shape = getattr(tensor, "shape", None)
        if shape is None:
            shape = self._backend.shape(tensor)
        return [int(dim) for dim in shape]

    def _compute_drift(self, tensor_a: Any, tensor_b: Any) -> float:
        b = self._backend
        diff = tensor_a - tensor_b
        diff_sq = b.sum(diff * diff)
        norm_a_sq = b.sum(tensor_a * tensor_a)
        b.eval(diff_sq, norm_a_sq)

        diff_norm = b.sqrt(diff_sq)
        norm_a = b.sqrt(norm_a_sq)
        eps = float(machine_epsilon(b, tensor_a))
        denom = b.maximum(norm_a, b.array([eps]))
        drift = diff_norm / denom
        b.eval(drift)
        return float(b.to_scalar(drift))


def _numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
