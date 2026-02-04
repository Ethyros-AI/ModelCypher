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

"""Backend-based model probe implementation.

This module implements the ModelProbePort protocol using the Backend abstraction,
making it framework-agnostic (works with MLX, JAX, or CUDA backends).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from modelcypher.ports.model_probe import (
    AlignmentAnalysisResult,
    LayerDrift,
    LayerInfo,
    MergeValidationResult,
    ModelProbeResult,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class BackendModelProbe:
    """Model probe using Backend protocol.

    This implementation uses the Backend abstraction to load and analyze model
    weights, making it framework-agnostic (works with MLX, JAX, or CUDA backends).
    """

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details."""
        path = Path(model_path)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Extract architecture info
        model_type = config.get("model_type", "unknown")
        hidden_size = config.get("hidden_size", 0)
        num_layers = config.get("num_hidden_layers", 0)
        num_attention_heads = config.get("num_attention_heads", 0)
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        intermediate_size = config.get("intermediate_size", 0)
        vocab_size = config.get("vocab_size", 0)

        # Load weights to get layer info
        layers = self._analyze_layers(path)

        return ModelProbeResult(
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            layers=layers,
            config=config,
        )

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models."""
        source_result = self.probe(source)
        target_result = self.probe(target)

        compatible = (
            source_result.hidden_size == target_result.hidden_size
            and source_result.num_layers == target_result.num_layers
            and source_result.intermediate_size == target_result.intermediate_size
        )

        messages = []
        if source_result.hidden_size != target_result.hidden_size:
            messages.append(
                f"Hidden size mismatch: {source_result.hidden_size} vs {target_result.hidden_size}"
            )
        if source_result.num_layers != target_result.num_layers:
            messages.append(
                f"Layer count mismatch: {source_result.num_layers} vs {target_result.num_layers}"
            )
        if source_result.intermediate_size != target_result.intermediate_size:
            messages.append(
                f"Intermediate size mismatch: {source_result.intermediate_size} vs {target_result.intermediate_size}"
            )
        if source_result.vocab_size != target_result.vocab_size:
            messages.append(
                f"Vocab size mismatch: {source_result.vocab_size} vs {target_result.vocab_size}"
            )

        return MergeValidationResult(
            compatible=compatible,
            message="; ".join(messages) if messages else "Models are compatible",
            source_layers=source_result.num_layers,
            target_layers=target_result.num_layers,
            hidden_size_match=source_result.hidden_size == target_result.hidden_size,
        )

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models."""
        weights_a = self._load_weights(Path(model_a))
        weights_b = self._load_weights(Path(model_b))

        # Find common layers
        common_keys = set(weights_a.keys()) & set(weights_b.keys())
        layer_drifts = []

        for key in sorted(common_keys):
            tensor_a = weights_a[key]
            tensor_b = weights_b[key]

            # Check shape compatibility
            shape_a = list(tensor_a.shape)
            shape_b = list(tensor_b.shape)

            if shape_a != shape_b:
                continue

            # Compute drift
            drift = self._compute_drift(tensor_a, tensor_b)
            layer_type = self._infer_layer_type(key)

            layer_drifts.append(
                LayerDrift(
                    layer_name=key,
                    layer_type=layer_type,
                    drift_score=drift,
                    shape=shape_a,
                )
            )

        # Compute statistics
        drifts = [ld.drift_score for ld in layer_drifts]
        drift_mean = sum(drifts) / len(drifts) if drifts else 0.0
        drift_max = max(drifts) if drifts else 0.0
        sorted_drifts = sorted(drifts)
        drift_p90 = sorted_drifts[int(len(sorted_drifts) * 0.9)] if sorted_drifts else 0.0

        return AlignmentAnalysisResult(
            drift_mean=drift_mean,
            drift_max=drift_max,
            drift_p90=drift_p90,
            common_layer_count=len(common_keys),
            comparable_layer_count=len(layer_drifts),
            missing_layer_count=len(set(weights_a.keys()) ^ set(weights_b.keys())),
            layer_drifts=layer_drifts,
        )

    def _analyze_layers(self, model_path: Path) -> list[LayerInfo]:
        """Analyze layer structure from model weights."""
        weights = self._load_weights(model_path)
        layers = []

        for key in sorted(weights.keys()):
            tensor = weights[key]
            shape = list(tensor.shape)
            layer_type = self._infer_layer_type(key)

            layers.append(
                LayerInfo(
                    name=key,
                    layer_type=layer_type,
                    shape=shape,
                    dtype=str(tensor.dtype),
                )
            )

        return layers

    def _load_weights(self, model_path: Path) -> dict[str, Any]:
        """Load model weights using backend."""
        return self._backend.load_binary_weights(model_path)

    def _compute_drift(self, tensor_a: Any, tensor_b: Any) -> float:
        """Compute normalized drift between two tensors."""
        # Frobenius norm of difference divided by average norm
        diff = tensor_a - tensor_b
        norm_diff = float(self._backend.norm(diff))
        norm_a = float(self._backend.norm(tensor_a))
        norm_b = float(self._backend.norm(tensor_b))
        avg_norm = (norm_a + norm_b) / 2.0

        if avg_norm < 1e-8:
            return 0.0
        return norm_diff / avg_norm

    @staticmethod
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
        if "norm" in key_lower:
            return "norm"
        if "lm_head" in key_lower:
            return "lm_head"
        return "unknown"


__all__ = ["BackendModelProbe"]
