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

"""Abstract interface for model probing across backends.

Weight loading is inherently backend-specific:
- MLX: mx.load() supports bfloat16 natively
- PyTorch/CUDA: torch.load() or safetensors with framework="pt"
- JAX: jax.numpy loading

Each backend must implement its own weight loading strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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
    quantization: str | None = None


@dataclass(frozen=True)
class MergeValidationResult:
    """Result of validating merge effort between two models.

    Note: Models are ALWAYS compatible. This result indicates how much
    transformation effort is needed, not whether merge is possible.
    """

    low_effort: bool  # True if merge requires minimal transformation
    warnings: list[str]
    architecture_match: bool
    vocab_match: bool
    dimension_match: bool


@dataclass(frozen=True)
class LayerDrift:
    """Drift information for a single layer."""

    layer_name: str
    drift_magnitude: float
    direction: str


@dataclass(frozen=True)
class AlignmentAnalysisResult:
    """Result of analyzing alignment drift between two models."""

    drift_magnitude: float
    layer_drifts: list[LayerDrift]
    assessment: str
    interpretation: str


@runtime_checkable
class ModelProbePort(Protocol):
    """
    Abstract interface for model probing.

    Implementations must handle backend-specific weight loading:
    - MLXModelProbe: Uses mx.load() for bfloat16 support
    - CUDAModelProbe: Uses torch.load() or safetensors with framework="pt"
    - JAXModelProbe: Uses JAX-native loading
    """

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details.

        Args:
            model_path: Path to the model directory containing config.json and weight files.

        Returns:
            ModelProbeResult with architecture details.
        """
        ...

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models.

        Args:
            source: Path to the source model directory.
            target: Path to the target model directory.

        Returns:
            MergeValidationResult with compatibility assessment.
        """
        ...

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models.

        Args:
            model_a: Path to the first model directory.
            model_b: Path to the second model directory.

        Returns:
            AlignmentAnalysisResult with drift metrics.
        """
        ...


class BaseModelProbe(ABC):
    """
    Base implementation with shared logic.

    Subclasses must implement _load_weight_tensors() for their backend.
    """

    @abstractmethod
    def _load_weight_tensors(self, model_path: Path) -> dict[str, Any]:
        """Load weight tensors from model files.

        Backend-specific implementation required:
        - MLX: Use mx.load() for bfloat16 support
        - CUDA: Use torch.load() or safetensors
        - JAX: Use JAX-native loading

        Returns:
            Dictionary mapping layer names to tensors.
        """
        ...

    @abstractmethod
    def _get_tensor_shape(self, tensor: Any) -> list[int]:
        """Get shape of a tensor in backend-agnostic way."""
        ...

    @abstractmethod
    def _compute_layer_drift(self, tensor_a: Any, tensor_b: Any) -> float:
        """Compute normalized drift between two tensors."""
        ...

    @staticmethod
    def infer_layer_type(key: str) -> str:
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
