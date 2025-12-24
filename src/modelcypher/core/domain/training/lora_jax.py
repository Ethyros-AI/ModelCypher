"""
LoRA (Low-Rank Adaptation) Support for Parameter-Efficient Fine-Tuning (JAX Backend).

This module provides a JAX/Flax implementation of LoRA adapters.
Currently a stub - implement when JAX support is needed.

For other backends:
- MLX/macOS: see lora_mlx.py
- CUDA/PyTorch: see lora_cuda.py

Use _platform.get_lora_config_class() for automatic platform selection.

Implementation Notes:
- Replace mlx.nn.Linear with flax.linen.Dense
- Replace mlx.utils.tree_flatten with jax.tree_util.tree_flatten
- Use flax.linen.Module for LoRADense implementation
- Handle JAX arrays instead of mx.array

Research Basis:
- LoRA: arxiv:2106.09685
- DoRA: arxiv:2402.09353
- Flax: https://flax.readthedocs.io/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class FineTuneTypeJAX(str, Enum):
    """Fine-tuning method type."""
    LORA = "lora"
    DORA = "dora"  # Weight-decomposed LoRA


@dataclass
class LoRAConfigJAX:
    """Configuration for LoRA adapters (JAX version)."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fine_tune_type: FineTuneTypeJAX = FineTuneTypeJAX.LORA
    num_layers: Optional[int] = None  # None = all layers

    @property
    def scale(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.alpha / max(self.rank, 1)

    @classmethod
    def default(cls) -> "LoRAConfigJAX":
        return cls()

    @classmethod
    def for_mistral(cls) -> "LoRAConfigJAX":
        """Preset for Mistral-style models."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @classmethod
    def for_llama(cls) -> "LoRAConfigJAX":
        """Preset for Llama-style models."""
        return cls(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )


@dataclass
class TargetResolutionJAX:
    """Result of resolving LoRA target modules."""
    resolved_keys: List[str]
    unmatched_modules: List[str]
    layer_count: int


@dataclass
class LoRAExportResultJAX:
    """Result of exporting LoRA adapters."""
    path: Path
    parameter_count: int
    file_size_bytes: int


class LoRADenseJAX:
    """
    Dense layer with LoRA adapters (JAX/Flax version).

    This is a stub. When JAX support is needed, implement as flax.linen.Module:

    class LoRADense(nn.Module):
        features: int
        rank: int = 8
        alpha: float = 16.0

        @nn.compact
        def __call__(self, x):
            # Base dense (frozen)
            y = nn.Dense(self.features, name='base')(x)

            # LoRA adapters
            lora_a = self.param('lora_a', nn.initializers.normal(0.01), (x.shape[-1], self.rank))
            lora_b = self.param('lora_b', nn.initializers.zeros, (self.rank, self.features))

            scale = self.alpha / self.rank
            lora_out = x @ lora_a @ lora_b * scale

            return y + lora_out
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JAX LoRA not yet implemented. "
            "See lora_mlx.py for the MLX implementation to port. "
            "Key differences:\n"
            "  - Implement as flax.linen.Module\n"
            "  - Use nn.Dense instead of nn.Linear\n"
            "  - Use self.param() for weight initialization\n"
            "  - Handle frozen/trainable params with TrainState"
        )


def resolve_lora_targets_jax(
    model: Any,  # flax.linen.Module
    config: LoRAConfigJAX,
) -> TargetResolutionJAX:
    """
    Resolve LoRA target modules within a JAX/Flax model.

    Raises:
        NotImplementedError: This is a stub.
    """
    raise NotImplementedError(
        "JAX LoRA target resolution not yet implemented. "
        "Use jax.tree_util.tree_flatten to scan module parameters."
    )


def apply_lora_to_model_jax(
    model: Any,  # flax.linen.Module
    config: LoRAConfigJAX,
) -> Any:
    """
    Inject LoRA adapters into targeted Dense modules.

    Raises:
        NotImplementedError: This is a stub.
    """
    raise NotImplementedError(
        "JAX LoRA injection not yet implemented. "
        "Flax modules are immutable - need to return new module tree."
    )


def export_lora_adapters_jax(
    params: Dict[str, Any],  # Flax params pytree
    output_path: Path,
    config: LoRAConfigJAX,
    model_id: str = "",
) -> LoRAExportResultJAX:
    """
    Export trained LoRA adapter weights.

    Raises:
        NotImplementedError: This is a stub.
    """
    raise NotImplementedError(
        "JAX LoRA export not yet implemented. "
        "Use flax.serialization or orbax.checkpoint for saving."
    )


def load_lora_adapters_jax(
    params: Dict[str, Any],  # Flax params pytree
    adapter_path: Path,
) -> Dict[str, Any]:
    """
    Load LoRA adapter weights into Flax params.

    Raises:
        NotImplementedError: This is a stub.
    """
    raise NotImplementedError(
        "JAX LoRA loading not yet implemented. "
        "Use flax.serialization or orbax.checkpoint for loading."
    )


__all__ = [
    "LoRAConfigJAX",
    "LoRADenseJAX",
    "LoRAExportResultJAX",
    "TargetResolutionJAX",
    "FineTuneTypeJAX",
    "resolve_lora_targets_jax",
    "apply_lora_to_model_jax",
    "export_lora_adapters_jax",
    "load_lora_adapters_jax",
]
