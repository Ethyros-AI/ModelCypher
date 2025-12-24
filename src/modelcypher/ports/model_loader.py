"""Model loading port for hexagonal architecture.

This port abstracts model loading operations used for:
- Training preparation (model + tokenizer loading)
- Activation extraction for geometry analysis
- Weight loading for merge operations
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from modelcypher.core.domain.training.lora_mlx import LoRAConfig

@runtime_checkable
class ModelLoaderPort(Protocol):
    """Port for loading models and their components.

    Implementations handle backend-specific model loading (MLX, JAX, etc.)
    while domain code depends only on this abstract interface.
    """

    def load_model_for_training(
        self,
        model_path: str,
        lora_config: "LoRAConfig | None" = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer for training or inference.

        Args:
            model_path: Path to model directory
            lora_config: Optional LoRA configuration to apply

        Returns:
            Tuple of (model, tokenizer) where:
            - model: nn.Module with optional LoRA adapters applied
            - tokenizer: Tokenizer compatible with the model

        Raises:
            ImportError: If required backend (MLX/mlx_vlm) is unavailable
            RuntimeError: If model loading fails
        """
        ...

    def load_weights_as_numpy(self, model_path: str) -> "dict[str, np.ndarray]":
        """Load model weights as numpy arrays.

        Handles bfloat16 conversion via the backend.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to numpy float32 arrays

        Raises:
            FileNotFoundError: If no safetensors files found
        """
        ...
