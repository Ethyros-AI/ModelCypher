"""MLX-based model loader implementing ModelLoaderPort.

This adapter wraps the existing model loading functions to implement
the ModelLoaderPort protocol for hexagonal architecture compliance.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from modelcypher.core.domain.training.lora_mlx import LoRAConfig

class MLXModelLoader:
    """MLX-based implementation of ModelLoaderPort.

    Wraps the existing model_loader functions to provide a clean interface
    for dependency injection.
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
            Tuple of (model, tokenizer)
        """
        # Import here to avoid circular imports and MLX dependency at module level
        from modelcypher.adapters.model_loader import (
            load_model_for_training as _load_model_for_training,
        )

        return _load_model_for_training(model_path, lora_config)

    def load_weights_as_numpy(self, model_path: str) -> dict[str, np.ndarray]:
        """Load model weights as numpy arrays.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to numpy float32 arrays
        """
        # Import here to avoid circular imports and MLX dependency at module level
        from modelcypher.adapters.model_loader import (
            load_weights_as_numpy as _load_weights_as_numpy,
        )

        return _load_weights_as_numpy(model_path)
