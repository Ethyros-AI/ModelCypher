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

"""MLX-based model loader implementing ModelLoaderPort.

This adapter wraps the existing model loading functions to implement
the ModelLoaderPort protocol for hexagonal architecture compliance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    def load_weights_as_numpy(self, model_path: str) -> dict[str, Any]:
        """Load model weights as numpy-compatible arrays.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to numpy-compatible float32 arrays
        """
        # Import here to avoid circular imports and MLX dependency at module level
        from modelcypher.adapters.model_loader import (
            load_weights_as_numpy as _load_weights_as_numpy,
        )

        return _load_weights_as_numpy(model_path)

    def load_weights(self, model_path: str) -> dict[str, Any]:
        """Load model weights as native MLX arrays (GPU-accelerated).

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to mx.array (runs on GPU)
        """
        from pathlib import Path

        import mlx.core as mx

        model_dir = Path(model_path)

        # Find safetensors files
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensor_files:
            # mx.load handles safetensors natively and keeps as mx.array
            file_weights = mx.load(str(sf_path))
            weights.update(file_weights)

        # Force evaluation of all weights to make them concrete tensors
        # This prevents lazy computation graphs from later stages causing issues
        mx.eval(*weights.values())

        return weights
