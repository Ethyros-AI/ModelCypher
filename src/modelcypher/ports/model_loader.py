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

"""Model loading port for hexagonal architecture.

This port abstracts model loading operations used for:
- Training preparation (model + tokenizer loading)
- Activation extraction for geometry analysis
- Weight loading for merge operations
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class ModelLoaderPort(Protocol):
    """Port for loading models and their components.

    Implementations handle backend-specific model loading
    while domain code depends only on this abstract interface.

    NOTE: LoRA injection is NOT part of model loading. Use TrainingPort.apply_lora()
    after loading the model. Separation of concerns: loading vs training.
    """

    def load_model(
        self,
        model_path: str,
        adapter_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer.

        Args:
            model_path: Path to model directory
            adapter_path: Optional adapter directory to load (e.g., saved LoRA)

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ImportError: If required backend is unavailable
            RuntimeError: If model loading fails
        """
        ...



    def load_weights(self, model_path: str) -> "dict[str, Any]":
        """Load model weights as native backend arrays (GPU-accelerated).

        Returns arrays in the backend's native format. Operations on these
        arrays run on the configured accelerator.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to native backend arrays

        Raises:
            FileNotFoundError: If no safetensors files found
        """
        ...

    def iter_weights(self, model_path: str) -> "Iterator[tuple[str, Any]]":
        """Iterate model weights as (name, native backend array) pairs.

        This API allows streaming analysis without materializing every tensor
        in memory at once.

        Args:
            model_path: Path to model directory with safetensors

        Yields:
            Tuples of (weight_name, native backend array)

        Raises:
            FileNotFoundError: If no safetensors files found
        """
        ...


__all__ = ["ModelLoaderPort"]
