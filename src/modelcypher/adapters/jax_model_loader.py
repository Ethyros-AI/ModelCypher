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

"""JAX Model Loader implementing ModelLoaderPort.

This adapter wraps JAX and HuggingFace Transformers for model loading
on TPU/GPU, implementing the ModelLoaderPort protocol for hexagonal
architecture compliance.

Usage:
    from modelcypher.adapters.jax_model_loader import JAXModelLoader

    loader = JAXModelLoader()
    model, tokenizer = loader.load_model_for_training("/path/to/model")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.model_loader import ModelLoaderPort

if TYPE_CHECKING:
    from modelcypher.core.domain.training.lora_mlx import LoRASettings

logger = logging.getLogger(__name__)


class JAXModelLoader(ModelLoaderPort):
    """JAX implementation of ModelLoaderPort.

    Loads models using HuggingFace Transformers with Flax/JAX backend.
    Supports TPU, GPU, and CPU.

    Requires: pip install jax jaxlib transformers flax safetensors
    """

    def __init__(self) -> None:
        """Initialize JAX model loader."""
        try:
            import jax
            import jax.numpy as jnp

            self.jax = jax
            self.jnp = jnp
            self._available = True
        except ImportError:
            self._available = False
            self.jax = None
            self.jnp = None
            logger.warning("JAX not available. Install with: pip install jax jaxlib")

    @property
    def available(self) -> bool:
        """Check if JAX backend is available."""
        return self._available

    def load_model_for_training(
        self,
        model_path: str,
        lora_config: "LoRASettings | None" = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer for training or inference.

        Args:
            model_path: Path to model directory
            lora_config: Optional LoRA settings to apply

        Returns:
            Tuple of (model, tokenizer)
        """
        if not self._available:
            raise RuntimeError("JAX not available. Install: pip install jax jaxlib")

        try:
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "transformers with Flax not available. "
                "Install: pip install transformers flax"
            ) from exc

        logger.info("Loading model from %s with JAX backend...", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Try Flax model first
        try:
            model = FlaxAutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("Flax model loading failed, trying PyTorch conversion: %s", e)
            # Fall back to loading PyTorch weights and converting
            model = FlaxAutoModelForCausalLM.from_pretrained(
                model_path,
                from_pt=True,
                trust_remote_code=True,
            )

        if lora_config is not None:
            logger.warning("LoRA config provided but not yet implemented for JAX loader")

        return model, tokenizer

    def load_weights_as_numpy(self, model_path: str) -> dict[str, Any]:
        """Load model weights as CPU JAX arrays.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to float32 jax.Array on CPU
        """
        if not self._available:
            raise RuntimeError("JAX not available. Install: pip install jax jaxlib")

        from safetensors import safe_open

        model_dir = Path(model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        cpu_device = None
        try:
            cpu_devices = self.jax.devices("cpu")
            if cpu_devices:
                cpu_device = cpu_devices[0]
        except Exception:
            cpu_device = None

        weights: dict[str, Any] = {}
        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="flax") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    array = self.jnp.asarray(tensor, dtype=self.jnp.float32)
                    if cpu_device is not None:
                        array = self.jax.device_put(array, cpu_device)
                    weights[key] = array

        return weights

    def load_weights(self, model_path: str) -> dict[str, Any]:
        """Load model weights as native JAX arrays (GPU/TPU-accelerated).

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to jax.Array (runs on accelerator)
        """
        if not self._available:
            raise RuntimeError("JAX not available. Install: pip install jax jaxlib")

        from safetensors import safe_open

        model_dir = Path(model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        weights: dict[str, Any] = {}
        for sf_path in safetensor_files:
            # Use Flax framework for JAX-compatible arrays with bfloat16 support
            with safe_open(sf_path, framework="flax") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        return weights


def get_model_loader() -> JAXModelLoader:
    """Get the JAX model loader instance."""
    return JAXModelLoader()


__all__ = ["JAXModelLoader", "get_model_loader"]
