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

"""Unified model loading - ONE loader that uses Backend.

This is THE model loader. It detects the backend and loads models appropriately.
No mlx_model_loader.py, jax_model_loader.py, cuda_model_loader.py needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.model_loader import ModelLoaderPort

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class ModelLoader(ModelLoaderPort):
    """Unified model loader - uses Backend for tensor operations.

    Handles MLX, JAX, and CUDA models through a single interface.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize with optional backend.

        Args:
            backend: If None, auto-detects from platform.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self._backend = backend
        self._backend_type = type(backend).__name__.lower().replace("backend", "")

    def load_model(
        self,
        model_path: str,
        adapter_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer.

        Auto-selects loading method based on backend type.

        Args:
            model_path: Path to model directory
            adapter_path: Optional adapter directory to load

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path_obj = Path(model_path).expanduser().resolve()

        # Check model type from config for multimodal detection
        config_path = model_path_obj / "config.json"
        model_type = "unknown"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    model_type = json.load(f).get("model_type", "unknown")
            except Exception:
                pass

        if self._backend_type == "mlx":
            return self._load_mlx(str(model_path_obj), adapter_path, model_type)
        elif self._backend_type == "cuda":
            return self._load_cuda(str(model_path_obj), adapter_path)
        elif self._backend_type == "jax":
            return self._load_jax(str(model_path_obj), adapter_path)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def load_weights(self, model_path: str) -> dict[str, Any]:
        """Load model weights as native backend arrays.

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to backend arrays
        """
        model_dir = Path(model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        weights: dict[str, Any] = {}
        for sf_path in safetensor_files:
            file_weights = self._backend.load_safetensors(str(sf_path))
            weights.update(file_weights)

        self._backend.eval(*weights.values())
        return weights

    def _load_mlx(self, model_path: str, adapter_path: str | None, model_type: str) -> tuple[Any, Any]:
        """Load model using MLX/mlx_lm."""
        from modelcypher.backends.mlx_probe import get_mlx_probe_error, probe_mlx_available

        if not probe_mlx_available(explicit=True):
            detail = get_mlx_probe_error() or "Unknown MLX initialization error"
            raise RuntimeError(f"MLX runtime unavailable: {detail}")

        adapter_dir = Path(adapter_path).expanduser().resolve() if adapter_path else None

        # Multimodal models
        MULTIMODAL_TYPES = {"glm4v", "qwen2_vl", "llava", "paligemma", "idefics2", "phi3_v"}
        if model_type in MULTIMODAL_TYPES:
            try:
                from mlx_vlm import load as mlx_vlm_load
                if adapter_dir:
                    return mlx_vlm_load(model_path, adapter_path=str(adapter_dir))
                return mlx_vlm_load(model_path)
            except ImportError as e:
                raise ImportError(f"mlx_vlm required for {model_type}. Install: pip install mlx-vlm") from e

        # Standard text models
        try:
            from mlx_lm import load as mlx_lm_load
        except ImportError as e:
            raise ImportError("mlx_lm required. Install: pip install mlx-lm") from e

        if adapter_dir:
            try:
                return mlx_lm_load(model_path, adapter_path=str(adapter_dir))
            except AttributeError as exc:
                if "num_layers" in str(exc):
                    from modelcypher.adapters.training.mlx.self_reflection import load_self_reflection_adapters
                    return load_self_reflection_adapters(model_path, str(adapter_dir))
                raise
        return mlx_lm_load(model_path)

    def _load_cuda(self, model_path: str, adapter_path: str | None) -> tuple[Any, Any]:
        """Load model using PyTorch/transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("torch and transformers required. Install: pip install torch transformers") from e

        from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code

        warn_trust_remote_code(logger)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code_enabled())
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=trust_remote_code_enabled(),
        )

        if adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
            except ImportError as e:
                raise ImportError("peft required for adapters. Install: pip install peft") from e

        return model, tokenizer

    def _load_jax(self, model_path: str, adapter_path: str | None) -> tuple[Any, Any]:
        """Load model using JAX/Flax/transformers."""
        try:
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        except ImportError as e:
            raise ImportError("transformers with Flax required. Install: pip install transformers flax") from e

        from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code

        warn_trust_remote_code(logger)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code_enabled())

        try:
            model = FlaxAutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code_enabled())
        except Exception:
            model = FlaxAutoModelForCausalLM.from_pretrained(model_path, from_pt=True, trust_remote_code=trust_remote_code_enabled())

        if adapter_path:
            raise NotImplementedError("JAX adapter loading not yet implemented")

        return model, tokenizer


# Convenience functions for backwards compatibility
def load_model(model_path: str | Path, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load model and tokenizer."""
    return ModelLoader().load_model(str(model_path), adapter_path)


def load_model_for_training(model_path: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load model for training (same as load_model)."""
    return ModelLoader().load_model(model_path, adapter_path)


def get_model_loader(backend: "Backend | None" = None) -> ModelLoader:
    """Get a model loader instance."""
    return ModelLoader(backend)


__all__ = ["ModelLoader", "load_model", "load_model_for_training", "get_model_loader"]
