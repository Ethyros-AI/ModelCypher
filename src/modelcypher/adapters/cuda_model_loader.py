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

"""CUDA/PyTorch Model Loader implementing ModelLoaderPort.

This adapter wraps PyTorch and HuggingFace Transformers for model loading
on CUDA GPUs, implementing the ModelLoaderPort protocol for hexagonal
architecture compliance.

Usage:
    from modelcypher.adapters.cuda_model_loader import CUDAModelLoader

    loader = CUDAModelLoader()
    model, tokenizer = loader.load_model_for_training("/path/to/model")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.model_loader import ModelLoaderPort
from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code

if TYPE_CHECKING:
    from modelcypher.adapters.training.mlx.lora import LoRASettings

logger = logging.getLogger(__name__)


class CUDAModelLoader(ModelLoaderPort):
    """PyTorch/CUDA implementation of ModelLoaderPort.

    Loads models using HuggingFace Transformers with PyTorch backend.
    Keeps all tensors on CUDA GPU.

    Requires: pip install torch transformers safetensors
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize CUDA model loader.

        Args:
            device: PyTorch device ("cuda", "cuda:0", "cuda:1", etc.)
        """
        self.device = device
        try:
            import torch

            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. ModelCypher requires GPU acceleration. "
                    "CPU fallback is not supported. Ensure CUDA is properly installed."
                )
            self._available = True
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch"
            ) from exc

    @property
    def available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._available

    def load_model_for_training(
        self,
        model_path: str,
        lora_config: "LoRASettings | None" = None,
        adapter_path: str | None = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer for training or inference.

        Args:
            model_path: Path to model directory
            lora_config: Optional LoRA settings to apply
            adapter_path: Optional adapter directory to load

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers not available. Install: pip install transformers"
            ) from exc

        logger.info("Loading model from %s with CUDA backend...", model_path)
        adapter_dir = Path(adapter_path).expanduser().resolve() if adapter_path else None
        if lora_config is not None and adapter_dir is not None:
            raise ValueError("Cannot combine lora_config with adapter_path")

        warn_trust_remote_code(logger)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code_enabled()
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch.bfloat16,
            device_map=self.device,
            trust_remote_code=trust_remote_code_enabled(),
        )

        if adapter_dir is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required to load adapters on CUDA. Install: pip install peft"
                ) from exc
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            logger.info("Loaded adapter from %s", adapter_dir)
        elif lora_config is not None:
            logger.warning("LoRA config provided but not yet implemented for CUDA loader")

        return model, tokenizer



    def load_weights(self, model_path: str) -> dict[str, Any]:
        """Load model weights as native PyTorch tensors (GPU-accelerated).

        Args:
            model_path: Path to model directory with safetensors

        Returns:
            Dictionary mapping weight names to torch.Tensor (runs on CUDA)
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        from safetensors import safe_open

        model_dir = Path(model_path)
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        weights: dict[str, Any] = {}
        for sf_path in safetensor_files:
            # Use PyTorch framework for native tensor loading
            with safe_open(sf_path, framework="pt", device=self.device) as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        return weights


def get_model_loader(device: str = "cuda") -> CUDAModelLoader:
    """Get the CUDA model loader instance."""
    return CUDAModelLoader(device=device)


__all__ = ["CUDAModelLoader", "get_model_loader"]
