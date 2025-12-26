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

"""Model loading infrastructure for training."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_lm_load

from modelcypher.core.domain._backend import get_default_backend

from modelcypher.core.domain.training.lora_mlx import (
    LoRAConfig,
    apply_lora_to_model,
)

logger = logging.getLogger(__name__)


def load_model_for_training(
    model_path: str,
    lora_config: LoRAConfig | None = None,
) -> tuple[nn.Module, any]:
    """Load model and tokenizer for training.

    Parameters
    ----------
    model_path : str
        Path to model directory.
    lora_config : LoRAConfig or None
        Optional LoRA configuration for adapter training.

    Returns
    -------
    tuple of (nn.Module, any)
        Model with optional LoRA adapters and tokenizer.
        Base weights are frozen if LoRA is used.
    """
    logger.info("Loading model for training from %s", model_path)

    # Check model type from config
    config_path = Path(model_path) / "config.json"
    model_type = "unknown"
    full_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                full_config = json.load(f)
                model_type = full_config.get("model_type", "unknown")
        except Exception:
            pass

    # Multimodal VL model types that require mlx_vlm
    MULTIMODAL_TYPES = {"glm4v", "qwen2_vl", "llava", "paligemma", "idefics2", "phi3_v"}

    if model_type in MULTIMODAL_TYPES:
        logger.info("Multimodal model detected (%s), loading with mlx_vlm", model_type)
        try:
            from mlx_vlm import load as mlx_vlm_load

            model, tokenizer = mlx_vlm_load(model_path)

            # Count parameters for logging
            from mlx.utils import tree_flatten

            flat_params = tree_flatten(model.parameters())
            all_params = sum(param.size for _, param in flat_params)

            logger.info("Multimodal model loaded: %s, ~%d total parameters", model_type, all_params)

            # Note: LoRA on VL models requires special handling
            if lora_config is not None:
                logger.warning(
                    "LoRA on multimodal models may require architecture-specific adapter placement. "
                    "Consider using text-only model for LoRA training."
                )
                # For now, we freeze and apply LoRA to language backbone only
                model.freeze()
                model = apply_lora_to_model(model, lora_config)

            return model, tokenizer

        except ImportError as e:
            raise ImportError(
                f"mlx_vlm is required to load {model_type} models. "
                f"Install with: pip install mlx-vlm"
            ) from e
        except Exception as e:
            # Do NOT silently fallback to stripping vision tower
            # That would produce scientifically invalid results
            raise RuntimeError(
                f"Failed to load multimodal model {model_type}: {e}. "
                f"Ensure mlx_vlm is properly installed and the model is compatible."
            ) from e
    else:
        model, tokenizer = mlx_lm_load(model_path)

    if lora_config is not None:
        # Freeze base weights first
        model.freeze()

        logger.info("Injecting LoRA adapters (rank=%d)", lora_config.rank)
        model = apply_lora_to_model(model, lora_config)

        # Count parameters for logging
        trainable_params = 0
        all_params = 0

        from mlx.utils import tree_flatten

        flat_params = tree_flatten(model.parameters())
        for name, param in flat_params:
            all_params += param.size
            if "lora" in name.lower():
                trainable_params += param.size

        logger.info(
            "LoRA: ~%d trainable parameters (%.2f%% of %d total)",
            trainable_params,
            (trainable_params / all_params) * 100 if all_params > 0 else 0,
            all_params,
        )

    return model, tokenizer


def load_weights_as_numpy(model_path: str) -> "dict[str, Any]":  # noqa: F821
    """Load model weights as numpy-compatible arrays while preserving dtype.

    Uses the backend's to_numpy() method for conversion at the boundary.
    bfloat16 is promoted to float32; integer/quantized dtypes are preserved.
    """
    import glob
    from pathlib import Path

    backend = get_default_backend()
    path = Path(model_path)
    safetensor_files = glob.glob(str(path / "*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    weights: dict[str, Any] = {}
    try:
        from safetensors.numpy import load_file

        for sf_path in safetensor_files:
            weights.update(load_file(sf_path))
        return weights
    except Exception as e:
        logger.warning(
            "safetensors numpy load failed (%s); falling back to MLX loader",
            e,
        )

    for sf_path in safetensor_files:
        mlx_weights = mx.load(sf_path)
        for key, value in mlx_weights.items():
            weights[key] = backend.to_numpy(value)

    return weights
