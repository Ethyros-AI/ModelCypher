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

"""Model loading infrastructure for training and inference."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend, get_mlx_probe_error, probe_mlx_available
if TYPE_CHECKING:
    from modelcypher.core.domain.training.lora_mlx import LoRASettings

logger = logging.getLogger(__name__)
mlx_lm_load: Any | None = None


def load_model(
    model_path: str | Path,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load model and tokenizer for inference.

    This is a simple wrapper around mlx_lm.load for inference use cases.
    For training with LoRA, use load_model_for_training() instead.

    Parameters
    ----------
    model_path : str or Path
        Path to model directory.
    adapter_path : str or None
        Optional adapter directory to load (e.g., LoRA weights).

    Returns
    -------
    tuple of (model, tokenizer)
        The loaded model and tokenizer ready for inference.

    Raises
    ------
    RuntimeError
        If MLX is not available.
    ImportError
        If mlx_lm is not installed.
    """
    _ensure_mlx()

    model_path = Path(model_path).expanduser().resolve()
    adapter_dir = Path(adapter_path).expanduser().resolve() if adapter_path else None

    # Check model type from config
    config_path = model_path / "config.json"
    model_type = "unknown"
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

            if adapter_dir is not None:
                model, tokenizer = mlx_vlm_load(
                    str(model_path),
                    adapter_path=str(adapter_dir),
                )
            else:
                model, tokenizer = mlx_vlm_load(str(model_path))
            return model, tokenizer

        except ImportError as e:
            raise ImportError(
                f"mlx_vlm is required to load {model_type} models. "
                f"Install with: poetry add mlx-vlm"
            ) from e

    # Standard text model
    try:
        from mlx_lm import load as _mlx_lm_load
    except ModuleNotFoundError as exc:
        raise ImportError(
            "mlx_lm is required to load text models. "
            "Install with: pip install mlx-lm"
        ) from exc

    if adapter_dir is not None:
        model, tokenizer = _mlx_lm_load(
            str(model_path),
            adapter_path=str(adapter_dir),
        )
    else:
        model, tokenizer = _mlx_lm_load(str(model_path))

    return model, tokenizer


def _ensure_mlx() -> tuple[Any, Any]:
    if not probe_mlx_available(explicit=True):
        detail = get_mlx_probe_error() or "Unknown MLX initialization error"
        raise RuntimeError(f"MLX runtime unavailable: {detail}")

    import mlx.core as mx
    import mlx.nn as nn

    return mx, nn


def load_model_for_training(
    model_path: str,
    lora_settings: "LoRASettings | None" = None,
    adapter_path: str | None = None,
) -> tuple["nn.Module", Any]:
    """Load model and tokenizer for training.

    Parameters
    ----------
    model_path : str
        Path to model directory.
    lora_settings : LoRASettings or None
        Optional LoRA settings for adapter training.
    adapter_path : str or None
        Optional adapter directory to load (e.g., LoRA weights).

    Returns
    -------
    tuple of (nn.Module, any)
        Model with optional LoRA adapters and tokenizer.
        Base weights are frozen if LoRA is used.
    """
    logger.info("Loading model from %s", model_path)
    adapter_dir = Path(adapter_path).expanduser().resolve() if adapter_path else None
    if lora_settings is not None and adapter_dir is not None:
        raise ValueError("Cannot combine lora_settings with adapter_path")
    _ensure_mlx()

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

            if adapter_dir is not None:
                try:
                    model, tokenizer = mlx_vlm_load(
                        model_path,
                        adapter_path=str(adapter_dir),
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "mlx_vlm.load does not support adapter_path for multimodal models."
                    ) from exc
            else:
                model, tokenizer = mlx_vlm_load(model_path)

            # Count parameters for logging
            from mlx.utils import tree_flatten

            flat_params = tree_flatten(model.parameters())
            all_params = sum(param.size for _, param in flat_params)

            logger.info("Multimodal model loaded: %s, ~%d total parameters", model_type, all_params)

            # Note: LoRA on VL models requires special handling
            if lora_settings is not None:
                logger.warning(
                    "LoRA on multimodal models may require architecture-specific adapter placement. "
                    "Consider using text-only model for LoRA training."
                )
                # For now, we freeze and apply LoRA to language backbone only
                from modelcypher.core.domain.training.lora_mlx import apply_lora_to_model

                model.freeze()
                model = apply_lora_to_model(model, lora_settings)

            return model, tokenizer

        except ImportError as e:
            raise ImportError(
                f"mlx_vlm is required to load {model_type} models. "
                f"Install with: poetry add mlx-vlm"
            ) from e
        except Exception as e:
            # Do NOT silently fallback to stripping vision tower
            # That would produce scientifically invalid results
            raise RuntimeError(
                f"Failed to load multimodal model {model_type}: {e}. "
                f"Ensure mlx_vlm is properly installed and the model is compatible."
            ) from e
    else:
        try:
            from mlx_lm import load as _mlx_lm_load
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "mlx_lm is required to load text models for training. "
                "Install with: pip install mlx-lm"
            ) from exc
        if adapter_dir is not None:
            try:
                model, tokenizer = _mlx_lm_load(
                    model_path,
                    adapter_path=str(adapter_dir),
                )
            except TypeError as exc:
                raise RuntimeError(
                    "mlx_lm.load does not support adapter_path for this model."
                ) from exc
        else:
            model, tokenizer = _mlx_lm_load(model_path)

    if lora_settings is not None:
        # Freeze base weights first
        model.freeze()

        logger.info("Injecting LoRA adapters (rank=%d)", lora_settings.rank)
        from modelcypher.core.domain.training.lora_mlx import apply_lora_to_model

        model = apply_lora_to_model(model, lora_settings)

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


