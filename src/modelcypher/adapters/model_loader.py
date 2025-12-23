"""Model loading infrastructure for training."""
import logging
import json
import os
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_lm_load
from mlx_lm.utils import _get_classes, load_model as mlx_lm_load_model

from modelcypher.core.domain.training.lora import (
    LoRAConfig,
    apply_lora_to_model,
)

logger = logging.getLogger(__name__)

def load_model_for_training(
    model_path: str,
    lora_config: LoRAConfig | None = None,
) -> Tuple[nn.Module, any]:
    """
    Load model and tokenizer for training.
    
    Returns tokenized and model with LoRA adapters injected if config provided.
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

    if model_type == "glm4v":
        logger.info("GLM-4V detected, loading full multimodal model using mlx_vlm")
        try:
            from mlx_vlm import load as mlx_vlm_load
            model, processor = mlx_vlm_load(model_path)
            tokenizer = processor.tokenizer
            
            # Count parameters for logging
            trainable_params = 0
            all_params = 0
            from mlx.utils import tree_flatten
            flat_params = tree_flatten(model.parameters())
            for name, param in flat_params:
                all_params += param.size
            
            logger.info("Multimodal Model loaded: ~%d total parameters", all_params)
            return model, tokenizer
        except Exception as e:
            logger.error("Failed to load multimodal GLM-4V: %s", str(e))
            logger.info("Falling back to isolated language core...")
            # (Fallback logic remains below if multimodal load fails)
            
        # Extract text config
        text_config = full_config.get("text_config", {})
        if not text_config:
            logger.error("No text_config found in GLM-4V config")
            raise ValueError("Invalid GLM-4V config")
            
        # Standardize for mlx_lm
        text_config["model_type"] = "glm4"
        
        # Flatten rope parameters if they are nested (GLM-4V specific)
        rope_params = text_config.get("rope_parameters", {})
        if rope_params:
            if "partial_rotary_factor" in rope_params:
                text_config["partial_rotary_factor"] = rope_params["partial_rotary_factor"]
            if "rope_theta" in rope_params:
                text_config["rope_theta"] = rope_params["rope_theta"]
        
        # Ensure head_dim is set
        if "head_dim" not in text_config:
            hidden_size = text_config.get("hidden_size", 4096)
            num_heads = text_config.get("num_attention_heads", 32)
            text_config["head_dim"] = hidden_size // num_heads
            
        # Ensure all required GLM4 args are present
        text_config.setdefault("partial_rotary_factor", 0.5)
        text_config.setdefault("rope_theta", 500000.0)
        
        # Instantiate model
        model_class, model_args_class = _get_classes(config=text_config)
        model_args = model_args_class.from_dict(text_config)
        model = model_class(model_args)
        
        # Apply quantization if present in config
        quant_config = full_config.get("quantization") or full_config.get("quantization_config")
        if quant_config:
            from mlx_lm.utils import quantize_model
            logger.info("Applying quantization to isolated model: %s", quant_config)
            model, _ = quantize_model(
                model, 
                quant_config, 
                group_size=quant_config.get("group_size", 64),
                bits=quant_config.get("bits", 4)
            )
        
        # Load weights and strip "language_model." prefix
        weight_files = list(Path(model_path).glob("*.safetensors"))
        weights = {}
        for wf in weight_files:
            raw_weights = mx.load(str(wf))
            for k, v in raw_weights.items():
                if k.startswith("language_model."):
                    new_k = k[len("language_model."):]
                    weights[new_k] = v
                else:
                    weights[k] = v
        
        # Filter weights to only include keys that exist in the model
        from mlx.utils import tree_flatten
        model_params = dict(tree_flatten(model.parameters()))
        filtered_weights = {k: v for k, v in weights.items() if k in model_params}
        
        # Load the remapped and filtered weights
        model.load_weights(list(filtered_weights.items()))
        
        # Get tokenizer
        from mlx_lm.utils import load_tokenizer
        tokenizer = load_tokenizer(model_path)
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
            all_params
        )
    
    return model, tokenizer
