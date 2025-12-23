"""Model loading infrastructure for training."""
import logging
from typing import Tuple

import mlx.nn as nn
from mlx_lm import load as mlx_lm_load

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
    model, tokenizer = mlx_lm_load(model_path)
    
    if lora_config is not None:
        logger.info("Injecting LoRA adapters (rank=%d)", lora_config.rank)
        model = apply_lora_to_model(model, lora_config)
        
        # Freeze base weights
        model.freeze()
        
        # Unfreeze LoRA weights
        trainable_params = 0
        all_params = 0
        
        # We need to unfreeze the lora_a and lora_b parameters
        # In MLX, we can do this by traversing the modules
        for name, module in model.named_modules():
            if "lora" in name.lower():
                module.unfreeze()
        
        # Count parameters for logging
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
        
        # DEBUG
        trainable = list(model.trainable_parameters().keys())
        logger.info("DEBUG: Trainable parameters: %s", trainable)
    
    return model, tokenizer
