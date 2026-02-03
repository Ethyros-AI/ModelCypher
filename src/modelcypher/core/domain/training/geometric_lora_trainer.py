# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometric LoRA trainer.

Trains LoRA adapters where all configuration is derived from geometry:
- Target modules: spectral decay < 100×
- Rank: min(tail_dims) across targets
- Scale: σ_k per layer (via spectral normalization)

No hyperparameters except learning rate and epochs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .geometric_lora import (
    LayerGeometry,
    analyze_model_geometry,
    apply_geometric_lora,
    compute_geometric_rank,
    get_lora_parameters,
    select_target_modules,
)

logger = logging.getLogger(__name__)


@dataclass
class GeometricLoRAConfig:
    """Configuration derived from model geometry."""

    target_modules: list[str]
    rank: int
    geometries: dict[str, LayerGeometry]

    # Training params (these ARE hyperparameters - task dependent)
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 4

    def to_dict(self) -> dict:
        return {
            "target_modules": self.target_modules,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "layer_geometries": {
                key: {
                    "sigma_k": g.sigma_k,
                    "sigma_max": g.sigma_max,
                    "decay_ratio": g.decay_ratio,
                    "tail_dims": g.tail_dims,
                }
                for key, g in self.geometries.items()
                if key in self.target_modules
            },
        }


@dataclass
class GeometricLoRAResult:
    """Result of geometric LoRA training."""

    success: bool
    config: Optional[GeometricLoRAConfig] = None
    adapter_path: Optional[Path] = None
    final_loss: float = 0.0
    training_time_seconds: float = 0.0
    error: Optional[str] = None


def derive_config_from_geometry(
    model,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 4,
) -> GeometricLoRAConfig:
    """Derive LoRA configuration from model geometry.

    Args:
        model: The loaded model
        learning_rate: Learning rate (task-dependent)
        epochs: Number of epochs (task-dependent)
        batch_size: Batch size (hardware-dependent)

    Returns:
        GeometricLoRAConfig with all geometry-derived parameters
    """
    logger.info("Analyzing model geometry...")

    # Compute geometry for all layers
    geometries = analyze_model_geometry(model)

    if not geometries:
        raise ValueError("No targetable layers found in model")

    # Select targets based on spectral decay
    target_modules = select_target_modules(geometries)

    if not target_modules:
        raise ValueError("No layers with decay_ratio < 100 found")

    # Derive rank from geometry
    rank = compute_geometric_rank(geometries, target_modules)

    logger.info(
        "Geometry-derived config: %d targets, rank=%d",
        len(target_modules), rank
    )

    return GeometricLoRAConfig(
        target_modules=target_modules,
        rank=rank,
        geometries=geometries,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )


def train_geometric_lora(
    model,
    tokenizer,
    training_data: list[dict],
    output_path: Path,
    config: GeometricLoRAConfig,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> GeometricLoRAResult:
    """Train a geometric LoRA adapter.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        training_data: List of training examples (prompt/completion format)
        output_path: Where to save the adapter
        config: Geometry-derived configuration
        progress_callback: Optional callback for progress updates

    Returns:
        GeometricLoRAResult with training outcomes
    """
    start_time = time.time()

    try:
        # Apply geometric LoRA to model
        lora_layers = apply_geometric_lora(
            model,
            config.geometries,
            config.target_modules,
            config.rank,
        )

        if not lora_layers:
            return GeometricLoRAResult(
                success=False,
                error="No LoRA layers were applied",
            )

        # Get trainable parameters
        lora_params = get_lora_parameters(lora_layers)

        logger.info("Training %d LoRA parameters", len(lora_params) * 2)

        # Tokenize training data
        tokenized = _tokenize_data(training_data, tokenizer)

        if not tokenized:
            return GeometricLoRAResult(
                success=False,
                error="No valid training data after tokenization",
            )

        # Create optimizer (only for LoRA params)
        optimizer = optim.AdamW(learning_rate=config.learning_rate)

        # Training loop
        final_loss = 0.0
        total_steps = 0

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(tokenized), config.batch_size):
                batch = tokenized[batch_start:batch_start + config.batch_size]

                # Forward and backward pass
                loss, grads = _compute_loss_and_grads(model, batch, lora_layers)

                # Update LoRA parameters
                optimizer.update(model, grads)
                mx.eval(model.parameters())

                epoch_loss += float(loss)
                n_batches += 1
                total_steps += 1

                if progress_callback:
                    progress_callback({
                        "epoch": epoch,
                        "step": total_steps,
                        "loss": float(loss),
                    })

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            logger.info("Epoch %d: loss=%.4f", epoch, avg_loss)
            final_loss = avg_loss

        # Save adapter
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        _save_geometric_adapter(
            lora_layers,
            config,
            output_path,
        )

        training_time = time.time() - start_time

        return GeometricLoRAResult(
            success=True,
            config=config,
            adapter_path=output_path,
            final_loss=final_loss,
            training_time_seconds=training_time,
        )

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return GeometricLoRAResult(
            success=False,
            error=str(e),
            training_time_seconds=time.time() - start_time,
        )


def _tokenize_data(data: list[dict], tokenizer, max_length: int = 512) -> list[mx.array]:
    """Tokenize training data."""
    tokenized = []

    for sample in data:
        if "prompt" in sample and "completion" in sample:
            text = sample["prompt"] + sample["completion"]
        elif "text" in sample:
            text = sample["text"]
        elif "input" in sample and "output" in sample:
            text = sample["input"] + sample["output"]
        else:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(mx.array(tokens))

    return tokenized


def _compute_loss_and_grads(model, batch: list[mx.array], lora_layers):
    """Compute loss and gradients for a batch."""

    def loss_fn(model):
        total_loss = 0.0
        n_tokens = 0

        for tokens in batch:
            # Shift for language modeling
            input_ids = tokens[:-1]
            target_ids = tokens[1:]

            # Forward pass
            logits = model(input_ids[None, :])

            # Cross entropy loss
            logits_flat = logits[0]  # [seq_len, vocab]
            loss = nn.losses.cross_entropy(
                logits_flat,
                target_ids,
                reduction="sum",
            )
            total_loss += loss
            n_tokens += len(target_ids)

        return total_loss / n_tokens if n_tokens > 0 else mx.array(0.0)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    return loss, grads


def _save_geometric_adapter(
    lora_layers: dict,
    config: GeometricLoRAConfig,
    output_path: Path,
):
    """Save the geometric LoRA adapter."""
    # Collect weights
    weights = {}
    for layer_key, lora_layer in lora_layers.items():
        weights[f"{layer_key}.lora_a"] = lora_layer.lora_a
        weights[f"{layer_key}.lora_b"] = lora_layer.lora_b

    # Save weights
    weights_path = output_path / "lora_weights.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    # Save config with geometry info
    config_dict = {
        "type": "geometric_lora",
        "rank": config.rank,
        "target_modules": config.target_modules,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        # Store σ_k for each layer (needed for inference)
        "layer_sigma_k": {
            key: config.geometries[key].sigma_k
            for key in config.target_modules
        },
        # Store full geometry for reference
        "geometry": config.to_dict()["layer_geometries"],
    }

    config_path = output_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Saved geometric adapter to %s", output_path)


__all__ = [
    "GeometricLoRAConfig",
    "GeometricLoRAResult",
    "derive_config_from_geometry",
    "train_geometric_lora",
]
