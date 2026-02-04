# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA Training Service.

High-level service for training LoRA adapters with geometry-guided settings.
All LoRA parameters are derived from the spectral geometry - no hyperparameters.

Architecture:
    - Receives TrainingPort and ModelLoaderPort via dependency injection
    - Uses Backend protocol for tensor operations (via TrainingPort.backend)
    - Domain logic from core/domain/training/geometric_lora.py
    - NO framework-specific imports (mlx, jax, torch)

Usage:
    # At composition root (infrastructure/cli):
    from modelcypher.backends.training.mlx_adapter import MLXTrainingAdapter
    from modelcypher.adapters.model_loader import ModelLoader

    service = LoRATrainingService(
        training_port=MLXTrainingAdapter(),
        model_loader=ModelLoader(),
    )
    result = service.train_lora(...)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    derive_lora_configs,
    select_target_modules,
)
from modelcypher.core.domain.training.types import TrainingSpec
from modelcypher.ports.training import LoRALayerConfig, TrainingPort

if TYPE_CHECKING:
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass
class LoRATrainingResult:
    """Result of LoRA training."""

    success: bool
    adapter_path: Optional[Path] = None

    # Geometry metrics
    final_loss: float = 0.0
    barrier_to_base: float = 0.0
    cka_from_base: float = 0.0

    # Training stats
    steps_trained: int = 0
    samples_used: int = 0
    training_time_seconds: float = 0.0

    # Configs used (geometry-derived)
    lora_configs: list[LoRALayerConfig] = field(default_factory=list)

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "final_loss": self.final_loss,
            "barrier_to_base": self.barrier_to_base,
            "cka_from_base": self.cka_from_base,
            "steps_trained": self.steps_trained,
            "samples_used": self.samples_used,
            "training_time_seconds": self.training_time_seconds,
            "lora_configs": [
                {"layer": c.layer_key, "rank": c.rank, "sigma_k": c.sigma_k}
                for c in self.lora_configs
            ],
            "error": self.error,
        }


class LoRATrainingService:
    """Service for training LoRA adapters with geometry guidance.

    All LoRA parameters are derived from the spectral structure:
    - Target modules: layers with tail_dims > 0
    - Rank: bounded by tail_dims (null-space capacity)
    - Scale: σ_k per layer (smallest significant singular value)

    Dependencies are injected - no framework-specific imports.
    """

    def __init__(
        self,
        training_port: TrainingPort,
        model_loader: "ModelLoaderPort",
    ):
        """Initialize with injected dependencies.

        Args:
            training_port: Adapter implementing TrainingPort (e.g., MLXTrainingAdapter)
            model_loader: Adapter implementing ModelLoaderPort (e.g., ModelLoader)
        """
        self._training = training_port
        self._loader = model_loader
        self._safety_service = None

    @property
    def backend(self):
        """Get compute backend from training port."""
        return self._training.backend

    @property
    def safety_service(self):
        """Lazy-load safety service."""
        if self._safety_service is None:
            from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

            self._safety_service = LoRASafetyService()
        return self._safety_service

    def train_lora(
        self,
        model_path: Path,
        training_data_path: Path,
        output_path: Path,
        *,
        # Training settings
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        # Optional overrides (None = derive from geometry)
        target_modules: Optional[list[str]] = None,
        # Safety settings
        check_barrier: bool = True,
        barrier_threshold: float = 0.03,
        # Callbacks
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> LoRATrainingResult:
        """Train a LoRA adapter with geometry-derived configuration.

        All LoRA parameters (rank, scale, targets) are derived from the
        spectral structure of the base weights. No hyperparameters.

        Args:
            model_path: Path to base model
            training_data_path: Path to JSONL training data
            output_path: Path to save adapter weights
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            target_modules: Override target modules (default: geometry-derived)
            check_barrier: Whether to check mode connectivity barrier
            barrier_threshold: Maximum barrier before warning
            progress_callback: Call with progress updates

        Returns:
            LoRATrainingResult with adapter path and metrics
        """
        import time

        start_time = time.time()

        try:
            # Load model using injected loader
            logger.info("Loading model from %s", model_path)
            model, tokenizer = self._loader.load_model(str(model_path))

            # Get weight matrices for geometry analysis
            weights = self._training.get_weight_matrices(
                model, layer_pattern=r"(q_proj|v_proj)"
            )

            # Analyze geometry and derive LoRA configs (pure domain logic)
            geometries = analyze_weight_geometries(weights, self.backend)
            target_keys = target_modules or select_target_modules(geometries)
            lora_configs = derive_lora_configs(
                geometries, target_keys, adaptive_rank=True
            )

            if not lora_configs:
                return LoRATrainingResult(
                    success=False,
                    error="No targetable layers found (all have tail_dims=0)",
                    training_time_seconds=time.time() - start_time,
                )

            logger.info(
                "Derived LoRA configs: %d layers, ranks=%s",
                len(lora_configs),
                [c.rank for c in lora_configs],
            )

            # Apply LoRA using injected training port
            lora_layers = self._training.apply_lora(model, lora_configs)

            # Freeze base model, unfreeze LoRA
            self._training.freeze_model(model)
            self._training.unfreeze_lora(model, lora_layers)

            # Load and validate training data
            training_samples = self._load_training_data(training_data_path)
            if not training_samples:
                return LoRATrainingResult(
                    success=False,
                    error="No training data found",
                    training_time_seconds=time.time() - start_time,
                )

            logger.info("Loaded %d training samples", len(training_samples))

            # Train
            final_loss, steps_trained = self._run_training_loop(
                model=model,
                tokenizer=tokenizer,
                samples=training_samples,
                lora_configs=lora_configs,
                lora_layers=lora_layers,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
            )

            # Save adapter
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            self._training.save_lora_adapter(
                lora_layers=lora_layers,
                configs=lora_configs,
                output_path=output_path,
                metadata={"model_id": str(model_path)},
            )

            logger.info("Exported adapter to %s", output_path)

            # Compute geometry metrics
            barrier, cka = self._compute_geometry_metrics(
                model_path, output_path, check_barrier
            )

            training_time = time.time() - start_time

            result = LoRATrainingResult(
                success=True,
                adapter_path=output_path,
                final_loss=final_loss,
                barrier_to_base=barrier,
                cka_from_base=cka,
                steps_trained=steps_trained,
                samples_used=len(training_samples),
                training_time_seconds=training_time,
                lora_configs=lora_configs,
            )

            # Safety check
            if check_barrier and barrier > barrier_threshold:
                logger.warning(
                    "Barrier %.4f exceeds threshold %.4f - adapter may fight base model",
                    barrier,
                    barrier_threshold,
                )

            return result

        except Exception as e:
            logger.exception("Training failed: %s", e)
            return LoRATrainingResult(
                success=False,
                error=str(e),
                training_time_seconds=time.time() - start_time,
            )

    def _run_training_loop(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[dict],
        lora_configs: list[LoRALayerConfig],
        lora_layers: dict[str, Any],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        progress_callback: Optional[Callable[[dict], None]],
    ) -> tuple[float, int]:
        """Run the training loop.

        Returns:
            Tuple of (final_loss, steps_trained)
        """
        b = self.backend

        # Tokenize samples
        tokenized = []
        for sample in samples:
            text = self._extract_text(sample)
            if not text:
                continue
            tokens = self._training.tokenize(tokenizer, text, max_length=512)
            tokenized.append(tokens)

        if not tokenized:
            return 0.0, 0

        # Training loop
        global_step = 0
        final_loss = 0.0

        # Per-parameter learning rates (uniform for now)
        param_info = self._training.get_parameter_info(model)
        learning_rates = {p.key: learning_rate for p in param_info}

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_start in range(0, len(tokenized), batch_size):
                batch_tokens = tokenized[batch_start : batch_start + batch_size]
                if not batch_tokens:
                    continue

                # Pad batch to same length
                max_len = max(int(t.shape[0]) for t in batch_tokens)
                padded = []
                for t in batch_tokens:
                    pad_len = max_len - int(t.shape[0])
                    if pad_len > 0:
                        padding = b.zeros((pad_len,), dtype="int32")
                        t = b.concatenate([t, padding], axis=0)
                    padded.append(t)

                input_ids = b.stack(padded, axis=0)
                # Target is input shifted by 1
                target_ids = b.concatenate(
                    [input_ids[:, 1:], b.zeros((input_ids.shape[0], 1), dtype="int32")],
                    axis=1,
                )

                # Compute loss and gradients
                loss, grads = self._training.compute_loss_and_gradients(
                    model, input_ids, target_ids
                )

                # Apply gradients
                self._training.apply_gradients(model, grads, learning_rates)

                # Enforce spectral bounds after update
                self._training.enforce_spectral_bounds(lora_layers, lora_configs)

                epoch_loss += loss
                batch_count += 1
                global_step += 1

                if progress_callback:
                    progress_callback(
                        {
                            "step": global_step,
                            "loss": loss,
                            "epoch": epoch,
                        }
                    )

            if batch_count > 0:
                final_loss = epoch_loss / batch_count
                logger.info("Epoch %d: loss=%.4f", epoch + 1, final_loss)

        return final_loss, global_step

    def _extract_text(self, sample: dict) -> str:
        """Extract text from various training data formats."""
        if "prompt" in sample and "completion" in sample:
            return sample["prompt"] + sample["completion"]
        elif "text" in sample:
            return sample["text"]
        elif "input" in sample and "output" in sample:
            return sample["input"] + sample["output"]
        return ""

    def _load_training_data(self, path: Path) -> list[dict]:
        """Load training data from JSONL file."""
        samples = []
        path = Path(path)

        if not path.exists():
            logger.error("Training data not found: %s", path)
            return []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return samples

    def _compute_geometry_metrics(
        self,
        base_path: Path,
        adapter_path: Path,
        compute: bool = True,
    ) -> tuple[float, float]:
        """Compute barrier and CKA metrics."""
        if not compute:
            return 0.0, 1.0

        try:
            result = self.safety_service.check_barrier_safety(
                base_path=base_path,
                target_path=adapter_path,
                prompts=[
                    "Hello, how are you?",
                    "What is 2+2?",
                    "Explain machine learning.",
                ],
            )
            return result.barrier_height, result.cka_at_target
        except Exception as e:
            logger.warning("Failed to compute geometry metrics: %s", e)
            return 0.0, 1.0


__all__ = ["LoRATrainingService", "LoRATrainingResult"]
