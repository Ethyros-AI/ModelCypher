# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA Training Service.

High-level service for training LoRA adapters with geometry-guided settings
and safety checks. Integrates:

- LoRA settings derivation from model geometry
- Mode connectivity barrier checks
- Training engine with checkpoints
- Adapter export

Usage:
    from modelcypher.core.use_cases.lora_training_service import LoRATrainingService

    service = LoRATrainingService()
    result = service.train_lora(
        model_path=Path("/path/to/model"),
        training_data_path=Path("/path/to/data.jsonl"),
        output_path=Path("/path/to/adapter"),
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

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
    
    # Settings used
    lora_rank: int = 8
    lora_alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
    
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
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "error": self.error,
        }


class LoRATrainingService:
    """Service for training LoRA adapters with geometry guidance.
    
    Wraps the training engine and LoRA utilities to provide a simple
    high-level interface for LoRA training.
    """
    
    def __init__(self):
        """Initialize the training service."""
        self._safety_service = None
    
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
        # LoRA settings (None = derive from geometry)
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        target_modules: Optional[list[str]] = None,
        # Safety settings
        check_barrier: bool = True,
        barrier_threshold: float = 0.03,
        # Callbacks
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> LoRATrainingResult:
        """Train a LoRA adapter.
        
        Args:
            model_path: Path to base model
            training_data_path: Path to JSONL training data
            output_path: Path to save adapter weights
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            rank: LoRA rank (None = derive from geometry)
            alpha: LoRA alpha (None = 2 * rank)
            target_modules: Modules to target (None = q_proj, v_proj)
            check_barrier: Whether to check mode connectivity barrier
            barrier_threshold: Maximum barrier before warning
            progress_callback: Call with progress updates
            
        Returns:
            LoRATrainingResult with adapter path and metrics
        """
        import time
        
        start_time = time.time()
        
        try:
            # Load model
            from modelcypher.adapters.model_loader import load_model_for_training
            
            logger.info("Loading model from %s", model_path)
            model, tokenizer = load_model_for_training(str(model_path))
            
            # Derive LoRA settings from geometry if not provided
            lora_settings = self._derive_or_use_settings(
                model, rank, alpha, target_modules
            )
            
            # Apply LoRA adapters
            from modelcypher.core.domain.training.lora_mlx import (
                apply_lora_to_model,
                export_lora_adapters,
                derive_lora_settings_from_model,
            )
            
            logger.info(
                "Applying LoRA: rank=%d, alpha=%.1f, modules=%s",
                lora_settings.rank, lora_settings.alpha, lora_settings.target_modules
            )
            apply_lora_to_model(model, lora_settings)
            
            # Load training data
            training_samples = self._load_training_data(training_data_path)
            if not training_samples:
                return LoRATrainingResult(
                    success=False,
                    error="No training data found",
                )
            
            logger.info("Loaded %d training samples", len(training_samples))
            
            # Create training config
            from modelcypher.core.domain.training.types import TrainingSpec
            
            job_id = str(uuid.uuid4())[:8]
            config = TrainingSpec(
                job_id=job_id,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=str(output_path.parent),
                checkpoint_steps=100,
            )
            
            # Create optimizer
            import mlx.optimizers as optim
            optimizer = optim.AdamW(learning_rate=learning_rate)
            
            # Create data provider
            data_provider = self._create_data_provider(
                training_samples, tokenizer, batch_size
            )
            
            # Train
            from modelcypher.core.domain.training.engine_mlx import TrainingEngine
            
            engine = TrainingEngine()
            final_progress = None
            
            def on_progress(progress):
                nonlocal final_progress
                final_progress = progress
                if progress_callback:
                    progress_callback({
                        "step": progress.global_step,
                        "loss": progress.loss,
                        "epoch": progress.epoch_index,
                    })
            
            logger.info("Starting training: %d epochs, %d samples", epochs, len(training_samples))
            engine.train(
                job_id=job_id,
                config=config,
                model=model,
                optimizer=optimizer,
                data_provider=data_provider,
                progress_callback=on_progress,
            )
            
            # Export adapters
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            adapter_file = output_path / "adapter.safetensors"
            
            export_result = export_lora_adapters(
                model=model,
                output_path=adapter_file,
                settings=lora_settings,
                model_id=str(model_path),
            )
            
            logger.info("Exported adapter to %s (%d params)", 
                       export_result.path, export_result.parameter_count)
            
            # Compute geometry metrics
            barrier, cka = self._compute_geometry_metrics(
                model_path, output_path, check_barrier
            )
            
            training_time = time.time() - start_time
            final_loss = final_progress.loss if final_progress else 0.0
            
            result = LoRATrainingResult(
                success=True,
                adapter_path=output_path,
                final_loss=final_loss,
                barrier_to_base=barrier,
                cka_from_base=cka,
                steps_trained=final_progress.global_step if final_progress else 0,
                samples_used=len(training_samples),
                training_time_seconds=training_time,
                lora_rank=lora_settings.rank,
                lora_alpha=lora_settings.alpha,
                target_modules=lora_settings.target_modules,
            )
            
            # Safety check
            if check_barrier and barrier > barrier_threshold:
                logger.warning(
                    "Barrier %.4f exceeds threshold %.4f - adapter may fight base model",
                    barrier, barrier_threshold
                )
            
            return result
            
        except Exception as e:
            logger.exception("Training failed: %s", e)
            return LoRATrainingResult(
                success=False,
                error=str(e),
                training_time_seconds=time.time() - start_time,
            )
    
    def _derive_or_use_settings(
        self,
        model,
        rank: Optional[int],
        alpha: Optional[float],
        target_modules: Optional[list[str]],
    ):
        """Derive LoRA settings from geometry or use provided values."""
        from modelcypher.core.domain.training.lora_mlx import (
            LoRASettings,
            derive_lora_settings_from_model,
        )
        
        if rank is None:
            # Derive from geometry
            logger.info("Deriving LoRA settings from model geometry")
            settings = derive_lora_settings_from_model(model, target_modules)
        else:
            settings = LoRASettings(
                rank=rank,
                alpha=alpha or (rank * 2.0),
                target_modules=target_modules or ["q_proj", "v_proj"],
            )
        
        return settings
    
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
    
    def _create_data_provider(self, samples: list[dict], tokenizer, batch_size: int):
        """Create a data provider for training."""
        import mlx.core as mx
        
        # Tokenize all samples
        tokenized = []
        for sample in samples:
            # Handle different data formats
            if "prompt" in sample and "completion" in sample:
                text = sample["prompt"] + sample["completion"]
            elif "text" in sample:
                text = sample["text"]
            elif "input" in sample and "output" in sample:
                text = sample["input"] + sample["output"]
            else:
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > 512:
                tokens = tokens[:512]
            tokenized.append(tokens)
        
        # Create batches
        class SimpleDataProvider:
            def __init__(self, data, bs):
                self.data = data
                self.batch_size = bs
                self.idx = 0
            
            def __iter__(self):
                self.idx = 0
                return self
            
            def __next__(self):
                if self.idx >= len(self.data):
                    raise StopIteration
                
                batch = self.data[self.idx:self.idx + self.batch_size]
                self.idx += self.batch_size
                
                # Pad to same length
                max_len = max(len(t) for t in batch)
                padded = []
                for t in batch:
                    padded.append(t + [0] * (max_len - len(t)))
                
                x = mx.array(padded)
                # For language modeling, y is x shifted by 1
                y = mx.concatenate([x[:, 1:], mx.zeros((x.shape[0], 1), dtype=mx.int32)], axis=1)
                
                return x, y
            
            def __len__(self):
                return (len(self.data) + self.batch_size - 1) // self.batch_size
        
        return SimpleDataProvider(tokenized, batch_size)
    
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
            # Use safety service to check barrier
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
