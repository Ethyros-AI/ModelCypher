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

"""
CUDA Training Engine (PyTorch Backend).

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see engine_mlx.py
- JAX/TPU: see engine_jax.py

Use _platform.get_training_engine() for automatic platform selection.

Implementation based on PyTorch 2.9 best practices (2025):
- torch.autocast for mixed precision forward passes
- torch.amp.GradScaler for gradient scaling with fp16
- Gradient accumulation with proper loss scaling
- NaN/Inf detection via GradScaler

References:
- https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- https://docs.pytorch.org/docs/stable/amp.html
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .resources import TrainingResourceGuard
from .types import TrainingConfig, TrainingProgress
from .validation import TrainingHyperparameterValidator

logger = logging.getLogger(__name__)


class TrainingErrorCUDA(Exception):
    """Base exception for CUDA training errors."""

    pass


class TrainingCancelledExceptionCUDA(TrainingErrorCUDA):
    """Raised when training is cancelled."""

    pass


@dataclass
class GradientAccumulationContextCUDA:
    """
    Manages gradient accumulation across micro-batches (CUDA version).

    With AMP/GradScaler, gradients stay scaled during accumulation.
    We only call scaler.step() and scaler.update() after all micro-batches.

    Per PyTorch docs: "grads should remain scaled, and the scale factor
    should remain constant, while grads for a given effective batch are
    accumulated."
    """

    total_steps: int
    current_step: int = 0
    accumulated_loss: float = 0.0

    def should_update(self) -> bool:
        """Returns True when optimizer should step."""
        return self.current_step >= self.total_steps

    def accumulate_loss(self, loss: float) -> None:
        """Track loss for averaging (gradients accumulate in .grad attrs)."""
        self.current_step += 1
        self.accumulated_loss += loss

    def get_averaged_loss(self) -> float:
        """Get averaged loss across accumulated steps."""
        return self.accumulated_loss / max(self.total_steps, 1)

    def reset(self) -> None:
        """Reset accumulator for next batch."""
        self.current_step = 0
        self.accumulated_loss = 0.0


@dataclass
class ResumeStateCUDA:
    """State for resuming from checkpoint."""

    global_step: int
    epoch_index: int
    step_offset: int
    loss_history: list[float]
    best_loss: float


class TrainingEngineCUDA:
    """
    CUDA Training Engine (PyTorch backend).

    Features (matching MLX parity):
    - Mixed precision training with torch.autocast + GradScaler
    - Gradient accumulation with proper AMP handling
    - Checkpoint resume support
    - Pause/resume state machine
    - NaN/Inf detection via GradScaler (skips bad steps)
    - Learning rate warmup
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resource_guard = TrainingResourceGuard()

        # Import checkpoint manager lazily to avoid circular imports
        from .checkpoints_cuda import CheckpointManagerCUDA

        self.checkpoint_manager = CheckpointManagerCUDA()

        # Job state
        self._cancelled_jobs: set[str] = set()
        self._paused_jobs: set[str] = set()
        self._pause_events: dict[str, asyncio.Event] = {}

        # Training state
        self.best_loss: float = float("inf")
        self.loss_history: list[float] = []

        # Mixed precision
        self.scaler = GradScaler()

    async def train(
        self,
        job_id: str,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_provider: Any,
        progress_callback: Callable[[TrainingProgress], None],
        loss_fn: Callable | None = None,
    ) -> None:
        """
        Execute a complete training job on CUDA.

        Args:
            job_id: Unique identifier for job control
            config: Training configuration
            model: PyTorch model to train (will be moved to device)
            optimizer: PyTorch optimizer
            data_provider: Iterable yielding (inputs, targets) batches
            progress_callback: Called with progress updates
            loss_fn: Optional custom loss function (default: cross_entropy)
        """
        # Preflight checks
        TrainingHyperparameterValidator.validate_for_engine(config.hyperparameters)

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        # Move model to device
        model = model.to(self.device)
        logger.info("Starting training job %s on %s", job_id, self.device)

        # Memory info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            logger.info("GPU Memory: Allocated=%.2fGB, Reserved=%.2fGB", allocated, reserved)

        # Reset state
        self._cancelled_jobs.discard(job_id)
        self._paused_jobs.discard(job_id)
        self._pause_events[job_id] = asyncio.Event()
        self._pause_events[job_id].set()  # Not paused initially
        self.loss_history = []
        self.best_loss = float("inf")
        self.scaler = GradScaler()  # Fresh scaler per job

        # Deterministic seed if configured
        if config.hyperparameters.deterministic:
            torch.manual_seed(config.hyperparameters.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.hyperparameters.seed)
            logger.info("Deterministic training enabled with seed %d", config.hyperparameters.seed)

        # Resource locking
        async with self.resource_guard.training_session(job_id):
            try:
                await self._execute_training(
                    job_id=job_id,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    data_provider=data_provider,
                    progress_callback=progress_callback,
                    loss_fn=loss_fn,
                )
            finally:
                self._pause_events.pop(job_id, None)

    async def _execute_training(
        self,
        job_id: str,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_provider: Any,
        progress_callback: Callable[[TrainingProgress], None],
        loss_fn: Callable | None,
    ) -> None:
        """Core training loop with AMP and gradient accumulation."""
        hp = config.hyperparameters
        epochs = hp.epochs
        accumulation_steps = max(hp.gradient_accumulation_steps, 1)
        warmup_steps = hp.warmup_steps
        base_lr = hp.learning_rate

        # Get total steps
        try:
            steps_per_epoch = len(data_provider)
        except TypeError:
            steps_per_epoch = 100  # Fallback

        total_steps = epochs * steps_per_epoch

        # Check for resume
        resume_state = await self._check_resume(config)
        resume_epoch_idx = 0
        resume_step_offset = 0
        global_step = 0

        if resume_state is not None:
            resume_epoch_idx = resume_state.epoch_index
            resume_step_offset = resume_state.step_offset
            global_step = resume_state.global_step
            self.loss_history = resume_state.loss_history.copy()
            self.best_loss = resume_state.best_loss
            logger.info("Resuming from step %d (epoch %d)", global_step, resume_epoch_idx)

        # Default loss function
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        acc_context = GradientAccumulationContextCUDA(total_steps=accumulation_steps)
        start_time = time.time()

        for epoch in range(resume_epoch_idx, epochs):
            self._check_cancellation(job_id)
            await self._wait_if_paused(job_id)

            logger.info("Epoch %d/%d", epoch + 1, epochs)
            model.train()

            for batch_idx, (inputs, targets) in enumerate(data_provider):
                # Skip to resume offset
                if epoch == resume_epoch_idx and batch_idx < resume_step_offset:
                    continue

                self._check_cancellation(job_id)
                await self._wait_if_paused(job_id)

                step_start = time.time()

                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Learning rate warmup
                if global_step < warmup_steps:
                    warmup_lr = base_lr * (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = base_lr

                # Forward pass with autocast (PyTorch 2.x pattern)
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()

                # Track loss (unscaled)
                acc_context.accumulate_loss(loss.item() * accumulation_steps)

                if acc_context.should_update():
                    # Unscale before clipping (if needed)
                    self.scaler.unscale_(optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Step (skips if grads contain inf/nan)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                    avg_loss = acc_context.get_averaged_loss()

                    # Track loss
                    self.loss_history.append(avg_loss)
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss

                    acc_context.reset()
                    global_step += 1

                    elapsed = time.time() - step_start
                    tokens_per_sec = inputs.numel() / elapsed if elapsed > 0 else 0.0

                    # Progress update
                    if global_step % 10 == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        progress = TrainingProgress(
                            job_id=job_id,
                            epoch=epoch + 1,
                            step=global_step,
                            total_steps=total_steps,
                            loss=avg_loss,
                            learning_rate=current_lr,
                            tokens_per_second=tokens_per_sec,
                            metrics={"batch_time": elapsed, "best_loss": self.best_loss},
                        )
                        progress_callback(progress)

                    # Periodic checkpoint
                    if global_step % 100 == 0:
                        await self.checkpoint_manager.save_checkpoint(
                            model_weights={k: v.cpu() for k, v in model.state_dict().items()},
                            optimizer_state=optimizer.state_dict(),
                            step=global_step,
                            total_steps=total_steps,
                            loss_history=self.loss_history,
                            config=config,
                            output_dir=config.output_path,
                        )

                    # Memory cleanup
                    if global_step % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Final checkpoint
        await self.checkpoint_manager.save_checkpoint(
            model_weights={k: v.cpu() for k, v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),
            step=global_step,
            total_steps=total_steps,
            loss_history=self.loss_history,
            config=config,
            output_dir=config.output_path,
        )

        logger.info(
            "Training completed in %.2fs, final step %d", time.time() - start_time, global_step
        )

    async def _check_resume(self, config: TrainingConfig) -> ResumeStateCUDA | None:
        """Check for checkpoint to resume from."""
        if config.resume_from_checkpoint_path:
            metadata = await self.checkpoint_manager.load_latest_checkpoint(
                config.resume_from_checkpoint_path
            )
            if metadata:
                try:
                    steps_per_epoch = config.hyperparameters.epochs
                    epoch_idx = metadata.step // max(steps_per_epoch, 1)
                    step_offset = metadata.step % max(steps_per_epoch, 1)
                except Exception:
                    epoch_idx = 0
                    step_offset = 0

                return ResumeStateCUDA(
                    global_step=metadata.step,
                    epoch_index=epoch_idx,
                    step_offset=step_offset,
                    loss_history=metadata.loss_history,
                    best_loss=min(metadata.loss_history) if metadata.loss_history else float("inf"),
                )
        return None

    def _check_cancellation(self, job_id: str) -> None:
        """Raise if job is cancelled."""
        if job_id in self._cancelled_jobs:
            raise TrainingCancelledExceptionCUDA(f"Job {job_id} was cancelled")

    async def _wait_if_paused(self, job_id: str) -> None:
        """Block while job is paused."""
        if job_id in self._paused_jobs:
            event = self._pause_events.get(job_id)
            if event:
                await event.wait()

    def pause(self, job_id: str) -> None:
        """Pause training at next step boundary."""
        self._paused_jobs.add(job_id)
        event = self._pause_events.get(job_id)
        if event:
            event.clear()
        logger.info("Pause requested for job %s", job_id)

    def resume(self, job_id: str) -> None:
        """Resume paused training."""
        self._paused_jobs.discard(job_id)
        event = self._pause_events.get(job_id)
        if event:
            event.set()
        logger.info("Resume requested for job %s", job_id)

    def cancel(self, job_id: str) -> None:
        """Cancel training at next step boundary."""
        self._cancelled_jobs.add(job_id)
        self.resume(job_id)
        logger.info("Cancel requested for job %s", job_id)


__all__ = [
    "TrainingEngineCUDA",
    "TrainingErrorCUDA",
    "TrainingCancelledExceptionCUDA",
    "GradientAccumulationContextCUDA",
    "ResumeStateCUDA",
]
