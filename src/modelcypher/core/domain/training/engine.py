"""
MLX Training Engine.

Ported 1:1 from TrainingCypher/MLXTrainingEngine.swift.

Orchestrates:
- Resource locking (via TrainingResourceGuard)
- Checkpoint management with resume support
- Training loop with gradient accumulation
- Pause/resume state machine
- NaN/Inf detection and recovery
- Learning rate scheduling
"""
import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List, Dict, Set

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .types import TrainingConfig, TrainingProgress, Hyperparameters
from .validation import TrainingHyperparameterValidator
from .resources import TrainingResourceGuard, ResourceIntensiveOperation
from .checkpoints import CheckpointManager
from modelcypher.infrastructure.services.memory import MLXMemoryService


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class TrainingCancelledException(TrainingError):
    """Raised when training is cancelled."""
    pass


@dataclass
class GradientAccumulationContext:
    """
    Manages gradient accumulation across micro-batches.
    
    Ported from Swift's GradientAccumulationContext.
    """
    total_steps: int
    current_step: int = 0
    accumulated_grads: Optional[Dict[str, mx.array]] = None
    accumulated_loss: float = 0.0

    def should_update(self) -> bool:
        """Returns True when optimizer should step."""
        return self.current_step >= self.total_steps

    def accumulate(self, grads: Dict[str, mx.array], loss: float):
        """Add gradients to accumulator."""
        self.current_step += 1
        self.accumulated_loss += loss

        if self.accumulated_grads is None:
            # First accumulation - copy
            self.accumulated_grads = {k: v for k, v in grads.items()}
        else:
            # Sum gradients
            for key in grads:
                if key in self.accumulated_grads:
                    self.accumulated_grads[key] = self.accumulated_grads[key] + grads[key]
                else:
                    self.accumulated_grads[key] = grads[key]

    def get_averaged(self) -> tuple[Dict[str, mx.array], float]:
        """Get averaged gradients and loss."""
        avg_grads = {}
        for key, grad in (self.accumulated_grads or {}).items():
            avg_grads[key] = grad / float(self.total_steps)
        avg_loss = self.accumulated_loss / float(self.total_steps)
        return avg_grads, avg_loss

    def reset(self):
        """Reset accumulator for next batch."""
        self.current_step = 0
        self.accumulated_grads = None
        self.accumulated_loss = 0.0


@dataclass
class ResumeState:
    """State for resuming from checkpoint."""
    global_step: int
    epoch_index: int
    step_offset: int
    loss_history: List[float]
    best_loss: float


class TrainingEngine:
    """
    Core MLX training engine.
    
    Ported 1:1 from TrainingCypher/MLXTrainingEngine.swift with:
    - Gradient accumulation
    - Checkpoint resume
    - Pause/resume state machine
    - NaN detection
    - Learning rate scheduling
    """

    def __init__(self):
        self.resource_guard = TrainingResourceGuard()
        self.checkpoint_manager = CheckpointManager()
        self.memory_service = MLXMemoryService()
        
        # Job state
        self._cancelled_jobs: Set[str] = set()
        self._paused_jobs: Set[str] = set()
        self._pause_events: Dict[str, asyncio.Event] = {}
        
        # Training state
        self.best_loss: float = float('inf')
        self.loss_history: List[float] = []

    async def train(
        self,
        job_id: str,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data_provider: Any,
        progress_callback: Callable[[TrainingProgress], None],
        loss_fn: Optional[Callable] = None,
    ):
        """
        Executes a complete training job.
        
        Args:
            job_id: Unique identifier for job control
            config: Training configuration
            model: MLX model to train
            optimizer: MLX optimizer
            data_provider: Iterable yielding (inputs, targets) batches
            progress_callback: Called with progress updates
            loss_fn: Optional custom loss function (default: cross_entropy)
        """
        # 1. Preflight Checks
        TrainingHyperparameterValidator.validate_for_engine(config.hyperparameters)

        mem_stats = self.memory_service.get_memory_stats()
        if mem_stats.pressure == "critical":
            raise TrainingError(f"Insufficient memory: {mem_stats.available_gb}GB available.")

        print(f"Starting training job {job_id} with MLX.")
        print(f"Memory: Active={mem_stats.mlx_active_gb}GB, Peak={mem_stats.mlx_peak_gb}GB")

        # Reset state
        self._cancelled_jobs.discard(job_id)
        self._paused_jobs.discard(job_id)
        self._pause_events[job_id] = asyncio.Event()
        self._pause_events[job_id].set()  # Not paused initially
        self.loss_history = []
        self.best_loss = float('inf')

        # Set deterministic seed if configured
        if config.hyperparameters.deterministic:
            mx.random.seed(config.hyperparameters.seed)
            print(f"Deterministic training enabled with seed {config.hyperparameters.seed}")

        # 2. Resource Locking
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
        optimizer: optim.Optimizer,
        data_provider: Any,
        progress_callback: Callable[[TrainingProgress], None],
        loss_fn: Optional[Callable],
    ):
        """Core training loop with all features."""
        hp = config.hyperparameters
        epochs = hp.epochs
        accumulation_steps = max(hp.gradient_accumulation_steps, 1)
        warmup_steps = hp.warmup_steps
        base_lr = hp.learning_rate

        # Try to get total steps
        try:
            steps_per_epoch = len(data_provider)
        except TypeError:
            steps_per_epoch = 100  # Fallback estimate

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
            print(f"Resuming from step {global_step} (epoch {resume_epoch_idx}, offset {resume_step_offset})")

            # Send baseline progress for UI
            progress_callback(TrainingProgress(
                job_id=job_id,
                epoch=resume_epoch_idx + 1,
                step=global_step,
                total_steps=total_steps,
                loss=self.loss_history[-1] if self.loss_history else 0.0,
                learning_rate=base_lr,
                metrics={"resumed": 1.0},
            ))

        # Define loss function
        if loss_fn is None:
            def loss_fn(model_inner, X, y):
                logits = model_inner(X)
                return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        acc_context = GradientAccumulationContext(total_steps=accumulation_steps)

        start_time = time.time()
        nan_recovery_count = 0
        max_nan_recoveries = 3

        for epoch in range(resume_epoch_idx, epochs):
            self._check_cancellation(job_id)
            await self._wait_if_paused(job_id)

            print(f"Epoch {epoch + 1}/{epochs}")

            for batch_idx, (inputs, targets) in enumerate(data_provider):
                # Skip to resume offset on first epoch
                if epoch == resume_epoch_idx and batch_idx < resume_step_offset:
                    continue

                self._check_cancellation(job_id)
                await self._wait_if_paused(job_id)

                step_start = time.time()

                # Convert to MLX arrays
                X = mx.array(inputs) if not isinstance(inputs, mx.array) else inputs
                y = mx.array(targets) if not isinstance(targets, mx.array) else targets

                # Learning rate warmup
                if global_step < warmup_steps:
                    warmup_lr = base_lr * (global_step + 1) / warmup_steps
                    optimizer.learning_rate = warmup_lr
                else:
                    optimizer.learning_rate = base_lr

                # Forward + Backward
                loss, grads = loss_and_grad_fn(model, X, y)
                current_loss = float(loss.item())

                # NaN detection
                if not math.isfinite(current_loss):
                    nan_recovery_count += 1
                    print(f"⚠️ NaN/Inf detected at step {global_step}. Recovery attempt {nan_recovery_count}/{max_nan_recoveries}")

                    if nan_recovery_count >= max_nan_recoveries:
                        raise TrainingError(f"Training diverged: NaN/Inf loss after {max_nan_recoveries} recovery attempts")

                    # Skip this batch and clear accumulator
                    acc_context.reset()
                    mx.eval(model.parameters())
                    continue

                # Gradient accumulation
                acc_context.accumulate(grads, current_loss)

                if acc_context.should_update():
                    avg_grads, avg_loss = acc_context.get_averaged()

                    # Update
                    optimizer.update(model, avg_grads)
                    mx.eval(model.parameters(), optimizer.state)

                    # Track loss
                    self.loss_history.append(avg_loss)
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss

                    acc_context.reset()
                    global_step += 1

                    elapsed = time.time() - step_start
                    tokens_per_sec = X.size / elapsed if elapsed > 0 else 0.0

                    # Progress Update
                    if global_step % 10 == 0:
                        progress = TrainingProgress(
                            job_id=job_id,
                            epoch=epoch + 1,
                            step=global_step,
                            total_steps=total_steps,
                            loss=avg_loss,
                            learning_rate=optimizer.learning_rate if isinstance(optimizer.learning_rate, float) else float(optimizer.learning_rate),
                            tokens_per_second=tokens_per_sec,
                            metrics={"batch_time": elapsed, "best_loss": self.best_loss},
                        )
                        progress_callback(progress)

                    # Periodic Checkpoint
                    if global_step % 100 == 0:
                        await self.checkpoint_manager.save_checkpoint(
                            model_weights=dict(model.parameters()),
                            optimizer_state=dict(optimizer.state) if hasattr(optimizer, 'state') else None,
                            step=global_step,
                            total_steps=total_steps,
                            loss_history=self.loss_history,
                            config=config,
                            output_dir=config.output_path,
                        )

                    # Memory Cleanup
                    if global_step % 50 == 0:
                        self.memory_service.clear_cache()

        # Final Checkpoint
        await self.checkpoint_manager.save_checkpoint(
            model_weights=dict(model.parameters()),
            optimizer_state=dict(optimizer.state) if hasattr(optimizer, 'state') else None,
            step=global_step,
            total_steps=total_steps,
            loss_history=self.loss_history,
            config=config,
            output_dir=config.output_path,
        )

        print(f"Training completed in {time.time() - start_time:.2f}s, final step {global_step}")

    async def _check_resume(self, config: TrainingConfig) -> Optional[ResumeState]:
        """Check for checkpoint to resume from."""
        if config.resume_from_checkpoint_path:
            metadata = await self.checkpoint_manager.load_latest_checkpoint(
                config.resume_from_checkpoint_path
            )
            if metadata:
                # Calculate epoch and step offset
                try:
                    steps_per_epoch = config.hyperparameters.epochs  # Simplified
                    epoch_idx = metadata.step // max(steps_per_epoch, 1)
                    step_offset = metadata.step % max(steps_per_epoch, 1)
                except Exception:
                    epoch_idx = 0
                    step_offset = 0

                return ResumeState(
                    global_step=metadata.step,
                    epoch_index=epoch_idx,
                    step_offset=step_offset,
                    loss_history=metadata.loss_history,
                    best_loss=min(metadata.loss_history) if metadata.loss_history else float('inf'),
                )
        return None

    def _check_cancellation(self, job_id: str):
        """Raise if job is cancelled."""
        if job_id in self._cancelled_jobs:
            raise TrainingCancelledException(f"Job {job_id} was cancelled")

    async def _wait_if_paused(self, job_id: str):
        """Block while job is paused."""
        if job_id in self._paused_jobs:
            event = self._pause_events.get(job_id)
            if event:
                await event.wait()

    def pause(self, job_id: str):
        """Pause training at next step boundary."""
        self._paused_jobs.add(job_id)
        event = self._pause_events.get(job_id)
        if event:
            event.clear()
        print(f"Pause requested for job {job_id}")

    def resume(self, job_id: str):
        """Resume paused training."""
        self._paused_jobs.discard(job_id)
        event = self._pause_events.get(job_id)
        if event:
            event.set()
        print(f"Resume requested for job {job_id}")

    def cancel(self, job_id: str):
        """Cancel training at next step boundary."""
        self._cancelled_jobs.add(job_id)
        # Also resume if paused so the cancellation check runs
        self.resume(job_id)
        print(f"Cancel requested for job {job_id}")

