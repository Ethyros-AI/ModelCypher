import asyncio
import time
import uuid
from typing import Callable, Optional, Dict, Any, List
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from .types import TrainingConfig, TrainingProgress, Hyperparameters
from .validation import TrainingHyperparameterValidator
from .resources import TrainingResourceGuard, ResourceIntensiveOperation
from .checkpoints import CheckpointManager
from modelcypher.infrastructure.services.memory import MLXMemoryService

class TrainingError(Exception):
    pass

class TrainingEngine:
    """
    Core MLX training engine.
    Orchestrates:
    - Resource locking (via TrainingResourceGuard)
    - Checkpoint management
    - Training loop (Forward/Backward/Update)
    - Callback dispatch
    """
    
    def __init__(self):
        self.resource_guard = TrainingResourceGuard()
        self.checkpoint_manager = CheckpointManager()
        self.memory_service = MLXMemoryService()
        self.should_stop = False
        
    async def train(
        self,
        job_id: str,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        # TODO: Abstract Dataset provider. For now simpler interface:
        # data_provider should be an iterable yielding batches (inputs, targets)
        data_provider: Any, 
        progress_callback: Callable[[TrainingProgress], None]
    ):
        """
        Executes a complete training job.
        """
        # 1. Preflight Checks
        TrainingHyperparameterValidator.validate_for_engine(config.hyperparameters)
        
        mem_stats = self.memory_service.get_memory_stats()
        if mem_stats.pressure == "critical":
            raise TrainingError(f"Insufficient memory: {mem_stats.available_gb}GB available.")
            
        print(f"Starting training job {job_id} with MLX.")
        print(f"Memory: Active={mem_stats.mlx_active_gb}GB, Peak={mem_stats.mlx_peak_gb}GB")
        
        self.should_stop = False
        
        # 2. Resource Locking
        async with self.resource_guard.training_session(job_id):
            
            # 3. Training Loop
            loss_history: List[float] = []
            global_step = 0
            total_steps = config.hyperparameters.epochs * len(data_provider) # simplified
            
            # Define loss function
            def loss_fn(model, X, y):
                logits = model(X)
                return nn.losses.cross_entropy(logits, y, reduction="mean")
            
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            
            start_time = time.time()
            
            for epoch in range(config.hyperparameters.epochs):
                if self.should_stop: break
                
                print(f"Epoch {epoch+1}/{config.hyperparameters.epochs}")
                
                for batch_idx, (inputs, targets) in enumerate(data_provider):
                    if self.should_stop: break
                    
                    step_start = time.time()
                    
                    # Convert to MLX arrays if needed
                    X = mx.array(inputs)
                    y = mx.array(targets)
                    
                    # Forward + Backward
                    loss, grads = loss_and_grad_fn(model, X, y)
                    
                    # Update
                    optimizer.update(model, grads)
                    
                    # Force eval to realize computation
                    mx.eval(model.parameters(), optimizer.state)
                    
                    # Metrics
                    current_loss = loss.item()
                    loss_history.append(current_loss)
                    global_step += 1
                    
                    elapsed = time.time() - step_start
                    
                    # Progress Update
                    if global_step % 10 == 0:
                        progress = TrainingProgress(
                            job_id=job_id,
                            epoch=epoch + 1,
                            step=global_step,
                            total_steps=total_steps,
                            loss=current_loss,
                            learning_rate=optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else optimizer.learning_rate,
                            tokens_per_second=(X.size / elapsed),
                            metrics={"batch_time": elapsed}
                        )
                        progress_callback(progress)
                        
                    # Periodic Checkpoint
                    if global_step % 100 == 0:
                         await self.checkpoint_manager.save_checkpoint(
                             model_weights=dict(model.parameters()), # simplified, flattened
                             optimizer_state=None,
                             step=global_step,
                             total_steps=total_steps,
                             loss_history=loss_history,
                             config=config,
                             output_dir=config.output_path
                         )
                         
                    # Memory Cleanup
                    if global_step % 50 == 0:
                        self.memory_service.clear_cache()

            # Final Checkpoint
            await self.checkpoint_manager.save_checkpoint(
                 model_weights=dict(model.parameters()),
                 optimizer_state=None,
                 step=global_step,
                 total_steps=total_steps,
                 loss_history=loss_history,
                 config=config,
                 output_dir=config.output_path
             )
             
            print(f"Training completed in {time.time() - start_time:.2f}s")
            
    def cancel(self):
        self.should_stop = True

    # NOTE: Helper to setup LoRA model would go here (using mlx-lm)
