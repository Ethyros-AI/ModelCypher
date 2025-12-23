"""
Checkpoint Manager for Training Persistence.

Ported 1:1 from the reference Swift implementation.

Features:
- Atomic checkpoint writes (temp dir → validate → rename)
- SHA-256 checksum validation
- Optimizer state serialization (MLX arrays)
- Disk space validation
- Best checkpoint alias
- Retention-based pruning
"""
import os
import json
import shutil
import hashlib
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
import mlx.core as mx
from .types import CheckpointMetadata, TrainingConfig, Hyperparameters, LoRAConfig, ComputePrecision
from .exceptions import CheckpointError


# Minimum required disk space in bytes (500MB)
MIN_DISK_SPACE_BYTES = 500 * 1024 * 1024


"""Raised when checkpoint operations fail."""


class InsufficientDiskSpaceError(CheckpointError):
    """Raised when there's not enough disk space for checkpoint."""
    pass


class CheckpointManager:
    """
    Manages atomic writing and loading of training checkpoints.
    
    Ported from CheckpointManager.swift with:
    - Optimizer state serialization
    - Disk space validation
    - Best checkpoint alias
    """

    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
        self._best_loss: float = float('inf')
        self._best_step: int = -1

    async def save_checkpoint(
        self,
        model_weights: Dict[str, mx.array],
        optimizer_state: Optional[Dict[str, Any]],
        step: int,
        total_steps: int,
        loss_history: List[float],
        config: TrainingConfig,
        output_dir: str
    ) -> CheckpointMetadata:
        """
        Saves a checkpoint atomically with full state preservation.
        """
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Check disk space
        await self._validate_disk_space(checkpoints_dir)

        # Temp directory for atomic writes
        temp_dir = os.path.join(checkpoints_dir, ".tmp", f"step_{step}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            # 1. Save model weights
            weights_filename = f"checkpoint-{step}.safetensors"
            temp_weights_path = os.path.join(temp_dir, weights_filename)

            flat_weights = self._flatten_weights(model_weights)
            mx.save_safetensors(temp_weights_path, flat_weights)

            # 2. Save optimizer state
            optimizer_filename = None
            if optimizer_state is not None:
                optimizer_filename = f"optimizer-{step}.safetensors"
                temp_optimizer_path = os.path.join(temp_dir, optimizer_filename)
                optimizer_arrays = self._extract_optimizer_arrays(optimizer_state)
                if optimizer_arrays:
                    mx.save_safetensors(temp_optimizer_path, optimizer_arrays)

            # 3. Calculate Checksum
            checksum = await self._calculate_checksum(temp_weights_path)

            # 4. Create Metadata
            sanitized_loss = [
                float(x) for x in loss_history
                if x == x and x != float('inf') and x != float('-inf')  # Filter NaN/Inf
            ]

            metadata = CheckpointMetadata(
                version=2,
                step=step,
                total_steps=total_steps,
                train_config=config,
                loss_history=sanitized_loss,
                timestamp=datetime.now(),
                checksum=checksum,
                weights_file=weights_filename,
                optimizer_file=optimizer_filename
            )

            # 5. Save Metadata JSON
            metadata_filename = f"checkpoint-{step}.json"
            temp_metadata_path = os.path.join(temp_dir, metadata_filename)

            json_str = self._serialize_metadata(metadata)
            with open(temp_metadata_path, 'w') as f:
                f.write(json_str)

            # 6. Atomic Move
            final_weights_path = os.path.join(checkpoints_dir, weights_filename)
            final_metadata_path = os.path.join(checkpoints_dir, metadata_filename)

            os.rename(temp_weights_path, final_weights_path)
            os.rename(temp_metadata_path, final_metadata_path)

            # Move optimizer if saved
            if optimizer_filename:
                final_optimizer_path = os.path.join(checkpoints_dir, optimizer_filename)
                os.rename(
                    os.path.join(temp_dir, optimizer_filename),
                    final_optimizer_path
                )

            # 7. Update Best Checkpoint Alias
            current_loss = sanitized_loss[-1] if sanitized_loss else float('inf')
            await self._update_best_alias(checkpoints_dir, step, current_loss)

            # 8. Prune Old Checkpoints
            await self._prune_checkpoints(checkpoints_dir)

            return metadata

        except InsufficientDiskSpaceError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def load_latest_checkpoint(self, output_dir: str) -> Optional[CheckpointMetadata]:
        """Load metadata for the latest checkpoint."""
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return None

        # Try to load best checkpoint first
        best_path = os.path.join(checkpoints_dir, "checkpoint-best.json")
        if os.path.exists(best_path):
            try:
                with open(best_path, 'r') as f:
                    best_info = json.load(f)
                best_step = best_info.get("step")
                if best_step is not None:
                    return await self.load_checkpoint_metadata(checkpoints_dir, best_step)
            except Exception:
                pass

        # Fallback to finding latest step
        files = [f for f in os.listdir(checkpoints_dir)
                 if f.endswith(".json") and f.startswith("checkpoint-") and f != "checkpoint-best.json"]
        if not files:
            return None

        steps = []
        for f in files:
            try:
                step_str = f.replace("checkpoint-", "").replace(".json", "")
                steps.append(int(step_str))
            except ValueError:
                continue

        if not steps:
            return None

        latest_step = max(steps)
        return await self.load_checkpoint_metadata(checkpoints_dir, latest_step)

    async def load_checkpoint_metadata(self, checkpoints_dir: str, step: int) -> CheckpointMetadata:
        """Load checkpoint metadata for a specific step."""
        path = os.path.join(checkpoints_dir, f"checkpoint-{step}.json")
        with open(path, 'r') as f:
            data = json.load(f)

        # Deserialize TrainingConfig
        config = self._deserialize_config(data.get("train_config"))

        return CheckpointMetadata(
            version=data.get("version", 1),
            step=data["step"],
            total_steps=data.get("total_steps", 0),
            train_config=config,
            loss_history=data.get("loss_history", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            checksum=data.get("checksum", ""),
            weights_file=data.get("weights_file", f"checkpoint-{step}.safetensors"),
            optimizer_file=data.get("optimizer_file")
        )

    async def load_weights(self, checkpoints_dir: str, step: int) -> Dict[str, mx.array]:
        """Load model weights from checkpoint."""
        path = os.path.join(checkpoints_dir, f"checkpoint-{step}.safetensors")
        return mx.load(path)

    async def load_optimizer_state(self, checkpoints_dir: str, step: int) -> Optional[Dict[str, mx.array]]:
        """Load optimizer state from checkpoint if it exists."""
        path = os.path.join(checkpoints_dir, f"optimizer-{step}.safetensors")
        if os.path.exists(path):
            return mx.load(path)
        return None

    async def _validate_disk_space(self, checkpoints_dir: str):
        """Ensure sufficient disk space for checkpoint."""
        try:
            usage = shutil.disk_usage(checkpoints_dir)
            if usage.free < MIN_DISK_SPACE_BYTES:
                raise InsufficientDiskSpaceError(
                    f"Insufficient disk space: {usage.free / (1024**2):.1f}MB available, "
                    f"need at least {MIN_DISK_SPACE_BYTES / (1024**2):.1f}MB"
                )
        except OSError:
            # Can't check disk space, proceed anyway
            pass

    async def _update_best_alias(self, checkpoints_dir: str, step: int, loss: float):
        """Update best checkpoint alias if this is the best loss."""
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_step = step

            best_path = os.path.join(checkpoints_dir, "checkpoint-best.json")
            with open(best_path, 'w') as f:
                json.dump({
                    "step": step,
                    "loss": loss,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

    async def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _prune_checkpoints(self, checkpoints_dir: str):
        """Remove old checkpoints beyond retention limit."""
        files = [f for f in os.listdir(checkpoints_dir)
                 if f.endswith(".json") and f.startswith("checkpoint-") and f != "checkpoint-best.json"]

        steps = []
        for f in files:
            try:
                steps.append(int(f.replace("checkpoint-", "").replace(".json", "")))
            except ValueError:
                continue

        steps.sort(reverse=True)

        if len(steps) > self.max_checkpoints:
            to_delete = steps[self.max_checkpoints:]
            for step in to_delete:
                # Don't delete best checkpoint
                if step == self._best_step:
                    continue

                # Remove all files for this step
                for ext in [".json", ".safetensors"]:
                    try:
                        os.remove(os.path.join(checkpoints_dir, f"checkpoint-{step}{ext}"))
                    except OSError:
                        pass
                    try:
                        os.remove(os.path.join(checkpoints_dir, f"optimizer-{step}{ext}"))
                    except OSError:
                        pass

    def _flatten_weights(self, weights: Dict[str, Any], prefix: str = "") -> Dict[str, mx.array]:
        """Flatten nested weight dictionary for safetensors format."""
        flat: Dict[str, mx.array] = {}

        for k, v in weights.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_weights(v, key))
            elif isinstance(v, mx.array):
                flat[key] = v

        return flat

    def _extract_optimizer_arrays(self, state: Dict[str, Any]) -> Dict[str, mx.array]:
        """Extract MLX arrays from optimizer state for serialization."""
        arrays: Dict[str, mx.array] = {}

        def extract(d: Dict[str, Any], prefix: str = ""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, mx.array):
                    arrays[key] = v
                elif isinstance(v, dict):
                    extract(v, key)
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, mx.array):
                            arrays[f"{key}.{i}"] = item
                        elif isinstance(item, dict):
                            extract(item, f"{key}.{i}")

        extract(state)
        return arrays

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        """Serialize metadata to JSON string."""
        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, datetime):
                    return o.isoformat()
                if hasattr(o, '__dict__'):
                    return {k: v for k, v in o.__dict__.items() if not k.startswith('_')}
                if hasattr(o, 'value'):  # Enum
                    return o.value
                return super().default(o)

        return json.dumps(metadata, cls=EnhancedJSONEncoder, indent=2)

    def _deserialize_config(self, data: Optional[Dict[str, Any]]) -> Optional[TrainingConfig]:
        """Deserialize TrainingConfig from JSON data."""
        if data is None:
            return None

        try:
            hp_data = data.get("hyperparameters", {})
            hyperparameters = Hyperparameters(
                batch_size=hp_data.get("batch_size", 4),
                learning_rate=hp_data.get("learning_rate", 3e-5),
                epochs=hp_data.get("epochs", 3),
                sequence_length=hp_data.get("sequence_length", 1024),
                gradient_accumulation_steps=hp_data.get("gradient_accumulation_steps", 1),
                gradient_checkpointing=hp_data.get("gradient_checkpointing", True),
                mixed_precision=hp_data.get("mixed_precision", True),
                compute_precision=ComputePrecision(hp_data.get("compute_precision", "float16")),
                warmup_steps=hp_data.get("warmup_steps", 10),
                weight_decay=hp_data.get("weight_decay", 0.01),
                seed=hp_data.get("seed", 42),
                deterministic=hp_data.get("deterministic", True),
                optimizer_type=hp_data.get("optimizer_type", "adamw"),
            )

            lora_config = None
            lora_data = data.get("lora_config")
            if lora_data:
                lora_config = LoRAConfig(
                    rank=lora_data.get("rank", 8),
                    alpha=lora_data.get("alpha", 16.0),
                    dropout=lora_data.get("dropout", 0.05),
                    target_modules=lora_data.get("target_modules", ["q_proj", "v_proj"]),
                )

            return TrainingConfig(
                model_id=data.get("model_id", ""),
                dataset_path=data.get("dataset_path", ""),
                output_path=data.get("output_path", ""),
                hyperparameters=hyperparameters,
                lora_config=lora_config,
                resume_from_checkpoint_path=data.get("resume_from_checkpoint_path"),
            )
        except Exception:
            return None

