import os
import json
import shutil
import hashlib
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import mlx.core as mx
from .types import CheckpointMetadata, TrainingConfig

class CheckpointError(Exception):
    pass

class CheckpointManager:
    """
    Manages atomic writing and loading of training checkpoints.
    Ports `CheckpointManager.swift` logic including atomic moves and retention.
    """
    
    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints

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
        Saves a checkpoint atomically.
        """
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Temp directory for atomic writes
        temp_dir = os.path.join(checkpoints_dir, ".tmp", f"step_{step}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            # 1. Save weights
            weights_filename = f"checkpoint-{step}.safetensors"
            temp_weights_path = os.path.join(temp_dir, weights_filename)
            
            # Flatten weights if nested
            flat_weights = {}
            def flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        flatten(v, key)
                    elif isinstance(v, mx.array):
                        flat_weights[key] = v
            
            flatten(model_weights)
            mx.save_safetensors(temp_weights_path, flat_weights)
            
            # 2. Save optimizer state (if any)
            # MLX persistence for optimizers is tricky; usually we save state dicts as np/mlx arrays too
            # For now, we assume optimizer_state is serializable or managed externally.
            # If optimizer state contains arrays, we can save them in a separate safetensors or pickle.
            optimizer_filename = None
            if optimizer_state:
                # TODO: Implement robust MLX optimizer state serialization
                # For prototype, we skip saving full optimizer state to disk or assume it's small metadata.
                # If arrays, save as safetensors:
                # mx.save_safetensors(opt_path, optimizer_arrays)
                pass

            # 3. Calculate Checksum (simple file hash)
            checksum = await self._calculate_checksum(temp_weights_path)
            
            # 4. Create Metadata
            metadata = CheckpointMetadata(
                version=2,
                step=step,
                total_steps=total_steps,
                train_config=config,
                loss_history=[float(x) for x in loss_history if not (x != x or x == float('inf'))], # sanitize
                timestamp=datetime.now(),
                checksum=checksum,
                weights_file=weights_filename,
                optimizer_file=optimizer_filename
            )
            
            # 5. Save Metadata JSON
            metadata_filename = f"checkpoint-{step}.json"
            temp_metadata_path = os.path.join(temp_dir, metadata_filename)
            
            # Helper to sanitize dataclasses/enums for JSON
            json_str = self._serialize_metadata(metadata)
            with open(temp_metadata_path, 'w') as f:
                f.write(json_str)

            # 6. Atomic Move
            final_weights_path = os.path.join(checkpoints_dir, weights_filename)
            final_metadata_path = os.path.join(checkpoints_dir, metadata_filename)
            
            # Move files from temp to final
            os.rename(temp_weights_path, final_weights_path)
            os.rename(temp_metadata_path, final_metadata_path)
            
            # 7. Prune Old Checkpoints
            await self._prune_checkpoints(checkpoints_dir)
            
            return metadata

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def load_latest_checkpoint(self, output_dir: str) -> Optional[CheckpointMetadata]:
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return None
            
        # List all JSON files
        files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".json") and f.startswith("checkpoint-")]
        if not files:
            return None
            
        # Parse step numbers
        # checkpoint-100.json
        steps = []
        for f in files:
            try:
                parts = f.replace("checkpoint-", "").replace(".json", "")
                steps.append(int(parts))
            except ValueError:
                continue
        
        if not steps:
            return None
            
        latest_step = max(steps)
        return await self.load_checkpoint_metadata(checkpoints_dir, latest_step)

    async def load_checkpoint_metadata(self, checkpoints_dir: str, step: int) -> CheckpointMetadata:
        path = os.path.join(checkpoints_dir, f"checkpoint-{step}.json")
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Deserialize back to object (simplified)
        # In a real app, use dacite or similar library for robust rehydration
        # Reconstructing manually for now to avoid external deps if possible
        # Or use simple dicts if types are too complex.
        
        # NOTE: For now, returning raw dict might be easier, but sticking to type
        # Ideally, we should use a library or helper.
        # Let's assume we return a simplified object or type ignore for complex nested config.
        # TODO: Implement full deserialization
        config_data = data.get("train_config", {})
        # Making a dummy config for now to satisfy type checker in runtime
        # You'd fully hydrate this in production
        
        return CheckpointMetadata(
            version=data["version"],
            step=data["step"],
            total_steps=data["total_steps"],
            train_config=None, # Placeholder
            loss_history=data["loss_history"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            checksum=data["checksum"],
            weights_file=data["weights_file"]
        )

    async def _calculate_checksum(self, filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _prune_checkpoints(self, checkpoints_dir: str):
        # List all checkpoints, sort by step
        files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".json") and f.startswith("checkpoint-")]
        steps = []
        for f in files:
            try:
                steps.append(int(f.replace("checkpoint-", "").replace(".json", "")))
            except: pass
            
        steps.sort(reverse=True)
        
        if len(steps) > self.max_checkpoints:
            to_delete = steps[self.max_checkpoints:]
            for step in to_delete:
                # Remove json and safetensors
                try:
                    os.remove(os.path.join(checkpoints_dir, f"checkpoint-{step}.json"))
                    os.remove(os.path.join(checkpoints_dir, f"checkpoint-{step}.safetensors"))
                    # Also optimizer if it exists
                except OSError:
                    pass

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        # Custom encoder for datetimes and dataclasses
        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, datetime):
                    return o.isoformat()
                if hasattr(o, '__dict__'):
                    return o.__dict__
                return super().default(o)
        
        return json.dumps(metadata, cls=EnhancedJSONEncoder, indent=2)
