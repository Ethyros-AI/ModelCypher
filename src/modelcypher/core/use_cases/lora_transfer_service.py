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

"""LoRA Transfer Service.

Provides high-level orchestration for transferring LoRA adapters between
different base models using geometric subspace alignment.

This service coordinates:
1. Loading adapter weights from safetensors format
2. Computing SVD of source and target base model layers
3. Calling the UniversalLoRAProjector for geometric transfer
4. Saving the projected adapter in HuggingFace PEFT format

Usage:
    from modelcypher.core.use_cases.lora_transfer_service import LoRATransferService

    service = LoRATransferService()
    result = await service.transfer_adapter(
        source_adapter_path=Path("./my-llama-adapter"),
        source_base_model_id="meta-llama/Llama-3-8B",
        target_base_model_id="mistralai/Mistral-7B-v0.3",
        output_path=Path("./my-mistral-adapter"),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.universal_lora_projector import (
    LoRATransferResult,
    SVDComponents,
    UniversalLoRAProjector,
    compute_lora_delta,
    decompose_to_lora,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TransferConfig:
    """Configuration for LoRA transfer operation."""

    # Target rank for output adapter (None = preserve source rank)
    target_rank: int | None = None
    
    # Maximum layers to process (None = all layers)
    max_layers: int | None = None
    
    # Cache SVD computations to disk
    cache_svd: bool = True
    
    # Preserve adapter metadata (alpha, dropout, etc.)
    preserve_metadata: bool = True
    
    # Layer name mapping (source pattern -> target pattern)
    layer_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class TransferProgress:
    """Progress update during transfer operation."""

    phase: str  # "loading", "svd_source", "svd_target", "transferring", "saving"
    current_layer: int
    total_layers: int
    message: str


# =============================================================================
# LoRA Transfer Service
# =============================================================================


class LoRATransferService:
    """High-level service for cross-model LoRA adapter transfer.

    This service orchestrates the complete workflow for transferring a LoRA
    adapter from one base model to another using geometric alignment.

    Example:
        service = LoRATransferService()
        result = await service.transfer_adapter(
            source_adapter_path=Path("./my-llama-adapter"),
            source_base_model_id="meta-llama/Llama-3-8B",
            target_base_model_id="mistralai/Mistral-7B-v0.3",
            output_path=Path("./my-mistral-adapter"),
        )
        
        if result.success:
            print(f"Transferred {result.layers_transferred} layers")
        else:
            print(f"Transfer failed: {result.warnings}")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        cache_dir: Path | None = None,
        model_loader: "ModelLoaderPort | None" = None,
    ) -> None:
        """Initialize the transfer service.

        Args:
            backend: Compute backend (uses default if None)
            cache_dir: Directory for SVD caches (uses tmp if None)
            model_loader: Port for loading base model weights
        """
        self._backend = backend or get_default_backend()
        self._projector = UniversalLoRAProjector(backend=self._backend)
        self._cache_dir = cache_dir or Path("/tmp/modelcypher/svd_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_loader = model_loader

    @property
    def backend(self) -> "Backend":
        """Get compute backend."""
        return self._backend

    async def transfer_adapter(
        self,
        source_adapter_path: Path,
        source_base_model_id: str,
        target_base_model_id: str,
        output_path: Path,
        config: TransferConfig | None = None,
        progress_callback: Any = None,
    ) -> LoRATransferResult:
        """Transfer a LoRA adapter from source to target base model.

        This is the main entry point for adapter transfer. It handles:
        1. Loading the source adapter weights
        2. Computing or loading cached SVD for source base model
        3. Computing or loading cached SVD for target base model
        4. Projecting adapter weights using geometric alignment
        5. Saving the projected adapter in PEFT format

        Args:
            source_adapter_path: Path to source adapter directory
            source_base_model_id: HuggingFace model ID of source base
            target_base_model_id: HuggingFace model ID of target base
            output_path: Path to save projected adapter
            config: Transfer configuration (uses defaults if None)
            progress_callback: Optional callback for progress updates

        Returns:
            LoRATransferResult with transfer metrics and warnings
        """
        config = config or TransferConfig()
        
        def _report(phase: str, current: int, total: int, msg: str) -> None:
            if progress_callback:
                progress_callback(TransferProgress(phase, current, total, msg))
            logger.info("[%s] %d/%d: %s", phase, current, total, msg)

        # Phase 1: Load source adapter
        _report("loading", 0, 4, "Loading source adapter weights...")
        adapter_weights_raw, adapter_config = self._load_adapter(source_adapter_path)
        
        # Convert A/B matrices to deltas
        adapter_weights = self._convert_to_deltas(adapter_weights_raw)
        logger.info("Loaded %d layer deltas from adapter", len(adapter_weights))

        # Phase 2: Compute source SVD
        _report("svd_source", 1, 4, f"Computing SVD for {source_base_model_id}...")
        source_svd = await self._get_model_svd(
            source_base_model_id,
            list(adapter_weights.keys()),
            config,
        )

        # Phase 3: Compute target SVD
        _report("svd_target", 2, 4, f"Computing SVD for {target_base_model_id}...")
        target_layer_keys = self._map_layer_keys(
            list(adapter_weights.keys()),
            config.layer_mapping,
        )
        target_svd = await self._get_model_svd(
            target_base_model_id,
            target_layer_keys,
            config,
        )

        # Phase 4: Transfer
        _report("transferring", 3, 4, "Projecting adapter weights...")
        transferred_weights, result = self._projector.transfer(
            source_adapter_weights=adapter_weights,
            source_base_svd=source_svd,
            target_base_svd=target_svd,
            source_base_model=source_base_model_id,
            target_base_model=target_base_model_id,
        )

        # Phase 5: Save
        _report("saving", 4, 4, f"Saving to {output_path}...")
        target_rank = config.target_rank or adapter_config.get("r", 16)
        self._save_adapter(
            transferred_weights,
            output_path,
            target_base_model_id,
            target_rank,
            adapter_config if config.preserve_metadata else {},
        )

        logger.info(
            "Transfer complete: %d layers, mean_error=%.4f",
            result.layers_transferred,
            result.mean_projection_error,
        )

        return result

    def _load_adapter(self, adapter_path: Path) -> tuple[dict[str, "Array"], dict]:
        """Load adapter weights and config from a PEFT adapter directory."""
        from safetensors import safe_open
        
        # Load weights
        weights_path = adapter_path / "adapter_model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Adapter weights not found: {weights_path}")

        weights: dict[str, "Array"] = {}
        with safe_open(str(weights_path), framework="numpy") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
                weights[key] = self._backend.array(arr)

        # Load config
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        return weights, config

    def _convert_to_deltas(
        self,
        adapter_weights: dict[str, "Array"],
    ) -> dict[str, "Array"]:
        """Convert LoRA A/B weight pairs to delta matrices.

        LoRA stores weights as:
            base_layer.lora_A.weight: [rank, in_features]
            base_layer.lora_B.weight: [out_features, rank]

        We combine them: delta = B @ A
        """
        deltas: dict[str, "Array"] = {}

        # Group by base layer name
        a_weights: dict[str, "Array"] = {}
        b_weights: dict[str, "Array"] = {}

        for key, weight in adapter_weights.items():
            if ".lora_A." in key:
                base_key = key.replace(".lora_A.weight", "").replace(".lora_A.default", "")
                a_weights[base_key] = weight
            elif ".lora_B." in key:
                base_key = key.replace(".lora_B.weight", "").replace(".lora_B.default", "")
                b_weights[base_key] = weight

        # Combine A and B into deltas
        for base_key in a_weights:
            if base_key in b_weights:
                delta = compute_lora_delta(
                    a_weights[base_key],
                    b_weights[base_key],
                    self._backend,
                )
                deltas[base_key] = delta
            else:
                logger.warning("Missing B weight for %s", base_key)

        return deltas

    async def _get_model_svd(
        self,
        model_id: str,
        layer_keys: list[str],
        config: TransferConfig,
    ) -> dict[str, SVDComponents]:
        """Get SVD components for model layers.

        Attempts to load from cache if available, otherwise computes and caches.
        """
        # For now, compute directly using model loading
        # In production, this would use cached SVDs or lazy loading
        return await self._compute_model_svd(model_id, layer_keys, config)

    async def _compute_model_svd(
        self,
        model_id: str,
        layer_keys: list[str],
        config: TransferConfig,
    ) -> dict[str, SVDComponents]:
        """Compute SVD for specified layers of a base model."""
        svd_components: dict[str, SVDComponents] = {}

        if self._model_loader is None:
            raise RuntimeError(
                "LoRATransferService requires a ModelLoaderPort to load base model weights."
            )

        try:
            # Load model weights via hexagonal port.
            weights = self._model_loader.load_weights(model_id)
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_id, e)
            raise RuntimeError(f"Cannot load base model {model_id}: {e}") from e

        for key in layer_keys:
            # Map adapter key to base model weight key
            weight_key = self._adapter_key_to_weight_key(key)
            
            if weight_key not in weights:
                logger.warning("Weight key %s not found in model", weight_key)
                continue

            weight = weights[weight_key]
            weight_arr = self._backend.array(weight)
            
            try:
                svd = self._projector.compute_layer_svd(weight_arr)
                svd_components[key] = svd
            except Exception as e:
                logger.warning("SVD failed for %s: %s", key, e)

        return svd_components

    def _adapter_key_to_weight_key(self, adapter_key: str) -> str:
        """Convert adapter layer key to base model weight key.

        PEFT adapter keys look like:
            base_model.model.model.layers.0.self_attn.q_proj

        Base model weight keys look like:
            model.layers.0.self_attn.q_proj.weight
        """
        # Remove common prefixes
        key = adapter_key
        for prefix in ["base_model.model.", "base_model."]:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break

        # Add .weight suffix if needed
        if not key.endswith(".weight"):
            key = key + ".weight"

        return key

    def _map_layer_keys(
        self,
        source_keys: list[str],
        mapping: dict[str, str],
    ) -> list[str]:
        """Map source layer keys to target layer keys."""
        if not mapping:
            return source_keys

        mapped = []
        for key in source_keys:
            for pattern, replacement in mapping.items():
                if pattern in key:
                    key = key.replace(pattern, replacement)
                    break
            mapped.append(key)
        return mapped

    def _save_adapter(
        self,
        transferred_weights: dict[str, "Array"],
        output_path: Path,
        target_base_model: str,
        rank: int,
        source_config: dict,
    ) -> None:
        """Save transferred weights as a PEFT adapter.

        Decomposes delta matrices back into A/B LoRA format.
        """
        from safetensors.numpy import save_file
        import numpy as np
        
        output_path.mkdir(parents=True, exist_ok=True)

        # Decompose deltas back to A/B format
        output_weights: dict[str, Any] = {}
        for key, delta in transferred_weights.items():
            lora_A, lora_B = decompose_to_lora(delta, rank, self._backend)
            
            # Convert to numpy for safetensors
            a_np = np.array(self._backend.to_numpy(lora_A))
            b_np = np.array(self._backend.to_numpy(lora_B))
            
            output_weights[f"{key}.lora_A.weight"] = a_np
            output_weights[f"{key}.lora_B.weight"] = b_np

        # Save weights
        weights_path = output_path / "adapter_model.safetensors"
        save_file(output_weights, str(weights_path))

        # Create adapter config
        config = {
            "r": rank,
            "lora_alpha": source_config.get("lora_alpha", rank),
            "lora_dropout": source_config.get("lora_dropout", 0.0),
            "target_modules": source_config.get("target_modules", []),
            "bias": source_config.get("bias", "none"),
            "peft_type": "LORA",
            "base_model_name_or_path": target_base_model,
            "task_type": source_config.get("task_type", "CAUSAL_LM"),
            "transferred_from": source_config.get("base_model_name_or_path", "unknown"),
        }
        
        config_path = output_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved adapter to %s with %d weight tensors", output_path, len(output_weights))


# =============================================================================
# Convenience Functions
# =============================================================================


async def transfer_lora_adapter(
    source_adapter: str | Path,
    source_base: str,
    target_base: str,
    output: str | Path,
    rank: int | None = None,
    model_loader: "ModelLoaderPort | None" = None,
) -> LoRATransferResult:
    """Convenience function for adapter transfer.

    Args:
        source_adapter: Path to source adapter directory
        source_base: HuggingFace model ID of source base
        target_base: HuggingFace model ID of target base
        output: Path to save projected adapter
        rank: Target LoRA rank (None = preserve source)

    Returns:
        LoRATransferResult with transfer metrics
    """
    service = LoRATransferService(model_loader=model_loader)
    config = TransferConfig(target_rank=rank)
    
    return await service.transfer_adapter(
        source_adapter_path=Path(source_adapter),
        source_base_model_id=source_base,
        target_base_model_id=target_base,
        output_path=Path(output),
        config=config,
    )


__all__ = [
    "LoRATransferService",
    "TransferConfig",
    "TransferProgress",
    "transfer_lora_adapter",
]
