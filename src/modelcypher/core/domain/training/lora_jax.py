"""
JAX LoRA (Low-Rank Adaptation) Support for Parameter-Efficient Fine-Tuning.

This is the JAX/Flax implementation. For other backends:
- MLX/macOS: see lora_mlx.py
- CUDA/PyTorch: see lora_cuda.py

Use _platform.get_lora_config_class() for automatic platform selection.

Implementation based on JAX/Flax best practices (2025):
- Flax NNX or Linen module patterns
- Pure functional parameter handling
- JAX pytree operations for parameter traversal
- numpy serialization for checkpoints

Research Basis:
- LoRA: arxiv:2106.09685
- DoRA: arxiv:2402.09353
- RSLoRA: arxiv:2312.03732

References:
- https://flax.readthedocs.io/en/stable/
- https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class FineTuneTypeJAX(str, Enum):
    """Fine-tuning method type."""
    LORA = "lora"
    DORA = "dora"  # Weight-decomposed LoRA


@dataclass
class LoRAConfigJAX:
    """Configuration for LoRA adapters (JAX version)."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fine_tune_type: FineTuneTypeJAX = FineTuneTypeJAX.LORA
    num_layers: Optional[int] = None  # None = all layers
    use_rslora: bool = False  # Rank-Stabilized LoRA scaling

    @property
    def scale(self) -> float:
        """LoRA scaling factor: alpha / rank (or sqrt(rank) for RSLoRA)."""
        if self.use_rslora:
            return self.alpha / math.sqrt(max(self.rank, 1))
        return self.alpha / max(self.rank, 1)

    @classmethod
    def default(cls) -> "LoRAConfigJAX":
        return cls()

    @classmethod
    def for_mistral(cls) -> "LoRAConfigJAX":
        """Preset for Mistral-style models."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @classmethod
    def for_llama(cls) -> "LoRAConfigJAX":
        """Preset for Llama-style models."""
        return cls(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )

    @classmethod
    def for_qwen(cls) -> "LoRAConfigJAX":
        """Preset for Qwen-style models (gate in MLP)."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )


@dataclass
class TargetResolutionJAX:
    """Result of resolving LoRA target modules."""
    resolved_keys: List[str]
    unmatched_modules: List[str]
    layer_count: int


@dataclass
class LoRAExportResultJAX:
    """Result of exporting LoRA adapters."""
    path: Path
    parameter_count: int
    file_size_bytes: int


# =============================================================================
# LoRA Parameter Initialization and Forward Pass
# =============================================================================

def init_lora_params(
    key: jax.random.PRNGKey,
    in_features: int,
    out_features: int,
    rank: int = 8,
) -> Dict[str, jnp.ndarray]:
    """
    Initialize LoRA adapter parameters.

    Args:
        key: JAX random key
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank

    Returns:
        Dict with 'lora_a' and 'lora_b' arrays
    """
    key_a, key_b = jax.random.split(key)

    # lora_a: (rank, in_features), initialized with small normal
    lora_a = jax.random.normal(key_a, (rank, in_features)) * 0.01

    # lora_b: (out_features, rank), initialized to zeros
    lora_b = jnp.zeros((out_features, rank))

    return {"lora_a": lora_a, "lora_b": lora_b}


def lora_forward(
    x: jnp.ndarray,
    base_weight: jnp.ndarray,
    lora_a: jnp.ndarray,
    lora_b: jnp.ndarray,
    scale: float,
    base_bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    LoRA forward pass: y = Wx + b + (BA)x * scale

    Args:
        x: Input tensor [batch, ..., in_features]
        base_weight: Frozen weight [out_features, in_features]
        lora_a: Down-projection [rank, in_features]
        lora_b: Up-projection [out_features, rank]
        scale: LoRA scaling factor (alpha / rank)
        base_bias: Optional bias [out_features]

    Returns:
        Output tensor [batch, ..., out_features]
    """
    # Base forward: x @ W^T
    y = x @ base_weight.T
    if base_bias is not None:
        y = y + base_bias

    # LoRA forward: x @ A^T @ B^T * scale
    lora_out = (x @ lora_a.T) @ lora_b.T
    y = y + lora_out * scale

    return y


# =============================================================================
# Target Resolution
# =============================================================================

def resolve_lora_targets_jax(
    params: Dict[str, Any],
    config: LoRAConfigJAX,
) -> TargetResolutionJAX:
    """
    Resolve LoRA target modules within a JAX params pytree.

    Scans flattened parameters to find Dense/Linear layers matching target patterns.

    Args:
        params: Model parameters (JAX pytree)
        config: LoRA configuration with target_modules

    Returns:
        TargetResolutionJAX with matched keys and any unmatched targets
    """
    resolved_keys: List[str] = []
    matched_targets: Set[str] = set()

    # Build regex patterns for each target
    patterns = [re.compile(rf"(^|/)({target})(/|$)") for target in config.target_modules]

    # Flatten params to get all paths
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)

    for path, _ in flat_params:
        # Convert path to string
        path_str = "/".join(str(p.key) for p in path if hasattr(p, 'key'))

        # Check if this is a kernel/weight parameter
        if not (path_str.endswith("kernel") or path_str.endswith("weight")):
            continue

        for i, pattern in enumerate(patterns):
            if pattern.search(path_str):
                # Get module path (remove kernel/weight suffix)
                module_path = "/".join(path_str.split("/")[:-1])
                resolved_keys.append(module_path)
                matched_targets.add(config.target_modules[i])
                break

    # Apply layer limit if configured
    if config.num_layers is not None:
        filtered_keys = []
        for key in resolved_keys:
            match = re.search(r"/layers/(\d+)/", key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx < config.num_layers:
                    filtered_keys.append(key)
            else:
                filtered_keys.append(key)
        resolved_keys = filtered_keys

    # Find unmatched targets
    unmatched = [t for t in config.target_modules if t not in matched_targets]

    # Count layers
    layer_indices = set()
    for key in resolved_keys:
        match = re.search(r"/layers/(\d+)/", key)
        if match:
            layer_indices.add(int(match.group(1)))

    return TargetResolutionJAX(
        resolved_keys=sorted(set(resolved_keys)),
        unmatched_modules=unmatched,
        layer_count=len(layer_indices),
    )


def create_lora_params(
    key: jax.random.PRNGKey,
    params: Dict[str, Any],
    config: LoRAConfigJAX,
    target_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Create LoRA parameters for targeted modules.

    Args:
        key: JAX random key
        params: Model parameters (to get shapes)
        config: LoRA configuration
        target_keys: Optional specific keys to target

    Returns:
        Dict mapping module paths to LoRA params
    """
    if target_keys is None:
        resolution = resolve_lora_targets_jax(params, config)
        target_keys = resolution.resolved_keys
        logger.info(
            "LoRA: Auto-resolved %d targets across %d layers",
            len(resolution.resolved_keys),
            resolution.layer_count,
        )

    lora_params = {}

    for i, module_path in enumerate(target_keys):
        key, subkey = jax.random.split(key)

        # Get weight shape from params
        # Navigate to the weight in the pytree
        parts = module_path.split("/")
        weight = params
        for part in parts:
            if isinstance(weight, dict) and part in weight:
                weight = weight[part]

        # Get kernel/weight
        if isinstance(weight, dict):
            if "kernel" in weight:
                weight = weight["kernel"]
            elif "weight" in weight:
                weight = weight["weight"]

        if hasattr(weight, "shape") and len(weight.shape) == 2:
            out_features, in_features = weight.shape
            lora_params[module_path] = init_lora_params(
                subkey, in_features, out_features, config.rank
            )

    logger.info("LoRA: Created adapters for %d modules", len(lora_params))
    return lora_params


# =============================================================================
# Export / Import
# =============================================================================

def export_lora_adapters_jax(
    lora_params: Dict[str, Dict[str, jnp.ndarray]],
    output_path: Path,
    config: LoRAConfigJAX,
    model_id: str = "",
) -> LoRAExportResultJAX:
    """
    Export trained LoRA adapter weights.

    Args:
        lora_params: LoRA parameters dict
        output_path: Destination path
        config: LoRA configuration
        model_id: Optional model identifier

    Returns:
        LoRAExportResultJAX with path and statistics
    """
    # Flatten for saving
    flat_params = {}
    for module_path, adapters in lora_params.items():
        for key, value in adapters.items():
            flat_key = f"{module_path}/{key}"
            flat_params[flat_key] = np.array(value)

    if not flat_params:
        raise ValueError("No LoRA parameters to export")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as npz
    np.savez(str(output_path), **flat_params)

    # Calculate stats
    param_count = sum(v.size for v in flat_params.values())
    file_size = output_path.stat().st_size

    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "model_id": model_id,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "target_modules": config.target_modules,
        "fine_tune_type": config.fine_tune_type.value,
        "use_rslora": config.use_rslora,
        "parameter_count": param_count,
        "exported_at": datetime.now().isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "LoRA: Exported %d parameters (%.2f MB) to %s",
        param_count,
        file_size / 1e6,
        output_path,
    )

    return LoRAExportResultJAX(
        path=output_path,
        parameter_count=param_count,
        file_size_bytes=file_size,
    )


def load_lora_adapters_jax(
    adapter_path: Path,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Load LoRA adapter weights.

    Args:
        adapter_path: Path to adapter npz file

    Returns:
        Dict mapping module paths to LoRA params
    """
    loaded = np.load(str(adapter_path))

    # Reconstruct nested structure
    lora_params: Dict[str, Dict[str, jnp.ndarray]] = {}

    for key, value in loaded.items():
        parts = key.rsplit("/", 1)
        if len(parts) == 2:
            module_path, param_name = parts
            if module_path not in lora_params:
                lora_params[module_path] = {}
            lora_params[module_path][param_name] = jnp.array(value)

    logger.info("LoRA: Loaded adapters for %d modules from %s", len(lora_params), adapter_path)
    return lora_params


# =============================================================================
# Adapter Geometry (for tracking)
# =============================================================================

def snapshot_lora_parameters_jax(
    lora_params: Dict[str, Dict[str, jnp.ndarray]],
) -> Dict[str, jnp.ndarray]:
    """
    Snapshot LoRA trainable parameters for trajectory tracking.

    Flattens the nested structure for easier norm computation.
    """
    snapshot = {}
    for module_path, adapters in lora_params.items():
        for key, value in adapters.items():
            snapshot[f"{module_path}/{key}"] = value
    return snapshot


def compute_adapter_norm_jax(adapters: Dict[str, jnp.ndarray]) -> float:
    """Compute Frobenius norm of all adapter weights."""
    total = 0.0
    for weight in adapters.values():
        total += float(jnp.sum(weight ** 2))
    return math.sqrt(total)


def compute_adapter_delta_norm_jax(
    initial: Dict[str, jnp.ndarray],
    current: Dict[str, jnp.ndarray],
) -> float:
    """Compute norm of weight change from initial to current."""
    total = 0.0
    for name in initial:
        if name in current:
            delta = current[name] - initial[name]
            total += float(jnp.sum(delta ** 2))
    return math.sqrt(total)


__all__ = [
    "FineTuneTypeJAX",
    "LoRAConfigJAX",
    "TargetResolutionJAX",
    "LoRAExportResultJAX",
    "init_lora_params",
    "lora_forward",
    "resolve_lora_targets_jax",
    "create_lora_params",
    "export_lora_adapters_jax",
    "load_lora_adapters_jax",
    "snapshot_lora_parameters_jax",
    "compute_adapter_norm_jax",
    "compute_adapter_delta_norm_jax",
]
