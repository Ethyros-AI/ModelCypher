"""
CUDA LoRA (Low-Rank Adaptation) Support for Parameter-Efficient Fine-Tuning.

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see lora_mlx.py
- JAX/TPU: see lora_jax.py

Use _platform.get_lora_config_class() for automatic platform selection.

Implementation based on PyTorch 2.x and safetensors 0.5.x best practices (2025):
- torch.nn.Module subclass with proper Parameter registration
- safetensors.torch for checkpoint I/O
- Gradient freezing via requires_grad=False
- Kaiming initialization for lora_a, zeros for lora_b

Research Basis:
- LoRA: arxiv:2106.09685
- DoRA: arxiv:2402.09353
- RSLoRA: arxiv:2312.03732

References:
- https://huggingface.co/docs/peft/en/developer_guides/lora
- https://huggingface.co/docs/safetensors/torch_shared_tensors
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
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class FineTuneTypeCUDA(str, Enum):
    """Fine-tuning method type."""
    LORA = "lora"
    DORA = "dora"  # Weight-Decomposed LoRA


@dataclass
class LoRAConfigCUDA:
    """Configuration for LoRA adapters (CUDA version)."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fine_tune_type: FineTuneTypeCUDA = FineTuneTypeCUDA.LORA
    num_layers: Optional[int] = None  # None = all layers
    use_rslora: bool = False  # Rank-Stabilized LoRA scaling

    @property
    def scale(self) -> float:
        """LoRA scaling factor: alpha / rank (or sqrt(rank) for RSLoRA)."""
        if self.use_rslora:
            return self.alpha / math.sqrt(max(self.rank, 1))
        return self.alpha / max(self.rank, 1)

    @classmethod
    def default(cls) -> "LoRAConfigCUDA":
        return cls()

    @classmethod
    def for_mistral(cls) -> "LoRAConfigCUDA":
        """Preset for Mistral-style models."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @classmethod
    def for_llama(cls) -> "LoRAConfigCUDA":
        """Preset for Llama-style models."""
        return cls(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )

    @classmethod
    def for_qwen(cls) -> "LoRAConfigCUDA":
        """Preset for Qwen-style models (gate in MLP)."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )


@dataclass
class TargetResolutionCUDA:
    """Result of resolving LoRA target modules."""
    resolved_keys: List[str]
    unmatched_modules: List[str]
    layer_count: int


@dataclass
class LoRAExportResultCUDA:
    """Result of exporting LoRA adapters."""
    path: Path
    parameter_count: int
    file_size_bytes: int


# =============================================================================
# LoRA Linear Layer
# =============================================================================

class LoRALinearCUDA(nn.Module):
    """
    Linear layer with LoRA adapters (CUDA/PyTorch version).

    Implements: y = Wx + (BA)x * scale
    Where A ∈ R^{r×d}, B ∈ R^{d×r}, scale = α/r

    The base weight W is frozen; only lora_a and lora_b are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.use_rslora = use_rslora

        # Compute scaling factor
        if use_rslora:
            self.scale = alpha / math.sqrt(max(rank, 1))
        else:
            self.scale = alpha / max(rank, 1)

        # Base linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA adapters (trainable)
        # A: down-projection (rank x in_features), initialized with Kaiming
        # B: up-projection (out_features x rank), initialized to zeros
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize lora_a with Kaiming uniform (as per PEFT best practices)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        # Dropout on input before LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base forward
        y = self.linear(x)

        # LoRA forward: x @ A^T @ B^T * scale
        lora_x = self.dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_a), self.lora_b)
        y = y + lora_out * self.scale

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ) -> "LoRALinearCUDA":
        """Create LoRALinear by wrapping an existing Linear layer."""
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        lora = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
            use_rslora=use_rslora,
        )

        # Copy frozen weights
        with torch.no_grad():
            lora.linear.weight.copy_(linear.weight)
            if has_bias and linear.bias is not None:
                lora.linear.bias.copy_(linear.bias)

        return lora

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base Linear layer."""
        merged_weight = self.linear.weight + self.scale * (self.lora_b @ self.lora_a)

        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.linear.bias is not None,
        )

        with torch.no_grad():
            linear.weight.copy_(merged_weight)
            if self.linear.bias is not None:
                linear.bias.copy_(self.linear.bias)

        return linear

    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get LoRA adapter parameters for export."""
        return {
            "lora_a": self.lora_a.data,
            "lora_b": self.lora_b.data,
        }


# =============================================================================
# Target Resolution
# =============================================================================

def resolve_lora_targets_cuda(
    model: nn.Module,
    config: LoRAConfigCUDA,
) -> TargetResolutionCUDA:
    """
    Resolve LoRA target modules within a PyTorch model.

    Scans model named_modules to find Linear layers matching target patterns.

    Args:
        model: The model to analyze
        config: LoRA configuration with target_modules

    Returns:
        TargetResolutionCUDA with matched keys and any unmatched targets
    """
    resolved_keys: List[str] = []
    matched_targets: Set[str] = set()

    # Build regex patterns for each target
    patterns = [re.compile(rf"(^|\.){target}$") for target in config.target_modules]

    # Scan all modules
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        for i, pattern in enumerate(patterns):
            if pattern.search(name):
                resolved_keys.append(name)
                matched_targets.add(config.target_modules[i])
                break

    # Apply layer limit if configured
    if config.num_layers is not None:
        filtered_keys = []
        for key in resolved_keys:
            # Extract layer index from paths like "model.layers.5.self_attn.q_proj"
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx < config.num_layers:
                    filtered_keys.append(key)
            else:
                # Keep modules without layer indices
                filtered_keys.append(key)
        resolved_keys = filtered_keys

    # Find unmatched targets
    unmatched = [t for t in config.target_modules if t not in matched_targets]

    # Count layers
    layer_indices = set()
    for key in resolved_keys:
        match = re.search(r"\.layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

    return TargetResolutionCUDA(
        resolved_keys=sorted(set(resolved_keys)),
        unmatched_modules=unmatched,
        layer_count=len(layer_indices),
    )


def apply_lora_to_model_cuda(
    model: nn.Module,
    config: LoRAConfigCUDA,
    target_keys: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject LoRA adapters into targeted Linear modules.

    Args:
        model: PyTorch model to modify
        config: LoRA configuration
        target_keys: Optional specific keys to target (auto-resolves if None)

    Returns:
        Modified model with LoRA adapters injected
    """
    if target_keys is None:
        resolution = resolve_lora_targets_cuda(model, config)
        target_keys = resolution.resolved_keys
        logger.info(
            "LoRA: Auto-resolved %d targets across %d layers",
            len(resolution.resolved_keys),
            resolution.layer_count,
        )
        if resolution.unmatched_modules:
            logger.warning(
                "LoRA: Unmatched target patterns: %s",
                resolution.unmatched_modules,
            )

    def get_parent_and_name(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
        """Get parent module and child name from dotted path."""
        parts = path.rsplit(".", 1)
        if len(parts) == 1:
            return root, parts[0]

        parent_path = parts[0]
        child_name = parts[1]

        parent = root
        for part in parent_path.split("."):
            parent = getattr(parent, part)

        return parent, child_name

    count = 0
    for key in target_keys:
        try:
            parent, child_name = get_parent_and_name(model, key)
            linear = getattr(parent, child_name)

            if isinstance(linear, nn.Linear):
                lora = LoRALinearCUDA.from_linear(
                    linear,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    use_rslora=config.use_rslora,
                )
                setattr(parent, child_name, lora)
                count += 1
        except (AttributeError, KeyError) as e:
            logger.warning("LoRA: Failed to inject at %s: %s", key, e)

    logger.info("LoRA: Injected adapters into %d modules", count)
    return model


# =============================================================================
# Export / Import
# =============================================================================

def export_lora_adapters_cuda(
    model: nn.Module,
    output_path: Path,
    config: LoRAConfigCUDA,
    model_id: str = "",
) -> LoRAExportResultCUDA:
    """
    Export trained LoRA adapter weights to a safetensors file.

    Only extracts LoRA A/B matrices, not frozen weights.

    Args:
        model: Trained model with LoRA layers
        output_path: Destination path for adapter weights
        config: LoRA configuration used during training
        model_id: Optional model identifier for metadata

    Returns:
        LoRAExportResultCUDA with path and statistics
    """
    adapter_weights: Dict[str, torch.Tensor] = {}

    # Extract all LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearCUDA):
            # Store with module path prefix
            adapter_weights[f"{name}.lora_a"] = module.lora_a.data.cpu()
            adapter_weights[f"{name}.lora_b"] = module.lora_b.data.cpu()

    if not adapter_weights:
        raise ValueError("No LoRA parameters found in model")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using safetensors
    save_file(adapter_weights, str(output_path))

    # Calculate stats
    param_count = sum(w.numel() for w in adapter_weights.values())
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

    return LoRAExportResultCUDA(
        path=output_path,
        parameter_count=param_count,
        file_size_bytes=file_size,
    )


def load_lora_adapters_cuda(
    model: nn.Module,
    adapter_path: Path,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Load LoRA adapter weights into a model.

    Args:
        model: Model with LoRA layers to load adapters into
        adapter_path: Path to adapter safetensors file
        device: Target device for loaded weights

    Returns:
        Model with loaded adapter weights
    """
    # Load with device placement
    adapters = load_file(str(adapter_path), device=device)

    loaded_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinearCUDA):
            lora_a_key = f"{name}.lora_a"
            lora_b_key = f"{name}.lora_b"

            if lora_a_key in adapters and lora_b_key in adapters:
                with torch.no_grad():
                    module.lora_a.copy_(adapters[lora_a_key])
                    module.lora_b.copy_(adapters[lora_b_key])
                loaded_count += 1

    logger.info("LoRA: Loaded adapters into %d modules from %s", loaded_count, adapter_path)

    return model


# =============================================================================
# Adapter Geometry (for tracking)
# =============================================================================

def snapshot_lora_parameters_cuda(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot LoRA trainable parameters for trajectory tracking.

    Used by geometric metrics collector to track training dynamics.
    """
    snapshot: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinearCUDA):
            snapshot[f"{name}.lora_a"] = module.lora_a.data.clone()
            snapshot[f"{name}.lora_b"] = module.lora_b.data.clone()

    return snapshot


def compute_adapter_norm_cuda(adapters: Dict[str, torch.Tensor]) -> float:
    """Compute Frobenius norm of all adapter weights."""
    total = 0.0
    for weight in adapters.values():
        total += float(torch.sum(weight ** 2).item())
    return math.sqrt(total)


def compute_adapter_delta_norm_cuda(
    initial: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
) -> float:
    """Compute norm of weight change from initial to current."""
    total = 0.0
    for name in initial:
        if name in current:
            delta = current[name] - initial[name]
            total += float(torch.sum(delta ** 2).item())
    return math.sqrt(total)


__all__ = [
    "FineTuneTypeCUDA",
    "LoRAConfigCUDA",
    "TargetResolutionCUDA",
    "LoRAExportResultCUDA",
    "LoRALinearCUDA",
    "resolve_lora_targets_cuda",
    "apply_lora_to_model_cuda",
    "export_lora_adapters_cuda",
    "load_lora_adapters_cuda",
    "snapshot_lora_parameters_cuda",
    "compute_adapter_norm_cuda",
    "compute_adapter_delta_norm_cuda",
]
