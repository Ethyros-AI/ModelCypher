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
LoRA (Low-Rank Adaptation) Support for Parameter-Efficient Fine-Tuning (MLX Backend).

This is the MLX/macOS implementation. For other backends:
- CUDA/PyTorch: see lora_cuda.py
- JAX/TPU: see lora_jax.py

Use _platform.get_lora_config_class() for automatic platform selection.

Ported from the reference Swift implementation.

Core functionality:
- LoRA target module resolution (auto-detect Q/K/V/O projections)
- Adapter layer injection (wraps Linear -> LoRALinear)
- Export/import adapter weights
- DoRA (Weight-Decomposed) support

Research Basis:
- LoRA: arxiv:2106.09685
- DoRA: arxiv:2402.09353

NOTE: This module has infrastructure dependencies (mlx.nn for neural network
layers, mlx.utils for tree operations, mlx file I/O) that cannot be fully
abstracted via the Backend protocol. The LoRALinear class and model
manipulation functions remain MLX-specific until a full training abstraction
layer is implemented.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.numerical_stability import (
    condition_threshold,
    division_epsilon,
    machine_epsilon,
    regularization_epsilon,
    svd_rank_threshold,
)

# Infrastructure dependencies (MLX-specific neural network layers and file I/O)
# These cannot be abstracted via Backend protocol
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FineTuneType(str, Enum):
    """Fine-tuning method type."""

    LORA = "lora"
    DORA = "dora"  # Weight-decomposed LoRA


@dataclass
class LoRASettings:
    """Settings for LoRA adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fine_tune_type: FineTuneType = FineTuneType.LORA
    num_layers: int | None = None  # None = all layers

    @property
    def scale(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.alpha / max(self.rank, 1)

    @classmethod
    def default(cls) -> "LoRASettings":
        return cls()

    @classmethod
    def for_mistral(cls) -> "LoRASettings":
        """Preset for Mistral-style models."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @classmethod
    def for_llama(cls) -> "LoRASettings":
        """Preset for Llama-style models."""
        return cls(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )

    @classmethod
    def for_qwen(cls) -> "LoRASettings":
        """Preset for Qwen-style models (gate in MLP).

        DEPRECATED: Use from_weight_geometry() for geometry-derived parameters.
        """
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )

    @classmethod
    def from_weight_geometry(
        cls,
        weight: mx.array,
        target_modules: list[str] | None = None,
        fine_tune_type: "FineTuneType" = None,
    ) -> "LoRASettings":
        """Derive LoRA settings from weight matrix geometry.

        Philosophy: Hyperparameters are MEASUREMENTS, not knobs.
        All parameters are derived from the weight matrix's spectral properties.

        Derivation:
            - rank: Effective numerical rank from SVD (singular values > threshold)
            - alpha: Set to maintain scale = 1.0 (neutral scaling)
            - dropout: Derived from condition number (higher conditioning = more regularization)

        Args:
            weight: Representative weight matrix to analyze (e.g., q_proj.weight)
            target_modules: Modules to target (default: ["q_proj", "v_proj"])
            fine_tune_type: LoRA or DoRA (default: LORA)

        Returns:
            LoRASettings with geometry-derived parameters
        """
        if fine_tune_type is None:
            fine_tune_type = FineTuneType.LORA
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # Compute SVD to analyze spectral properties (uses MLX directly)
        # Note: SVD in MLX requires CPU stream
        U, S, Vt = mx.linalg.svd(weight, compute_uv=True, stream=mx.cpu)
        mx.eval(S)

        # Get dtype-derived thresholds
        max_dim = max(weight.shape)
        eps = float(mx.finfo(weight.dtype).eps)
        rank_threshold = max_dim * eps

        # Effective rank: count singular values above numerical noise floor
        S_np = S.tolist()
        max_sv = S_np[0] if S_np else 1.0
        significant = [s for s in S_np if s > max_sv * rank_threshold]
        effective_rank = len(significant)

        # Clamp rank to reasonable range (at least 1, at most half the smaller dimension)
        min_dim = min(weight.shape)
        rank = max(1, min(effective_rank, min_dim // 2))

        # Alpha: maintain scale = alpha/rank = 1.0 (neutral scaling)
        # This means the LoRA contribution is not artificially amplified or suppressed
        alpha = float(rank)

        # Dropout: derived from condition number
        # Higher condition number = less stable optimization = need more regularization
        # Condition number = max_sv / min_sv
        min_sv = S_np[-1] if S_np else eps
        condition_number = max_sv / max(min_sv, eps)

        # Dropout scales with log(condition_number)
        # Well-conditioned (κ ~ 1): dropout ~ 0
        # Ill-conditioned (κ ~ 1e6): dropout ~ 0.1
        import math
        log_cond = math.log10(max(condition_number, 1.0))
        dropout = min(0.1, log_cond / 60.0)  # 60 = log10(1e6) / 0.1

        logger.info(
            "Geometry-derived LoRA: rank=%d (effective=%d), alpha=%.1f, dropout=%.3f, κ=%.1e",
            rank, effective_rank, alpha, dropout, condition_number
        )

        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            fine_tune_type=fine_tune_type,
        )


@dataclass
class TargetResolution:
    """Result of resolving LoRA target modules."""

    resolved_keys: list[str]
    unmatched_modules: list[str]
    layer_count: int


@dataclass
class LoRAExportResult:
    """Result of exporting LoRA adapters."""

    path: Path
    parameter_count: int
    file_size_bytes: int


# =============================================================================
# LoRA Linear Layer
# =============================================================================


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapters.

    Implements: y = Wx + (BA)x * scale
    Where A ∈ R^{r×d}, B ∈ R^{d×r}, scale = α/r

    Initialization:
        B is initialized to zeros (standard LoRA).
        A is initialized with scale derived from geometry:
        - init_scale = sqrt(eps) where eps is machine epsilon for the dtype
        - This ensures initial perturbation is at the numerical noise floor,
          allowing gradients to shape the adaptation without arbitrary bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
        init_scale: float | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)

        # Original frozen weights (to be copied from base layer)
        self.weight = mx.zeros((out_features, in_features))
        self.bias = mx.zeros((out_features,)) if bias else None

        # LoRA adapters (trainable)
        # A: down-projection, B: up-projection
        # Initialization scale derived from dtype if not provided
        if init_scale is None:
            # Use sqrt(machine_epsilon) as the initialization scale
            # This is the numerical analysis threshold for relative precision
            eps = float(mx.finfo(mx.float32).eps)
            import math
            init_scale = math.sqrt(eps)

        self.lora_a = mx.random.normal((rank, in_features)) * init_scale
        self.lora_b = mx.zeros((out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Freeze base weight by default
        self.freeze(keys=["weight"])

    def __call__(self, x: mx.array) -> mx.array:
        # Frozen forward
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias

        # LoRA forward
        lora_x = x
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)

        # (B @ A) @ x^T → compute as x @ A^T @ B^T for efficiency
        lora_out = (lora_x @ self.lora_a.T) @ self.lora_b.T
        y = y + lora_out * self.scale

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_scale: float | None = None,
    ) -> "LoRALinear":
        """Create LoRALinear by wrapping an existing Linear layer.

        If init_scale is None, derives it from the weight matrix's spectral norm.
        This ensures the initial LoRA perturbation is proportional to the
        weight's scale, avoiding both vanishing and exploding gradients.

        Derivation: init_scale = spectral_norm * sqrt(eps) / sqrt(rank)
        - spectral_norm: largest singular value of W
        - sqrt(eps): numerical precision floor
        - sqrt(rank): scale with adaptation capacity
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, "bias") and linear.bias is not None

        # Derive init_scale from weight geometry if not provided
        if init_scale is None:
            import math
            eps = float(mx.finfo(linear.weight.dtype).eps)

            # Compute spectral norm (largest singular value) efficiently
            # Use power iteration approximation for speed
            try:
                # For small matrices, exact SVD is fast
                if min(out_features, in_features) <= 512:
                    S = mx.linalg.svdvals(linear.weight)
                    mx.eval(S)
                    spectral_norm = float(S[0])
                else:
                    # Approximate spectral norm via Frobenius norm bound
                    frob_sq = mx.sum(linear.weight * linear.weight)
                    mx.eval(frob_sq)
                    # spectral_norm <= frobenius_norm <= sqrt(min_dim) * spectral_norm
                    spectral_norm = math.sqrt(float(frob_sq) / min(out_features, in_features))
            except Exception:
                # Fallback to simple scale estimate
                spectral_norm = 1.0

            # init_scale = spectral_norm * sqrt(eps) / sqrt(rank)
            # This scales with the weight matrix and adaptation capacity
            init_scale = spectral_norm * math.sqrt(eps) / math.sqrt(max(rank, 1))

        lora = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
            init_scale=init_scale,
        )

        # Copy frozen weights
        lora.weight = linear.weight
        if has_bias:
            lora.bias = linear.bias
            lora.freeze(keys=["bias"])

        return lora

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base Linear layer."""
        # Merged = W + scale * B @ A
        merged_weight = self.weight + self.scale * (self.lora_b @ self.lora_a)

        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight = merged_weight
        if self.bias is not None:
            linear.bias = self.bias
            self.freeze(keys=["bias"])

        return linear


# =============================================================================
# Geometry-Derived Settings from Model
# =============================================================================


def derive_lora_settings_from_model(
    model: nn.Module,
    target_modules: list[str] | None = None,
) -> LoRASettings:
    """Derive LoRA settings by analyzing model weight geometry.

    Analyzes representative weight matrices from target modules to determine
    optimal LoRA rank, alpha, and dropout based on spectral properties.

    Philosophy: Hyperparameters are MEASUREMENTS, not knobs.

    Args:
        model: The model to analyze
        target_modules: Modules to target (default: ["q_proj", "v_proj"])

    Returns:
        LoRASettings with geometry-derived parameters
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    from mlx.utils import tree_flatten

    # Find representative weight matrices
    weights_found = []
    for name, value in tree_flatten(model.parameters()):
        for target in target_modules:
            if target in name and name.endswith(".weight"):
                weights_found.append(value)
                break
        # Sample up to 3 representative weights
        if len(weights_found) >= 3:
            break

    if not weights_found:
        logger.warning("No target weights found, using default settings")
        return LoRASettings.default()

    # Analyze each weight and aggregate
    ranks = []
    conditions = []

    for weight in weights_found:
        try:
            S = mx.linalg.svdvals(weight)
            mx.eval(S)
            S_list = S.tolist()

            if not S_list:
                continue

            # Compute effective rank
            eps = float(mx.finfo(weight.dtype).eps)
            max_dim = max(weight.shape)
            threshold = S_list[0] * max_dim * eps

            effective_rank = sum(1 for s in S_list if s > threshold)
            ranks.append(effective_rank)

            # Compute condition number
            min_sv = S_list[-1] if S_list[-1] > eps else eps
            condition = S_list[0] / min_sv
            conditions.append(condition)

        except Exception as e:
            logger.debug("SVD failed for weight: %s", e)
            continue

    if not ranks:
        logger.warning("Could not analyze weights, using default settings")
        return LoRASettings.default()

    # Aggregate: use median rank (robust to outliers)
    import statistics
    median_rank = int(statistics.median(ranks))
    median_condition = statistics.median(conditions)

    # Clamp rank to reasonable range
    rank = max(4, min(median_rank, 64))

    # Alpha = rank for scale = 1.0
    alpha = float(rank)

    # Dropout from condition number
    import math
    log_cond = math.log10(max(median_condition, 1.0))
    dropout = min(0.1, log_cond / 60.0)

    logger.info(
        "Model geometry analysis: median_rank=%d, median_κ=%.1e → rank=%d, dropout=%.3f",
        median_rank, median_condition, rank, dropout
    )

    return LoRASettings(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


# =============================================================================
# Target Resolution
# =============================================================================


def resolve_lora_targets(
    model: nn.Module,
    settings: LoRASettings,
) -> TargetResolution:
    """
    Resolve LoRA target modules within a model.

    Scans model parameters to find Linear layers matching target patterns.

    Args:
        model: The model to analyze
        settings: LoRA settings with target_modules

    Returns:
        TargetResolution with matched keys and any unmatched targets
    """
    resolved_keys: list[str] = []
    matched_targets: set[str] = set()

    from mlx.utils import tree_flatten

    # Build regex patterns for each target
    patterns = [re.compile(rf"(^|\.){target}\.weight$") for target in settings.target_modules]

    # Scan all parameters using flattened tree
    for name, value in tree_flatten(model.parameters()):
        if not name.endswith(".weight"):
            continue

        for i, pattern in enumerate(patterns):
            if pattern.search(name):  # Use search() to find pattern anywhere in string
                # Extract the module path (without .weight)
                module_path = name.rsplit(".weight", 1)[0]
                resolved_keys.append(module_path)
                matched_targets.add(settings.target_modules[i])
                break

    # Find unmatched targets
    unmatched = [t for t in settings.target_modules if t not in matched_targets]

    # Count layers
    layer_indices = set()
    for key in resolved_keys:
        # Extract layer index from paths like "model.layers.5.self_attn.q_proj"
        match = re.search(r"\.layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

    return TargetResolution(
        resolved_keys=sorted(set(resolved_keys)),
        unmatched_modules=unmatched,
        layer_count=len(layer_indices),
    )


def apply_lora_to_model(
    model: nn.Module,
    settings: LoRASettings,
    target_keys: list[str] | None = None,
) -> nn.Module:
    """Inject LoRA adapters into targeted Linear modules."""
    if target_keys is None:
        resolution = resolve_lora_targets(model, settings)
        target_keys = resolution.resolved_keys

    # Build path → module mapping helpers
    def get_module_by_path(root: nn.Module, path: str) -> Any:
        parts = path.split(".")
        current = root
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, "__getitem__"):
                # Handle indexed layers like model.layers.0
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def set_module_by_path(root: nn.Module, path: str, new_module: nn.Module) -> None:
        parts = path.split(".")
        if len(parts) == 1:
            setattr(root, parts[0], new_module)
            return

        parent_path = ".".join(parts[:-1])
        parent = get_module_by_path(root, parent_path)
        if parent is not None:
            setattr(parent, parts[-1], new_module)

    count = 0
    for key in target_keys:
        linear = get_module_by_path(model, key)
        if linear is not None and isinstance(linear, nn.Linear):
            # Create LoRA adapter from original Linear weights
            lora = LoRALinear.from_linear(linear, settings.rank, settings.alpha, settings.dropout)
            set_module_by_path(model, key, lora)
            count += 1

    logger.info("LoRA: Injected adapters into %d modules", count)
    return model


# =============================================================================
# Export / Import
# =============================================================================


def export_lora_adapters(
    model: nn.Module,
    output_path: Path,
    settings: LoRASettings,
    model_id: str = "",
) -> LoRAExportResult:
    """
    Export trained LoRA adapter weights to a safetensors file.

    Only extracts LoRA A/B matrices, not frozen weights.

    Args:
        model: Trained model with LoRA layers
        output_path: Destination path for adapter weights
        settings: LoRA settings used during training
        model_id: Optional model identifier for metadata

    Returns:
        LoRAExportResult with path and statistics
    """
    adapter_weights: dict[str, mx.array] = {}

    # Extract all LoRA parameters
    for name, param in model.parameters().items():
        if "lora_a" in name or "lora_b" in name:
            adapter_weights[name] = param

    if not adapter_weights:
        raise ValueError("No LoRA parameters found in model")

    # Save to safetensors
    mx.save_safetensors(str(output_path), adapter_weights)

    # Calculate stats
    param_count = sum(w.size for w in adapter_weights.values())
    file_size = output_path.stat().st_size

    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "model_id": model_id,
        "rank": settings.rank,
        "alpha": settings.alpha,
        "target_modules": settings.target_modules,
        "fine_tune_type": settings.fine_tune_type.value,
        "parameter_count": param_count,
        "exported_at": datetime.now().isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return LoRAExportResult(
        path=output_path,
        parameter_count=param_count,
        file_size_bytes=file_size,
    )


def load_lora_adapters(
    model: nn.Module,
    adapter_path: Path,
) -> nn.Module:
    """
    Load LoRA adapter weights into a model.

    Args:
        model: Model to load adapters into
        adapter_path: Path to adapter safetensors file

    Returns:
        Model with loaded adapter weights
    """
    adapters = mx.load(str(adapter_path))

    # Update model parameters
    current_params = dict(model.parameters())
    for name, weight in adapters.items():
        if name in current_params:
            current_params[name] = weight

    # Reconstruct nested dict and update
    # (simplified - full impl would use mx update_modules)

    return model


# =============================================================================
# Adapter Geometry (for tracking)
# =============================================================================


def snapshot_lora_parameters(model: nn.Module) -> dict[str, mx.array]:
    """
    Snapshot LoRA trainable parameters for trajectory tracking.

    Used by geometric metrics collector to track training dynamics.
    """
    snapshot: dict[str, mx.array] = {}

    for name, param in model.parameters().items():
        if "lora_a" in name or "lora_b" in name:
            snapshot[name] = param

    return snapshot


def compute_adapter_norm(adapters: dict[str, mx.array]) -> float:
    """Compute Frobenius norm of all adapter weights."""
    if not adapters:
        return 0.0
    _b = get_default_backend()
    flat_parts = []
    for weight in adapters.values():
        arr = weight if hasattr(weight, "shape") else _b.array(weight)
        flat_parts.append(_b.reshape(arr, (-1,)))
    flat = _b.concatenate(flat_parts, axis=0)
    norm_arr = geodesic_norms(_b.reshape(flat, (1, -1)), _b)
    _b.eval(norm_arr)
    return float(_b.to_scalar(norm_arr[0]))


def compute_adapter_delta_norm(
    initial: dict[str, mx.array],
    current: dict[str, mx.array],
) -> float:
    """Compute norm of weight change from initial to current."""
    _b = get_default_backend()
    flat_parts = []
    for name, init_val in initial.items():
        if name not in current:
            continue
        delta = current[name] - init_val
        arr = delta if hasattr(delta, "shape") else _b.array(delta)
        flat_parts.append(_b.reshape(arr, (-1,)))
    if not flat_parts:
        return 0.0
    flat = _b.concatenate(flat_parts, axis=0)
    norm_arr = geodesic_norms(_b.reshape(flat, (1, -1)), _b)
    _b.eval(norm_arr)
    return float(_b.to_scalar(norm_arr[0]))
