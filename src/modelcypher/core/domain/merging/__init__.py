"""
Model Merging Package.

Provides tools for merging models using geometric alignment.

Platform-Specific Implementations:
- MLX (macOS): *_mlx.py files
- CUDA (Linux): *_cuda.py files
- JAX (TPU/GPU): *_jax.py files
- Use _platform module for automatic selection
"""
from modelcypher.core.domain.merging.exceptions import MergeError
from .entropy_merge_validator import (
    EntropyMergeValidator,
    LayerEntropyProfile,
    ModelEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    MergeStability,
)
from .exceptions import MergeError
from .lora_adapter_merger_mlx import (
    LoRAAdapterMerger,
    Strategy as LoRAMergeStrategy,
    Config as LoRAMergeConfig,
    MergeReport as LoRAMergeReport,
)

# Platform selection (auto-detects MLX on macOS, CUDA on Linux, JAX on TPU)
from ._platform import (
    get_merging_platform,
    get_lora_adapter_merger_class,
    get_lora_merge_strategy_enum,
    get_lora_merge_config_class,
)

# Re-export from merge_engine (the canonical source)
from modelcypher.core.use_cases.merge_engine import (
    AnchorMode,
    ModuleScope,
    LayerMergeMetric,
    MergeAnalysisResult,
    RotationalMergeOptions as MergeOptions,
    RotationalMerger,
)

__all__ = [
    # Entropy Merge Validator
    "EntropyMergeValidator",
    "LayerEntropyProfile",
    "ModelEntropyProfile",
    "LayerMergeValidation",
    "MergeEntropyValidation",
    "MergeStability",
    # Merge Engine (canonical)
    "RotationalMerger",
    "MergeOptions",
    "MergeAnalysisResult",
    "LayerMergeMetric",
    "AnchorMode",
    "ModuleScope",
    "MergeError",
    # LoRA Adapter Merger (TIES/DARE-TIES)
    "LoRAAdapterMerger",
    "LoRAMergeStrategy",
    "LoRAMergeConfig",
    "LoRAMergeReport",
    # Platform selection
    "get_merging_platform",
    "get_lora_adapter_merger_class",
    "get_lora_merge_strategy_enum",
    "get_lora_merge_config_class",
]
