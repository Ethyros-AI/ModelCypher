"""
Model Merging Package.

Provides tools for merging models using geometric alignment.
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
from .rotational_merger import (
    RotationalModelMerger,
    MergeOptions,
    MergeAnalysisResult,
    AnchorMode,
    ModuleScope,
    LayerMergeMetric,
    merge_lora_adapters,
    weighted_merge,
)
from .exceptions import MergeError
from .lora_adapter_merger import (
    LoRAAdapterMerger,
    Strategy as LoRAMergeStrategy,
    Config as LoRAMergeConfig,
    MergeReport as LoRAMergeReport,
)

__all__ = [
    # Entropy Merge Validator
    "EntropyMergeValidator",
    "LayerEntropyProfile",
    "ModelEntropyProfile",
    "LayerMergeValidation",
    "MergeEntropyValidation",
    "MergeStability",
    # Rotational Merger
    "RotationalModelMerger",
    "MergeOptions",
    "MergeAnalysisResult",
    "LayerMergeMetric",
    "AnchorMode",
    "ModuleScope",
    "MergeError",
    "merge_lora_adapters",
    "weighted_merge",
    # LoRA Adapter Merger (TIES/DARE-TIES)
    "LoRAAdapterMerger",
    "LoRAMergeStrategy",
    "LoRAMergeConfig",
    "LoRAMergeReport",
]

