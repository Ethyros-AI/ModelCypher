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
from .exceptions import MergeError
from .lora_adapter_merger import (
    LoRAAdapterMerger,
    Strategy as LoRAMergeStrategy,
    Config as LoRAMergeConfig,
    MergeReport as LoRAMergeReport,
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
]
