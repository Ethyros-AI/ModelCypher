"""
Model Merging Package.

Provides geometric alignment for merging models and adapters.

The ONE correct merge method:
1. Probe models with semantic primes to build intersection map
2. Permutation align (re-basin neurons)
3. Procrustes rotate (align weight spaces)
4. Blend with confidence-weighted alpha

No strategy options. No heuristic dropout. Just geometry.
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
from .lora_adapter_merger import (
    LoRAAdapterMerger,
    MergeReport as LoRAMergeReport,
)

# Platform detection (for backend selection, not merger selection)
from ._platform import (
    get_merging_platform,
    get_lora_adapter_merger_class,
)

# Re-export from merge_engine (the canonical geometric merge)
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
    # Merge Engine (canonical geometric merge)
    "RotationalMerger",
    "MergeOptions",
    "MergeAnalysisResult",
    "LayerMergeMetric",
    "AnchorMode",
    "ModuleScope",
    "MergeError",
    # LoRA Adapter Merger (geometric)
    "LoRAAdapterMerger",
    "LoRAMergeReport",
    # Platform detection
    "get_merging_platform",
    "get_lora_adapter_merger_class",
]
