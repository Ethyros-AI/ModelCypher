"""
Model Merging Package.

Provides tools for merging models using geometric alignment.
"""
from modelcypher.core.domain.merging.exceptions import MergeError
from .rotational_merger import (
    RotationalModelMerger,
    MergeOptions,
    MergeAnalysisResult,
    LayerMergeMetric,
    AnchorMode,
    ModuleScope,
    merge_lora_adapters,
    weighted_merge,
)
from .lora_adapter_merger import (
    LoRAAdapterMerger,
    Strategy as LoRAMergeStrategy,
    Config as LoRAMergeConfig,
    MergeReport as LoRAMergeReport,
)
from .unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    UnifiedMergeResult,
    LayerAlphaProfile,
    compute_adaptive_alpha_profile,
    compute_spectral_penalty,
    DimensionBlendingWeights,
)

__all__ = [
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
    # Unified Manifold Merger
    "UnifiedManifoldMerger",
    "UnifiedMergeConfig",
    "UnifiedMergeResult",
    "LayerAlphaProfile",
    "compute_adaptive_alpha_profile",
    "compute_spectral_penalty",
    "DimensionBlendingWeights",
]

