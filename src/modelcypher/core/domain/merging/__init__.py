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
from .unified_manifold_merger import (
    UnifiedManifoldMerger,
    UnifiedMergeConfig,
    UnifiedMergeResult,
    LayerAlphaProfile,
    compute_adaptive_alpha_profile,
    compute_spectral_penalty,
    DimensionBlendingWeights,
    # New enums (Phase 1 parity)
    BlendMode,
    LayerMappingStrategy,
    MLPInternalIntersectionMode,
    IntersectionSimilarityMode,
    ModuleScope as UnifiedModuleScope,  # Alias to avoid conflict with rotational_merger
    SequenceFamily,
    # Nested config classes
    InvariantLayerMapperConfig,
    TangentSpaceConfig,
    SharedSubspaceConfig,
    TransportGuidedConfig,
    AffineStitchingConfig,
    VerbNounConfig,
    IntersectionEnsembleWeights,
    AnchorCategoryWeights,
    ModuleBlendPolicy,
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
    # Unified Manifold Merger
    "UnifiedManifoldMerger",
    "UnifiedMergeConfig",
    "UnifiedMergeResult",
    "LayerAlphaProfile",
    "compute_adaptive_alpha_profile",
    "compute_spectral_penalty",
    "DimensionBlendingWeights",
    # Unified Merger Enums (Phase 1 parity)
    "BlendMode",
    "LayerMappingStrategy",
    "MLPInternalIntersectionMode",
    "IntersectionSimilarityMode",
    "UnifiedModuleScope",
    "SequenceFamily",
    # Nested Config Classes
    "InvariantLayerMapperConfig",
    "TangentSpaceConfig",
    "SharedSubspaceConfig",
    "TransportGuidedConfig",
    "AffineStitchingConfig",
    "VerbNounConfig",
    "IntersectionEnsembleWeights",
    "AnchorCategoryWeights",
    "ModuleBlendPolicy",
]

