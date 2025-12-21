"""
Model Merging Package.

Provides tools for merging models using geometric alignment.
"""
from .rotational_merger import (
    RotationalModelMerger,
    MergeOptions,
    MergeAnalysisResult,
    LayerMergeMetric,
    AnchorMode,
    ModuleScope,
    MergeError,
    merge_lora_adapters,
    weighted_merge,
)

__all__ = [
    "RotationalModelMerger",
    "MergeOptions",
    "MergeAnalysisResult",
    "LayerMergeMetric",
    "AnchorMode",
    "ModuleScope",
    "MergeError",
    "merge_lora_adapters",
    "weighted_merge",
]
