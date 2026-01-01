"""Backward-compatible unified merge package exports."""

from modelcypher.core.use_cases.merge import (
    CrossArchitectureInfo,
    LayerMergeState,
    UnifiedGeometricMerger,
    UnifiedMergeConfig,
    UnifiedMergeResult,
    unified_merge,
)

__all__ = [
    "CrossArchitectureInfo",
    "LayerMergeState",
    "UnifiedGeometricMerger",
    "UnifiedMergeConfig",
    "UnifiedMergeResult",
    "unified_merge",
]
