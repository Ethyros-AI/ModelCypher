"""
Merge pipeline stages.

Each stage is a standalone module that can be imported and tested independently.
The UnifiedGeometricMerger orchestrates these stages in sequence.

Stage 0: VOCABULARY - Cross-vocabulary embedding alignment
Stage 1: PROBE - Build intersection map from probe responses
Stage 2: PERMUTE - Permutation alignment for MLP neurons
Stage 3-5: ROTATE + BLEND + PROPAGATE - Geometric merge loop
Stage 6: VALIDATE - Safety checks (numerical + content)
"""

from .stage_0_vocabulary import (
    stage_vocabulary_align,
    VocabularyConfig,
    VocabularyResult,
)
from .stage_1_probe import (
    stage_probe,
    ProbeConfig,
    ProbeResult,
    collect_layer_activations_mlx,
)
from .stage_2_permute import (
    stage_permute,
    PermuteConfig,
    PermuteResult,
    infer_hidden_dim,
)
from .stage_3_5_rotate_blend import (
    stage_rotate_blend_propagate,
    RotateBlendConfig,
    RotateBlendResult,
)
from .stage_6_validate import (
    stage_validate,
    ValidateConfig,
    ValidateResult,
)

__all__ = [
    # Stage 0
    "stage_vocabulary_align",
    "VocabularyConfig",
    "VocabularyResult",
    # Stage 1
    "stage_probe",
    "ProbeConfig",
    "ProbeResult",
    "collect_layer_activations_mlx",
    # Stage 2
    "stage_permute",
    "PermuteConfig",
    "PermuteResult",
    "infer_hidden_dim",
    # Stage 3-5
    "stage_rotate_blend_propagate",
    "RotateBlendConfig",
    "RotateBlendResult",
    # Stage 6
    "stage_validate",
    "ValidateConfig",
    "ValidateResult",
]
