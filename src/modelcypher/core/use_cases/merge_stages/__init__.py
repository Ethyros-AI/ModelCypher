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
    VocabularyConfig,
    VocabularyResult,
    stage_vocabulary_align,
)
from .stage_1_probe import (
    ProbeConfig,
    ProbeResult,
    collect_layer_activations_mlx,
    stage_probe,
)
from .stage_2_permute import (
    PermuteConfig,
    PermuteResult,
    infer_hidden_dim,
    stage_permute,
)
from .stage_3_5_rotate_blend import (
    RotateBlendConfig,
    RotateBlendResult,
    stage_rotate_blend_propagate,
)
from .stage_6_validate import (
    ValidateConfig,
    ValidateResult,
    stage_validate,
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
