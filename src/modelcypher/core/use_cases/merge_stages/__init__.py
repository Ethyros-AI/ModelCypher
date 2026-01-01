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

Pipeline: VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE

Stage 0: VOCABULARY - Cross-vocabulary embedding alignment
Stage 1: PROBE - Build intersection map from probe responses
Stage 2: PERMUTE - Git Re-Basin permutation alignment for MLP neurons (same-arch)
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting
Stage 4: VALIDATE - Safety checks (numerical + content)

REMOVED (proven broken):
- ROTATE/BLEND/PROPAGATE: Alpha-blending produces gibberish even for same-arch models.
  No mathematical guarantee of boundary preservation.

References:
- Git Re-Basin: Ainsworth et al. (2023) arXiv:2209.04836
- AlphaEdit (null-space transplant): Fang et al. (2025) ICLR Outstanding Paper
"""

from .stage_0_vocabulary import (
    VocabularyConfig,
    VocabularyResult,
    stage_vocabulary_align,
)
from .stage_1_probe import (
    ProbeResult,
    collect_layer_activations_mlx,
    stage_probe,
)
from .stage_2_permute import (
    PermuteResult,
    infer_hidden_dim,
    stage_permute,
)
# NOTE: ProbeConfig and PermuteConfig were REMOVED.
# Probe always uses precise mode with all probes.
# Permute always runs (no enable_permutation toggle).
from .stage_3_transplant import (
    TransplantStageConfig,
    TransplantStageResult,
    stage_transplant,
)
from .stage_6_validate import (
    ValidateConfig,
    ValidateResult,
    stage_validate,
)

__all__ = [
    # Stage 0: Vocabulary
    "stage_vocabulary_align",
    "VocabularyConfig",
    "VocabularyResult",
    # Stage 1: Probe (ProbeConfig REMOVED - always precise mode, all probes)
    "stage_probe",
    "ProbeResult",
    "collect_layer_activations_mlx",
    # Stage 2: Permute (PermuteConfig REMOVED - always runs)
    "stage_permute",
    "PermuteResult",
    "infer_hidden_dim",
    # Stage 3: Transplant (simplified - only core_domains and graft_mask)
    "stage_transplant",
    "TransplantStageConfig",
    "TransplantStageResult",
    # Stage 4: Validate
    "stage_validate",
    "ValidateConfig",
    "ValidateResult",
]
