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

Pipeline: VOCAB → PROBE → DENSITY → PERMUTE → TRANSPLANT → VALIDATE

Stage 0: VOCABULARY - Cross-vocabulary embedding alignment
Stage 1: PROBE - Build intersection map from probe responses
Stage 2a: DENSITY - Knowledge density profiling for graft mask
Stage 2b: PERMUTE - Git Re-Basin permutation alignment for MLP neurons (same-arch)
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting
Stage 6: VALIDATE - Safety checks (numerical + content)

REMOVED (proven broken):
- ROTATE/BLEND/PROPAGATE: Alpha-blending produces gibberish even for same-arch models.
  No mathematical guarantee of boundary preservation.

References:
- Git Re-Basin: Ainsworth et al. (2023) arXiv:2209.04836
- AlphaEdit (null-space transplant): Fang et al. (2025) ICLR Outstanding Paper
"""

from .vocabulary import (
    VocabularyResult,
    stage_vocabulary_align,
)
# NOTE: VocabularyConfig is INTERNAL ONLY. All defaults are optimal.
# Users should not configure vocabulary alignment - it just works.
from .vocabulary import VocabularyConfig as _VocabularyConfig
from .probe import (
    ProbeResult,
    collect_layer_activations_mlx,
    stage_probe,
)
from .density import (
    DensityStageResult,
    stage_density,
)
from .permute import (
    PermuteResult,
    infer_hidden_dim,
    stage_permute,
)
# NOTE: ProbeConfig and PermuteConfig were REMOVED.
# Probe always uses precise mode with all probes.
# Permute always runs (no enable_permutation toggle).
from .transplant import (
    TransplantStageConfig,
    TransplantStageResult,
    stage_transplant,
)
from .validate import (
    ValidateResult,
    stage_validate,
)
# NOTE: ValidateConfig was REMOVED. Validation always runs all checks.
# entropy_phase is passed directly to stage_validate (input data, not config).

__all__ = [
    # Stage 0: Vocabulary (VocabularyConfig INTERNAL ONLY - not exported)
    "stage_vocabulary_align",
    "VocabularyResult",
    # Stage 1: Probe (ProbeConfig REMOVED - always precise mode, all probes)
    "stage_probe",
    "ProbeResult",
    "collect_layer_activations_mlx",
    # Stage 2a: Density
    "stage_density",
    "DensityStageResult",
    # Stage 2b: Permute (PermuteConfig REMOVED - always runs)
    "stage_permute",
    "PermuteResult",
    "infer_hidden_dim",
    # Stage 3: Transplant (simplified - only core_domains and graft_mask)
    "stage_transplant",
    "TransplantStageConfig",
    "TransplantStageResult",
    # Stage 6: Validate (ValidateConfig REMOVED - always runs all checks)
    "stage_validate",
    "ValidateResult",
]
