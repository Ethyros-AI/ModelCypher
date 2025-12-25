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

# Re-export from merge_engine (the canonical geometric merge)
from modelcypher.core.use_cases.merge_engine import (
    AnchorMode,
    LayerMergeMetric,
    MergeAnalysisResult,
    ModuleScope,
    RotationalMerger,
)
from modelcypher.core.use_cases.merge_engine import (
    RotationalMergeOptions as MergeOptions,
)

# Platform detection (for backend selection, not merger selection)
from ._platform import (
    get_lora_adapter_merger_class,
    get_merging_platform,
)
from .entropy_merge_validator import (
    EntropyMergeValidator,
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    ModelEntropyProfile,
)
from .lora_adapter_merger import (
    LoRAAdapterMerger,
)
from .lora_adapter_merger import (
    MergeReport as LoRAMergeReport,
)

__all__ = [
    # Entropy Merge Validator
    "EntropyMergeValidator",
    "LayerEntropyProfile",
    "ModelEntropyProfile",
    "LayerMergeValidation",
    "MergeEntropyValidation",
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
