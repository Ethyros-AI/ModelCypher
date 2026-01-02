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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VocabularyConfig:
    """Configuration for Stage 0 vocabulary alignment.

    Uses null space addition for merging - no blending.
    """

    # Projection strategy: procrustes, pca, optimal_transport, cca
    projection_strategy: str = "procrustes"

    # Embedding merge options
    preserve_special_tokens: bool = True

    # Advanced
    anchor_count: int = 1000
    similarity_batch_size: int = 128

    # Phase-lock alignment tuning
    alignment_iterations: int = 8
    alignment_solver_iterations: int = 5000
    alignment_solver_rounds: int = 1
    # Tolerance is overridden by dtype's machine_epsilon when smaller.
    # Effective tolerance = max(alignment_tolerance, machine_epsilon).
    alignment_tolerance: float = 0.0  # 0.0 = use dtype-derived threshold only
    phase_lock_max_iterations: int = 0
    use_all_support_texts: bool = True
    use_byte_anchors_for_atlas: bool = True
    balance_anchor_weights: bool = True
    use_coverage_anchor_selection: bool = True
    coverage_k_neighbors: int | None = None
    coverage_candidate_multiplier: int = 3
    strict_token_alignment: bool = True


@dataclass
class VocabularyResult:
    """Result of Stage 0 vocabulary alignment."""

    modified_weights: dict[str, "object"]
    metrics: dict[str, Any]
    was_aligned: bool
    alignment_map: Any | None = None
