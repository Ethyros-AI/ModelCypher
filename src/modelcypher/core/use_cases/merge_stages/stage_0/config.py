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

    Note on thresholds:
        All thresholds default to None and are derived from data at runtime:

        - similarity_threshold: If None, derived from distribution of embedding
          cosine similarities. Uses spectral gap to find natural boundary.

        - confidence_threshold: If None, derived from alignment confidence
          distribution using spectral gap detection.

        - blend_alpha: If None, computed from relative alignment quality
          (CKA scores) between source and target. Higher alignment quality
          means higher weight for that model's embeddings.

        - min_alignment_score: If None, set to machine_epsilon - any
          measurable alignment is acceptable.

        - min_coverage: If None, set to 0.0 - no arbitrary coverage floor.
    """

    # Projection strategy: procrustes, pca, optimal_transport, cca
    projection_strategy: str = "procrustes"

    # Alignment thresholds - None means derive from data
    similarity_threshold: float | None = None
    confidence_threshold: float | None = None

    # Embedding blending - None means compute from alignment quality
    blend_alpha: float | None = None
    preserve_special_tokens: bool = True

    # Quality thresholds - None means no arbitrary floor
    min_alignment_score: float | None = None
    min_coverage: float | None = None

    # Advanced
    use_embedding_similarity: bool = True
    anchor_count: int = 1000
    max_similarity_pairs: int = 5_000_000
    max_unmapped_similarity: int = 5000
    max_prefix_length: int = 8
    max_prefix_matches: int = 3
    similarity_batch_size: int = 128

    # Phase-lock alignment tuning
    alignment_iterations: int = 8
    alignment_solver_iterations: int = 5000
    alignment_solver_rounds: int = 1
    alignment_tolerance: float = 1e-12
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
