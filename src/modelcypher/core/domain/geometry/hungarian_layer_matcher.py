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

"""Hungarian algorithm-based layer matching using CKA similarity.

This module provides closed-form optimal layer matching between source and
target models. Instead of assuming proportional depth = semantic alignment
(a heuristic), we compute CKA for all (source_layer, target_layer) pairs
and use the Hungarian algorithm to find the optimal 1-to-1 assignment.

The Hungarian algorithm is O(N³) and closed-form - no iteration, no heuristics.
It guarantees the maximum total CKA across all matched layer pairs.

References:
    - Kuhn, H. W. (1955). "The Hungarian Method for the Assignment Problem."
    - Kornblith et al. (2019). "Similarity of Neural Network Representations."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.hungarian import hungarian_assignment
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerMatchingResult:
    """Result of Hungarian layer matching."""

    # Mapping: target_layer_idx -> source_layer_idx
    layer_mapping: dict[int, int]

    # CKA matrix [n_source, n_target] where cka_matrix[i][j] = CKA(source_i, target_j)
    cka_matrix: dict[tuple[int, int], float]

    # Per-pair CKA for the matched layers
    matched_cka_scores: dict[int, float]

    # Total CKA (sum of matched pairs)
    total_cka: float

    # Mean CKA across matched pairs
    mean_cka: float

    # Source and target layer indices used
    source_layers: list[int]
    target_layers: list[int]


def compute_all_pairs_cka(
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    backend: "Backend",
) -> tuple[dict[tuple[int, int], float], list[int], list[int]]:
    """Compute CKA for all (source_layer, target_layer) pairs.

    Args:
        source_layer_activations: Source activations by layer index.
        target_layer_activations: Target activations by layer index.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (cka_dict, source_layers, target_layers) where:
        - cka_dict maps (source_idx, target_idx) -> CKA score
        - source_layers is sorted list of source layer indices
        - target_layers is sorted list of target layer indices
    """
    b = backend

    source_layers = sorted(source_layer_activations.keys())
    target_layers = sorted(target_layer_activations.keys())

    n_source = len(source_layers)
    n_target = len(target_layers)

    logger.info(
        "HUNGARIAN LAYER MATCH: Computing CKA for %d x %d = %d layer pairs",
        n_source,
        n_target,
        n_source * n_target,
    )

    cka_dict: dict[tuple[int, int], float] = {}

    for src_layer in source_layers:
        src_acts = source_layer_activations[src_layer]
        if isinstance(src_acts, list):
            src_acts = b.stack(src_acts, axis=0)
        src_acts = _promote_precision(src_acts, b)
        b.eval(src_acts)

        for tgt_layer in target_layers:
            tgt_acts = target_layer_activations[tgt_layer]
            if isinstance(tgt_acts, list):
                tgt_acts = b.stack(tgt_acts, axis=0)
            tgt_acts = _promote_precision(tgt_acts, b)
            b.eval(tgt_acts)

            # CKA requires same number of samples
            n_src = int(b.shape(src_acts)[0])
            n_tgt = int(b.shape(tgt_acts)[0])
            n_samples = min(n_src, n_tgt)

            if n_samples < 2:
                cka_dict[(src_layer, tgt_layer)] = 0.0
                continue

            src_subset = src_acts[:n_samples]
            tgt_subset = tgt_acts[:n_samples]
            b.eval(src_subset, tgt_subset)

            cka_result = compute_cka(src_subset, tgt_subset, backend=b)
            cka_score = cka_result.cka if cka_result.is_valid else 0.0
            cka_dict[(src_layer, tgt_layer)] = cka_score

    return cka_dict, source_layers, target_layers


def hungarian_layer_matching(
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    backend: "Backend | None" = None,
) -> LayerMatchingResult:
    """Find optimal layer matching using Hungarian algorithm on CKA matrix.

    This is a closed-form solution to the layer matching problem. Instead of
    assuming proportional depth correlates with semantic alignment, we compute
    the actual CKA similarity between all layer pairs and find the assignment
    that maximizes total CKA.

    The algorithm handles different layer counts:
    - If n_source == n_target: 1-to-1 matching
    - If n_source < n_target: Some target layers unmatched (padded with dummy)
    - If n_source > n_target: Some source layers unmatched (padded with dummy)

    Args:
        source_layer_activations: Source activations by layer index.
        target_layer_activations: Target activations by layer index.
        backend: Backend for tensor operations.

    Returns:
        LayerMatchingResult with optimal mapping and diagnostics.
    """
    b = backend or get_default_backend()

    # Compute all-pairs CKA
    cka_dict, source_layers, target_layers = compute_all_pairs_cka(
        source_layer_activations,
        target_layer_activations,
        b,
    )

    n_source = len(source_layers)
    n_target = len(target_layers)

    if n_source == 0 or n_target == 0:
        logger.warning("HUNGARIAN LAYER MATCH: Empty layer activations")
        return LayerMatchingResult(
            layer_mapping={},
            cka_matrix=cka_dict,
            matched_cka_scores={},
            total_cka=0.0,
            mean_cka=0.0,
            source_layers=source_layers,
            target_layers=target_layers,
        )

    # Build cost matrix for Hungarian algorithm
    # Hungarian minimizes cost, so we use (1 - CKA) as cost
    # Pad to square matrix if needed
    n = max(n_source, n_target)

    cost_list: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i < n_source and j < n_target:
                src_layer = source_layers[i]
                tgt_layer = target_layers[j]
                cka = cka_dict.get((src_layer, tgt_layer), 0.0)
                # Cost = 1 - CKA (minimize cost = maximize CKA)
                cost = 1.0 - cka
            else:
                # Dummy entries for padding (high cost)
                cost = 1.0
            row.append(cost)
        cost_list.append(row)

    cost_matrix = b.array(cost_list)
    b.eval(cost_matrix)

    # Run Hungarian algorithm
    assignment = hungarian_assignment(cost_matrix, b)
    b.eval(assignment)
    assignment_list = b.tolist(assignment)

    # Extract layer mapping (target -> source)
    layer_mapping: dict[int, int] = {}
    matched_cka_scores: dict[int, float] = {}
    total_cka = 0.0

    for src_idx, tgt_idx in enumerate(assignment_list):
        if src_idx >= n_source or tgt_idx >= n_target:
            # Skip dummy assignments
            continue

        src_layer = source_layers[src_idx]
        tgt_layer = target_layers[tgt_idx]

        # Store as target -> source (consistent with existing code)
        layer_mapping[tgt_layer] = src_layer

        cka = cka_dict.get((src_layer, tgt_layer), 0.0)
        matched_cka_scores[tgt_layer] = cka
        total_cka += cka

    n_matched = len(layer_mapping)
    mean_cka = total_cka / n_matched if n_matched > 0 else 0.0

    # Log the matching
    logger.info(
        "HUNGARIAN LAYER MATCH: Optimal matching found, mean_cka=%.4f, total_cka=%.4f",
        mean_cka,
        total_cka,
    )

    for tgt_layer in sorted(layer_mapping.keys()):
        src_layer = layer_mapping[tgt_layer]
        cka = matched_cka_scores[tgt_layer]
        logger.info(
            "  target[%d] <- source[%d] (CKA=%.4f)",
            tgt_layer,
            src_layer,
            cka,
        )

    return LayerMatchingResult(
        layer_mapping=layer_mapping,
        cka_matrix=cka_dict,
        matched_cka_scores=matched_cka_scores,
        total_cka=total_cka,
        mean_cka=mean_cka,
        source_layers=source_layers,
        target_layers=target_layers,
    )


__all__ = [
    "LayerMatchingResult",
    "compute_all_pairs_cka",
    "hungarian_layer_matching",
]
