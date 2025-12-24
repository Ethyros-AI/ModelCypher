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

import math

from modelcypher.core.domain.geometry.traversal_coherence import (
    Path,
    TraversalCoherence,
    standard_computational_paths,
)


def test_transition_inner_product_identity():
    gram = [1.0, 0.0, 0.0, 1.0]
    value = TraversalCoherence.transition_inner_product(gram, n=2, a=0, b=1, c=0, d=1)
    assert value == 2.0
    norm_sq = TraversalCoherence.transition_norm_squared(gram, n=2, a=0, b=1)
    assert norm_sq == 2.0
    normalized = TraversalCoherence.normalized_transition_inner_product(gram, n=2, a=0, b=1, c=0, d=1)
    assert math.isfinite(normalized)
    assert abs(normalized - 1.0) < 1e-6


def test_compare_identical_transition_gram():
    gram = [
        1.0, 0.1, 0.2,
        0.1, 1.0, 0.3,
        0.2, 0.3, 1.0,
    ]
    paths = [Path(anchor_ids=["A", "B", "C"])]
    result = TraversalCoherence.compare(paths, gram, gram, anchor_ids=["A", "B", "C"])
    assert result is not None
    assert abs(result.transition_gram_correlation - 1.0) < 1e-6


def test_path_transition_count():
    """Path correctly counts transitions."""
    path = Path(anchor_ids=["A", "B", "C", "D"])
    assert path.transition_count == 3

    empty_path = Path(anchor_ids=[])
    assert empty_path.transition_count == 0

    single_path = Path(anchor_ids=["A"])
    assert single_path.transition_count == 0


def test_transition_inner_product_invalid_indices():
    """Invalid indices return NaN."""
    gram = [1.0, 0.0, 0.0, 1.0]
    value = TraversalCoherence.transition_inner_product(gram, n=2, a=-1, b=0, c=0, d=1)
    assert math.isnan(value)


def test_transition_gram_empty_paths():
    """Empty paths return empty transition gram."""
    gram = [1.0, 0.0, 0.0, 1.0]
    trans_gram, count = TraversalCoherence.transition_gram([], gram, ["A", "B"])
    assert trans_gram == []
    assert count == 0


def test_standard_computational_paths_exist():
    """Standard computational paths are defined."""
    assert len(standard_computational_paths) > 0
    for path in standard_computational_paths:
        assert isinstance(path, Path)
        assert len(path.anchor_ids) >= 2


def test_compare_different_grams():
    """Different Gram matrices produce correlation < 1."""
    gram_a = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    gram_b = [
        1.0, 0.1, 0.8,
        0.1, 1.0, 0.2,
        0.8, 0.2, 1.0,
    ]
    # Multiple paths to get enough asymmetric transitions for meaningful correlation.
    # A single path with 3 anchors gives symmetric off-diagonal values (zero variance).
    paths = [
        Path(anchor_ids=["A", "B", "C"]),
        Path(anchor_ids=["C", "A", "B"]),
        Path(anchor_ids=["B", "C", "A"]),
    ]
    result = TraversalCoherence.compare(paths, gram_a, gram_b, anchor_ids=["A", "B", "C"])
    assert result is not None
    assert result.transition_gram_correlation < 1.0


def test_compare_insufficient_transitions():
    """Insufficient transitions return None."""
    gram = [1.0, 0.5, 0.5, 1.0]
    # Path with only 2 anchors = 1 transition, needs at least 2 for correlation
    paths = [Path(anchor_ids=["A", "B"])]
    result = TraversalCoherence.compare(paths, gram, gram, anchor_ids=["A", "B"])
    # 1 transition means 1x1 matrix, which can't compute off-diagonal correlation
    assert result is None
