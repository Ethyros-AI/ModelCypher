from __future__ import annotations

import math

from modelcypher.core.domain.traversal_coherence import Path, TraversalCoherence


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
