# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Representation Consistency: Measuring stability across related inputs.

Analyzes whether related inputs produce similar representations and
unrelated/contradictory inputs produce different representations.

Computes:
1. Implication consistency: Similarity between original and related representations
2. Contradiction distance: Distance between original and contradictory representations
3. Separation: How well the model distinguishes related from unrelated inputs

All outputs are raw measurements. No interpretation of what high/low
values "mean" about the model - that's for the researcher to determine.

Methods:
- Cosine distance: 1 - cosine_similarity (range [0, 2])
- Consistency score: Combined metric from implication/contradiction distances
- Separation score: Effect size between related and unrelated distances
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


@dataclass
class RepresentationConsistencyResult:
    """Result of representation consistency measurement.

    All values are raw measurements without interpretation.
    """

    implication_consistency: float
    """Mean similarity between original and related representations. Range [0, 1]."""

    contradiction_distance: float
    """Mean distance between original and contradictory representations. Range [0, 2]."""

    consistency_score: float
    """Combined score: high when related are similar AND contradictions are distant."""

    separation_score: float
    """Effect size between related and unrelated distances. Higher = better separation."""

    n_implications: int
    """Number of related representations measured."""

    n_contradictions: int
    """Number of contradictory representations measured."""

    representation_distances: List[float]
    """All individual distance measurements."""


class RepresentationConsistencyAnalyzer:
    """Measure representation consistency across related inputs.

    Given a set of related inputs (implications) and contradictory inputs,
    measures how consistently the model represents them relative to an
    original input.

    This is useful for:
    - Detecting representation stability for semantically similar inputs
    - Measuring separation between related and unrelated concepts
    - Evaluating representation quality for downstream tasks

    Usage:
        analyzer = RepresentationConsistencyAnalyzer(backend)

        # Get representations for related statements
        orig = get_activations("The sky is blue")
        related = [get_activations(r) for r in ["The sky has a blue color", ...]]
        unrelated = [get_activations(u) for u in ["The sky is green", ...]]

        result = analyzer.compute(orig, related, unrelated)
        print(f"Consistency: {result.consistency_score:.2f}")
    """

    def __init__(self, backend: Optional["Backend"] = None):
        """Initialize with compute backend.

        Args:
            backend: Backend for array operations (defaults to system default)
        """
        self._backend = backend or get_default_backend()

    def cosine_distance(self, a: "Array", b: "Array") -> float:
        """Compute cosine distance between two representations.

        Args:
            a: First representation vector
            b: Second representation vector

        Returns:
            Distance in [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite
        """
        backend = self._backend

        # Flatten if needed
        a_flat = backend.reshape(a, (-1,))
        b_flat = backend.reshape(b, (-1,))
        backend.eval(a_flat, b_flat)

        # Compute norms
        a_norm = backend.sqrt(backend.sum(a_flat * a_flat))
        b_norm = backend.sqrt(backend.sum(b_flat * b_flat))
        backend.eval(a_norm, b_norm)

        a_norm_val = float(backend.to_scalar(a_norm))
        b_norm_val = float(backend.to_scalar(b_norm))

        # Use dtype-derived epsilon for norm check
        div_eps = division_epsilon(backend, a)
        if a_norm_val < div_eps or b_norm_val < div_eps:
            return 1.0  # Orthogonal if zero vector

        a_unit = a_flat / a_norm_val
        b_unit = b_flat / b_norm_val

        # Cosine similarity
        similarity = backend.sum(a_unit * b_unit)
        backend.eval(similarity)
        sim_val = float(backend.to_scalar(similarity))

        # Distance = 1 - similarity
        return 1.0 - sim_val

    def compute(
        self,
        original: "Array",
        related: List["Array"],
        contradictory: Optional[List["Array"]] = None,
    ) -> ConsistencyResult:
        """Compute consistency metrics.

        Args:
            original: Representation of the original input
            related: Representations of semantically related inputs
            contradictory: Representations of contradictory/unrelated inputs

        Returns:
            ConsistencyResult with all measurements
        """
        # Measure distances from original to related inputs
        related_distances = []
        for rel in related:
            dist = self.cosine_distance(original, rel)
            related_distances.append(dist)

        # Measure distances from original to contradictory inputs
        contra_distances = []
        if contradictory:
            for contra in contradictory:
                dist = self.cosine_distance(original, contra)
                contra_distances.append(dist)

        # Implication consistency: lower distance = higher consistency
        avg_related_dist = (
            sum(related_distances) / len(related_distances)
            if related_distances
            else 1.0
        )
        implication_consistency = 1.0 - min(1.0, avg_related_dist)

        # Contradiction distance: higher is better separation
        avg_contra_dist = (
            sum(contra_distances) / len(contra_distances)
            if contra_distances
            else 0.0
        )
        contradiction_distance = avg_contra_dist

        # Combined consistency score
        if contradictory:
            # High consistency = related close + contradictions far
            consistency_score = implication_consistency * min(1.0, contradiction_distance)
        else:
            consistency_score = implication_consistency

        # Separation score: effect size between related and unrelated distances
        if contradictory and related_distances and contra_distances:
            related_mean = sum(related_distances) / len(related_distances)
            contra_mean = sum(contra_distances) / len(contra_distances)

            # Pooled variance estimate
            all_dists = related_distances + contra_distances
            overall_mean = sum(all_dists) / len(all_dists)
            variance = sum((d - overall_mean) ** 2 for d in all_dists) / len(all_dists)
            std = math.sqrt(variance) if variance > 0 else 1.0

            # Effect size (Cohen's d approximation)
            effect_size = abs(contra_mean - related_mean) / std if std > 0 else 0.0

            # Cap at reasonable range
            separation_score = min(2.0, effect_size)
        else:
            separation_score = 0.0

        return ConsistencyResult(
            implication_consistency=implication_consistency,
            contradiction_distance=contradiction_distance,
            consistency_score=consistency_score,
            separation_score=separation_score,
            n_implications=len(related),
            n_contradictions=len(contradictory) if contradictory else 0,
            representation_distances=related_distances + (contra_distances or []),
        )


__all__ = [
    "ConsistencyResult",
    "RepresentationConsistencyAnalyzer",
]
