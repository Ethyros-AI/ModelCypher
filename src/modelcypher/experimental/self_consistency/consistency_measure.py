# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Consistency Measurement using Representation Geometry.

The core insight from counterfactual_self_play.py:
- If a model "knows" something, its representation should be stable
- Related statements should have similar representations
- Contradictory statements should have different representations

Consistency is measured as:
- High: original and implications have similar representations
- Low: original and implications have different representations

This leverages the counterfactual sensitivity metric (effect size 1.44)
but applies it to self-generated implications rather than manually
constructed counterfactuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry._primitives.epsilon_utils import division_epsilon


@dataclass
class ConsistencyResult:
    """Result of consistency measurement."""

    # Core metrics
    implication_consistency: float  # How consistent are implications with original
    contradiction_distance: float  # How different are contradictions from original

    # Derived scores
    consistency_score: float  # Combined score (0-1)
    knowledge_confidence: float  # How confident is the model about this

    # Details
    n_implications: int
    n_contradictions: int
    representation_distances: List[float]


class ConsistencyMeasure:
    """Measure semantic consistency using representation geometry.

    The key insight: if a model's understanding is coherent, then:
    1. A statement and its implications should have SIMILAR representations
    2. A statement and its contradictions should have DIFFERENT representations

    This is exactly what we observed in counterfactual_self_play.py with
    effect size 1.44 - the model's representations encode what it "knows."

    Usage:
        measure = ConsistencyMeasure(backend)

        # Get representations
        orig_repr = get_activations("2 + 2 = 4")
        impl_reprs = [get_activations(impl) for impl in implications]
        contra_reprs = [get_activations(contra) for contra in contradictions]

        result = measure.compute(orig_repr, impl_reprs, contra_reprs)
        print(f"Consistency: {result.consistency_score:.2%}")
    """

    def __init__(self, backend: Optional["Backend"] = None):
        self._backend = backend or get_default_backend()

    def cosine_distance(self, a: "Array", b: "Array") -> float:
        """Compute cosine distance between two representations.

        Returns:
            Distance in [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite
        """
        b_ = self._backend

        # Flatten if needed
        a_flat = b_.reshape(a, (-1,))
        b_flat = b_.reshape(b, (-1,))
        b_.eval(a_flat, b_flat)

        # Normalize
        a_norm = b_.sqrt(b_.sum(a_flat * a_flat))
        b_norm = b_.sqrt(b_.sum(b_flat * b_flat))
        b_.eval(a_norm, b_norm)

        a_norm_val = float(b_.to_scalar(a_norm))
        b_norm_val = float(b_.to_scalar(b_norm))

        # Use dtype-derived epsilon for norm check
        div_eps = division_epsilon(b_, a)
        if a_norm_val < div_eps or b_norm_val < div_eps:
            return 1.0  # Orthogonal if zero

        a_unit = a_flat / a_norm_val
        b_unit = b_flat / b_norm_val

        # Cosine similarity
        similarity = b_.sum(a_unit * b_unit)
        b_.eval(similarity)
        sim_val = float(b_.to_scalar(similarity))

        # Distance = 1 - similarity
        return 1.0 - sim_val

    def compute(
        self,
        original: "Array",
        implications: List["Array"],
        contradictions: Optional[List["Array"]] = None,
    ) -> ConsistencyResult:
        """Compute consistency metrics.

        Args:
            original: Representation of the original statement
            implications: Representations of implied statements
            contradictions: Representations of contradicting statements

        Returns:
            ConsistencyResult with consistency metrics
        """
        # Measure distances from original to implications
        impl_distances = []
        for impl in implications:
            dist = self.cosine_distance(original, impl)
            impl_distances.append(dist)

        # Measure distances from original to contradictions
        contra_distances = []
        if contradictions:
            for contra in contradictions:
                dist = self.cosine_distance(original, contra)
                contra_distances.append(dist)

        # Implication consistency: lower distance = higher consistency
        # We want implications to be CLOSE to original
        avg_impl_dist = sum(impl_distances) / len(impl_distances) if impl_distances else 1.0
        implication_consistency = 1.0 - min(1.0, avg_impl_dist)

        # Contradiction distance: higher is better
        # We want contradictions to be FAR from original
        avg_contra_dist = sum(contra_distances) / len(contra_distances) if contra_distances else 0.0
        contradiction_distance = avg_contra_dist

        # Combined consistency score
        # High consistency = implications close + contradictions far
        if contradictions:
            # Ideal: impl_dist → 0, contra_dist → 1
            # Score = (1 - impl_dist) * contra_dist
            consistency_score = implication_consistency * min(1.0, contradiction_distance)
        else:
            # Just use implication consistency
            consistency_score = implication_consistency

        # Knowledge confidence: how different are contradictions from implications?
        # This is the "effect size" - separation between things that should be same vs different
        if contradictions and impl_distances and contra_distances:
            impl_mean = sum(impl_distances) / len(impl_distances)
            contra_mean = sum(contra_distances) / len(contra_distances)

            # Pooled std estimate
            all_dists = impl_distances + contra_distances
            variance = sum((d - sum(all_dists)/len(all_dists))**2 for d in all_dists) / len(all_dists)
            std = math.sqrt(variance) if variance > 0 else 1.0

            # Effect size (Cohen's d approximation)
            effect_size = abs(contra_mean - impl_mean) / std if std > 0 else 0.0

            # Normalize to 0-1 (effect size of 1.44 was observed in our experiments)
            knowledge_confidence = min(1.0, effect_size / 1.5)
        else:
            knowledge_confidence = implication_consistency

        return ConsistencyResult(
            implication_consistency=implication_consistency,
            contradiction_distance=contradiction_distance,
            consistency_score=consistency_score,
            knowledge_confidence=knowledge_confidence,
            n_implications=len(implications),
            n_contradictions=len(contradictions) if contradictions else 0,
            representation_distances=impl_distances + (contra_distances or []),
        )


__all__ = ["ConsistencyMeasure", "ConsistencyResult"]
