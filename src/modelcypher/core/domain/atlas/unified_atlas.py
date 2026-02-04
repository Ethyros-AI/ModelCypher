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

"""Unified atlas.

Provides a unified probe inventory for cross-domain triangulation in layer
mapping operations. Probes are loaded from JSON files in `data/probes/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    sqrt_scalar,
)


class AtlasSource(str, Enum):
    """Source atlas for a unified probe."""

    SEQUENCE_INVARIANT = "sequence_invariant"
    SEMANTIC_PRIME = "semantic_prime"
    COMPUTATIONAL_GATE = "computational_gate"
    EMOTION_CONCEPT = "emotion_concept"
    TEMPORAL_CONCEPT = "temporal_concept"
    SPATIAL_CONCEPT = "spatial_concept"
    SOCIAL_CONCEPT = "social_concept"
    MORAL_CONCEPT = "moral_concept"
    COMPOSITIONAL = "compositional"
    PHILOSOPHICAL_CONCEPT = "philosophical_concept"
    CONCEPTUAL_GENEALOGY = "conceptual_genealogy"
    METAPHOR_INVARIANT = "metaphor_invariant"
    CONCEPTUAL_METAPHOR = "conceptual_metaphor"
    SYNTAX_CONCEPT = "syntax_concept"
    SAFETY_ETHICS = "safety_ethics"
    PHYSICAL_EXISTENCE = "physical_existence"
    PERCEPTUAL = "perceptual"
    NUMERIC = "numeric"
    COMMON_OBJECT = "common_object"
    ACTION_VERB = "action_verb"
    DOMAIN_SPECIFIC = "domain_specific"
    ABSTRACT_RELATION = "abstract_relation"
    PRONOUN_PERSPECTIVE = "pronoun_perspective"
    PRIME_NUMBER = "prime_number"


# Import AtlasDomain from the canonical location
from modelcypher.core.domain.domains import AtlasDomain


@dataclass(frozen=True)
class AtlasProbe:
    """
    Unified probe from any atlas source.

    Normalizes probes from all domains into a common format for cross-domain
    triangulation.
    """

    id: str  # Unique ID (prefixed with source)
    source: AtlasSource  # Which atlas this came from
    domain: AtlasDomain  # Unified triangulation domain
    name: str  # Human-readable name
    description: str  # Brief description
    cross_domain_weight: float  # Weight for cross-domain scoring (0.0-2.0)

    # Original category (for filtering)
    category_name: str  # Original category name

    # Optional metadata
    support_texts: tuple[str, ...] = ()  # Example texts for embedding

    @property
    def probe_id(self) -> str:
        """Full probe ID including source prefix."""
        return f"{self.source.value}:{self.id}"


_DEFAULT_WEIGHTS: dict[AtlasSource, float] = {source: 1.0 for source in AtlasSource}


class UnifiedAtlasInventory:
    """
    Unified inventory of all atlas probes.

    Loads probes from JSON files in data/probes/. This provides a clean
    separation between probe definitions (JSON) and probe usage (Python).
    """

    _cached_probes: list[AtlasProbe] | None = None

    @classmethod
    def all_probes(cls) -> list[AtlasProbe]:
        """Get all probes from JSON data files."""
        if cls._cached_probes is not None:
            return list(cls._cached_probes)

        from modelcypher.core.domain.atlas.probe_loader import load_all_probes

        cls._cached_probes = load_all_probes()
        return list(cls._cached_probes)

    @classmethod
    def probes_by_source(cls, sources: set[AtlasSource]) -> list[AtlasProbe]:
        """Get probes from specific atlas sources."""
        return [p for p in cls.all_probes() if p.source in sources]

    @classmethod
    def probes_by_domain(cls, domains: set[AtlasDomain]) -> list[AtlasProbe]:
        """Get probes from specific triangulation domains."""
        return [p for p in cls.all_probes() if p.domain in domains]

    @classmethod
    def probe_count(cls) -> dict[AtlasSource, int]:
        """Get probe counts by source."""
        counts: dict[AtlasSource, int] = {}
        for probe in cls.all_probes():
            counts[probe.source] = counts.get(probe.source, 0) + 1
        return counts

    @classmethod
    def total_probe_count(cls) -> int:
        """Get total number of probes."""
        return len(cls.all_probes())

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the probe cache to force reload from JSON."""
        cls._cached_probes = None


# Convenience constants
ALL_ATLAS_SOURCES = frozenset(AtlasSource)
MATHEMATICAL_DOMAINS = frozenset([AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL])
LINGUISTIC_DOMAINS = frozenset([AtlasDomain.LINGUISTIC, AtlasDomain.MENTAL])
COMPUTATIONAL_DOMAINS = frozenset([AtlasDomain.COMPUTATIONAL, AtlasDomain.STRUCTURAL])
AFFECTIVE_DOMAINS = frozenset([AtlasDomain.AFFECTIVE, AtlasDomain.RELATIONAL])
SPATIOTEMPORAL_DOMAINS = frozenset([AtlasDomain.TEMPORAL, AtlasDomain.SPATIAL])
MORAL_DOMAINS = frozenset([AtlasDomain.MORAL])
PHILOSOPHICAL_DOMAINS = frozenset([AtlasDomain.PHILOSOPHICAL, AtlasDomain.LOGICAL])
SAFETY_DOMAINS = frozenset([AtlasDomain.SAFETY])

# Default sources for layer mapping (all sources enabled)
DEFAULT_ATLAS_SOURCES = frozenset(AtlasSource)


@dataclass(frozen=True)
class MultiAtlasTriangulationScore:
    """
    Cross-atlas triangulation score for a layer.

    Higher scores indicate concept detection across multiple atlas sources
    and domains, which provides stronger anchoring for layer mapping.
    """

    layer_index: int
    sources_detected: set[AtlasSource]
    domains_detected: set[AtlasDomain]
    source_multiplier: float  # Boost for detection across sources
    domain_multiplier: float  # Boost for detection across domains
    combined_multiplier: float  # Final combined multiplier


class MultiAtlasTriangulationScorer:
    """
    Scorer for cross-atlas triangulation.

    Provides confidence boosts when concepts are detected across multiple
    atlas sources and triangulation domains.
    """

    @staticmethod
    def compute_score(
        activations: dict[AtlasProbe, float],
    ) -> MultiAtlasTriangulationScore:
        """
        Compute triangulation score from probe activations.

        Args:
            activations: Map of probes to their activation values

        Returns:
            MultiAtlasTriangulationScore with multipliers
        """
        sources_detected: set[AtlasSource] = set()
        domains_detected: set[AtlasDomain] = set()

        activation_values = [abs(value) for value in activations.values()]
        if activation_values:
            backend = get_default_backend()
            eps = division_epsilon(backend, backend.array(activation_values))
            if len(activation_values) < 3:
                threshold = eps
            else:
                threshold = find_magnitude_gap_threshold(sorted(activation_values), eps=eps)
                max_value = max(activation_values)
                if threshold >= max_value - eps:
                    threshold = eps
            for probe, activation in activations.items():
                if abs(activation) > threshold:
                    sources_detected.add(probe.source)
                    domains_detected.add(probe.domain)

        source_count = len(sources_detected)
        domain_count = len(domains_detected)
        source_multiplier = max(1.0, float(source_count))
        domain_multiplier = max(1.0, float(domain_count))
        _b = get_default_backend()
        combined_multiplier = sqrt_scalar(source_multiplier * domain_multiplier, _b)

        return MultiAtlasTriangulationScore(
            layer_index=-1,  # Set by caller
            sources_detected=sources_detected,
            domains_detected=domains_detected,
            source_multiplier=source_multiplier,
            domain_multiplier=domain_multiplier,
            combined_multiplier=combined_multiplier,
        )


def get_probe_ids(sources: set[AtlasSource] | None = None) -> list[str]:
    """
    Get all probe IDs for specified sources.

    Args:
        sources: Set of atlas sources to include, or None for all

    Returns:
        List of probe IDs in format "source:category_id"
    """
    if sources is None:
        sources = set(ALL_ATLAS_SOURCES)

    probes = UnifiedAtlasInventory.probes_by_source(sources)
    return [probe.probe_id for probe in probes]
