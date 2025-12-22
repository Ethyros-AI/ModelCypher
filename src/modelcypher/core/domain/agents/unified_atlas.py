"""
Unified Atlas.

Combines all atlas sources (sequence invariants, semantic primes, computational gates,
emotion concepts) into a single unified probe system for cross-domain triangulation
in layer mapping operations.

Total probe count: 237
- Sequence Invariants: 68 probes (10 families)
- Semantic Primes: 65 probes (17 categories)
- Computational Gates: 72 probes (14 categories)
- Emotion Concepts: 32 probes (8 categories + 8 dyads)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariant,
    SequenceInvariantInventory,
    ExpressionDomain,
    DEFAULT_FAMILIES,
)
from modelcypher.core.domain.agents.semantic_prime_atlas import (
    SemanticPrime,
    SemanticPrimeCategory,
    SemanticPrimeInventory,
)
from modelcypher.core.domain.agents.computational_gate_atlas import (
    ComputationalGate,
    ComputationalGateCategory,
    ComputationalGateInventory,
)
from modelcypher.core.domain.agents.emotion_concept_atlas import (
    EmotionConcept,
    EmotionDyad,
    EmotionCategory,
    EmotionConceptInventory,
)


class AtlasSource(str, Enum):
    """Source atlas for a unified probe."""
    SEQUENCE_INVARIANT = "sequence_invariant"
    SEMANTIC_PRIME = "semantic_prime"
    COMPUTATIONAL_GATE = "computational_gate"
    EMOTION_CONCEPT = "emotion_concept"


class AtlasDomain(str, Enum):
    """
    Cross-domain categories for triangulation scoring.

    Maps different atlas categories into unified triangulation domains.
    When a concept is detected across multiple domains, triangulation
    confidence increases.
    """
    # Mathematical/logical domains
    MATHEMATICAL = "mathematical"       # Sequences, ratios, patterns
    LOGICAL = "logical"                 # Logic, conditionals, causality

    # Language/semantic domains
    LINGUISTIC = "linguistic"           # Semantic primes, speech acts
    MENTAL = "mental"                   # Mental predicates, cognitive

    # Computational domains
    COMPUTATIONAL = "computational"     # Code gates, algorithms
    STRUCTURAL = "structural"           # Data types, modularity

    # Affective domains
    AFFECTIVE = "affective"             # Emotions, valence
    RELATIONAL = "relational"           # Social, interpersonal

    # Temporal/spatial domains
    TEMPORAL = "temporal"               # Time concepts
    SPATIAL = "spatial"                 # Place, location


# Domain mapping for each atlas category
_SEQUENCE_DOMAIN_MAP: dict[SequenceFamily, AtlasDomain] = {
    SequenceFamily.FIBONACCI: AtlasDomain.MATHEMATICAL,
    SequenceFamily.LUCAS: AtlasDomain.MATHEMATICAL,
    SequenceFamily.TRIBONACCI: AtlasDomain.MATHEMATICAL,
    SequenceFamily.PRIMES: AtlasDomain.MATHEMATICAL,
    SequenceFamily.CATALAN: AtlasDomain.MATHEMATICAL,
    SequenceFamily.RAMANUJAN: AtlasDomain.MATHEMATICAL,
    SequenceFamily.LOGIC: AtlasDomain.LOGICAL,
    SequenceFamily.ORDERING: AtlasDomain.LOGICAL,
    SequenceFamily.ARITHMETIC: AtlasDomain.MATHEMATICAL,
    SequenceFamily.CAUSALITY: AtlasDomain.LOGICAL,
}

_SEMANTIC_DOMAIN_MAP: dict[SemanticPrimeCategory, AtlasDomain] = {
    SemanticPrimeCategory.SUBSTANTIVES: AtlasDomain.LINGUISTIC,
    SemanticPrimeCategory.RELATIONAL_SUBSTANTIVES: AtlasDomain.RELATIONAL,
    SemanticPrimeCategory.DETERMINERS: AtlasDomain.LINGUISTIC,
    SemanticPrimeCategory.QUANTIFIERS: AtlasDomain.MATHEMATICAL,
    SemanticPrimeCategory.EVALUATORS: AtlasDomain.AFFECTIVE,
    SemanticPrimeCategory.DESCRIPTORS: AtlasDomain.LINGUISTIC,
    SemanticPrimeCategory.MENTAL_PREDICATES: AtlasDomain.MENTAL,
    SemanticPrimeCategory.SPEECH: AtlasDomain.LINGUISTIC,
    SemanticPrimeCategory.ACTIONS_EVENTS_MOVEMENT: AtlasDomain.STRUCTURAL,
    SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION: AtlasDomain.SPATIAL,
    SemanticPrimeCategory.POSSESSION: AtlasDomain.RELATIONAL,
    SemanticPrimeCategory.LIFE_AND_DEATH: AtlasDomain.TEMPORAL,
    SemanticPrimeCategory.TIME: AtlasDomain.TEMPORAL,
    SemanticPrimeCategory.PLACE: AtlasDomain.SPATIAL,
    SemanticPrimeCategory.LOGICAL_CONCEPTS: AtlasDomain.LOGICAL,
    SemanticPrimeCategory.AUGMENTOR_INTENSIFIER: AtlasDomain.LINGUISTIC,
    SemanticPrimeCategory.SIMILARITY: AtlasDomain.RELATIONAL,
}

_GATE_DOMAIN_MAP: dict[ComputationalGateCategory, AtlasDomain] = {
    ComputationalGateCategory.CORE_CONCEPTS: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.CONTROL_FLOW: AtlasDomain.LOGICAL,
    ComputationalGateCategory.FUNCTIONS_SCOPING: AtlasDomain.STRUCTURAL,
    ComputationalGateCategory.DATA_TYPES: AtlasDomain.STRUCTURAL,
    ComputationalGateCategory.DOMAIN_SPECIFIC: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.CONCURRENCY_PARALLELISM: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.MEMORY_MANAGEMENT: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.SYSTEM_IO: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.MODULARITY: AtlasDomain.STRUCTURAL,
    ComputationalGateCategory.ERROR_HANDLING: AtlasDomain.LOGICAL,
    ComputationalGateCategory.OBJECT_ORIENTED: AtlasDomain.STRUCTURAL,
    ComputationalGateCategory.METAPROGRAMMING: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.UNCATEGORIZED: AtlasDomain.COMPUTATIONAL,
    ComputationalGateCategory.COMPOSITE: AtlasDomain.COMPUTATIONAL,
}

_EMOTION_DOMAIN_MAP: dict[EmotionCategory, AtlasDomain] = {
    EmotionCategory.JOY: AtlasDomain.AFFECTIVE,
    EmotionCategory.TRUST: AtlasDomain.RELATIONAL,
    EmotionCategory.FEAR: AtlasDomain.AFFECTIVE,
    EmotionCategory.SURPRISE: AtlasDomain.MENTAL,
    EmotionCategory.SADNESS: AtlasDomain.AFFECTIVE,
    EmotionCategory.DISGUST: AtlasDomain.AFFECTIVE,
    EmotionCategory.ANGER: AtlasDomain.AFFECTIVE,
    EmotionCategory.ANTICIPATION: AtlasDomain.MENTAL,
}


@dataclass(frozen=True)
class AtlasProbe:
    """
    Unified probe from any atlas source.

    Normalizes probes from sequence invariants, semantic primes, computational gates,
    and emotion concepts into a common format for cross-domain triangulation.
    """
    id: str                          # Unique ID (prefixed with source)
    source: AtlasSource              # Which atlas this came from
    domain: AtlasDomain              # Unified triangulation domain
    name: str                        # Human-readable name
    description: str                 # Brief description
    cross_domain_weight: float       # Weight for cross-domain scoring (0.0-2.0)

    # Original category (for filtering)
    category_name: str               # Original category name

    # Optional metadata
    support_texts: tuple[str, ...] = ()  # Example texts for embedding

    @property
    def probe_id(self) -> str:
        """Full probe ID including source prefix."""
        return f"{self.source.value}:{self.id}"


# Default cross-domain weights for each source
_DEFAULT_WEIGHTS: dict[AtlasSource, float] = {
    AtlasSource.SEQUENCE_INVARIANT: 1.2,   # Mathematical invariants are strong anchors
    AtlasSource.SEMANTIC_PRIME: 1.0,       # Linguistic primes are reliable
    AtlasSource.COMPUTATIONAL_GATE: 1.1,   # Computational patterns are robust
    AtlasSource.EMOTION_CONCEPT: 0.9,      # Emotions are softer but useful
}


class UnifiedAtlasInventory:
    """
    Unified inventory of all atlas probes.

    Combines:
    - 68 sequence invariants
    - 65 semantic primes
    - 72 computational gates
    - 32 emotion concepts

    Total: 237 probes for cross-domain triangulation
    """

    _cached_probes: list[AtlasProbe] | None = None

    @classmethod
    def all_probes(cls) -> list[AtlasProbe]:
        """Get all probes from all atlases."""
        if cls._cached_probes is not None:
            return list(cls._cached_probes)

        probes: list[AtlasProbe] = []
        probes.extend(cls._sequence_invariant_probes())
        probes.extend(cls._semantic_prime_probes())
        probes.extend(cls._computational_gate_probes())
        probes.extend(cls._emotion_concept_probes())

        cls._cached_probes = probes
        return list(probes)

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
    def _sequence_invariant_probes(cls) -> list[AtlasProbe]:
        """Convert sequence invariants to unified probes."""
        invariants = SequenceInvariantInventory.probes_for_families()
        probes: list[AtlasProbe] = []

        for inv in invariants:
            domain = _SEQUENCE_DOMAIN_MAP.get(inv.family, AtlasDomain.MATHEMATICAL)
            probes.append(AtlasProbe(
                id=f"{inv.family.value}_{inv.id}",
                source=AtlasSource.SEQUENCE_INVARIANT,
                domain=domain,
                name=inv.name,
                description=inv.description,
                cross_domain_weight=inv.cross_domain_weight,
                category_name=inv.family.value,
                support_texts=inv.support_texts,
            ))

        return probes

    @classmethod
    def _semantic_prime_probes(cls) -> list[AtlasProbe]:
        """Convert semantic primes to unified probes."""
        primes = SemanticPrimeInventory.english_2014()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.SEMANTIC_PRIME]

        for prime in primes:
            domain = _SEMANTIC_DOMAIN_MAP.get(prime.category, AtlasDomain.LINGUISTIC)
            # Create support texts from English exponents
            support_texts = tuple(f"The concept of '{exp}'" for exp in prime.english_exponents)

            probes.append(AtlasProbe(
                id=prime.id,
                source=AtlasSource.SEMANTIC_PRIME,
                domain=domain,
                name=prime.canonical_english,
                description=f"Semantic prime: {prime.id}",
                cross_domain_weight=base_weight,
                category_name=prime.category.value,
                support_texts=support_texts,
            ))

        return probes

    @classmethod
    def _computational_gate_probes(cls) -> list[AtlasProbe]:
        """Convert computational gates to unified probes."""
        gates = ComputationalGateInventory.all_gates()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.COMPUTATIONAL_GATE]

        for gate in gates:
            domain = _GATE_DOMAIN_MAP.get(gate.category, AtlasDomain.COMPUTATIONAL)
            # Create support texts from examples
            support_texts = tuple(gate.examples) if gate.examples else ()

            probes.append(AtlasProbe(
                id=gate.id,
                source=AtlasSource.COMPUTATIONAL_GATE,
                domain=domain,
                name=gate.name,
                description=gate.description,
                cross_domain_weight=base_weight,
                category_name=gate.category.value,
                support_texts=support_texts,
            ))

        return probes

    @classmethod
    def _emotion_concept_probes(cls) -> list[AtlasProbe]:
        """Convert emotion concepts and dyads to unified probes."""
        emotions = EmotionConceptInventory.all_emotions()
        dyads = EmotionConceptInventory.all_dyads()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.EMOTION_CONCEPT]

        # Add emotions
        for emotion in emotions:
            domain = _EMOTION_DOMAIN_MAP.get(emotion.category, AtlasDomain.AFFECTIVE)
            probes.append(AtlasProbe(
                id=emotion.id,
                source=AtlasSource.EMOTION_CONCEPT,
                domain=domain,
                name=emotion.name,
                description=emotion.description,
                cross_domain_weight=base_weight,
                category_name=emotion.category.value,
                support_texts=emotion.support_texts,
            ))

        # Add dyads with blended domain
        for dyad in dyads:
            probes.append(AtlasProbe(
                id=f"dyad_{dyad.id}",
                source=AtlasSource.EMOTION_CONCEPT,
                domain=AtlasDomain.AFFECTIVE,  # Dyads are affective blends
                name=dyad.name,
                description=dyad.description,
                cross_domain_weight=base_weight * 0.9,  # Slightly lower for blends
                category_name="dyad",
                support_texts=dyad.support_texts,
            ))

        return probes


# Convenience constants
ALL_ATLAS_SOURCES = frozenset(AtlasSource)
MATHEMATICAL_DOMAINS = frozenset([AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL])
LINGUISTIC_DOMAINS = frozenset([AtlasDomain.LINGUISTIC, AtlasDomain.MENTAL])
COMPUTATIONAL_DOMAINS = frozenset([AtlasDomain.COMPUTATIONAL, AtlasDomain.STRUCTURAL])
AFFECTIVE_DOMAINS = frozenset([AtlasDomain.AFFECTIVE, AtlasDomain.RELATIONAL])
SPATIOTEMPORAL_DOMAINS = frozenset([AtlasDomain.TEMPORAL, AtlasDomain.SPATIAL])

# Default sources for layer mapping (all sources enabled)
DEFAULT_ATLAS_SOURCES = frozenset([
    AtlasSource.SEQUENCE_INVARIANT,
    AtlasSource.SEMANTIC_PRIME,
    AtlasSource.COMPUTATIONAL_GATE,
    AtlasSource.EMOTION_CONCEPT,
])


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
    source_multiplier: float      # Boost for detection across sources
    domain_multiplier: float      # Boost for detection across domains
    combined_multiplier: float    # Final combined multiplier


class MultiAtlasTriangulationScorer:
    """
    Scorer for cross-atlas triangulation.

    Provides confidence boosts when concepts are detected across multiple
    atlas sources and triangulation domains.
    """

    @staticmethod
    def compute_score(
        activations: dict[AtlasProbe, float],
        threshold: float = 0.3,
    ) -> MultiAtlasTriangulationScore:
        """
        Compute triangulation score from probe activations.

        Args:
            activations: Map of probes to their activation values
            threshold: Minimum activation to count as detected

        Returns:
            MultiAtlasTriangulationScore with multipliers
        """
        sources_detected: set[AtlasSource] = set()
        domains_detected: set[AtlasDomain] = set()

        for probe, activation in activations.items():
            if activation > threshold:
                sources_detected.add(probe.source)
                domains_detected.add(probe.domain)

        # Source multiplier: boost when detected across multiple atlas sources
        # 1 source = 1.0, 2 = 1.1, 3 = 1.2, 4 = 1.3
        source_count = len(sources_detected)
        source_multiplier = 1.0 + (source_count - 1) * 0.1 if source_count > 0 else 1.0

        # Domain multiplier: boost when detected across multiple domains
        # 1 domain = 1.0, 2 = 1.15, 3 = 1.3, 4 = 1.45, 5+ = 1.6+
        domain_count = len(domains_detected)
        domain_multiplier = 1.0 + (domain_count - 1) * 0.15 if domain_count > 0 else 1.0

        # Combined: geometric mean of source and domain multipliers
        combined_multiplier = (source_multiplier * domain_multiplier) ** 0.5

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
        sources = DEFAULT_ATLAS_SOURCES

    probes = UnifiedAtlasInventory.probes_by_source(sources)
    return [probe.probe_id for probe in probes]
