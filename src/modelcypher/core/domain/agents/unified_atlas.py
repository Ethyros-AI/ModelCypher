"""
Unified Atlas.

Combines all atlas sources (sequence invariants, semantic primes, computational gates,
emotion concepts, temporal, social, moral, probe corpus, compositional) into a single
unified probe system for cross-domain triangulation in layer mapping operations.

Total probe count: 403
- Sequence Invariants: 68 probes (10 families)
- Semantic Primes: 65 probes (17 categories)
- Computational Gates: 76 probes (14 categories)
- Emotion Concepts: 32 probes (8 categories + 8 dyads)
- Temporal Concepts: 25 probes (tense, duration, causality, lifecycle, sequence)
- Social Concepts: 25 probes (power, formality, kinship, status, age)
- Moral Concepts: 30 probes (Haidt Moral Foundations Theory)
- Probe Corpus: 60 probes (code, math, creative, reasoning, factual, general)
- Compositional: 22 probes (semantic prime compositions)
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
from modelcypher.core.domain.agents.temporal_atlas import (
    TemporalConcept,
    TemporalCategory,
    TemporalAxis,
    TemporalConceptInventory,
)
from modelcypher.core.domain.agents.social_atlas import (
    SocialConcept,
    SocialCategory,
    SocialAxis,
    SocialConceptInventory,
)
from modelcypher.core.domain.agents.moral_atlas import (
    MoralConcept,
    MoralFoundation,
    MoralAxis,
    MoralConceptInventory,
)


class AtlasSource(str, Enum):
    """Source atlas for a unified probe."""
    SEQUENCE_INVARIANT = "sequence_invariant"
    SEMANTIC_PRIME = "semantic_prime"
    COMPUTATIONAL_GATE = "computational_gate"
    EMOTION_CONCEPT = "emotion_concept"
    TEMPORAL_CONCEPT = "temporal_concept"
    SOCIAL_CONCEPT = "social_concept"
    MORAL_CONCEPT = "moral_concept"
    PROBE_CORPUS = "probe_corpus"        # General probing (code, math, creative, reasoning)
    COMPOSITIONAL = "compositional"      # Semantic prime compositions (I THINK, GOOD THINGS, etc.)


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

    # Moral/ethical domains
    MORAL = "moral"                     # Ethics, virtue, vice


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

_TEMPORAL_DOMAIN_MAP: dict[TemporalCategory, AtlasDomain] = {
    TemporalCategory.TENSE: AtlasDomain.TEMPORAL,
    TemporalCategory.DURATION: AtlasDomain.TEMPORAL,
    TemporalCategory.CAUSALITY: AtlasDomain.LOGICAL,
    TemporalCategory.LIFECYCLE: AtlasDomain.TEMPORAL,
    TemporalCategory.SEQUENCE: AtlasDomain.TEMPORAL,
}

_SOCIAL_DOMAIN_MAP: dict[SocialCategory, AtlasDomain] = {
    SocialCategory.POWER_HIERARCHY: AtlasDomain.RELATIONAL,
    SocialCategory.FORMALITY: AtlasDomain.LINGUISTIC,
    SocialCategory.KINSHIP: AtlasDomain.RELATIONAL,
    SocialCategory.STATUS_MARKERS: AtlasDomain.RELATIONAL,
    SocialCategory.AGE: AtlasDomain.RELATIONAL,
}

_MORAL_DOMAIN_MAP: dict[MoralFoundation, AtlasDomain] = {
    MoralFoundation.CARE_HARM: AtlasDomain.MORAL,
    MoralFoundation.FAIRNESS_CHEATING: AtlasDomain.MORAL,
    MoralFoundation.LOYALTY_BETRAYAL: AtlasDomain.RELATIONAL,
    MoralFoundation.AUTHORITY_SUBVERSION: AtlasDomain.RELATIONAL,
    MoralFoundation.SANCTITY_DEGRADATION: AtlasDomain.MORAL,
    MoralFoundation.LIBERTY_OPPRESSION: AtlasDomain.MORAL,
}

# Domain mapping for probe corpus domains
_PROBE_CORPUS_DOMAIN_MAP: dict[str, AtlasDomain] = {
    "general_language": AtlasDomain.LINGUISTIC,
    "code": AtlasDomain.COMPUTATIONAL,
    "math": AtlasDomain.MATHEMATICAL,
    "factual": AtlasDomain.LOGICAL,
    "creative": AtlasDomain.AFFECTIVE,
    "reasoning": AtlasDomain.LOGICAL,
}

# Domain mapping for compositional probe categories
_COMPOSITIONAL_DOMAIN_MAP: dict[str, AtlasDomain] = {
    "mentalPredicate": AtlasDomain.MENTAL,
    "action": AtlasDomain.STRUCTURAL,
    "evaluative": AtlasDomain.AFFECTIVE,
    "temporal": AtlasDomain.TEMPORAL,
    "spatial": AtlasDomain.SPATIAL,
    "quantified": AtlasDomain.MATHEMATICAL,
    "relational": AtlasDomain.RELATIONAL,
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
    AtlasSource.TEMPORAL_CONCEPT: 1.1,     # Temporal probes validated 2025-12-23
    AtlasSource.SOCIAL_CONCEPT: 1.15,      # Social probes validated 2025-12-23 (SMS=0.53)
    AtlasSource.MORAL_CONCEPT: 1.2,        # Moral probes (Haidt Moral Foundations)
    AtlasSource.PROBE_CORPUS: 1.0,         # General probing (code, math, creative, reasoning)
    AtlasSource.COMPOSITIONAL: 1.05,       # Semantic prime compositions
}


class UnifiedAtlasInventory:
    """
    Unified inventory of all atlas probes.

    Combines:
    - 68 sequence invariants
    - 65 semantic primes
    - 76 computational gates
    - 32 emotion concepts
    - 25 temporal concepts (validated 2025-12-23)
    - 25 social concepts (validated 2025-12-23, SMS=0.53)
    - 30 moral concepts (Haidt Moral Foundations Theory)
    - 60 probe corpus (code, math, creative, reasoning)
    - 22 compositional probes (semantic prime compositions)

    Total: 403 probes for cross-domain triangulation
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
        probes.extend(cls._temporal_concept_probes())
        probes.extend(cls._social_concept_probes())
        probes.extend(cls._moral_concept_probes())
        probes.extend(cls._probe_corpus_probes())
        probes.extend(cls._compositional_probes())

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

    @classmethod
    def _temporal_concept_probes(cls) -> list[AtlasProbe]:
        """Convert temporal concepts to unified probes.

        Temporal probes validated 2025-12-23:
        - Tests Latent Chronologist hypothesis
        - Direction, Duration, Causality axes
        - Arrow of Time detection
        """
        concepts = TemporalConceptInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.TEMPORAL_CONCEPT]

        for concept in concepts:
            domain = _TEMPORAL_DOMAIN_MAP.get(concept.category, AtlasDomain.TEMPORAL)
            probes.append(AtlasProbe(
                id=concept.id,
                source=AtlasSource.TEMPORAL_CONCEPT,
                domain=domain,
                name=concept.name,
                description=concept.description,
                cross_domain_weight=concept.cross_domain_weight * base_weight,
                category_name=concept.category.value,
                support_texts=concept.support_texts,
            ))

        return probes

    @classmethod
    def _social_concept_probes(cls) -> list[AtlasProbe]:
        """Convert social concepts to unified probes.

        Social probes validated 2025-12-23:
        - Tests Latent Sociologist hypothesis
        - Power, Kinship, Formality axes
        - Mean SMS: 0.53, Orthogonality: 94.8%
        - Qwen2.5-3B shows monotonic power hierarchy (r=1.0)
        """
        concepts = SocialConceptInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.SOCIAL_CONCEPT]

        for concept in concepts:
            domain = _SOCIAL_DOMAIN_MAP.get(concept.category, AtlasDomain.RELATIONAL)
            probes.append(AtlasProbe(
                id=concept.id,
                source=AtlasSource.SOCIAL_CONCEPT,
                domain=domain,
                name=concept.name,
                description=concept.description,
                cross_domain_weight=concept.cross_domain_weight * base_weight,
                category_name=concept.category.value,
                support_texts=concept.support_texts,
            ))

        return probes

    @classmethod
    def _moral_concept_probes(cls) -> list[AtlasProbe]:
        """Convert moral concepts to unified probes.

        Moral probes based on Haidt's Moral Foundations Theory:
        - Care/Harm, Fairness/Cheating, Loyalty/Betrayal
        - Authority/Subversion, Sanctity/Degradation, Liberty/Oppression
        - Valence, Agency, Scope axes for triangulation
        """
        concepts = MoralConceptInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.MORAL_CONCEPT]

        for concept in concepts:
            domain = _MORAL_DOMAIN_MAP.get(concept.foundation, AtlasDomain.MORAL)
            probes.append(AtlasProbe(
                id=concept.id,
                source=AtlasSource.MORAL_CONCEPT,
                domain=domain,
                name=concept.name,
                description=concept.description,
                cross_domain_weight=concept.cross_domain_weight * base_weight,
                category_name=concept.foundation.value,
                support_texts=concept.support_texts,
            ))

        return probes

    @classmethod
    def _probe_corpus_probes(cls) -> list[AtlasProbe]:
        """Convert probe corpus samples to unified probes.

        Probe corpus provides diverse samples across 6 domains:
        - general_language: Natural language pangrams, narratives
        - code: Python, Go, Rust, SQL, shell snippets
        - math: Calculus, algebra, number theory
        - factual: Scientific facts, geography, biology
        - creative: Poetry, fiction, metaphor
        - reasoning: Syllogisms, logic, critical thinking

        Total: 60 probes (10 per domain)
        """
        from modelcypher.core.domain.geometry.probe_corpus import ProbeCorpus

        corpus = ProbeCorpus.get_standard()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.PROBE_CORPUS]

        for sample in corpus.samples:
            domain = _PROBE_CORPUS_DOMAIN_MAP.get(sample.domain.value, AtlasDomain.LINGUISTIC)
            probes.append(AtlasProbe(
                id=sample.id,
                source=AtlasSource.PROBE_CORPUS,
                domain=domain,
                name=sample.id.replace("_", " ").title(),
                description=f"Probe corpus sample: {sample.domain.value}",
                cross_domain_weight=base_weight,
                category_name=sample.domain.value,
                support_texts=(sample.text,),
            ))

        return probes

    @classmethod
    def _compositional_probes(cls) -> list[AtlasProbe]:
        """Convert compositional probes to unified probes.

        Compositional probes test semantic prime compositions:
        - MENTAL_PREDICATE: I THINK, I KNOW, I WANT, I FEEL, I SEE, I HEAR
        - ACTION: SOMEONE DO, PEOPLE DO, I SAY
        - EVALUATIVE: GOOD THINGS, BAD THINGS, GOOD PEOPLE
        - TEMPORAL: BEFORE NOW, AFTER THIS, A LONG TIME BEFORE
        - SPATIAL: ABOVE HERE, FAR FROM HERE, NEAR THIS
        - QUANTIFIED: MUCH GOOD, MANY PEOPLE
        - Complex: I WANT GOOD THINGS, SOMEONE DO BAD THINGS

        Total: 22 probes testing compositional semantics
        """
        from modelcypher.core.domain.geometry.probes import CompositionalProbes

        probes: list[AtlasProbe] = []
        base_weight = _DEFAULT_WEIGHTS[AtlasSource.COMPOSITIONAL]

        for probe in CompositionalProbes.STANDARD_PROBES:
            domain = _COMPOSITIONAL_DOMAIN_MAP.get(probe.category.value, AtlasDomain.MENTAL)
            # Create support text from the phrase and its components
            component_text = " + ".join(probe.components)
            probes.append(AtlasProbe(
                id=probe.phrase.lower().replace(" ", "_"),
                source=AtlasSource.COMPOSITIONAL,
                domain=domain,
                name=probe.phrase,
                description=f"Composition: {component_text}",
                cross_domain_weight=base_weight,
                category_name=probe.category.value,
                support_texts=(probe.phrase, component_text),
            ))

        return probes


# Convenience constants
ALL_ATLAS_SOURCES = frozenset(AtlasSource)
MATHEMATICAL_DOMAINS = frozenset([AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL])
LINGUISTIC_DOMAINS = frozenset([AtlasDomain.LINGUISTIC, AtlasDomain.MENTAL])
COMPUTATIONAL_DOMAINS = frozenset([AtlasDomain.COMPUTATIONAL, AtlasDomain.STRUCTURAL])
AFFECTIVE_DOMAINS = frozenset([AtlasDomain.AFFECTIVE, AtlasDomain.RELATIONAL])
SPATIOTEMPORAL_DOMAINS = frozenset([AtlasDomain.TEMPORAL, AtlasDomain.SPATIAL])
MORAL_DOMAINS = frozenset([AtlasDomain.MORAL])

# Default sources for layer mapping (all sources enabled)
DEFAULT_ATLAS_SOURCES = frozenset([
    AtlasSource.SEQUENCE_INVARIANT,
    AtlasSource.SEMANTIC_PRIME,
    AtlasSource.COMPUTATIONAL_GATE,
    AtlasSource.EMOTION_CONCEPT,
    AtlasSource.TEMPORAL_CONCEPT,
    AtlasSource.SOCIAL_CONCEPT,
    AtlasSource.MORAL_CONCEPT,
    AtlasSource.PROBE_CORPUS,
    AtlasSource.COMPOSITIONAL,
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
        # 1 source = 1.0, 2 = 1.1, 3 = 1.2, 4 = 1.3, 5 = 1.4, 6 = 1.5, 7 = 1.6
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
