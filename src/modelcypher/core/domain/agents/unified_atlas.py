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

"""
Unified Atlas.

Combines all atlas sources (sequence invariants, semantic primes, computational gates,
emotion concepts, temporal, social, moral, compositional, philosophical, conceptual
genealogy) into a single
unified probe system for cross-domain triangulation in layer mapping operations.

Philosophy IS conceptual math. These probes measure the fundamental categories of
thought - the structural preconditions for coherent reasoning that are INVARIANT
across all models. Knowledge occupies fixed probability clouds in hyperspace.

Total probe count: 439
- Sequence Invariants: 68 probes (10 families)
- Semantic Primes: 65 probes (17 categories)
- Computational Gates: 76 probes (14 categories)
- Emotion Concepts: 32 probes (8 categories + 8 dyads)
- Temporal Concepts: 25 probes (tense, duration, causality, lifecycle, sequence)
- Spatial Concepts: 23 probes (vertical, lateral, depth, mass, furniture)
- Social Concepts: 25 probes (power, formality, kinship, status, age)
- Moral Concepts: 30 probes (Haidt Moral Foundations Theory)
- Compositional: 22 probes (semantic prime compositions)
- Philosophical: 30 probes (ontological, epistemological, logical, modal, mereological)
- Conceptual Genealogy: 29 probes (etymology + lineage)
- Metaphor Invariants: 14 probes (cross-cultural semantic anchors)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.agents.conceptual_genealogy_atlas import (
    ConceptDomain,
    ConceptualGenealogyInventory,
)
from modelcypher.core.domain.agents.computational_gate_atlas import (
    ComputationalGateCategory,
    ComputationalGateInventory,
)
from modelcypher.core.domain.agents.emotion_concept_atlas import (
    EmotionCategory,
    EmotionConceptInventory,
)
from modelcypher.core.domain.agents.moral_atlas import (
    MoralConceptInventory,
    MoralFoundation,
)
from modelcypher.core.domain.agents.metaphor_invariant_atlas import (
    MetaphorFamily,
    MetaphorInvariantInventory,
)
from modelcypher.core.domain.agents.philosophical_atlas import (
    PhilosophicalCategory,
    PhilosophicalConceptInventory,
)
from modelcypher.core.domain.agents.semantic_prime_atlas import (
    SemanticPrimeCategory,
    SemanticPrimeInventory,
)
from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.domain.agents.social_atlas import (
    SocialCategory,
    SocialConceptInventory,
)
from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
from modelcypher.core.domain.agents.temporal_atlas import (
    TemporalCategory,
    TemporalConceptInventory,
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
    COMPOSITIONAL = "compositional"  # Semantic prime compositions (I THINK, GOOD THINGS, etc.)
    PHILOSOPHICAL_CONCEPT = "philosophical_concept"  # Fundamental categories of thought
    CONCEPTUAL_GENEALOGY = "conceptual_genealogy"
    METAPHOR_INVARIANT = "metaphor_invariant"


class AtlasDomain(str, Enum):
    """
    Cross-domain categories for triangulation scoring.

    Maps different atlas categories into unified triangulation domains.
    When a concept is detected across multiple domains, triangulation
    confidence increases.
    """

    # Mathematical/logical domains
    MATHEMATICAL = "mathematical"  # Sequences, ratios, patterns
    LOGICAL = "logical"  # Logic, conditionals, causality

    # Language/semantic domains
    LINGUISTIC = "linguistic"  # Semantic primes, speech acts
    MENTAL = "mental"  # Mental predicates, cognitive

    # Computational domains
    COMPUTATIONAL = "computational"  # Code gates, algorithms
    STRUCTURAL = "structural"  # Data types, modularity

    # Affective domains
    AFFECTIVE = "affective"  # Emotions, valence
    RELATIONAL = "relational"  # Social, interpersonal

    # Temporal/spatial domains
    TEMPORAL = "temporal"  # Time concepts
    SPATIAL = "spatial"  # Place, location

    # Moral/ethical domains
    MORAL = "moral"  # Ethics, virtue, vice

    # Philosophical domains (fundamental categories of thought)
    PHILOSOPHICAL = "philosophical"  # Ontology, epistemology, logic, modality, mereology


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

# Domain mapping for philosophical categories
# All philosophical probes map to PHILOSOPHICAL domain - they are fundamental
# categories of thought that underlie all other domains
_PHILOSOPHICAL_DOMAIN_MAP: dict[PhilosophicalCategory, AtlasDomain] = {
    PhilosophicalCategory.ONTOLOGICAL: AtlasDomain.PHILOSOPHICAL,
    PhilosophicalCategory.EPISTEMOLOGICAL: AtlasDomain.PHILOSOPHICAL,
    PhilosophicalCategory.LOGICAL: AtlasDomain.LOGICAL,  # Overlaps with sequence invariants
    PhilosophicalCategory.MODAL: AtlasDomain.PHILOSOPHICAL,
    PhilosophicalCategory.MEREOLOGICAL: AtlasDomain.PHILOSOPHICAL,
}

_GENEALOGY_DOMAIN_MAP: dict[ConceptDomain, AtlasDomain] = {
    ConceptDomain.PHILOSOPHY: AtlasDomain.PHILOSOPHICAL,
    ConceptDomain.SCIENCE: AtlasDomain.MATHEMATICAL,
    ConceptDomain.MATHEMATICS: AtlasDomain.MATHEMATICAL,
    ConceptDomain.POLITICS: AtlasDomain.RELATIONAL,
    ConceptDomain.ETHICS: AtlasDomain.MORAL,
    ConceptDomain.AESTHETICS: AtlasDomain.AFFECTIVE,
}

_METAPHOR_DOMAIN_MAP: dict[MetaphorFamily, AtlasDomain] = {
    MetaphorFamily.FUTILITY: AtlasDomain.LINGUISTIC,
    MetaphorFamily.IMPOSSIBILITY: AtlasDomain.LINGUISTIC,
    MetaphorFamily.OBVIOUSNESS: AtlasDomain.LINGUISTIC,
    MetaphorFamily.CONSEQUENCE: AtlasDomain.LINGUISTIC,
    MetaphorFamily.FRAGILITY: AtlasDomain.LINGUISTIC,
    MetaphorFamily.DECEPTION: AtlasDomain.LINGUISTIC,
    MetaphorFamily.RESILIENCE: AtlasDomain.LINGUISTIC,
}


@dataclass(frozen=True)
class AtlasProbe:
    """
    Unified probe from any atlas source.

    Normalizes probes from sequence invariants, semantic primes, computational gates,
    and emotion concepts into a common format for cross-domain triangulation.
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


_DEFAULT_WEIGHTS: dict[AtlasSource, float] = {
    AtlasSource.SEQUENCE_INVARIANT: 1.0,
    AtlasSource.SEMANTIC_PRIME: 1.0,
    AtlasSource.COMPUTATIONAL_GATE: 1.0,
    AtlasSource.EMOTION_CONCEPT: 1.0,
    AtlasSource.TEMPORAL_CONCEPT: 1.0,
    AtlasSource.SPATIAL_CONCEPT: 1.0,
    AtlasSource.SOCIAL_CONCEPT: 1.0,
    AtlasSource.MORAL_CONCEPT: 1.0,
    AtlasSource.COMPOSITIONAL: 1.0,
    AtlasSource.PHILOSOPHICAL_CONCEPT: 1.0,
    AtlasSource.CONCEPTUAL_GENEALOGY: 1.0,
    AtlasSource.METAPHOR_INVARIANT: 1.0,
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
    - 23 spatial concepts (validated 2025-12-23)
    - 25 social concepts (validated 2025-12-23, SMS=0.53)
    - 30 moral concepts (Haidt Moral Foundations Theory)
    - 22 compositional probes (semantic prime compositions)
    - 30 philosophical concepts (ontological, epistemological, logical, modal, mereological)
    - 29 conceptual genealogy probes (etymology + lineage)
    - 14 metaphor invariants (cross-cultural semantic anchors)

    Total: 439 probes for cross-domain triangulation
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
        probes.extend(cls._spatial_concept_probes())
        probes.extend(cls._social_concept_probes())
        probes.extend(cls._moral_concept_probes())
        probes.extend(cls._compositional_probes())
        probes.extend(cls._philosophical_concept_probes())
        probes.extend(cls._conceptual_genealogy_probes())
        probes.extend(cls._metaphor_invariant_probes())

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
            probes.append(
                AtlasProbe(
                    id=f"{inv.family.value}_{inv.id}",
                    source=AtlasSource.SEQUENCE_INVARIANT,
                    domain=domain,
                    name=inv.name,
                    description=inv.description,
                    cross_domain_weight=inv.cross_domain_weight,
                    category_name=inv.family.value,
                    support_texts=inv.support_texts,
                )
            )

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

            probes.append(
                AtlasProbe(
                    id=prime.id,
                    source=AtlasSource.SEMANTIC_PRIME,
                    domain=domain,
                    name=prime.canonical_english,
                    description=f"Semantic prime: {prime.id}",
                    cross_domain_weight=base_weight,
                    category_name=prime.category.value,
                    support_texts=support_texts,
                )
            )

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

            probes.append(
                AtlasProbe(
                    id=gate.id,
                    source=AtlasSource.COMPUTATIONAL_GATE,
                    domain=domain,
                    name=gate.name,
                    description=gate.description,
                    cross_domain_weight=base_weight,
                    category_name=gate.category.value,
                    support_texts=support_texts,
                )
            )

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
            probes.append(
                AtlasProbe(
                    id=emotion.id,
                    source=AtlasSource.EMOTION_CONCEPT,
                    domain=domain,
                    name=emotion.name,
                    description=emotion.description,
                    cross_domain_weight=base_weight,
                    category_name=emotion.category.value,
                    support_texts=emotion.support_texts,
                )
            )

        # Add dyads with blended domain
        for dyad in dyads:
            probes.append(
                AtlasProbe(
                    id=f"dyad_{dyad.id}",
                    source=AtlasSource.EMOTION_CONCEPT,
                    domain=AtlasDomain.AFFECTIVE,  # Dyads are affective blends
                    name=dyad.name,
                    description=dyad.description,
                    cross_domain_weight=base_weight,  # Uniform - calibration determines weight
                    category_name="dyad",
                    support_texts=dyad.support_texts,
                )
            )

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
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.TEMPORAL_CONCEPT,
                    domain=domain,
                    name=concept.name,
                    description=concept.description,
                    cross_domain_weight=concept.cross_domain_weight * base_weight,
                    category_name=concept.category.value,
                    support_texts=concept.support_texts,
                )
            )

        return probes

    @classmethod
    def _spatial_concept_probes(cls) -> list[AtlasProbe]:
        """Convert spatial concepts to unified probes."""
        concepts = SpatialConceptInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.SPATIAL_CONCEPT]

        for concept in concepts:
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.SPATIAL_CONCEPT,
                    domain=AtlasDomain.SPATIAL,
                    name=concept.name,
                    description=f"Spatial anchor: {concept.name}",
                    cross_domain_weight=base_weight,
                    category_name=concept.category.value,
                    support_texts=(concept.prompt,),
                )
            )

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
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.SOCIAL_CONCEPT,
                    domain=domain,
                    name=concept.name,
                    description=concept.description,
                    cross_domain_weight=concept.cross_domain_weight * base_weight,
                    category_name=concept.category.value,
                    support_texts=concept.support_texts,
                )
            )

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
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.MORAL_CONCEPT,
                    domain=domain,
                    name=concept.name,
                    description=concept.description,
                    cross_domain_weight=concept.cross_domain_weight * base_weight,
                    category_name=concept.foundation.value,
                    support_texts=concept.support_texts,
                )
            )

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
        from modelcypher.core.domain.geometry.compositional_probes import CompositionalProbes

        probes: list[AtlasProbe] = []
        base_weight = _DEFAULT_WEIGHTS[AtlasSource.COMPOSITIONAL]

        for probe in CompositionalProbes.STANDARD_PROBES:
            domain = _COMPOSITIONAL_DOMAIN_MAP.get(probe.category.value, AtlasDomain.MENTAL)
            # Create support text from the phrase and its components
            component_text = " + ".join(probe.components)
            probes.append(
                AtlasProbe(
                    id=probe.phrase.lower().replace(" ", "_"),
                    source=AtlasSource.COMPOSITIONAL,
                    domain=domain,
                    name=probe.phrase,
                    description=f"Composition: {component_text}",
                    cross_domain_weight=base_weight,
                    category_name=probe.category.value,
                    support_texts=(probe.phrase, component_text),
                )
            )

        return probes

    @classmethod
    def _philosophical_concept_probes(cls) -> list[AtlasProbe]:
        """Convert philosophical concepts to unified probes.

        Philosophical probes test the fundamental categories of thought:
        - ONTOLOGICAL: Being, substance, attribute, potential, actual
        - EPISTEMOLOGICAL: Knowledge, belief, understanding, wisdom
        - LOGICAL: Identity, contradiction, implication, necessity
        - MODAL: Possible, impossible, necessary, contingent
        - MEREOLOGICAL: Part, whole, unity, plurality

        Total: 30 probes testing conceptual invariants.
        Philosophy IS conceptual math - these categories are the structural
        preconditions for coherent thought.
        """
        concepts = PhilosophicalConceptInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.PHILOSOPHICAL_CONCEPT]

        for concept in concepts:
            domain = _PHILOSOPHICAL_DOMAIN_MAP.get(
                concept.category, AtlasDomain.PHILOSOPHICAL
            )
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.PHILOSOPHICAL_CONCEPT,
                    domain=domain,
                    name=concept.name,
                    description=concept.description,
                    cross_domain_weight=concept.cross_domain_weight * base_weight,
                    category_name=concept.category.value,
                    support_texts=concept.support_texts,
                )
            )

        return probes

    @classmethod
    def _conceptual_genealogy_probes(cls) -> list[AtlasProbe]:
        """Convert conceptual genealogy probes to unified probes."""
        concepts = ConceptualGenealogyInventory.all_concepts()
        probes: list[AtlasProbe] = []

        base_weight = _DEFAULT_WEIGHTS[AtlasSource.CONCEPTUAL_GENEALOGY]

        for concept in concepts:
            domain = _GENEALOGY_DOMAIN_MAP.get(concept.domain, AtlasDomain.PHILOSOPHICAL)
            probes.append(
                AtlasProbe(
                    id=concept.id,
                    source=AtlasSource.CONCEPTUAL_GENEALOGY,
                    domain=domain,
                    name=concept.name,
                    description=concept.description,
                    cross_domain_weight=concept.cross_domain_weight * base_weight,
                    category_name=concept.domain.value,
                    support_texts=concept.support_texts,
                )
            )

        return probes

    @classmethod
    def _metaphor_invariant_probes(cls) -> list[AtlasProbe]:
        """Convert metaphor invariants to unified probes."""
        probes: list[AtlasProbe] = []
        base_weight = _DEFAULT_WEIGHTS[AtlasSource.METAPHOR_INVARIANT]

        for invariant in MetaphorInvariantInventory.ALL_PROBES:
            domain = _METAPHOR_DOMAIN_MAP.get(invariant.family, AtlasDomain.LINGUISTIC)
            support_texts = tuple(variation.phrase for variation in invariant.variations)
            probes.append(
                AtlasProbe(
                    id=invariant.id,
                    source=AtlasSource.METAPHOR_INVARIANT,
                    domain=domain,
                    name=invariant.universal_concept,
                    description=invariant.universal_concept,
                    cross_domain_weight=base_weight,
                    category_name=invariant.family.value,
                    support_texts=support_texts,
                )
            )

        return probes


# Convenience constants
ALL_ATLAS_SOURCES = frozenset(AtlasSource)
MATHEMATICAL_DOMAINS = frozenset([AtlasDomain.MATHEMATICAL, AtlasDomain.LOGICAL])
LINGUISTIC_DOMAINS = frozenset([AtlasDomain.LINGUISTIC, AtlasDomain.MENTAL])
COMPUTATIONAL_DOMAINS = frozenset([AtlasDomain.COMPUTATIONAL, AtlasDomain.STRUCTURAL])
AFFECTIVE_DOMAINS = frozenset([AtlasDomain.AFFECTIVE, AtlasDomain.RELATIONAL])
SPATIOTEMPORAL_DOMAINS = frozenset([AtlasDomain.TEMPORAL, AtlasDomain.SPATIAL])
MORAL_DOMAINS = frozenset([AtlasDomain.MORAL])
PHILOSOPHICAL_DOMAINS = frozenset([AtlasDomain.PHILOSOPHICAL, AtlasDomain.LOGICAL])

# Default sources for layer mapping (all sources enabled)
DEFAULT_ATLAS_SOURCES = frozenset(
    [
        AtlasSource.SEQUENCE_INVARIANT,
        AtlasSource.SEMANTIC_PRIME,
        AtlasSource.COMPUTATIONAL_GATE,
        AtlasSource.EMOTION_CONCEPT,
        AtlasSource.TEMPORAL_CONCEPT,
        AtlasSource.SOCIAL_CONCEPT,
        AtlasSource.MORAL_CONCEPT,
        AtlasSource.COMPOSITIONAL,
        AtlasSource.PHILOSOPHICAL_CONCEPT,
        AtlasSource.CONCEPTUAL_GENEALOGY,
    ]
)


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

        source_multiplier = 1.0
        domain_multiplier = 1.0
        combined_multiplier = 1.0

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
