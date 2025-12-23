"""
Conceptual Genealogy for Extended Probing.

Provides probe texts based on etymology and concept lineage,
testing whether models understand how concepts evolved and relate historically.

These probes test the model's understanding of:
- Word etymology and semantic drift
- Concept borrowing across languages/cultures
- Historical development of abstract ideas
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ConceptDomain(str, Enum):
    """Domains of conceptual genealogy."""

    PHILOSOPHY = "philosophy"
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    POLITICS = "politics"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"


class LanguageOrigin(str, Enum):
    """Origin languages for etymological tracking."""

    GREEK = "greek"
    LATIN = "latin"
    ARABIC = "arabic"
    SANSKRIT = "sanskrit"
    CHINESE = "chinese"
    GERMANIC = "germanic"


@dataclass(frozen=True)
class EtymologyEntry:
    """An etymology entry tracing a word's origin."""

    modern_word: str
    origin_word: str
    origin_language: LanguageOrigin
    original_meaning: str
    semantic_shift: str  # How meaning changed over time


@dataclass(frozen=True)
class ConceptualGenealogyProbe:
    """A probe text derived from conceptual genealogy."""

    domain: ConceptDomain
    concept: str
    etymology: EtymologyEntry
    probe_text: str
    probe_id: str
    tests_semantic_drift: bool  # Whether probe tests meaning change

    @property
    def category(self) -> str:
        return f"genealogy_{self.domain.value}"


# =============================================================================
# Etymology Database
# =============================================================================

PHILOSOPHY_ETYMOLOGIES = [
    EtymologyEntry(
        modern_word="philosophy",
        origin_word="philosophia",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="love of wisdom",
        semantic_shift="From love of wisdom to academic discipline",
    ),
    EtymologyEntry(
        modern_word="logic",
        origin_word="logike",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="the art of reasoning",
        semantic_shift="From reasoning art to formal symbolic system",
    ),
    EtymologyEntry(
        modern_word="ethics",
        origin_word="ethikos",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="relating to character",
        semantic_shift="From character study to moral philosophy",
    ),
    EtymologyEntry(
        modern_word="metaphysics",
        origin_word="ta meta ta physika",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="the things after physics",
        semantic_shift="From book ordering to study of reality beyond physics",
    ),
    EtymologyEntry(
        modern_word="ontology",
        origin_word="ontos + logos",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="study of being",
        semantic_shift="From being-study to existence taxonomy",
    ),
]

SCIENCE_ETYMOLOGIES = [
    EtymologyEntry(
        modern_word="algorithm",
        origin_word="al-Khwarizmi",
        origin_language=LanguageOrigin.ARABIC,
        original_meaning="name of Persian mathematician",
        semantic_shift="From person's name to computational procedure",
    ),
    EtymologyEntry(
        modern_word="algebra",
        origin_word="al-jabr",
        origin_language=LanguageOrigin.ARABIC,
        original_meaning="reunion of broken parts",
        semantic_shift="From bone-setting metaphor to mathematical operations",
    ),
    EtymologyEntry(
        modern_word="chemistry",
        origin_word="al-kimiya",
        origin_language=LanguageOrigin.ARABIC,
        original_meaning="the Egyptian art",
        semantic_shift="From alchemy to modern science of matter",
    ),
    EtymologyEntry(
        modern_word="atom",
        origin_word="atomos",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="indivisible",
        semantic_shift="From indivisible unit to now-divisible particle",
    ),
    EtymologyEntry(
        modern_word="energy",
        origin_word="energeia",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="activity, operation",
        semantic_shift="From activity to quantifiable physical property",
    ),
]

MATHEMATICS_ETYMOLOGIES = [
    EtymologyEntry(
        modern_word="geometry",
        origin_word="geometria",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="earth measurement",
        semantic_shift="From land surveying to abstract spatial study",
    ),
    EtymologyEntry(
        modern_word="calculus",
        origin_word="calculus",
        origin_language=LanguageOrigin.LATIN,
        original_meaning="small pebble (for counting)",
        semantic_shift="From counting stones to infinitesimal analysis",
    ),
    EtymologyEntry(
        modern_word="zero",
        origin_word="sunya",
        origin_language=LanguageOrigin.SANSKRIT,
        original_meaning="void, emptiness",
        semantic_shift="From philosophical void to mathematical number",
    ),
    EtymologyEntry(
        modern_word="digit",
        origin_word="digitus",
        origin_language=LanguageOrigin.LATIN,
        original_meaning="finger",
        semantic_shift="From finger (for counting) to numerical symbol",
    ),
]

POLITICS_ETYMOLOGIES = [
    EtymologyEntry(
        modern_word="democracy",
        origin_word="demokratia",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="rule by the people",
        semantic_shift="From Athenian direct rule to representative systems",
    ),
    EtymologyEntry(
        modern_word="republic",
        origin_word="res publica",
        origin_language=LanguageOrigin.LATIN,
        original_meaning="public affair",
        semantic_shift="From public matter to political system",
    ),
    EtymologyEntry(
        modern_word="tyrant",
        origin_word="tyrannos",
        origin_language=LanguageOrigin.GREEK,
        original_meaning="absolute ruler (neutral)",
        semantic_shift="From neutral ruler to oppressive despot",
    ),
]

ETHICS_ETYMOLOGIES = [
    EtymologyEntry(
        modern_word="virtue",
        origin_word="virtus",
        origin_language=LanguageOrigin.LATIN,
        original_meaning="manliness, valor",
        semantic_shift="From masculine valor to moral excellence",
    ),
    EtymologyEntry(
        modern_word="conscience",
        origin_word="conscientia",
        origin_language=LanguageOrigin.LATIN,
        original_meaning="joint knowledge",
        semantic_shift="From shared knowledge to inner moral sense",
    ),
    EtymologyEntry(
        modern_word="sin",
        origin_word="synn",
        origin_language=LanguageOrigin.GERMANIC,
        original_meaning="guilt, true (being truly guilty)",
        semantic_shift="From legal guilt to religious transgression",
    ),
]


# =============================================================================
# Probe Text Generation
# =============================================================================


def generate_etymology_probe(
    etymology: EtymologyEntry,
    domain: ConceptDomain,
    probe_index: int,
) -> ConceptualGenealogyProbe:
    """Generate a probe from an etymology entry."""

    templates = [
        # Direct etymology statement
        f"The word '{etymology.modern_word}' comes from {etymology.origin_language.value} "
        f"'{etymology.origin_word}', meaning '{etymology.original_meaning}'.",
        # Semantic shift focus
        f"'{etymology.modern_word}' originally meant '{etymology.original_meaning}' "
        f"but now refers to something broader.",
        # Connection statement
        f"When we use '{etymology.modern_word}', we echo its {etymology.origin_language.value} "
        f"roots in '{etymology.origin_word}'.",
        # Evolution statement
        f"The concept of {etymology.modern_word} evolved from {etymology.original_meaning} "
        f"to its modern meaning.",
    ]

    probe_text = templates[probe_index % len(templates)]
    tests_drift = probe_index % 2 == 1  # Alternate probes test semantic drift

    return ConceptualGenealogyProbe(
        domain=domain,
        concept=etymology.modern_word,
        etymology=etymology,
        probe_text=probe_text,
        probe_id=f"genealogy_{domain.value}_{etymology.modern_word}_{probe_index}",
        tests_semantic_drift=tests_drift,
    )


def generate_domain_probes(domain: ConceptDomain) -> List[ConceptualGenealogyProbe]:
    """Generate all probes for a specific domain."""

    etymologies = {
        ConceptDomain.PHILOSOPHY: PHILOSOPHY_ETYMOLOGIES,
        ConceptDomain.SCIENCE: SCIENCE_ETYMOLOGIES,
        ConceptDomain.MATHEMATICS: MATHEMATICS_ETYMOLOGIES,
        ConceptDomain.POLITICS: POLITICS_ETYMOLOGIES,
        ConceptDomain.ETHICS: ETHICS_ETYMOLOGIES,
        ConceptDomain.AESTHETICS: [],  # Can be extended
    }

    probes = []
    for i, etymology in enumerate(etymologies.get(domain, [])):
        # Generate multiple probes per etymology
        for j in range(2):
            probe = generate_etymology_probe(etymology, domain, i * 2 + j)
            probes.append(probe)

    return probes


def generate_all_genealogy_probes(
    domains: Optional[set[ConceptDomain]] = None,
) -> List[ConceptualGenealogyProbe]:
    """
    Generate all conceptual genealogy probes.

    Args:
        domains: Set of domains to include (None = all)

    Returns:
        List of all genealogy probes
    """
    if domains is None:
        domains = set(ConceptDomain)

    all_probes = []
    for domain in domains:
        probes = generate_domain_probes(domain)
        all_probes.extend(probes)

    return all_probes


# =============================================================================
# Concept Lineage Anchors
# =============================================================================


@dataclass(frozen=True)
class ConceptLineageAnchor:
    """
    An anchor testing concept relationships across historical development.

    These test whether models understand how concepts evolved and influenced each other.
    """

    ancestor_concept: str
    descendant_concept: str
    relationship: str  # e.g., "evolved_into", "split_from", "merged_with"
    domain: ConceptDomain
    probe_text: str
    anchor_id: str


def generate_concept_lineage_anchors() -> List[ConceptLineageAnchor]:
    """Generate anchors testing concept lineage relationships."""
    anchors = []

    # Philosophy lineages
    philosophy_lineages = [
        ("natural philosophy", "physics", "evolved_into", "Natural philosophy became modern physics."),
        ("alchemy", "chemistry", "evolved_into", "Alchemy transformed into the science of chemistry."),
        ("rhetoric", "communication studies", "evolved_into", "Classical rhetoric evolved into communication studies."),
        ("logic", "computer science", "influenced", "Aristotelian logic influenced the foundations of computer science."),
    ]

    for i, (ancestor, descendant, rel, text) in enumerate(philosophy_lineages):
        anchors.append(
            ConceptLineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.PHILOSOPHY,
                probe_text=text,
                anchor_id=f"lineage_philosophy_{i}",
            )
        )

    # Science lineages
    science_lineages = [
        ("phlogiston theory", "oxidation theory", "replaced_by", "Phlogiston theory was replaced by oxidation."),
        ("atomism", "quantum mechanics", "evolved_into", "Ancient atomism evolved into quantum mechanics."),
        ("vitalism", "biochemistry", "replaced_by", "Vitalism gave way to biochemistry."),
    ]

    for i, (ancestor, descendant, rel, text) in enumerate(science_lineages):
        anchors.append(
            ConceptLineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.SCIENCE,
                probe_text=text,
                anchor_id=f"lineage_science_{i}",
            )
        )

    # Mathematics lineages
    math_lineages = [
        ("Euclidean geometry", "non-Euclidean geometry", "generalized_by",
         "Euclidean geometry was generalized by non-Euclidean systems."),
        ("arithmetic", "number theory", "formalized_into",
         "Basic arithmetic was formalized into number theory."),
    ]

    for i, (ancestor, descendant, rel, text) in enumerate(math_lineages):
        anchors.append(
            ConceptLineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.MATHEMATICS,
                probe_text=text,
                anchor_id=f"lineage_math_{i}",
            )
        )

    return anchors


@dataclass
class ConceptualGenealogyConfig:
    """Configuration for conceptual genealogy probing."""

    domains: Optional[set[ConceptDomain]] = None  # None = all
    include_lineage_anchors: bool = True
    languages: Optional[set[LanguageOrigin]] = None  # None = all
    genealogy_weight: float = 0.25  # Weight in intersection correlation
