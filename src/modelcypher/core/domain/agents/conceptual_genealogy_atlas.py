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
Conceptual Genealogy Atlas.

Multi-domain probes that encode etymology and concept lineage. These probes
capture how ideas evolve across time and culture, providing cross-domain
anchors for triangulation beyond surface semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
class GenealogyConcept:
    """A genealogy probe for triangulating historical concept structure."""

    id: str
    domain: ConceptDomain
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class LineageAnchor:
    """Anchor testing concept relationships across historical development."""

    ancestor_concept: str
    descendant_concept: str
    relationship: str
    domain: ConceptDomain
    probe_text: str


def _slugify(value: str) -> str:
    """Normalize identifiers for probe IDs."""
    cleaned: list[str] = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return "_".join(part for part in slug.split("_") if part)


def _etymology_support_texts(entry: EtymologyEntry) -> tuple[str, ...]:
    return (
        f"The word '{entry.modern_word}' comes from {entry.origin_language.value} "
        f"'{entry.origin_word}', meaning '{entry.original_meaning}'.",
        f"'{entry.modern_word}' originally meant '{entry.original_meaning}' "
        f"but now refers to something broader.",
        f"When we use '{entry.modern_word}', we echo its {entry.origin_language.value} "
        f"roots in '{entry.origin_word}'.",
        f"The concept of {entry.modern_word} evolved from {entry.original_meaning} "
        f"to its modern meaning.",
    )


def _build_etymology_probes(
    domain: ConceptDomain,
    entries: list[EtymologyEntry],
) -> list[GenealogyConcept]:
    probes: list[GenealogyConcept] = []
    for entry in entries:
        slug = _slugify(entry.modern_word)
        probes.append(
            GenealogyConcept(
                id=f"etymology_{domain.value}_{slug}",
                domain=domain,
                name=entry.modern_word,
                description=(
                    f"Etymology: {entry.origin_word} ({entry.origin_language.value}), "
                    f"{entry.original_meaning}; shift: {entry.semantic_shift}"
                ),
                support_texts=_etymology_support_texts(entry),
                cross_domain_weight=1.0,
            )
        )
    return probes


# =============================================================================
# Etymology Inventory
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


def _lineage_anchors() -> list[LineageAnchor]:
    anchors: list[LineageAnchor] = []

    philosophy_lineages = [
        (
            "natural philosophy",
            "physics",
            "evolved_into",
            "Natural philosophy became modern physics.",
        ),
        (
            "alchemy",
            "chemistry",
            "evolved_into",
            "Alchemy transformed into the science of chemistry.",
        ),
        (
            "rhetoric",
            "communication studies",
            "evolved_into",
            "Classical rhetoric evolved into communication studies.",
        ),
        (
            "logic",
            "computer science",
            "influenced",
            "Aristotelian logic influenced the foundations of computer science.",
        ),
    ]

    for ancestor, descendant, rel, text in philosophy_lineages:
        anchors.append(
            LineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.PHILOSOPHY,
                probe_text=text,
            )
        )

    science_lineages = [
        (
            "phlogiston theory",
            "oxidation theory",
            "replaced_by",
            "Phlogiston theory was replaced by oxidation.",
        ),
        (
            "atomism",
            "quantum mechanics",
            "evolved_into",
            "Ancient atomism evolved into quantum mechanics.",
        ),
        ("vitalism", "biochemistry", "replaced_by", "Vitalism gave way to biochemistry."),
    ]

    for ancestor, descendant, rel, text in science_lineages:
        anchors.append(
            LineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.SCIENCE,
                probe_text=text,
            )
        )

    math_lineages = [
        (
            "Euclidean geometry",
            "non-Euclidean geometry",
            "generalized_by",
            "Euclidean geometry was generalized by non-Euclidean systems.",
        ),
        (
            "arithmetic",
            "number theory",
            "formalized_into",
            "Basic arithmetic was formalized into number theory.",
        ),
    ]

    for ancestor, descendant, rel, text in math_lineages:
        anchors.append(
            LineageAnchor(
                ancestor_concept=ancestor,
                descendant_concept=descendant,
                relationship=rel,
                domain=ConceptDomain.MATHEMATICS,
                probe_text=text,
            )
        )

    return anchors


def _build_lineage_probes() -> list[GenealogyConcept]:
    probes: list[GenealogyConcept] = []
    for anchor in _lineage_anchors():
        ancestor_slug = _slugify(anchor.ancestor_concept)
        descendant_slug = _slugify(anchor.descendant_concept)
        probes.append(
            GenealogyConcept(
                id=f"lineage_{anchor.domain.value}_{ancestor_slug}_{descendant_slug}",
                domain=anchor.domain,
                name=f"{anchor.ancestor_concept} -> {anchor.descendant_concept}",
                description=f"Lineage: {anchor.ancestor_concept} {anchor.relationship} {anchor.descendant_concept}",
                support_texts=(anchor.probe_text,),
                cross_domain_weight=1.0,
            )
        )
    return probes


ALL_PROBES: tuple[GenealogyConcept, ...] = tuple(
    _build_etymology_probes(ConceptDomain.PHILOSOPHY, PHILOSOPHY_ETYMOLOGIES)
    + _build_etymology_probes(ConceptDomain.SCIENCE, SCIENCE_ETYMOLOGIES)
    + _build_etymology_probes(ConceptDomain.MATHEMATICS, MATHEMATICS_ETYMOLOGIES)
    + _build_etymology_probes(ConceptDomain.POLITICS, POLITICS_ETYMOLOGIES)
    + _build_etymology_probes(ConceptDomain.ETHICS, ETHICS_ETYMOLOGIES)
    + _build_lineage_probes()
)


class ConceptualGenealogyInventory:
    """Static inventory of conceptual genealogy probes."""

    @staticmethod
    def all_concepts() -> tuple[GenealogyConcept, ...]:
        return ALL_PROBES

    @staticmethod
    def concepts_by_domain(domain: ConceptDomain) -> list[GenealogyConcept]:
        return [probe for probe in ALL_PROBES if probe.domain == domain]

    @staticmethod
    def probe_count_by_domain() -> dict[ConceptDomain, int]:
        counts: dict[ConceptDomain, int] = {}
        for probe in ALL_PROBES:
            counts[probe.domain] = counts.get(probe.domain, 0) + 1
        return counts


__all__ = [
    "ConceptDomain",
    "LanguageOrigin",
    "EtymologyEntry",
    "GenealogyConcept",
    "LineageAnchor",
    "ConceptualGenealogyInventory",
]
