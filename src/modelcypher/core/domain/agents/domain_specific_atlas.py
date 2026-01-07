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
Domain-Specific Atlas - Technical terms for cross-dimensional alignment.

Covers specialized vocabulary from major domains:
- Science (physics, chemistry, biology)
- Medicine (health, treatment, diagnosis)
- Law (legal terms, rights, contracts)
- Technology (software, hardware, data)
- Business (commerce, finance, markets)
- Education (learning, teaching, assessment)

Total: ~60 probes for domain-specific grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DomainCategory(str, Enum):
    """Category of domain-specific term."""
    
    SCIENCE = "science"
    MEDICINE = "medicine"
    LAW = "law"
    TECHNOLOGY_DOMAIN = "technology_domain"  # Avoid conflict with category name
    BUSINESS = "business"
    EDUCATION = "education"


@dataclass(frozen=True)
class DomainConcept:
    """A domain-specific concept."""
    
    id: str
    name: str
    category: DomainCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class DomainSpecificInventory:
    """Inventory of domain-specific concepts."""
    
    # Science terms
    SCIENCE = (
        DomainConcept("atom", "atom", DomainCategory.SCIENCE, "Basic matter unit", ("atom", "atomic", "particle")),
        DomainConcept("molecule", "molecule", DomainCategory.SCIENCE, "Bonded atoms", ("molecule", "molecular", "compound")),
        DomainConcept("cell", "cell", DomainCategory.SCIENCE, "Life unit", ("cell", "cellular", "organism")),
        DomainConcept("gene", "gene", DomainCategory.SCIENCE, "Heredity unit", ("gene", "genetic", "DNA")),
        DomainConcept("energy", "energy", DomainCategory.SCIENCE, "Capacity for work", ("energy", "power", "force")),
        DomainConcept("force", "force", DomainCategory.SCIENCE, "Push or pull", ("force", "pressure", "thrust")),
        DomainConcept("mass", "mass", DomainCategory.SCIENCE, "Amount of matter", ("mass", "weight", "density")),
        DomainConcept("gravity", "gravity", DomainCategory.SCIENCE, "Attractive force", ("gravity", "gravitational", "attraction")),
        DomainConcept("evolution", "evolution", DomainCategory.SCIENCE, "Species change", ("evolution", "evolutionary", "natural selection")),
        DomainConcept("experiment", "experiment", DomainCategory.SCIENCE, "Scientific test", ("experiment", "test", "trial")),
    )
    
    # Medicine terms
    MEDICINE = (
        DomainConcept("disease", "disease", DomainCategory.MEDICINE, "Health disorder", ("disease", "illness", "sickness")),
        DomainConcept("symptom", "symptom", DomainCategory.MEDICINE, "Disease sign", ("symptom", "sign", "indication")),
        DomainConcept("treatment", "treatment", DomainCategory.MEDICINE, "Medical care", ("treatment", "therapy", "cure")),
        DomainConcept("diagnosis", "diagnosis", DomainCategory.MEDICINE, "Disease identification", ("diagnosis", "diagnostic", "assessment")),
        DomainConcept("health", "health", DomainCategory.MEDICINE, "Body condition", ("health", "healthy", "wellness")),
        DomainConcept("medicine", "medicine", DomainCategory.MEDICINE, "Healing substance", ("medicine", "drug", "medication")),
        DomainConcept("surgery", "surgery", DomainCategory.MEDICINE, "Medical operation", ("surgery", "operation", "procedure")),
        DomainConcept("patient", "patient", DomainCategory.MEDICINE, "Medical recipient", ("patient", "sick person", "case")),
        DomainConcept("doctor", "doctor", DomainCategory.MEDICINE, "Medical practitioner", ("doctor", "physician", "practitioner")),
        DomainConcept("hospital", "hospital", DomainCategory.MEDICINE, "Medical facility", ("hospital", "clinic", "infirmary")),
    )
    
    # Law terms
    LAW = (
        DomainConcept("law", "law", DomainCategory.LAW, "Legal rule", ("law", "legal", "statute")),
        DomainConcept("right", "right", DomainCategory.LAW, "Legal entitlement", ("right", "entitlement", "privilege")),
        DomainConcept("contract", "contract", DomainCategory.LAW, "Legal agreement", ("contract", "agreement", "deal")),
        DomainConcept("court", "court", DomainCategory.LAW, "Legal venue", ("court", "tribunal", "bench")),
        DomainConcept("judge", "judge", DomainCategory.LAW, "Legal arbiter", ("judge", "justice", "magistrate")),
        DomainConcept("crime", "crime", DomainCategory.LAW, "Legal violation", ("crime", "offense", "violation")),
        DomainConcept("justice", "justice", DomainCategory.LAW, "Fairness principle", ("justice", "fairness", "equity")),
        DomainConcept("evidence", "evidence", DomainCategory.LAW, "Proof material", ("evidence", "proof", "testimony")),
        DomainConcept("verdict", "verdict", DomainCategory.LAW, "Legal decision", ("verdict", "judgment", "ruling")),
        DomainConcept("lawyer", "lawyer", DomainCategory.LAW, "Legal representative", ("lawyer", "attorney", "counsel")),
    )
    
    # Technology terms
    TECHNOLOGY = (
        DomainConcept("software", "software", DomainCategory.TECHNOLOGY_DOMAIN, "Computer programs", ("software", "program", "application")),
        DomainConcept("hardware", "hardware", DomainCategory.TECHNOLOGY_DOMAIN, "Physical devices", ("hardware", "equipment", "device")),
        DomainConcept("data", "data", DomainCategory.TECHNOLOGY_DOMAIN, "Information units", ("data", "information", "dataset")),
        DomainConcept("algorithm", "algorithm", DomainCategory.TECHNOLOGY_DOMAIN, "Computational procedure", ("algorithm", "procedure", "method")),
        DomainConcept("network", "network", DomainCategory.TECHNOLOGY_DOMAIN, "Connected systems", ("network", "connection", "internet")),
        DomainConcept("code", "code", DomainCategory.TECHNOLOGY_DOMAIN, "Program instructions", ("code", "programming", "source")),
        DomainConcept("database", "database", DomainCategory.TECHNOLOGY_DOMAIN, "Data storage", ("database", "storage", "repository")),
        DomainConcept("interface", "interface", DomainCategory.TECHNOLOGY_DOMAIN, "Interaction point", ("interface", "UI", "interaction")),
        DomainConcept("security", "security", DomainCategory.TECHNOLOGY_DOMAIN, "Protection system", ("security", "protection", "safety")),
        DomainConcept("cloud", "cloud", DomainCategory.TECHNOLOGY_DOMAIN, "Remote computing", ("cloud", "remote", "distributed")),
    )
    
    # Business terms
    BUSINESS = (
        DomainConcept("company", "company", DomainCategory.BUSINESS, "Business entity", ("company", "corporation", "firm")),
        DomainConcept("product", "product", DomainCategory.BUSINESS, "Sellable good", ("product", "goods", "merchandise")),
        DomainConcept("market", "market", DomainCategory.BUSINESS, "Trading space", ("market", "marketplace", "exchange")),
        DomainConcept("profit", "profit", DomainCategory.BUSINESS, "Financial gain", ("profit", "earnings", "revenue")),
        DomainConcept("investment", "investment", DomainCategory.BUSINESS, "Capital allocation", ("investment", "capital", "funding")),
        DomainConcept("trade", "trade", DomainCategory.BUSINESS, "Exchange activity", ("trade", "commerce", "transaction")),
        DomainConcept("customer", "customer", DomainCategory.BUSINESS, "Buyer of goods", ("customer", "client", "consumer")),
        DomainConcept("employee", "employee", DomainCategory.BUSINESS, "Company worker", ("employee", "worker", "staff")),
        DomainConcept("salary", "salary", DomainCategory.BUSINESS, "Worker payment", ("salary", "wage", "compensation")),
        DomainConcept("contract_business", "contract", DomainCategory.BUSINESS, "Business agreement", ("contract", "deal", "agreement")),
    )
    
    # Education terms
    EDUCATION = (
        DomainConcept("school", "school", DomainCategory.EDUCATION, "Learning institution", ("school", "academy", "institution")),
        DomainConcept("teacher", "teacher", DomainCategory.EDUCATION, "Educator", ("teacher", "instructor", "educator")),
        DomainConcept("student", "student", DomainCategory.EDUCATION, "Learner", ("student", "pupil", "learner")),
        DomainConcept("lesson", "lesson", DomainCategory.EDUCATION, "Teaching unit", ("lesson", "class", "lecture")),
        DomainConcept("exam", "exam", DomainCategory.EDUCATION, "Assessment", ("exam", "test", "examination")),
        DomainConcept("grade", "grade", DomainCategory.EDUCATION, "Performance measure", ("grade", "score", "mark")),
        DomainConcept("curriculum", "curriculum", DomainCategory.EDUCATION, "Study plan", ("curriculum", "syllabus", "program")),
        DomainConcept("homework", "homework", DomainCategory.EDUCATION, "Home assignment", ("homework", "assignment", "task")),
        DomainConcept("degree", "degree", DomainCategory.EDUCATION, "Academic credential", ("degree", "diploma", "certificate")),
        DomainConcept("research", "research", DomainCategory.EDUCATION, "Scholarly inquiry", ("research", "study", "investigation")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[DomainConcept]:
        """Get all domain-specific concepts."""
        concepts: list[DomainConcept] = []
        concepts.extend(cls.SCIENCE)
        concepts.extend(cls.MEDICINE)
        concepts.extend(cls.LAW)
        concepts.extend(cls.TECHNOLOGY)
        concepts.extend(cls.BUSINESS)
        concepts.extend(cls.EDUCATION)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: DomainCategory) -> list[DomainConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
