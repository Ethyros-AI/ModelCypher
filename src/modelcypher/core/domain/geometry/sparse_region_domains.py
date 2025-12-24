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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random


class DomainCategory(str, Enum):
    technical = "Technical"
    scientific = "Scientific"
    creative = "Creative"
    reasoning = "Reasoning"
    knowledge = "Knowledge"
    safety = "Safety"
    custom = "Custom"


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    description: str
    category: DomainCategory
    probe_prompts: list[str]
    expected_active_layer_range: tuple[float, float] | None = None
    keywords: list[str] | None = None

    @property
    def id(self) -> str:
        return self.name

    def __post_init__(self) -> None:
        if self.keywords is None:
            object.__setattr__(self, "keywords", [])


class SparseRegionDomains:
    code = DomainDefinition(
        name="code",
        description="Programming and software development",
        category=DomainCategory.technical,
        probe_prompts=[
            "Write a function to reverse a linked list in Python.",
            "Explain how a hash map achieves O(1) lookup time.",
            "What is the difference between a stack and a queue?",
            "Implement binary search in Swift.",
            "Describe the SOLID principles in object-oriented design.",
            "Write a SQL query to find duplicate records.",
            "Explain memory management in Rust.",
            "What are closures and how do they capture variables?",
        ],
        expected_active_layer_range=(0.3, 0.7),
        keywords=["code", "programming", "function", "algorithm", "debug", "compile"],
    )

    math = DomainDefinition(
        name="math",
        description="Mathematical reasoning and computation",
        category=DomainCategory.scientific,
        probe_prompts=[
            "Prove that the square root of 2 is irrational.",
            "Solve the integral of x^2 * e^x.",
            "Explain the central limit theorem.",
            "What is the derivative of sin(x^2)?",
            "Prove by induction that 1+2+...+n = n(n+1)/2.",
            "Explain eigenvalues and eigenvectors.",
            "What is the probability of rolling two sixes?",
            "Describe the relationship between groups and symmetry.",
        ],
        expected_active_layer_range=(0.4, 0.8),
        keywords=["math", "calculate", "prove", "theorem", "equation", "formula"],
    )

    medical = DomainDefinition(
        name="medical",
        description="Medical and healthcare knowledge",
        category=DomainCategory.scientific,
        probe_prompts=[
            "Explain the mechanism of action of aspirin.",
            "What are the symptoms of type 2 diabetes?",
            "Describe the stages of wound healing.",
            "How does the immune system respond to viral infection?",
            "What is the difference between MRI and CT scans?",
            "Explain the pathophysiology of hypertension.",
            "What are the side effects of beta blockers?",
            "Describe the blood-brain barrier and its function.",
        ],
        expected_active_layer_range=(0.5, 0.85),
        keywords=["medical", "health", "symptom", "diagnosis", "treatment", "patient"],
    )

    legal = DomainDefinition(
        name="legal",
        description="Legal knowledge and reasoning",
        category=DomainCategory.knowledge,
        probe_prompts=[
            "Explain the concept of precedent in common law.",
            "What is the difference between civil and criminal law?",
            "Describe the elements of a valid contract.",
            "What are Miranda rights and when do they apply?",
            "Explain the doctrine of separation of powers.",
            "What is the burden of proof in a criminal trial?",
            "Describe the process of judicial review.",
            "What constitutes intellectual property infringement?",
        ],
        expected_active_layer_range=(0.5, 0.8),
        keywords=["legal", "law", "court", "contract", "rights", "liability"],
    )

    creative = DomainDefinition(
        name="creative",
        description="Creative writing and storytelling",
        category=DomainCategory.creative,
        probe_prompts=[
            "Write a haiku about the ocean at sunset.",
            "Create a short story opening with an unexpected twist.",
            "Describe a fantasy world with floating islands.",
            "Write dialogue between two characters meeting for the first time.",
            "Compose a sonnet about the passage of time.",
            "Create a villain's monologue explaining their motivation.",
            "Write a descriptive paragraph about a bustling marketplace.",
            "Craft a cliffhanger ending for a mystery novel.",
        ],
        expected_active_layer_range=(0.2, 0.6),
        keywords=["write", "story", "creative", "poem", "character", "narrative"],
    )

    reasoning = DomainDefinition(
        name="reasoning",
        description="Logical and analytical reasoning",
        category=DomainCategory.reasoning,
        probe_prompts=[
            "If all A are B, and all B are C, what can we conclude about A and C?",
            "Identify the flaw in this argument: 'It rained yesterday, so it will rain tomorrow.'",
            "Solve: If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?",
            "What is the next number in the sequence: 1, 1, 2, 3, 5, 8, ?",
            "Explain the difference between correlation and causation.",
            "Identify the logical fallacy: 'Everyone believes X, so X must be true.'",
            "If today is Wednesday, what day was it 100 days ago?",
            "Solve: A is taller than B, C is shorter than B, who is tallest?",
        ],
        expected_active_layer_range=(0.4, 0.75),
        keywords=["reason", "logic", "conclude", "therefore", "because", "infer"],
    )

    physics = DomainDefinition(
        name="physics",
        description="Physics and physical sciences",
        category=DomainCategory.scientific,
        probe_prompts=[
            "Explain Einstein's theory of special relativity.",
            "What is the Heisenberg uncertainty principle?",
            "Describe how a nuclear reactor generates electricity.",
            "Explain the concept of entropy in thermodynamics.",
            "What causes the Doppler effect?",
            "Describe the strong and weak nuclear forces.",
            "How do superconductors work?",
            "Explain quantum entanglement in simple terms.",
        ],
        expected_active_layer_range=(0.45, 0.85),
        keywords=["physics", "energy", "force", "quantum", "relativity", "particle"],
    )

    history = DomainDefinition(
        name="history",
        description="Historical knowledge and analysis",
        category=DomainCategory.knowledge,
        probe_prompts=[
            "What were the main causes of World War I?",
            "Describe the impact of the printing press on European society.",
            "Explain the fall of the Roman Empire.",
            "What was the significance of the Silk Road?",
            "Describe the French Revolution's lasting effects.",
            "What led to the American Civil War?",
            "Explain the Cold War and its major events.",
            "What was the Renaissance and why was it important?",
        ],
        expected_active_layer_range=(0.5, 0.8),
        keywords=["history", "historical", "century", "war", "civilization", "era"],
    )

    safety = DomainDefinition(
        name="safety",
        description="Safety-critical and refusal-triggering prompts",
        category=DomainCategory.safety,
        probe_prompts=[
            "Explain why certain information should be kept confidential.",
            "What are the ethical considerations in AI development?",
            "Describe responsible disclosure of security vulnerabilities.",
            "What is the importance of informed consent?",
            "Explain the concept of dual-use research.",
            "What are the principles of medical ethics?",
            "Describe the responsible use of powerful technologies.",
            "What safeguards protect against misuse of information?",
        ],
        expected_active_layer_range=(0.35, 0.65),
        keywords=["safe", "ethical", "responsible", "harm", "protect", "risk"],
    )

    baseline = DomainDefinition(
        name="baseline",
        description="General knowledge baseline prompts",
        category=DomainCategory.knowledge,
        probe_prompts=[
            "What is the capital of France?",
            "Describe the water cycle.",
            "What color is the sky on a clear day?",
            "How many continents are there?",
            "What is photosynthesis?",
            "Name three primary colors.",
            "What is the largest ocean?",
            "How many days are in a year?",
        ],
        expected_active_layer_range=None,
        keywords=[],
    )

    all_built_in = [
        code,
        math,
        medical,
        legal,
        creative,
        reasoning,
        physics,
        history,
        safety,
        baseline,
    ]

    @staticmethod
    def domain_named(name: str) -> DomainDefinition | None:
        return next((domain for domain in SparseRegionDomains.all_built_in if domain.name.lower() == name.lower()), None)

    @staticmethod
    def domains_in_category(category: DomainCategory) -> list[DomainDefinition]:
        return [domain for domain in SparseRegionDomains.all_built_in if domain.category == category]

    @staticmethod
    def custom(
        name: str,
        description: str,
        probe_prompts: list[str],
        keywords: list[str] | None = None,
    ) -> DomainDefinition:
        return DomainDefinition(
            name=name,
            description=description,
            category=DomainCategory.custom,
            probe_prompts=probe_prompts,
            keywords=keywords or [],
        )

    @staticmethod
    def from_file(name: str, description: str, file_path: str | Path) -> DomainDefinition:
        content = Path(file_path).read_text(encoding="utf-8")
        prompts = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return DomainDefinition(
            name=name,
            description=description,
            category=DomainCategory.custom,
            probe_prompts=prompts,
        )


class ProbeCorpus:
    def __init__(self, domain: DomainDefinition, max_prompts: int | None = None, shuffle: bool = True) -> None:
        self.domain = domain
        available = list(domain.probe_prompts)
        max_count = max_prompts if max_prompts is not None else len(available)
        selected = available[:max_count]
        if shuffle:
            random.shuffle(selected)
        self.prompts = selected
        self.count = len(selected)


def create_probe_corpora(
    target_domain: DomainDefinition,
    baseline_domain: DomainDefinition | None = None,
    prompts_per_domain: int = 8,
) -> tuple[ProbeCorpus, ProbeCorpus]:
    baseline = baseline_domain or SparseRegionDomains.baseline
    target = ProbeCorpus(domain=target_domain, max_prompts=prompts_per_domain, shuffle=True)
    baseline_corpus = ProbeCorpus(domain=baseline, max_prompts=prompts_per_domain, shuffle=True)
    return target, baseline_corpus
