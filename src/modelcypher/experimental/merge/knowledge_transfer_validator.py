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
Knowledge Transfer Validation for Model Merging.

Verifies that merged models actually retain knowledge from source models
by running targeted probes and comparing outputs.

Integrates with:
- ProbeCorpus: Standard prompt sets across domains
- CompositionalProbes: Semantic compositionality tests
- ConceptResponseMatrix: Layer-wise anchor activations
- MergeValidationService: Perplexity and coherence scoring
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class KnowledgeDomain(str, Enum):
    """Domains for knowledge validation."""

    MATH = "math"
    CODE = "code"
    FACTUAL = "factual"
    REASONING = "reasoning"
    LANGUAGE = "language"
    CREATIVE = "creative"


# ValidationStatus enum removed - expose raw retention metrics only.


# =============================================================================
# Knowledge Probes
# =============================================================================


@dataclass(frozen=True)
class KnowledgeProbe:
    """A question with expected answer pattern for knowledge validation.

    Attributes
    ----------
    id : str
        Unique identifier for this probe.
    domain : KnowledgeDomain
        Knowledge domain being tested.
    prompt : str
        The question/prompt to send to the model.
    expected_pattern : str
        Regex pattern or exact substring expected in response.
    is_regex : bool
        Whether expected_pattern is a regex (True) or exact match (False).
    variations : tuple of str
        Alternative phrasings of the same question.
    difficulty : str
        Probe difficulty: easy, medium, hard.
    """

    id: str
    domain: KnowledgeDomain
    prompt: str
    expected_pattern: str
    is_regex: bool = True
    variations: tuple[str, ...] = field(default_factory=tuple)
    difficulty: str = "medium"

    def matches(self, response: str) -> bool:
        """Check if response matches expected pattern.

        Parameters
        ----------
        response : str
            Model response to check.

        Returns
        -------
        bool
            True if response matches expected pattern.
        """
        response_lower = response.lower().strip()
        pattern_lower = self.expected_pattern.lower()

        if self.is_regex:
            return bool(re.search(pattern_lower, response_lower))
        else:
            return pattern_lower in response_lower


# =============================================================================
# Probe Corpus
# =============================================================================


class KnowledgeProbeCorpus:
    """Collection of knowledge probes organized by domain."""

    def __init__(self):
        self._probes: dict[KnowledgeDomain, list[KnowledgeProbe]] = {
            domain: [] for domain in KnowledgeDomain
        }
        self._probe_index: dict[str, KnowledgeProbe] = {}  # O(1) lookup by ID
        self._load_default_probes()

    def _load_default_probes(self):
        """Load default knowledge probes."""
        # Math probes
        self._probes[KnowledgeDomain.MATH].extend(
            [
                KnowledgeProbe(
                    id="math_001",
                    domain=KnowledgeDomain.MATH,
                    prompt="What is 15 * 17?",
                    expected_pattern=r"255",
                    variations=("Calculate 15 times 17", "Multiply 15 by 17"),
                ),
                KnowledgeProbe(
                    id="math_002",
                    domain=KnowledgeDomain.MATH,
                    prompt="What is the square root of 144?",
                    expected_pattern=r"12",
                    variations=("sqrt(144) = ?", "The square root of 144 is"),
                ),
                KnowledgeProbe(
                    id="math_003",
                    domain=KnowledgeDomain.MATH,
                    prompt="What is the derivative of x^2?",
                    expected_pattern=r"2x",
                    variations=(
                        "d/dx of x squared",
                        "Differentiate x^2",
                    ),
                ),
                KnowledgeProbe(
                    id="math_004",
                    domain=KnowledgeDomain.MATH,
                    prompt="What is the integral of 2x?",
                    expected_pattern=r"x\^?2|x²",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="math_005",
                    domain=KnowledgeDomain.MATH,
                    prompt="What is 7 factorial (7!)?",
                    expected_pattern=r"5040",
                ),
            ]
        )

        # Code probes
        self._probes[KnowledgeDomain.CODE].extend(
            [
                KnowledgeProbe(
                    id="code_001",
                    domain=KnowledgeDomain.CODE,
                    prompt="What does 'def' keyword mean in Python?",
                    expected_pattern=r"function|define|declaration",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="code_002",
                    domain=KnowledgeDomain.CODE,
                    prompt="What is the time complexity of binary search?",
                    expected_pattern=r"O\(log\s*n\)|log.*n|logarithmic",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="code_003",
                    domain=KnowledgeDomain.CODE,
                    prompt="What is a linked list?",
                    expected_pattern=r"node|pointer|next|element",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="code_004",
                    domain=KnowledgeDomain.CODE,
                    prompt="What does SQL stand for?",
                    expected_pattern=r"structured query language",
                    is_regex=False,
                ),
                KnowledgeProbe(
                    id="code_005",
                    domain=KnowledgeDomain.CODE,
                    prompt="What is recursion in programming?",
                    expected_pattern=r"call.*itself|function.*itself|self.*call",
                    is_regex=True,
                ),
            ]
        )

        # Factual probes
        self._probes[KnowledgeDomain.FACTUAL].extend(
            [
                KnowledgeProbe(
                    id="fact_001",
                    domain=KnowledgeDomain.FACTUAL,
                    prompt="What is the capital of France?",
                    expected_pattern=r"paris",
                    is_regex=False,
                ),
                KnowledgeProbe(
                    id="fact_002",
                    domain=KnowledgeDomain.FACTUAL,
                    prompt="What is the chemical symbol for gold?",
                    expected_pattern=r"\bau\b",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="fact_003",
                    domain=KnowledgeDomain.FACTUAL,
                    prompt="Who wrote Romeo and Juliet?",
                    expected_pattern=r"shakespeare",
                    is_regex=False,
                ),
                KnowledgeProbe(
                    id="fact_004",
                    domain=KnowledgeDomain.FACTUAL,
                    prompt="What is the speed of light in vacuum?",
                    expected_pattern=r"3.*10\^?8|300.*000|299",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="fact_005",
                    domain=KnowledgeDomain.FACTUAL,
                    prompt="What is the largest planet in our solar system?",
                    expected_pattern=r"jupiter",
                    is_regex=False,
                ),
            ]
        )

        # Reasoning probes
        self._probes[KnowledgeDomain.REASONING].extend(
            [
                KnowledgeProbe(
                    id="reason_001",
                    domain=KnowledgeDomain.REASONING,
                    prompt="If all cats are animals, and Whiskers is a cat, is Whiskers an animal?",
                    expected_pattern=r"yes|animal",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="reason_002",
                    domain=KnowledgeDomain.REASONING,
                    prompt="If A > B and B > C, what is the relationship between A and C?",
                    expected_pattern=r"A.*>.*C|A.*greater.*C|A.*larger.*C",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="reason_003",
                    domain=KnowledgeDomain.REASONING,
                    prompt="Complete the pattern: 2, 4, 6, 8, ?",
                    expected_pattern=r"10",
                ),
                KnowledgeProbe(
                    id="reason_004",
                    domain=KnowledgeDomain.REASONING,
                    prompt="If today is Monday, what day was it 3 days ago?",
                    expected_pattern=r"friday",
                    is_regex=False,
                ),
                KnowledgeProbe(
                    id="reason_005",
                    domain=KnowledgeDomain.REASONING,
                    prompt="Which is heavier: a kilogram of steel or a kilogram of feathers?",
                    expected_pattern=r"same|equal|both|neither|weigh.*same",
                    is_regex=True,
                ),
            ]
        )

        # Language probes
        self._probes[KnowledgeDomain.LANGUAGE].extend(
            [
                KnowledgeProbe(
                    id="lang_001",
                    domain=KnowledgeDomain.LANGUAGE,
                    prompt="What is an antonym of 'happy'?",
                    expected_pattern=r"sad|unhappy|miserable|depressed",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="lang_002",
                    domain=KnowledgeDomain.LANGUAGE,
                    prompt="What part of speech is the word 'quickly'?",
                    expected_pattern=r"adverb",
                    is_regex=False,
                ),
                KnowledgeProbe(
                    id="lang_003",
                    domain=KnowledgeDomain.LANGUAGE,
                    prompt="What is a synonym for 'big'?",
                    expected_pattern=r"large|huge|enormous|massive|giant",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="lang_004",
                    domain=KnowledgeDomain.LANGUAGE,
                    prompt="Correct this sentence: 'She don't like apples.'",
                    expected_pattern=r"doesn't|does not",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="lang_005",
                    domain=KnowledgeDomain.LANGUAGE,
                    prompt="What is the plural of 'child'?",
                    expected_pattern=r"children",
                    is_regex=False,
                ),
            ]
        )

        # Creative probes (structure-based matching)
        self._probes[KnowledgeDomain.CREATIVE].extend(
            [
                KnowledgeProbe(
                    id="creative_001",
                    domain=KnowledgeDomain.CREATIVE,
                    prompt="Complete this simile: 'As brave as a...'",
                    expected_pattern=r"lion|soldier|warrior|hero",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="creative_002",
                    domain=KnowledgeDomain.CREATIVE,
                    prompt="What rhymes with 'cat'?",
                    expected_pattern=r"bat|hat|mat|rat|sat|flat|that",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="creative_003",
                    domain=KnowledgeDomain.CREATIVE,
                    prompt="Give an example of alliteration.",
                    expected_pattern=r"\b(\w)\w*\s+\1\w*",  # Two words starting with same letter
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="creative_004",
                    domain=KnowledgeDomain.CREATIVE,
                    prompt="What is a metaphor?",
                    expected_pattern=r"comparison|figure.*speech|represent|symbol",
                    is_regex=True,
                ),
                KnowledgeProbe(
                    id="creative_005",
                    domain=KnowledgeDomain.CREATIVE,
                    prompt="Name a famous fictional detective.",
                    expected_pattern=r"sherlock|poirot|marple|bond|columbo",
                    is_regex=True,
                ),
            ]
        )

        # Build O(1) lookup index from all loaded probes
        for probes in self._probes.values():
            for probe in probes:
                self._probe_index[probe.id] = probe

    def get_probes(self, domain: KnowledgeDomain | None = None) -> list[KnowledgeProbe]:
        """Get probes, optionally filtered by domain.

        Parameters
        ----------
        domain : KnowledgeDomain, optional
            Domain to filter by. If None, returns all probes.

        Returns
        -------
        list of KnowledgeProbe
            Probes matching the filter.
        """
        if domain:
            return self._probes.get(domain, [])
        return [probe for probes in self._probes.values() for probe in probes]

    def get_probe_by_id(self, probe_id: str) -> KnowledgeProbe | None:
        """Get a specific probe by ID in O(1) time.

        Parameters
        ----------
        probe_id : str
            Probe identifier.

        Returns
        -------
        KnowledgeProbe or None
            Probe if found, None otherwise.
        """
        return self._probe_index.get(probe_id)

    def add_probe(self, probe: KnowledgeProbe) -> None:
        """Add a custom probe.

        Parameters
        ----------
        probe : KnowledgeProbe
            Probe to add to the corpus.
        """
        self._probes[probe.domain].append(probe)
        self._probe_index[probe.id] = probe

    @property
    def domain_counts(self) -> dict[KnowledgeDomain, int]:
        """Get probe counts per domain."""
        return {domain: len(probes) for domain, probes in self._probes.items()}


# =============================================================================
# Results
# =============================================================================


@dataclass
class KnowledgeProbeResult:
    """Result of running a single knowledge probe.

    Attributes
    ----------
    probe_id : str
        Unique identifier for this probe.
    domain : KnowledgeDomain
        Knowledge domain tested.
    prompt : str
        The prompt sent to the model.
    response : str
        Model response text.
    expected_pattern : str
        Pattern expected in response.
    passed : bool
        Whether the main prompt matched.
    variation_results : dict
        Results for each variation: variation_prompt -> passed.
    """

    probe_id: str
    domain: KnowledgeDomain
    prompt: str
    response: str
    expected_pattern: str
    passed: bool
    variation_results: dict[str, bool] = field(default_factory=dict)

    @property
    def variation_pass_rate(self) -> float:
        """Pass rate across variations."""
        if not self.variation_results:
            return 1.0 if self.passed else 0.0
        total = 1 + len(self.variation_results)  # main + variations
        passed = (1 if self.passed else 0) + sum(self.variation_results.values())
        return passed / total


@dataclass
class KnowledgeRetentionResult:
    """Per-domain knowledge retention metrics.

    Attributes
    ----------
    domain : KnowledgeDomain
        Knowledge domain measured.
    source_pass_rate : float
        Baseline pass rate from source model.
    merged_pass_rate : float
        Pass rate on merged model.
    probes_tested : int
        Number of probes tested in this domain.
    passed_probes : list of str
        IDs of probes that passed.
    failed_probes : list of str
        IDs of probes that failed.
    """

    domain: KnowledgeDomain
    source_pass_rate: float
    merged_pass_rate: float

    @property
    def retention_score(self) -> float:
        """Retention = merged / source (capped at 1.0)."""
        _b = get_default_backend()
        eps = division_epsilon(_b, _b.array([1.0], dtype=precision_dtype(_b)))
        if self.source_pass_rate <= eps:
            return 1.0  # Avoid division by zero/near-zero
        return min(1.0, self.merged_pass_rate / self.source_pass_rate)

    probes_tested: int = 0
    passed_probes: list[str] = field(default_factory=list)
    failed_probes: list[str] = field(default_factory=list)

@dataclass
class KnowledgeTransferReport:
    """Comprehensive post-merge knowledge validation report.

    Attributes
    ----------
    per_domain : dict
        Results broken down by domain.
    probe_results : list of KnowledgeProbeResult
        Individual probe results.
    compositional_consistency : float
        Consistency of semantic compositions.
    crm_correlation : float
        CRM similarity between source and merged model.
    """

    per_domain: dict[KnowledgeDomain, KnowledgeRetentionResult]
    probe_results: list[KnowledgeProbeResult] = field(default_factory=list)

    @property
    def overall_retention(self) -> float:
        """Weighted average retention across domains."""
        if not self.per_domain:
            return 0.0

        total_probes = sum(r.probes_tested for r in self.per_domain.values())
        if total_probes == 0:
            return 0.0

        weighted_sum = sum(r.retention_score * r.probes_tested for r in self.per_domain.values())
        return weighted_sum / total_probes

    @property
    def overall_pass_rate(self) -> float:
        """Overall pass rate on merged model."""
        if not self.probe_results:
            return 0.0
        passed = sum(1 for r in self.probe_results if r.passed)
        return passed / len(self.probe_results)

    # NOTE: The status property was removed to comply with "no vibes" principle.
    # Validation returns raw measurements (overall_retention); callers decide interpretation.

    compositional_consistency: float = 0.0
    """Consistency of semantic compositions (from CompositionalProbes)."""

    crm_correlation: float = 0.0
    """CRM similarity between source and merged model."""

    def summary(self) -> dict[str, any]:
        """Get summary dict for JSON output.

        Returns
        -------
        dict
            Summary dictionary with raw retention metrics.
        """
        return {
            "overall_retention": round(self.overall_retention, 4),
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "compositional_consistency": round(self.compositional_consistency, 4),
            "crm_correlation": round(self.crm_correlation, 4),
            "domain_retention": {
                domain.value: round(result.retention_score, 4)
                for domain, result in self.per_domain.items()
            },
            "total_probes": len(self.probe_results),
            "passed_probes": sum(1 for r in self.probe_results if r.passed),
            "failed_probes": sum(1 for r in self.probe_results if not r.passed),
        }


# =============================================================================
# Validation Functions
# =============================================================================


def run_knowledge_probes(
    generate_fn: Callable[[str], str],
    probes: list[KnowledgeProbe],
) -> list[KnowledgeProbeResult]:
    """Run knowledge probes against a model.

    Parameters
    ----------
    generate_fn : callable
        Function that takes a prompt and returns model response.
    probes : list of KnowledgeProbe
        List of probes to run.

    Returns
    -------
    list of KnowledgeProbeResult
        List of KnowledgeProbeResult for each probe.
    """
    results = []

    for probe in probes:
        # Run main prompt
        response = generate_fn(probe.prompt)
        passed = probe.matches(response)

        # Run variations for the full probe set
        variation_results = {}
        if probe.variations:
            for variation in probe.variations:
                var_response = generate_fn(variation)
                variation_results[variation] = probe.matches(var_response)

        results.append(
            KnowledgeProbeResult(
                probe_id=probe.id,
                domain=probe.domain,
                prompt=probe.prompt,
                response=response,
                expected_pattern=probe.expected_pattern,
                passed=passed,
                variation_results=variation_results,
            )
        )

    return results


def compute_retention_by_domain(
    source_results: list[KnowledgeProbeResult],
    merged_results: list[KnowledgeProbeResult],
) -> dict[KnowledgeDomain, KnowledgeRetentionResult]:
    """Compute per-domain retention from probe results.

    Parameters
    ----------
    source_results : list of KnowledgeProbeResult
        Results from running probes on source model.
    merged_results : list of KnowledgeProbeResult
        Results from running probes on merged model.

    Returns
    -------
    dict
        Dict mapping domain to retention result.
    """
    # Group by domain
    source_by_domain: dict[KnowledgeDomain, list[KnowledgeProbeResult]] = {}
    merged_by_domain: dict[KnowledgeDomain, list[KnowledgeProbeResult]] = {}

    for result in source_results:
        source_by_domain.setdefault(result.domain, []).append(result)
    for result in merged_results:
        merged_by_domain.setdefault(result.domain, []).append(result)

    retention: dict[KnowledgeDomain, KnowledgeRetentionResult] = {}

    for domain in KnowledgeDomain:
        source_probes = source_by_domain.get(domain, [])
        merged_probes = merged_by_domain.get(domain, [])

        if not source_probes or not merged_probes:
            continue

        # Single pass over source probes
        source_passed = sum(1 for r in source_probes if r.passed)
        source_pass_rate = source_passed / len(source_probes)

        # Single pass over merged probes - collect all stats at once
        passed_ids: list[str] = []
        failed_ids: list[str] = []
        for r in merged_probes:
            if r.passed:
                passed_ids.append(r.probe_id)
            else:
                failed_ids.append(r.probe_id)

        merged_pass_rate = len(passed_ids) / len(merged_probes)

        retention[domain] = KnowledgeRetentionResult(
            domain=domain,
            source_pass_rate=source_pass_rate,
            merged_pass_rate=merged_pass_rate,
            probes_tested=len(merged_probes),
            passed_probes=passed_ids,
            failed_probes=failed_ids,
        )

    return retention


def validate_knowledge_transfer(
    source_generate_fn: Callable[[str], str],
    merged_generate_fn: Callable[[str], str],
    corpus: KnowledgeProbeCorpus | None = None,
) -> KnowledgeTransferReport:
    """Run full knowledge transfer validation.

    Parameters
    ----------
    source_generate_fn : callable
        Generation function for source model.
    merged_generate_fn : callable
        Generation function for merged model.
    corpus : KnowledgeProbeCorpus, optional
        Probe corpus (uses default if not provided).

    Returns
    -------
    KnowledgeTransferReport
        Comprehensive knowledge transfer report.
    """
    probe_corpus = corpus or KnowledgeProbeCorpus()

    # Use the full probe corpus to avoid arbitrary truncation
    probes = probe_corpus.get_probes()

    logger.info(f"Running {len(probes)} knowledge probes across all domains")

    # Run probes on source model
    source_results = run_knowledge_probes(source_generate_fn, probes)

    # Run probes on merged model
    merged_results = run_knowledge_probes(merged_generate_fn, probes)

    # Compute retention
    per_domain = compute_retention_by_domain(source_results, merged_results)

    return KnowledgeTransferReport(
        per_domain=per_domain,
        probe_results=merged_results,
        compositional_consistency=0.0,  # To be filled by service
        crm_correlation=0.0,  # To be filled by service
    )
