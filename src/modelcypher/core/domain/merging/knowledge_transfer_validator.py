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

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class KnowledgeDomain(str, Enum):
    """Domains for knowledge validation."""

    MATH = "math"
    CODE = "code"
    FACTUAL = "factual"
    REASONING = "reasoning"
    LANGUAGE = "language"
    CREATIVE = "creative"


# ValidationStatus enum removed - the overall_retention IS the validation state.
# Use status_for_thresholds() with caller-provided thresholds to classify.


@dataclass
class KnowledgeValidationConfig:
    """Configuration for knowledge validation.

    Attributes
    ----------
    retention_threshold_excellent : float
        Threshold for excellent status (e.g., 0.95 = 95% retention).
    retention_threshold_acceptable : float
        Threshold for acceptable status (e.g., 0.80 = 80% retention).
    retention_threshold_degraded : float
        Threshold for degraded status (e.g., 0.60 = 60% retention).
    domains : list of KnowledgeDomain
        Which domains to test.
    min_probes_per_domain : int
        Minimum number of probes per domain.
    use_variations : bool
        Whether to test paraphrased variations of prompts.
    max_response_length : int
        Maximum tokens for response generation.
    temperature : float
        Generation temperature (0 for deterministic).

    Notes
    -----
    All retention thresholds must be explicitly provided or derived from
    calibration data. Use from_calibration_data() to derive thresholds from
    historical retention scores, or from_standard_testing() for commonly-used
    values that should be explicitly acknowledged.
    """

    retention_threshold_excellent: float
    retention_threshold_acceptable: float
    retention_threshold_degraded: float

    domains: list[KnowledgeDomain] = field(default_factory=lambda: list(KnowledgeDomain))
    min_probes_per_domain: int = 5
    use_variations: bool = True
    max_response_length: int = 256
    temperature: float = 0.0

    @classmethod
    def from_calibration_data(
        cls,
        retention_scores: list[float],
        *,
        excellent_percentile: float = 0.95,
        acceptable_percentile: float = 0.70,
        degraded_percentile: float = 0.30,
        domains: list[KnowledgeDomain] | None = None,
    ) -> "KnowledgeValidationConfig":
        """Derive thresholds from historical retention score distribution.

        Parameters
        ----------
        retention_scores : list of float
            Historical retention scores from prior validations.
        excellent_percentile : float, optional
            Percentile for excellent threshold (top X%).
        acceptable_percentile : float, optional
            Percentile for acceptable threshold.
        degraded_percentile : float, optional
            Percentile for degraded threshold.
        domains : list of KnowledgeDomain, optional
            Domains to test (defaults to all).

        Returns
        -------
        KnowledgeValidationConfig
            Configuration with percentile-derived thresholds.
        """
        if not retention_scores:
            raise ValueError("retention_scores cannot be empty for calibration")

        sorted_scores = sorted(retention_scores)
        n = len(sorted_scores)

        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_scores[idx]

        return cls(
            retention_threshold_excellent=percentile(excellent_percentile),
            retention_threshold_acceptable=percentile(acceptable_percentile),
            retention_threshold_degraded=percentile(degraded_percentile),
            domains=domains if domains is not None else list(KnowledgeDomain),
        )

    @classmethod
    def from_baseline_variance(
        cls,
        retention_scores: list[float],
        *,
        domains: list[KnowledgeDomain] | None = None,
    ) -> "KnowledgeValidationConfig":
        """Derive thresholds from baseline retention score variance.

        Parameters
        ----------
        retention_scores : list of float
            Historical retention scores from prior validations.
        domains : list of KnowledgeDomain, optional
            Domains to test (defaults to all).

        Returns
        -------
        KnowledgeValidationConfig
            Configuration with variance-derived thresholds.

        Raises
        ------
        ValueError
            If retention_scores is empty.

        Notes
        -----
        Uses the mean and standard deviation of observed retention scores
        to define thresholds:
        - Excellent: mean (baseline performance)
        - Acceptable: mean - 1*std (one std below baseline)
        - Degraded: mean - 2*std (two stds below baseline)
        """
        if not retention_scores:
            raise ValueError("retention_scores cannot be empty for baseline derivation")

        n = len(retention_scores)
        mean = sum(retention_scores) / n
        variance = sum((x - mean) ** 2 for x in retention_scores) / n
        std = variance ** 0.5

        # Thresholds based on deviations from mean
        # Excellent = at or above mean
        # Acceptable = within 1 std of mean
        # Degraded = within 2 stds of mean
        # Failed = more than 2 stds below mean
        return cls(
            retention_threshold_excellent=mean,
            retention_threshold_acceptable=max(0.0, mean - std),
            retention_threshold_degraded=max(0.0, mean - 2 * std),
            domains=domains if domains is not None else list(KnowledgeDomain),
        )

    @classmethod
    def with_explicit_thresholds(
        cls,
        excellent: float,
        acceptable: float,
        degraded: float,
        *,
        domains: list[KnowledgeDomain] | None = None,
    ) -> "KnowledgeValidationConfig":
        """Create configuration with explicitly specified thresholds.

        Parameters
        ----------
        excellent : float
            Retention threshold for excellent status.
        acceptable : float
            Retention threshold for acceptable status.
        degraded : float
            Retention threshold for degraded status.
        domains : list of KnowledgeDomain, optional
            Domains to test (defaults to all).

        Returns
        -------
        KnowledgeValidationConfig
            Configuration with the specified thresholds.

        Notes
        -----
        Use this when you have domain-specific requirements for what
        constitutes acceptable knowledge retention. The caller must
        explicitly specify all thresholds to acknowledge they are
        making a deliberate choice.
        """
        return cls(
            retention_threshold_excellent=excellent,
            retention_threshold_acceptable=acceptable,
            retention_threshold_degraded=degraded,
            domains=domains if domains is not None else list(KnowledgeDomain),
        )

    @classmethod
    def from_standard_testing(
        cls,
        domains: list[KnowledgeDomain] | None = None,
    ) -> "KnowledgeValidationConfig":
        """Create configuration with standard testing thresholds.

        Parameters
        ----------
        domains : list of KnowledgeDomain, optional
            Domains to test (defaults to all).

        Returns
        -------
        KnowledgeValidationConfig
            Configuration with thresholds of 95%/80%/60%.

        Notes
        -----
        Returns thresholds of 95%/80%/60% for backward compatibility.
        For data-driven thresholds, consider using from_calibration_data()
        or from_baseline_variance() instead.
        """
        return cls.with_explicit_thresholds(
            excellent=0.95,
            acceptable=0.80,
            degraded=0.60,
            domains=domains,
        )


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
        """Get a specific probe by ID.

        Parameters
        ----------
        probe_id : str
            Probe identifier.

        Returns
        -------
        KnowledgeProbe or None
            Probe if found, None otherwise.
        """
        for probes in self._probes.values():
            for probe in probes:
                if probe.id == probe_id:
                    return probe
        return None

    def add_probe(self, probe: KnowledgeProbe) -> None:
        """Add a custom probe.

        Parameters
        ----------
        probe : KnowledgeProbe
            Probe to add to the corpus.
        """
        self._probes[probe.domain].append(probe)

    @property
    def domain_counts(self) -> dict[KnowledgeDomain, int]:
        """Get probe counts per domain."""
        return {domain: len(probes) for domain, probes in self._probes.items()}


# =============================================================================
# Results
# =============================================================================


@dataclass
class ProbeResult:
    """Result of running a single knowledge probe."""

    probe_id: str
    domain: KnowledgeDomain
    prompt: str
    response: str
    expected_pattern: str
    passed: bool
    variation_results: dict[str, bool] = field(default_factory=dict)
    """Results for each variation: variation_prompt -> passed."""

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
    """Per-domain knowledge retention metrics."""

    domain: KnowledgeDomain
    source_pass_rate: float
    """Baseline pass rate from source model."""

    merged_pass_rate: float
    """Pass rate on merged model."""

    @property
    def retention_score(self) -> float:
        """Retention = merged / source (capped at 1.0)."""
        if self.source_pass_rate < 0.01:
            return 1.0  # Avoid division by zero
        return min(1.0, self.merged_pass_rate / self.source_pass_rate)

    probes_tested: int = 0
    """Number of probes tested in this domain."""

    passed_probes: list[str] = field(default_factory=list)
    """IDs of probes that passed."""

    failed_probes: list[str] = field(default_factory=list)
    """IDs of probes that failed."""

    @property
    def degraded_probes(self) -> list[str]:
        """Alias for failed_probes for compatibility."""
        return self.failed_probes


@dataclass
class KnowledgeTransferReport:
    """Comprehensive post-merge knowledge validation report."""

    per_domain: dict[KnowledgeDomain, KnowledgeRetentionResult]
    """Results broken down by domain."""

    probe_results: list[ProbeResult] = field(default_factory=list)
    """Individual probe results."""

    config: KnowledgeValidationConfig | None = None
    """Configuration used for validation (for threshold reference)."""

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

    compositional_consistency: float = 0.0
    """Consistency of semantic compositions (from CompositionalProbes)."""

    crm_correlation: float = 0.0
    """CRM similarity between source and merged model."""

    def status_for_thresholds(
        self,
        excellent_threshold: float,
        acceptable_threshold: float,
        degraded_threshold: float,
    ) -> str:
        """Classify validation status using caller-provided thresholds.

        Parameters
        ----------
        excellent_threshold : float
            Retention above this is "excellent".
        acceptable_threshold : float
            Retention above this is "acceptable".
        degraded_threshold : float
            Retention above this is "degraded".

        Returns
        -------
        str
            Status label: "excellent", "acceptable", "degraded", or "failed".

        Notes
        -----
        The overall_retention IS the validation state. This method classifies
        it using explicit thresholds.
        """
        retention = self.overall_retention
        if retention >= excellent_threshold:
            return "excellent"
        elif retention >= acceptable_threshold:
            return "acceptable"
        elif retention >= degraded_threshold:
            return "degraded"
        else:
            return "failed"

    def compute_status(self, config: KnowledgeValidationConfig) -> str:
        """Compute validation status using config thresholds.

        Parameters
        ----------
        config : KnowledgeValidationConfig
            Configuration with retention thresholds.

        Returns
        -------
        str
            Status label based on overall retention vs thresholds.
        """
        return self.status_for_thresholds(
            excellent_threshold=config.retention_threshold_excellent,
            acceptable_threshold=config.retention_threshold_acceptable,
            degraded_threshold=config.retention_threshold_degraded,
        )

    @property
    def status(self) -> str:
        """Overall validation status.

        Uses stored config if available, otherwise falls back to standard thresholds.
        Prefer status_for_thresholds() with explicit thresholds.
        """
        if self.config is not None:
            return self.compute_status(self.config)
        # Backward compatibility: use standard thresholds
        return self.compute_status(KnowledgeValidationConfig.from_standard_testing())

    def compute_recommendation(self, config: KnowledgeValidationConfig) -> str:
        """Compute recommendation based on status.

        Parameters
        ----------
        config : KnowledgeValidationConfig
            Configuration with retention thresholds.

        Returns
        -------
        str
            Human-readable recommendation.
        """
        status = self.compute_status(config)
        if status == "excellent":
            return "Knowledge transfer is excellent. Merged model is production-ready."
        elif status == "acceptable":
            return (
                "Knowledge transfer is acceptable. Minor degradation in some domains. "
                "Review failed probes before deployment."
            )
        elif status == "degraded":
            return (
                "Knowledge transfer shows significant degradation. "
                "Recommend adjusting merge parameters or using different alpha values."
            )
        else:
            return (
                "Knowledge transfer failed. Merged model has lost critical knowledge. "
                "Do not deploy. Review merge strategy."
            )

    @property
    def recommendation(self) -> str:
        """Human-readable recommendation based on results.

        Returns
        -------
        str
            Human-readable recommendation.

        Notes
        -----
        Deprecated: Use compute_recommendation(config) with explicit thresholds.
        """
        if self.config is not None:
            return self.compute_recommendation(self.config)
        return self.compute_recommendation(KnowledgeValidationConfig.from_standard_testing())

    def get_failed_domains(self, threshold: float) -> list[KnowledgeDomain]:
        """Get domains with retention below threshold.

        Parameters
        ----------
        threshold : float
            Retention threshold (e.g., 0.8 for 80%).

        Returns
        -------
        list of KnowledgeDomain
            List of domains below the threshold.
        """
        return [
            domain
            for domain, result in self.per_domain.items()
            if result.retention_score < threshold
        ]

    def summary(self, config: KnowledgeValidationConfig | None = None) -> dict[str, any]:
        """Get summary dict for JSON output.

        Parameters
        ----------
        config : KnowledgeValidationConfig, optional
            Configuration with thresholds. Uses stored config if not provided.

        Returns
        -------
        dict
            Summary dictionary with status, retention, and recommendations.
        """
        cfg = config or self.config or KnowledgeValidationConfig.from_standard_testing()
        return {
            "status": self.compute_status(cfg),
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
            "recommendation": self.compute_recommendation(cfg),
        }


# =============================================================================
# Validation Functions
# =============================================================================


def run_knowledge_probes(
    generate_fn: Callable[[str], str],
    probes: list[KnowledgeProbe],
    config: KnowledgeValidationConfig | None = None,
) -> list[ProbeResult]:
    """Run knowledge probes against a model.

    Parameters
    ----------
    generate_fn : callable
        Function that takes a prompt and returns model response.
    probes : list of KnowledgeProbe
        List of probes to run.
    config : KnowledgeValidationConfig, optional
        Validation configuration. Uses standard thresholds if not provided.

    Returns
    -------
    list of ProbeResult
        List of ProbeResult for each probe.
    """
    cfg = config or KnowledgeValidationConfig.from_standard_testing()
    results = []

    for probe in probes:
        # Run main prompt
        response = generate_fn(probe.prompt)
        passed = probe.matches(response)

        # Run variations if configured
        variation_results = {}
        if cfg.use_variations and probe.variations:
            for variation in probe.variations:
                var_response = generate_fn(variation)
                variation_results[variation] = probe.matches(var_response)

        results.append(
            ProbeResult(
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
    source_results: list[ProbeResult],
    merged_results: list[ProbeResult],
) -> dict[KnowledgeDomain, KnowledgeRetentionResult]:
    """Compute per-domain retention from probe results.

    Parameters
    ----------
    source_results : list of ProbeResult
        Results from running probes on source model.
    merged_results : list of ProbeResult
        Results from running probes on merged model.

    Returns
    -------
    dict
        Dict mapping domain to retention result.
    """
    # Group by domain
    source_by_domain: dict[KnowledgeDomain, list[ProbeResult]] = {}
    merged_by_domain: dict[KnowledgeDomain, list[ProbeResult]] = {}

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

        source_pass_rate = sum(1 for r in source_probes if r.passed) / len(source_probes)
        merged_pass_rate = sum(1 for r in merged_probes if r.passed) / len(merged_probes)

        passed_ids = [r.probe_id for r in merged_probes if r.passed]
        failed_ids = [r.probe_id for r in merged_probes if not r.passed]

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
    config: KnowledgeValidationConfig | None = None,
    corpus: KnowledgeProbeCorpus | None = None,
) -> KnowledgeTransferReport:
    """Run full knowledge transfer validation.

    Parameters
    ----------
    source_generate_fn : callable
        Generation function for source model.
    merged_generate_fn : callable
        Generation function for merged model.
    config : KnowledgeValidationConfig, optional
        Validation configuration. If not provided, uses standard testing
        thresholds (95%/80%/60%).
    corpus : KnowledgeProbeCorpus, optional
        Probe corpus (uses default if not provided).

    Returns
    -------
    KnowledgeTransferReport
        Comprehensive knowledge transfer report.
    """
    cfg = config or KnowledgeValidationConfig.from_standard_testing()
    probe_corpus = corpus or KnowledgeProbeCorpus()

    # Get probes for configured domains
    probes = []
    for domain in cfg.domains:
        domain_probes = probe_corpus.get_probes(domain)
        probes.extend(domain_probes[: cfg.min_probes_per_domain])

    logger.info(f"Running {len(probes)} knowledge probes across {len(cfg.domains)} domains")

    # Run probes on source model
    source_results = run_knowledge_probes(source_generate_fn, probes, cfg)

    # Run probes on merged model
    merged_results = run_knowledge_probes(merged_generate_fn, probes, cfg)

    # Compute retention
    per_domain = compute_retention_by_domain(source_results, merged_results)

    return KnowledgeTransferReport(
        per_domain=per_domain,
        probe_results=merged_results,
        config=cfg,  # Store config for threshold reference
        compositional_consistency=0.0,  # To be filled by service
        crm_correlation=0.0,  # To be filled by service
    )
