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

"""Chunk entropy analyzer for RAG security.

Analyzes retrieved chunks for trust and injection risk using entropy-based signals.
Provides defense-in-depth by scoring chunk trustworthiness before content reaches
the generation model.

Threat Model:
Attackers may inject malicious content into documents that get indexed for RAG:
- Prompt injections hidden in PDFs/docs ("Ignore previous instructions...")
- Jailbreak sequences embedded in technical documentation
- Adversarial text designed to manipulate model behavior
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from modelcypher.core.domain.geometry.vector_math import VectorMath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkTrustAssessment:
    """Detailed assessment of a chunk's trustworthiness based on entropy analysis."""

    semantic_coherence: float
    """Semantic coherence score (0-1). How well the embedding represents the content."""

    linguistic_entropy: float
    """Linguistic entropy (bits). Lower = more predictable/stable text."""

    cross_reference_score: float | None
    """Cross-reference consistency (0-1). Agreement with related chunks.
    None when embedding context unavailable for cross-reference computation."""

    injection_risk: float
    """Injection detection confidence (0-1). 0 = likely clean, 1 = likely injection."""

    verdict: TrustVerdict
    """Overall trust verdict."""

    suspicious_patterns: tuple[str, ...] = ()
    """Detected suspicious patterns (if any)."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "semantic_coherence": self.semantic_coherence,
            "linguistic_entropy": self.linguistic_entropy,
            "cross_reference_score": self.cross_reference_score,
            "injection_risk": self.injection_risk,
            "suspicious_patterns": list(self.suspicious_patterns),
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True)
class TrustComponentAggregates:
    """Aggregate trust component metrics across chunks.

    Returns raw component aggregates instead of weighted composite scores.
    Consumers decide how to interpret these measurements.
    """

    avg_semantic_coherence: float
    """Average semantic coherence across chunks (0-1)."""

    min_semantic_coherence: float
    """Minimum semantic coherence (weakest link)."""

    avg_linguistic_entropy: float
    """Average linguistic entropy in bits."""

    max_linguistic_entropy: float
    """Maximum linguistic entropy (most uncertain chunk)."""

    avg_cross_reference_score: float | None
    """Average cross-reference consistency. None if unavailable for all chunks."""

    max_injection_risk: float
    """Maximum injection risk across chunks."""

    avg_injection_risk: float
    """Average injection risk across chunks."""


@dataclass(frozen=True)
class RetrievalTrustMetrics:
    """Aggregate trust metrics for a RAG retrieval session."""

    component_aggregates: TrustComponentAggregates
    """Per-component aggregate metrics across chunks."""

    trusted_chunk_count: int
    """Number of chunks that passed trust validation."""

    flagged_chunk_count: int
    """Number of chunks flagged as suspicious or untrusted."""

    overall_state: RetrievalTrustState
    """Overall retrieval trust state."""

    trust_analysis_duration_ms: float
    """Time spent on trust analysis (milliseconds)."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "avg_semantic_coherence": self.component_aggregates.avg_semantic_coherence,
            "min_semantic_coherence": self.component_aggregates.min_semantic_coherence,
            "avg_linguistic_entropy": self.component_aggregates.avg_linguistic_entropy,
            "max_linguistic_entropy": self.component_aggregates.max_linguistic_entropy,
            "avg_cross_reference_score": self.component_aggregates.avg_cross_reference_score,
            "max_injection_risk": self.component_aggregates.max_injection_risk,
            "avg_injection_risk": self.component_aggregates.avg_injection_risk,
            "trusted_chunk_count": self.trusted_chunk_count,
            "flagged_chunk_count": self.flagged_chunk_count,
            "overall_state": self.overall_state.value,
            "trust_analysis_duration_ms": self.trust_analysis_duration_ms,
        }


@dataclass(frozen=True)
class ChunkEntropyConfiguration:
    """Configuration for chunk entropy analyzer.

    All thresholds must be explicitly provided or derived from calibration data.
    Use from_calibration_data() to derive from empirical measurements.
    """

    # Basic analysis parameters
    minimum_text_length: int
    """Minimum text length for analysis (shorter texts get default trust)."""

    ngram_size: int
    """Character n-gram size for entropy estimation."""

    injection_sensitivity: float
    """Injection pattern detection sensitivity (0-1)."""

    entropy_threshold: float
    """Entropy threshold above which text is considered cautious."""

    # Verdict thresholds
    injection_untrusted_threshold: float
    """Injection risk above this → UNTRUSTED."""

    injection_suspicious_threshold: float
    """Injection risk above this → SUSPICIOUS."""

    injection_cautious_threshold: float
    """Injection risk above this → CAUTIOUS."""

    coherence_cautious_threshold: float
    """Semantic coherence below this → CAUTIOUS."""

    # Overall state thresholds
    injection_compromised_threshold: float
    """Max injection risk above this → COMPROMISED."""

    flagged_ratio_low_threshold: float
    """Flagged ratio above this → LOW_CONFIDENCE."""

    coherence_low_threshold: float
    """Min coherence below this → LOW_CONFIDENCE."""

    flagged_ratio_moderate_threshold: float
    """Flagged ratio above this → MODERATE."""

    entropy_moderate_threshold: float
    """Max entropy above this → MODERATE."""

    coherence_moderate_threshold: float
    """Min coherence below this → MODERATE."""

    @classmethod
    def from_calibration_data(
        cls,
        injection_risk_samples: list[float],
        coherence_samples: list[float],
        entropy_samples: list[float],
        *,
        untrusted_percentile: float = 0.98,
        suspicious_percentile: float = 0.90,
        cautious_percentile: float = 0.70,
        ngram_size: int = 3,
        minimum_text_length: int = 50,
    ) -> "ChunkEntropyConfiguration":
        """Derive thresholds from calibration data.

        Args:
            injection_risk_samples: Historical injection risk scores from safe texts.
            coherence_samples: Historical coherence scores from safe texts.
            entropy_samples: Historical entropy values from safe texts.
            untrusted_percentile: Percentile for untrusted threshold.
            suspicious_percentile: Percentile for suspicious threshold.
            cautious_percentile: Percentile for cautious threshold.
            ngram_size: Character n-gram size for entropy estimation.
            minimum_text_length: Minimum text length for analysis.

        Returns:
            Configuration with calibration-derived thresholds.
        """
        if not injection_risk_samples or not coherence_samples or not entropy_samples:
            raise ValueError("All sample lists required for calibration")

        def percentile(samples: list[float], p: float) -> float:
            sorted_samples = sorted(samples)
            idx = int(p * (len(sorted_samples) - 1))
            return sorted_samples[idx]

        # Injection thresholds from upper percentiles of safe data
        sorted_injection = sorted(injection_risk_samples)
        injection_untrusted = percentile(injection_risk_samples, untrusted_percentile)
        injection_suspicious = percentile(injection_risk_samples, suspicious_percentile)
        injection_cautious = percentile(injection_risk_samples, cautious_percentile)

        # Coherence thresholds from lower percentiles of safe data
        sorted_coherence = sorted(coherence_samples)
        coherence_cautious = percentile(coherence_samples, 1 - cautious_percentile)
        coherence_low = percentile(coherence_samples, 0.10)  # 10th percentile
        coherence_moderate = percentile(coherence_samples, 0.25)  # 25th percentile

        # Entropy thresholds from upper percentiles
        entropy_threshold = percentile(entropy_samples, suspicious_percentile)
        entropy_moderate = percentile(entropy_samples, 0.90)

        return cls(
            minimum_text_length=minimum_text_length,
            ngram_size=ngram_size,
            injection_sensitivity=0.8,  # Moderate sensitivity
            entropy_threshold=entropy_threshold,
            injection_untrusted_threshold=injection_untrusted,
            injection_suspicious_threshold=injection_suspicious,
            injection_cautious_threshold=injection_cautious,
            coherence_cautious_threshold=coherence_cautious,
            injection_compromised_threshold=injection_untrusted,
            flagged_ratio_low_threshold=0.5,
            coherence_low_threshold=coherence_low,
            flagged_ratio_moderate_threshold=0.2,
            entropy_moderate_threshold=entropy_moderate,
            coherence_moderate_threshold=coherence_moderate,
        )

    @classmethod
    def standard(cls) -> "ChunkEntropyConfiguration":
        """Standard configuration with reasonable defaults.

        Deprecated: Use from_calibration_data() when calibration data is available.
        """
        return cls(
            minimum_text_length=50,
            ngram_size=3,
            injection_sensitivity=0.7,
            entropy_threshold=4.5,
            # Verdict thresholds
            injection_untrusted_threshold=0.8,
            injection_suspicious_threshold=0.5,
            injection_cautious_threshold=0.2,
            coherence_cautious_threshold=0.6,
            # Overall state thresholds
            injection_compromised_threshold=0.8,
            flagged_ratio_low_threshold=0.5,
            coherence_low_threshold=0.3,
            flagged_ratio_moderate_threshold=0.2,
            entropy_moderate_threshold=4.0,
            coherence_moderate_threshold=0.6,
        )


# Known injection patterns to detect
_INJECTION_PATTERNS: list[tuple[str, float, str]] = [
    # Direct instruction overrides
    (r"ignore.*previous.*instructions", 1.0, "instruction_override"),
    (r"disregard.*above", 0.9, "disregard_instruction"),
    (r"forget.*everything", 0.9, "forget_instruction"),
    (r"you are now", 0.8, "persona_injection"),
    (r"act as if", 0.7, "persona_injection"),
    (r"pretend.*you.*are", 0.8, "persona_injection"),
    # System prompt extraction
    (r"what.*system.*prompt", 0.9, "prompt_extraction"),
    (r"repeat.*instructions", 0.8, "prompt_extraction"),
    (r"show.*hidden.*prompt", 0.9, "prompt_extraction"),
    # Jailbreak patterns
    (r"DAN mode", 1.0, "jailbreak"),
    (r"developer mode", 0.8, "jailbreak"),
    (r"unrestricted mode", 0.9, "jailbreak"),
    (r"no ethical", 0.7, "jailbreak"),
    (r"bypass.*safety", 0.9, "jailbreak"),
    # Code injection
    (r"```.*eval\(", 0.8, "code_injection"),
    (r"```.*exec\(", 0.8, "code_injection"),
    (r"```.*system\(", 0.8, "code_injection"),
    (r"<script>", 0.9, "script_injection"),
    (r"javascript:", 0.7, "script_injection"),
    # Hidden instructions
    (r"\[INST\]", 0.6, "hidden_instruction"),
    (r"\[/INST\]", 0.6, "hidden_instruction"),
    (r"<<SYS>>", 0.8, "hidden_instruction"),
    (r"<\|im_start\|>", 0.9, "hidden_instruction"),
    (r"<\|im_end\|>", 0.9, "hidden_instruction"),
    # Manipulation
    (r"do not reveal", 0.5, "secrecy_instruction"),
    (r"keep this secret", 0.5, "secrecy_instruction"),
    (r"never mention", 0.5, "secrecy_instruction"),
]


class ChunkEntropyAnalyzer:
    """Analyzes retrieved chunks for trust and injection risk using entropy-based signals.

    This analyzer computes trust scores for RAG-retrieved text chunks based on:
    - Semantic coherence: How well the chunk's structure matches expected patterns
    - Linguistic entropy: Text predictability/stability
    - Injection detection: Identifies prompt injection attempts, jailbreak patterns
    - Cross-reference consistency: Agreement between related chunks
    """

    def __init__(self, configuration: ChunkEntropyConfiguration | None = None) -> None:
        """Create a chunk entropy analyzer.

        Args:
            configuration: Analyzer configuration. Uses standard() if not provided.
        """
        self._config = configuration or ChunkEntropyConfiguration.standard()

    def analyze_chunk(self, text: str) -> ChunkTrustAssessment:
        """Analyze a single text chunk for trust assessment.

        Args:
            text: The chunk text content.

        Returns:
            Trust assessment with computed scores.
        """
        # Skip analysis for very short texts
        if len(text) < self._config.minimum_text_length:
            return ChunkTrustAssessment(
                semantic_coherence=1.0,
                linguistic_entropy=2.0,
                cross_reference_score=None,  # No embedding context available
                injection_risk=0.0,
                verdict=TrustVerdict.TRUSTED,
            )

        # Compute linguistic entropy
        linguistic_entropy = self._compute_character_entropy(text)

        # Detect injection patterns
        injection_risk, suspicious_patterns = self._detect_injection_patterns(text)

        # Compute semantic coherence (based on text structure)
        semantic_coherence = self._compute_semantic_coherence(text)

        # Cross-reference score: None when analyzing single chunk without embedding context.
        # Real scores computed by analyze_chunks() when embeddings are provided.
        cross_reference_score: float | None = None

        # Determine verdict
        verdict = self._determine_verdict(
            semantic_coherence=semantic_coherence,
            linguistic_entropy=linguistic_entropy,
            injection_risk=injection_risk,
        )

        return ChunkTrustAssessment(
            semantic_coherence=semantic_coherence,
            linguistic_entropy=linguistic_entropy,
            cross_reference_score=cross_reference_score,
            injection_risk=injection_risk,
            suspicious_patterns=tuple(suspicious_patterns),
            verdict=verdict,
        )

    def analyze_chunks(
        self,
        texts: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> list[ChunkTrustAssessment]:
        """Analyze chunks with embedding context for cross-reference scoring.

        Args:
            texts: Array of chunk texts.
            embeddings: Optional embedding vectors for semantic comparison.

        Returns:
            Array of trust assessments.
        """
        # First pass: individual analysis
        assessments = [self.analyze_chunk(text) for text in texts]

        # Second pass: cross-reference scoring if embeddings provided
        if embeddings is not None and len(embeddings) == len(texts):
            cross_ref_scores = self._compute_cross_reference_scores(embeddings)
            assessments = [
                ChunkTrustAssessment(
                    semantic_coherence=a.semantic_coherence,
                    linguistic_entropy=a.linguistic_entropy,
                    cross_reference_score=cross_ref_scores[i],
                    injection_risk=a.injection_risk,
                    suspicious_patterns=a.suspicious_patterns,
                    verdict=a.verdict,
                )
                for i, a in enumerate(assessments)
            ]

        return assessments

    def aggregate_metrics(self, assessments: list[ChunkTrustAssessment]) -> RetrievalTrustMetrics:
        """Compute aggregate trust metrics from individual chunk assessments.

        Args:
            assessments: Array of chunk trust assessments.

        Returns:
            Aggregate retrieval trust metrics.
        """
        if not assessments:
            return RetrievalTrustMetrics(
                component_aggregates=TrustComponentAggregates(
                    avg_semantic_coherence=1.0,
                    min_semantic_coherence=1.0,
                    avg_linguistic_entropy=0.0,
                    max_linguistic_entropy=0.0,
                    avg_cross_reference_score=None,
                    max_injection_risk=0.0,
                    avg_injection_risk=0.0,
                ),
                trusted_chunk_count=0,
                flagged_chunk_count=0,
                overall_state=RetrievalTrustState.HIGH_CONFIDENCE,
                trust_analysis_duration_ms=0.0,
            )

        n = len(assessments)

        # Compute per-component aggregates
        coherences = [a.semantic_coherence for a in assessments]
        entropies = [a.linguistic_entropy for a in assessments]
        injection_risks = [a.injection_risk for a in assessments]

        # Cross-reference may be None for some chunks
        cross_refs = [a.cross_reference_score for a in assessments if a.cross_reference_score is not None]
        avg_cross_ref = sum(cross_refs) / len(cross_refs) if cross_refs else None

        component_aggregates = TrustComponentAggregates(
            avg_semantic_coherence=sum(coherences) / n,
            min_semantic_coherence=min(coherences),
            avg_linguistic_entropy=sum(entropies) / n,
            max_linguistic_entropy=max(entropies),
            avg_cross_reference_score=avg_cross_ref,
            max_injection_risk=max(injection_risks),
            avg_injection_risk=sum(injection_risks) / n,
        )

        trusted_count = sum(
            1 for a in assessments if a.verdict in (TrustVerdict.TRUSTED, TrustVerdict.CAUTIOUS)
        )
        flagged_count = sum(
            1 for a in assessments if a.verdict in (TrustVerdict.SUSPICIOUS, TrustVerdict.UNTRUSTED)
        )

        overall_state = self._compute_overall_state(
            min_coherence=component_aggregates.min_semantic_coherence,
            max_entropy=component_aggregates.max_linguistic_entropy,
            flagged_ratio=flagged_count / n,
            max_injection_risk=component_aggregates.max_injection_risk,
        )

        return RetrievalTrustMetrics(
            component_aggregates=component_aggregates,
            trusted_chunk_count=trusted_count,
            flagged_chunk_count=flagged_count,
            overall_state=overall_state,
            trust_analysis_duration_ms=0.0,  # Caller should measure
        )

    def _compute_character_entropy(self, text: str) -> float:
        """Compute character-level entropy using n-grams."""
        chars = list(text.lower())
        if len(chars) <= self._config.ngram_size:
            return 0.0

        ngram_counts: dict[str, int] = {}
        for i in range(len(chars) - self._config.ngram_size + 1):
            ngram = "".join(chars[i : i + self._config.ngram_size])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        total = sum(ngram_counts.values())
        entropy = 0.0

        for count in ngram_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _detect_injection_patterns(self, text: str) -> tuple[float, list[str]]:
        """Detect injection patterns in text."""
        lowercased = text.lower()
        max_risk = 0.0
        detected_patterns: list[str] = []

        for pattern, weight, name in _INJECTION_PATTERNS:
            try:
                if re.search(pattern, lowercased, re.IGNORECASE):
                    adjusted_weight = weight * self._config.injection_sensitivity
                    max_risk = max(max_risk, adjusted_weight)
                    if name not in detected_patterns:
                        detected_patterns.append(name)
            except re.error:
                continue

        return max_risk, detected_patterns

    def _compute_semantic_coherence(self, text: str) -> float:
        """Compute semantic coherence based on text structure."""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

        if not sentences:
            return 0.5

        # Check average sentence length (ideal: 15-30 words)
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_words < 5:
            length_score = 0.6  # Very short sentences
        elif avg_words > 50:
            length_score = 0.7  # Very long sentences
        else:
            length_score = 1.0

        # Check for unusual character ratios
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = len(text)
        alpha_ratio = alpha_count / max(1, total_count)

        if alpha_ratio < 0.5:
            char_score = 0.6  # Too many non-letter characters
        elif alpha_ratio > 0.95:
            char_score = 0.8  # Suspiciously few punctuation/numbers
        else:
            char_score = 1.0

        return (length_score + char_score) / 2.0

    def _compute_cross_reference_scores(self, embeddings: list[list[float]]) -> list[float]:
        """Compute cross-reference consistency scores from embeddings."""
        if len(embeddings) <= 1:
            return [1.0] * len(embeddings)

        # Validate all embeddings have the same dimension
        dims = len(embeddings[0])
        if not all(len(e) == dims for e in embeddings) or dims == 0:
            logger.warning(
                "Cross-reference scoring skipped: embedding dimensions mismatch or empty"
            )
            return [1.0] * len(embeddings)

        # Compute centroid
        centroid = [0.0] * dims
        for embedding in embeddings:
            for i, val in enumerate(embedding):
                centroid[i] += val
        centroid = [v / len(embeddings) for v in centroid]

        # Score each embedding by similarity to centroid
        return [VectorMath.cosine_similarity(e, centroid) or 0.0 for e in embeddings]

    def _determine_verdict(
        self,
        semantic_coherence: float,
        linguistic_entropy: float,
        injection_risk: float,
    ) -> TrustVerdict:
        """Determine trust verdict from component scores using config thresholds."""
        cfg = self._config

        # Automatic untrusted if high injection risk
        if injection_risk > cfg.injection_untrusted_threshold:
            return TrustVerdict.UNTRUSTED

        # Automatic suspicious if moderate injection risk
        if injection_risk > cfg.injection_suspicious_threshold:
            return TrustVerdict.SUSPICIOUS

        # Check entropy bounds
        if linguistic_entropy > cfg.entropy_threshold:
            return TrustVerdict.CAUTIOUS

        # Check coherence
        if semantic_coherence < cfg.coherence_cautious_threshold:
            return TrustVerdict.CAUTIOUS

        # Low injection risk but some patterns detected
        if injection_risk > cfg.injection_cautious_threshold:
            return TrustVerdict.CAUTIOUS

        return TrustVerdict.TRUSTED

    def _compute_overall_state(
        self,
        min_coherence: float,
        max_entropy: float,
        flagged_ratio: float,
        max_injection_risk: float,
    ) -> RetrievalTrustState:
        """Compute overall retrieval trust state from components using config thresholds."""
        cfg = self._config

        # Any high injection risk = compromised
        if max_injection_risk > cfg.injection_compromised_threshold:
            return RetrievalTrustState.COMPROMISED

        # Majority flagged or very low coherence = low confidence
        if flagged_ratio > cfg.flagged_ratio_low_threshold or min_coherence < cfg.coherence_low_threshold:
            return RetrievalTrustState.LOW_CONFIDENCE

        # Some concerns: high entropy or moderate flagging
        if (
            flagged_ratio > cfg.flagged_ratio_moderate_threshold
            or max_entropy > cfg.entropy_moderate_threshold
            or min_coherence < cfg.coherence_moderate_threshold
        ):
            return RetrievalTrustState.MODERATE

        return RetrievalTrustState.HIGH_CONFIDENCE
