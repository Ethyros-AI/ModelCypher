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
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class TrustVerdict(str, Enum):
    """Trust verdict for a retrieved chunk."""

    TRUSTED = "trusted"
    """Chunk passed all trust checks."""

    CAUTIOUS = "cautious"
    """Minor concerns but usable."""

    SUSPICIOUS = "suspicious"
    """Significant concerns, use with care."""

    UNTRUSTED = "untrusted"
    """Failed trust checks, should be filtered."""


class RetrievalTrustState(str, Enum):
    """Overall trust state for a retrieval session."""

    HIGH_CONFIDENCE = "highConfidence"
    """All chunks passed trust validation - proceed with confidence."""

    MODERATE = "moderate"
    """Most chunks trusted but some caution advised."""

    LOW_CONFIDENCE = "lowConfidence"
    """Significant trust concerns - review results carefully."""

    COMPROMISED = "compromised"
    """Trust validation failed - results may be compromised."""


@dataclass(frozen=True)
class ChunkTrustAssessment:
    """Detailed assessment of a chunk's trustworthiness based on entropy analysis."""

    semantic_coherence: float
    """Semantic coherence score (0-1). How well the embedding represents the content."""

    linguistic_entropy: float
    """Linguistic entropy (bits). Lower = more predictable/stable text."""

    cross_reference_score: float
    """Cross-reference consistency (0-1). Agreement with related chunks."""

    injection_risk: float
    """Injection detection confidence (0-1). 0 = likely clean, 1 = likely injection."""

    verdict: TrustVerdict
    """Overall trust verdict."""

    suspicious_patterns: tuple[str, ...] = ()
    """Detected suspicious patterns (if any)."""

    @property
    def computed_trust_score(self) -> float:
        """Compute trust score from assessment components."""
        coherence_weight = 0.35
        entropy_weight = 0.25
        cross_ref_weight = 0.20
        injection_weight = 0.20

        # Normalize linguistic entropy (lower is better, cap at 5 bits)
        normalized_entropy = max(0, 1.0 - self.linguistic_entropy / 5.0)

        # Invert injection risk (0 risk = full trust)
        injection_trust = 1.0 - self.injection_risk

        score = (
            coherence_weight * self.semantic_coherence
            + entropy_weight * normalized_entropy
            + cross_ref_weight * self.cross_reference_score
            + injection_weight * injection_trust
        )

        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "semantic_coherence": self.semantic_coherence,
            "linguistic_entropy": self.linguistic_entropy,
            "cross_reference_score": self.cross_reference_score,
            "injection_risk": self.injection_risk,
            "suspicious_patterns": list(self.suspicious_patterns),
            "verdict": self.verdict.value,
            "computed_trust_score": self.computed_trust_score,
        }


@dataclass(frozen=True)
class RetrievalTrustMetrics:
    """Aggregate trust metrics for a RAG retrieval session."""

    average_trust_score: float
    """Average trust score across all retrieved chunks."""

    minimum_trust_score: float
    """Minimum trust score (weakest link)."""

    trusted_chunk_count: int
    """Number of chunks that passed trust validation."""

    flagged_chunk_count: int
    """Number of chunks flagged as suspicious or untrusted."""

    overall_state: RetrievalTrustState
    """Overall retrieval trust state."""

    aggregate_injection_risk: float
    """Aggregate injection risk across chunks."""

    trust_analysis_duration_ms: float
    """Time spent on trust analysis (milliseconds)."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "average_trust_score": self.average_trust_score,
            "minimum_trust_score": self.minimum_trust_score,
            "trusted_chunk_count": self.trusted_chunk_count,
            "flagged_chunk_count": self.flagged_chunk_count,
            "overall_state": self.overall_state.value,
            "aggregate_injection_risk": self.aggregate_injection_risk,
            "trust_analysis_duration_ms": self.trust_analysis_duration_ms,
        }


@dataclass(frozen=True)
class ChunkEntropyConfiguration:
    """Configuration for chunk entropy analyzer."""

    minimum_text_length: int = 50
    """Minimum text length for analysis (shorter texts get default trust)."""

    ngram_size: int = 3
    """Character n-gram size for entropy estimation."""

    injection_sensitivity: float = 0.7
    """Injection pattern detection sensitivity (0-1)."""

    entropy_threshold: float = 4.5
    """Entropy threshold above which text is considered suspicious."""

    trust_flag_threshold: float = 0.4
    """Trust score below which chunk is flagged."""

    @classmethod
    def default(cls) -> ChunkEntropyConfiguration:
        """Create default configuration."""
        return cls()

    @classmethod
    def paranoid(cls) -> ChunkEntropyConfiguration:
        """Paranoid configuration for high-security environments."""
        return cls(
            minimum_text_length=30,
            ngram_size=2,
            injection_sensitivity=0.9,
            entropy_threshold=4.0,
            trust_flag_threshold=0.6,
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

    def __init__(
        self, configuration: ChunkEntropyConfiguration | None = None
    ) -> None:
        """Create a chunk entropy analyzer.

        Args:
            configuration: Analyzer configuration. Defaults to standard settings.
        """
        self._config = configuration or ChunkEntropyConfiguration.default()

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
                cross_reference_score=1.0,
                injection_risk=0.0,
                verdict=TrustVerdict.TRUSTED,
            )

        # Compute linguistic entropy
        linguistic_entropy = self._compute_character_entropy(text)

        # Detect injection patterns
        injection_risk, suspicious_patterns = self._detect_injection_patterns(text)

        # Compute semantic coherence (based on text structure)
        semantic_coherence = self._compute_semantic_coherence(text)

        # Cross-reference score (placeholder - requires embedding context)
        cross_reference_score = 1.0

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

    def aggregate_metrics(
        self, assessments: list[ChunkTrustAssessment]
    ) -> RetrievalTrustMetrics:
        """Compute aggregate trust metrics from individual chunk assessments.

        Args:
            assessments: Array of chunk trust assessments.

        Returns:
            Aggregate retrieval trust metrics.
        """
        if not assessments:
            return RetrievalTrustMetrics(
                average_trust_score=1.0,
                minimum_trust_score=1.0,
                trusted_chunk_count=0,
                flagged_chunk_count=0,
                overall_state=RetrievalTrustState.HIGH_CONFIDENCE,
                aggregate_injection_risk=0.0,
                trust_analysis_duration_ms=0.0,
            )

        trust_scores = [a.computed_trust_score for a in assessments]
        average_trust = sum(trust_scores) / len(trust_scores)
        min_trust = min(trust_scores)

        trusted_count = sum(
            1
            for a in assessments
            if a.verdict in (TrustVerdict.TRUSTED, TrustVerdict.CAUTIOUS)
        )
        flagged_count = sum(
            1
            for a in assessments
            if a.verdict in (TrustVerdict.SUSPICIOUS, TrustVerdict.UNTRUSTED)
        )

        max_injection_risk = max(a.injection_risk for a in assessments)

        overall_state = self._compute_overall_state(
            average_trust=average_trust,
            min_trust=min_trust,
            flagged_ratio=flagged_count / len(assessments),
            injection_risk=max_injection_risk,
        )

        return RetrievalTrustMetrics(
            average_trust_score=average_trust,
            minimum_trust_score=min_trust,
            trusted_chunk_count=trusted_count,
            flagged_chunk_count=flagged_count,
            overall_state=overall_state,
            aggregate_injection_risk=max_injection_risk,
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
        sentences = [
            s.strip()
            for s in re.split(r"[.!?]", text)
            if s.strip()
        ]

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

    def _compute_cross_reference_scores(
        self, embeddings: list[list[float]]
    ) -> list[float]:
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
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        return [cosine_similarity(e, centroid) for e in embeddings]

    def _determine_verdict(
        self,
        semantic_coherence: float,
        linguistic_entropy: float,
        injection_risk: float,
    ) -> TrustVerdict:
        """Determine trust verdict from component scores."""
        # Automatic untrusted if high injection risk
        if injection_risk > 0.8:
            return TrustVerdict.UNTRUSTED

        # Automatic suspicious if moderate injection risk
        if injection_risk > 0.5:
            return TrustVerdict.SUSPICIOUS

        # Check entropy bounds
        if linguistic_entropy > self._config.entropy_threshold:
            return TrustVerdict.CAUTIOUS

        # Check coherence
        if semantic_coherence < 0.6:
            return TrustVerdict.CAUTIOUS

        # Low injection risk but some patterns detected
        if injection_risk > 0.2:
            return TrustVerdict.CAUTIOUS

        return TrustVerdict.TRUSTED

    def _compute_overall_state(
        self,
        average_trust: float,
        min_trust: float,
        flagged_ratio: float,
        injection_risk: float,
    ) -> RetrievalTrustState:
        """Compute overall retrieval trust state."""
        # Any high injection risk = compromised
        if injection_risk > 0.8:
            return RetrievalTrustState.COMPROMISED

        # Majority flagged = low confidence
        if flagged_ratio > 0.5 or min_trust < 0.3:
            return RetrievalTrustState.LOW_CONFIDENCE

        # Some concerns but mostly okay
        if flagged_ratio > 0.2 or average_trust < 0.7:
            return RetrievalTrustState.MODERATE

        return RetrievalTrustState.HIGH_CONFIDENCE
