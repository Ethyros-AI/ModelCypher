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
import re
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_finite,
    log2_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkTrustAssessment:
    """Trust assessment of a single text chunk.

    Attributes
    ----------
    semantic_coherence : float
        Coherence score in [0, 1]. Higher = more coherent.
    linguistic_entropy : float
        Character n-gram entropy in bits. Lower = more predictable.
    cross_reference_score : float or None
        Normalized geodesic agreement with other chunks in [0, 1]. None if unavailable.
    injection_risk : float
        Injection detection confidence in [0, 1]. Higher = more likely injection.
    suspicious_patterns : tuple[str, ...]
        Names of detected injection patterns.
    """

    semantic_coherence: float
    linguistic_entropy: float
    cross_reference_score: float | None
    injection_risk: float
    suspicious_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "semantic_coherence": self.semantic_coherence,
            "linguistic_entropy": self.linguistic_entropy,
            "cross_reference_score": self.cross_reference_score,
            "injection_risk": self.injection_risk,
            "suspicious_patterns": list(self.suspicious_patterns),
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
    """Aggregate trust metrics for a RAG retrieval session.

    Attributes
    ----------
    component_aggregates : TrustComponentAggregates
        Per-component aggregate metrics across chunks.
    total_chunk_count : int
        Total number of chunks analyzed.
    trust_analysis_duration_ms : float
        Time spent on trust analysis (milliseconds).
    """

    component_aggregates: TrustComponentAggregates
    total_chunk_count: int
    trust_analysis_duration_ms: float

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
            "total_chunk_count": self.total_chunk_count,
            "trust_analysis_duration_ms": self.trust_analysis_duration_ms,
        }


# =============================================================================
# Module Constants (derived from standard text analysis practices)
# =============================================================================

# Minimum text length for meaningful entropy analysis.
# Character 3-grams need at least ~30 chars for stable statistics.
# 50 provides margin for varied text quality.
_MINIMUM_TEXT_LENGTH = 50

# Character n-gram size for entropy estimation.
# 3-grams are standard in computational linguistics for character-level
# entropy estimation, balancing context capture with statistical stability.
_NGRAM_SIZE = 3


# Known injection patterns to detect.
#
# WARNING: The confidence scores (second element) are HEURISTIC ESTIMATES,
# not calibrated from empirical data. These represent relative severity
# rankings based on attack pattern analysis, NOT detection probabilities.
#
# Interpretation:
# - 1.0 = High-confidence indicator (explicit jailbreak keyword)
# - 0.7-0.9 = Medium confidence (common attack patterns)
# - 0.5-0.6 = Lower confidence (could be legitimate in some contexts)
#
# Users deploying this in production SHOULD calibrate these scores
# against their specific threat model and false-positive tolerance.
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

    No configuration needed - uses standard text analysis parameters.
    """

    def __init__(self) -> None:
        """Create a chunk entropy analyzer.

        No configuration needed - uses standard text analysis parameters
        derived from computational linguistics best practices.
        """
        self._compiled_patterns: list[tuple[re.Pattern[str], float, str]] = []
        for pattern, weight, name in _INJECTION_PATTERNS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                logger.warning("Invalid injection pattern '%s': %s", pattern, exc)
                continue
            self._compiled_patterns.append((compiled, weight, name))

    def analyze_chunk(self, text: str) -> ChunkTrustAssessment:
        """Analyze a single text chunk for trust assessment.

        Parameters
        ----------
        text : str
            The chunk text content.

        Returns
        -------
        ChunkTrustAssessment
            Trust scores and detected patterns.
        """
        # Skip analysis for very short texts
        if len(text) < _MINIMUM_TEXT_LENGTH:
            return ChunkTrustAssessment(
                semantic_coherence=1.0,
                linguistic_entropy=2.0,
                cross_reference_score=None,
                injection_risk=0.0,
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

        return ChunkTrustAssessment(
            semantic_coherence=semantic_coherence,
            linguistic_entropy=linguistic_entropy,
            cross_reference_score=cross_reference_score,
            injection_risk=injection_risk,
            suspicious_patterns=tuple(suspicious_patterns),
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
            Array of trust assessments with raw measurements.
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
                )
                for i, a in enumerate(assessments)
            ]

        return assessments

    def aggregate_metrics(self, assessments: list[ChunkTrustAssessment]) -> RetrievalTrustMetrics:
        """Compute aggregate trust metrics from individual chunk assessments.

        Parameters
        ----------
        assessments : list[ChunkTrustAssessment]
            Individual chunk trust assessments.

        Returns
        -------
        RetrievalTrustMetrics
            Aggregate metrics across all chunks.
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
                total_chunk_count=0,
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

        return RetrievalTrustMetrics(
            component_aggregates=component_aggregates,
            total_chunk_count=n,
            trust_analysis_duration_ms=0.0,  # Caller should measure
        )

    def _compute_character_entropy(self, text: str) -> float:
        """Compute character-level entropy using n-grams."""
        chars = list(text.lower())
        if len(chars) <= _NGRAM_SIZE:
            return 0.0

        ngram_counts: dict[str, int] = {}
        for i in range(len(chars) - _NGRAM_SIZE + 1):
            ngram = "".join(chars[i : i + _NGRAM_SIZE])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        total = sum(ngram_counts.values())
        entropy = 0.0

        backend = get_default_backend()
        for count in ngram_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2_scalar(p, backend)

        return entropy

    def _detect_injection_patterns(self, text: str) -> tuple[float, list[str]]:
        """Detect injection patterns in text.

        Uses pattern weights directly - each pattern has a carefully calibrated
        weight based on its severity (1.0 for severe like 'DAN mode', 0.5 for
        mild like 'do not reveal').
        """
        lowercased = text.lower()
        max_risk = 0.0
        detected_patterns: list[str] = []

        for compiled, weight, name in self._compiled_patterns:
            if compiled.search(lowercased):
                # Use pattern weight directly - no arbitrary scaling
                max_risk = max(max_risk, weight)
                if name not in detected_patterns:
                    detected_patterns.append(name)

        return max_risk, detected_patterns

    def _compute_semantic_coherence(self, text: str) -> float:
        """Compute semantic coherence based on text structure.

        Returns the alphabetic character ratio as a direct measure of
        text structure. Higher ratio = more textual content vs. special
        characters, code, or formatting. No arbitrary classification
        thresholds - the raw ratio is the signal.

        Returns:
            Alpha ratio in [0, 1]. Caller interprets based on their context.
        """
        # Alpha ratio: proportion of alphabetic characters
        # This is a structural measurement without arbitrary cutoffs
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = len(text)
        alpha_ratio = alpha_count / max(1, total_count)

        return alpha_ratio

    def _compute_cross_reference_scores(self, embeddings: list[list[float]]) -> list[float]:
        """Compute cross-reference agreement from geodesic structure."""
        if len(embeddings) <= 1:
            return [1.0] * len(embeddings)

        dims = [len(e) for e in embeddings]
        min_dim = min(dims) if dims else 0
        if min_dim == 0:
            logger.warning("Cross-reference scoring skipped: empty embeddings detected")
            return [0.0] * len(embeddings)

        # Project to a shared dimension if needed (truncate to min dimension).
        if len(set(dims)) > 1:
            logger.debug(
                "Cross-reference scoring using shared dimension %d for %d embeddings",
                min_dim,
                len(embeddings),
            )
            projected = [e[:min_dim] for e in embeddings]
        else:
            projected = embeddings

        backend = get_default_backend()
        points = backend.array(projected)
        backend.eval(points)

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points)
        backend.eval(geo_result.distances)
        dist = geo_result.distances
        backend.eval(dist)

        n = len(projected)
        if n == 0:
            return []

        finite_mask = backend.isfinite(dist)
        neg_inf = -float(backend.finfo().max)
        finite_vals = backend.where(finite_mask, dist, backend.full(dist.shape, neg_inf))
        max_distance_arr = backend.max(finite_vals)
        backend.eval(max_distance_arr)
        max_distance = float(backend.to_scalar(max_distance_arr))

        if not is_finite(max_distance, backend):
            logger.warning("Cross-reference scoring skipped: no finite distances")
            return [0.0] * n
        if max_distance <= 0.0:
            return [1.0] * n

        dist_filled = backend.where(finite_mask, dist, backend.full(dist.shape, max_distance))
        off_diag = backend.ones_like(dist_filled) - backend.eye(n)
        sum_rows = backend.sum(dist_filled * off_diag, axis=1)
        mean_distances = sum_rows / max(1, n - 1)
        # Use epsilon guard for division to handle extremely small max_distance
        eps = division_epsilon(backend, mean_distances)
        safe_max_distance = max(max_distance, eps)
        scores = 1.0 - (mean_distances / safe_max_distance)
        scores = backend.maximum(backend.minimum(scores, backend.full(scores.shape, 1.0)), backend.zeros_like(scores))
        backend.eval(scores)
        return backend.tolist(scores)
