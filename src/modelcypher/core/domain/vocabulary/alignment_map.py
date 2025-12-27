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
Vocabulary Alignment Map.

Stores token mappings and alignment information between vocabularies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


class AlignmentQuality(str, Enum):
    """Quality levels for token alignments."""

    EXACT = "exact"  # Identical tokens
    SIMILAR = "similar"  # High semantic similarity
    APPROXIMATE = "approximate"  # Partial match
    INTERPOLATED = "interpolated"  # Synthesized from neighbors
    UNMAPPED = "unmapped"  # No alignment found


@dataclass
class TokenAlignment:
    """Alignment between a source token and target token(s)."""

    source_id: int
    source_token: str
    target_ids: list[int]  # Can map to multiple targets
    target_tokens: list[str]
    weights: list[float]  # Weights for each target (sum to 1)
    quality: AlignmentQuality
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_token": self.source_token,
            "target_ids": self.target_ids,
            "target_tokens": self.target_tokens,
            "weights": self.weights,
            "quality": self.quality.value,
            "confidence": self.confidence,
            **self.metadata,
        }


@dataclass
class VocabularyAlignmentMap:
    """
    Complete alignment map between source and target vocabularies.

    Supports:
    - One-to-one mappings (exact matches)
    - One-to-many mappings (subword splits)
    - Many-to-one mappings (token merges)
    - Interpolated mappings (synthesized tokens)
    """

    source_vocab_size: int
    target_vocab_size: int
    alignments: dict[int, TokenAlignment] = field(default_factory=dict)
    reverse_map: dict[int, list[int]] = field(default_factory=dict)  # target -> sources
    projection_matrix: "Array | None" = None

    # Statistics
    exact_matches: int = 0
    similar_matches: int = 0
    approximate_matches: int = 0
    interpolated_count: int = 0
    unmapped_count: int = 0

    def add_alignment(self, alignment: TokenAlignment) -> None:
        """Add a token alignment."""
        self.alignments[alignment.source_id] = alignment

        # Update reverse map
        for target_id in alignment.target_ids:
            if target_id not in self.reverse_map:
                self.reverse_map[target_id] = []
            if alignment.source_id not in self.reverse_map[target_id]:
                self.reverse_map[target_id].append(alignment.source_id)

        # Update statistics
        if alignment.quality == AlignmentQuality.EXACT:
            self.exact_matches += 1
        elif alignment.quality == AlignmentQuality.SIMILAR:
            self.similar_matches += 1
        elif alignment.quality == AlignmentQuality.APPROXIMATE:
            self.approximate_matches += 1
        elif alignment.quality == AlignmentQuality.INTERPOLATED:
            self.interpolated_count += 1
        elif alignment.quality == AlignmentQuality.UNMAPPED:
            self.unmapped_count += 1

    def get_alignment(self, source_id: int) -> TokenAlignment | None:
        """Get alignment for a source token."""
        return self.alignments.get(source_id)

    def get_target_sources(self, target_id: int) -> list[int]:
        """Get source tokens that map to a target token."""
        return self.reverse_map.get(target_id, [])

    def iter_alignments(self) -> Iterator[TokenAlignment]:
        """Iterate over all alignments."""
        yield from self.alignments.values()

    @property
    def coverage(self) -> float:
        """Fraction of source tokens with alignments."""
        if self.source_vocab_size == 0:
            return 0.0
        mapped = self.source_vocab_size - self.unmapped_count
        return mapped / self.source_vocab_size

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all alignments."""
        if not self.alignments:
            return 0.0
        return sum(a.confidence for a in self.alignments.values()) / len(self.alignments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (summary only, not full alignments)."""
        return {
            "source_vocab_size": self.source_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "total_alignments": len(self.alignments),
            "coverage": self.coverage,
            "mean_confidence": self.mean_confidence,
            "exact_matches": self.exact_matches,
            "similar_matches": self.similar_matches,
            "approximate_matches": self.approximate_matches,
            "interpolated_count": self.interpolated_count,
            "unmapped_count": self.unmapped_count,
            "has_projection_matrix": self.projection_matrix is not None,
        }

    def quality_distribution(self) -> dict[str, int]:
        """Get distribution of alignment qualities."""
        return {
            "exact": self.exact_matches,
            "similar": self.similar_matches,
            "approximate": self.approximate_matches,
            "interpolated": self.interpolated_count,
            "unmapped": self.unmapped_count,
        }


def build_alignment_from_vocabs(
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    similarity_threshold: float = 0.8,
    max_prefix_length: int = 8,
    max_prefix_matches: int = 3,
    *,
    exact_only: bool = False,
) -> VocabularyAlignmentMap:
    """
    Build alignment map from source and target vocabulary dictionaries.

    Uses exact string matching and simple heuristics. For more sophisticated
    alignment, use the CrossVocabMerger with embedding-based similarity.

    Args:
        source_vocab: Source token -> id mapping
        target_vocab: Target token -> id mapping
        similarity_threshold: Minimum similarity for approximate matches
        exact_only: If True, only exact string matches are accepted

    Returns:
        VocabularyAlignmentMap with token alignments
    """
    alignment_map = VocabularyAlignmentMap(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
    )

    # Build direct and normalized lookups for fast matching
    target_by_token = target_vocab
    normalized_target: dict[str, tuple[str, int]] = {}
    prefix_map: dict[str, list[tuple[str, int]]] = {}

    if max_prefix_length < 0:
        max_prefix_length = 0

    for token, tid in target_vocab.items():
        if not exact_only:
            normalized = token.lower().strip()
            existing = normalized_target.get(normalized)
            if existing is None or tid < existing[1]:
                normalized_target[normalized] = (token, tid)

        if not exact_only and max_prefix_length > 0:
            for i in range(1, min(max_prefix_length, len(token)) + 1):
                prefix = token[:i]
                prefix_map.setdefault(prefix, []).append((token, tid))

    # Deterministic iteration by token id
    source_items = sorted(source_vocab.items(), key=lambda x: x[1])

    for source_token, source_id in source_items:
        if source_token in target_by_token:
            # Exact match
            target_id = target_by_token[source_token]
            alignment = TokenAlignment(
                source_id=source_id,
                source_token=source_token,
                target_ids=[target_id],
                target_tokens=[source_token],
                weights=[1.0],
                quality=AlignmentQuality.EXACT,
                confidence=1.0,
            )
        else:
            if exact_only:
                alignment = TokenAlignment(
                    source_id=source_id,
                    source_token=source_token,
                    target_ids=[],
                    target_tokens=[],
                    weights=[],
                    quality=AlignmentQuality.UNMAPPED,
                    confidence=0.0,
                )
                alignment_map.add_alignment(alignment)
                continue

            # Try normalized matching (lowercase, strip whitespace)
            normalized = source_token.lower().strip()
            normalized_match = normalized_target.get(normalized)
            if normalized_match is not None:
                target_token, target_id = normalized_match
                alignment = TokenAlignment(
                    source_id=source_id,
                    source_token=source_token,
                    target_ids=[target_id],
                    target_tokens=[target_token],
                    weights=[1.0],
                    quality=AlignmentQuality.SIMILAR,
                    confidence=0.9,
                )
            else:
                # Try prefix matching for subwords (bounded to avoid O(N*M))
                prefix_matches: list[tuple[str, int]] = []
                if max_prefix_length > 0:
                    if len(source_token) <= max_prefix_length:
                        prefix_matches.extend(prefix_map.get(source_token, []))

                    max_len = min(len(source_token), max_prefix_length)
                    for i in range(1, max_len + 1):
                        prefix = source_token[:i]
                        if prefix in target_by_token:
                            prefix_matches.append((prefix, target_by_token[prefix]))

                if prefix_matches:
                    # Deduplicate and bound matches deterministically
                    unique: dict[int, str] = {}
                    for token, tid in prefix_matches:
                        if tid not in unique:
                            unique[tid] = token
                    if len(unique) <= max_prefix_matches:
                        target_ids = sorted(unique.keys())
                        target_tokens = [unique[tid] for tid in target_ids]
                        weights = [1.0 / len(target_ids)] * len(target_ids)
                        alignment = TokenAlignment(
                            source_id=source_id,
                            source_token=source_token,
                            target_ids=target_ids,
                            target_tokens=target_tokens,
                            weights=weights,
                            quality=AlignmentQuality.APPROXIMATE,
                            confidence=0.6,
                        )
                    else:
                        alignment = TokenAlignment(
                            source_id=source_id,
                            source_token=source_token,
                            target_ids=[],
                            target_tokens=[],
                            weights=[],
                            quality=AlignmentQuality.UNMAPPED,
                            confidence=0.0,
                        )
                else:
                    # No match found
                    alignment = TokenAlignment(
                        source_id=source_id,
                        source_token=source_token,
                        target_ids=[],
                        target_tokens=[],
                        weights=[],
                        quality=AlignmentQuality.UNMAPPED,
                        confidence=0.0,
                    )

        alignment_map.add_alignment(alignment)

    return alignment_map


@dataclass
class TokenizerComparisonResult:
    """Result of comparing two tokenizers' vocabularies."""

    source_vocab_size: int
    target_vocab_size: int
    overlap_count: int  # Exact match tokens
    overlap_ratio: float  # Fraction of source vocab with exact match
    approximate_count: int  # Tokens matched via normalization/prefix
    unmapped_count: int  # Tokens with no mapping
    coverage: float  # Fraction of source tokens with any mapping

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sourceVocabSize": self.source_vocab_size,
            "targetVocabSize": self.target_vocab_size,
            "overlapCount": self.overlap_count,
            "overlapRatio": round(self.overlap_ratio, 4),
            "approximateCount": self.approximate_count,
            "unmappedCount": self.unmapped_count,
            "coverage": round(self.coverage, 4),
        }


def compare_tokenizers(
    source_tokenizer: Any,
    target_tokenizer: Any,
) -> TokenizerComparisonResult:
    """
    Compare two tokenizers' vocabularies for merge compatibility.

    This is the canonical tokenizer comparison function. It analyzes
    vocabulary overlap and reports alignment statistics.

    Args:
        source_tokenizer: Source model's tokenizer (HuggingFace or tokenizers)
        target_tokenizer: Target model's tokenizer

    Returns:
        TokenizerComparisonResult with comparison metrics
    """
    # Extract vocabularies
    source_vocab = _extract_vocab(source_tokenizer)
    target_vocab = _extract_vocab(target_tokenizer)

    # Build alignment map
    alignment_map = build_alignment_from_vocabs(source_vocab, target_vocab)

    # Compute overlap ratio
    overlap_ratio = alignment_map.exact_matches / max(len(source_vocab), 1)

    return TokenizerComparisonResult(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        overlap_count=alignment_map.exact_matches,
        overlap_ratio=overlap_ratio,
        approximate_count=alignment_map.similar_matches + alignment_map.approximate_matches,
        unmapped_count=alignment_map.unmapped_count,
        coverage=alignment_map.coverage,
    )


def format_comparison_report(result: TokenizerComparisonResult) -> str:
    """Format tokenizer comparison result as human-readable report."""
    lines = [
        "Vocabulary Comparison Report",
        "=" * 40,
        f"Source vocab size: {result.source_vocab_size:,}",
        f"Target vocab size: {result.target_vocab_size:,}",
        "",
        "Alignment Statistics:",
        f"  Exact matches:   {result.overlap_count:,} ({result.overlap_ratio:.1%})",
        f"  Approximate:     {result.approximate_count:,}",
        f"  Unmapped:        {result.unmapped_count:,}",
        f"  Total coverage:  {result.coverage:.1%}",
    ]
    return "\n".join(lines)


def _extract_vocab(tokenizer: Any) -> dict[str, int]:
    """Extract vocabulary mapping from tokenizer."""
    # HuggingFace tokenizers
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    # Fast tokenizers / tokenizers library
    if hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
        if isinstance(vocab, dict):
            return vocab
    # GPT2 style
    if hasattr(tokenizer, "encoder"):
        return tokenizer.encoder
    # Fallback: try vocab attribute on underlying tokenizer
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
        return tokenizer.tokenizer.get_vocab()

    logger.warning("Could not extract vocabulary from tokenizer type %s", type(tokenizer))
    return {}
