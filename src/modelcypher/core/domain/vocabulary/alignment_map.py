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
    EXACT = "exact"           # Identical tokens
    SIMILAR = "similar"       # High semantic similarity
    APPROXIMATE = "approximate"  # Partial match
    INTERPOLATED = "interpolated"  # Synthesized from neighbors
    UNMAPPED = "unmapped"     # No alignment found


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
) -> VocabularyAlignmentMap:
    """
    Build alignment map from source and target vocabulary dictionaries.

    Uses exact string matching and simple heuristics. For more sophisticated
    alignment, use the CrossVocabMerger with embedding-based similarity.

    Args:
        source_vocab: Source token -> id mapping
        target_vocab: Target token -> id mapping
        similarity_threshold: Minimum similarity for approximate matches

    Returns:
        VocabularyAlignmentMap with token alignments
    """
    alignment_map = VocabularyAlignmentMap(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
    )

    # Build reverse lookups
    source_by_token = {token: id_ for token, id_ in source_vocab.items()}
    target_by_token = {token: id_ for token, id_ in target_vocab.items()}
    target_id_to_token = {id_: token for token, id_ in target_vocab.items()}

    for source_token, source_id in source_vocab.items():
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
            # Try normalized matching (lowercase, strip whitespace)
            normalized = source_token.lower().strip()
            matched = False

            for target_token, target_id in target_vocab.items():
                if target_token.lower().strip() == normalized:
                    alignment = TokenAlignment(
                        source_id=source_id,
                        source_token=source_token,
                        target_ids=[target_id],
                        target_tokens=[target_token],
                        weights=[1.0],
                        quality=AlignmentQuality.SIMILAR,
                        confidence=0.9,
                    )
                    matched = True
                    break

            if not matched:
                # Try prefix/suffix matching for subwords
                prefix_matches = [
                    (t, tid) for t, tid in target_vocab.items()
                    if t.startswith(source_token) or source_token.startswith(t)
                ]

                if prefix_matches and len(prefix_matches) <= 3:
                    target_tokens = [t for t, _ in prefix_matches]
                    target_ids = [tid for _, tid in prefix_matches]
                    weights = [1.0 / len(prefix_matches)] * len(prefix_matches)
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
