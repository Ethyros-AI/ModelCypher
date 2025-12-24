"""
Vocabulary Alignment for Cross-Tokenizer Model Merging.

Detects and quantifies tokenizer mismatches between models,
enabling merging of models with different vocabularies.

Implements multi-pass alignment:
1. Exact string match (fastest, highest confidence)
2. Decomposition match (token → subtokens in other vocab)
3. Semantic anchor match (via semantic prime embeddings)

References:
- Minixhofer et al. (2024). Zero-Shot Tokenizer Transfer. NeurIPS 2024.
- Moayeri et al. (2023). Text-To-Concept via Cross-Model Alignment. ICML 2023.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class AlignmentMethod(str, Enum):
    """Method used to align a token between vocabularies."""

    EXACT = "exact"  # Same token string in both vocabs
    DECOMPOSED = "decomposed"  # Token decomposes to known tokens
    SEMANTIC = "semantic"  # Aligned via embedding similarity
    UNMAPPED = "unmapped"  # No alignment found


@dataclass(frozen=True)
class TokenMapping:
    """Mapping of a source token to target vocabulary."""

    source_token_id: int
    target_token_id: int | None  # None if unmapped
    method: AlignmentMethod
    confidence: float  # 0-1, higher is better
    decomposition: tuple[int, ...] | None = None  # Target token IDs for DECOMPOSED

    @property
    def is_mapped(self) -> bool:
        """Check if token has a mapping."""
        return self.method != AlignmentMethod.UNMAPPED


@dataclass
class VocabularyAlignmentResult:
    """Result of aligning two tokenizer vocabularies."""

    source_vocab_size: int
    target_vocab_size: int
    overlap_count: int  # Exact match tokens
    decomposed_count: int  # Decomposition-aligned tokens
    semantic_count: int  # Semantic-aligned tokens
    unmapped_count: int  # Tokens with no mapping
    mappings: dict[int, TokenMapping] = field(default_factory=dict)

    @property
    def overlap_ratio(self) -> float:
        """Fraction of source vocab with exact match in target."""
        if self.source_vocab_size == 0:
            return 0.0
        return self.overlap_count / self.source_vocab_size

    @property
    def coverage(self) -> float:
        """Fraction of source tokens with any mapping."""
        if self.source_vocab_size == 0:
            return 0.0
        return 1.0 - (self.unmapped_count / self.source_vocab_size)

    @property
    def recommended_method(self) -> str:
        """Recommend embedding bridge method based on overlap."""
        if self.overlap_ratio > 0.9:
            return "fvt"  # Fast Vocabulary Transfer sufficient
        elif self.overlap_ratio > 0.5:
            return "fvt+procrustes"  # FVT with Procrustes refinement
        else:
            return "procrustes+affine"  # Need full alignment

    @property
    def merge_feasibility(self) -> str:
        """Assess feasibility of cross-vocabulary merge."""
        if self.coverage > 0.95:
            return "high"
        elif self.coverage > 0.8:
            return "medium"
        elif self.coverage > 0.5:
            return "low"
        else:
            return "infeasible"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sourceVocabSize": self.source_vocab_size,
            "targetVocabSize": self.target_vocab_size,
            "overlapCount": self.overlap_count,
            "decomposedCount": self.decomposed_count,
            "semanticCount": self.semantic_count,
            "unmappedCount": self.unmapped_count,
            "overlapRatio": round(self.overlap_ratio, 4),
            "coverage": round(self.coverage, 4),
            "recommendedMethod": self.recommended_method,
            "mergeFeasibility": self.merge_feasibility,
        }


@dataclass
class VocabularyAlignmentConfig:
    """Configuration for vocabulary alignment."""

    max_decomposition_tokens: int = 5  # Max subtokens for decomposition
    semantic_threshold: float = 0.7  # Min cosine similarity for semantic match
    use_semantic_primes: bool = True  # Use semantic primes for anchor alignment
    batch_size: int = 1000  # Batch size for embedding comparisons


class VocabularyAligner:
    """
    Aligns vocabularies between two tokenizers for cross-vocabulary merging.

    Implements multi-pass alignment strategy:
    1. Exact match: Same token string in both vocabularies
    2. Decomposition: Token in source can be represented as sequence in target
    3. Semantic: Embedding similarity using anchor tokens

    Usage:
        aligner = VocabularyAligner()
        result = aligner.align(source_tokenizer, target_tokenizer)
        print(f"Overlap: {result.overlap_ratio:.1%}")
        print(f"Recommended: {result.recommended_method}")
    """

    def __init__(self, config: VocabularyAlignmentConfig | None = None):
        self.config = config or VocabularyAlignmentConfig()

    def align(
        self,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_embeddings: np.ndarray | None = None,
        target_embeddings: np.ndarray | None = None,
    ) -> VocabularyAlignmentResult:
        """
        Compute token-level alignment between vocabularies.

        Args:
            source_tokenizer: Source model's tokenizer.
            target_tokenizer: Target model's tokenizer.
            source_embeddings: Optional source embedding matrix for semantic matching.
            target_embeddings: Optional target embedding matrix for semantic matching.

        Returns:
            VocabularyAlignmentResult with per-token mappings and statistics.
        """
        source_vocab = self._get_vocab(source_tokenizer)
        target_vocab = self._get_vocab(target_tokenizer)

        source_vocab_size = len(source_vocab)
        target_vocab_size = len(target_vocab)

        mappings: dict[int, TokenMapping] = {}

        # Pass 1: Exact match
        exact_mappings = self._exact_match(source_vocab, target_vocab)
        mappings.update(exact_mappings)
        exact_count = len(exact_mappings)

        # Pass 2: Decomposition match for unmapped tokens
        unmapped_ids = [
            tok_id for tok_id in source_vocab.values() if tok_id not in mappings
        ]
        decomp_mappings = self._decomposition_match(
            unmapped_ids, source_vocab, target_tokenizer
        )
        mappings.update(decomp_mappings)
        decomp_count = len(decomp_mappings)

        # Pass 3: Semantic match (if embeddings provided)
        semantic_count = 0
        if source_embeddings is not None and target_embeddings is not None:
            unmapped_ids = [
                tok_id for tok_id in source_vocab.values() if tok_id not in mappings
            ]
            semantic_mappings = self._semantic_match(
                unmapped_ids,
                source_embeddings,
                target_embeddings,
                source_vocab,
                target_vocab,
            )
            mappings.update(semantic_mappings)
            semantic_count = len(semantic_mappings)

        # Mark remaining as unmapped
        for tok_id in source_vocab.values():
            if tok_id not in mappings:
                mappings[tok_id] = TokenMapping(
                    source_token_id=tok_id,
                    target_token_id=None,
                    method=AlignmentMethod.UNMAPPED,
                    confidence=0.0,
                )

        unmapped_count = sum(
            1 for m in mappings.values() if m.method == AlignmentMethod.UNMAPPED
        )

        return VocabularyAlignmentResult(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            overlap_count=exact_count,
            decomposed_count=decomp_count,
            semantic_count=semantic_count,
            unmapped_count=unmapped_count,
            mappings=mappings,
        )

    def compare_vocabularies(
        self,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
    ) -> dict:
        """
        Quick comparison of vocabularies without full alignment.

        Returns statistics about vocabulary overlap for compatibility check.
        """
        source_vocab = self._get_vocab(source_tokenizer)
        target_vocab = self._get_vocab(target_tokenizer)

        source_set = set(source_vocab.keys())
        target_set = set(target_vocab.keys())

        overlap = source_set & target_set
        source_only = source_set - target_set
        target_only = target_set - source_set

        return {
            "sourceVocabSize": len(source_vocab),
            "targetVocabSize": len(target_vocab),
            "overlapCount": len(overlap),
            "overlapRatio": len(overlap) / len(source_vocab) if source_vocab else 0,
            "sourceOnlyCount": len(source_only),
            "targetOnlyCount": len(target_only),
            "compatible": len(overlap) / len(source_vocab) > 0.9 if source_vocab else False,
        }

    def _get_vocab(self, tokenizer: Tokenizer) -> dict[str, int]:
        """Extract vocabulary from tokenizer."""
        return tokenizer.get_vocab()

    def _exact_match(
        self,
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> dict[int, TokenMapping]:
        """Find tokens present in both vocabularies."""
        mappings = {}
        for token_str, source_id in source_vocab.items():
            if token_str in target_vocab:
                target_id = target_vocab[token_str]
                mappings[source_id] = TokenMapping(
                    source_token_id=source_id,
                    target_token_id=target_id,
                    method=AlignmentMethod.EXACT,
                    confidence=1.0,
                )
        return mappings

    def _decomposition_match(
        self,
        unmapped_ids: list[int],
        source_vocab: dict[str, int],
        target_tokenizer: Tokenizer,
    ) -> dict[int, TokenMapping]:
        """
        Find tokens that decompose to known target tokens.

        For each unmapped source token, tokenize its string representation
        using the target tokenizer. If the result is a short sequence of
        valid target tokens, create a decomposition mapping.
        """
        # Build reverse lookup
        id_to_token = {v: k for k, v in source_vocab.items()}

        mappings = {}
        for source_id in unmapped_ids:
            token_str = id_to_token.get(source_id)
            if not token_str:
                continue

            # Tokenize using target tokenizer
            try:
                encoded = target_tokenizer.encode(token_str, add_special_tokens=False)
                target_ids = tuple(encoded.ids)
            except Exception:
                continue

            # Check if decomposition is reasonable
            if 1 < len(target_ids) <= self.config.max_decomposition_tokens:
                # Confidence decreases with more tokens
                confidence = 1.0 - (len(target_ids) - 1) * 0.15
                confidence = max(0.3, confidence)

                mappings[source_id] = TokenMapping(
                    source_token_id=source_id,
                    target_token_id=target_ids[0],  # Primary token
                    method=AlignmentMethod.DECOMPOSED,
                    confidence=confidence,
                    decomposition=target_ids,
                )

        return mappings

    def _semantic_match(
        self,
        unmapped_ids: list[int],
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> dict[int, TokenMapping]:
        """
        Find semantically similar tokens via embedding cosine similarity.

        For each unmapped source token, find the most similar target token
        by cosine similarity in embedding space.
        """
        if len(unmapped_ids) == 0:
            return {}

        # Normalize embeddings for cosine similarity
        source_norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)

        source_normed = source_embeddings / np.maximum(source_norms, 1e-8)
        target_normed = target_embeddings / np.maximum(target_norms, 1e-8)

        mappings = {}

        # Process in batches
        for i in range(0, len(unmapped_ids), self.config.batch_size):
            batch_ids = unmapped_ids[i : i + self.config.batch_size]

            # Get source embeddings for batch
            batch_embeds = source_normed[batch_ids]  # [batch, dim]

            # Compute similarities to all target tokens
            similarities = batch_embeds @ target_normed.T  # [batch, target_vocab]

            # Find best matches
            best_targets = np.argmax(similarities, axis=1)
            best_scores = similarities[np.arange(len(batch_ids)), best_targets]

            for j, source_id in enumerate(batch_ids):
                score = float(best_scores[j])
                if score >= self.config.semantic_threshold:
                    mappings[source_id] = TokenMapping(
                        source_token_id=source_id,
                        target_token_id=int(best_targets[j]),
                        method=AlignmentMethod.SEMANTIC,
                        confidence=score,
                    )

        return mappings


def load_tokenizer(model_path: str) -> Tokenizer:
    """Load tokenizer from model directory."""
    path = Path(model_path).expanduser().resolve()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {path}")
    return Tokenizer.from_file(str(path))


def compare_model_vocabularies(
    source_model_path: str,
    target_model_path: str,
) -> dict:
    """
    Quick vocabulary compatibility check between two models.

    Args:
        source_model_path: Path to source model directory.
        target_model_path: Path to target model directory.

    Returns:
        Dictionary with compatibility statistics.
    """
    source_tok = load_tokenizer(source_model_path)
    target_tok = load_tokenizer(target_model_path)

    aligner = VocabularyAligner()
    return aligner.compare_vocabularies(source_tok, target_tok)


def format_alignment_report(result: VocabularyAlignmentResult) -> str:
    """Format alignment result for human readability."""
    lines = [
        "=" * 60,
        "VOCABULARY ALIGNMENT REPORT",
        "=" * 60,
        "",
        f"Source Vocabulary: {result.source_vocab_size:,} tokens",
        f"Target Vocabulary: {result.target_vocab_size:,} tokens",
        "",
        "Alignment Breakdown:",
        f"  Exact Match:    {result.overlap_count:,} ({result.overlap_ratio:.1%})",
        f"  Decomposition:  {result.decomposed_count:,}",
        f"  Semantic:       {result.semantic_count:,}",
        f"  Unmapped:       {result.unmapped_count:,}",
        "",
        f"Total Coverage: {result.coverage:.1%}",
        f"Merge Feasibility: {result.merge_feasibility.upper()}",
        f"Recommended Method: {result.recommended_method}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
