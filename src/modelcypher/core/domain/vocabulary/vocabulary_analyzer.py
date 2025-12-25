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
Vocabulary Analyzer for Cross-Model Merging.

Analyzes vocabulary statistics and detects compatibility between models
with different tokenizers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class TokenizerType(str, Enum):
    """Known tokenizer types."""

    BPE = "bpe"  # GPT-style BPE
    SENTENCEPIECE = "sentencepiece"  # Llama, T5
    WORDPIECE = "wordpiece"  # BERT
    UNIGRAM = "unigram"  # SentencePiece unigram
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VocabularyStats:
    """Statistics about a model's vocabulary and embeddings."""

    vocab_size: int
    hidden_dim: int
    embedding_mean_norm: float
    embedding_std: float
    tokenizer_type: TokenizerType
    special_token_count: int
    has_tie_weights: bool  # embed_tokens == lm_head

    # Optional extended stats
    mean_token_length: float = 0.0
    bpe_merge_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "embedding_mean_norm": self.embedding_mean_norm,
            "embedding_std": self.embedding_std,
            "tokenizer_type": self.tokenizer_type.value,
            "special_token_count": self.special_token_count,
            "has_tie_weights": self.has_tie_weights,
            "mean_token_length": self.mean_token_length,
            "bpe_merge_count": self.bpe_merge_count,
        }


@dataclass(frozen=True)
class VocabularyCompatibility:
    """Vocabulary geometry assessment between two models."""

    compatibility_score: float  # 0.0 (high effort) to 1.0 (minimal effort)
    vocab_overlap_ratio: float  # Fraction of tokens shared
    dimension_ratio: float  # hidden_dim ratio
    requires_projection: bool  # Needs embedding projection
    requires_vocab_mapping: bool  # Needs token ID remapping
    recommendation: str

    # Detailed analysis
    shared_token_count: int = 0
    source_only_tokens: int = 0
    target_only_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compatibility_score": self.compatibility_score,
            "vocab_overlap_ratio": self.vocab_overlap_ratio,
            "dimension_ratio": self.dimension_ratio,
            "requires_projection": self.requires_projection,
            "requires_vocab_mapping": self.requires_vocab_mapping,
            "recommendation": self.recommendation,
            "shared_token_count": self.shared_token_count,
            "source_only_tokens": self.source_only_tokens,
            "target_only_tokens": self.target_only_tokens,
        }


class VocabularyAnalyzer:
    """
    Analyzes model vocabularies for merge compatibility.

    Detects:
    - Vocabulary size and embedding dimensions
    - Tokenizer type (BPE, SentencePiece, etc.)
    - Embedding statistics (norms, variance)
    - Token overlap between models
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze_embeddings(
        self,
        embeddings: "Array",
        tokenizer_config: dict[str, Any] | None = None,
    ) -> VocabularyStats:
        """
        Analyze embedding matrix statistics.

        Args:
            embeddings: Embedding matrix [vocab_size, hidden_dim]
            tokenizer_config: Optional tokenizer configuration

        Returns:
            VocabularyStats with computed statistics
        """
        b = self._backend

        # Get dimensions
        shape = embeddings.shape
        if len(shape) != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {shape}")

        vocab_size, hidden_dim = shape

        # Compute embedding statistics
        norms = b.norm(embeddings, axis=1)
        mean_norm = float(b.to_numpy(b.mean(norms)))

        # Compute standard deviation across all values
        flat = b.reshape(embeddings, (-1,))
        mean_val = b.mean(flat)
        variance = b.mean((flat - mean_val) ** 2)
        std_val = float(b.to_numpy(b.sqrt(variance)))

        # Detect tokenizer type
        tokenizer_type = self._detect_tokenizer_type(tokenizer_config)

        # Count special tokens
        special_count = self._count_special_tokens(tokenizer_config)

        # Detect weight tying (would need lm_head to check, default to False)
        has_tie = False

        return VocabularyStats(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embedding_mean_norm=mean_norm,
            embedding_std=std_val,
            tokenizer_type=tokenizer_type,
            special_token_count=special_count,
            has_tie_weights=has_tie,
        )

    def analyze_compatibility(
        self,
        source_stats: VocabularyStats,
        target_stats: VocabularyStats,
        source_vocab: dict[str, int] | None = None,
        target_vocab: dict[str, int] | None = None,
    ) -> VocabularyCompatibility:
        """
        Analyze compatibility between source and target vocabularies.

        Args:
            source_stats: Source model vocabulary statistics
            target_stats: Target model vocabulary statistics
            source_vocab: Optional source token->id mapping
            target_vocab: Optional target token->id mapping

        Returns:
            VocabularyCompatibility assessment
        """
        # Check dimension compatibility
        dim_ratio = source_stats.hidden_dim / target_stats.hidden_dim
        requires_projection = source_stats.hidden_dim != target_stats.hidden_dim

        # Check vocabulary overlap
        vocab_overlap = 0.0
        shared_count = 0
        source_only = 0
        target_only = 0

        if source_vocab and target_vocab:
            source_set = set(source_vocab.keys())
            target_set = set(target_vocab.keys())
            shared = source_set & target_set
            shared_count = len(shared)
            source_only = len(source_set - target_set)
            target_only = len(target_set - source_set)

            # Compute overlap as Jaccard index
            union_size = len(source_set | target_set)
            vocab_overlap = shared_count / union_size if union_size > 0 else 0.0
        else:
            # Without vocab dicts, estimate based on size
            min_size = min(source_stats.vocab_size, target_stats.vocab_size)
            max_size = max(source_stats.vocab_size, target_stats.vocab_size)
            vocab_overlap = min_size / max_size if max_size > 0 else 0.0

        requires_mapping = (
            source_stats.vocab_size != target_stats.vocab_size or vocab_overlap < 0.99
        )

        # Dimension ratio and overlap ratio provide compatibility information.
        # These geometric measurements specify the alignment method.
        compatibility_score = vocab_overlap

        # Recommend alignment method based on geometric properties
        if not requires_projection and vocab_overlap > 0.99:
            recommendation = "Direct merge: vocabularies aligned, dimensions match."
        elif not requires_projection:
            recommendation = (
                f"Vocabulary mapping required ({shared_count} shared tokens). "
                "Use token ID remapping with shared anchor alignment."
            )
        elif vocab_overlap > 0.5:
            recommendation = (
                f"Dimension projection required (ratio {dim_ratio:.2f}). "
                "Use PCA or Procrustes to align embedding spaces."
            )
        else:
            recommendation = (
                f"Cross-space alignment required (dim ratio {dim_ratio:.2f}, "
                f"overlap {vocab_overlap:.2%}). Use optimal transport for embedding alignment."
            )

        return VocabularyCompatibility(
            compatibility_score=compatibility_score,
            vocab_overlap_ratio=vocab_overlap,
            dimension_ratio=dim_ratio,
            requires_projection=requires_projection,
            requires_vocab_mapping=requires_mapping,
            recommendation=recommendation,
            shared_token_count=shared_count,
            source_only_tokens=source_only,
            target_only_tokens=target_only,
        )

    def _detect_tokenizer_type(
        self,
        config: dict[str, Any] | None,
    ) -> TokenizerType:
        """Detect tokenizer type from config."""
        if not config:
            return TokenizerType.UNKNOWN

        # Check for common indicators
        tokenizer_class = config.get("tokenizer_class", "").lower()
        model_type = config.get("model_type", "").lower()

        if "bpe" in tokenizer_class or "gpt" in tokenizer_class:
            return TokenizerType.BPE
        if "sentencepiece" in tokenizer_class or "llama" in model_type:
            return TokenizerType.SENTENCEPIECE
        if "wordpiece" in tokenizer_class or "bert" in model_type:
            return TokenizerType.WORDPIECE
        if "unigram" in tokenizer_class:
            return TokenizerType.UNIGRAM

        # Check tokenizer.json structure
        if "model" in config:
            model_config = config["model"]
            if isinstance(model_config, dict):
                model_type_str = model_config.get("type", "").lower()
                if model_type_str == "bpe":
                    return TokenizerType.BPE
                if model_type_str == "unigram":
                    return TokenizerType.UNIGRAM
                if model_type_str == "wordpiece":
                    return TokenizerType.WORDPIECE

        return TokenizerType.UNKNOWN

    def _count_special_tokens(
        self,
        config: dict[str, Any] | None,
    ) -> int:
        """Count special tokens from config."""
        if not config:
            return 0

        count = 0
        special_keys = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ]

        for key in special_keys:
            if config.get(key) is not None:
                count += 1

        # Check added_tokens
        added = config.get("added_tokens", [])
        if isinstance(added, list):
            count += len([t for t in added if isinstance(t, dict) and t.get("special", False)])

        return count

    def compute_token_overlap(
        self,
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> tuple[set[str], set[str], set[str]]:
        """
        Compute token overlap between vocabularies.

        Returns:
            Tuple of (shared, source_only, target_only) token sets
        """
        source_set = set(source_vocab.keys())
        target_set = set(target_vocab.keys())

        shared = source_set & target_set
        source_only = source_set - target_set
        target_only = target_set - source_set

        return shared, source_only, target_only
