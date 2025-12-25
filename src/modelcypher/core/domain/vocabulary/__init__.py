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
Cross-Vocabulary Merging Domain.

Provides algorithms for merging models with different vocabularies/tokenizers:
- VocabularyAnalyzer: Detect vocab size, embedding dimensions, tokenizer type
- EmbeddingProjector: Project embeddings between different vocabulary spaces
- VocabularyAlignmentMap: Store token mappings and projection matrices
- CrossVocabMerger: Orchestrate full cross-vocabulary merging pipeline

This module enables merging models that have:
- Different vocabulary sizes
- Different tokenizers (e.g., Llama tokenizer vs Qwen tokenizer)
- Different embedding dimensions
"""

from .alignment_map import (
    AlignmentQuality,
    TokenAlignment,
    TokenizerComparisonResult,
    VocabularyAlignmentMap,
    build_alignment_from_vocabs,
    compare_tokenizers,
    format_comparison_report,
)
from .cross_vocab_merger import (
    CrossVocabMergeConfig,
    CrossVocabMerger,
    CrossVocabMergeResult,
)
from .embedding_projector import (
    EmbeddingProjector,
    ProjectionConfig,
    ProjectionResult,
    ProjectionStrategy,
)
from .vocabulary_analyzer import (
    TokenizerType,
    VocabularyAnalyzer,
    VocabularyCompatibility,
    VocabularyStats,
)

__all__ = [
    # Analyzer
    "VocabularyStats",
    "VocabularyCompatibility",
    "VocabularyAnalyzer",
    "TokenizerType",
    # Projector
    "ProjectionStrategy",
    "ProjectionConfig",
    "ProjectionResult",
    "EmbeddingProjector",
    # Alignment
    "TokenAlignment",
    "VocabularyAlignmentMap",
    "AlignmentQuality",
    "TokenizerComparisonResult",
    "compare_tokenizers",
    "format_comparison_report",
    "build_alignment_from_vocabs",
    # Merger
    "CrossVocabMergeConfig",
    "CrossVocabMergeResult",
    "CrossVocabMerger",
]
