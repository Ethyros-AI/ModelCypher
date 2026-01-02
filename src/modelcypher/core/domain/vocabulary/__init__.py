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
Vocabulary Alignment Utilities.

Provides exact vocabulary alignment helpers used for tokenizer comparison
and probe ID mapping. No embedding interpolation or blending.
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
    CrossVocabMergeResult,
    CrossVocabMerger,
)
from .embedding_projector import (
    EmbeddingProjector,
    ProjectionConfig,
    ProjectionResult,
    ProjectionStrategy,
)
from .vocabulary_analyzer import (
    TokenizerType,
    VocabularyAlignment,
    VocabularyAnalyzer,
    VocabularyStats,
)
__all__ = [
    # Alignment
    "TokenAlignment",
    "VocabularyAlignmentMap",
    "AlignmentQuality",
    "TokenizerComparisonResult",
    "compare_tokenizers",
    "format_comparison_report",
    "build_alignment_from_vocabs",
    # Analyzer
    "VocabularyAnalyzer",
    "VocabularyStats",
    "VocabularyAlignment",
    "TokenizerType",
    # Projection
    "EmbeddingProjector",
    "ProjectionConfig",
    "ProjectionResult",
    "ProjectionStrategy",
    # Merger
    "CrossVocabMerger",
    "CrossVocabMergeConfig",
    "CrossVocabMergeResult",
]
