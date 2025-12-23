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

from .vocabulary_analyzer import (
    VocabularyStats,
    VocabularyCompatibility,
    VocabularyAnalyzer,
    TokenizerType,
)
from .embedding_projector import (
    ProjectionStrategy,
    ProjectionConfig,
    ProjectionResult,
    EmbeddingProjector,
)
from .alignment_map import (
    TokenAlignment,
    VocabularyAlignmentMap,
    AlignmentQuality,
)
from .cross_vocab_merger import (
    CrossVocabMergeConfig,
    CrossVocabMergeResult,
    CrossVocabMerger,
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
    # Merger
    "CrossVocabMergeConfig",
    "CrossVocabMergeResult",
    "CrossVocabMerger",
]
