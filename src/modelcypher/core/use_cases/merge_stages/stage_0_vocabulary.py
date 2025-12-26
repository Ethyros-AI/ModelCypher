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
Stage 0: VOCABULARY ALIGNMENT - Cross-vocabulary merging.

Uses the superior CrossVocabMerger pipeline:
1. Analyze vocabularies (stats, compatibility)
2. Build token alignment map (exact + embedding similarity)
3. Project source embeddings to target space (Procrustes/OT)
4. Blend aligned embeddings with quality-weighted alpha
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain._backend import get_default_backend

logger = logging.getLogger(__name__)


@dataclass
class VocabularyConfig:
    """Configuration for Stage 0 vocabulary alignment."""

    # Projection strategy: procrustes, pca, optimal_transport, cca
    projection_strategy: str = "procrustes"

    # Alignment thresholds
    similarity_threshold: float = 0.8
    confidence_threshold: float = 0.5

    # Embedding blending
    blend_alpha: float = 0.5
    preserve_special_tokens: bool = True

    # Quality thresholds
    min_compatibility_score: float = 0.3
    min_coverage: float = 0.5

    # Advanced
    use_embedding_similarity: bool = True
    anchor_count: int = 1000
    max_similarity_pairs: int = 5_000_000
    max_unmapped_similarity: int = 5000
    max_prefix_length: int = 8
    max_prefix_matches: int = 3


@dataclass
class VocabularyResult:
    """Result of Stage 0 vocabulary alignment."""

    modified_weights: dict[str, "object"]
    metrics: dict[str, Any]
    was_aligned: bool


def stage_vocabulary_align(
    source_weights: dict[str, "object"],
    target_weights: dict[str, "object"],
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
    config: VocabularyConfig,
) -> VocabularyResult:
    """
    Stage 0: Align source vocabulary to target vocabulary.

    Uses CrossVocabMerger for sophisticated vocabulary alignment with:
    - Multi-strategy projection (Procrustes, PCA, Optimal Transport)
    - Embedding similarity for unmapped tokens
    - Quality-weighted blending

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        config: Vocabulary alignment configuration

    Returns:
        VocabularyResult with modified weights, metrics, and alignment status
    """
    backend = get_default_backend()

    metrics: dict[str, Any] = {
        "enabled": True,
        "tokenizers_provided": source_tokenizer is not None and target_tokenizer is not None,
    }

    # Skip if tokenizers not provided
    if source_tokenizer is None or target_tokenizer is None:
        logger.info("Tokenizers not provided, skipping vocabulary alignment")
        metrics["skipped"] = True
        metrics["reason"] = "tokenizers_not_provided"
        return VocabularyResult(source_weights, metrics, False)

    # Find embedding layer keys
    embed_keys = [k for k in source_weights if "embed" in k.lower() and "weight" in k.lower()]
    if not embed_keys:
        logger.info("No embedding layer found, skipping vocabulary alignment")
        metrics["skipped"] = True
        metrics["reason"] = "no_embedding_layer"
        return VocabularyResult(source_weights, metrics, False)

    # Import CrossVocabMerger
    try:
        from modelcypher.core.domain.vocabulary.cross_vocab_merger import (
            CrossVocabMergeConfig,
            CrossVocabMerger,
        )
        from modelcypher.core.domain.vocabulary.embedding_projector import (
            ProjectionStrategy,
        )
    except ImportError as e:
        logger.warning("CrossVocabMerger not available: %s", e)
        metrics["skipped"] = True
        metrics["reason"] = f"import_error: {e}"
        return VocabularyResult(source_weights, metrics, False)

    # Map config string to ProjectionStrategy enum
    strategy_map = {
        "procrustes": ProjectionStrategy.PROCRUSTES,
        "pca": ProjectionStrategy.PCA,
        "optimal_transport": ProjectionStrategy.OPTIMAL_TRANSPORT,
        "cca": ProjectionStrategy.CCA,
        "truncate": ProjectionStrategy.TRUNCATE,
    }
    projection_strategy = strategy_map.get(
        config.projection_strategy.lower(),
        ProjectionStrategy.PROCRUSTES,
    )

    # Configure merger
    merge_config = CrossVocabMergeConfig(
        projection_strategy=projection_strategy,
        similarity_threshold=config.similarity_threshold,
        confidence_threshold=config.confidence_threshold,
        blend_alpha=config.blend_alpha,
        preserve_special_tokens=config.preserve_special_tokens,
        use_embedding_similarity=config.use_embedding_similarity,
        anchor_count=config.anchor_count,
        max_similarity_pairs=config.max_similarity_pairs,
        max_unmapped_similarity=config.max_unmapped_similarity,
        max_prefix_length=config.max_prefix_length,
        max_prefix_matches=config.max_prefix_matches,
    )

    merger = CrossVocabMerger(merge_config)

    # Extract vocabulary mappings from tokenizers
    source_vocab = _extract_vocab(source_tokenizer)
    target_vocab = _extract_vocab(target_tokenizer)

    if source_vocab is None or target_vocab is None:
        logger.warning("Could not extract vocabulary from tokenizers")
        metrics["skipped"] = True
        metrics["reason"] = "vocab_extraction_failed"
        return VocabularyResult(source_weights, metrics, False)

    metrics["source_vocab_size"] = len(source_vocab)
    metrics["target_vocab_size"] = len(target_vocab)

    # Check for vocab compatibility before doing expensive operations
    overlap = set(source_vocab.keys()) & set(target_vocab.keys())
    overlap_ratio = len(overlap) / max(len(source_vocab), 1)
    metrics["overlap_count"] = len(overlap)
    metrics["overlap_ratio"] = overlap_ratio

    if overlap_ratio > 0.95:
        logger.info(
            "Vocabulary overlap %.1f%% - vocabularies compatible, skipping alignment",
            overlap_ratio * 100,
        )
        metrics["skipped"] = True
        metrics["reason"] = "compatible_vocabulary"
        return VocabularyResult(source_weights, metrics, False)

    # Apply merger to each embedding layer
    modified_weights = source_weights.copy()
    aligned_layers = 0

    for embed_key in embed_keys:
        source_embed = source_weights.get(embed_key)
        target_embed = target_weights.get(embed_key)

        if source_embed is None or target_embed is None:
            logger.warning("Missing embedding for key %s", embed_key)
            continue

        # Vocabulary is the 2D compression plane; dequantize the 1D binary basis first.
        from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

        source_embed = dequantize_if_needed(source_embed, embed_key, source_weights, backend)
        target_embed = dequantize_if_needed(target_embed, embed_key, target_weights, backend)

        # Ensure backend arrays with stable dtype for linear algebra.
        source_embed = backend.array(source_embed)
        target_embed = backend.array(target_embed)
        source_embed = backend.astype(source_embed, "float32")
        target_embed = backend.astype(target_embed, "float32")
        backend.eval(source_embed, target_embed)

        logger.info(
            "Aligning %s: source=%s, target=%s",
            embed_key,
            source_embed.shape,
            target_embed.shape,
        )

        try:
            # Run CrossVocabMerger
            result = merger.merge(
                source_embeddings=source_embed,
                target_embeddings=target_embed,
                source_vocab=source_vocab,
                target_vocab=target_vocab,
            )

            # Check quality
            quality_metrics = merger.analyze_merge_quality(result)

            if result.compatibility.compatibility_score < config.min_compatibility_score:
                logger.warning(
                    "Low compatibility score %.2f for %s (continuing alignment)",
                    result.compatibility.compatibility_score,
                    embed_key,
                )
                metrics[f"{embed_key}_warning"] = "low_compatibility"

            if result.alignment_map.coverage < config.min_coverage:
                logger.warning(
                    "Low coverage %.2f for %s (continuing alignment)",
                    result.alignment_map.coverage,
                    embed_key,
                )
                metrics[f"{embed_key}_warning"] = "low_coverage"

            # Convert result to backend array format, preserving original dtype
            merged_embed = result.merged_embeddings

            # Ensure we have a backend array
            if hasattr(merged_embed, "numpy"):
                # PyTorch or TensorFlow tensor - convert via numpy
                merged_np = merged_embed.numpy()
                merged_embed = backend.array(merged_np)
            elif not hasattr(merged_embed, "shape") or not hasattr(merged_embed, "dtype"):
                # Raw python data - convert to backend array
                merged_embed = backend.array(merged_embed)

            # Keep float32 to preserve the aligned vocabulary plane; requantize at final save.
            merged_embed = backend.astype(merged_embed, "float32")
            backend.eval(merged_embed)
            # Keep on CPU to reduce GPU pressure; will be moved to backend on demand.
            modified_weights[embed_key] = backend.to_numpy(merged_embed)
            aligned_layers += 1

            # Record metrics
            metrics[f"{embed_key}_projection_strategy"] = config.projection_strategy
            metrics[f"{embed_key}_alignment_coverage"] = result.alignment_map.coverage
            metrics[f"{embed_key}_alignment_confidence"] = result.alignment_map.mean_confidence
            metrics[f"{embed_key}_projection_score"] = result.projection_result.alignment_score
            metrics[f"{embed_key}_compatibility_score"] = result.compatibility.compatibility_score
            metrics[f"{embed_key}_overall_quality"] = quality_metrics["overall_quality_score"]
            metrics[f"{embed_key}_recommendation"] = quality_metrics["recommendation"]
            metrics[f"{embed_key}_warnings"] = result.warnings

            logger.info(
                "Aligned %s: coverage=%.2f, quality=%.2f, %s",
                embed_key,
                result.alignment_map.coverage,
                quality_metrics["overall_quality_score"],
                quality_metrics["recommendation"],
            )

        except Exception as e:
            logger.error("Failed to align %s: %s", embed_key, e)
            metrics[f"{embed_key}_error"] = str(e)
            continue

    metrics["aligned_layers"] = aligned_layers
    metrics["alignment_applied"] = aligned_layers > 0

    if aligned_layers > 0:
        logger.info("Vocabulary alignment applied to %d layers", aligned_layers)
    else:
        logger.info("No vocabulary alignment applied")

    return VocabularyResult(modified_weights, metrics, aligned_layers > 0)


def _extract_vocab(tokenizer: Any) -> dict[str, int] | None:
    """Extract vocabulary mapping from tokenizer."""
    # Try different tokenizer APIs
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    if hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
        if isinstance(vocab, dict):
            return vocab
    if hasattr(tokenizer, "encoder"):
        return tokenizer.encoder
    if hasattr(tokenizer, "token_to_id"):
        # Tokenizers library - need to iterate
        try:
            vocab = {}
            for token in tokenizer.get_vocab():
                vocab[token] = tokenizer.token_to_id(token)
            return vocab
        except Exception:
            pass

    logger.warning("Could not extract vocabulary from tokenizer type %s", type(tokenizer))
    return None
