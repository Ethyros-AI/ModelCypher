"""
Stage 0: VOCABULARY ALIGNMENT - Cross-vocabulary merging.

Detects vocabulary mismatch and applies embedding bridge if needed.
Uses intelligent method selection: FVT → Procrustes → Affine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VocabularyConfig:
    """Configuration for Stage 0 vocabulary alignment."""

    vocab_bridge_method: str = "auto"
    vocab_quality_threshold: float = 0.5
    vocab_compatible_threshold: float = 0.95
    vocab_min_anchor_pairs: int = 10
    vocab_use_semantic_primes: bool = True


@dataclass
class VocabularyResult:
    """Result of Stage 0 vocabulary alignment."""

    modified_weights: dict[str, np.ndarray]
    metrics: dict[str, Any]
    was_aligned: bool


def stage_vocabulary_align(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    source_tokenizer: Optional[Any],
    target_tokenizer: Optional[Any],
    config: VocabularyConfig,
) -> VocabularyResult:
    """
    Stage 0: Align source vocabulary to target vocabulary.

    Detects vocabulary mismatch and applies embedding bridge if needed.
    Uses intelligent method selection: FVT → Procrustes → Affine.

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        config: Vocabulary alignment configuration

    Returns:
        VocabularyResult with modified weights, metrics, and alignment status
    """
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

    # Import vocabulary alignment modules
    try:
        from modelcypher.core.domain.merging.vocabulary_alignment import (
            VocabularyAligner,
        )
        from modelcypher.core.domain.merging.embedding_bridge import (
            EmbeddingBridgeBuilder,
            EmbeddingBridgeConfig,
        )
    except ImportError as e:
        logger.warning("Vocabulary alignment modules not available: %s", e)
        metrics["skipped"] = True
        metrics["reason"] = f"import_error: {e}"
        return VocabularyResult(source_weights, metrics, False)

    # Align vocabularies
    aligner = VocabularyAligner()
    alignment = aligner.align(source_tokenizer, target_tokenizer)

    metrics["source_vocab_size"] = alignment.source_vocab_size
    metrics["target_vocab_size"] = alignment.target_vocab_size
    metrics["overlap_count"] = alignment.overlap_count
    metrics["overlap_ratio"] = alignment.overlap_ratio
    metrics["coverage"] = alignment.coverage
    metrics["recommended_method"] = alignment.recommended_method
    metrics["merge_feasibility"] = alignment.merge_feasibility

    # Check if vocabulary is already compatible
    if alignment.overlap_ratio >= config.vocab_compatible_threshold:
        logger.info(
            "Vocabulary overlap %.1f%% >= %.1f%% threshold, no bridge needed",
            alignment.overlap_ratio * 100,
            config.vocab_compatible_threshold * 100,
        )
        metrics["bridge_applied"] = False
        metrics["reason"] = "compatible_vocabulary"
        return VocabularyResult(source_weights, metrics, False)

    # Check feasibility
    if alignment.merge_feasibility == "infeasible":
        logger.warning(
            "Vocabulary merge marked as infeasible (coverage: %.1f%%)",
            alignment.coverage * 100,
        )
        metrics["bridge_applied"] = False
        metrics["reason"] = "infeasible_vocabulary"
        return VocabularyResult(source_weights, metrics, False)

    # Configure bridge builder
    bridge_config = EmbeddingBridgeConfig(
        auto_select=(config.vocab_bridge_method == "auto"),
        quality_threshold=config.vocab_quality_threshold,
        min_anchor_pairs=config.vocab_min_anchor_pairs,
        use_semantic_primes=config.vocab_use_semantic_primes,
    )

    if config.vocab_bridge_method != "auto":
        bridge_config = EmbeddingBridgeConfig(
            auto_select=False,
            fallback_chain=(config.vocab_bridge_method,),
            quality_threshold=config.vocab_quality_threshold,
            min_anchor_pairs=config.vocab_min_anchor_pairs,
        )

    bridge_builder = EmbeddingBridgeBuilder(bridge_config)

    # Apply bridge to each embedding layer
    modified_weights = source_weights.copy()
    bridged_layers = 0

    for embed_key in embed_keys:
        source_embed = source_weights.get(embed_key)
        target_embed = target_weights.get(embed_key)

        if source_embed is None or target_embed is None:
            continue

        if source_embed.shape[1] != target_embed.shape[1]:
            logger.warning(
                "Hidden dimension mismatch for %s: %d vs %d",
                embed_key,
                source_embed.shape[1],
                target_embed.shape[1],
            )
            continue

        # Build embedding bridge
        result = bridge_builder.build(
            source_embed,
            target_embed,
            alignment,
        )

        if result.alignment_quality < config.vocab_quality_threshold:
            logger.warning(
                "Bridge quality %.3f below threshold %.3f for %s",
                result.alignment_quality,
                config.vocab_quality_threshold,
                embed_key,
            )
            metrics[f"{embed_key}_quality"] = result.alignment_quality
            metrics[f"{embed_key}_warnings"] = result.warnings
            continue

        modified_weights[embed_key] = result.bridged_embeddings
        bridged_layers += 1

        metrics[f"{embed_key}_method"] = result.method_used.value
        metrics[f"{embed_key}_quality"] = result.alignment_quality
        logger.info(
            "Applied %s bridge to %s (quality: %.3f)",
            result.method_used.value.upper(),
            embed_key,
            result.alignment_quality,
        )

    metrics["bridge_applied"] = bridged_layers > 0
    metrics["bridged_layers"] = bridged_layers

    return VocabularyResult(modified_weights, metrics, bridged_layers > 0)
