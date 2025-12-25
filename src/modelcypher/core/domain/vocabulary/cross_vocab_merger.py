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
Cross-Vocabulary Merger.

Orchestrates full cross-vocabulary merging pipeline for models with
different tokenizers/vocabularies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

from .alignment_map import (
    AlignmentQuality,
    TokenAlignment,
    VocabularyAlignmentMap,
    build_alignment_from_vocabs,
)
from .embedding_projector import (
    EmbeddingProjector,
    ProjectionConfig,
    ProjectionResult,
    ProjectionStrategy,
)
from .vocabulary_analyzer import (
    VocabularyAnalyzer,
    VocabularyCompatibility,
    VocabularyStats,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class CrossVocabMergeConfig:
    """Configuration for cross-vocabulary merging."""

    # Projection strategy
    projection_strategy: ProjectionStrategy = ProjectionStrategy.PROCRUSTES

    # Alignment thresholds
    similarity_threshold: float = 0.8  # Min similarity for similar match
    confidence_threshold: float = 0.5  # Min confidence to include alignment

    # Embedding blending
    blend_alpha: float = 0.5  # Weight for source embeddings (0=target, 1=source)
    preserve_special_tokens: bool = True  # Keep target special tokens unchanged

    # Advanced options
    use_embedding_similarity: bool = True  # Use embedding cosine for alignment
    max_alignments_per_token: int = 3  # Max target tokens per source token
    anchor_count: int = 1000  # Anchors for projection alignment
    regularization: float = 1e-6

    def to_projection_config(self) -> ProjectionConfig:
        """Convert to ProjectionConfig."""
        return ProjectionConfig(
            strategy=self.projection_strategy,
            regularization=self.regularization,
            anchor_count=self.anchor_count,
            preserve_norms=True,
        )


@dataclass
class CrossVocabMergeResult:
    """Result of cross-vocabulary merging."""

    merged_embeddings: "Array"
    output_vocab_size: int
    output_hidden_dim: int

    # Alignment info
    alignment_map: VocabularyAlignmentMap
    projection_result: ProjectionResult

    # Quality metrics
    compatibility: VocabularyCompatibility
    source_stats: VocabularyStats
    target_stats: VocabularyStats

    # Diagnostics
    tokens_preserved_from_source: int
    tokens_preserved_from_target: int
    tokens_interpolated: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_vocab_size": self.output_vocab_size,
            "output_hidden_dim": self.output_hidden_dim,
            "alignment_summary": self.alignment_map.to_dict(),
            "projection_summary": self.projection_result.to_dict(),
            "compatibility_summary": self.compatibility.to_dict(),
            "source_stats": self.source_stats.to_dict(),
            "target_stats": self.target_stats.to_dict(),
            "tokens_preserved_from_source": self.tokens_preserved_from_source,
            "tokens_preserved_from_target": self.tokens_preserved_from_target,
            "tokens_interpolated": self.tokens_interpolated,
            "warnings": self.warnings,
        }


class CrossVocabMerger:
    """
    Merges embeddings from models with different vocabularies.

    Pipeline:
    1. Analyze vocabularies (stats, compatibility)
    2. Build token alignment map
    3. Project source embeddings to target space
    4. Blend aligned embeddings
    5. Handle unaligned tokens (interpolation or dropout)

    Supports:
    - Different vocabulary sizes
    - Different embedding dimensions
    - Different tokenizer types (BPE, SentencePiece, etc.)
    """

    def __init__(
        self,
        config: CrossVocabMergeConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or CrossVocabMergeConfig()
        self._backend = backend or get_default_backend()

        self._analyzer = VocabularyAnalyzer(backend=self._backend)
        self._projector = EmbeddingProjector(
            config=self.config.to_projection_config(),
            backend=self._backend,
        )

    def merge(
        self,
        source_embeddings: "Array",
        target_embeddings: "Array",
        source_vocab: dict[str, int] | None = None,
        target_vocab: dict[str, int] | None = None,
        source_tokenizer_config: dict[str, Any] | None = None,
        target_tokenizer_config: dict[str, Any] | None = None,
    ) -> CrossVocabMergeResult:
        """
        Merge source embeddings into target vocabulary space.

        Args:
            source_embeddings: Source embedding matrix [vocab, hidden]
            target_embeddings: Target embedding matrix [vocab, hidden]
            source_vocab: Optional source token->id mapping
            target_vocab: Optional target token->id mapping
            source_tokenizer_config: Optional source tokenizer config
            target_tokenizer_config: Optional target tokenizer config

        Returns:
            CrossVocabMergeResult with merged embeddings
        """
        b = self._backend
        warnings = []

        # Step 1: Analyze vocabularies
        logger.info("Analyzing source vocabulary...")
        source_stats = self._analyzer.analyze_embeddings(source_embeddings, source_tokenizer_config)

        logger.info("Analyzing target vocabulary...")
        target_stats = self._analyzer.analyze_embeddings(target_embeddings, target_tokenizer_config)

        # Step 2: Check compatibility
        logger.info("Checking vocabulary compatibility...")
        compatibility = self._analyzer.analyze_compatibility(
            source_stats, target_stats, source_vocab, target_vocab
        )

        if not compatibility.is_compatible:
            warnings.append(f"Low compatibility score: {compatibility.compatibility_score:.2f}")
            warnings.append(compatibility.recommendation)

        # Step 3: Build alignment map
        logger.info("Building alignment map...")
        if source_vocab and target_vocab:
            alignment_map = self._build_embedding_alignment(
                source_embeddings,
                target_embeddings,
                source_vocab,
                target_vocab,
            )
        else:
            # No vocab dicts - use index-based alignment
            alignment_map = self._build_index_alignment(
                source_stats.vocab_size, target_stats.vocab_size
            )
            warnings.append("No vocabulary dicts provided - using index-based alignment")

        # Step 4: Project source embeddings
        logger.info(f"Projecting embeddings using {self.config.projection_strategy.value}...")
        shared_indices = self._get_shared_indices(alignment_map)
        projection_result = self._projector.project(
            source_embeddings, target_embeddings, shared_indices
        )

        # Step 5: Blend embeddings
        logger.info("Blending embeddings...")
        merged, blend_stats = self._blend_embeddings(
            projection_result.projected_embeddings,
            target_embeddings,
            alignment_map,
        )

        # Log results
        logger.info(f"Merge complete: {merged.shape[0]} tokens, {merged.shape[1]} dims")
        logger.info(f"Alignment coverage: {alignment_map.coverage:.1%}")
        logger.info(f"Projection alignment score: {projection_result.alignment_score:.2f}")

        return CrossVocabMergeResult(
            merged_embeddings=merged,
            output_vocab_size=merged.shape[0],
            output_hidden_dim=merged.shape[1],
            alignment_map=alignment_map,
            projection_result=projection_result,
            compatibility=compatibility,
            source_stats=source_stats,
            target_stats=target_stats,
            tokens_preserved_from_source=blend_stats["source_preserved"],
            tokens_preserved_from_target=blend_stats["target_preserved"],
            tokens_interpolated=blend_stats["interpolated"],
            warnings=warnings,
        )

    def _build_embedding_alignment(
        self,
        source_embeddings: "Array",
        target_embeddings: "Array",
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> VocabularyAlignmentMap:
        """Build alignment using both string matching and embedding similarity."""
        b = self._backend

        # Start with string-based alignment
        alignment_map = build_alignment_from_vocabs(
            source_vocab, target_vocab, self.config.similarity_threshold
        )

        if not self.config.use_embedding_similarity:
            return alignment_map

        # Enhance unmapped tokens with embedding similarity
        target_id_to_token = {id_: token for token, id_ in target_vocab.items()}

        source_dim = source_embeddings.shape[1]
        target_dim = target_embeddings.shape[1]

        # Handle dimension mismatch for similarity computation
        if source_dim != target_dim:
            shared_dim = min(source_dim, target_dim)
            # Truncate both to shared dimension for similarity
            source_for_sim = source_embeddings[:, :shared_dim]
            target_for_sim = target_embeddings[:, :shared_dim]
        else:
            source_for_sim = source_embeddings
            target_for_sim = target_embeddings

        # Normalize embeddings for cosine similarity
        source_norms = b.norm(source_for_sim, axis=1, keepdims=True) + self.config.regularization
        target_norms = b.norm(target_for_sim, axis=1, keepdims=True) + self.config.regularization
        source_normalized = source_for_sim / source_norms
        target_normalized = target_for_sim / target_norms

        # Process unmapped tokens
        unmapped_count = 0
        for alignment in list(alignment_map.iter_alignments()):
            if alignment.quality != AlignmentQuality.UNMAPPED:
                continue

            # Compute cosine similarity with all target tokens
            if alignment.source_id >= source_normalized.shape[0]:
                continue
            source_vec = source_normalized[alignment.source_id]
            similarities = b.matmul(source_vec[None, :], target_normalized.T)[0]

            # Get top-k matches
            k = self.config.max_alignments_per_token
            sim_np = b.to_numpy(similarities)

            # Find top k indices
            top_indices = sim_np.argsort()[-k:][::-1]
            top_sims = sim_np[top_indices]

            # Filter by threshold
            mask = top_sims >= self.config.similarity_threshold
            if not mask.any():
                # Take best match even if below threshold
                top_indices = top_indices[:1]
                top_sims = top_sims[:1]
            else:
                top_indices = top_indices[mask]
                top_sims = top_sims[mask]

            if len(top_indices) > 0:
                # Create new alignment
                target_ids = [int(idx) for idx in top_indices]
                target_tokens = [target_id_to_token.get(idx, f"<{idx}>") for idx in target_ids]

                # Normalize weights
                weights = (
                    top_sims / top_sims.sum()
                    if top_sims.sum() > 0
                    else [1.0 / len(top_sims)] * len(top_sims)
                )
                weights = [float(w) for w in weights]

                # Determine quality
                max_sim = float(top_sims[0])
                if max_sim >= 0.95:
                    quality = AlignmentQuality.SIMILAR
                elif max_sim >= self.config.similarity_threshold:
                    quality = AlignmentQuality.APPROXIMATE
                else:
                    quality = AlignmentQuality.INTERPOLATED
                    unmapped_count += 1

                new_alignment = TokenAlignment(
                    source_id=alignment.source_id,
                    source_token=alignment.source_token,
                    target_ids=target_ids,
                    target_tokens=target_tokens,
                    weights=weights,
                    quality=quality,
                    confidence=max_sim,
                    metadata={"cosine_similarities": [float(s) for s in top_sims]},
                )

                # Update alignment map
                alignment_map.alignments[alignment.source_id] = new_alignment

                # Update stats
                if alignment.quality == AlignmentQuality.UNMAPPED:
                    alignment_map.unmapped_count -= 1
                if quality == AlignmentQuality.SIMILAR:
                    alignment_map.similar_matches += 1
                elif quality == AlignmentQuality.APPROXIMATE:
                    alignment_map.approximate_matches += 1
                elif quality == AlignmentQuality.INTERPOLATED:
                    alignment_map.interpolated_count += 1

        return alignment_map

    def _build_index_alignment(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
    ) -> VocabularyAlignmentMap:
        """Build simple index-based alignment when vocab dicts not available."""
        alignment_map = VocabularyAlignmentMap(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
        )

        shared_size = min(source_vocab_size, target_vocab_size)

        for i in range(source_vocab_size):
            if i < shared_size:
                # Direct index mapping
                alignment = TokenAlignment(
                    source_id=i,
                    source_token=f"<{i}>",
                    target_ids=[i],
                    target_tokens=[f"<{i}>"],
                    weights=[1.0],
                    quality=AlignmentQuality.EXACT,
                    confidence=1.0,
                )
            else:
                # No corresponding target token
                alignment = TokenAlignment(
                    source_id=i,
                    source_token=f"<{i}>",
                    target_ids=[],
                    target_tokens=[],
                    weights=[],
                    quality=AlignmentQuality.UNMAPPED,
                    confidence=0.0,
                )

            alignment_map.add_alignment(alignment)

        return alignment_map

    def _get_shared_indices(
        self,
        alignment_map: VocabularyAlignmentMap,
    ) -> tuple[list[int], list[int]] | None:
        """Extract shared token indices from alignment map."""
        source_indices = []
        target_indices = []

        for alignment in alignment_map.iter_alignments():
            if alignment.quality in (AlignmentQuality.EXACT, AlignmentQuality.SIMILAR):
                if len(alignment.target_ids) == 1:
                    source_indices.append(alignment.source_id)
                    target_indices.append(alignment.target_ids[0])

        if not source_indices:
            return None

        return source_indices, target_indices

    def _blend_embeddings(
        self,
        projected_source: "Array",
        target_embeddings: "Array",
        alignment_map: VocabularyAlignmentMap,
    ) -> tuple["Array", dict[str, int]]:
        """
        Blend projected source embeddings with target embeddings.

        Returns:
            Tuple of (merged_embeddings, blend_statistics)
        """
        b = self._backend

        target_vocab_size, hidden_dim = target_embeddings.shape
        alpha = self.config.blend_alpha

        # Initialize with target embeddings
        merged = b.array(b.to_numpy(target_embeddings).copy())

        stats = {
            "source_preserved": 0,
            "target_preserved": 0,
            "interpolated": 0,
            "blended": 0,
        }

        # Process each alignment
        for alignment in alignment_map.iter_alignments():
            if alignment.quality == AlignmentQuality.UNMAPPED:
                continue

            source_id = alignment.source_id
            if source_id >= projected_source.shape[0]:
                continue

            source_vec = projected_source[source_id]

            if len(alignment.target_ids) == 1:
                # One-to-one mapping
                target_id = alignment.target_ids[0]
                if target_id >= target_vocab_size:
                    continue

                target_vec = target_embeddings[target_id]

                # Skip special tokens if configured
                if self.config.preserve_special_tokens and self._is_special_token(
                    alignment.source_token
                ):
                    stats["target_preserved"] += 1
                    continue

                # Blend based on quality and confidence
                effective_alpha = alpha * alignment.confidence

                if alignment.quality == AlignmentQuality.EXACT:
                    # For exact matches, use higher source weight
                    effective_alpha = max(alpha, 0.7) * alignment.confidence
                elif alignment.quality == AlignmentQuality.INTERPOLATED:
                    # For interpolated, use lower source weight
                    effective_alpha = min(alpha, 0.3) * alignment.confidence

                blended = effective_alpha * source_vec + (1 - effective_alpha) * target_vec

                # Update merged embeddings
                merged_np = b.to_numpy(merged)
                merged_np[target_id] = b.to_numpy(blended)
                merged = b.array(merged_np)

                stats["blended"] += 1

            elif len(alignment.target_ids) > 1:
                # One-to-many mapping - distribute source to targets
                for target_id, weight in zip(alignment.target_ids, alignment.weights):
                    if target_id >= target_vocab_size:
                        continue

                    target_vec = target_embeddings[target_id]
                    effective_alpha = alpha * alignment.confidence * weight

                    blended = effective_alpha * source_vec + (1 - effective_alpha) * target_vec

                    merged_np = b.to_numpy(merged)
                    merged_np[target_id] = b.to_numpy(blended)
                    merged = b.array(merged_np)

                stats["interpolated"] += 1

        return merged, stats

    def _is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        special_patterns = [
            "<|",
            "|>",
            "<s>",
            "</s>",
            "<pad>",
            "<unk>",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PAD]",
            "[UNK]",
            "<bos>",
            "<eos>",
        ]
        token_lower = token.lower()
        return any(p.lower() in token_lower for p in special_patterns)

    def analyze_merge_quality(
        self,
        result: CrossVocabMergeResult,
        test_prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze quality of merged embeddings.

        Args:
            result: Merge result to analyze
            test_prompts: Optional test prompts for semantic validation

        Returns:
            Dictionary of quality metrics
        """
        alignment = result.alignment_map
        projection = result.projection_result

        metrics = {
            "alignment_coverage": alignment.coverage,
            "alignment_confidence": alignment.mean_confidence,
            "alignment_quality_distribution": alignment.quality_distribution(),
            "projection_alignment_score": projection.alignment_score,
            "projection_reconstruction_error": projection.reconstruction_error,
            "compatibility_score": result.compatibility.compatibility_score,
            "vocab_overlap_ratio": result.compatibility.vocab_overlap_ratio,
            "warnings_count": len(result.warnings),
        }

        # Overall quality score
        quality_score = (
            0.3 * alignment.coverage
            + 0.2 * alignment.mean_confidence
            + 0.3 * projection.alignment_score
            + 0.2 * result.compatibility.compatibility_score
        )
        metrics["overall_quality_score"] = quality_score

        # Recommendation
        if quality_score >= 0.8:
            recommendation = "High quality merge. Safe for production use."
        elif quality_score >= 0.6:
            recommendation = "Acceptable merge. Recommend validation testing."
        elif quality_score >= 0.4:
            recommendation = "Low quality merge. Extensive testing required."
        else:
            recommendation = "Poor quality merge. Consider alternative strategies."

        metrics["recommendation"] = recommendation

        return metrics
