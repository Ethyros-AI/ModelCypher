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

"""Cross-vocabulary embedding merger utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    ProjectionResult,
)
from .vocabulary_analyzer import (
    VocabularyAlignment,
    VocabularyAnalyzer,
    VocabularyStats,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class AlignmentMethod(str, Enum):
    VOCABULARY = "vocabulary"
    INDEX = "index"


@dataclass
class CrossVocabMergeResult:
    merged_embeddings: "Array"
    output_vocab_size: int
    output_hidden_dim: int
    alignment_map: VocabularyAlignmentMap
    projection_result: ProjectionResult
    alignment: VocabularyAlignment
    source_stats: VocabularyStats
    target_stats: VocabularyStats
    alignment_method: AlignmentMethod
    tokens_preserved_from_source: int
    tokens_preserved_from_target: int
    tokens_interpolated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_vocab_size": self.output_vocab_size,
            "output_hidden_dim": self.output_hidden_dim,
            "alignment_summary": self.alignment_map.to_dict(),
            "projection_summary": self.projection_result.to_dict(),
            "vocabulary_alignment": self.alignment.to_dict(),
            "source_stats": self.source_stats.to_dict(),
            "target_stats": self.target_stats.to_dict(),
            "alignment_method": self.alignment_method.value,
            "tokens_preserved_from_source": self.tokens_preserved_from_source,
            "tokens_preserved_from_target": self.tokens_preserved_from_target,
            "tokens_interpolated": self.tokens_interpolated,
        }


class CrossVocabMerger:
    """Merge embeddings from different vocabularies using geometric alignment."""

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._analyzer = VocabularyAnalyzer(backend=self._backend)
        self._projector = EmbeddingProjector(
            backend=self._backend,
        )

    def merge(
        self,
        source_embeddings: "Array",
        target_embeddings: "Array",
        source_vocab: dict[str, int] | None = None,
        target_vocab: dict[str, int] | None = None,
    ) -> CrossVocabMergeResult:
        backend = self._backend
        source_arr = backend.array(source_embeddings)
        target_arr = backend.array(target_embeddings)
        backend.eval(source_arr, target_arr)

        if source_vocab is None or target_vocab is None:
            alignment_map = self._build_index_alignment(
                int(source_arr.shape[0]),
                int(target_arr.shape[0]),
            )
            alignment_method = AlignmentMethod.INDEX
        else:
            alignment_map = build_alignment_from_vocabs(source_vocab, target_vocab)
            alignment_method = AlignmentMethod.VOCABULARY

        shared_indices = self._get_shared_indices(alignment_map)
        projection_result = self._projector.project(
            source_arr,
            target_arr,
            shared_token_indices=shared_indices,
        )

        merged_embeddings = target_arr
        source_stats = self._analyzer.analyze_embeddings(source_arr)
        target_stats = self._analyzer.analyze_embeddings(target_arr)
        alignment = self._analyzer.analyze_alignment(
            source_stats,
            target_stats,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
        )

        mapped_count = (
            alignment_map.exact_matches
            + alignment_map.similar_matches
            + alignment_map.approximate_matches
        )

        return CrossVocabMergeResult(
            merged_embeddings=merged_embeddings,
            output_vocab_size=int(merged_embeddings.shape[0]),
            output_hidden_dim=int(merged_embeddings.shape[1]),
            alignment_map=alignment_map,
            projection_result=projection_result,
            alignment=alignment,
            source_stats=source_stats,
            target_stats=target_stats,
            alignment_method=alignment_method,
            tokens_preserved_from_source=int(mapped_count),
            tokens_preserved_from_target=int(target_arr.shape[0]),
            tokens_interpolated=int(alignment_map.interpolated_count),
        )

    def analyze_merge_quality(self, result: CrossVocabMergeResult) -> dict[str, Any]:
        alignment_map = result.alignment_map
        coverage = alignment_map.coverage
        confidence = (
            alignment_map.exact_matches / float(alignment_map.source_vocab_size)
            if alignment_map.source_vocab_size
            else 0.0
        )
        projection_meta = result.projection_result.metadata

        return {
            "coverage_ratio": float(coverage),
            "exact_match_ratio": float(confidence),
            "match_quality_distribution": alignment_map.quality_distribution(),
            "projection_reconstruction_error": result.projection_result.reconstruction_error,
            "projection_mean_cosine_similarity": projection_meta.get(
                "mean_cosine_similarity"
            ),
            "projection_norm_preservation_ratio": projection_meta.get(
                "norm_preservation_ratio"
            ),
            "vocab_overlap_ratio": result.alignment.vocab_overlap_ratio,
            "dimension_ratio": result.alignment.dimension_ratio,
            "requires_projection": result.alignment.requires_projection,
            "requires_vocab_mapping": result.alignment.requires_vocab_mapping,
            "alignment_method": result.alignment_method.value,
        }

    def _is_special_token(self, token: str) -> bool:
        lowered = token.lower()
        if lowered in {
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "[cls]",
            "[sep]",
            "[mask]",
            "[pad]",
            "[unk]",
        }:
            return True
        if lowered.startswith("<|") and lowered.endswith("|>"):
            return True
        return False

    def _build_index_alignment(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
    ) -> VocabularyAlignmentMap:
        alignment = VocabularyAlignmentMap(
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
        )

        shared = min(source_vocab_size, target_vocab_size)
        for idx in range(shared):
            alignment.add_alignment(
                TokenAlignment(
                    source_id=idx,
                    source_token=str(idx),
                    target_ids=[idx],
                    target_tokens=[str(idx)],
                    quality=AlignmentQuality.EXACT,
                )
            )

        for idx in range(shared, source_vocab_size):
            alignment.add_alignment(
                TokenAlignment(
                    source_id=idx,
                    source_token=str(idx),
                    target_ids=[],
                    target_tokens=[],
                    quality=AlignmentQuality.UNMAPPED,
                )
            )

        return alignment

    def _get_shared_indices(
        self,
        alignment_map: VocabularyAlignmentMap,
    ) -> tuple[list[int], list[int]]:
        source_indices: list[int] = []
        target_indices: list[int] = []
        for alignment in alignment_map.iter_alignments():
            if alignment.quality in {
                AlignmentQuality.EXACT,
                AlignmentQuality.SIMILAR,
                AlignmentQuality.APPROXIMATE,
            } and len(alignment.target_ids) == 1:
                source_indices.append(int(alignment.source_id))
                target_indices.append(int(alignment.target_ids[0]))
        return source_indices, target_indices
