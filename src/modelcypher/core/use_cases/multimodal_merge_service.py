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

"""Multi-Modal Merge Service.

Orchestrates cross-modal knowledge transfer from vision (CLIP) and audio
(Whisper) models into an LLM's null space.

Key insight: The 99% null space in LLMs isn't waste - it's capacity waiting
to be filled. Multi-modal knowledge from CLIP and Whisper can be projected
into this capacity, giving the LLM richer multi-modal understanding without
destroying existing knowledge.

Pipeline:
1. Extract embeddings from all modalities for shared concepts
2. Align each modality to target LLM geometry (CKA=1.0)
3. Project aligned knowledge into null space
4. Validate geometry preservation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Backend

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.multimodal import (
    ModalityType,
    MultiModalEmbeddingExtractor,
    ModalityEmbeddings,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning one modality to target."""

    source_modality: ModalityType
    cka_before: float
    cka_after: float
    transform_shape: tuple[int, int]


@dataclass(frozen=True)
class MergeResult:
    """Result of merging knowledge into target null space."""

    source_modality: ModalityType
    preserved_fraction: float
    projection_loss: float
    delta_norm_before: float
    delta_norm_after: float


@dataclass(frozen=True)
class MultiModalMergeResult:
    """Complete result of multi-modal merge operation."""

    concepts: tuple[str, ...]
    alignment_results: tuple[AlignmentResult, ...]
    merge_results: tuple[MergeResult, ...]
    cka_preservation: float
    target_model: str
    source_models: tuple[str, ...]


class MultiModalMergeService:
    """Service for merging multi-modal knowledge into LLMs.

    This service orchestrates the complete pipeline:
    1. Extract embeddings from target LLM and source modalities
    2. Align each source to target geometry (CKA=1.0)
    3. Merge into null space
    4. Validate preservation

    Example:
        >>> service = MultiModalMergeService()
        >>> concepts = ["a red ball", "music playing", "running fast"]
        >>> result = service.merge(
        ...     target_model="/path/to/lfm2",
        ...     concepts=concepts,
        ...     include_clip=True,
        ...     include_whisper=True,
        ... )
        >>> print(f"CKA preservation: {result.cka_preservation:.4f}")
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize the service.

        Args:
            backend: Optional backend instance. If None, uses default.
        """
        if backend is None:
            backend = get_default_backend()
        self._backend = backend
        self._extractor = MultiModalEmbeddingExtractor(backend=backend)
        self._aligner = GramAligner(backend=backend)

    def merge(
        self,
        target_model: str,
        concepts: list[str],
        include_clip: bool = True,
        include_whisper: bool = True,
        highway_layers: tuple[int, int, int] = (7, 8, 9),
    ) -> MultiModalMergeResult:
        """Merge multi-modal knowledge into target LLM.

        Args:
            target_model: Path to target LLM (receives knowledge).
            concepts: Shared concepts across modalities.
            include_clip: Whether to include CLIP (vision) knowledge.
            include_whisper: Whether to include Whisper (audio) knowledge.
            highway_layers: LLM layers to extract from (semantic highway).

        Returns:
            MultiModalMergeResult with alignment and merge metrics.
        """
        logger.info("=" * 60)
        logger.info("MULTI-MODAL KNOWLEDGE MERGE")
        logger.info("=" * 60)

        # Step 1: Extract embeddings
        logger.info("\nSTEP 1: EXTRACT EMBEDDINGS")
        logger.info("-" * 40)

        target_embeds = self._extractor.extract_llm(
            target_model, concepts, highway_layers
        )
        logger.info(f"Target LLM: {target_embeds.hidden_dim}D")

        source_embeds: list[ModalityEmbeddings] = []
        source_models: list[str] = [target_model]

        if include_clip:
            clip_embeds = self._extractor.extract_clip(concepts)
            source_embeds.append(clip_embeds)
            source_models.append(clip_embeds.model_name)
            logger.info(f"CLIP: {clip_embeds.hidden_dim}D")

        if include_whisper:
            whisper_embeds = self._extractor.extract_whisper(concepts)
            source_embeds.append(whisper_embeds)
            source_models.append(whisper_embeds.model_name)
            logger.info(f"Whisper: {whisper_embeds.hidden_dim}D")

        # Step 2: Align each source to target
        logger.info("\nSTEP 2: ALIGN TO TARGET GEOMETRY")
        logger.info("-" * 40)

        alignment_results: list[AlignmentResult] = []
        aligned_embeds: list["Backend.Array"] = []  # type: ignore

        for source in source_embeds:
            cka_before = self._compute_cka(source.embeddings, target_embeds.embeddings)

            # Use GramAligner to find CKA=1.0 transform
            result = self._aligner.find_perfect_alignment(
                source.embeddings,
                target_embeds.embeddings,
            )

            # Apply transform
            aligned = self._backend.matmul(
                source.embeddings,
                self._backend.array(result.feature_transform),
            )
            self._backend.eval(aligned)

            cka_after = self._compute_cka(aligned, target_embeds.embeddings)

            alignment_results.append(
                AlignmentResult(
                    source_modality=source.modality,
                    cka_before=cka_before,
                    cka_after=cka_after,
                    transform_shape=result.feature_transform.shape,
                )
            )
            aligned_embeds.append(aligned)

            logger.info(f"{source.modality.value}: CKA {cka_before:.4f} → {cka_after:.4f}")

        # Step 3: Merge into null space
        logger.info("\nSTEP 3: MERGE INTO NULL SPACE")
        logger.info("-" * 40)

        merge_results: list[MergeResult] = []
        current_target = target_embeds.embeddings

        for i, (source, aligned) in enumerate(zip(source_embeds, aligned_embeds)):
            merged, metrics = self._merge_into_null_space(
                aligned, current_target, target_embeds.embeddings
            )

            merge_results.append(
                MergeResult(
                    source_modality=source.modality,
                    preserved_fraction=metrics["preserved_fraction"],
                    projection_loss=metrics["projection_loss"],
                    delta_norm_before=metrics["delta_norm_before"],
                    delta_norm_after=metrics["delta_norm_after"],
                )
            )
            current_target = merged

            logger.info(
                f"{source.modality.value}: preserved {metrics['preserved_fraction']:.4f}, "
                f"lost {metrics['projection_loss']:.4f}"
            )

        # Step 4: Validate preservation
        logger.info("\nSTEP 4: VALIDATE PRESERVATION")
        logger.info("-" * 40)

        cka_preservation = self._compute_cka(target_embeds.embeddings, current_target)
        logger.info(f"CKA(original, merged) = {cka_preservation:.4f}")

        return MultiModalMergeResult(
            concepts=tuple(concepts),
            alignment_results=tuple(alignment_results),
            merge_results=tuple(merge_results),
            cka_preservation=cka_preservation,
            target_model=target_model,
            source_models=tuple(source_models),
        )

    def _compute_cka(
        self,
        X: "Backend.Array",  # type: ignore
        Y: "Backend.Array",  # type: ignore
    ) -> float:
        """Compute CKA between two embedding matrices."""
        backend = self._backend

        X = backend.astype(X, "float32")
        Y = backend.astype(Y, "float32")

        # Center
        X = X - backend.mean(X, axis=0, keepdims=True)
        Y = Y - backend.mean(Y, axis=0, keepdims=True)

        # Gram matrices
        K = backend.matmul(X, backend.transpose(X))
        L = backend.matmul(Y, backend.transpose(Y))

        n = int(K.shape[0])
        H = backend.eye(n) - backend.ones((n, n)) / n

        KH = backend.matmul(K, H)
        LH = backend.matmul(L, H)

        hsic_xy = backend.sum(KH * backend.transpose(LH)) / ((n - 1) ** 2)
        hsic_xx = backend.sum(KH * backend.transpose(KH)) / ((n - 1) ** 2)
        hsic_yy = backend.sum(LH * backend.transpose(LH)) / ((n - 1) ** 2)

        backend.eval(hsic_xy, hsic_xx, hsic_yy)

        hsic_xy_val = float(backend.to_scalar(hsic_xy))
        hsic_xx_val = float(backend.to_scalar(hsic_xx))
        hsic_yy_val = float(backend.to_scalar(hsic_yy))

        return hsic_xy_val / (hsic_xx_val**0.5 * hsic_yy_val**0.5 + 1e-10)

    def _merge_into_null_space(
        self,
        source: "Backend.Array",  # type: ignore
        target: "Backend.Array",  # type: ignore
        prior: "Backend.Array",  # type: ignore
    ) -> tuple["Backend.Array", dict]:  # type: ignore
        """Merge source into target's null space using variance-weighted projection.

        Uses simplified variance-based projection for embedding-level merging.
        The full geodesic filter is too expensive for large vocabularies.

        Args:
            source: Aligned source embeddings.
            target: Current target embeddings.
            prior: Original target embeddings for variance estimation.

        Returns:
            Merged embeddings and metrics dict.
        """
        backend = self._backend

        # Compute delta
        delta = source - target

        # Variance-weighted projection (simplified null-space)
        # High variance = dense region = scale down
        # Low variance = sparse region = preserve
        prior_var = backend.var(prior, axis=0, keepdims=True)
        backend.eval(prior_var)

        # Normalize variance to [0, 1]
        var_max = backend.max(prior_var)
        var_normalized = prior_var / (var_max + 1e-10)

        # Inverse variance weighting (sparse regions get more)
        weights = 1.0 - var_normalized

        # Apply weights to delta
        filtered_delta = delta * weights

        # Compute norms for metrics
        delta_norm_before = float(
            backend.to_scalar(backend.sqrt(backend.sum(delta * delta)))
        )
        delta_norm_after = float(
            backend.to_scalar(backend.sqrt(backend.sum(filtered_delta * filtered_delta)))
        )

        preserved_fraction = delta_norm_after / (delta_norm_before + 1e-10)
        projection_loss = 1.0 - preserved_fraction

        # Merge
        merged = target + filtered_delta
        backend.eval(merged)

        metrics = {
            "preserved_fraction": preserved_fraction,
            "projection_loss": projection_loss,
            "delta_norm_before": delta_norm_before,
            "delta_norm_after": delta_norm_after,
        }

        return merged, metrics
