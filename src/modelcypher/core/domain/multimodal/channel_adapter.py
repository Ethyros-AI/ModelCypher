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

"""Multi-Modal Channel Adapter.

Bridges multi-modal encoder models (CLIP, Whisper) with the multi-channel
merge infrastructure, creating "offramp" projections for inference-time access.

Key insight from mHC-null-space connection (docs/research/mhc_null_space_connection.md):
Both doubly stochastic routing and null-space projection are "invariant-preserving
projections onto constrained manifolds." The adapter creates entry/exit ramps
for different modalities to access the shared geometry.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    MULTI-MODAL ADAPTER                               │
    │                                                                     │
    │  CLIP Encoder ──────┐                                               │
    │                     │   ┌──────────────┐   ┌───────────────┐        │
    │  Whisper Encoder ───┼──►│  GramAligner │──►│  Null Space   │        │
    │                     │   │  (CKA=1.0)   │   │  Projection   │        │
    │  Target LLM ────────┘   └──────────────┘   └───────┬───────┘        │
    │                                                    │                │
    │                         ┌──────────────────────────┴───────────┐    │
    │                         │           OFFRAMPS                    │    │
    │                         │  ┌─────────────┐  ┌─────────────┐    │    │
    │                         │  │ Vision Ramp │  │ Audio Ramp  │    │    │
    │                         │  └─────────────┘  └─────────────┘    │    │
    │                         └──────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    adapter = MultiModalChannelAdapter(backend)
    result = adapter.create_offramps(
        target_model="/path/to/lfm2",
        include_clip=True,
        include_whisper=True,
    )
    # result.offramps contains projection matrices for inference-time access
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Backend

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.multimodal import (
    ModalityType,
    MultiModalEmbeddingExtractor,
    ModalityEmbeddings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfframpProjection:
    """Projection matrix for accessing modality-specific knowledge.

    An offramp allows the LLM to project its hidden states into a space
    aligned with a specific modality, enabling cross-modal reasoning.

    Attributes:
        modality: The modality this offramp projects to.
        projection_matrix: Shape [llm_dim, modality_dim] projection.
        inverse_projection: Shape [modality_dim, llm_dim] for embedding lookup.
        source_model: The encoder model this offramp connects to.
        cka_alignment: CKA score achieved during alignment.
    """

    modality: ModalityType
    projection_matrix: Any  # [llm_dim, modality_dim]
    inverse_projection: Any  # [modality_dim, llm_dim]
    source_model: str
    cka_alignment: float


@dataclass(frozen=True)
class MultiModalOfframpResult:
    """Result of creating multi-modal offramps.

    Contains all the projection matrices needed for inference-time
    access to multi-modal knowledge.
    """

    target_model: str
    offramps: tuple[OfframpProjection, ...]
    merged_embeddings: Any  # Enhanced embedding matrix
    cka_preservation: float  # CKA between original and merged embeddings
    concepts_used: tuple[str, ...]


class MultiModalChannelAdapter:
    """Adapter for connecting multi-modal encoders to LLM merging infrastructure.

    This class creates "offramps" - projection matrices that allow an LLM to
    access knowledge from different modalities (vision, audio) that has been
    merged into its embedding space.

    The key innovation is that offramps are BIDIRECTIONAL:
    1. Forward projection: LLM hidden state → modality-aligned space
       (for cross-modal reasoning during generation)
    2. Inverse projection: modality embeddings → LLM token space
       (for embedding lookup from multi-modal inputs)

    Mathematical Foundation:
        From GramAligner, we get transform F such that:
            CKA(source @ F, target) = 1.0

        The offramp projection is:
            P_forward = F^T (transpose of alignment transform)
            P_inverse = F (the alignment transform itself)

        This ensures:
            - Forward: llm_hidden @ P_forward projects to modality-aligned space
            - Inverse: modality_embed @ P_inverse projects to LLM token space
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize the adapter.

        Args:
            backend: Optional backend instance. If None, uses default.
        """
        if backend is None:
            backend = get_default_backend()
        self._backend = backend
        self._extractor = MultiModalEmbeddingExtractor(backend=backend)
        self._aligner = GramAligner(backend=backend)

    def create_offramps(
        self,
        target_model: str,
        include_clip: bool = True,
        include_whisper: bool = True,
        concepts: list[str] | None = None,
        highway_layers: tuple[int, int, int] = (7, 8, 9),
    ) -> MultiModalOfframpResult:
        """Create offramp projections for multi-modal knowledge access.

        This method:
        1. Extracts embeddings from target LLM and source encoders
        2. Computes alignment transforms (CKA=1.0)
        3. Creates bidirectional offramp projections
        4. Merges aligned knowledge into target's null space

        Args:
            target_model: Path to target LLM (e.g., LFM2-350M).
            include_clip: Whether to create CLIP (vision) offramp.
            include_whisper: Whether to create Whisper (audio) offramp.
            concepts: Shared concepts for alignment. If None, uses defaults.
            highway_layers: LLM layers to extract from (semantic highway).

        Returns:
            MultiModalOfframpResult with offramps and merged embeddings.
        """
        backend = self._backend

        if concepts is None:
            concepts = self._default_concepts()

        logger.info("=" * 60)
        logger.info("CREATING MULTI-MODAL OFFRAMPS")
        logger.info("=" * 60)

        # Extract target LLM embeddings
        logger.info("\nEXTRACTING TARGET EMBEDDINGS")
        target_embeds = self._extractor.extract_llm(
            target_model, concepts, highway_layers
        )
        logger.info(f"  Target: {target_embeds.hidden_dim}D")

        # Collect source modalities and create offramps
        offramps: list[OfframpProjection] = []
        aligned_sources: list[tuple[ModalityEmbeddings, Any]] = []

        if include_clip:
            logger.info("\nCREATING CLIP OFFRAMP")
            clip_embeds = self._extractor.extract_clip(concepts)
            clip_offramp, clip_aligned = self._create_offramp(
                clip_embeds, target_embeds
            )
            offramps.append(clip_offramp)
            aligned_sources.append((clip_embeds, clip_aligned))
            logger.info(f"  CLIP: CKA = {clip_offramp.cka_alignment:.4f}")

        if include_whisper:
            logger.info("\nCREATING WHISPER OFFRAMP")
            whisper_embeds = self._extractor.extract_whisper(concepts)
            whisper_offramp, whisper_aligned = self._create_offramp(
                whisper_embeds, target_embeds
            )
            offramps.append(whisper_offramp)
            aligned_sources.append((whisper_embeds, whisper_aligned))
            logger.info(f"  Whisper: CKA = {whisper_offramp.cka_alignment:.4f}")

        # Merge aligned knowledge into target null space
        logger.info("\nMERGING INTO NULL SPACE")
        merged_embeds = self._merge_aligned_sources(
            target_embeds.embeddings, aligned_sources
        )

        # Validate geometry preservation
        cka_preservation = self._compute_cka(
            target_embeds.embeddings, merged_embeds
        )
        logger.info(f"  CKA preservation: {cka_preservation:.4f}")

        return MultiModalOfframpResult(
            target_model=target_model,
            offramps=tuple(offramps),
            merged_embeddings=merged_embeds,
            cka_preservation=cka_preservation,
            concepts_used=tuple(concepts),
        )

    def _create_offramp(
        self,
        source_embeds: ModalityEmbeddings,
        target_embeds: ModalityEmbeddings,
    ) -> tuple[OfframpProjection, Any]:
        """Create an offramp projection for a single modality.

        Args:
            source_embeds: Embeddings from source modality (CLIP/Whisper).
            target_embeds: Embeddings from target LLM.

        Returns:
            Tuple of (OfframpProjection, aligned_embeddings).
        """
        backend = self._backend

        # Find perfect alignment transform
        result = self._aligner.find_perfect_alignment(
            source_embeds.embeddings,
            target_embeds.embeddings,
        )

        # Transform is [source_dim, target_dim]
        # Projects source → target space
        transform = backend.array(result.feature_transform)
        backend.eval(transform)

        # Apply transform to get aligned embeddings
        aligned = backend.matmul(source_embeds.embeddings, transform)
        backend.eval(aligned)

        # Compute CKA to verify alignment
        cka = self._compute_cka(aligned, target_embeds.embeddings)

        # Create bidirectional projections:
        # - projection_matrix: [target_dim, source_dim] = transform.T
        #   Used for: target_hidden @ projection_matrix → source-aligned space
        # - inverse_projection: [source_dim, target_dim] = transform
        #   Used for: source_embed @ inverse_projection → target-aligned space
        projection_matrix = backend.transpose(transform)
        backend.eval(projection_matrix)

        return (
            OfframpProjection(
                modality=source_embeds.modality,
                projection_matrix=projection_matrix,
                inverse_projection=transform,
                source_model=source_embeds.model_name,
                cka_alignment=cka,
            ),
            aligned,
        )

    def _merge_aligned_sources(
        self,
        target_embeds: Any,
        aligned_sources: list[tuple[ModalityEmbeddings, Any]],
    ) -> Any:
        """Merge aligned source embeddings into target's null space.

        Uses variance-weighted projection to add knowledge where the
        target has sparse activations (unused capacity).

        Args:
            target_embeds: Original target embeddings.
            aligned_sources: List of (source_metadata, aligned_embeddings) tuples.

        Returns:
            Merged embeddings with multi-modal knowledge added.
        """
        backend = self._backend

        if not aligned_sources:
            return target_embeds

        # Compute variance-based weights (null space proxy)
        # sqrt(float32 machine epsilon) for safe division
        import math
        eps = math.sqrt(2.0 ** -23)

        target_var = backend.var(target_embeds, axis=0, keepdims=True)
        backend.eval(target_var)

        var_max = backend.max(target_var)
        var_normalized = target_var / (var_max + eps)

        # Inverse variance weighting (sparse regions get more)
        # Geometry determines the scale - no arbitrary factors
        weights = 1.0 - var_normalized

        # Merge each aligned source
        merged = target_embeds
        for source_meta, aligned in aligned_sources:
            delta = aligned - target_embeds
            filtered_delta = delta * weights  # Variance weighting IS the scale
            merged = merged + filtered_delta

        backend.eval(merged)
        return merged

    def _compute_cka(self, X: Any, Y: Any) -> float:
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

        # sqrt(float32 machine epsilon) for safe division
        import math
        eps = math.sqrt(2.0 ** -23)
        return hsic_xy_val / (hsic_xx_val**0.5 * hsic_yy_val**0.5 + eps)

    def _default_concepts(self) -> list[str]:
        """Return default concepts for multi-modal alignment.

        These concepts span visual, auditory, and abstract domains
        to ensure good coverage of the shared semantic space.
        """
        return [
            # Visual concepts
            "a red ball bouncing",
            "blue sky with white clouds",
            "green forest with tall trees",
            "yellow sunflower in bloom",
            "orange sunset over ocean",
            # Auditory concepts
            "music playing softly",
            "birds singing in morning",
            "thunder rumbling loudly",
            "rain falling on roof",
            "wind whistling through trees",
            # Abstract/semantic concepts
            "running fast",
            "thinking deeply",
            "feeling happy",
            "understanding clearly",
            "remembering fondly",
            # Spatial concepts
            "above the clouds",
            "below the surface",
            "inside the building",
            "outside in nature",
            "between two objects",
        ]


__all__ = [
    "MultiModalChannelAdapter",
    "MultiModalOfframpResult",
    "OfframpProjection",
]
