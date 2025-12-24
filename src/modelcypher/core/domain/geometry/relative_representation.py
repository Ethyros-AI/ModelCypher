"""
Relative Representations for Dimension-Agnostic Transfer.

Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot
Latent Space Communication"
https://arxiv.org/abs/2209.15430

Key insight: Pairwise similarities to a fixed anchor set are quasi-isometric
across models, regardless of their hidden dimension. This enables transfer
between models of different sizes (e.g., 2048-dim to 896-dim) by working in
anchor-relative space.

The 321 anchors from UnifiedAtlasInventory serve as the universal bridge:
- Any hidden state h in R^d maps to s in R^321 via cosine similarities
- Alignment happens in R^321 (dimension-agnostic)
- Transfer back to target space uses pseudo-inverse projection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# unified_atlas imported lazily to avoid circular imports

if TYPE_CHECKING:
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelativeRepresentation:
    """Anchor-relative representation (dimension-agnostic).

    Attributes:
        similarities: Cosine similarities to anchors [n_samples, n_anchors]
        anchor_ids: List of anchor probe IDs
        hidden_dim: Original hidden dimension (for reference)
    """

    similarities: np.ndarray  # [n_samples, n_anchors]
    anchor_ids: tuple[str, ...]
    hidden_dim: int

    @property
    def n_samples(self) -> int:
        return self.similarities.shape[0]

    @property
    def n_anchors(self) -> int:
        return self.similarities.shape[1]


def compute_anchor_embeddings(
    embedding_matrix: np.ndarray,
    tokenizer: "Tokenizer",
    vocab_size: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute anchor embeddings from token embedding matrix.

    Args:
        embedding_matrix: Token embedding matrix [vocab, hidden_dim]
        tokenizer: Tokenizer for encoding probe texts
        vocab_size: Vocabulary size (defaults to embedding_matrix.shape[0])

    Returns:
        Tuple of (anchor_embeddings [n_anchors, hidden_dim], anchor_ids)
    """
    # Lazy import to avoid circular dependency
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    if vocab_size is None:
        vocab_size = embedding_matrix.shape[0]

    probes = UnifiedAtlasInventory.all_probes()
    anchors: list[np.ndarray] = []
    anchor_ids: list[str] = []

    for probe in probes:
        vectors: list[np.ndarray] = []
        for text in probe.support_texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            valid = [tid for tid in ids if 0 <= tid < vocab_size]
            if valid:
                vectors.append(embedding_matrix[valid].mean(axis=0))

        if vectors:
            anchors.append(np.mean(np.stack(vectors), axis=0))
            anchor_ids.append(probe.probe_id)

    logger.info(
        "Computed %d anchor embeddings from %d probes",
        len(anchors),
        len(probes),
    )
    return np.stack(anchors), anchor_ids


def compute_relative_representation(
    hidden_states: np.ndarray,
    anchor_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute anchor-relative representation.

    This maps any hidden state h in R^d to s in R^n_anchors via:
        s_i = cos(h, anchor_i)

    The result is dimension-agnostic: models with d=2048 and d=896
    both produce s in R^n_anchors.

    Args:
        hidden_states: Hidden states [n, d_model]
        anchor_embeddings: Anchor embeddings [n_anchors, d_model]

    Returns:
        Relative representation [n, n_anchors]
    """
    # Normalize anchors
    anchor_norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    anchors_normalized = anchor_embeddings / np.maximum(anchor_norms, 1e-8)

    # Normalize hidden states
    hidden_norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
    hidden_normalized = hidden_states / np.maximum(hidden_norms, 1e-8)

    # Compute cosine similarities: [n, d] @ [d, n_anchors] = [n, n_anchors]
    similarities = hidden_normalized @ anchors_normalized.T

    return similarities


def align_relative_representations(
    source_rel: np.ndarray,
    target_rel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find optimal rotation in anchor space using Procrustes.

    Args:
        source_rel: Source relative representation [n, n_anchors]
        target_rel: Target relative representation [n, n_anchors]

    Returns:
        Tuple of (rotation_matrix [n_anchors, n_anchors], alignment_error)
    """
    # Center the representations
    source_centered = source_rel - source_rel.mean(axis=0, keepdims=True)
    target_centered = target_rel - target_rel.mean(axis=0, keepdims=True)

    # Procrustes: find R such that ||R @ source - target||_F is minimized
    M = source_centered.T @ target_centered  # [n_anchors, n_anchors]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Ensure proper rotation (det = +1)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute alignment error
    aligned = source_rel @ R.T
    error = np.linalg.norm(aligned - target_rel) / max(np.linalg.norm(target_rel), 1e-8)

    return R, float(error)


def transfer_via_relative_space(
    source_hidden: np.ndarray,
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    alignment_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Transfer hidden states from source to target space via anchors.

    This is the core transfer algorithm:
    1. Map source hidden states to relative space (R^321)
    2. Optionally compute alignment rotation from paired samples
    3. Project back to target space using pseudo-inverse

    Args:
        source_hidden: Source hidden states [n, d_source]
        source_anchors: Source anchor embeddings [n_anchors, d_source]
        target_anchors: Target anchor embeddings [n_anchors, d_target]
        alignment_samples: Optional paired samples for Procrustes alignment

    Returns:
        Transferred hidden states [n, d_target]
    """
    # Step 1: Map to relative space
    source_rel = compute_relative_representation(source_hidden, source_anchors)

    # Step 2: Optional alignment in relative space
    if alignment_samples is not None:
        # Compute alignment from paired samples
        sample_source_rel = compute_relative_representation(
            alignment_samples[:, : source_anchors.shape[1]],
            source_anchors,
        )
        sample_target_rel = compute_relative_representation(
            alignment_samples[:, source_anchors.shape[1] :],
            target_anchors,
        )
        R, error = align_relative_representations(sample_source_rel, sample_target_rel)
        logger.info("Relative space alignment error: %.4f", error)
        source_rel = source_rel @ R.T

    # Step 3: Project back to target space using pseudo-inverse
    # target_hidden = source_rel @ pinv(target_rel_anchors)
    # where target_rel_anchors[i, j] = cos(anchor_j, anchor_i)
    target_anchor_norms = np.linalg.norm(target_anchors, axis=1, keepdims=True)
    target_anchors_normalized = target_anchors / np.maximum(target_anchor_norms, 1e-8)

    # Pseudo-inverse of anchor similarities
    target_rel_anchors = target_anchors_normalized @ target_anchors_normalized.T
    pinv = np.linalg.pinv(target_rel_anchors)

    # Project: [n, n_anchors] @ [n_anchors, n_anchors] @ [n_anchors, d_target]
    transferred = source_rel @ pinv @ target_anchors

    return transferred


@dataclass(frozen=True)
class CrossDimensionTransferResult:
    """Result of cross-dimension transfer via relative representations."""

    transferred_states: np.ndarray  # [n, d_target]
    relative_representation: np.ndarray  # [n, n_anchors]
    alignment_rotation: np.ndarray | None  # [n_anchors, n_anchors]
    alignment_error: float
    source_dim: int
    target_dim: int
    n_anchors: int


def cross_dimension_transfer(
    source_hidden: np.ndarray,
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_tokenizer: "Tokenizer",
    target_tokenizer: "Tokenizer",
) -> CrossDimensionTransferResult:
    """Full cross-dimension transfer pipeline.

    This is the main entry point for transferring hidden states between
    models of different dimensions using anchor-relative representations.

    Args:
        source_hidden: Hidden states to transfer [n, d_source]
        source_embedding: Source token embedding matrix [vocab_source, d_source]
        target_embedding: Target token embedding matrix [vocab_target, d_target]
        source_tokenizer: Source model tokenizer
        target_tokenizer: Target model tokenizer

    Returns:
        CrossDimensionTransferResult with transferred states and metadata
    """
    d_source = source_hidden.shape[1]
    d_target = target_embedding.shape[1]

    # Compute anchor embeddings for both models
    source_anchors, source_ids = compute_anchor_embeddings(
        source_embedding,
        source_tokenizer,
    )
    target_anchors, target_ids = compute_anchor_embeddings(
        target_embedding,
        target_tokenizer,
    )

    # Find common anchors
    common_ids = set(source_ids) & set(target_ids)
    if len(common_ids) < 100:
        logger.warning(
            "Only %d common anchors found (expected ~321). Transfer may be unreliable.",
            len(common_ids),
        )

    # Filter to common anchors
    source_mask = [i for i, aid in enumerate(source_ids) if aid in common_ids]
    target_mask = [i for i, aid in enumerate(target_ids) if aid in common_ids]
    source_anchors_common = source_anchors[source_mask]
    target_anchors_common = target_anchors[target_mask]

    # Compute relative representations
    source_rel = compute_relative_representation(source_hidden, source_anchors_common)

    # Compute alignment in anchor space
    # Use anchor self-similarities as the alignment target
    source_anchor_rel = compute_relative_representation(
        source_anchors_common,
        source_anchors_common,
    )
    target_anchor_rel = compute_relative_representation(
        target_anchors_common,
        target_anchors_common,
    )
    R, error = align_relative_representations(source_anchor_rel, target_anchor_rel)

    # Apply alignment and transfer
    aligned_rel = source_rel @ R.T
    transferred = transfer_via_relative_space(
        source_hidden,
        source_anchors_common,
        target_anchors_common,
    )

    return CrossDimensionTransferResult(
        transferred_states=transferred,
        relative_representation=aligned_rel,
        alignment_rotation=R,
        alignment_error=error,
        source_dim=d_source,
        target_dim=d_target,
        n_anchors=len(common_ids),
    )
