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

from modelcypher.core.domain._backend import get_default_backend

# unified_atlas imported lazily to avoid circular imports

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelativeRepresentation:
    """Anchor-relative representation (dimension-agnostic).

    Attributes:
        similarities: Cosine similarities to anchors [n_samples, n_anchors]
        anchor_ids: List of anchor probe IDs
        hidden_dim: Original hidden dimension (for reference)
    """

    similarities: "Array"  # [n_samples, n_anchors]
    anchor_ids: tuple[str, ...]
    hidden_dim: int

    @property
    def n_samples(self) -> int:
        backend = get_default_backend()
        return backend.shape(self.similarities)[0]

    @property
    def n_anchors(self) -> int:
        backend = get_default_backend()
        return backend.shape(self.similarities)[1]


def compute_anchor_embeddings(
    embedding_matrix: "Array",
    tokenizer: "Tokenizer",
    vocab_size: int | None = None,
) -> tuple["Array", list[str]]:
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

    backend = get_default_backend()
    if vocab_size is None:
        vocab_size = backend.shape(embedding_matrix)[0]

    probes = UnifiedAtlasInventory.all_probes()
    anchors: list["Array"] = []
    anchor_ids: list[str] = []

    for probe in probes:
        vectors: list["Array"] = []
        for text in probe.support_texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            valid = [tid for tid in ids if 0 <= tid < vocab_size]
            if valid:
                selected = backend.take(embedding_matrix, backend.array(valid), axis=0)
                mean_vec = backend.mean(selected, axis=0)
                backend.eval(mean_vec)
                vectors.append(mean_vec)

        if vectors:
            stacked = backend.stack(vectors, axis=0)
            mean_anchor = backend.mean(stacked, axis=0)
            backend.eval(mean_anchor)
            anchors.append(mean_anchor)
            anchor_ids.append(probe.probe_id)

    logger.info(
        "Computed %d anchor embeddings from %d probes",
        len(anchors),
        len(probes),
    )
    result = backend.stack(anchors, axis=0)
    backend.eval(result)
    return result, anchor_ids


def compute_relative_representation(
    hidden_states: "Array",
    anchor_embeddings: "Array",
) -> "Array":
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
    backend = get_default_backend()
    # Normalize anchors
    anchor_norms = backend.norm(anchor_embeddings, axis=1, keepdims=True)
    anchors_normalized = anchor_embeddings / backend.maximum(anchor_norms, 1e-8)

    # Normalize hidden states
    hidden_norms = backend.norm(hidden_states, axis=1, keepdims=True)
    hidden_normalized = hidden_states / backend.maximum(hidden_norms, 1e-8)

    # Compute cosine similarities: [n, d] @ [d, n_anchors] = [n, n_anchors]
    backend.eval(anchors_normalized, hidden_normalized)
    similarities = backend.matmul(hidden_normalized, backend.transpose(anchors_normalized))
    backend.eval(similarities)

    return similarities


def align_relative_representations(
    source_rel: "Array",
    target_rel: "Array",
) -> tuple["Array", float]:
    """Find optimal rotation in anchor space using Procrustes.

    Args:
        source_rel: Source relative representation [n, n_anchors]
        target_rel: Target relative representation [n, n_anchors]

    Returns:
        Tuple of (rotation_matrix [n_anchors, n_anchors], alignment_error)
    """
    backend = get_default_backend()
    # Center the representations
    source_mean = backend.mean(source_rel, axis=0, keepdims=True)
    target_mean = backend.mean(target_rel, axis=0, keepdims=True)
    source_centered = source_rel - source_mean
    target_centered = target_rel - target_mean

    # Procrustes: find R such that ||R @ source - target||_F is minimized
    backend.eval(source_centered, target_centered)
    M = backend.matmul(backend.transpose(source_centered), target_centered)  # [n_anchors, n_anchors]
    backend.eval(M)
    U, S, Vt = backend.svd(M, full_matrices=False)
    backend.eval(U, S, Vt)

    # Ensure proper rotation (det = +1)
    R = backend.matmul(U, Vt)
    backend.eval(R)
    det_val = backend.det(R)
    backend.eval(det_val)
    if backend.to_numpy(det_val).item() < 0:
        U_np = backend.to_numpy(U)
        U_np[:, -1] *= -1
        U = backend.array(U_np)
        R = backend.matmul(U, Vt)
        backend.eval(R)

    # Compute alignment error
    aligned = backend.matmul(source_rel, backend.transpose(R))
    diff = aligned - target_rel
    backend.eval(aligned, diff)
    error_num = backend.norm(diff)
    error_denom = backend.maximum(backend.norm(target_rel), 1e-8)
    backend.eval(error_num, error_denom)
    error = float(backend.to_numpy(error_num).item() / backend.to_numpy(error_denom).item())

    return R, float(error)


def transfer_via_relative_space(
    source_hidden: "Array",
    source_anchors: "Array",
    target_anchors: "Array",
    alignment_samples: "Array | None" = None,
) -> "Array":
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
    backend = get_default_backend()
    # Step 1: Map to relative space
    source_rel = compute_relative_representation(source_hidden, source_anchors)

    # Step 2: Optional alignment in relative space
    if alignment_samples is not None:
        # Compute alignment from paired samples
        source_dim = backend.shape(source_anchors)[1]
        sample_source_rel = compute_relative_representation(
            alignment_samples[:, :source_dim],
            source_anchors,
        )
        sample_target_rel = compute_relative_representation(
            alignment_samples[:, source_dim:],
            target_anchors,
        )
        R, error = align_relative_representations(sample_source_rel, sample_target_rel)
        logger.info("Relative space alignment error: %.4f", error)
        source_rel = backend.matmul(source_rel, backend.transpose(R))
        backend.eval(source_rel)

    # Step 3: Project back to target space using pseudo-inverse
    # target_hidden = source_rel @ pinv(target_rel_anchors)
    # where target_rel_anchors[i, j] = cos(anchor_j, anchor_i)
    target_anchor_norms = backend.norm(target_anchors, axis=1, keepdims=True)
    target_anchors_normalized = target_anchors / backend.maximum(target_anchor_norms, 1e-8)

    # Pseudo-inverse of anchor similarities
    backend.eval(target_anchors_normalized)
    target_rel_anchors = backend.matmul(target_anchors_normalized, backend.transpose(target_anchors_normalized))
    backend.eval(target_rel_anchors)
    pinv = backend.pinv(target_rel_anchors)
    backend.eval(pinv)

    # Project: [n, n_anchors] @ [n_anchors, n_anchors] @ [n_anchors, d_target]
    temp = backend.matmul(source_rel, pinv)
    transferred = backend.matmul(temp, target_anchors)
    backend.eval(transferred)

    return transferred


@dataclass(frozen=True)
class CrossDimensionTransferResult:
    """Result of cross-dimension transfer via relative representations."""

    transferred_states: "Array"  # [n, d_target]
    relative_representation: "Array"  # [n, n_anchors]
    alignment_rotation: "Array | None"  # [n_anchors, n_anchors]
    alignment_error: float
    source_dim: int
    target_dim: int
    n_anchors: int


def cross_dimension_transfer(
    source_hidden: "Array",
    source_embedding: "Array",
    target_embedding: "Array",
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
    backend = get_default_backend()
    d_source = backend.shape(source_hidden)[1]
    d_target = backend.shape(target_embedding)[1]

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
    source_anchors_common = backend.take(source_anchors, backend.array(source_mask), axis=0)
    target_anchors_common = backend.take(target_anchors, backend.array(target_mask), axis=0)
    backend.eval(source_anchors_common, target_anchors_common)

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
    aligned_rel = backend.matmul(source_rel, backend.transpose(R))
    backend.eval(aligned_rel)
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
