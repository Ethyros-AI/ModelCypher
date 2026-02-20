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

"""Relative representations for dimension-agnostic transfer.

Implements anchor-relative representations based on cosine similarities to a
fixed probe set.

Reference:
    - Moschella et al. (2023). "Relative Representations Enable Zero-Shot
      Latent Space Communication." https://arxiv.org/abs/2209.15430
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_cosine_between_sets,
    geodesic_norms,
    geodesic_pairwise_metrics,
)
from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes

_cache = ComputationCache.shared()

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
    probes: Sequence[AtlasProbeProtocol] | None = None,
) -> tuple["Array", list[str]]:
    """Compute anchor embeddings from token embedding matrix.

    Args:
        embedding_matrix: Token embedding matrix [vocab, hidden_dim]
        tokenizer: Tokenizer for encoding probe texts
        vocab_size: Vocabulary size (defaults to embedding_matrix.shape[0])
        probes: Optional probe inventory (defaults to registry)

    Returns:
        Tuple of (anchor_embeddings [n_anchors, hidden_dim], anchor_ids)
    """
    backend = get_default_backend()
    if vocab_size is None:
        vocab_size = backend.shape(embedding_matrix)[0]

    probes = list(probes or get_atlas_probes())
    if not probes:
        raise ValueError(
            "No atlas probes registered. Call "
            "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
            "before computing anchor embeddings."
        )
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
    backend: "Backend | None" = None,
) -> "Array":
    """Compute anchor-relative representation.

    This maps any hidden state h in R^d to s in R^n_anchors via:
        s_i = cos(h, anchor_i)

    The result is dimension-agnostic: models with d=2048 and d=896
    both produce s in R^n_anchors.

    Args:
        hidden_states: Hidden states [n, d_model]
        anchor_embeddings: Anchor embeddings [n_anchors, d_model]
        backend: Backend for tensor operations. If None, uses default.

    Returns:
        Relative representation [n, n_anchors]
    """
    backend = backend or get_default_backend()
    hidden_arr = hidden_states if hasattr(hidden_states, "shape") else backend.array(hidden_states)
    anchors_arr = (
        anchor_embeddings if hasattr(anchor_embeddings, "shape") else backend.array(anchor_embeddings)
    )
    backend.eval(hidden_arr, anchors_arr)
    similarities = geodesic_cosine_between_sets(hidden_arr, anchors_arr, backend)
    backend.eval(similarities)
    return similarities


def align_relative_representations(
    source_rel: "Array",
    target_rel: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """Find optimal rotation in anchor space using Procrustes.

    Args:
        source_rel: Source relative representation [n, n_anchors]
        target_rel: Target relative representation [n, n_anchors]
        backend: Backend for tensor operations. If None, uses default.

    Returns:
        Tuple of (rotation_matrix [n_anchors, n_anchors], alignment_error)
    """
    backend = backend or get_default_backend()

    # Handle degenerate cases: single sample leads to all-zero centered matrices
    # after mean subtraction, causing singular matrices in SVD/det
    n_samples = int(source_rel.shape[0])
    n_anchors = int(source_rel.shape[1])
    if n_samples <= 1:
        # Return identity rotation - no meaningful alignment possible with single sample
        R = backend.eye(n_anchors)
        backend.eval(R)
        return R, 0.0

    # Center the representations
    source_mean = backend.mean(source_rel, axis=0, keepdims=True)
    target_mean = backend.mean(target_rel, axis=0, keepdims=True)
    source_centered = source_rel - source_mean
    target_centered = target_rel - target_mean
    backend.eval(source_centered, target_centered)

    # Check for degenerate input (all zeros or constant values)
    source_norm = backend.sum(source_centered * source_centered)
    target_norm = backend.sum(target_centered * target_centered)
    backend.eval(source_norm, target_norm)
    # Use dtype-derived epsilon, not arbitrary 1e-10
    eps = float(division_epsilon(backend, source_norm))
    if float(backend.to_scalar(source_norm)) < eps or float(backend.to_scalar(target_norm)) < eps:
        # Degenerate case: return identity
        R = backend.eye(n_anchors)
        backend.eval(R)
        return R, 0.0

    # Procrustes: find R such that ||R @ source - target||_F is minimized
    M = backend.matmul(backend.transpose(source_centered), target_centered)  # [n_anchors, n_anchors]
    backend.eval(M)
    # Geodesic SVD (GPU-only)
    U, S, Vt = geodesic_svd(backend, M)
    backend.eval(U, S, Vt)

    # Check for singular SVD result (would cause det to crash)
    min_sv = backend.min(S)
    backend.eval(min_sv)
    if float(backend.to_scalar(min_sv)) < eps:
        # Singular matrix - return identity
        R = backend.eye(n_anchors)
        backend.eval(R)
        return R, 0.0

    # Ensure proper rotation (det = +1)
    R = backend.matmul(U, Vt)
    backend.eval(R)

    # Compute determinant - safe now that we've checked for singularity
    det_val = backend.det(R)
    backend.eval(det_val)
    det_scalar = float(backend.to_scalar(det_val))
    if det_scalar < 0:
        sign = backend.ones((U.shape[1],))
        idx = backend.arange(U.shape[1])
        sign = backend.where(idx == (U.shape[1] - 1), backend.full(sign.shape, -1.0), sign)
        U = U * sign
        R = backend.matmul(U, Vt)
        backend.eval(R)

    # Compute alignment error
    aligned = backend.matmul(source_rel, backend.transpose(R))
    diff = aligned - target_rel
    backend.eval(aligned, diff)
    _, dist_vals = geodesic_pairwise_metrics(aligned, target_rel, backend)
    mean_dist_arr = backend.mean(dist_vals)
    target_norms = geodesic_norms(target_rel, backend)
    mean_norm_arr = backend.mean(target_norms) if int(target_rel.shape[0]) > 0 else backend.array(0.0)
    error_eps = division_epsilon(backend, target_rel)
    denom_arr = backend.maximum(mean_norm_arr, backend.array(error_eps))
    backend.eval(mean_dist_arr, denom_arr)
    error = float(backend.to_scalar(mean_dist_arr)) / float(backend.to_scalar(denom_arr))

    return R, float(error)


def transfer_via_relative_space(
    source_hidden: "Array",
    source_anchors: "Array",
    target_anchors: "Array",
    alignment_samples: "Array | None" = None,
) -> "Array":
    """Transfer hidden states from source to target space via anchors.

    This is the core transfer algorithm:
    1. Map source hidden states to relative space (R^N)
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
    # Pseudo-inverse of anchor similarities (with SVD caching for efficiency)
    target_rel_anchors = _cache.get_or_compute_gram(
        target_anchors, backend, kernel_type="geodesic_cosine"
    )

    # Use cached SVD to compute pseudo-inverse: A^+ = V @ S^{-1} @ U^T
    U, S, Vh = _cache.get_or_compute_svd(target_rel_anchors, backend)
    # Threshold singular values using machine epsilon
    eps = float(backend.finfo().eps)
    max_s = float(backend.max(S))
    threshold = eps * max_s * float(max(target_rel_anchors.shape))
    S_safe = backend.where(S > threshold, S, backend.full(S.shape, 1.0))
    S_inv = backend.where(S > threshold, 1.0 / S_safe, backend.zeros_like(S))
    backend.eval(S_inv)
    # pinv = V @ diag(S_inv) @ U^T
    pinv = backend.matmul(backend.transpose(Vh), S_inv[:, None] * backend.transpose(U))
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
    expected_count = max(len(source_ids), len(target_ids))
    logger.info(
        "Cross-dimension transfer: %d common anchors of %d expected",
        len(common_ids),
        expected_count,
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


# ---------------------------------------------------------------------------
# Outer similarity metrics (Kucukahmetler et al. 2026)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OuterSimilarityResult:
    """Cross-model similarity metrics on relative representations.

    Three complementary measures of how similarly two models organize their
    latent spaces relative to a shared anchor set.

    Reference:
        Kucukahmetler, D. et al. (2026). "Relative Geometry of Neural
        Forecasters: Linking Accuracy and Alignment in Learned Latent
        Geometry." Transactions on Machine Learning Research (TMLR).

    Note:
        High alignment does NOT imply high accuracy. Kucukahmetler et al.
        (2026) demonstrate that models can achieve high accuracy with weak
        alignment and vice versa. These metrics measure geometric similarity
        of latent organization, not task performance. Use online_eval for
        accuracy assessment.

    Attributes:
        cosine_rss: Mean row-wise cosine similarity between relative
            embeddings. Range [-1, 1]; 1.0 = identical geometry.
        spearman_rank: Mean row-wise Spearman rank correlation over
            anchor orderings. Range [-1, 1]; 1.0 = identical ranking.
        top1_agreement: Fraction of points where both models agree on
            the closest anchor. Range [0, 1]; 1.0 = perfect agreement.
        n_samples: Number of data points compared.
        n_anchors: Number of anchors in the relative representations.
    """

    cosine_rss: float
    spearman_rank: float
    top1_agreement: float
    n_samples: int
    n_anchors: int


def _cosine_rss(
    rel_1: "Array",
    rel_2: "Array",
    backend: "Backend",
) -> float:
    """Row-wise cosine similarity, averaged over samples.

    alpha_cos = (1/N) * SUM_i cos(z'_i^(1), z'_i^(2))
    """
    dots = backend.sum(rel_1 * rel_2, axis=1)  # [N]
    norms_1 = backend.norm(rel_1, axis=1)  # [N]
    norms_2 = backend.norm(rel_2, axis=1)  # [N]
    eps = division_epsilon(backend, rel_1)
    denom = backend.maximum(norms_1 * norms_2, backend.full(backend.shape(norms_1), eps))
    cosines = dots / denom
    backend.eval(cosines)
    mean_cos = backend.mean(cosines)
    backend.eval(mean_cos)
    return float(backend.to_scalar(mean_cos))


def _spearman_rank(
    rel_1: "Array",
    rel_2: "Array",
    backend: "Backend",
) -> float:
    """Row-wise Spearman rank correlation, averaged over samples.

    alpha_rank = (1/N) * SUM_i rho(rank(z'_i^(1)), rank(z'_i^(2)))

    Uses the d^2 formula: rho = 1 - 6*sum(d^2) / (n*(n^2-1)).
    No-ties assumption is valid for float32 cosine similarities to distinct
    anchors (exact ties have measure zero).
    """
    n_anchors = int(backend.shape(rel_1)[1])
    if n_anchors < 3:
        # Spearman undefined for fewer than 3 values
        return 0.0

    # Double argsort to get ranks: argsort(argsort(x)) = rank
    ranks_1 = backend.argsort(backend.argsort(rel_1, axis=1), axis=1)  # [N, n_anchors]
    ranks_2 = backend.argsort(backend.argsort(rel_2, axis=1), axis=1)
    backend.eval(ranks_1, ranks_2)

    # Cast to float for arithmetic
    ranks_1_f = backend.astype(ranks_1, rel_1.dtype)
    ranks_2_f = backend.astype(ranks_2, rel_1.dtype)

    # Spearman d^2 formula
    d = ranks_1_f - ranks_2_f
    d_sq = backend.sum(d * d, axis=1)  # [N]
    backend.eval(d_sq)

    n = float(n_anchors)
    rho = 1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
    mean_rho = backend.mean(rho)
    backend.eval(mean_rho)
    return float(backend.to_scalar(mean_rho))


def _top1_agreement(
    rel_1: "Array",
    rel_2: "Array",
    backend: "Backend",
) -> float:
    """Fraction of points where both models agree on the closest anchor.

    alpha_T1 = (1/N) * SUM_i 1[argmax_k z'_{ik}^(1) == argmax_k z'_{ik}^(2)]
    """
    argmax_1 = backend.argmax(rel_1, axis=1)  # [N]
    argmax_2 = backend.argmax(rel_2, axis=1)  # [N]
    backend.eval(argmax_1, argmax_2)

    # Integer equality: |diff| < 0.5 is exact for integers (which differ by >= 1)
    diff = backend.astype(argmax_1, rel_1.dtype) - backend.astype(argmax_2, rel_1.dtype)
    matches = backend.where(
        backend.abs(diff) < backend.full(backend.shape(diff), 0.5),
        backend.ones_like(diff),
        backend.zeros_like(diff),
    )
    backend.eval(matches)
    agreement = backend.mean(matches)
    backend.eval(agreement)
    return float(backend.to_scalar(agreement))


def compute_outer_similarity(
    rel_1: "Array",
    rel_2: "Array",
    backend: "Backend | None" = None,
) -> OuterSimilarityResult:
    """Compute cross-model similarity on relative representations.

    Implements three outer similarity metrics from Kucukahmetler et al.
    (TMLR 2026): cosine RSS, Spearman rank correlation, and top-1
    anchor agreement.

    Both inputs must be relative representations with the same anchor
    set, i.e., [N, n_anchors] arrays where N is the number of data
    points and n_anchors is the number of shared anchors.

    Args:
        rel_1: Relative representation from model 1, shape [N, n_anchors].
        rel_2: Relative representation from model 2, shape [N, n_anchors].
        backend: Backend for tensor operations. If None, uses default.

    Returns:
        OuterSimilarityResult with all three metrics.

    Raises:
        ValueError: If input shapes don't match or are degenerate.
    """
    backend = backend or get_default_backend()

    shape_1 = backend.shape(rel_1)
    shape_2 = backend.shape(rel_2)

    if len(shape_1) != 2 or len(shape_2) != 2:
        raise ValueError(
            f"Relative representations must be 2D [N, n_anchors], "
            f"got shapes {shape_1} and {shape_2}"
        )
    if shape_1 != shape_2:
        raise ValueError(
            f"Relative representations must have matching shapes, "
            f"got {shape_1} and {shape_2}"
        )
    if shape_1[0] < 1:
        raise ValueError("Need at least 1 sample for outer similarity")

    n_samples, n_anchors = shape_1

    return OuterSimilarityResult(
        cosine_rss=_cosine_rss(rel_1, rel_2, backend),
        spearman_rank=_spearman_rank(rel_1, rel_2, backend),
        top1_agreement=_top1_agreement(rel_1, rel_2, backend),
        n_samples=n_samples,
        n_anchors=n_anchors,
    )
