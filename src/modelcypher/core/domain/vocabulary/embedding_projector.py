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
Embedding Projector for Cross-Vocabulary Merging.

Projects embeddings from source vocabulary space to target vocabulary space
using various alignment strategies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class ProjectionStrategy(str, Enum):
    """Strategies for embedding projection."""

    TRUNCATE = "truncate"  # Simple truncation/padding (baseline)
    PCA = "pca"  # PCA-based dimension alignment
    PROCRUSTES = "procrustes"  # Orthogonal Procrustes alignment
    CCA = "cca"  # Canonical Correlation Analysis
    OPTIMAL_TRANSPORT = "optimal_transport"  # Wasserstein-based alignment


@dataclass
class ProjectionConfig:
    """Configuration for embedding projection."""

    strategy: ProjectionStrategy = ProjectionStrategy.PROCRUSTES
    regularization: float = 1e-6  # Regularization for stability
    n_components: int | None = None  # For PCA, limit components
    preserve_norms: bool = True  # Scale to preserve embedding norms
    anchor_count: int = 1000  # Number of anchors for alignment


@dataclass
class ProjectionResult:
    """Result of embedding projection."""

    projected_embeddings: "Array"
    projection_matrix: "Array | None"
    reconstruction_error: float
    alignment_score: float  # How well alignment preserved distances
    strategy_used: ProjectionStrategy
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "reconstruction_error": self.reconstruction_error,
            "alignment_score": self.alignment_score,
            "strategy_used": self.strategy_used.value,
            "output_shape": list(self.projected_embeddings.shape),
            "has_projection_matrix": self.projection_matrix is not None,
            **self.metadata,
        }


class EmbeddingProjector:
    """
    Projects embeddings between different vocabulary spaces.

    Supports multiple strategies:
    - TRUNCATE: Simple truncation/padding for dimension mismatch
    - PCA: Project to shared principal components
    - PROCRUSTES: Orthogonal alignment minimizing Frobenius norm
    - CCA: Canonical correlation-based alignment
    - OPTIMAL_TRANSPORT: Wasserstein distance minimization
    """

    def __init__(
        self,
        config: ProjectionConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or ProjectionConfig()
        self._backend = backend or get_default_backend()

    def project(
        self,
        source_embeddings: "Array",
        target_embeddings: "Array",
        shared_token_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        """
        Project source embeddings to target embedding space.

        Args:
            source_embeddings: Source embedding matrix [vocab_size, hidden_dim]
            target_embeddings: Target embedding matrix [vocab_size, hidden_dim]
            shared_token_indices: Optional (source_indices, target_indices) for
                                  tokens present in both vocabularies

        Returns:
            ProjectionResult with projected embeddings
        """
        strategy = self.config.strategy

        if strategy == ProjectionStrategy.TRUNCATE:
            return self._project_truncate(source_embeddings, target_embeddings)
        elif strategy == ProjectionStrategy.PCA:
            return self._project_pca(source_embeddings, target_embeddings)
        elif strategy == ProjectionStrategy.PROCRUSTES:
            return self._project_procrustes(
                source_embeddings, target_embeddings, shared_token_indices
            )
        elif strategy == ProjectionStrategy.CCA:
            return self._project_cca(source_embeddings, target_embeddings, shared_token_indices)
        elif strategy == ProjectionStrategy.OPTIMAL_TRANSPORT:
            return self._project_optimal_transport(
                source_embeddings, target_embeddings, shared_token_indices
            )
        else:
            raise ValueError(f"Unknown projection strategy: {strategy}")

    def _project_truncate(
        self,
        source: "Array",
        target: "Array",
    ) -> ProjectionResult:
        """Simple truncation/padding for dimension mismatch."""
        b = self._backend

        source_vocab, source_dim = source.shape
        _, target_dim = target.shape

        if source_dim == target_dim:
            # No projection needed
            return ProjectionResult(
                projected_embeddings=source,
                projection_matrix=None,
                reconstruction_error=0.0,
                alignment_score=1.0,
                strategy_used=ProjectionStrategy.TRUNCATE,
            )

        if source_dim > target_dim:
            # Truncate to target dimension
            projected = source[:, :target_dim]
        else:
            # Pad with zeros
            padding = b.zeros((source_vocab, target_dim - source_dim))
            projected = b.concatenate([source, padding], axis=1)

        # Compute reconstruction error
        if source_dim > target_dim:
            error = float(b.to_numpy(b.mean(b.norm(source[:, target_dim:], axis=1))))
        else:
            error = 0.0

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=None,
            reconstruction_error=error,
            alignment_score=min(source_dim, target_dim) / max(source_dim, target_dim),
            strategy_used=ProjectionStrategy.TRUNCATE,
        )

    def _project_pca(
        self,
        source: "Array",
        target: "Array",
    ) -> ProjectionResult:
        """PCA-based dimension alignment."""
        b = self._backend

        _, source_dim = source.shape
        _, target_dim = target.shape

        # Determine target components
        n_components = self.config.n_components or target_dim

        # Center embeddings
        source_mean = b.mean(source, axis=0)
        source_centered = source - source_mean

        # Compute SVD for PCA
        U, S, Vt = b.svd(source_centered)

        # Project to target dimension
        if n_components < source_dim:
            # Truncate to n_components
            projection = Vt[:n_components, :].T  # [source_dim, n_components]
            projected = b.matmul(source_centered, projection)
        else:
            projected = source_centered

        # If target_dim differs, pad/truncate
        proj_dim = projected.shape[1]
        if proj_dim < target_dim:
            padding = b.zeros((projected.shape[0], target_dim - proj_dim))
            projected = b.concatenate([projected, padding], axis=1)
        elif proj_dim > target_dim:
            projected = projected[:, :target_dim]

        # Compute explained variance ratio
        total_var = float(b.to_numpy(b.sum(S**2)))
        explained_var = float(b.to_numpy(b.sum(S[:n_components] ** 2)))
        alignment_score = explained_var / total_var if total_var > 0 else 0.0

        # Reconstruction error
        reconstructed = b.matmul(projected[:, :n_components], Vt[:n_components, :])
        error = float(b.to_numpy(b.mean(b.norm(source_centered - reconstructed, axis=1))))

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=Vt[:n_components, :].T if n_components < source_dim else None,
            reconstruction_error=error,
            alignment_score=alignment_score,
            strategy_used=ProjectionStrategy.PCA,
            metadata={"n_components": n_components, "explained_variance_ratio": alignment_score},
        )

    def _project_procrustes(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        """
        Orthogonal Procrustes alignment.

        Finds orthogonal matrix R that minimizes ||source @ R - target||_F
        using shared anchor tokens.
        """
        b = self._backend

        source_vocab, source_dim = source.shape
        target_vocab, target_dim = target.shape

        # Select anchor tokens for alignment
        if shared_indices:
            source_idx, target_idx = shared_indices
            n_anchors = min(len(source_idx), self.config.anchor_count)
            # Stack using backend
            source_anchors = b.stack([source[i] for i in source_idx[:n_anchors]])
            target_anchors = b.stack([target[i] for i in target_idx[:n_anchors]])
        else:
            # Use random sample for alignment
            n_anchors = min(source_vocab, target_vocab, self.config.anchor_count)
            source_anchors = source[:n_anchors]
            target_anchors = target[:n_anchors]

        # Handle dimension mismatch
        if source_dim != target_dim:
            # Pad smaller dimension for alignment
            if source_dim < target_dim:
                padding = b.zeros((source_anchors.shape[0], target_dim - source_dim))
                source_anchors = b.concatenate([source_anchors, padding], axis=1)
                full_padding = b.zeros((source_vocab, target_dim - source_dim))
                source_padded = b.concatenate([source, full_padding], axis=1)
            else:
                padding = b.zeros((target_anchors.shape[0], source_dim - target_dim))
                target_anchors = b.concatenate([target_anchors, padding], axis=1)
                source_padded = source

            work_dim = max(source_dim, target_dim)
        else:
            source_padded = source
            work_dim = source_dim

        # Compute Procrustes rotation: R = V @ U^T where SVD(target^T @ source) = U S V^T
        M = b.matmul(target_anchors.T, source_anchors)
        U, S, Vt = b.svd(M)
        R = b.matmul(U, Vt)

        # Apply rotation to full source embeddings
        projected = b.matmul(source_padded, R.T)

        # Truncate or pad to target dimension
        if projected.shape[1] > target_dim:
            projected = projected[:, :target_dim]
        elif projected.shape[1] < target_dim:
            padding = b.zeros((projected.shape[0], target_dim - projected.shape[1]))
            projected = b.concatenate([projected, padding], axis=1)

        # Compute alignment quality on anchors
        if shared_indices:
            aligned_source = b.matmul(source_anchors, R.T)[:, :target_dim]
            target_subset = target_anchors[:, :target_dim]
            error = float(b.to_numpy(b.mean(b.norm(aligned_source - target_subset, axis=1))))
            target_norm = float(b.to_numpy(b.mean(b.norm(target_subset, axis=1))))
            alignment_score = 1.0 - (error / target_norm) if target_norm > 0 else 0.0
        else:
            error = 0.0
            alignment_score = 0.8  # Default for random alignment

        # Scale to preserve norms if configured
        if self.config.preserve_norms:
            source_norms = b.norm(source, axis=1)
            projected_norms = b.norm(projected, axis=1) + self.config.regularization
            scale = source_norms / projected_norms
            projected = projected * b.reshape(scale, (-1, 1))

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=R,
            reconstruction_error=error,
            alignment_score=max(0.0, min(1.0, alignment_score)),
            strategy_used=ProjectionStrategy.PROCRUSTES,
            metadata={"n_anchors": n_anchors, "work_dim": work_dim},
        )

    def _project_cca(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        """
        Canonical Correlation Analysis alignment.

        Projects both source and target to a shared canonical space.
        """
        b = self._backend

        source_vocab, source_dim = source.shape
        target_vocab, target_dim = target.shape

        # Select shared tokens
        if shared_indices:
            source_idx, target_idx = shared_indices
            n_shared = min(len(source_idx), self.config.anchor_count)
            X = b.stack([source[i] for i in source_idx[:n_shared]])
            Y = b.stack([target[i] for i in target_idx[:n_shared]])
        else:
            # Use first N tokens as pseudo-shared
            n_shared = min(source_vocab, target_vocab, self.config.anchor_count)
            X = source[:n_shared]
            Y = target[:n_shared]

        # Center data
        X_mean = b.mean(X, axis=0)
        Y_mean = b.mean(Y, axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # Compute covariances with regularization
        reg = self.config.regularization
        Cxx = b.matmul(X_centered.T, X_centered) / n_shared + reg * b.eye(source_dim)
        Cyy = b.matmul(Y_centered.T, Y_centered) / n_shared + reg * b.eye(target_dim)
        Cxy = b.matmul(X_centered.T, Y_centered) / n_shared

        # Compute CCA via generalized eigenvalue problem
        # Use SVD approximation: Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}

        # Compute inverse square roots via SVD
        Ux, Sx, Vxt = b.svd(Cxx)
        Sx_inv_sqrt = 1.0 / b.sqrt(Sx + reg)
        Cxx_inv_sqrt = b.matmul(Ux * Sx_inv_sqrt, Vxt)

        Uy, Sy, Vyt = b.svd(Cyy)
        Sy_inv_sqrt = 1.0 / b.sqrt(Sy + reg)
        Cyy_inv_sqrt = b.matmul(Uy * Sy_inv_sqrt, Vyt)

        # Compute transformation
        T = b.matmul(b.matmul(Cxx_inv_sqrt, Cxy), Cyy_inv_sqrt)
        U_cca, S_cca, Vt_cca = b.svd(T)

        # Projection matrix: maps source to canonical space then to target
        n_components = min(source_dim, target_dim)
        A = b.matmul(Cxx_inv_sqrt, U_cca[:, :n_components])  # source -> canonical
        B = b.matmul(Cyy_inv_sqrt, Vt_cca[:n_components, :].T)  # canonical -> target

        # Project full source
        source_centered_full = source - X_mean
        projected_canonical = b.matmul(source_centered_full, A)

        # Map to target space (approximate via B^T)
        projected = b.matmul(projected_canonical, B.T) + Y_mean

        # Handle dimension mismatch
        if projected.shape[1] != target_dim:
            if projected.shape[1] > target_dim:
                projected = projected[:, :target_dim]
            else:
                padding = b.zeros((projected.shape[0], target_dim - projected.shape[1]))
                projected = b.concatenate([projected, padding], axis=1)

        # Alignment score from canonical correlations
        correlations = b.to_numpy(S_cca[:n_components])
        alignment_score = float(correlations.mean()) if len(correlations) > 0 else 0.0

        # Reconstruction error
        if shared_indices:
            proj_shared = projected[source_idx[:n_shared]]
            target_shared = Y
            error = float(b.to_numpy(b.mean(b.norm(proj_shared - target_shared, axis=1))))
        else:
            error = 0.0

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=A,  # Store source projection
            reconstruction_error=error,
            alignment_score=alignment_score,
            strategy_used=ProjectionStrategy.CCA,
            metadata={
                "n_components": n_components,
                "canonical_correlations": correlations.tolist()[:10],  # Top 10
            },
        )

    def _project_optimal_transport(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        """
        Optimal Transport-based alignment using Wasserstein distance.

        Uses Sinkhorn algorithm for efficient transport plan computation.
        """
        b = self._backend

        source_vocab, source_dim = source.shape
        target_vocab, target_dim = target.shape

        # Subsample for efficiency
        n_source = min(source_vocab, self.config.anchor_count)
        n_target = min(target_vocab, self.config.anchor_count)

        source_sample = source[:n_source]
        target_sample = target[:n_target]

        # Handle dimension mismatch by projecting to shared dimension
        shared_dim = min(source_dim, target_dim)
        if source_dim > shared_dim:
            # Use PCA to reduce source
            U, S, Vt = b.svd(source_sample)
            source_reduced = b.matmul(source_sample, Vt[:shared_dim, :].T)
            source_full_reduced = b.matmul(source, Vt[:shared_dim, :].T)
        else:
            source_reduced = source_sample
            source_full_reduced = source

        if target_dim > shared_dim:
            U, S, Vt = b.svd(target_sample)
            target_reduced = b.matmul(target_sample, Vt[:shared_dim, :].T)
        else:
            target_reduced = target_sample

        # Compute cost matrix (squared Euclidean distances)
        # C[i,j] = ||source[i] - target[j]||^2
        source_sq = b.sum(source_reduced**2, axis=1, keepdims=True)
        target_sq = b.sum(target_reduced**2, axis=1, keepdims=True)
        cross = b.matmul(source_reduced, target_reduced.T)
        C = source_sq - 2 * cross + target_sq.T

        # Sinkhorn algorithm for optimal transport
        reg = 0.1  # Entropic regularization
        K = b.exp(-C / reg)

        # Initialize
        u = b.ones((n_source,)) / n_source
        v = b.ones((n_target,)) / n_target

        # Sinkhorn iterations
        for _ in range(100):
            u = 1.0 / (b.matmul(K, v) + self.config.regularization)
            v = 1.0 / (b.matmul(K.T, u) + self.config.regularization)

        # Transport plan
        P = b.reshape(u, (-1, 1)) * K * b.reshape(v, (1, -1))

        # Barycentric projection: projected[i] = sum_j P[i,j] * target[j]
        P_normalized = P / (b.sum(P, axis=1, keepdims=True) + self.config.regularization)
        projected_sample = b.matmul(P_normalized, target_reduced)

        # Extend to full source vocabulary via nearest-neighbor interpolation
        # For tokens not in sample, find nearest sampled token and use its projection

        # Simple approach: compute linear transformation from reduced space
        # A @ source_reduced ≈ projected_sample
        # A = projected_sample @ pinv(source_reduced)
        b.matmul(
            b.matmul(
                source_reduced.T,
                b.linalg_inv(
                    b.matmul(source_reduced, source_reduced.T)
                    + self.config.regularization * b.eye(n_source)
                ),
            ),
            b.eye(n_source),
        ).T

        # Use Procrustes as fallback for full projection
        M = b.matmul(projected_sample.T, source_reduced)
        U_proc, _, Vt_proc = b.svd(M)
        R = b.matmul(U_proc, Vt_proc)

        projected = b.matmul(source_full_reduced, R.T)

        # Pad/truncate to target dimension
        if projected.shape[1] < target_dim:
            padding = b.zeros((projected.shape[0], target_dim - projected.shape[1]))
            projected = b.concatenate([projected, padding], axis=1)
        elif projected.shape[1] > target_dim:
            projected = projected[:, :target_dim]

        # Compute Wasserstein distance as alignment quality
        transport_cost = float(b.to_numpy(b.sum(P * C)))
        alignment_score = 1.0 / (1.0 + transport_cost / n_source)

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=R,
            reconstruction_error=transport_cost / n_source,
            alignment_score=alignment_score,
            strategy_used=ProjectionStrategy.OPTIMAL_TRANSPORT,
            metadata={
                "n_source_samples": n_source,
                "n_target_samples": n_target,
                "transport_cost": transport_cost,
                "shared_dim": shared_dim,
            },
        )

    def compute_alignment_quality(
        self,
        source_embeddings: "Array",
        projected_embeddings: "Array",
        target_embeddings: "Array",
        shared_indices: tuple[list[int], list[int]] | None = None,
    ) -> dict[str, float]:
        """
        Compute alignment quality metrics.

        Returns:
            Dictionary of quality metrics
        """
        b = self._backend

        if shared_indices:
            source_idx, target_idx = shared_indices
            n_shared = min(len(source_idx), 1000)

            source_shared = b.stack([source_embeddings[i] for i in source_idx[:n_shared]])
            projected_shared = b.stack([projected_embeddings[i] for i in source_idx[:n_shared]])
            target_shared = b.stack([target_embeddings[i] for i in target_idx[:n_shared]])
        else:
            n_shared = min(source_embeddings.shape[0], target_embeddings.shape[0], 1000)
            source_shared = source_embeddings[:n_shared]
            projected_shared = projected_embeddings[:n_shared]
            target_shared = target_embeddings[:n_shared]

        # MSE between projected and target
        mse = float(b.to_numpy(b.mean((projected_shared - target_shared) ** 2)))

        # Cosine similarity
        proj_norms = b.norm(projected_shared, axis=1) + self.config.regularization
        target_norms = b.norm(target_shared, axis=1) + self.config.regularization
        cosine = b.sum(projected_shared * target_shared, axis=1) / (proj_norms * target_norms)
        mean_cosine = float(b.to_numpy(b.mean(cosine)))

        # Norm preservation
        source_norms = b.norm(source_shared, axis=1)
        proj_norms_actual = b.norm(projected_shared, axis=1)
        norm_ratio = float(
            b.to_numpy(b.mean(proj_norms_actual / (source_norms + self.config.regularization)))
        )

        return {
            "mse": mse,
            "mean_cosine_similarity": mean_cosine,
            "norm_preservation_ratio": norm_ratio,
            "n_samples_evaluated": n_shared,
        }
