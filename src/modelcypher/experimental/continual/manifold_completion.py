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
Manifold completion via geometric constraints and optional retrieval.

Identifies sparse regions, interpolates from neighbors, optionally uses
external retrieval, and encodes the resulting target via null-space projection.

References:
    - Word2Vec analogy geometry (Mikolov et al., 2013)
    - Manifold hypothesis in deep learning
    - Geometric deep learning (Bronstein et al., 2021)
    - Sleep-Time Compute (Letta 2025) - consolidation during idle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.continual.knowledge_encoder import KnowledgeEncoder
from modelcypher.core.domain.geometry.null_space_tracker import NullSpaceTracker
from modelcypher.experimental.continual.surprise_detector import SurpriseEvent

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Type alias for external knowledge retrieval function
# Signature: (sparse_embedding, neighbor_indices) -> (attractor_vector, confidence) | None
# The function queries external sources (web, RAG, aligned model, etc.) and returns
# an attractor vector that pulls the sparse point toward "ground truth"
RetrievalFunction = Callable[
    ["Array", list[int]],  # (sparse_embedding, neighbor_indices)
    tuple["Array", float] | None,  # (attractor_vector, confidence) or None if no knowledge
]


@dataclass(frozen=True)
class CompletionStep:
    """Result of a single manifold completion step.

    Attributes:
        iteration: Current iteration number.
        sparse_point_idx: Index of the sparse point being completed.
        initial_entropy: Entropy before completion.
        final_entropy: Entropy after completion.
        entropy_reduction: Reduction achieved.
        constraint_loss: Final constraint satisfaction loss.
        n_neighbors_used: Number of dense neighbors used.
        encoding_applied: Whether a weight update was applied.
        converged: Whether this point converged.
    """

    iteration: int
    sparse_point_idx: int
    initial_entropy: float
    final_entropy: float
    entropy_reduction: float
    constraint_loss: float
    n_neighbors_used: int
    encoding_applied: bool
    converged: bool

    def as_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "sparse_point_idx": self.sparse_point_idx,
            "initial_entropy": self.initial_entropy,
            "final_entropy": self.final_entropy,
            "entropy_reduction": self.entropy_reduction,
            "constraint_loss": self.constraint_loss,
            "n_neighbors_used": self.n_neighbors_used,
            "encoding_applied": self.encoding_applied,
            "converged": self.converged,
        }


class ManifoldCompletion:
    """Self-guided manifold completion via geometric constraints.

    Fills in sparse regions of the model's representational manifold
    by following invariant geometric relationships.

    Usage:
        completion = ManifoldCompletion(model, tracker, encoder)

        # Generate probe points covering the manifold
        probes = completion.generate_probe_points(n_probes=1000)

        # Run completion
        for step in completion.complete(probes):
            print(f"Iteration {step.iteration}: "
                  f"entropy {step.initial_entropy:.3f} -> {step.final_entropy:.3f}")

            if step.converged:
                print("Converged!")
                break
    """

    def __init__(
        self,
        model: Any,
        null_space_tracker: NullSpaceTracker,
        knowledge_encoder: KnowledgeEncoder,
        backend: Backend,
        knowledge_retrieval_fn: RetrievalFunction | None = None,
    ) -> None:
        """Initialize manifold completion.

        Args:
            model: The language model.
            null_space_tracker: Tracker for null-space availability.
            knowledge_encoder: Encoder for weight updates.
            backend: Compute backend.
            knowledge_retrieval_fn: Optional function to query external knowledge.
                Signature: (sparse_embedding, neighbor_indices) -> (attractor, confidence) | None
                When provided, completion blends local geometry with external attractors.
        """
        self._backend = backend
        self._model = model
        self._tracker = null_space_tracker
        self._encoder = knowledge_encoder
        self._retrieval_fn = knowledge_retrieval_fn

        # State
        self._iteration = 0
        self._best_mean_entropy = float("inf")

        # Derived thresholds (computed once from dtype)
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            sqrt_scalar,
        )
        sample = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, sample)
        self._sqrt_eps = sqrt_scalar(eps, self._backend)

    def complete(
        self,
        probe_embeddings: Array,
        probe_ids: list[int] | None = None,
    ) -> Iterator[CompletionStep]:
        """Run manifold completion on probe points.

        Uses convergence-based stopping with geometry-derived parameters:
        - Stops when entropy is below sqrt(machine_epsilon) (numerically zero)
        - k_neighbors derived from intrinsic dimension estimate
        - step_size derived from condition number for numerical stability

        Args:
            probe_embeddings: Embeddings of probe points [n_probes, hidden_dim].
            probe_ids: Optional token IDs for probes (for encoding).

        Yields:
            CompletionStep for each iteration.
        """
        b = self._backend

        n_probes = int(probe_embeddings.shape[0])
        if probe_ids is None:
            probe_ids = list(range(n_probes))

        # Derive parameters from data
        convergence_threshold = self._get_convergence_threshold()
        k_neighbors = self._get_k_neighbors(probe_embeddings)
        step_size = self._get_step_size(probe_embeddings)

        # Track density per probe
        densities = self._compute_densities(probe_embeddings, k_neighbors)

        # Safety bound: prevent infinite loops (n_probes iterations is sufficient)
        max_iterations = n_probes
        self._iteration = 0

        while self._iteration < max_iterations:
            # Find sparsest point
            sparse_idx = self._find_sparsest(densities)
            sparse_entropy = 1.0 - densities[sparse_idx]  # Density ~ 1 - entropy

            # Check convergence: entropy below numerical precision threshold
            if sparse_entropy < convergence_threshold:
                yield CompletionStep(
                    iteration=self._iteration,
                    sparse_point_idx=sparse_idx,
                    initial_entropy=sparse_entropy,
                    final_entropy=sparse_entropy,
                    entropy_reduction=0.0,
                    constraint_loss=0.0,
                    n_neighbors_used=0,
                    encoding_applied=False,
                    converged=True,
                )
                break

            # Find dense neighbors
            neighbors = self._find_dense_neighbors(
                probe_embeddings, sparse_idx, k_neighbors
            )

            if len(neighbors) < 2:
                # Not enough dense neighbors - skip this point
                self._iteration += 1
                continue

            # Query external knowledge source if available
            # The retrieval function gets the sparse embedding and neighbor indices,
            # allowing it to query external sources (web, RAG, aligned model, etc.)
            external_attractor: tuple[Array, float] | None = None
            if self._retrieval_fn is not None:
                sparse_embedding = b.take(probe_embeddings, b.array([sparse_idx]), axis=0)[0]
                try:
                    external_attractor = self._retrieval_fn(sparse_embedding, neighbors)
                except Exception:
                    # Retrieval failed - continue with local geometry only
                    external_attractor = None

            # Compute constraint target, blending local geometry with external knowledge
            target_position = self._solve_constraints(
                probe_embeddings, sparse_idx, neighbors, external_attractor
            )

            # Compute loss before update
            initial_loss = self._compute_constraint_loss(
                probe_embeddings, sparse_idx, neighbors, target_position
            )

            # Encode the update
            sparse_embedding = b.take(probe_embeddings, b.array([sparse_idx]), axis=0)[0]
            delta = target_position - sparse_embedding

            # Create a synthetic surprise event for encoding
            # Uses raw metrics - caller determines encoding threshold
            activation_delta = float(b.to_scalar(b.sum(delta * delta))) ** 0.5
            event = SurpriseEvent(
                timestep=self._iteration,
                token_id=probe_ids[sparse_idx],
                predicted_token_id=probe_ids[sparse_idx],
                token_surprise=sparse_entropy,
                token_surprise_baseline=0.0,  # No baseline for synthetic event
                token_surprise_zscore=0.0,  # Not applicable
                rank_surprise=0,
                rank_log=0.0,
                activation_surprise=activation_delta,
                percentile=1.0,  # Synthetic events are maximally surprising
                context_tokens=tuple(),
            )

            # Encode via null-space projection
            results = self._encoder.encode(
                event=event,
                hidden_state=target_position,
            )

            encoding_applied = any(r.applied for r in results)

            # Update probe embedding if encoding was applied
            if encoding_applied:
                # Update the embedding with a step toward target
                # step_size derived from condition number for stability
                new_embedding = (
                    sparse_embedding
                    + step_size * (target_position - sparse_embedding)
                )
                # This is approximate - true update would re-run forward pass
                probe_embeddings = self._update_embedding(
                    probe_embeddings, sparse_idx, new_embedding
                )

            # Recompute density at sparse point
            new_densities = self._compute_densities(probe_embeddings, k_neighbors)
            new_entropy = 1.0 - new_densities[sparse_idx]

            mean_entropy = sum(1.0 - d for d in new_densities) / len(new_densities)
            if mean_entropy < self._best_mean_entropy:
                self._best_mean_entropy = mean_entropy

            # Compute final loss
            final_loss = self._compute_constraint_loss(
                probe_embeddings, sparse_idx, neighbors, target_position
            )

            yield CompletionStep(
                iteration=self._iteration,
                sparse_point_idx=sparse_idx,
                initial_entropy=sparse_entropy,
                final_entropy=new_entropy,
                entropy_reduction=sparse_entropy - new_entropy,
                constraint_loss=final_loss,
                n_neighbors_used=len(neighbors),
                encoding_applied=encoding_applied,
                converged=False,
            )

            # Update densities
            densities = new_densities
            self._iteration += 1


    def _get_convergence_threshold(self) -> float:
        """Get convergence threshold, derived from sqrt(machine_epsilon) if not set.

        sqrt(eps) is the natural threshold below which values are numerically
        indistinguishable from zero in relative terms.
        """
        return self._sqrt_eps

    def _get_k_neighbors(self, embeddings: Array) -> int:
        """Get k_neighbors, derived from intrinsic dimension if not set.

        Uses k = max(2, int(intrinsic_dim + 1)) where intrinsic_dim is estimated
        from the data's effective rank.
        """
        b = self._backend
        n = int(embeddings.shape[0])

        # Estimate intrinsic dimension from effective rank
        # Effective rank = exp(entropy of normalized singular values)
        centered = embeddings - b.mean(embeddings, axis=0, keepdims=True)
        _, s, _ = b.svd(centered, full_matrices=False)
        b.eval(s)

        # Normalize singular values to probability distribution
        s_sum = b.sum(s)
        b.eval(s_sum)
        s_sum_val = float(b.to_scalar(s_sum))
        if s_sum_val < self._sqrt_eps:
            # Degenerate case: all zeros
            return 2

        s_norm = s / s_sum
        # Entropy: -sum(p * log(p)), avoiding log(0)
        log_s = b.log(s_norm + self._sqrt_eps)
        entropy = -b.sum(s_norm * log_s)
        b.eval(entropy)
        entropy_val = float(b.to_scalar(entropy))

        # Effective rank = exp(entropy)
        intrinsic_dim = float(b.to_scalar(b.exp(b.array([entropy_val]))))

        # k = intrinsic_dim + 1, minimum 2 for meaningful neighbors
        k = max(2, int(intrinsic_dim + 1))
        return min(k, n - 1)  # Can't exceed available points

    def _get_step_size(self, embeddings: Array) -> float:
        """Get step_size, derived from condition number if not set.

        Uses step = 1.0 / condition_number for numerical stability.
        A well-conditioned system (κ ≈ 1) allows full steps.
        An ill-conditioned system (κ >> 1) requires smaller steps.
        """
        b = self._backend

        # Compute condition number of the Gram matrix
        centered = embeddings - b.mean(embeddings, axis=0, keepdims=True)
        gram = b.matmul(centered, b.transpose(centered))
        eigvals = b.eigh(gram)[0]
        b.eval(eigvals)

        # Condition number = max(eigval) / min(eigval)
        max_eig = float(b.to_scalar(b.max(eigvals)))
        min_eig = float(b.to_scalar(b.min(b.maximum(eigvals, b.array([self._sqrt_eps])))))

        if min_eig < self._sqrt_eps:
            # Near-singular: use very small steps
            return self._sqrt_eps

        condition_number = max_eig / min_eig
        return 1.0 / condition_number

    def _compute_densities(self, embeddings: Array, k: int | None = None) -> list[float]:
        """Compute local density for each embedding.

        Uses k-NN distance as a density proxy.

        Args:
            embeddings: Probe embeddings [n_probes, hidden_dim].
            k: Number of neighbors (derived from intrinsic dim if None).
        """
        b = self._backend

        n = int(embeddings.shape[0])
        if k is None:
            k = self._get_k_neighbors(embeddings)
        k = min(k, n - 1)

        # Compute pairwise distances
        # ||a - b||² = ||a||² + ||b||² - 2<a,b>
        norms_sq = b.sum(embeddings * embeddings, axis=1, keepdims=True)  # [n, 1]
        dots = b.matmul(embeddings, b.transpose(embeddings))  # [n, n]
        dists_sq = norms_sq + b.transpose(norms_sq) - 2 * dots  # [n, n]

        # Clamp to non-negative
        dists_sq = b.maximum(dists_sq, b.zeros_like(dists_sq))

        b.eval(dists_sq)

        # For each point, find k-nearest neighbor distance
        kth_dists: list[float] = []
        for i in range(n):
            row = b.take(dists_sq, b.array([i]), axis=0)[0]
            # Sort distances
            sorted_dists = b.sort(row)
            # Take k-th distance (skip 0 which is self)
            kth_dist_sq = b.take(sorted_dists, b.array([k]), axis=0)
            b.eval(kth_dist_sq)
            kth_dist = float(b.to_scalar(kth_dist_sq)) ** 0.5
            kth_dists.append(kth_dist)

        if not kth_dists:
            return []

        # Scale distances by the mean k-NN distance (data-derived)
        mean_kth = sum(kth_dists) / len(kth_dists)
        scale = mean_kth if mean_kth > 0 else 1.0

        # Density proxy in [0, 1], normalized by data scale
        return [1.0 / (1.0 + (d / scale)) for d in kth_dists]

    def _find_sparsest(self, densities: list[float]) -> int:
        """Find index of sparsest point."""
        return densities.index(min(densities))

    def _find_dense_neighbors(
        self,
        embeddings: Array,
        sparse_idx: int,
        k_neighbors: int | None = None,
    ) -> list[int]:
        """Find nearest neighbors of a sparse point.

        Args:
            embeddings: Probe embeddings [n_probes, hidden_dim].
            sparse_idx: Index of the sparse point.
            k_neighbors: Max neighbors to return (derived if None).
        """
        b = self._backend

        n = int(embeddings.shape[0])
        if k_neighbors is None:
            k_neighbors = self._get_k_neighbors(embeddings)

        # Get sparse point embedding
        sparse_embed = b.take(embeddings, b.array([sparse_idx]), axis=0)[0]

        # Compute distances to all points
        diffs = embeddings - sparse_embed[None, :]  # [n, d]
        dists_sq = b.sum(diffs * diffs, axis=1)  # [n]
        b.eval(dists_sq)

        dists_list = b.tolist(dists_sq)

        # Sort by distance
        indexed = [(i, dists_list[i]) for i in range(n) if i != sparse_idx]
        indexed.sort(key=lambda x: x[1])  # Sort by distance

        # Take k nearest neighbors
        neighbors = []
        for idx, _dist in indexed:
            neighbors.append(idx)
            if len(neighbors) >= k_neighbors:
                break

        return neighbors

    def _solve_constraints(
        self,
        embeddings: Array,
        sparse_idx: int,
        neighbors: list[int],
        external_attractor: tuple[Array, float] | None = None,
    ) -> Array:
        """Solve for target position using geometric constraints + external knowledge.

        Uses weighted average of relationships from neighbors, optionally blended
        with an external attractor from a knowledge source.

        Args:
            embeddings: All probe embeddings.
            sparse_idx: Index of the sparse point.
            neighbors: Indices of dense neighbors.
            external_attractor: Optional (attractor_vector, confidence) from external source.
                When provided, final target blends local geometry with external knowledge.
        """
        b = self._backend

        # Get sparse and neighbor embeddings
        sparse_embed = b.take(embeddings, b.array([sparse_idx]), axis=0)[0]
        neighbor_embeds = b.take(embeddings, b.array(neighbors), axis=0)

        # Compute centroid of neighbors as initial target
        # This preserves local structure
        centroid = b.mean(neighbor_embeds, axis=0)

        # Compute relationship-preserving adjustment
        # For each neighbor pair, compute the relationship direction
        # and use it to refine the target
        n_neighbors = len(neighbors)
        if n_neighbors < 2:
            local_target = centroid
        else:
            # Compute pairwise relationship directions between neighbors
            # R(N_i, N_j) = (N_i - N_j) / ||N_i - N_j||
            # The sparse point should have similar relationships to these neighbors

            # Simple approach: weighted interpolation based on distances
            # More sophisticated: solve least squares for relationship preservation

            # Compute weights based on inverse distance to sparse point
            diffs_to_sparse = neighbor_embeds - sparse_embed[None, :]
            dists_to_sparse = b.sqrt(b.sum(diffs_to_sparse * diffs_to_sparse, axis=1))
            b.eval(dists_to_sparse)

            # Inverse distance weights with dtype-derived epsilon for stability
            from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
            eps = division_epsilon(b, dists_to_sparse)
            weights = 1.0 / (dists_to_sparse + eps)
            weights = weights / b.sum(weights)  # Normalize
            b.eval(weights)

            # Weighted average of neighbors
            local_target = b.sum(neighbor_embeds * weights[:, None], axis=0)

            # Blend with sparse point - alpha derived from relative density contrast
            # High density contrast = trust neighbors more (move more toward target)
            # Low density contrast = sparse point may be intentionally sparse (move less)
            # alpha = mean(neighbor_densities) / (mean(neighbor_densities) + sparse_density)
            # This is the natural geometric ratio, not a hardcoded constant
            #
            # For simplicity, use the weight variance as a proxy for confidence:
            # High variance = one neighbor dominates = trust target more
            # Low variance = neighbors agree = trust target more
            # Since weights are normalized, variance is bounded [0, 1/n] to [1-1/n, 0]
            weights_var = b.sum((weights - 1.0 / n_neighbors) ** 2) * n_neighbors
            b.eval(weights_var)

            # Map variance to blend: low variance (neighbors agree) = higher alpha
            # alpha = 1 - sqrt(weights_var)  (derived from Cauchy-Schwarz bound)
            alpha_value = 1.0 - float(b.to_scalar(b.sqrt(weights_var)))
            alpha_value = max(0.0, min(1.0, alpha_value))  # Clamp to [0, 1]

            local_target = alpha_value * local_target + (1 - alpha_value) * sparse_embed

        # Blend with external attractor if provided
        # The external attractor represents "ground truth" from an external source
        # (web search, RAG, aligned model, knowledge graph, etc.)
        if external_attractor is not None:
            ext_vector, ext_confidence = external_attractor
            # Confidence in [0, 1] determines blend weight
            # High confidence = trust external source more
            # Low confidence = trust local geometry more
            ext_confidence = max(0.0, min(1.0, ext_confidence))
            target = ext_confidence * ext_vector + (1 - ext_confidence) * local_target
        else:
            target = local_target

        b.eval(target)
        return target

    def _compute_constraint_loss(
        self,
        embeddings: Array,
        sparse_idx: int,
        neighbors: list[int],
        target: Array,
    ) -> float:
        """Compute constraint satisfaction loss."""
        b = self._backend

        # Loss = distance from target
        sparse_embed = b.take(embeddings, b.array([sparse_idx]), axis=0)[0]
        diff = sparse_embed - target
        loss = b.sum(diff * diff)
        b.eval(loss)

        return float(b.to_scalar(loss))

    def _update_embedding(
        self,
        embeddings: Array,
        idx: int,
        new_embed: Array,
    ) -> Array:
        """Update embedding at index (for tracking, not model update)."""
        b = self._backend

        # Create mask
        n = int(embeddings.shape[0])
        mask = b.zeros((n, 1))
        # Can't do direct assignment, so we use a workaround
        mask_list = [[0.0]] * n
        mask_list[idx] = [1.0]
        mask = b.array(mask_list)

        # Update: old * (1 - mask) + new * mask
        updated = embeddings * (1 - mask) + new_embed[None, :] * mask

        b.eval(updated)
        return updated

    def estimate_completion_coverage(self, embeddings: Array) -> dict[str, float]:
        """Compute raw coverage statistics for the manifold.

        Returns raw measurements without hardcoded thresholds.
        Caller interprets these based on application requirements.

        Returns:
            Dict with:
                mean_density: Mean local density across all points.
                min_density: Minimum local density (sparsest region).
                max_density: Maximum local density (densest region).
                std_density: Standard deviation of densities.
                density_contrast: max/min ratio (higher = more uneven).
                n_points: Number of probe points.
        """
        densities = self._compute_densities(embeddings)

        n = len(densities)
        mean_d = sum(densities) / n
        min_d = min(densities)
        max_d = max(densities)
        var_d = sum((d - mean_d) ** 2 for d in densities) / n
        std_d = var_d ** 0.5

        # Density contrast - ratio of max to min
        # Higher = more uneven coverage
        contrast = max_d / min_d if min_d > 0 else float("inf")

        return {
            "mean_density": mean_d,
            "min_density": min_d,
            "max_density": max_d,
            "std_density": std_d,
            "density_contrast": contrast,
            "n_points": float(n),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get completion statistics."""
        return {
            "iterations_run": self._iteration,
            "best_mean_entropy": self._best_mean_entropy,
        }

    def reset(self) -> None:
        """Reset completion state."""
        self._iteration = 0
        self._best_mean_entropy = float("inf")
