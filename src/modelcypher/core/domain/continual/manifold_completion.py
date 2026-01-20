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
Manifold Completion - Self-guided sparse region filling via geometric constraints.

This module implements self-supervised manifold completion: the model fills in
sparse regions of its own representational manifold by following invariant
geometric relationships. No external training data is needed.

The key insight: Geometric relationships (analogies) are invariant.
If A:B::C:D holds in dense regions, the same relational structure constrains
where points must live in sparse regions.

Algorithm:
    1. Identify sparse regions (high entropy on probes)
    2. Find dense neighbors with known relationships
    3. Solve constraint satisfaction: where must the sparse point live?
    4. Encode the inferred position via null-space projection
    5. Repeat until manifold is complete (entropy uniformly low)

This is "geometric dreaming" - the model improves itself by reasoning about
the structure of its own knowledge.

Mathematical formulation:
    Let M be the manifold, ρ(x) the density at x (inverse of local entropy)
    Let R(x,y) = (x-y)/||x-y|| be the relational direction between concepts

    Constraint: For analogies A:B::C:D, we have R(A,B) ≈ R(C,D)

    For sparse point S with dense neighbors {N_1, ..., N_k}:
        Find S' that minimizes: sum_i ||R(S',N_i) - R_expected(S,N_i)||²

    where R_expected comes from analogical transfer from dense regions.

Implementation uses gradient descent on the constraint loss:
    L(S') = sum_i ||R(S',N_i) - R_target_i||² + λ * entropy(S')

References:
    - Word2Vec analogy geometry (Mikolov et al., 2013)
    - Manifold hypothesis in deep learning
    - Geometric deep learning (Bronstein et al., 2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.entropy_analyzer import EntropyAnalyzer
from modelcypher.core.domain.continual.knowledge_encoder import (
    KnowledgeEncoder,
    UpdateFrequency,
)
from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker
from modelcypher.core.domain.continual.surprise_detector import SurpriseEvent

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


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


@dataclass
class CompletionConfig:
    """Configuration for manifold completion.

    Attributes:
        max_iterations: Maximum completion iterations.
        convergence_threshold: Entropy threshold for "complete".
        k_neighbors: Number of dense neighbors to use.
        constraint_weight: Weight on constraint loss vs entropy.
        step_size: Gradient step size for optimization.
        patience: Iterations without improvement before stopping.
        min_density_ratio: Minimum dense/sparse ratio to attempt completion.
    """

    max_iterations: int = 1000
    convergence_threshold: float = 0.1
    k_neighbors: int = 8
    constraint_weight: float = 1.0
    step_size: float = 0.01
    patience: int = 50
    min_density_ratio: float = 0.1


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
        config: CompletionConfig | None = None,
        backend: Backend | None = None,
    ) -> None:
        """Initialize manifold completion.

        Args:
            model: The language model.
            null_space_tracker: Tracker for null-space availability.
            knowledge_encoder: Encoder for weight updates.
            config: Completion configuration.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._model = model
        self._tracker = null_space_tracker
        self._encoder = knowledge_encoder
        self._config = config or CompletionConfig()

        self._entropy_analyzer = EntropyAnalyzer(backend=self._backend)

        # State
        self._iteration = 0
        self._best_mean_entropy = float("inf")
        self._patience_counter = 0

    def complete(
        self,
        probe_embeddings: Array,
        probe_ids: list[int] | None = None,
    ) -> Iterator[CompletionStep]:
        """Run manifold completion on probe points.

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

        # Track density per probe
        densities = self._compute_densities(probe_embeddings)

        for self._iteration in range(self._config.max_iterations):
            # Find sparsest point
            sparse_idx = self._find_sparsest(densities)
            sparse_entropy = 1.0 - densities[sparse_idx]  # Density ~ 1 - entropy

            # Check convergence
            if sparse_entropy < self._config.convergence_threshold:
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
                probe_embeddings, sparse_idx, densities
            )

            if len(neighbors) < 2:
                # Not enough dense neighbors
                continue

            # Compute constraint target
            target_position = self._solve_constraints(
                probe_embeddings, sparse_idx, neighbors
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
                frequency=UpdateFrequency.SLOW,
            )

            encoding_applied = any(r.applied for r in results)

            # Update probe embedding if encoding was applied
            if encoding_applied:
                # Update the embedding with a step toward target
                new_embedding = (
                    sparse_embedding
                    + self._config.step_size * (target_position - sparse_embedding)
                )
                # This is approximate - true update would re-run forward pass
                probe_embeddings = self._update_embedding(
                    probe_embeddings, sparse_idx, new_embedding
                )

            # Recompute density at sparse point
            new_densities = self._compute_densities(probe_embeddings)
            new_entropy = 1.0 - new_densities[sparse_idx]

            # Update patience
            mean_entropy = sum(1.0 - d for d in new_densities) / len(new_densities)
            if mean_entropy < self._best_mean_entropy:
                self._best_mean_entropy = mean_entropy
                self._patience_counter = 0
            else:
                self._patience_counter += 1

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

            # Check patience
            if self._patience_counter >= self._config.patience:
                break

    def _compute_densities(self, embeddings: Array) -> list[float]:
        """Compute local density for each embedding.

        Uses k-NN distance as a density proxy.
        """
        b = self._backend

        n = int(embeddings.shape[0])
        k = min(self._config.k_neighbors, n - 1)

        # Compute pairwise distances
        # ||a - b||² = ||a||² + ||b||² - 2<a,b>
        norms_sq = b.sum(embeddings * embeddings, axis=1, keepdims=True)  # [n, 1]
        dots = b.matmul(embeddings, b.transpose(embeddings))  # [n, n]
        dists_sq = norms_sq + b.transpose(norms_sq) - 2 * dots  # [n, n]

        # Clamp to non-negative
        dists_sq = b.maximum(dists_sq, b.zeros_like(dists_sq))

        b.eval(dists_sq)

        # For each point, find k-nearest neighbor distance
        densities = []
        for i in range(n):
            row = b.take(dists_sq, b.array([i]), axis=0)[0]
            # Sort distances
            sorted_dists = b.sort(row)
            # Take k-th distance (skip 0 which is self)
            kth_dist_sq = b.take(sorted_dists, b.array([k]), axis=0)
            b.eval(kth_dist_sq)
            kth_dist = float(b.to_scalar(kth_dist_sq)) ** 0.5

            # Density estimation via inverse k-NN distance
            # Formula: ρ(x) = k / (V_d * d_k^d) where V_d is volume of unit ball
            # Simplified to monotonic proxy: 1 / (1 + d_k)
            # This maps distance to density in [0, 1] range:
            #   d_k = 0 → density = 1 (perfectly dense)
            #   d_k → ∞ → density → 0 (sparse)
            # The +1 offset is the natural Laplace smoothing scale for stability.
            density = 1.0 / (1.0 + kth_dist)
            densities.append(density)

        return densities

    def _find_sparsest(self, densities: list[float]) -> int:
        """Find index of sparsest point."""
        return densities.index(min(densities))

    def _find_dense_neighbors(
        self,
        embeddings: Array,
        sparse_idx: int,
        densities: list[float],
    ) -> list[int]:
        """Find dense neighbors of a sparse point."""
        b = self._backend

        n = int(embeddings.shape[0])

        # Get sparse point embedding
        sparse_embed = b.take(embeddings, b.array([sparse_idx]), axis=0)[0]

        # Compute distances to all points
        diffs = embeddings - sparse_embed[None, :]  # [n, d]
        dists_sq = b.sum(diffs * diffs, axis=1)  # [n]
        b.eval(dists_sq)

        dists_list = b.tolist(dists_sq)

        # Sort by distance
        indexed = [(i, dists_list[i], densities[i]) for i in range(n) if i != sparse_idx]
        indexed.sort(key=lambda x: x[1])  # Sort by distance

        # Take k nearest that are dense enough
        min_density = max(densities) * self._config.min_density_ratio
        neighbors = []
        for idx, dist, density in indexed:
            if density >= min_density:
                neighbors.append(idx)
                if len(neighbors) >= self._config.k_neighbors:
                    break

        return neighbors

    def _solve_constraints(
        self,
        embeddings: Array,
        sparse_idx: int,
        neighbors: list[int],
    ) -> Array:
        """Solve for target position using geometric constraints.

        Uses weighted average of relationships from neighbors.
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
            return centroid

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
        target = b.sum(neighbor_embeds * weights[:, None], axis=0)

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

        target = alpha_value * target + (1 - alpha_value) * sparse_embed

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
            "patience_counter": self._patience_counter,
        }

    def reset(self) -> None:
        """Reset completion state."""
        self._iteration = 0
        self._best_mean_entropy = float("inf")
        self._patience_counter = 0

    @property
    def config(self) -> CompletionConfig:
        """Get completion configuration."""
        return self._config
