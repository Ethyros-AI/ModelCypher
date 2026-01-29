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

"""Consensus correction for outlier concepts.

Computes weight deltas that move target concepts toward a consensus position
without null-space projection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.behavioral_norm import behavioral_norm
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    gpu_lstsq,
    precision_dtype,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.cross_grounding_transfer import (
        RelationalStressProfile,
    )
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectionResult:
    """Result of consensus correction."""

    weight_delta: "Array"  # Delta to add to target weights [out_dim, in_dim]
    activation_delta: "Array"  # Delta in activation space [n, out_dim]
    stress_reduction: float  # How much stress was reduced
    correction_magnitude: float  # Behavioral norm of weight delta


class ConsensusCorrector:
    """Compute correction deltas to move outlier concepts to consensus.

    Produces a direct correction delta (no null-space projection).
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize corrector.

        Args:
            backend: Compute backend (defaults to system default).
        """
        self._backend = backend or get_default_backend()

    def compute_correction_delta(
        self,
        target_position: "Array",
        consensus_stress: "Array",
        target_anchors: dict[str, "Array"],
        target_activations: "Array",
        target_weights: "Array",
    ) -> CorrectionResult:
        """Compute weight delta to move target toward consensus.

        Args:
            target_position: Target's current position [d] for this concept.
            consensus_stress: Consensus stress vector [n_anchors] (Fréchet mean).
            target_anchors: Dict mapping anchor names to positions [d].
            target_activations: Target activations [n, in_dim] for lstsq.
            target_weights: Target weights [out_dim, in_dim] for scale reference.

        Returns:
            CorrectionResult with weight delta and diagnostics.
        """
        b = self._backend

        # Step 1: Solve for consensus position using multilateration
        consensus_position = self._solve_position_from_stress(
            stress_vector=consensus_stress,
            anchor_positions=target_anchors,
        )
        b.eval(consensus_position)

        # Step 2: Compute activation delta
        # delta = consensus - target (move target toward consensus)
        delta_activation = consensus_position - target_position
        b.eval(delta_activation)

        # Reshape for lstsq: [1, out_dim] since this is one concept
        delta_activation_2d = b.reshape(delta_activation, (1, -1))

        # Step 3: Compute weight delta via least squares
        # Unlike addition, we do NOT project into null-space.
        #
        # Solve: activations @ W_delta.T = delta_activations
        # => W_delta = lstsq(activations.T, delta_activations.T).T
        #
        # But we need to handle the case where we have one sample
        # In that case, use pseudoinverse directly

        n_samples = int(b.shape(target_activations)[0])
        in_dim = int(b.shape(target_activations)[1])
        out_dim = int(b.shape(delta_activation_2d)[1])

        # For a single concept correction, we have one delta but potentially
        # multiple activation samples. Broadcast delta to match sample count.
        delta_broadcast = b.broadcast_to(delta_activation_2d, (n_samples, out_dim))
        b.eval(delta_broadcast)

        # Compute weight delta via least squares
        # Unlike addition, we do NOT project into null-space.
        #
        # Solve: activations @ W_delta.T = delta_broadcast
        # => W_delta.T = lstsq(activations, delta_broadcast)
        # => W_delta = lstsq(activations, delta_broadcast).T

        weight_delta = gpu_lstsq(
            b,
            target_activations,  # [n, in_dim]
            delta_broadcast,  # [n, out_dim]
        )  # [in_dim, out_dim]
        weight_delta = b.transpose(weight_delta)  # [out_dim, in_dim]

        b.eval(weight_delta)

        # Compute behavioral magnitude (impact on outputs)
        correction_magnitude = behavioral_norm(weight_delta, target_activations, b)

        # Stress reduction: geodesic distance between target and consensus stress
        target_stress = self._compute_stress_from_position(
            target_position, target_anchors
        )
        # Stack target and consensus stress to compute geodesic distance
        stress_pair = b.stack([target_stress, consensus_stress], axis=0)
        b.eval(stress_pair)
        rg = RiemannianGeometry(b)
        geo_result = rg.geodesic_distances(stress_pair, use_cache=False)
        stress_reduction = float(b.to_scalar(geo_result.distances[0, 1]))

        logger.info(
            "Correction computed: magnitude=%.4f, stress_reduction=%.4f",
            correction_magnitude,
            stress_reduction,
        )

        return CorrectionResult(
            weight_delta=weight_delta,
            activation_delta=delta_activation,
            stress_reduction=stress_reduction,
            correction_magnitude=correction_magnitude,
        )

    def apply_correction(
        self,
        target_weights: "Array",
        weight_delta: "Array",
    ) -> "Array":
        """Apply correction delta to target weights.

        Unlike null-space addition which preserves behavior,
        this intentionally changes behavior to fix errors.

        Args:
            target_weights: Original target weights [out_dim, in_dim].
            weight_delta: Correction delta [out_dim, in_dim].

        Returns:
            Corrected weights [out_dim, in_dim].
        """
        b = self._backend
        corrected = target_weights + weight_delta
        b.eval(corrected)
        return corrected

    def _solve_position_from_stress(
        self,
        stress_vector: "Array",
        anchor_positions: dict[str, "Array"],
    ) -> "Array":
        """Solve for position in target space that matches stress vector.

        Uses multilateration via linear solve (same math as cross_grounding_transfer).

        Args:
            stress_vector: Distances to each anchor [n_anchors].
            anchor_positions: Dict mapping anchor names to positions [d].

        Returns:
            Position [d] that minimizes stress residual.
        """
        b = self._backend

        anchor_list = sorted(anchor_positions.keys())
        n_anchors = len(anchor_list)

        if n_anchors < 2:
            # Can't multilaterate with < 2 anchors
            first_anchor = anchor_list[0] if anchor_list else list(anchor_positions.keys())[0]
            return b.array(anchor_positions[first_anchor])

        # Build anchor matrix [n_anchors, d]
        anchor_arrays = [b.reshape(anchor_positions[a], (1, -1)) for a in anchor_list]
        anchor_matrix = b.concatenate(anchor_arrays, axis=0)
        anchor_arr = b.array(anchor_matrix)
        stress_vec = stress_vector if hasattr(stress_vector, "shape") else b.array(stress_vector)
        compute_dtype = precision_dtype(b, reference=anchor_arr)
        if hasattr(stress_vec, "dtype"):
            try:
                if b.finfo(stress_vec.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = stress_vec.dtype
            except Exception as exc:
                logger.debug("Failed to compare dtype eps for stress vector: %s", exc)
        anchor_arr = b.astype(anchor_arr, compute_dtype)
        stress_vec = b.astype(stress_vec, compute_dtype)
        b.eval(anchor_arr, stress_vec)

        d = int(b.shape(anchor_arr)[1])
        eps = division_epsilon(b, anchor_arr)

        # Compute anchor norms squared
        anchor_norms_sq = b.sum(anchor_arr ** 2, axis=1)
        b.eval(anchor_norms_sq)

        # Target distances from stress vector
        target_dists_sq = stress_vec ** 2
        b.eval(target_dists_sq)

        # Build linear system using a_0 as reference
        a_0 = b.take(anchor_arr, b.array([0]), axis=0)
        a_rest = anchor_arr[1:]
        A_mat = 2.0 * (b.broadcast_to(a_0, (n_anchors - 1, d)) - a_rest)

        d_0_sq = b.take(target_dists_sq, b.array([0]), axis=0)
        d_rest_sq = target_dists_sq[1:]
        norm_0_sq = b.take(anchor_norms_sq, b.array([0]), axis=0)
        norm_rest_sq = anchor_norms_sq[1:]

        b_vec = (
            d_rest_sq
            - b.broadcast_to(d_0_sq, (n_anchors - 1,))
            + b.broadcast_to(norm_0_sq, (n_anchors - 1,))
            - norm_rest_sq
        )
        b.eval(A_mat, b_vec)

        # Solve via least squares
        b_col = b.reshape(b_vec, (-1, 1))
        try:
            position = gpu_lstsq(b, A_mat, b_col)
            position = b.squeeze(position, axis=1)
        except Exception:
            # Fallback: weighted centroid
            stress_1d = b.reshape(stress_vec, (-1,))
            weights_arr = 1.0 / (stress_1d + eps)
            total_weight = b.sum(weights_arr)
            normalized_weights = weights_arr / total_weight
            position = b.sum(
                anchor_arr * b.reshape(normalized_weights, (-1, 1)),
                axis=0,
            )

        b.eval(position)
        return position

    def _compute_stress_from_position(
        self,
        position: "Array",
        anchor_positions: dict[str, "Array"],
    ) -> "Array":
        """Compute stress vector (geodesic distances to anchors) from position.

        Uses k-NN graph shortest paths to compute true manifold distances.

        Args:
            position: Position in target space [d].
            anchor_positions: Dict mapping anchor names to positions [d].

        Returns:
            Stress vector [n_anchors] of geodesic distances to each anchor.
        """
        b = self._backend

        anchor_list = sorted(anchor_positions.keys())
        n_anchors = len(anchor_list)

        if n_anchors == 0:
            return b.array([])

        # Stack position and all anchors into point cloud
        # position at index 0, anchors at indices 1..n_anchors
        position_2d = b.reshape(position, (1, -1))
        anchor_arrays = [b.reshape(b.array(anchor_positions[name]), (1, -1))
                        for name in anchor_list]
        points = b.concatenate([position_2d] + anchor_arrays, axis=0)
        b.eval(points)

        # Compute geodesic distances
        rg = RiemannianGeometry(b)
        geo_result = rg.geodesic_distances(points, use_cache=False)

        # Extract distances from position (row 0) to each anchor (rows 1..n)
        distances = geo_result.distances[0, 1:]
        b.eval(distances)

        return distances
