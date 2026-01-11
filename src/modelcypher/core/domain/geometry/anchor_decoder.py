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
Anchor-Relative Decoder for Target Activation Space.

Decodes delta from anchor-relative space back to target activation space
using target's pseudo-inverse. This ensures source coordinates never
directly touch target weights.

Mathematical foundation:
    S_t = cos(A_t, anchors)           # Target relative representation [n, n_anchors]
    B = pinv(S_t) @ A_t               # Decoder [n_anchors, d_target]
    delta_A = (w * delta_S) @ B       # Weighted delta in target activation space

The decoder B maps from anchor space to target activation space using only
target's own coordinate system, preserving the invariant geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import geodesic_pinv
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

__all__ = [
    "AnchorDecodingResult",
    "compute_anchor_decoder",
    "decode_to_activation_space",
]


@dataclass(frozen=True)
class AnchorDecodingResult:
    """Result of anchor-relative decoding.

    Attributes:
        delta_activations: Delta in target activation space [n, d_target]
        decoder_matrix: Decoder B = pinv(S_t) @ A_t [n_anchors, d_target]
        density_weights: Per-sample weights used [n]
        reconstruction_error: Normalized error ||S_t @ B - A_t|| / ||A_t||
    """

    delta_activations: "Array"
    decoder_matrix: "Array"
    density_weights: "Array"
    reconstruction_error: float


def compute_anchor_decoder(
    target_relative_rep: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", float]:
    """Compute decoder from anchor space to target activation space.

    Computes B = pinv(S_t) @ A_t which maps anchor-relative coordinates
    back to target activation space using target's pseudo-inverse.

    Args:
        target_relative_rep: Target relative representation S_t [n, n_anchors]
        target_activations: Target activations A_t [n, d_target]
        backend: Compute backend (defaults to system default)

    Returns:
        Tuple of:
            - Decoder matrix B [n_anchors, d_target]
            - Reconstruction error ||S_t @ B - A_t|| / ||A_t||
    """
    b = backend or get_default_backend()

    # Compute pseudo-inverse of S_t
    # S_t is [n, n_anchors], pinv(S_t) is [n_anchors, n]
    S_t_pinv = geodesic_pinv(target_relative_rep, b)
    b.eval(S_t_pinv)

    # Decoder B = pinv(S_t) @ A_t
    # [n_anchors, n] @ [n, d_target] -> [n_anchors, d_target]
    decoder = b.matmul(S_t_pinv, target_activations)
    b.eval(decoder)

    # Compute reconstruction error
    # Reconstructed = S_t @ B, compare to A_t
    reconstructed = b.matmul(target_relative_rep, decoder)
    diff = reconstructed - target_activations
    b.eval(diff)

    # Compute normalized error
    diff_norm_arr = geodesic_norms(b.reshape(diff, (1, -1)), b, use_cache=False)
    target_norm_arr = geodesic_norms(
        b.reshape(target_activations, (1, -1)), b, use_cache=False
    )
    b.eval(diff_norm_arr, target_norm_arr)

    diff_norm = float(b.to_scalar(diff_norm_arr[0]))
    target_norm = float(b.to_scalar(target_norm_arr[0]))

    if target_norm > 0:
        reconstruction_error = diff_norm / target_norm
    else:
        reconstruction_error = 0.0

    logger.debug(
        "ANCHOR DECODER: shape=[%d, %d], reconstruction_error=%.4f",
        b.shape(decoder)[0],
        b.shape(decoder)[1],
        reconstruction_error,
    )

    return decoder, reconstruction_error


def decode_to_activation_space(
    delta_relative: "Array",
    decoder: "Array",
    density_weights: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Decode weighted delta from anchor space to target activation space.

    Computes: delta_A = (w * delta_S) @ B

    The density weights w control how much of the source delta is transferred
    at each sample point. Weights near 1.0 mean strong transfer (source denser),
    weights near 0.0 mean weak/no transfer (target denser).

    Args:
        delta_relative: Delta in anchor space [n, n_anchors]
        decoder: Decoder matrix B [n_anchors, d_target]
        density_weights: Per-sample weights [n] in [0, 1]
        backend: Compute backend

    Returns:
        Delta in target activation space [n, d_target]
    """
    b = backend or get_default_backend()

    # Reshape weights for broadcasting: [n] -> [n, 1]
    n_samples = b.shape(delta_relative)[0]
    weights_2d = b.reshape(density_weights, (n_samples, 1))

    # Apply density weighting: w * delta_S
    weighted_delta = delta_relative * weights_2d
    b.eval(weighted_delta)

    # Decode to activation space: (w * delta_S) @ B
    # [n, n_anchors] @ [n_anchors, d_target] -> [n, d_target]
    delta_activations = b.matmul(weighted_delta, decoder)
    b.eval(delta_activations)

    # Log transfer statistics
    mean_weight = float(b.to_scalar(b.mean(density_weights)))
    logger.debug(
        "DECODE TO ACTIVATION: n_samples=%d, mean_weight=%.3f",
        n_samples,
        mean_weight,
    )

    return delta_activations
