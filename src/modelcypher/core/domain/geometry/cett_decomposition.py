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

"""CETT (Contribution of Each Token per neuron) decomposition.

Experimental — not validated for gating. Research diagnostic only.

Implements per-neuron contribution analysis from:
    Gao et al., "H-Neurons: On the Existence, Impact, and Origin of
    Hallucination-Associated Neurons in LLMs", arXiv:2512.01797 (2025).

CETT decomposes the FFN output into per-neuron contributions:

    CETT_{j,t} = ||h_t^(j)||_2 / ||h_t||_2

where h_t^(j) = W_down[:, j] * z_{j,t} is the contribution of neuron j
to the hidden state at token position t.

Since W_down[:, j] is a fixed column vector, this simplifies to:

    CETT_{j,t} = |z_{j,t}| * ||W_down[:, j]||_2 / ||h_t||_2

No per-neuron matmuls needed — just scalar activations times pre-computed
column norms of W_down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

logger = logging.getLogger(__name__)

Array = Any


@dataclass(frozen=True)
class CETTResult:
    """Per-neuron CETT scores for one layer."""

    layer: int
    """Layer index."""

    mean_cett: Array
    """Mean CETT per neuron across all token positions. Shape: [intermediate_dim]."""

    max_cett: Array
    """Max CETT per neuron across all token positions. Shape: [intermediate_dim]."""

    n_tokens: int
    """Number of token positions used."""


def compute_down_proj_column_norms(
    model: Any,
    backend: Any,
) -> dict[int, Array]:
    """Pre-compute ||W_down[:, j]||_2 for each layer's down_proj.

    Args:
        model: Loaded model object.
        backend: Backend instance.

    Returns:
        Dict mapping layer_index to column norms [intermediate_dim].
    """
    base = getattr(model, "model", model)
    col_norms: dict[int, Array] = {}

    for layer_idx, layer in enumerate(base.layers):
        ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if ff_module is None:
            continue

        # Get down_proj weight: [hidden_dim, intermediate_dim]
        if hasattr(ff_module, "down_proj"):
            w_down = ff_module.down_proj.weight
        elif hasattr(ff_module, "w2"):
            w_down = ff_module.w2.weight
        elif hasattr(ff_module, "fc2"):
            w_down = ff_module.fc2.weight
        else:
            continue

        # Column norms: ||W_down[:, j]||_2 for each j
        # w_down shape: [hidden_dim, intermediate_dim]
        norms = backend.norm(w_down, axis=0)  # [intermediate_dim]
        backend.eval(norms)
        col_norms[layer_idx] = norms

    return col_norms


def compute_cett_per_layer(
    intermediate: Array,
    hidden_state: Array,
    down_proj_col_norms: Array,
    backend: Any,
    layer_idx: int,
) -> CETTResult:
    """Compute per-neuron CETT for a single layer.

    Args:
        intermediate: FFN intermediate activations [n_tokens, intermediate_dim].
            This is z_t = SiLU(gate) * up, before down_proj.
        hidden_state: Post-layer hidden states [n_tokens, hidden_dim].
        down_proj_col_norms: Pre-computed ||W_down[:, j]||_2 [intermediate_dim].
        backend: Backend instance.
        layer_idx: Layer index (for metadata).

    Returns:
        CETTResult with per-neuron mean and max CETT scores.
    """
    # ||h_t||_2 for each token position: [n_tokens]
    h_norms = backend.norm(hidden_state, axis=1)  # [n_tokens]

    # Guard against division by zero
    eps = division_epsilon(backend, h_norms)
    h_norms_safe = h_norms + eps

    # |z_{j,t}| * ||W_down[:, j]||_2 / ||h_t||_2
    # intermediate: [n_tokens, intermediate_dim]
    # down_proj_col_norms: [intermediate_dim]
    # h_norms_safe: [n_tokens]
    abs_z = backend.abs(intermediate)  # [n_tokens, intermediate_dim]
    numerator = abs_z * backend.reshape(down_proj_col_norms, (1, -1))  # broadcast
    cett = numerator / backend.reshape(h_norms_safe, (-1, 1))  # [n_tokens, intermediate_dim]

    # Aggregate over token positions
    mean_cett = backend.mean(cett, axis=0)  # [intermediate_dim]
    max_cett = backend.max(cett, axis=0)  # [intermediate_dim]
    backend.eval(mean_cett)
    backend.eval(max_cett)

    n_tokens = int(intermediate.shape[0])

    return CETTResult(
        layer=layer_idx,
        mean_cett=mean_cett,
        max_cett=max_cett,
        n_tokens=n_tokens,
    )
