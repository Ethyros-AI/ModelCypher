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

"""Behavioral norm utility for weight deltas.

Behavioral norm measures the actual OUTPUT change induced by a weight delta:
    ||A @ ΔW.T||_F

This is the correct preservation metric for transplant because it reflects
behavioral impact rather than weight magnitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def behavioral_norm(
    delta_W: "Array",
    input_activations: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute behavioral impact norm for a weight delta.

    Args:
        delta_W: Weight delta [out_dim, in_dim]
        input_activations: Activations that flow through this weight [n, in_dim]
        backend: Backend for tensor operations

    Returns:
        Behavioral impact norm (scalar)
    """
    b = backend or get_default_backend()
    delta_W = b.array(delta_W)
    input_activations = b.array(input_activations)
    if len(b.shape(delta_W)) == 1:
        delta_W = b.reshape(delta_W, (1, int(b.shape(delta_W)[0])))

    # Compute output change: [n, in_dim] @ [in_dim, out_dim] -> [n, out_dim]
    output_change = b.matmul(input_activations, b.transpose(delta_W))
    norm = b.sqrt(b.sum(output_change * output_change))
    b.eval(norm)
    return float(b.to_scalar(norm))
