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

"""Submultiplicative upper bound on transformer layer Jacobian spectral norm.

Pure Python — zero framework dependencies.

For a standard transformer layer with residual connections::

    J_l = I + dAttn/dh + dMLP/dh

The submultiplicative bound gives::

    ||J_l|| <= 1 + sigma(O) * sigma(V) + sigma(down) * (sigma(gate) + sigma(up))

where sigma(W) is the spectral norm (largest singular value) of weight matrix W.

Extracted from rl_spectral_anatomy_experiment.py.
"""

from __future__ import annotations


def submultiplicative_jacobian_bound(spectral_norms: dict[str, float]) -> float:
    """Compute submultiplicative upper bound on layer Jacobian spectral norm.

    Parameters
    ----------
    spectral_norms : dict
        Must contain keys: ``"o"``, ``"v"``, ``"down"``, ``"gate"``, ``"up"``.
        Values are spectral norms (largest singular values) of the
        corresponding weight matrices.

    Returns
    -------
    float
        Upper bound on ||J_l||.

    Raises
    ------
    KeyError
        If a required key is missing.
    """
    return (
        1.0
        + spectral_norms["o"] * spectral_norms["v"]
        + spectral_norms["down"] * (spectral_norms["gate"] + spectral_norms["up"])
    )
