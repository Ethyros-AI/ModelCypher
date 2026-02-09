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

"""Law-of-cosines decomposition of layer-to-layer velocity.

Pure Python — zero framework dependencies.

Decomposes the relative velocity ||h_{l+1} - h_l|| / ||h_l|| into:
- **radial**: magnitude change along the current direction
- **angular**: rotation perpendicular to the current direction

Uses the law of cosines on the triangle (h_l, h_{l+1}, h_{l+1} - h_l).

Extracted from rl_spectral_anatomy_experiment.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityDecompositionResult:
    """Result of decomposing velocity into radial and angular components."""

    radial_trajectory: list[float]
    angular_trajectory: list[float]
    cos_theta_trajectory: list[float]


def decompose_velocity(
    norm_trajectory: list[float],
    velocity_trajectory: list[float],
) -> VelocityDecompositionResult:
    """Decompose velocity into radial and angular components via law of cosines.

    Parameters
    ----------
    norm_trajectory : list[float]
        Per-layer mean activation norms, length N+1 (embedding + N layers).
    velocity_trajectory : list[float]
        Per-layer relative velocity ||h_{l+1} - h_l|| / ||h_l||, length N.

    Returns
    -------
    VelocityDecompositionResult
        Radial, angular, and cos_theta trajectories, each length N.
    """
    radial = []
    angular = []
    cos_theta = []

    for layer in range(len(velocity_trajectory)):
        vel = velocity_trajectory[layer]
        n_l = norm_trajectory[layer]
        n_l1 = norm_trajectory[layer + 1]

        if n_l < 1e-15 or n_l1 < 1e-15:
            radial.append(0.0)
            angular.append(0.0)
            cos_theta.append(1.0)
            continue

        step = vel * n_l

        # Law of cosines: h_l · h_{l+1} = (||h_{l+1}||² + ||h_l||² - step²) / 2
        dot = (n_l1**2 + n_l**2 - step**2) / 2.0

        ct = max(-1.0, min(1.0, dot / (n_l * n_l1)))
        cos_theta.append(ct)

        # Radial: magnitude change along h_l direction
        rad = ct * n_l1 / n_l - 1.0
        radial.append(rad)

        # Angular: perpendicular change
        ang_sq = max(0.0, vel**2 - rad**2)
        angular.append(math.sqrt(ang_sq))

    return VelocityDecompositionResult(
        radial_trajectory=radial,
        angular_trajectory=angular,
        cos_theta_trajectory=cos_theta,
    )
