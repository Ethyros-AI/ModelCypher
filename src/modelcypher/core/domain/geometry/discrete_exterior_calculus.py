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

"""Discrete Exterior Calculus (DEC) placeholder.

The previous implementation depended on NumPy/SciPy, which violates the
backend-only rule for domain code (GPU/accelerator safety). A backend-native
DEC implementation should live here. Until then, this module is a stub to
avoid accidental CPU fallbacks in core geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SimplicialComplex:
    """Simplicial complex placeholder (backend-native implementation pending)."""

    vertices: Any
    edges: Any
    triangles: Any
    edge_weights: Any


@dataclass
class HodgeDecomposition:
    """Hodge decomposition placeholder (backend-native implementation pending)."""

    gradient_component: Any
    curl_component: Any
    harmonic_component: Any
    gradient_norm: float
    curl_norm: float
    harmonic_norm: float


@dataclass
class DECGeodesicResult:
    """DEC geodesic result placeholder (backend-native implementation pending)."""

    distances: Any
    laplacian_eigenvalues: Any
    spectral_gap: float
    is_positive_semidefinite: bool
    heat_time: float
    mean_edge_length: float


class DiscreteExteriorCalculus:
    """Backend-native DEC is not yet implemented."""

    def __init__(self, sqrt_eps: Optional[float] = None) -> None:
        raise RuntimeError(
            "DiscreteExteriorCalculus requires a backend-native implementation. "
            "NumPy/SciPy are not permitted in the domain layer."
        )


__all__ = [
    "DECGeodesicResult",
    "DiscreteExteriorCalculus",
    "HodgeDecomposition",
    "SimplicialComplex",
]
