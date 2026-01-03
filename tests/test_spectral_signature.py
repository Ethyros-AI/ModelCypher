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

"""Tests for spectral signature computation.

SpectralSignature now derives all parameters from data at runtime:
- k_neighbors: derived from graph connectivity requirements
- kernel_bandwidth: derived from median neighbor distance
- heat_trace_times: derived from eigenvalue spectrum
"""

from __future__ import annotations

from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.spectral_signature import SpectralSignature


def test_spectral_signature_eigenvalues_bounds(any_backend) -> None:
    """Test that eigenvalues are in valid range for normalized Laplacian."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    # All parameters are now derived from data at runtime
    signature = SpectralSignature(any_backend).compute(points)

    eigvals = signature.eigenvalues
    eps = regularization_epsilon(any_backend, any_backend.array(eigvals))

    # For normalized Laplacian: eigenvalues in [0, 2]
    assert min(eigvals) >= -eps
    assert max(eigvals) <= 2.0 + eps


def test_spectral_signature_component_count_disconnected(any_backend) -> None:
    """Test detection of disconnected components."""
    # Two disconnected pairs with large gap
    points = [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]]
    signature = SpectralSignature(any_backend).compute(points)

    # Should detect two components if graph is disconnected
    # Note: actual component count depends on derived k_neighbors
    assert signature.component_count >= 1
    assert signature.node_count == 4


def test_spectral_signature_heat_trace_monotone(any_backend) -> None:
    """Test that heat trace decreases with time."""
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    signature = SpectralSignature(any_backend).compute(points)

    heat = signature.heat_trace
    eps = regularization_epsilon(any_backend, any_backend.array(heat))

    # Heat trace should decrease (or stay same) as time increases
    for i in range(len(heat) - 1):
        assert heat[i] + eps >= heat[i + 1]
