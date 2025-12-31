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

from __future__ import annotations

from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.spectral_signature import (
    SpectralSignature,
    SpectralSignatureConfig,
)


def test_spectral_signature_eigenvalues_bounds(any_backend) -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    config = SpectralSignatureConfig(k_neighbors=1, normalized_laplacian=True)
    signature = SpectralSignature(any_backend).compute(points, config)

    eigvals = signature.eigenvalues
    eps = regularization_epsilon(any_backend, any_backend.array(eigvals))

    assert min(eigvals) >= -eps
    assert max(eigvals) <= 2.0 + eps


def test_spectral_signature_component_count_disconnected(any_backend) -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]
    config = SpectralSignatureConfig(k_neighbors=1)
    signature = SpectralSignature(any_backend).compute(points, config)

    assert signature.component_count == 2
    assert signature.connected is False


def test_spectral_signature_heat_trace_monotone(any_backend) -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    config = SpectralSignatureConfig(k_neighbors=1, heat_trace_times=(0.1, 1.0, 10.0))
    signature = SpectralSignature(any_backend).compute(points, config)

    heat = signature.heat_trace
    eps = regularization_epsilon(any_backend, any_backend.array(heat))

    for i in range(len(heat) - 1):
        assert heat[i] + eps >= heat[i + 1]
