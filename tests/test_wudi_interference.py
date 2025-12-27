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

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.wudi_interference import (
    compute_wudi_interference,
    subspace_overlap,
)
from modelcypher.ports.backend import Backend


def test_wudi_overlap_identical(any_backend: Backend) -> None:
    b = any_backend
    a = b.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    overlap = subspace_overlap(a, a, b)
    tol = max(1e-5, machine_epsilon(b, a) * 100)
    assert abs(overlap - 1.0) <= tol


def test_wudi_overlap_orthogonal(any_backend: Backend) -> None:
    b = any_backend
    a = b.array([[1.0, 0.0], [0.0, 0.0]], dtype="float32")
    b_mat = b.array([[0.0, 1.0], [0.0, 0.0]], dtype="float32")
    overlap = subspace_overlap(a, b_mat, b)
    tol = max(1e-5, machine_epsilon(b, a) * 100)
    assert overlap <= tol


def test_wudi_loss_single_vector_zero(any_backend: Backend) -> None:
    b = any_backend
    tau = b.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    result = compute_wudi_interference({(2, 2): [tau]}, b)
    tol = max(1e-5, machine_epsilon(b, tau) * 100)
    assert abs(result.mean_loss) <= tol
