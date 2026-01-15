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

from modelcypher.core.domain.geometry.manifold_evidence import compute_manifold_evidence
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def test_manifold_evidence_support_diagnostics(any_backend) -> None:
    backend = any_backend
    points = backend.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.0],
            [0.5, 1.5],
            [1.5, 1.0],
            [2.5, 1.5],
            [3.0, 0.5],
            [3.5, 1.0],
        ]
    )

    report = compute_manifold_evidence(points, backend=backend)
    support = report.support_diagnostics

    expected_renyi_support = report.effective_rank.renyi_effective_rank / report.feature_dim
    expected_shannon_support = (
        report.effective_rank.shannon_effective_rank / report.feature_dim
    )
    eps = _eps(
        backend,
        support.renyi_support_ratio,
        support.shannon_support_ratio,
        report.intrinsic_dimension or 0.0,
    )

    assert abs(support.renyi_support_ratio - expected_renyi_support) <= eps
    assert abs(support.shannon_support_ratio - expected_shannon_support) <= eps
    assert abs(support.renyi_null_ratio - (1.0 - expected_renyi_support)) <= eps
    assert abs(support.shannon_null_ratio - (1.0 - expected_shannon_support)) <= eps

    assert report.intrinsic_dimension is not None
    assert support.renyi_id_gap is not None
    assert support.shannon_id_gap is not None
    assert (
        abs(
            support.renyi_id_gap
            - (report.effective_rank.renyi_effective_rank - report.intrinsic_dimension)
        )
        <= eps
    )
    assert (
        abs(
            support.shannon_id_gap
            - (
                report.effective_rank.shannon_effective_rank
                - report.intrinsic_dimension
            )
        )
        <= eps
    )


def test_manifold_evidence_id_gap_missing(any_backend) -> None:
    backend = any_backend
    points = backend.array([[0.0, 0.0], [1.0, 1.0]])

    report = compute_manifold_evidence(points, backend=backend)
    support = report.support_diagnostics

    assert report.intrinsic_dimension is None
    assert support.renyi_id_gap is None
    assert support.shannon_id_gap is None
