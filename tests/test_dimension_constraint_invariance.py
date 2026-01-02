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

"""Tests that constrained dimensions preserve manifold geometry."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.spectral_signature import (
    SpectralSignature,
    SpectralSignatureConfig,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _base_points() -> list[list[float]]:
    return [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [0.1, 0.7],
        [0.9, 0.2],
    ]


def _pad_points(points: list[list[float]], dims: int) -> list[list[float]]:
    return [row + [0.0] * (dims - len(row)) for row in points]


def test_gram_matrix_invariance_under_padding(any_backend) -> None:
    """Gram geometry is identical under zero-padding."""
    backend = any_backend
    points = _base_points()
    padded = _pad_points(points, 4)

    base = backend.array(points)
    expanded = backend.array(padded)
    backend.eval(base, expanded)

    gram_base = backend.matmul(base, backend.transpose(base))
    gram_expanded = backend.matmul(expanded, backend.transpose(expanded))
    backend.eval(gram_base, gram_expanded)

    cka = compute_cka_from_grams(gram_base, gram_expanded, backend=backend)
    eps = _eps(backend, float(cka), 1.0)
    assert abs(float(cka) - 1.0) <= eps


def test_geodesic_and_spectral_invariance_under_padding(any_backend) -> None:
    """Geodesic and spectral signatures are invariant under zero-padding."""
    backend = any_backend
    points = _base_points()
    padded = _pad_points(points, 3)
    k_neighbors = len(points) - 1

    geometry = RiemannianGeometry(backend)
    geo_base = geometry.geodesic_distances(points, k_neighbors=k_neighbors)
    geo_padded = geometry.geodesic_distances(padded, k_neighbors=k_neighbors)

    geo_diff = backend.abs(geo_base.distances - geo_padded.distances)
    geo_max = backend.max(geo_diff)
    backend.eval(geo_max)
    eps = _eps(backend, float(backend.to_numpy(geo_max).item()))
    assert float(backend.to_numpy(geo_max).item()) <= eps
    assert geo_base.connected == geo_padded.connected
    assert geo_base.k_neighbors == geo_padded.k_neighbors

    config = SpectralSignatureConfig(k_neighbors=k_neighbors)
    spectral = SpectralSignature(backend)
    sig_base = spectral.compute(points, config)
    sig_padded = spectral.compute(padded, config)

    eps = _eps(backend, sig_base.spectral_entropy, sig_padded.spectral_entropy)
    assert sig_base.eigenvalues == pytest.approx(sig_padded.eigenvalues, abs=eps)
    assert sig_base.heat_trace == pytest.approx(sig_padded.heat_trace, abs=eps)
    assert abs(sig_base.spectral_entropy - sig_padded.spectral_entropy) <= eps
    assert abs(sig_base.algebraic_connectivity - sig_padded.algebraic_connectivity) <= eps
    assert sig_base.component_count == sig_padded.component_count
    assert sig_base.edge_count == sig_padded.edge_count
    assert sig_base.k_neighbors == sig_padded.k_neighbors


def test_topological_fingerprint_invariance_under_padding() -> None:
    """Persistent homology summaries are invariant under zero-padding."""
    points = _base_points()
    padded = _pad_points(points, 4)

    fp_base = TopologicalFingerprint.compute(points, max_dimension=1)
    fp_padded = TopologicalFingerprint.compute(padded, max_dimension=1)

    assert fp_base.betti_numbers == fp_padded.betti_numbers
    assert fp_base.summary.component_count == fp_padded.summary.component_count
    assert fp_base.summary.cycle_count == fp_padded.summary.cycle_count
    backend = get_default_backend()
    eps = _eps(
        backend,
        fp_base.summary.average_persistence,
        fp_padded.summary.average_persistence,
        fp_base.summary.max_persistence,
        fp_padded.summary.max_persistence,
        fp_base.summary.persistence_entropy,
        fp_padded.summary.persistence_entropy,
    )
    assert abs(fp_base.summary.average_persistence - fp_padded.summary.average_persistence) <= eps
    assert abs(fp_base.summary.max_persistence - fp_padded.summary.max_persistence) <= eps
    assert abs(fp_base.summary.persistence_entropy - fp_padded.summary.persistence_entropy) <= eps


@given(
    sample_count=st.integers(min_value=4, max_value=8),
    base_dim=st.integers(min_value=2, max_value=4),
    pad_extra=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_padding_invariance_random_pointcloud(
    any_backend,
    sample_count: int,
    base_dim: int,
    pad_extra: int,
    seed: int,
) -> None:
    """Random point clouds remain invariant under zero-padding."""
    backend = any_backend
    backend.random_seed(seed)

    points_arr = backend.random_normal((sample_count, base_dim))
    backend.eval(points_arr)
    points = backend.to_numpy(points_arr).tolist()
    padded = _pad_points(points, base_dim + pad_extra)
    k_neighbors = sample_count - 1

    base = backend.array(points)
    expanded = backend.array(padded)
    backend.eval(base, expanded)

    gram_base = backend.matmul(base, backend.transpose(base))
    gram_expanded = backend.matmul(expanded, backend.transpose(expanded))
    backend.eval(gram_base, gram_expanded)

    cka = compute_cka_from_grams(gram_base, gram_expanded, backend=backend)
    eps = _eps(backend, float(cka), 1.0)
    assert abs(float(cka) - 1.0) <= eps

    geometry = RiemannianGeometry(backend)
    geo_base = geometry.geodesic_distances(points, k_neighbors=k_neighbors)
    geo_padded = geometry.geodesic_distances(padded, k_neighbors=k_neighbors)

    geo_diff = backend.abs(geo_base.distances - geo_padded.distances)
    geo_max = backend.max(geo_diff)
    backend.eval(geo_max)
    eps = _eps(backend, float(backend.to_numpy(geo_max).item()))
    assert float(backend.to_numpy(geo_max).item()) <= eps

    config = SpectralSignatureConfig(k_neighbors=k_neighbors)
    spectral = SpectralSignature(backend)
    sig_base = spectral.compute(points, config)
    sig_padded = spectral.compute(padded, config)

    eps = _eps(backend, sig_base.spectral_entropy, sig_padded.spectral_entropy)
    assert sig_base.eigenvalues == pytest.approx(sig_padded.eigenvalues, abs=eps)
    assert sig_base.heat_trace == pytest.approx(sig_padded.heat_trace, abs=eps)
