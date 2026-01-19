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

"""Hypothesis property tests for geometry metrics service payloads."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.geometry_metrics_service import (
    DimensionConstraintInvarianceResult,
    EffectiveRankResult,
    GeometryMetricsService,
    GromovWassersteinResult,
    IntrinsicDimensionResult,
    SpectralSignatureResult,
    TopologicalFingerprintResult,
)


_finite_float = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32)
_pos_int = st.integers(min_value=0, max_value=128)
_bool = st.booleans()


@settings(max_examples=10, deadline=None)
@given(
    distance=_finite_float,
    normalized=_finite_float,
    aligned=_bool,
    converged=_bool,
    iterations=_pos_int,
    n=_pos_int,
    m=_pos_int,
)
def test_gromov_wasserstein_payload_identity(
    distance: float,
    normalized: float,
    aligned: bool,
    converged: bool,
    iterations: int,
    n: int,
    m: int,
) -> None:
    result = GromovWassersteinResult(
        distance=distance,
        normalized_distance=normalized,
        aligned=aligned,
        converged=converged,
        iterations=iterations,
        coupling_shape=(n, m),
    )
    payload = GeometryMetricsService.gromov_wasserstein_payload(result)
    assert payload["distance"] == distance
    assert payload["normalizedDistance"] == normalized
    assert payload["aligned"] == aligned
    assert payload["converged"] == converged
    assert payload["iterations"] == iterations
    assert payload["couplingShape"] == [n, m]


@settings(max_examples=10, deadline=None)
@given(
    dimension=_finite_float,
    lower=_finite_float,
    upper=_finite_float,
    samples=_pos_int,
    method=st.text(min_size=1, max_size=12),
)
def test_intrinsic_dimension_payload_identity(
    dimension: float,
    lower: float,
    upper: float,
    samples: int,
    method: str,
) -> None:
    result = IntrinsicDimensionResult(
        dimension=dimension,
        confidence_lower=lower,
        confidence_upper=upper,
        sample_count=samples,
        method=method,
    )
    payload = GeometryMetricsService.intrinsic_dimension_payload(result)
    assert payload["intrinsicDimension"] == dimension
    assert payload["confidenceLower"] == lower
    assert payload["confidenceUpper"] == upper
    assert payload["sampleCount"] == samples
    assert payload["method"] == method


@settings(max_examples=10, deadline=None)
@given(
    renyi=_finite_float,
    shannon=_finite_float,
    entropy=_finite_float,
    samples=_pos_int,
    feature_dim=_pos_int,
    n_sv=_pos_int,
)
def test_effective_rank_payload_identity(
    renyi: float,
    shannon: float,
    entropy: float,
    samples: int,
    feature_dim: int,
    n_sv: int,
) -> None:
    result = EffectiveRankResult(
        renyi_effective_rank=renyi,
        shannon_effective_rank=shannon,
        spectral_entropy=entropy,
        sample_count=samples,
        feature_dim=feature_dim,
        n_singular_values=n_sv,
    )
    payload = GeometryMetricsService.effective_rank_payload(result)
    assert payload["renyiEffectiveRank"] == renyi
    assert payload["shannonEffectiveRank"] == shannon
    assert payload["spectralEntropy"] == entropy
    assert payload["sampleCount"] == samples
    assert payload["featureDim"] == feature_dim
    assert payload["singularValueCount"] == n_sv


@settings(max_examples=10, deadline=None)
@given(
    betti_0=st.integers(min_value=0, max_value=10),
    betti_1=st.integers(min_value=0, max_value=10),
    persistence=_finite_float,
    total=_finite_float,
)
def test_topological_fingerprint_payload_identity(
    betti_0: int,
    betti_1: int,
    persistence: float,
    total: float,
) -> None:
    result = TopologicalFingerprintResult(
        betti_0=betti_0,
        betti_1=betti_1,
        persistence_entropy=persistence,
        total_persistence=total,
    )
    payload = GeometryMetricsService.topological_fingerprint_payload(result)
    assert payload["betti0"] == betti_0
    assert payload["betti1"] == betti_1
    assert payload["persistenceEntropy"] == persistence
    assert payload["totalPersistence"] == total


@settings(max_examples=10, deadline=None)
@given(
    eigenvalues=st.lists(_finite_float, min_size=0, max_size=6),
    heat_trace=st.lists(_finite_float, min_size=0, max_size=6),
    heat_times=st.lists(_finite_float, min_size=0, max_size=6),
    entropy=_finite_float,
    alg_conn=_finite_float,
    component_count=_pos_int,
    node_count=_pos_int,
    edge_count=_pos_int,
    k_neighbors=_pos_int,
    bandwidth=_finite_float,
    normalized=_bool,
    connected=_bool,
)
def test_spectral_signature_payload_identity(
    eigenvalues: list[float],
    heat_trace: list[float],
    heat_times: list[float],
    entropy: float,
    alg_conn: float,
    component_count: int,
    node_count: int,
    edge_count: int,
    k_neighbors: int,
    bandwidth: float,
    normalized: bool,
    connected: bool,
) -> None:
    result = SpectralSignatureResult(
        eigenvalues=eigenvalues,
        heat_trace=heat_trace,
        heat_times=heat_times,
        spectral_entropy=entropy,
        algebraic_connectivity=alg_conn,
        component_count=component_count,
        node_count=node_count,
        edge_count=edge_count,
        k_neighbors=k_neighbors,
        kernel_bandwidth=bandwidth,
        normalized_laplacian=normalized,
        connected=connected,
    )
    payload = GeometryMetricsService.spectral_signature_payload(result)
    assert payload["eigenvalues"] == eigenvalues
    assert payload["eigenvalueCount"] == len(eigenvalues)
    assert payload["heatTrace"] == heat_trace
    assert payload["heatTimes"] == heat_times
    assert payload["spectralEntropy"] == entropy
    assert payload["algebraicConnectivity"] == alg_conn
    assert payload["componentCount"] == component_count
    assert payload["nodeCount"] == node_count
    assert payload["edgeCount"] == edge_count
    assert payload["kNeighbors"] == k_neighbors
    assert payload["kernelBandwidth"] == bandwidth
    assert payload["normalizedLaplacian"] == normalized
    assert payload["connected"] == connected


@settings(max_examples=5, deadline=None)
@given(
    base_dim=_pos_int,
    padded_dim=_pos_int,
    sample_count=_pos_int,
    k_neighbors=st.one_of(st.none(), _pos_int),
    gram_cka=_finite_float,
    geo_mean=_finite_float,
    geo_max=_finite_float,
    spec_mean=_finite_float,
    spec_max=_finite_float,
    spec_base=_finite_float,
    spec_padded=_finite_float,
    heat_trace_base=st.lists(_finite_float, min_size=0, max_size=4),
    heat_trace_padded=st.lists(_finite_float, min_size=0, max_size=4),
    heat_times=st.lists(_finite_float, min_size=0, max_size=4),
    betti_base=st.dictionaries(st.integers(min_value=0, max_value=3), _pos_int, max_size=3),
    betti_padded=st.dictionaries(st.integers(min_value=0, max_value=3), _pos_int, max_size=3),
    comp_base=_pos_int,
    comp_padded=_pos_int,
    cycle_base=_pos_int,
    cycle_padded=_pos_int,
    pe_base=_finite_float,
    pe_padded=_finite_float,
    max_p_base=_finite_float,
    max_p_padded=_finite_float,
)
def test_dimension_constraint_payload_identity(
    base_dim: int,
    padded_dim: int,
    sample_count: int,
    k_neighbors: int | None,
    gram_cka: float,
    geo_mean: float,
    geo_max: float,
    spec_mean: float,
    spec_max: float,
    spec_base: float,
    spec_padded: float,
    heat_trace_base: list[float],
    heat_trace_padded: list[float],
    heat_times: list[float],
    betti_base: dict[int, int],
    betti_padded: dict[int, int],
    comp_base: int,
    comp_padded: int,
    cycle_base: int,
    cycle_padded: int,
    pe_base: float,
    pe_padded: float,
    max_p_base: float,
    max_p_padded: float,
) -> None:
    result = DimensionConstraintInvarianceResult(
        base_dimension=base_dim,
        padded_dimension=padded_dim,
        sample_count=sample_count,
        k_neighbors=k_neighbors,
        gram_cka=gram_cka,
        geodesic_mean_abs_diff=geo_mean,
        geodesic_max_abs_diff=geo_max,
        spectral_eigen_mean_abs_diff=spec_mean,
        spectral_eigen_max_abs_diff=spec_max,
        spectral_entropy_base=spec_base,
        spectral_entropy_padded=spec_padded,
        heat_trace_base=heat_trace_base,
        heat_trace_padded=heat_trace_padded,
        heat_times=heat_times,
        betti_numbers_base=betti_base,
        betti_numbers_padded=betti_padded,
        component_count_base=comp_base,
        component_count_padded=comp_padded,
        cycle_count_base=cycle_base,
        cycle_count_padded=cycle_padded,
        persistence_entropy_base=pe_base,
        persistence_entropy_padded=pe_padded,
        max_persistence_base=max_p_base,
        max_persistence_padded=max_p_padded,
    )
    payload = GeometryMetricsService.dimension_constraint_invariance_payload(result)
    assert payload["baseDimension"] == base_dim
    assert payload["paddedDimension"] == padded_dim
    assert payload["sampleCount"] == sample_count
    assert payload["kNeighbors"] == k_neighbors
    assert payload["gramCka"] == gram_cka
    assert payload["geodesicDiff"]["meanAbs"] == geo_mean
    assert payload["geodesicDiff"]["maxAbs"] == geo_max
    assert payload["spectral"]["eigenMeanAbsDiff"] == spec_mean
    assert payload["spectral"]["eigenMaxAbsDiff"] == spec_max
    assert payload["spectral"]["spectralEntropyBase"] == spec_base
    assert payload["spectral"]["spectralEntropyPadded"] == spec_padded
    assert payload["spectral"]["heatTraceBase"] == heat_trace_base
    assert payload["spectral"]["heatTracePadded"] == heat_trace_padded
    assert payload["spectral"]["heatTimes"] == heat_times
    assert payload["topology"]["bettiNumbersBase"] == betti_base
    assert payload["topology"]["bettiNumbersPadded"] == betti_padded
    assert payload["topology"]["componentCountBase"] == comp_base
    assert payload["topology"]["componentCountPadded"] == comp_padded
    assert payload["topology"]["cycleCountBase"] == cycle_base
    assert payload["topology"]["cycleCountPadded"] == cycle_padded
    assert payload["topology"]["persistenceEntropyBase"] == pe_base
    assert payload["topology"]["persistenceEntropyPadded"] == pe_padded
    assert payload["topology"]["maxPersistenceBase"] == max_p_base
    assert payload["topology"]["maxPersistencePadded"] == max_p_padded
