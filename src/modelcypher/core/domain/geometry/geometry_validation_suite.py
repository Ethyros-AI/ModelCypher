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

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams

_cache = ComputationCache.shared()
from modelcypher.core.domain.geometry.gromov_wasserstein import GromovWassersteinDistance
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.path_geometry import PathGeometry, PathNode, PathSignature
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
)
from modelcypher.core.domain.geometry.spectral_signature import SpectralSignature
from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint
from modelcypher.core.domain.geometry.traversal_coherence import Path as TraversalPath
from modelcypher.core.domain.geometry.traversal_coherence import TraversalCoherence

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

SUITE_VERSION = "1.0"


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All algorithm parameters are derived from data (machine epsilon, dtype).
# There is exactly ONE correct way to validate geometry.
# =============================================================================


@dataclass(frozen=True)
class GromovWassersteinValidation:
    distance_identity: float
    distance_permutation: float
    symmetry_delta: float
    max_row_mass_error: float
    max_column_mass_error: float
    converged: bool
    iterations: int
    passed: bool


@dataclass(frozen=True)
class TraversalCoherenceValidation:
    self_correlation: float
    perturbed_correlation: float
    transition_count: int
    path_count: int
    passed: bool


@dataclass(frozen=True)
class PathSignatureValidation:
    signature_similarity: float
    signed_area: float
    signature_norm: float
    frechet_distance: float
    passed: bool


@dataclass(frozen=True)
class SpectralSignatureValidation:
    eigenvalue_min: float
    eigenvalue_max: float
    algebraic_connectivity: float
    component_count: int
    heat_trace: tuple[float, ...]
    heat_times: tuple[float, ...]
    connected: bool
    passed: bool


@dataclass(frozen=True)
class DimensionConstraintValidation:
    base_dimension: int
    padded_dimension: int
    sample_count: int
    k_neighbors: int
    gram_cka: float
    geodesic_mean_abs_diff: float
    geodesic_max_abs_diff: float
    spectral_eigen_mean_abs_diff: float
    spectral_eigen_max_abs_diff: float
    spectral_entropy_base: float
    spectral_entropy_padded: float
    heat_trace_base: tuple[float, ...]
    heat_trace_padded: tuple[float, ...]
    heat_times: tuple[float, ...]
    betti_numbers_base: dict[int, int]
    betti_numbers_padded: dict[int, int]
    component_count_base: int
    component_count_padded: int
    cycle_count_base: int
    cycle_count_padded: int
    persistence_entropy_base: float
    persistence_entropy_padded: float
    max_persistence_base: float
    max_persistence_padded: float
    passed: bool


@dataclass(frozen=True)
class GromovWassersteinFixture:
    points_a: tuple[tuple[float, ...], ...]
    points_b: tuple[tuple[float, ...], ...]
    permutation: tuple[int, ...]
    source_distances: tuple[tuple[float, ...], ...]
    target_distances: tuple[tuple[float, ...], ...]
    symmetry_source_distances: tuple[tuple[float, ...], ...]
    symmetry_target_distances: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TraversalCoherenceFixture:
    anchor_ids: tuple[str, ...]
    anchor_gram: tuple[float, ...]
    perturbed_gram: tuple[float, ...]
    paths: tuple[TraversalPath, ...]


@dataclass(frozen=True)
class PathSignatureFixture:
    gate_embeddings: dict[str, tuple[float, ...]]
    shifted_embeddings: dict[str, tuple[float, ...]]
    path: PathSignature


@dataclass(frozen=True)
class SpectralSignatureFixture:
    points: tuple[tuple[float, ...], ...]
    expected_component_count: int
    expected_connected: bool


@dataclass(frozen=True)
class DimensionConstraintFixture:
    points: tuple[tuple[float, ...], ...]
    padded_dimension: int


@dataclass(frozen=True)
class Fixtures:
    gromov_wasserstein: GromovWassersteinFixture
    traversal_coherence: TraversalCoherenceFixture
    path_signature: PathSignatureFixture
    spectral_signature: SpectralSignatureFixture
    spectral_signature_connected: SpectralSignatureFixture
    dimension_constraint: DimensionConstraintFixture


@dataclass(frozen=True)
class Report:
    suite_version: str
    timestamp: datetime
    passed: bool
    gromov_wasserstein: GromovWassersteinValidation
    traversal_coherence: TraversalCoherenceValidation
    path_signature: PathSignatureValidation
    spectral_signature: SpectralSignatureValidation
    spectral_signature_connected: SpectralSignatureValidation
    dimension_constraint: DimensionConstraintValidation


class GeometryValidationSuite:
    """Geometry validation suite using GPU-accelerated operations."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._gw = GromovWassersteinDistance(self._backend)

    def _array_to_2d_tuple(self, array: "Array") -> tuple[tuple[float, ...], ...]:
        """Convert 2D array to nested tuples using native tolist() - O(1) vs O(n*m)."""
        nested_list = self._backend.tolist(array)
        return tuple(tuple(row) for row in nested_list)

    def run(self) -> Report:
        """Run the full geometry validation suite.

        All algorithm parameters are derived from data (machine epsilon, dtype).
        """
        fixtures = self._build_fixtures()
        gw_validation = self._validate_gromov_wasserstein(
            fixture=fixtures.gromov_wasserstein,
        )
        traversal_validation = self._validate_traversal_coherence(
            fixture=fixtures.traversal_coherence,
        )
        path_validation = self._validate_path_signature(
            fixture=fixtures.path_signature,
        )
        spectral_validation = self._validate_spectral_signature(
            fixture=fixtures.spectral_signature,
        )
        spectral_connected_validation = self._validate_spectral_signature(
            fixture=fixtures.spectral_signature_connected,
        )
        dimension_constraint_validation = self._validate_dimension_constraint(
            fixture=fixtures.dimension_constraint,
        )

        passed = (
            gw_validation.passed
            and traversal_validation.passed
            and path_validation.passed
            and spectral_validation.passed
            and spectral_connected_validation.passed
            and dimension_constraint_validation.passed
        )

        return Report(
            suite_version=SUITE_VERSION,
            timestamp=datetime.utcnow(),
            passed=passed,
            gromov_wasserstein=gw_validation,
            traversal_coherence=traversal_validation,
            path_signature=path_validation,
            spectral_signature=spectral_validation,
            spectral_signature_connected=spectral_connected_validation,
            dimension_constraint=dimension_constraint_validation,
        )

    def _build_fixtures(self) -> Fixtures:
        backend = self._backend

        points_a_list = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        permutation_list = [2, 0, 1]
        points_b_list = [points_a_list[idx] for idx in permutation_list]

        # Convert to backend arrays and compute distances
        points_a_arr = backend.array(points_a_list)
        points_b_arr = backend.array(points_b_list)
        source_distances_arr = self._gw.compute_pairwise_distances(points_a_arr)
        target_distances_arr = self._gw.compute_pairwise_distances(points_b_arr)

        # Convert back to tuples for fixture storage
        backend.eval(source_distances_arr, target_distances_arr)
        source_distances = self._array_to_2d_tuple(source_distances_arr)
        target_distances = self._array_to_2d_tuple(target_distances_arr)
        symmetry_source_distances = (
            (0.0, 1.0, 3.0),
            (1.0, 0.0, 1.0),
            (3.0, 1.0, 0.0),
        )
        symmetry_target_distances = (
            (0.0, 2.0, 1.0),
            (2.0, 0.0, 2.0),
            (1.0, 2.0, 0.0),
        )
        gw_fixture = GromovWassersteinFixture(
            points_a=tuple(tuple(p) for p in points_a_list),
            points_b=tuple(tuple(p) for p in points_b_list),
            permutation=tuple(permutation_list),
            source_distances=source_distances,
            target_distances=target_distances,
            symmetry_source_distances=symmetry_source_distances,
            symmetry_target_distances=symmetry_target_distances,
        )

        anchor_ids_list = ["A", "B", "C", "D"]
        anchor_gram_list = [
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            1.0,
        ]
        perturbed_gram_list = list(anchor_gram_list)
        n = len(anchor_ids_list)
        perturbations = [
            (1, 2, -0.4),
            (2, 3, 0.4),
            (0, 3, -0.2),
        ]
        for i, j, delta in perturbations:
            perturbed_gram_list[i * n + j] += delta
            perturbed_gram_list[j * n + i] += delta
        traversal_fixture = TraversalCoherenceFixture(
            anchor_ids=tuple(anchor_ids_list),
            anchor_gram=tuple(anchor_gram_list),
            perturbed_gram=tuple(perturbed_gram_list),
            paths=(
                TraversalPath(anchor_ids=["A", "B", "C", "D"]),
                TraversalPath(anchor_ids=["A", "C", "D", "B"]),
            ),
        )

        gate_embeddings = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.0),
            "C": (1.0, 1.0),
        }
        shifted_embeddings = {
            "A": (2.0, -1.5),
            "B": (3.0, -1.5),
            "C": (3.0, -0.5),
        }
        path_id = UUID("00000000-0000-0000-0000-000000000101")
        path = PathSignature(
            id=path_id,
            model_id="validation-model",
            prompt_id="validation-prompt",
            nodes=[
                PathNode(gate_id="A", token_index=0, entropy=0.0),
                PathNode(gate_id="B", token_index=1, entropy=0.0),
                PathNode(gate_id="C", token_index=2, entropy=0.0),
            ],
        )
        path_fixture = PathSignatureFixture(
            gate_embeddings=gate_embeddings,
            shifted_embeddings=shifted_embeddings,
            path=path,
        )

        # Fixture points for spectral signature test.
        # Two clusters are spatially separated but derive_k_neighbors finds
        # the minimum k that yields a connected graph, so expected count is 1.
        spectral_fixture = SpectralSignatureFixture(
            points=(
                (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),  # Cluster A
                (100.0, 0.0), (101.0, 0.0), (102.0, 0.0),  # Cluster B
            ),
            expected_component_count=1,
            expected_connected=True,
        )
        # Fixture points for connected component test.
        # Linear arrangement ensures connectivity with any k >= 1.
        spectral_connected_fixture = SpectralSignatureFixture(
            points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
            expected_component_count=1,
            expected_connected=True,
        )
        dimension_constraint_fixture = DimensionConstraintFixture(
            points=(
                (0.0, 0.0),
                (1.0, 0.0),
                (0.5, 0.8),
                (0.1, 0.7),
                (0.9, 0.2),
            ),
            padded_dimension=4,
        )

        return Fixtures(
            gromov_wasserstein=gw_fixture,
            traversal_coherence=traversal_fixture,
            path_signature=path_fixture,
            spectral_signature=spectral_fixture,
            spectral_signature_connected=spectral_connected_fixture,
            dimension_constraint=dimension_constraint_fixture,
        )

    def _validate_gromov_wasserstein(
        self,
        fixture: GromovWassersteinFixture,
    ) -> GromovWassersteinValidation:
        backend = self._backend

        # Convert fixture data to backend arrays
        source_dist = backend.array(fixture.source_distances)
        target_dist = backend.array(fixture.target_distances)
        sym_source_dist = backend.array(fixture.symmetry_source_distances)
        sym_target_dist = backend.array(fixture.symmetry_target_distances)

        # All solver params derived from dtype - no config needed
        identity = self._gw.compute(
            source_distances=source_dist,
            target_distances=source_dist,
        )
        permuted = self._gw.compute(
            source_distances=source_dist,
            target_distances=target_dist,
        )
        symmetry_forward = self._gw.compute(
            source_distances=sym_source_dist,
            target_distances=sym_target_dist,
        )
        symmetry_reverse = self._gw.compute(
            source_distances=sym_target_dist,
            target_distances=sym_source_dist,
        )
        symmetry_delta = abs(symmetry_forward.distance - symmetry_reverse.distance)
        row_error, column_error = self._coupling_mass_errors(permuted.coupling)

        distance_eps = machine_epsilon(backend, source_dist)
        coupling_eps = regularization_epsilon(backend, permuted.coupling)
        passed = (
            abs(identity.distance) <= distance_eps
            and abs(permuted.distance) <= distance_eps
            and symmetry_delta <= distance_eps
            and row_error <= coupling_eps
            and column_error <= coupling_eps
            and permuted.converged
        )

        return GromovWassersteinValidation(
            distance_identity=float(identity.distance),
            distance_permutation=float(permuted.distance),
            symmetry_delta=float(symmetry_delta),
            max_row_mass_error=float(row_error),
            max_column_mass_error=float(column_error),
            converged=permuted.converged,
            iterations=permuted.iterations,
            passed=passed,
        )

    def _coupling_mass_errors(self, coupling: "Array") -> tuple[float, float]:
        """Compute coupling mass errors using GPU-accelerated operations."""
        backend = self._backend
        n = coupling.shape[0]
        m = coupling.shape[1]

        if n <= 0 or m <= 0:
            return float("inf"), float("inf")

        expected_row = 1.0 / float(n)
        expected_col = 1.0 / float(m)

        row_sums = backend.sum(coupling, axis=1)
        col_sums = backend.sum(coupling, axis=0)

        row_errors = backend.abs(row_sums - expected_row)
        col_errors = backend.abs(col_sums - expected_col)

        max_row_error = backend.max(row_errors)
        max_col_error = backend.max(col_errors)

        backend.eval(max_row_error, max_col_error)
        return float(backend.to_scalar(max_row_error)), float(backend.to_scalar(max_col_error))

    def _validate_traversal_coherence(
        self,
        fixture: TraversalCoherenceFixture,
    ) -> TraversalCoherenceValidation:
        backend = self._backend
        self_result = TraversalCoherence.compare(
            paths=fixture.paths,
            gram_a=fixture.anchor_gram,
            gram_b=fixture.anchor_gram,
            anchor_ids=fixture.anchor_ids,
        )
        perturbed_result = TraversalCoherence.compare(
            paths=fixture.paths,
            gram_a=fixture.anchor_gram,
            gram_b=fixture.perturbed_gram,
            anchor_ids=fixture.anchor_ids,
        )

        self_corr = self_result.transition_gram_correlation if self_result else float("nan")
        perturbed_corr = (
            perturbed_result.transition_gram_correlation if perturbed_result else float("nan")
        )
        transition_count = self_result.transition_count if self_result else 0
        path_count = self_result.path_count if self_result else 0

        corr_arr = backend.array(
            [self_corr, perturbed_corr], dtype=precision_dtype(backend)
        )
        corr_eps = machine_epsilon(backend, corr_arr)
        passed = (
            self_corr == self_corr
            and perturbed_corr == perturbed_corr
            and abs(self_corr - 1.0) <= corr_eps
            and perturbed_corr <= self_corr - corr_eps
        )

        return TraversalCoherenceValidation(
            self_correlation=float(self_corr),
            perturbed_correlation=float(perturbed_corr),
            transition_count=transition_count,
            path_count=path_count,
            passed=passed,
        )

    def _validate_path_signature(
        self,
        fixture: PathSignatureFixture,
    ) -> PathSignatureValidation:
        backend = self._backend
        signature = PathGeometry.compute_signature(
            fixture.path,
            gate_embeddings=fixture.gate_embeddings,
        )
        shifted_signature = PathGeometry.compute_signature(
            fixture.path,
            gate_embeddings=fixture.shifted_embeddings,
        )
        similarity = PathGeometry.signature_similarity(signature, shifted_signature)
        frechet = PathGeometry.frechet_distance(
            fixture.path,
            fixture.path,
            gate_embeddings=fixture.gate_embeddings,
        )

        signature_eps = machine_epsilon(
            backend,
            backend.array([signature.signature_norm, shifted_signature.signature_norm]),
        )
        similarity_eps = machine_epsilon(backend, backend.array([similarity]))
        frechet_eps = machine_epsilon(backend, backend.array([frechet.distance]))
        passed = (
            abs(similarity - 1.0) <= similarity_eps
            and abs(frechet.distance) <= frechet_eps
        )

        return PathSignatureValidation(
            signature_similarity=float(similarity),
            signed_area=float(signature.signed_area),
            signature_norm=float(signature.signature_norm),
            frechet_distance=float(frechet.distance),
            passed=passed,
        )

    def _validate_spectral_signature(
        self,
        fixture: SpectralSignatureFixture,
    ) -> SpectralSignatureValidation:
        backend = self._backend
        # All parameters derived from data - no config needed
        computer = SpectralSignature(backend)
        signature = computer.compute(points=fixture.points)

        eig_min = min(signature.eigenvalues) if signature.eigenvalues else 0.0
        eig_max = max(signature.eigenvalues) if signature.eigenvalues else 0.0

        eig_arr = backend.array(
            signature.eigenvalues or [0.0], dtype=precision_dtype(backend)
        )
        eig_eps = regularization_epsilon(backend, eig_arr)

        # Normalized Laplacian is always used (it's the correct approach)
        eigen_bounds_ok = eig_min >= -eig_eps and eig_max <= 2.0 + eig_eps

        heat_arr = backend.array(
            signature.heat_trace or [0.0], dtype=precision_dtype(backend)
        )
        heat_eps = regularization_epsilon(backend, heat_arr)

        # Vectorized monotonicity check: heat[i] + eps >= heat[i+1] for all i
        if len(signature.heat_trace) > 1:
            heat_diff = heat_arr[:-1] - heat_arr[1:]  # heat[i] - heat[i+1]
            # If any diff < -eps, it's not monotone
            violations = backend.sum(heat_diff < -heat_eps)
            backend.eval(violations)
            heat_monotone = int(backend.to_scalar(violations)) == 0
        else:
            heat_monotone = True

        component_ok = signature.component_count == fixture.expected_component_count
        connected_ok = signature.connected == fixture.expected_connected
        if fixture.expected_connected:
            connectivity_ok = signature.algebraic_connectivity > eig_eps
        else:
            connectivity_ok = signature.algebraic_connectivity <= eig_eps

        passed = eigen_bounds_ok and heat_monotone and component_ok and connected_ok and connectivity_ok

        return SpectralSignatureValidation(
            eigenvalue_min=float(eig_min),
            eigenvalue_max=float(eig_max),
            algebraic_connectivity=float(signature.algebraic_connectivity),
            component_count=signature.component_count,
            heat_trace=tuple(signature.heat_trace) if signature.heat_trace else (),
            heat_times=tuple(signature.heat_times) if signature.heat_times else (),
            connected=signature.connected,
            passed=passed,
        )

    def _validate_dimension_constraint(
        self,
        fixture: DimensionConstraintFixture,
    ) -> DimensionConstraintValidation:
        backend = self._backend
        points = fixture.points
        sample_count = len(points)
        base_dim = len(points[0]) if points else 0

        padded_points = [
            row + (0.0,) * (fixture.padded_dimension - base_dim) for row in points
        ]

        base_arr = backend.array(points)
        padded_arr = backend.array(padded_points)
        backend.eval(base_arr, padded_arr)

        gram_base = _cache.get_or_compute_gram(base_arr, backend)
        gram_padded = _cache.get_or_compute_gram(padded_arr, backend)
        gram_cka = compute_cka_from_grams(gram_base, gram_padded, backend=backend)

        geometry = RiemannianGeometry(backend)
        geo_base = geometry.geodesic_distances(points, k_neighbors=None)
        geo_padded = geometry.geodesic_distances(
            padded_points, k_neighbors=geo_base.k_neighbors
        )

        geo_diff = backend.abs(geo_base.distances - geo_padded.distances)
        geo_mean = backend.mean(geo_diff)
        geo_max = backend.max(geo_diff)
        backend.eval(geo_mean, geo_max)
        geodesic_mean_abs_diff = float(backend.to_scalar(geo_mean))
        geodesic_max_abs_diff = float(backend.to_scalar(geo_max))

        geo_eps = regularization_epsilon(backend, geo_base.distances)

        # All parameters derived from data - no config needed
        spectral = SpectralSignature(backend)
        sig_base = spectral.compute(points=points)
        sig_padded = spectral.compute(points=padded_points)

        # Vectorized eigenvalue diff computation
        if sig_base.eigenvalues and sig_padded.eigenvalues:
            min_eigen_len = min(len(sig_base.eigenvalues), len(sig_padded.eigenvalues))
            eigen_base_arr = backend.array(sig_base.eigenvalues[:min_eigen_len])
            eigen_padded_arr = backend.array(sig_padded.eigenvalues[:min_eigen_len])
            eigen_diff_arr = backend.abs(eigen_base_arr - eigen_padded_arr)
            eigen_mean_arr = backend.mean(eigen_diff_arr)
            eigen_max_arr = backend.max(eigen_diff_arr)
            backend.eval(eigen_mean_arr, eigen_max_arr)
            spectral_eigen_mean_abs_diff = float(backend.to_scalar(eigen_mean_arr))
            spectral_eigen_max_abs_diff = float(backend.to_scalar(eigen_max_arr))
        else:
            spectral_eigen_mean_abs_diff = 0.0
            spectral_eigen_max_abs_diff = 0.0

        spectral_entropy_diff = abs(sig_base.spectral_entropy - sig_padded.spectral_entropy)

        # Vectorized heat trace diff computation
        if sig_base.heat_trace and sig_padded.heat_trace:
            min_heat_len = min(len(sig_base.heat_trace), len(sig_padded.heat_trace))
            heat_base_arr = backend.array(sig_base.heat_trace[:min_heat_len])
            heat_padded_arr = backend.array(sig_padded.heat_trace[:min_heat_len])
            heat_diff_arr = backend.abs(heat_base_arr - heat_padded_arr)
            heat_max_arr = backend.max(heat_diff_arr)
            backend.eval(heat_max_arr)
            heat_trace_max_abs_diff = float(backend.to_scalar(heat_max_arr))
        else:
            heat_trace_max_abs_diff = 0.0

        fp_base = TopologicalFingerprint.compute(points)
        fp_padded = TopologicalFingerprint.compute(padded_points)

        persistence_entropy_diff = abs(
            fp_base.summary.persistence_entropy - fp_padded.summary.persistence_entropy
        )
        max_persistence_diff = abs(
            fp_base.summary.max_persistence - fp_padded.summary.max_persistence
        )
        topology_matches = (
            fp_base.betti_numbers == fp_padded.betti_numbers
            and fp_base.summary.component_count == fp_padded.summary.component_count
            and fp_base.summary.cycle_count == fp_padded.summary.cycle_count
        )

        cka_eps = machine_epsilon(backend, gram_base)
        eigen_eps = regularization_epsilon(
            backend,
            backend.array(
                sig_base.eigenvalues or [0.0], dtype=precision_dtype(backend)
            ),
        )
        entropy_eps = machine_epsilon(
            backend,
            backend.array([sig_base.spectral_entropy, sig_padded.spectral_entropy]),
        )
        heat_eps = regularization_epsilon(
            backend,
            backend.array(sig_base.heat_trace or [0.0], dtype=precision_dtype(backend)),
        )
        topo_eps = machine_epsilon(
            backend,
            backend.array(
                [
                    fp_base.summary.persistence_entropy,
                    fp_padded.summary.persistence_entropy,
                    fp_base.summary.max_persistence,
                    fp_padded.summary.max_persistence,
                ],
                dtype=precision_dtype(backend),
            ),
        )
        passed = (
            abs(float(gram_cka) - 1.0) <= cka_eps
            and geodesic_mean_abs_diff <= geo_eps
            and geodesic_max_abs_diff <= geo_eps
            and spectral_eigen_mean_abs_diff <= eigen_eps
            and spectral_eigen_max_abs_diff <= eigen_eps
            and spectral_entropy_diff <= entropy_eps
            and heat_trace_max_abs_diff <= heat_eps
            and topology_matches
            and persistence_entropy_diff <= topo_eps
            and max_persistence_diff <= topo_eps
        )

        return DimensionConstraintValidation(
            base_dimension=base_dim,
            padded_dimension=fixture.padded_dimension,
            sample_count=sample_count,
            k_neighbors=geo_base.k_neighbors,
            gram_cka=float(gram_cka),
            geodesic_mean_abs_diff=geodesic_mean_abs_diff,
            geodesic_max_abs_diff=geodesic_max_abs_diff,
            spectral_eigen_mean_abs_diff=spectral_eigen_mean_abs_diff,
            spectral_eigen_max_abs_diff=spectral_eigen_max_abs_diff,
            spectral_entropy_base=sig_base.spectral_entropy,
            spectral_entropy_padded=sig_padded.spectral_entropy,
            heat_trace_base=tuple(sig_base.heat_trace) if sig_base.heat_trace else (),
            heat_trace_padded=tuple(sig_padded.heat_trace) if sig_padded.heat_trace else (),
            heat_times=tuple(sig_base.heat_times) if sig_base.heat_times else (),
            betti_numbers_base=fp_base.betti_numbers,
            betti_numbers_padded=fp_padded.betti_numbers,
            component_count_base=fp_base.summary.component_count,
            component_count_padded=fp_padded.summary.component_count,
            cycle_count_base=fp_base.summary.cycle_count,
            cycle_count_padded=fp_padded.summary.cycle_count,
            persistence_entropy_base=fp_base.summary.persistence_entropy,
            persistence_entropy_padded=fp_padded.summary.persistence_entropy,
            max_persistence_base=fp_base.summary.max_persistence,
            max_persistence_padded=fp_padded.summary.max_persistence,
            passed=passed,
        )
