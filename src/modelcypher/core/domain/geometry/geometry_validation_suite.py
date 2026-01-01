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
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams
from modelcypher.core.domain.geometry.gromov_wasserstein import Config as GWConfig
from modelcypher.core.domain.geometry.gromov_wasserstein import GromovWassersteinDistance
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.path_geometry import PathGeometry, PathNode, PathSignature
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.spectral_signature import (
    SpectralSignature,
    SpectralSignatureConfig,
)
from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint
from modelcypher.core.domain.geometry.traversal_coherence import Path as TraversalPath
from modelcypher.core.domain.geometry.traversal_coherence import TraversalCoherence

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

SUITE_VERSION = "1.0"


@dataclass(frozen=True)
class Thresholds:
    """Validation thresholds for geometry suite.

    Use with_parameters() to create with explicit values.
    """

    identity_distance_max: float
    permutation_distance_max: float
    symmetry_delta_max: float
    coupling_mass_error_max: float
    traversal_self_correlation_min: float
    traversal_perturbed_correlation_max: float
    signature_similarity_min: float
    frechet_distance_max: float
    dimension_constraint_cka_min: float
    dimension_constraint_geodesic_mean_abs_diff_max: float
    dimension_constraint_geodesic_max_abs_diff_max: float
    dimension_constraint_spectral_eigen_mean_abs_diff_max: float
    dimension_constraint_spectral_eigen_max_abs_diff_max: float
    dimension_constraint_spectral_entropy_abs_diff_max: float
    dimension_constraint_heat_trace_max_abs_diff_max: float
    dimension_constraint_topology_abs_diff_max: float

    @classmethod
    def with_parameters(
        cls,
        *,
        identity_distance_max: float = 1e-6,
        permutation_distance_max: float = 0.02,
        symmetry_delta_max: float = 1e-3,
        coupling_mass_error_max: float = 0.02,
        traversal_self_correlation_min: float = 0.999,
        traversal_perturbed_correlation_max: float = 0.995,
        signature_similarity_min: float = 0.999,
        frechet_distance_max: float = 1e-5,
        dimension_constraint_cka_min: float = 0.999999,
        dimension_constraint_geodesic_mean_abs_diff_max: float = 1e-6,
        dimension_constraint_geodesic_max_abs_diff_max: float = 1e-6,
        dimension_constraint_spectral_eigen_mean_abs_diff_max: float = 1e-6,
        dimension_constraint_spectral_eigen_max_abs_diff_max: float = 1e-6,
        dimension_constraint_spectral_entropy_abs_diff_max: float = 1e-6,
        dimension_constraint_heat_trace_max_abs_diff_max: float = 1e-6,
        dimension_constraint_topology_abs_diff_max: float = 1e-9,
    ) -> "Thresholds":
        """Create thresholds with explicit parameters.

        Args:
            identity_distance_max: Maximum distance for identity test.
            permutation_distance_max: Maximum distance for permutation test.
            symmetry_delta_max: Maximum symmetry deviation.
            coupling_mass_error_max: Maximum coupling mass error.
            traversal_self_correlation_min: Minimum self-correlation.
            traversal_perturbed_correlation_max: Maximum perturbed correlation.
            signature_similarity_min: Minimum signature similarity.
            frechet_distance_max: Maximum Frechet distance.
            dimension_constraint_cka_min: Minimum Gram CKA for padding invariance.
            dimension_constraint_geodesic_mean_abs_diff_max: Maximum mean geodesic diff.
            dimension_constraint_geodesic_max_abs_diff_max: Maximum max geodesic diff.
            dimension_constraint_spectral_eigen_mean_abs_diff_max: Maximum mean eigen diff.
            dimension_constraint_spectral_eigen_max_abs_diff_max: Maximum max eigen diff.
            dimension_constraint_spectral_entropy_abs_diff_max: Maximum spectral entropy diff.
            dimension_constraint_heat_trace_max_abs_diff_max: Maximum heat trace diff.
            dimension_constraint_topology_abs_diff_max: Maximum topology summary diff.

        Returns:
            Thresholds with specified parameters.
        """
        return cls(
            identity_distance_max=identity_distance_max,
            permutation_distance_max=permutation_distance_max,
            symmetry_delta_max=symmetry_delta_max,
            coupling_mass_error_max=coupling_mass_error_max,
            traversal_self_correlation_min=traversal_self_correlation_min,
            traversal_perturbed_correlation_max=traversal_perturbed_correlation_max,
            signature_similarity_min=signature_similarity_min,
            frechet_distance_max=frechet_distance_max,
            dimension_constraint_cka_min=dimension_constraint_cka_min,
            dimension_constraint_geodesic_mean_abs_diff_max=(
                dimension_constraint_geodesic_mean_abs_diff_max
            ),
            dimension_constraint_geodesic_max_abs_diff_max=(
                dimension_constraint_geodesic_max_abs_diff_max
            ),
            dimension_constraint_spectral_eigen_mean_abs_diff_max=(
                dimension_constraint_spectral_eigen_mean_abs_diff_max
            ),
            dimension_constraint_spectral_eigen_max_abs_diff_max=(
                dimension_constraint_spectral_eigen_max_abs_diff_max
            ),
            dimension_constraint_spectral_entropy_abs_diff_max=(
                dimension_constraint_spectral_entropy_abs_diff_max
            ),
            dimension_constraint_heat_trace_max_abs_diff_max=(
                dimension_constraint_heat_trace_max_abs_diff_max
            ),
            dimension_constraint_topology_abs_diff_max=dimension_constraint_topology_abs_diff_max,
        )

    @classmethod
    def standard(cls) -> "Thresholds":
        """Return standard thresholds."""
        return cls.with_parameters()

    @classmethod
    def default(cls) -> "Thresholds":
        """Alias for standard thresholds."""
        return cls.standard()


@dataclass(frozen=True)
class GromovWassersteinConfig:
    """Configuration for GW solver using Frank-Wolfe algorithm.

    Use with_parameters() to create with explicit values.
    """

    # Frank-Wolfe parameters
    max_outer_iterations: int
    min_outer_iterations: int
    convergence_threshold: float
    relative_objective_threshold: float

    # Sinkhorn parameters for linear OT subproblem
    sinkhorn_epsilon: float
    sinkhorn_iterations: int
    sinkhorn_threshold: float

    # Loss function
    use_squared_loss: bool

    # Random restarts to escape local minima
    num_restarts: int

    @classmethod
    def with_parameters(
        cls,
        *,
        max_outer_iterations: int = 100,
        min_outer_iterations: int = 5,
        convergence_threshold: float = 1e-7,
        relative_objective_threshold: float = 1e-7,
        sinkhorn_epsilon: float = 0.001,
        sinkhorn_iterations: int = 200,
        sinkhorn_threshold: float = 1e-8,
        use_squared_loss: bool = True,
        num_restarts: int = 5,
    ) -> "GromovWassersteinConfig":
        """Create configuration with explicit parameters.

        Args:
            max_outer_iterations: Maximum Frank-Wolfe iterations.
            min_outer_iterations: Minimum iterations before convergence check.
            convergence_threshold: Absolute convergence threshold.
            relative_objective_threshold: Relative objective convergence threshold.
            sinkhorn_epsilon: Entropy regularization for Sinkhorn.
            sinkhorn_iterations: Maximum Sinkhorn iterations.
            sinkhorn_threshold: Sinkhorn convergence threshold.
            use_squared_loss: Whether to use squared loss function.
            num_restarts: Number of random restarts.

        Returns:
            Configuration with specified parameters.
        """
        if max_outer_iterations < 1:
            raise ValueError(f"max_outer_iterations must be >= 1, got {max_outer_iterations}")
        if min_outer_iterations < 1:
            raise ValueError(f"min_outer_iterations must be >= 1, got {min_outer_iterations}")
        if sinkhorn_epsilon <= 0:
            raise ValueError(f"sinkhorn_epsilon must be > 0, got {sinkhorn_epsilon}")
        if sinkhorn_iterations < 1:
            raise ValueError(f"sinkhorn_iterations must be >= 1, got {sinkhorn_iterations}")
        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")
        return cls(
            max_outer_iterations=max_outer_iterations,
            min_outer_iterations=min_outer_iterations,
            convergence_threshold=convergence_threshold,
            relative_objective_threshold=relative_objective_threshold,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_threshold=sinkhorn_threshold,
            use_squared_loss=use_squared_loss,
            num_restarts=num_restarts,
        )

    @classmethod
    def standard(cls) -> "GromovWassersteinConfig":
        """Return standard GW solver configuration."""
        return cls.with_parameters()

    @classmethod
    def default(cls) -> "GromovWassersteinConfig":
        """Alias for standard GW solver configuration."""
        return cls.standard()

    def solver_config(self) -> GWConfig:
        return GWConfig(
            max_outer_iterations=self.max_outer_iterations,
            min_outer_iterations=self.min_outer_iterations,
            convergence_threshold=self.convergence_threshold,
            relative_objective_threshold=self.relative_objective_threshold,
            sinkhorn_epsilon=self.sinkhorn_epsilon,
            sinkhorn_iterations=self.sinkhorn_iterations,
            sinkhorn_threshold=self.sinkhorn_threshold,
            use_squared_loss=self.use_squared_loss,
            num_restarts=self.num_restarts,
        )


@dataclass(frozen=True)
class Config:
    """Configuration for geometry validation suite.

    Use with_parameters() to create with explicit values.
    """

    include_fixtures: bool
    thresholds: Thresholds
    gromov_wasserstein: GromovWassersteinConfig

    @classmethod
    def with_parameters(
        cls,
        *,
        include_fixtures: bool = False,
        thresholds: Thresholds | None = None,
        gromov_wasserstein: GromovWassersteinConfig | None = None,
    ) -> "Config":
        """Create configuration with explicit parameters.

        Args:
            include_fixtures: Whether to include test fixtures in report.
            thresholds: Validation thresholds (uses with_parameters() defaults if None).
            gromov_wasserstein: GW solver config (uses with_parameters() defaults if None).

        Returns:
            Configuration with specified parameters.
        """
        return cls(
            include_fixtures=include_fixtures,
            thresholds=thresholds or Thresholds.with_parameters(),
            gromov_wasserstein=gromov_wasserstein or GromovWassersteinConfig.with_parameters(),
        )

    @classmethod
    def standard(cls) -> "Config":
        """Return standard suite configuration."""
        return cls.with_parameters()

    @classmethod
    def default(cls) -> "Config":
        """Alias for standard suite configuration."""
        return cls.standard()


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
    heat_trace: list[float]
    heat_times: list[float]
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
    heat_trace_base: list[float]
    heat_trace_padded: list[float]
    heat_times: list[float]
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
    points_a: list[list[float]]
    points_b: list[list[float]]
    permutation: list[int]
    source_distances: list[list[float]]
    target_distances: list[list[float]]
    symmetry_source_distances: list[list[float]]
    symmetry_target_distances: list[list[float]]


@dataclass(frozen=True)
class TraversalCoherenceFixture:
    anchor_ids: list[str]
    anchor_gram: list[float]
    perturbed_gram: list[float]
    paths: list[TraversalPath]


@dataclass(frozen=True)
class PathSignatureFixture:
    gate_embeddings: dict[str, list[float]]
    shifted_embeddings: dict[str, list[float]]
    path: PathSignature
    projection_dim: int


@dataclass(frozen=True)
class SpectralSignatureFixture:
    points: list[list[float]]
    k_neighbors: int | None
    normalized_laplacian: bool
    heat_times: list[float]
    expected_component_count: int
    expected_connected: bool


@dataclass(frozen=True)
class DimensionConstraintFixture:
    points: list[list[float]]
    padded_dimension: int
    k_neighbors: int
    heat_times: list[float]


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
    config: Config
    gromov_wasserstein: GromovWassersteinValidation
    traversal_coherence: TraversalCoherenceValidation
    path_signature: PathSignatureValidation
    spectral_signature: SpectralSignatureValidation
    spectral_signature_connected: SpectralSignatureValidation
    dimension_constraint: DimensionConstraintValidation
    fixtures: Fixtures | None


class GeometryValidationSuite:
    """Geometry validation suite using GPU-accelerated operations."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._gw = GromovWassersteinDistance(self._backend)

    def run(self, config: Config | None = None) -> Report:
        """Run the full geometry validation suite.

        Args:
            config: Suite configuration (use with_parameters() to create).
        """
        resolved = config or Config.default()
        fixtures = self._build_fixtures()
        gw_validation = self._validate_gromov_wasserstein(
            fixture=fixtures.gromov_wasserstein,
            config=resolved.gromov_wasserstein,
            thresholds=resolved.thresholds,
        )
        traversal_validation = self._validate_traversal_coherence(
            fixture=fixtures.traversal_coherence,
            thresholds=resolved.thresholds,
        )
        path_validation = self._validate_path_signature(
            fixture=fixtures.path_signature,
            thresholds=resolved.thresholds,
        )
        spectral_validation = self._validate_spectral_signature(
            fixture=fixtures.spectral_signature,
        )
        spectral_connected_validation = self._validate_spectral_signature(
            fixture=fixtures.spectral_signature_connected,
        )
        dimension_constraint_validation = self._validate_dimension_constraint(
            fixture=fixtures.dimension_constraint,
            thresholds=resolved.thresholds,
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
            config=resolved,
            gromov_wasserstein=gw_validation,
            traversal_coherence=traversal_validation,
            path_signature=path_validation,
            spectral_signature=spectral_validation,
            spectral_signature_connected=spectral_connected_validation,
            dimension_constraint=dimension_constraint_validation,
            fixtures=fixtures if resolved.include_fixtures else None,
        )

    def _build_fixtures(self) -> Fixtures:
        backend = self._backend

        points_a = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        permutation = [2, 0, 1]
        points_b = [points_a[idx] for idx in permutation]

        # Convert to backend arrays and compute distances
        points_a_arr = backend.array(points_a)
        points_b_arr = backend.array(points_b)
        source_distances_arr = self._gw.compute_pairwise_distances(points_a_arr)
        target_distances_arr = self._gw.compute_pairwise_distances(points_b_arr)

        # Convert back to lists for fixture storage
        backend.eval(source_distances_arr, target_distances_arr)
        source_distances = backend.to_numpy(source_distances_arr).tolist()
        target_distances = backend.to_numpy(target_distances_arr).tolist()
        symmetry_source_distances = [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]
        symmetry_target_distances = [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
        gw_fixture = GromovWassersteinFixture(
            points_a=points_a,
            points_b=points_b,
            permutation=permutation,
            source_distances=source_distances,
            target_distances=target_distances,
            symmetry_source_distances=symmetry_source_distances,
            symmetry_target_distances=symmetry_target_distances,
        )

        anchor_ids = ["A", "B", "C", "D"]
        anchor_gram = [
            1,
            0,
            -1,
            0,
            0,
            1,
            0,
            -1,
            -1,
            0,
            1,
            0,
            0,
            -1,
            0,
            1,
        ]
        perturbed_gram = list(anchor_gram)
        n = len(anchor_ids)
        perturbations = [
            (1, 2, -0.4),
            (2, 3, 0.4),
            (0, 3, -0.2),
        ]
        for i, j, delta in perturbations:
            perturbed_gram[i * n + j] += delta
            perturbed_gram[j * n + i] += delta
        traversal_fixture = TraversalCoherenceFixture(
            anchor_ids=anchor_ids,
            anchor_gram=anchor_gram,
            perturbed_gram=perturbed_gram,
            paths=[
                TraversalPath(anchor_ids=["A", "B", "C", "D"]),
                TraversalPath(anchor_ids=["A", "C", "D", "B"]),
            ],
        )

        gate_embeddings = {
            "A": [0.0, 0.0],
            "B": [1.0, 0.0],
            "C": [1.0, 1.0],
        }
        shifted_embeddings = {
            "A": [2.0, -1.5],
            "B": [3.0, -1.5],
            "C": [3.0, -0.5],
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
            projection_dim=3,
        )

        spectral_fixture = SpectralSignatureFixture(
            points=[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            k_neighbors=1,
            normalized_laplacian=True,
            heat_times=[0.1, 1.0, 10.0],
            expected_component_count=2,
            expected_connected=False,
        )
        spectral_connected_fixture = SpectralSignatureFixture(
            points=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            k_neighbors=1,
            normalized_laplacian=True,
            heat_times=[0.1, 1.0, 10.0],
            expected_component_count=1,
            expected_connected=True,
        )
        dimension_constraint_fixture = DimensionConstraintFixture(
            points=[
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.8],
                [0.1, 0.7],
                [0.9, 0.2],
            ],
            padded_dimension=4,
            k_neighbors=3,
            heat_times=[0.1, 1.0, 10.0],
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
        config: GromovWassersteinConfig,
        thresholds: Thresholds,
    ) -> GromovWassersteinValidation:
        backend = self._backend
        solver_config = config.solver_config()

        # Convert fixture data to backend arrays
        source_dist = backend.array(fixture.source_distances)
        target_dist = backend.array(fixture.target_distances)
        sym_source_dist = backend.array(fixture.symmetry_source_distances)
        sym_target_dist = backend.array(fixture.symmetry_target_distances)

        identity = self._gw.compute(
            source_distances=source_dist,
            target_distances=source_dist,
            config=solver_config,
        )
        permuted = self._gw.compute(
            source_distances=source_dist,
            target_distances=target_dist,
            config=solver_config,
        )
        symmetry_forward = self._gw.compute(
            source_distances=sym_source_dist,
            target_distances=sym_target_dist,
            config=solver_config,
        )
        symmetry_reverse = self._gw.compute(
            source_distances=sym_target_dist,
            target_distances=sym_source_dist,
            config=solver_config,
        )
        symmetry_delta = abs(symmetry_forward.distance - symmetry_reverse.distance)
        row_error, column_error = self._coupling_mass_errors(permuted.coupling)

        passed = (
            identity.distance <= thresholds.identity_distance_max
            and permuted.distance <= thresholds.permutation_distance_max
            and symmetry_delta <= thresholds.symmetry_delta_max
            and row_error <= thresholds.coupling_mass_error_max
            and column_error <= thresholds.coupling_mass_error_max
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
        return float(backend.to_numpy(max_row_error)), float(backend.to_numpy(max_col_error))

    def _validate_traversal_coherence(
        self,
        fixture: TraversalCoherenceFixture,
        thresholds: Thresholds,
    ) -> TraversalCoherenceValidation:
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

        passed = (
            self_corr == self_corr
            and perturbed_corr == perturbed_corr
            and self_corr >= thresholds.traversal_self_correlation_min
            and perturbed_corr <= thresholds.traversal_perturbed_correlation_max
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
        thresholds: Thresholds,
    ) -> PathSignatureValidation:
        signature = PathGeometry.compute_signature(
            fixture.path,
            gate_embeddings=fixture.gate_embeddings,
            projection_dim=fixture.projection_dim,
        )
        shifted_signature = PathGeometry.compute_signature(
            fixture.path,
            gate_embeddings=fixture.shifted_embeddings,
            projection_dim=fixture.projection_dim,
        )
        similarity = PathGeometry.signature_similarity(signature, shifted_signature)
        frechet = PathGeometry.frechet_distance(
            fixture.path,
            fixture.path,
            gate_embeddings=fixture.gate_embeddings,
        )

        passed = (
            similarity >= thresholds.signature_similarity_min
            and frechet.distance <= thresholds.frechet_distance_max
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
        config = SpectralSignatureConfig(
            k_neighbors=fixture.k_neighbors,
            normalized_laplacian=fixture.normalized_laplacian,
            heat_trace_times=tuple(fixture.heat_times),
        )
        computer = SpectralSignature(backend)
        signature = computer.compute(points=fixture.points, config=config)

        eig_min = min(signature.eigenvalues) if signature.eigenvalues else 0.0
        eig_max = max(signature.eigenvalues) if signature.eigenvalues else 0.0

        eig_arr = backend.array(signature.eigenvalues or [0.0], dtype="float32")
        eig_eps = regularization_epsilon(backend, eig_arr)

        eigen_bounds_ok = True
        if fixture.normalized_laplacian:
            eigen_bounds_ok = eig_min >= -eig_eps and eig_max <= 2.0 + eig_eps

        heat_arr = backend.array(signature.heat_trace or [0.0], dtype="float32")
        heat_eps = regularization_epsilon(backend, heat_arr)
        heat_monotone = True
        for i in range(len(signature.heat_trace) - 1):
            if signature.heat_trace[i] + heat_eps < signature.heat_trace[i + 1]:
                heat_monotone = False
                break

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
            heat_trace=signature.heat_trace,
            heat_times=signature.heat_times,
            connected=signature.connected,
            passed=passed,
        )

    def _validate_dimension_constraint(
        self,
        fixture: DimensionConstraintFixture,
        thresholds: Thresholds,
    ) -> DimensionConstraintValidation:
        backend = self._backend
        points = fixture.points
        sample_count = len(points)
        base_dim = len(points[0]) if points else 0

        padded_points = [
            row + [0.0] * (fixture.padded_dimension - base_dim) for row in points
        ]

        base_arr = backend.array(points)
        padded_arr = backend.array(padded_points)
        backend.eval(base_arr, padded_arr)

        gram_base = backend.matmul(base_arr, backend.transpose(base_arr))
        gram_padded = backend.matmul(padded_arr, backend.transpose(padded_arr))
        backend.eval(gram_base, gram_padded)
        gram_cka = compute_cka_from_grams(gram_base, gram_padded, backend=backend)

        geometry = RiemannianGeometry(backend)
        geo_base = geometry.geodesic_distances(points, k_neighbors=fixture.k_neighbors)
        geo_padded = geometry.geodesic_distances(padded_points, k_neighbors=fixture.k_neighbors)

        geo_diff = backend.abs(geo_base.distances - geo_padded.distances)
        geo_mean = backend.mean(geo_diff)
        geo_max = backend.max(geo_diff)
        backend.eval(geo_mean, geo_max)
        geodesic_mean_abs_diff = float(backend.to_numpy(geo_mean).item())
        geodesic_max_abs_diff = float(backend.to_numpy(geo_max).item())

        spectral_config = SpectralSignatureConfig(
            k_neighbors=fixture.k_neighbors,
            normalized_laplacian=True,
            heat_trace_times=tuple(fixture.heat_times),
        )
        spectral = SpectralSignature(backend)
        sig_base = spectral.compute(points=points, config=spectral_config)
        sig_padded = spectral.compute(points=padded_points, config=spectral_config)

        eigen_diffs = [
            abs(a - b) for a, b in zip(sig_base.eigenvalues, sig_padded.eigenvalues)
        ]
        spectral_eigen_mean_abs_diff = (
            sum(eigen_diffs) / len(eigen_diffs) if eigen_diffs else 0.0
        )
        spectral_eigen_max_abs_diff = max(eigen_diffs) if eigen_diffs else 0.0

        spectral_entropy_diff = abs(sig_base.spectral_entropy - sig_padded.spectral_entropy)
        heat_diffs = [
            abs(a - b) for a, b in zip(sig_base.heat_trace, sig_padded.heat_trace)
        ]
        heat_trace_max_abs_diff = max(heat_diffs) if heat_diffs else 0.0

        fp_base = TopologicalFingerprint.compute(points, max_dimension=1)
        fp_padded = TopologicalFingerprint.compute(padded_points, max_dimension=1)

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

        passed = (
            float(gram_cka) >= thresholds.dimension_constraint_cka_min
            and geodesic_mean_abs_diff
            <= thresholds.dimension_constraint_geodesic_mean_abs_diff_max
            and geodesic_max_abs_diff
            <= thresholds.dimension_constraint_geodesic_max_abs_diff_max
            and spectral_eigen_mean_abs_diff
            <= thresholds.dimension_constraint_spectral_eigen_mean_abs_diff_max
            and spectral_eigen_max_abs_diff
            <= thresholds.dimension_constraint_spectral_eigen_max_abs_diff_max
            and spectral_entropy_diff
            <= thresholds.dimension_constraint_spectral_entropy_abs_diff_max
            and heat_trace_max_abs_diff
            <= thresholds.dimension_constraint_heat_trace_max_abs_diff_max
            and topology_matches
            and persistence_entropy_diff <= thresholds.dimension_constraint_topology_abs_diff_max
            and max_persistence_diff <= thresholds.dimension_constraint_topology_abs_diff_max
        )

        return DimensionConstraintValidation(
            base_dimension=base_dim,
            padded_dimension=fixture.padded_dimension,
            sample_count=sample_count,
            k_neighbors=fixture.k_neighbors,
            gram_cka=float(gram_cka),
            geodesic_mean_abs_diff=geodesic_mean_abs_diff,
            geodesic_max_abs_diff=geodesic_max_abs_diff,
            spectral_eigen_mean_abs_diff=spectral_eigen_mean_abs_diff,
            spectral_eigen_max_abs_diff=spectral_eigen_max_abs_diff,
            spectral_entropy_base=sig_base.spectral_entropy,
            spectral_entropy_padded=sig_padded.spectral_entropy,
            heat_trace_base=sig_base.heat_trace,
            heat_trace_padded=sig_padded.heat_trace,
            heat_times=sig_base.heat_times,
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
