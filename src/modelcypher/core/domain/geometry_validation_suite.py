from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from modelcypher.core.domain.gromov_wasserstein import Config as GWConfig
from modelcypher.core.domain.gromov_wasserstein import GromovWassersteinDistance
from modelcypher.core.domain.path_geometry import PathGeometry, PathNode, PathSignature
from modelcypher.core.domain.traversal_coherence import Path as TraversalPath
from modelcypher.core.domain.traversal_coherence import TraversalCoherence


SUITE_VERSION = "1.0"


@dataclass(frozen=True)
class Thresholds:
    identity_distance_max: float
    permutation_distance_max: float
    symmetry_delta_max: float
    coupling_mass_error_max: float
    traversal_self_correlation_min: float
    traversal_perturbed_correlation_max: float
    signature_similarity_min: float
    frechet_distance_max: float

    @staticmethod
    def standard() -> "Thresholds":
        return Thresholds(
            identity_distance_max=1e-6,
            permutation_distance_max=0.02,
            symmetry_delta_max=1e-3,
            coupling_mass_error_max=0.02,
            traversal_self_correlation_min=0.999,
            traversal_perturbed_correlation_max=0.995,
            signature_similarity_min=0.999,
            frechet_distance_max=1e-5,
        )


@dataclass(frozen=True)
class GromovWassersteinConfig:
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    max_outer_iterations: int
    min_outer_iterations: int
    max_inner_iterations: int
    convergence_threshold: float
    relative_objective_threshold: float
    use_squared_loss: bool

    @staticmethod
    def standard() -> "GromovWassersteinConfig":
        return GromovWassersteinConfig(
            epsilon=0.05,
            epsilon_min=0.005,
            epsilon_decay=0.97,
            max_outer_iterations=60,
            min_outer_iterations=4,
            max_inner_iterations=150,
            convergence_threshold=1e-6,
            relative_objective_threshold=1e-6,
            use_squared_loss=True,
        )

    def solver_config(self) -> GWConfig:
        return GWConfig(
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            max_outer_iterations=self.max_outer_iterations,
            min_outer_iterations=self.min_outer_iterations,
            max_inner_iterations=self.max_inner_iterations,
            convergence_threshold=self.convergence_threshold,
            relative_objective_threshold=self.relative_objective_threshold,
            use_squared_loss=self.use_squared_loss,
        )


@dataclass(frozen=True)
class Config:
    include_fixtures: bool
    thresholds: Thresholds
    gromov_wasserstein: GromovWassersteinConfig

    @staticmethod
    def default() -> "Config":
        return Config(
            include_fixtures=False,
            thresholds=Thresholds.standard(),
            gromov_wasserstein=GromovWassersteinConfig.standard(),
        )


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
class Fixtures:
    gromov_wasserstein: GromovWassersteinFixture
    traversal_coherence: TraversalCoherenceFixture
    path_signature: PathSignatureFixture


@dataclass(frozen=True)
class Report:
    suite_version: str
    timestamp: datetime
    passed: bool
    config: Config
    gromov_wasserstein: GromovWassersteinValidation
    traversal_coherence: TraversalCoherenceValidation
    path_signature: PathSignatureValidation
    fixtures: Optional[Fixtures]


class GeometryValidationSuite:
    @staticmethod
    def run(config: Config | None = None) -> Report:
        resolved = config or Config.default()
        fixtures = GeometryValidationSuite._build_fixtures()
        gw_validation = GeometryValidationSuite._validate_gromov_wasserstein(
            fixture=fixtures.gromov_wasserstein,
            config=resolved.gromov_wasserstein,
            thresholds=resolved.thresholds,
        )
        traversal_validation = GeometryValidationSuite._validate_traversal_coherence(
            fixture=fixtures.traversal_coherence,
            thresholds=resolved.thresholds,
        )
        path_validation = GeometryValidationSuite._validate_path_signature(
            fixture=fixtures.path_signature,
            thresholds=resolved.thresholds,
        )

        passed = gw_validation.passed and traversal_validation.passed and path_validation.passed

        return Report(
            suite_version=SUITE_VERSION,
            timestamp=datetime.utcnow(),
            passed=passed,
            config=resolved,
            gromov_wasserstein=gw_validation,
            traversal_coherence=traversal_validation,
            path_signature=path_validation,
            fixtures=fixtures if resolved.include_fixtures else None,
        )

    @staticmethod
    def _build_fixtures() -> Fixtures:
        points_a = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        permutation = [2, 0, 1]
        points_b = [points_a[idx] for idx in permutation]
        source_distances = GromovWassersteinDistance.compute_pairwise_distances(points_a)
        target_distances = GromovWassersteinDistance.compute_pairwise_distances(points_b)
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
            1, 0, -1, 0,
            0, 1, 0, -1,
            -1, 0, 1, 0,
            0, -1, 0, 1,
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

        return Fixtures(
            gromov_wasserstein=gw_fixture,
            traversal_coherence=traversal_fixture,
            path_signature=path_fixture,
        )

    @staticmethod
    def _validate_gromov_wasserstein(
        fixture: GromovWassersteinFixture,
        config: GromovWassersteinConfig,
        thresholds: Thresholds,
    ) -> GromovWassersteinValidation:
        solver_config = config.solver_config()

        identity = GromovWassersteinDistance.compute(
            source_distances=fixture.source_distances,
            target_distances=fixture.source_distances,
            config=solver_config,
        )
        permuted = GromovWassersteinDistance.compute(
            source_distances=fixture.source_distances,
            target_distances=fixture.target_distances,
            config=solver_config,
        )
        symmetry_forward = GromovWassersteinDistance.compute(
            source_distances=fixture.symmetry_source_distances,
            target_distances=fixture.symmetry_target_distances,
            config=solver_config,
        )
        symmetry_reverse = GromovWassersteinDistance.compute(
            source_distances=fixture.symmetry_target_distances,
            target_distances=fixture.symmetry_source_distances,
            config=solver_config,
        )
        symmetry_delta = abs(symmetry_forward.distance - symmetry_reverse.distance)
        row_error, column_error = GeometryValidationSuite._coupling_mass_errors(permuted.coupling)

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

    @staticmethod
    def _coupling_mass_errors(coupling: list[list[float]]) -> tuple[float, float]:
        n = len(coupling)
        m = len(coupling[0]) if coupling else 0
        if n <= 0 or m <= 0:
            return float("inf"), float("inf")

        expected_row = 1.0 / float(n)
        expected_col = 1.0 / float(m)

        max_row_error = max(abs(sum(row) - expected_row) for row in coupling)

        max_col_error = 0.0
        for j in range(m):
            col_sum = 0.0
            for i in range(n):
                col_sum += coupling[i][j]
            max_col_error = max(max_col_error, abs(col_sum - expected_col))

        return max_row_error, max_col_error

    @staticmethod
    def _validate_traversal_coherence(
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
        perturbed_corr = perturbed_result.transition_gram_correlation if perturbed_result else float("nan")
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

    @staticmethod
    def _validate_path_signature(
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

        passed = similarity >= thresholds.signature_similarity_min and frechet.distance <= thresholds.frechet_distance_max

        return PathSignatureValidation(
            signature_similarity=float(similarity),
            signed_area=float(signature.signed_area),
            signature_norm=float(signature.signature_norm),
            frechet_distance=float(frechet.distance),
            passed=passed,
        )
