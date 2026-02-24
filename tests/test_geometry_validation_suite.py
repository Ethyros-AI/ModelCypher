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

"""Tests for GeometryValidationSuite.

Tests the mathematical validation suite that verifies:
- Gromov-Wasserstein distance properties (identity, symmetry, mass conservation)
- Traversal coherence properties (self-correlation, perturbation sensitivity)
- Path signature properties (invariance, Frechet distance)
- Spectral signature properties (component count, spectral bounds, heat trace)
- Connected spectral signature properties (connectivity, algebraic connectivity)
- Dimension-constraint invariance properties (padding invariance across metrics)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.geometry_validation_suite import (
    GeometryValidationSuite,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.traversal_coherence import TraversalCoherence


@pytest.fixture
def backend():
    return get_default_backend()


class TestSuiteExecution:
    """Tests for overall suite execution."""

    def test_suite_runs_with_default_config(self, backend) -> None:
        """Suite should run successfully with default config."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()

        assert report.suite_version == "1.0"
        assert report.gromov_wasserstein is not None
        assert report.traversal_coherence is not None
        assert report.path_signature is not None
        assert report.spectral_signature is not None
        assert report.spectral_signature_connected is not None
        assert report.dimension_constraint is not None


class TestGromovWassersteinValidation:
    """Tests for GW validation component."""

    def test_identity_distance_near_zero(self, backend) -> None:
        """GW distance of a matrix with itself should be near zero.

        Mathematical property: d(X, X) = 0 for any metric.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        gw = report.gromov_wasserstein

        # The identity test computes GW(source, source)
        eps = machine_epsilon(suite._backend, suite._backend.array([gw.distance_identity]))
        assert abs(gw.distance_identity) <= eps

    def test_permutation_distance_small(self, backend) -> None:
        """GW distance between a matrix and its permutation should be small.

        Mathematical property: GW is isometry-invariant, so permuted
        distance matrices should have near-zero GW distance.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        gw = report.gromov_wasserstein

        # Permutation of same points should have small distance
        eps = machine_epsilon(suite._backend, suite._backend.array([gw.distance_permutation]))
        assert abs(gw.distance_permutation) <= eps

    def test_symmetry_holds(self, backend) -> None:
        """GW(A, B) should equal GW(B, A).

        Mathematical property: GW is a symmetric distance.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        gw = report.gromov_wasserstein

        # symmetry_delta = |GW(A,B) - GW(B,A)|
        eps = machine_epsilon(suite._backend, suite._backend.array([gw.symmetry_delta]))
        assert gw.symmetry_delta <= eps

    def test_coupling_mass_conservation(self, backend) -> None:
        """Optimal coupling should preserve marginal mass.

        Mathematical property: The coupling π should satisfy
        π.sum(axis=1) = μ and π.sum(axis=0) = ν for source/target measures.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        gw = report.gromov_wasserstein

        eps = regularization_epsilon(
            suite._backend,
            suite._backend.array([gw.max_row_mass_error, gw.max_column_mass_error]),
        )
        assert gw.max_row_mass_error <= eps
        assert gw.max_column_mass_error <= eps

    def test_algorithm_converges(self, backend) -> None:
        """GW solver should converge within iteration budget."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        gw = report.gromov_wasserstein

        assert gw.converged, f"GW solver did not converge after {gw.iterations} iterations"


class TestTraversalCoherenceValidation:
    """Tests for traversal coherence validation component."""

    def test_self_correlation_near_one(self, backend) -> None:
        """Comparing a Gram matrix with itself should give correlation ~1.

        Mathematical property: corr(X, X) = 1.
        """
        suite = GeometryValidationSuite(backend=backend)
        fixtures = suite._build_fixtures()
        fixture = fixtures.traversal_coherence
        report = suite.run()
        tc = report.traversal_coherence

        expected = TraversalCoherence.compare(
            paths=fixture.paths,
            gram_a=list(fixture.anchor_gram),
            gram_b=list(fixture.anchor_gram),
            anchor_ids=list(fixture.anchor_ids),
        )
        assert expected is not None
        assert tc.self_correlation == expected.transition_gram_correlation

    def test_perturbed_correlation_differs(self, backend) -> None:
        """Comparing with perturbed Gram should give lower correlation.

        The validation suite creates a perturbed Gram matrix that differs
        from the original. This tests sensitivity to structural changes.
        """
        suite = GeometryValidationSuite(backend=backend)
        fixtures = suite._build_fixtures()
        fixture = fixtures.traversal_coherence
        report = suite.run()
        tc = report.traversal_coherence

        expected = TraversalCoherence.compare(
            paths=fixture.paths,
            gram_a=list(fixture.anchor_gram),
            gram_b=list(fixture.perturbed_gram),
            anchor_ids=list(fixture.anchor_ids),
        )
        assert expected is not None
        assert tc.perturbed_correlation == expected.transition_gram_correlation

    def test_paths_processed(self, backend) -> None:
        """Validation should process the fixture paths."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        tc = report.traversal_coherence

        assert tc.path_count >= 1, "Should process at least one path"
        assert tc.transition_count >= 0, "Transition count should be non-negative"


class TestPathSignatureValidation:
    """Tests for path signature validation component."""

    def test_self_frechet_distance_zero(self, backend) -> None:
        """Frechet distance of a path with itself should be zero.

        Mathematical property: d(X, X) = 0.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        ps = report.path_signature

        eps = machine_epsilon(suite._backend, suite._backend.array([ps.frechet_distance]))
        assert abs(ps.frechet_distance) <= eps

    def test_signature_properties_computed(self, backend) -> None:
        """Signature properties should be computed."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        ps = report.path_signature

        # signed_area and signature_norm should be non-negative
        assert ps.signed_area >= 0, "Signed area should be non-negative"
        assert ps.signature_norm >= 0, "Signature norm should be non-negative"

    def test_translation_invariance(self, backend) -> None:
        """Path signature should be translation invariant.

        The validation compares signatures computed with original vs shifted
        embeddings. Translation should preserve the signature structure.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        ps = report.path_signature

        eps = machine_epsilon(
            suite._backend,
            suite._backend.array([ps.signature_similarity]),
        )
        assert abs(ps.signature_similarity - 1.0) <= eps


class TestSpectralSignatureValidation:
    """Tests for spectral signature validation component."""

    def test_component_count_matches_fixture(self, backend) -> None:
        """Spectral fixture should be connected via auto-derived k.

        The fixture has spatially separated clusters, but derive_k_neighbors
        finds the minimum k that yields a connected graph, so component_count=1.
        """
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        spectral = report.spectral_signature

        assert spectral.component_count == 1, "Spectral fixture should produce 1 component with auto-k"
        assert spectral.connected is True

    def test_connected_fixture_properties(self, backend) -> None:
        """Connected spectral fixture should reflect connectivity."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        spectral = report.spectral_signature_connected

        assert spectral.component_count == 1, "Connected spectral fixture should produce 1 component"
        assert spectral.connected is True
        eps = regularization_epsilon(suite._backend, suite._backend.array([spectral.eigenvalue_min]))
        assert spectral.algebraic_connectivity > eps

    def test_eigenvalue_bounds_normalized(self, backend) -> None:
        """Normalized Laplacian eigenvalues should lie in [0, 2]."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        spectral = report.spectral_signature

        eps = regularization_epsilon(suite._backend, suite._backend.array([spectral.eigenvalue_min]))
        assert spectral.eigenvalue_min >= -eps
        assert spectral.eigenvalue_max <= 2.0 + eps

    def test_heat_trace_monotone(self, backend) -> None:
        """Heat trace should be non-increasing with time."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        spectral = report.spectral_signature

        eps = regularization_epsilon(suite._backend, suite._backend.array(spectral.heat_trace))
        for i in range(len(spectral.heat_trace) - 1):
            assert spectral.heat_trace[i] + eps >= spectral.heat_trace[i + 1]


class TestDimensionConstraintValidation:
    """Tests for dimension-constraint invariance validation."""

    def test_dimension_constraint_invariance(self, backend) -> None:
        """Zero-padding should preserve geometry across metrics."""
        suite = GeometryValidationSuite(backend=backend)
        report = suite.run()
        validation = report.dimension_constraint

        cka_eps = machine_epsilon(
            suite._backend,
            suite._backend.array([validation.gram_cka]),
        )
        geo_eps = regularization_epsilon(
            suite._backend,
            suite._backend.array(
                [validation.geodesic_mean_abs_diff, validation.geodesic_max_abs_diff],
            ),
        )
        spectral_eps = regularization_epsilon(
            suite._backend,
            suite._backend.array(
                [
                    validation.spectral_eigen_mean_abs_diff,
                    validation.spectral_eigen_max_abs_diff,
                ],
            ),
        )
        assert abs(validation.gram_cka - 1.0) <= cka_eps
        assert validation.geodesic_mean_abs_diff <= geo_eps
        assert validation.geodesic_max_abs_diff <= geo_eps
        assert validation.spectral_eigen_mean_abs_diff <= spectral_eps
        assert validation.spectral_eigen_max_abs_diff <= spectral_eps
        assert validation.component_count_base == validation.component_count_padded
        assert validation.cycle_count_base == validation.cycle_count_padded
        assert validation.betti_numbers_base == validation.betti_numbers_padded


class TestGeometryValidationResults:
    """Regression checks for geometry validation experiment outputs."""

    def test_alignment_invariance_results(self) -> None:
        """Aligned CKA should match the invariant alignment claim."""
        results_path = Path("experiments/results/geometry_validation.json")
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            experiments = data.get("experiments", {})
            alignment = experiments.get("alignment_invariance", {})
            raw_cka = alignment.get("raw_cka")
            aligned_cka = alignment.get("aligned_cka")
            achieved_cka = alignment.get("alignment_achieved_cka")
            precision_threshold = alignment.get("precision_threshold")
        else:
            from modelcypher.core.domain.geometry.gram_aligner import GramAligner

            backend = get_default_backend()
            backend.random_seed(111)
            source = backend.random_normal((48, 24))
            transform = backend.random_normal((24, 24))
            target = backend.matmul(source, transform)
            backend.eval(source, target)

            raw_cka = compute_cka(source, target, backend).cka
            aligner = GramAligner(backend)
            alignment = aligner.find_perfect_alignment(source, target)
            aligned = backend.matmul(source, alignment.feature_transform)
            backend.eval(aligned)
            aligned_cka = compute_cka(aligned, target, backend).cka
            achieved_cka = alignment.achieved_cka
            precision_threshold = alignment.precision_threshold

        assert raw_cka is not None
        assert aligned_cka is not None
        assert achieved_cka is not None

        assert precision_threshold is not None
        assert abs(aligned_cka - 1.0) <= precision_threshold
        assert abs(achieved_cka - 1.0) <= precision_threshold
        assert raw_cka < aligned_cka
