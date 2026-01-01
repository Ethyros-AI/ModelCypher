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

import pytest

from modelcypher.core.domain.geometry.geometry_validation_suite import (
    Config,
    GeometryValidationSuite,
    Thresholds,
)
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


class TestSuiteExecution:
    """Tests for overall suite execution."""

    def test_suite_runs_with_default_config(self) -> None:
        """Suite should run successfully with default config."""
        suite = GeometryValidationSuite()
        report = suite.run()

        assert report.suite_version == "1.0"
        assert report.gromov_wasserstein is not None
        assert report.traversal_coherence is not None
        assert report.path_signature is not None
        assert report.spectral_signature is not None
        assert report.spectral_signature_connected is not None
        assert report.dimension_constraint is not None

    def test_suite_reports_pass_status(self) -> None:
        """Suite should report overall pass/fail status correctly."""
        suite = GeometryValidationSuite()
        report = suite.run()

        # Overall pass should be AND of all component passes
        expected_pass = (
            report.gromov_wasserstein.passed
            and report.traversal_coherence.passed
            and report.path_signature.passed
            and report.spectral_signature.passed
            and report.spectral_signature_connected.passed
            and report.dimension_constraint.passed
        )
        assert report.passed == expected_pass

    def test_suite_with_fixtures_included(self) -> None:
        """Suite can include fixtures in report for debugging."""
        config = Config(
            include_fixtures=True,
            thresholds=Thresholds.standard(),
        )
        suite = GeometryValidationSuite()
        report = suite.run(config)

        assert report.fixtures is not None
        assert report.fixtures.gromov_wasserstein is not None
        assert report.fixtures.traversal_coherence is not None
        assert report.fixtures.path_signature is not None
        assert report.fixtures.spectral_signature is not None
        assert report.fixtures.spectral_signature_connected is not None
        assert report.fixtures.dimension_constraint is not None


class TestGromovWassersteinValidation:
    """Tests for GW validation component."""

    def test_identity_distance_near_zero(self) -> None:
        """GW distance of a matrix with itself should be near zero.

        Mathematical property: d(X, X) = 0 for any metric.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        gw = report.gromov_wasserstein

        # The identity test computes GW(source, source)
        assert gw.distance_identity < 1e-6, (
            f"Identity distance {gw.distance_identity} should be near zero"
        )

    def test_permutation_distance_small(self) -> None:
        """GW distance between a matrix and its permutation should be small.

        Mathematical property: GW is isometry-invariant, so permuted
        distance matrices should have near-zero GW distance.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        gw = report.gromov_wasserstein

        # Permutation of same points should have small distance
        assert gw.distance_permutation < 0.02, (
            f"Permutation distance {gw.distance_permutation} too large"
        )

    def test_symmetry_holds(self) -> None:
        """GW(A, B) should equal GW(B, A).

        Mathematical property: GW is a symmetric distance.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        gw = report.gromov_wasserstein

        # symmetry_delta = |GW(A,B) - GW(B,A)|
        assert gw.symmetry_delta < 1e-3, f"Symmetry delta {gw.symmetry_delta} too large"

    def test_coupling_mass_conservation(self) -> None:
        """Optimal coupling should preserve marginal mass.

        Mathematical property: The coupling π should satisfy
        π.sum(axis=1) = μ and π.sum(axis=0) = ν for source/target measures.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        gw = report.gromov_wasserstein

        assert gw.max_row_mass_error < 0.02, f"Row mass error {gw.max_row_mass_error} too large"
        assert gw.max_column_mass_error < 0.02, (
            f"Column mass error {gw.max_column_mass_error} too large"
        )

    def test_algorithm_converges(self) -> None:
        """GW solver should converge within iteration budget."""
        suite = GeometryValidationSuite()
        report = suite.run()
        gw = report.gromov_wasserstein

        assert gw.converged, f"GW solver did not converge after {gw.iterations} iterations"


class TestTraversalCoherenceValidation:
    """Tests for traversal coherence validation component."""

    def test_self_correlation_near_one(self) -> None:
        """Comparing a Gram matrix with itself should give correlation ~1.

        Mathematical property: corr(X, X) = 1.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        tc = report.traversal_coherence

        assert tc.self_correlation >= 0.999, (
            f"Self correlation {tc.self_correlation} should be ~1.0"
        )

    def test_perturbed_correlation_differs(self) -> None:
        """Comparing with perturbed Gram should give lower correlation.

        The validation suite creates a perturbed Gram matrix that differs
        from the original. This tests sensitivity to structural changes.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        tc = report.traversal_coherence

        # Perturbed should be noticeably different from self
        assert tc.perturbed_correlation < tc.self_correlation, (
            "Perturbed correlation should be lower than self correlation"
        )

    def test_paths_processed(self) -> None:
        """Validation should process the fixture paths."""
        suite = GeometryValidationSuite()
        report = suite.run()
        tc = report.traversal_coherence

        assert tc.path_count >= 1, "Should process at least one path"
        assert tc.transition_count >= 0, "Transition count should be non-negative"


class TestPathSignatureValidation:
    """Tests for path signature validation component."""

    def test_self_frechet_distance_zero(self) -> None:
        """Frechet distance of a path with itself should be zero.

        Mathematical property: d(X, X) = 0.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        ps = report.path_signature

        assert ps.frechet_distance == pytest.approx(0.0, abs=1e-5), (
            f"Self Frechet distance {ps.frechet_distance} should be zero"
        )

    def test_signature_properties_computed(self) -> None:
        """Signature properties should be computed."""
        suite = GeometryValidationSuite()
        report = suite.run()
        ps = report.path_signature

        # signed_area and signature_norm should be non-negative
        assert ps.signed_area >= 0, "Signed area should be non-negative"
        assert ps.signature_norm >= 0, "Signature norm should be non-negative"

    def test_translation_invariance(self) -> None:
        """Path signature should be translation invariant.

        The validation compares signatures computed with original vs shifted
        embeddings. Translation should preserve the signature structure.
        """
        suite = GeometryValidationSuite()
        report = suite.run()
        ps = report.path_signature

        # Similarity should be high for translated embeddings
        # (The fixture uses shifted_embeddings which are translations)
        assert ps.signature_similarity >= 0.999, (
            f"Signature similarity {ps.signature_similarity} should be ~1.0 for translations"
        )


class TestSpectralSignatureValidation:
    """Tests for spectral signature validation component."""

    def test_component_count_matches_fixture(self) -> None:
        """Spectral fixture should reflect disconnected components."""
        suite = GeometryValidationSuite()
        report = suite.run()
        spectral = report.spectral_signature

        assert spectral.component_count == 2, "Spectral fixture should produce 2 components"
        assert spectral.connected is False

    def test_connected_fixture_properties(self) -> None:
        """Connected spectral fixture should reflect connectivity."""
        suite = GeometryValidationSuite()
        report = suite.run()
        spectral = report.spectral_signature_connected

        assert spectral.component_count == 1, "Connected spectral fixture should produce 1 component"
        assert spectral.connected is True
        eps = regularization_epsilon(suite._backend, suite._backend.array([spectral.eigenvalue_min]))
        assert spectral.algebraic_connectivity > eps

    def test_eigenvalue_bounds_normalized(self) -> None:
        """Normalized Laplacian eigenvalues should lie in [0, 2]."""
        suite = GeometryValidationSuite()
        report = suite.run()
        spectral = report.spectral_signature

        eps = regularization_epsilon(suite._backend, suite._backend.array([spectral.eigenvalue_min]))
        assert spectral.eigenvalue_min >= -eps
        assert spectral.eigenvalue_max <= 2.0 + eps

    def test_heat_trace_monotone(self) -> None:
        """Heat trace should be non-increasing with time."""
        suite = GeometryValidationSuite()
        report = suite.run()
        spectral = report.spectral_signature

        eps = regularization_epsilon(suite._backend, suite._backend.array(spectral.heat_trace))
        for i in range(len(spectral.heat_trace) - 1):
            assert spectral.heat_trace[i] + eps >= spectral.heat_trace[i + 1]


class TestDimensionConstraintValidation:
    """Tests for dimension-constraint invariance validation."""

    def test_dimension_constraint_invariance(self) -> None:
        """Zero-padding should preserve geometry across metrics."""
        suite = GeometryValidationSuite()
        report = suite.run()
        validation = report.dimension_constraint

        assert validation.gram_cka >= 0.999999
        assert validation.geodesic_mean_abs_diff <= 1e-6
        assert validation.geodesic_max_abs_diff <= 1e-6
        assert validation.spectral_eigen_mean_abs_diff <= 1e-6
        assert validation.spectral_eigen_max_abs_diff <= 1e-6
        assert validation.component_count_base == validation.component_count_padded
        assert validation.cycle_count_base == validation.cycle_count_padded
        assert validation.betti_numbers_base == validation.betti_numbers_padded


class TestThresholds:
    """Tests for validation thresholds."""

    def test_standard_thresholds_are_reasonable(self) -> None:
        """Standard thresholds should be numerically reasonable."""
        thresholds = Thresholds.standard()

        assert thresholds.identity_distance_max > 0, "Identity threshold must be positive"
        assert thresholds.identity_distance_max < 1e-3, "Identity threshold should be tight"

        assert thresholds.symmetry_delta_max > 0, "Symmetry threshold must be positive"
        assert thresholds.symmetry_delta_max < 0.01, "Symmetry threshold should be tight"

        assert thresholds.traversal_self_correlation_min > 0.9, (
            "Self correlation threshold should be near 1.0"
        )
        assert thresholds.dimension_constraint_cka_min > 0.99, (
            "Dimension constraint CKA threshold should be near 1.0"
        )
        assert thresholds.dimension_constraint_geodesic_max_abs_diff_max < 1e-3, (
            "Dimension constraint geodesic threshold should be tight"
        )
        assert thresholds.dimension_constraint_spectral_eigen_max_abs_diff_max < 1e-3, (
            "Dimension constraint spectral threshold should be tight"
        )

    def test_custom_thresholds_affect_pass_status(self) -> None:
        """Custom thresholds should affect validation pass/fail."""
        # Create impossibly tight thresholds
        tight_thresholds = Thresholds(
            identity_distance_max=1e-20,  # Impossible
            permutation_distance_max=1e-20,
            symmetry_delta_max=1e-20,
            coupling_mass_error_max=1e-20,
            traversal_self_correlation_min=1.0001,  # Impossible (>1)
            traversal_perturbed_correlation_max=0.0,
            signature_similarity_min=1.0001,
            frechet_distance_max=1e-20,
            dimension_constraint_cka_min=1.1,
            dimension_constraint_geodesic_mean_abs_diff_max=-1.0,
            dimension_constraint_geodesic_max_abs_diff_max=-1.0,
            dimension_constraint_spectral_eigen_mean_abs_diff_max=-1.0,
            dimension_constraint_spectral_eigen_max_abs_diff_max=-1.0,
            dimension_constraint_spectral_entropy_abs_diff_max=-1.0,
            dimension_constraint_heat_trace_max_abs_diff_max=-1.0,
            dimension_constraint_topology_abs_diff_max=-1.0,
        )
        config = Config(
            include_fixtures=False,
            thresholds=tight_thresholds,
        )
        suite = GeometryValidationSuite()
        report = suite.run(config)

        # With impossible thresholds, validation should fail
        assert not report.passed, "Impossible thresholds should cause failure"
