"""Tests for GeometryValidationSuite.

Tests the mathematical validation suite that verifies:
- Gromov-Wasserstein distance properties (identity, symmetry, mass conservation)
- Traversal coherence properties (self-correlation, perturbation sensitivity)
- Path signature properties (invariance, Frechet distance)
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.geometry_validation_suite import (
    GeometryValidationSuite,
    Config,
    Thresholds,
    GromovWassersteinConfig,
)


class TestSuiteExecution:
    """Tests for overall suite execution."""

    def test_suite_runs_with_default_config(self) -> None:
        """Suite should run successfully with default config."""
        report = GeometryValidationSuite.run()

        assert report.suite_version == "1.0"
        assert report.gromov_wasserstein is not None
        assert report.traversal_coherence is not None
        assert report.path_signature is not None

    def test_suite_reports_pass_status(self) -> None:
        """Suite should report overall pass/fail status correctly."""
        report = GeometryValidationSuite.run()

        # Overall pass should be AND of all component passes
        expected_pass = (
            report.gromov_wasserstein.passed
            and report.traversal_coherence.passed
            and report.path_signature.passed
        )
        assert report.passed == expected_pass

    def test_suite_with_fixtures_included(self) -> None:
        """Suite can include fixtures in report for debugging."""
        config = Config(
            include_fixtures=True,
            thresholds=Thresholds.standard(),
            gromov_wasserstein=GromovWassersteinConfig.standard(),
        )
        report = GeometryValidationSuite.run(config)

        assert report.fixtures is not None
        assert report.fixtures.gromov_wasserstein is not None
        assert report.fixtures.traversal_coherence is not None
        assert report.fixtures.path_signature is not None


class TestGromovWassersteinValidation:
    """Tests for GW validation component."""

    def test_identity_distance_near_zero(self) -> None:
        """GW distance of a matrix with itself should be near zero.

        Mathematical property: d(X, X) = 0 for any metric.
        """
        report = GeometryValidationSuite.run()
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
        report = GeometryValidationSuite.run()
        gw = report.gromov_wasserstein

        # Permutation of same points should have small distance
        assert gw.distance_permutation < 0.02, (
            f"Permutation distance {gw.distance_permutation} too large"
        )

    def test_symmetry_holds(self) -> None:
        """GW(A, B) should equal GW(B, A).

        Mathematical property: GW is a symmetric distance.
        """
        report = GeometryValidationSuite.run()
        gw = report.gromov_wasserstein

        # symmetry_delta = |GW(A,B) - GW(B,A)|
        assert gw.symmetry_delta < 1e-3, (
            f"Symmetry delta {gw.symmetry_delta} too large"
        )

    def test_coupling_mass_conservation(self) -> None:
        """Optimal coupling should preserve marginal mass.

        Mathematical property: The coupling π should satisfy
        π.sum(axis=1) = μ and π.sum(axis=0) = ν for source/target measures.
        """
        report = GeometryValidationSuite.run()
        gw = report.gromov_wasserstein

        assert gw.max_row_mass_error < 0.02, (
            f"Row mass error {gw.max_row_mass_error} too large"
        )
        assert gw.max_column_mass_error < 0.02, (
            f"Column mass error {gw.max_column_mass_error} too large"
        )

    def test_algorithm_converges(self) -> None:
        """GW solver should converge within iteration budget."""
        report = GeometryValidationSuite.run()
        gw = report.gromov_wasserstein

        assert gw.converged, f"GW solver did not converge after {gw.iterations} iterations"


class TestTraversalCoherenceValidation:
    """Tests for traversal coherence validation component."""

    def test_self_correlation_near_one(self) -> None:
        """Comparing a Gram matrix with itself should give correlation ~1.

        Mathematical property: corr(X, X) = 1.
        """
        report = GeometryValidationSuite.run()
        tc = report.traversal_coherence

        assert tc.self_correlation >= 0.999, (
            f"Self correlation {tc.self_correlation} should be ~1.0"
        )

    def test_perturbed_correlation_differs(self) -> None:
        """Comparing with perturbed Gram should give lower correlation.

        The validation suite creates a perturbed Gram matrix that differs
        from the original. This tests sensitivity to structural changes.
        """
        report = GeometryValidationSuite.run()
        tc = report.traversal_coherence

        # Perturbed should be noticeably different from self
        assert tc.perturbed_correlation < tc.self_correlation, (
            "Perturbed correlation should be lower than self correlation"
        )

    def test_paths_processed(self) -> None:
        """Validation should process the fixture paths."""
        report = GeometryValidationSuite.run()
        tc = report.traversal_coherence

        assert tc.path_count >= 1, "Should process at least one path"
        assert tc.transition_count >= 0, "Transition count should be non-negative"


class TestPathSignatureValidation:
    """Tests for path signature validation component."""

    def test_self_frechet_distance_zero(self) -> None:
        """Frechet distance of a path with itself should be zero.

        Mathematical property: d(X, X) = 0.
        """
        report = GeometryValidationSuite.run()
        ps = report.path_signature

        assert ps.frechet_distance == pytest.approx(0.0, abs=1e-5), (
            f"Self Frechet distance {ps.frechet_distance} should be zero"
        )

    def test_signature_properties_computed(self) -> None:
        """Signature properties should be computed."""
        report = GeometryValidationSuite.run()
        ps = report.path_signature

        # signed_area and signature_norm should be non-negative
        assert ps.signed_area >= 0, "Signed area should be non-negative"
        assert ps.signature_norm >= 0, "Signature norm should be non-negative"

    def test_translation_invariance(self) -> None:
        """Path signature should be translation invariant.

        The validation compares signatures computed with original vs shifted
        embeddings. Translation should preserve the signature structure.
        """
        report = GeometryValidationSuite.run()
        ps = report.path_signature

        # Similarity should be high for translated embeddings
        # (The fixture uses shifted_embeddings which are translations)
        assert ps.signature_similarity >= 0.999, (
            f"Signature similarity {ps.signature_similarity} should be ~1.0 for translations"
        )


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
        )
        config = Config(
            include_fixtures=False,
            thresholds=tight_thresholds,
            gromov_wasserstein=GromovWassersteinConfig.standard(),
        )
        report = GeometryValidationSuite.run(config)

        # With impossible thresholds, validation should fail
        assert not report.passed, "Impossible thresholds should cause failure"


class TestGromovWassersteinConfig:
    """Tests for GW solver configuration."""

    def test_standard_config_produces_solver_config(self) -> None:
        """Standard config should produce a valid solver config."""
        gw_config = GromovWassersteinConfig.standard()
        solver_config = gw_config.solver_config()

        assert solver_config.epsilon > 0
        assert solver_config.max_outer_iterations > 0
        assert solver_config.convergence_threshold > 0

    def test_epsilon_decay_configured(self) -> None:
        """Epsilon decay should be configured for annealing."""
        gw_config = GromovWassersteinConfig.standard()

        assert 0 < gw_config.epsilon_decay < 1, (
            "Epsilon decay should be in (0, 1) for annealing"
        )
        assert gw_config.epsilon_min < gw_config.epsilon, (
            "Epsilon min should be less than initial epsilon"
        )
