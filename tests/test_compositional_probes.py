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

"""Tests for compositional_probes.py.

Tests cover:
- STANDARD_PROBES constant validation
- analyze_composition() with various inputs
- check_consistency() with matching/mismatching inputs
- analyze_all_probes() with complete/partial embeddings
- Edge cases: empty inputs, single components, high dimensions
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.compositional_probes import CompositionalProbes
from modelcypher.core.domain.geometry.types import (
    CompositionAnalysis,
    CompositionCategory,
    CompositionProbe,
)


def _scalar_tol():
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


# =============================================================================
# STANDARD_PROBES Tests
# =============================================================================


class TestStandardProbes:
    """Tests for STANDARD_PROBES constant."""

    def test_standard_probes_not_empty(self):
        """STANDARD_PROBES contains probes."""
        assert len(CompositionalProbes.STANDARD_PROBES) > 0

    def test_standard_probes_all_valid_types(self):
        """All items in STANDARD_PROBES are CompositionProbe."""
        for probe in CompositionalProbes.STANDARD_PROBES:
            assert isinstance(probe, CompositionProbe)
            assert isinstance(probe.phrase, str)
            assert isinstance(probe.components, tuple)
            assert isinstance(probe.category, CompositionCategory)

    def test_standard_probes_has_all_categories(self):
        """STANDARD_PROBES covers multiple categories."""
        categories = {probe.category for probe in CompositionalProbes.STANDARD_PROBES}
        # Should have at least mental, action, evaluative
        assert CompositionCategory.MENTAL_PREDICATE in categories
        assert CompositionCategory.ACTION in categories
        assert CompositionCategory.EVALUATIVE in categories

    def test_standard_probes_components_non_empty(self):
        """All probes have at least one component."""
        for probe in CompositionalProbes.STANDARD_PROBES:
            assert len(probe.components) >= 1


# =============================================================================
# analyze_composition Tests
# =============================================================================


class TestAnalyzeComposition:
    """Tests for analyze_composition method."""

    def test_analyze_composition_basic(self):
        """Basic analysis returns valid CompositionAnalysis."""
        probe = CompositionProbe("I WANT", ("I", "WANT"), CompositionCategory.MENTAL_PREDICATE)
        components = [[1.0, 0.0], [0.0, 1.0]]
        composition = [0.5, 0.5]
        analysis = CompositionalProbes.analyze_composition(composition, components, probe)

        tol = _scalar_tol()
        assert abs(analysis.barycentric_weights[0] - 0.5) <= tol
        assert abs(analysis.barycentric_weights[1] - 0.5) <= tol
        assert abs(analysis.residual_norm) <= tol
        assert analysis.is_compositional is True

    def test_analyze_composition_returns_correct_probe(self):
        """Analysis result contains the input probe."""
        probe = CompositionProbe("TEST", ("A", "B"), CompositionCategory.ACTION)
        analysis = CompositionalProbes.analyze_composition([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], probe)
        assert analysis.probe == probe

    def test_analyze_composition_single_component(self):
        """Analysis works with single component."""
        probe = CompositionProbe("SINGLE", ("A",), CompositionCategory.EVALUATIVE)
        comp = [1.0, 0.0, 0.0]
        analysis = CompositionalProbes.analyze_composition(comp, [comp], probe)
        # Single component matching composition should have weight ~1
        assert len(analysis.barycentric_weights) == 1
        assert analysis.residual_norm < 0.1  # Should reconstruct well

    def test_analyze_composition_empty_components(self):
        """Analysis handles empty component list gracefully."""
        probe = CompositionProbe("EMPTY", (), CompositionCategory.TEMPORAL)
        analysis = CompositionalProbes.analyze_composition([1.0, 0.0], [], probe)
        # Empty components returns default analysis with inf residual
        assert analysis.residual_norm == float("inf")

    def test_analyze_composition_high_dimensional(self):
        """Analysis works with high-dimensional vectors."""
        probe = CompositionProbe("HIGH_DIM", ("A", "B"), CompositionCategory.SPATIAL)
        dim = 768  # Typical embedding dimension
        import math
        # Normalized orthogonal vectors
        a = [1.0 / math.sqrt(dim)] * dim
        b = [0.0] * dim
        b[0] = 1.0
        composition = [(a[i] + b[i]) / 2 for i in range(dim)]
        
        analysis = CompositionalProbes.analyze_composition(composition, [a, b], probe)
        assert len(analysis.barycentric_weights) == 2
        assert len(analysis.component_angles) == 2

    def test_analyze_composition_with_2d_input(self):
        """Analysis handles 2D composition input (batch dimension)."""
        probe = CompositionProbe("TEST", ("A",), CompositionCategory.ACTION)
        # Composition as 2D array with batch dim of 1
        comp_2d = [[1.0, 0.0]]
        components = [[1.0, 0.0]]
        analysis = CompositionalProbes.analyze_composition(comp_2d, components, probe)
        assert len(analysis.barycentric_weights) == 1


# =============================================================================
# check_consistency Tests
# =============================================================================


class TestCheckConsistency:
    """Tests for check_consistency method."""

    def test_check_consistency_identical(self):
        """Check consistency returns raw measurements."""
        probe = CompositionProbe("I WANT", ("I", "WANT"), CompositionCategory.MENTAL_PREDICATE)
        analysis = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe)
        result = CompositionalProbes.check_consistency([analysis], [analysis])

        # Raw measurements. The numbers ARE the answer.
        tol = _scalar_tol()
        assert abs(result.barycentric_correlation - 1.0) <= tol
        assert abs(result.angular_correlation - 1.0) <= tol
        assert abs(result.consistency_score - 1.0) <= tol

    def test_check_consistency_empty_lists(self):
        """Consistency check handles empty lists."""
        result = CompositionalProbes.check_consistency([], [])
        assert result.probe_count == 0
        assert result.consistency_score == 0.0

    def test_check_consistency_mismatched_lengths(self):
        """Consistency check handles mismatched list lengths."""
        probe = CompositionProbe("TEST", ("A",), CompositionCategory.EVALUATIVE)
        a1 = CompositionalProbes.analyze_composition([1.0, 0.0], [[1.0, 0.0]], probe)
        a2 = CompositionalProbes.analyze_composition([0.0, 1.0], [[0.0, 1.0]], probe)
        
        result = CompositionalProbes.check_consistency([a1], [a1, a2])
        assert result.probe_count == 0

    def test_check_consistency_multiple_probes(self):
        """Consistency check works with multiple probes."""
        probe1 = CompositionProbe("P1", ("A", "B"), CompositionCategory.ACTION)
        probe2 = CompositionProbe("P2", ("C", "D"), CompositionCategory.MENTAL_PREDICATE)
        
        a1 = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe1)
        a2 = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe2)
        
        result = CompositionalProbes.check_consistency([a1, a2], [a1, a2])
        assert result.probe_count == 2


# =============================================================================
# _geodesic_correlation Tests
# =============================================================================


class TestGeodesicCorrelation:
    """Tests for _geodesic_correlation helper method."""

    def test_geodesic_correlation_identical_vectors(self):
        """Identical vectors have correlation 1.0."""
        a = [1.0, 2.0, 3.0, 4.0]
        result = CompositionalProbes._geodesic_correlation(a, a)
        assert abs(result - 1.0) < 0.01

    def test_geodesic_correlation_short_vectors(self):
        """Short vectors (len < 2) return 0.0."""
        result = CompositionalProbes._geodesic_correlation([1.0], [2.0])
        assert result == 0.0

    def test_geodesic_correlation_empty_vectors(self):
        """Empty vectors return 0.0."""
        result = CompositionalProbes._geodesic_correlation([], [])
        assert result == 0.0


# =============================================================================
# analyze_all_probes Tests
# =============================================================================


class TestAnalyzeAllProbes:
    """Tests for analyze_all_probes method."""

    def test_analyze_all_probes_custom(self):
        """Custom probes are analyzed correctly."""
        probe = CompositionProbe("TEST", ("A", "B"), CompositionCategory.RELATIONAL)
        prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
        composition_embeddings = {"TEST": [0.5, 0.5]}
        analyses = CompositionalProbes.analyze_all_probes(
            prime_embeddings=prime_embeddings,
            composition_embeddings=composition_embeddings,
            probes=[probe],
        )
        assert len(analyses) == 1

    def test_analyze_all_probes_missing_composition(self):
        """Probes with missing composition embedding are skipped."""
        probe = CompositionProbe("MISSING", ("A", "B"), CompositionCategory.ACTION)
        prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
        composition_embeddings = {}  # No composition embedding
        
        analyses = CompositionalProbes.analyze_all_probes(
            prime_embeddings=prime_embeddings,
            composition_embeddings=composition_embeddings,
            probes=[probe],
        )
        assert len(analyses) == 0

    def test_analyze_all_probes_missing_component(self):
        """Probes with missing component embedding are skipped."""
        probe = CompositionProbe("TEST", ("A", "B", "MISSING"), CompositionCategory.ACTION)
        prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0]}  # Missing "MISSING"
        composition_embeddings = {"TEST": [0.5, 0.5]}
        
        analyses = CompositionalProbes.analyze_all_probes(
            prime_embeddings=prime_embeddings,
            composition_embeddings=composition_embeddings,
            probes=[probe],
        )
        assert len(analyses) == 0

    def test_analyze_all_probes_uses_standard_probes_by_default(self):
        """Without explicit probes, uses STANDARD_PROBES."""
        # Create embeddings for some standard probes
        prime_embeddings = {
            "I": [1.0, 0.0],
            "THINK": [0.0, 1.0],
        }
        composition_embeddings = {
            "I THINK": [0.5, 0.5],
        }
        
        analyses = CompositionalProbes.analyze_all_probes(
            prime_embeddings=prime_embeddings,
            composition_embeddings=composition_embeddings,
            probes=None,  # Use default
        )
        # Should find "I THINK" in STANDARD_PROBES
        assert len(analyses) >= 1
        assert any(a.probe.phrase == "I THINK" for a in analyses)

    def test_analyze_all_probes_multiple_probes(self):
        """Multiple probes are all analyzed."""
        probes = [
            CompositionProbe("P1", ("A", "B"), CompositionCategory.ACTION),
            CompositionProbe("P2", ("C", "D"), CompositionCategory.MENTAL_PREDICATE),
        ]
        prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.5, 0.5], "D": [0.5, -0.5]}
        composition_embeddings = {"P1": [0.5, 0.5], "P2": [0.5, 0.0]}
        
        analyses = CompositionalProbes.analyze_all_probes(
            prime_embeddings=prime_embeddings,
            composition_embeddings=composition_embeddings,
            probes=probes,
        )
        assert len(analyses) == 2

