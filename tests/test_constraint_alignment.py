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

"""Tests for constraint-based alignment."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.constraint_alignment import (
    ConstraintAligner,
    ConstraintAlignmentResult,
    LayerCorrespondence,
    ProbeConstraint,
    diagnose_probe_conflict,
)


class TestProbeConstraint:
    """Tests for ProbeConstraint dataclass."""

    def test_frozen_dataclass(self):
        """ProbeConstraint should be immutable."""
        constraint = ProbeConstraint(
            probe_id="test",
            source_peak_layer=0,
            target_peak_layer=1,
            source_activation=1.0,
            target_activation=0.9,
            agreement_score=0.8,
        )
        with pytest.raises(AttributeError):
            constraint.probe_id = "modified"

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        constraint = ProbeConstraint(
            probe_id="probe1",
            source_peak_layer=2,
            target_peak_layer=3,
            source_activation=0.95,
            target_activation=0.92,
            agreement_score=0.88,
        )
        assert constraint.probe_id == "probe1"
        assert constraint.source_peak_layer == 2
        assert constraint.target_peak_layer == 3
        assert constraint.source_activation == 0.95
        assert constraint.target_activation == 0.92
        assert constraint.agreement_score == 0.88


class TestLayerCorrespondence:
    """Tests for LayerCorrespondence dataclass."""

    def test_frozen_dataclass(self):
        """LayerCorrespondence should be immutable."""
        correspondence = LayerCorrespondence(
            source_layer=0,
            target_layer=1,
            supporting_probes=("probe1",),
            conflicting_probes=(),
            consensus_ratio=1.0,
        )
        with pytest.raises(AttributeError):
            correspondence.source_layer = 5

    def test_is_unanimous_with_support_no_conflicts(self):
        """is_unanimous should be True when probes agree."""
        correspondence = LayerCorrespondence(
            source_layer=0,
            target_layer=1,
            supporting_probes=("probe1", "probe2"),
            conflicting_probes=(),
            consensus_ratio=1.0,
        )
        assert correspondence.is_unanimous is True
        assert correspondence.has_conflicts is False

    def test_is_unanimous_with_conflicts(self):
        """is_unanimous should be False when probes conflict."""
        correspondence = LayerCorrespondence(
            source_layer=0,
            target_layer=1,
            supporting_probes=("probe1",),
            conflicting_probes=("probe2",),
            consensus_ratio=0.5,
        )
        assert correspondence.is_unanimous is False
        assert correspondence.has_conflicts is True

    def test_is_unanimous_with_no_support(self):
        """is_unanimous should be False when no supporting probes."""
        correspondence = LayerCorrespondence(
            source_layer=0,
            target_layer=1,
            supporting_probes=(),
            conflicting_probes=(),
            consensus_ratio=0.0,
        )
        assert correspondence.is_unanimous is False


class TestConstraintAlignmentResult:
    """Tests for ConstraintAlignmentResult dataclass."""

    def test_frozen_dataclass(self):
        """ConstraintAlignmentResult should be immutable."""
        result = ConstraintAlignmentResult(
            layer_mappings=(),
            unanimous_mappings=0,
            conflicting_mappings=0,
            probes_needing_investigation=(),
        )
        with pytest.raises(AttributeError):
            result.unanimous_mappings = 10

    def test_is_fully_aligned_all_unanimous(self):
        """is_fully_aligned should be True when all mappings unanimous."""
        result = ConstraintAlignmentResult(
            layer_mappings=(
                LayerCorrespondence(
                    source_layer=0,
                    target_layer=0,
                    supporting_probes=("p1",),
                    conflicting_probes=(),
                    consensus_ratio=1.0,
                ),
            ),
            unanimous_mappings=1,
            conflicting_mappings=0,
            probes_needing_investigation=(),
        )
        assert result.is_fully_aligned is True

    def test_is_fully_aligned_with_conflicts(self):
        """is_fully_aligned should be False with conflicts."""
        result = ConstraintAlignmentResult(
            layer_mappings=(),
            unanimous_mappings=2,
            conflicting_mappings=1,
            probes_needing_investigation=("probe3",),
        )
        assert result.is_fully_aligned is False

    def test_is_fully_aligned_no_mappings(self):
        """is_fully_aligned should be False with no mappings."""
        result = ConstraintAlignmentResult(
            layer_mappings=(),
            unanimous_mappings=0,
            conflicting_mappings=0,
            probes_needing_investigation=(),
        )
        assert result.is_fully_aligned is False


class TestConstraintAlignerInit:
    """Tests for ConstraintAligner initialization."""

    def test_default_initialization(self):
        """Aligner should initialize without backend."""
        aligner = ConstraintAligner()
        assert aligner is not None
        assert aligner.backend is not None

    def test_with_explicit_backend(self):
        """Aligner should accept explicit backend."""
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        aligner = ConstraintAligner(backend)
        assert aligner.backend is backend


class TestFindPeakLayer:
    """Tests for find_peak_layer method."""

    def test_empty_activations(self):
        """Empty activations should return (-1, 0.0)."""
        aligner = ConstraintAligner()
        peak_layer, peak_activation = aligner.find_peak_layer({})
        assert peak_layer == -1
        assert peak_activation == 0.0

    def test_single_layer(self):
        """Single layer should be the peak."""
        aligner = ConstraintAligner()
        activations = {0: [1.0, 2.0, 3.0]}
        peak_layer, peak_activation = aligner.find_peak_layer(activations)
        assert peak_layer == 0
        assert peak_activation > 0

    def test_multiple_layers_finds_highest(self):
        """Should find layer with highest activation norm."""
        aligner = ConstraintAligner()
        activations = {
            0: [0.1, 0.1],  # Low
            1: [10.0, 10.0],  # High - should be peak
            2: [0.5, 0.5],  # Medium
        }
        peak_layer, peak_activation = aligner.find_peak_layer(activations)
        assert peak_layer == 1

    def test_empty_layer_skipped(self):
        """Layers with empty activations should be skipped."""
        aligner = ConstraintAligner()
        activations = {
            0: [],  # Empty - skip
            1: [1.0, 2.0],  # Has data
        }
        peak_layer, peak_activation = aligner.find_peak_layer(activations)
        assert peak_layer == 1


class TestExtractConstraints:
    """Tests for extract_constraints method."""

    def test_extract_basic_constraint(self):
        """Should extract constraint from probe activations."""
        aligner = ConstraintAligner()
        source = {0: [1.0, 2.0], 1: [0.1, 0.1]}
        target = {0: [0.1, 0.1], 1: [1.0, 2.0]}

        constraint = aligner.extract_constraints("probe1", source, target)

        assert isinstance(constraint, ProbeConstraint)
        assert constraint.probe_id == "probe1"
        assert constraint.source_peak_layer == 0  # Higher activation at layer 0
        assert constraint.target_peak_layer == 1  # Higher activation at layer 1

    def test_extract_constraint_has_agreement_score(self):
        """Constraint should have CKA agreement score."""
        aligner = ConstraintAligner()
        source = {0: [1.0, 2.0, 3.0]}
        target = {0: [1.0, 2.0, 3.0]}  # Same as source

        constraint = aligner.extract_constraints("probe1", source, target)

        assert constraint.agreement_score >= 0.0
        assert constraint.agreement_score <= 1.0


class TestAlignFromConstraints:
    """Tests for align_from_constraints method."""

    def test_empty_constraints(self):
        """Empty constraints should return empty result."""
        aligner = ConstraintAligner()
        result = aligner.align_from_constraints([])

        assert isinstance(result, ConstraintAlignmentResult)
        assert len(result.layer_mappings) == 0
        assert result.unanimous_mappings == 0
        assert result.conflicting_mappings == 0

    def test_single_constraint(self):
        """Single constraint should create one mapping."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id="probe1",
                source_peak_layer=0,
                target_peak_layer=1,
                source_activation=1.0,
                target_activation=0.9,
                agreement_score=0.95,
            )
        ]
        result = aligner.align_from_constraints(constraints)

        assert len(result.layer_mappings) == 1
        assert result.layer_mappings[0].source_layer == 0
        assert result.layer_mappings[0].target_layer == 1
        assert result.unanimous_mappings == 1

    def test_unanimous_constraints(self):
        """Agreeing constraints should be unanimous."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id="probe1",
                source_peak_layer=0,
                target_peak_layer=1,
                source_activation=1.0,
                target_activation=0.9,
                agreement_score=0.95,
            ),
            ProbeConstraint(
                probe_id="probe2",
                source_peak_layer=0,
                target_peak_layer=1,  # Same mapping
                source_activation=0.8,
                target_activation=0.85,
                agreement_score=0.92,
            ),
        ]
        result = aligner.align_from_constraints(constraints)

        assert result.unanimous_mappings == 1
        assert result.conflicting_mappings == 0
        assert result.is_fully_aligned is True

    def test_conflicting_constraints(self):
        """Disagreeing constraints should show conflicts."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id="probe1",
                source_peak_layer=0,
                target_peak_layer=1,
                source_activation=1.0,
                target_activation=0.9,
                agreement_score=0.95,
            ),
            ProbeConstraint(
                probe_id="probe2",
                source_peak_layer=0,
                target_peak_layer=2,  # Different target!
                source_activation=0.8,
                target_activation=0.85,
                agreement_score=0.92,
            ),
        ]
        result = aligner.align_from_constraints(constraints)

        assert result.conflicting_mappings == 1
        assert result.is_fully_aligned is False
        assert len(result.probes_needing_investigation) > 0

    def test_multiple_source_layers(self):
        """Should handle constraints from different source layers."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id="probe1",
                source_peak_layer=0,
                target_peak_layer=0,
                source_activation=1.0,
                target_activation=1.0,
                agreement_score=1.0,
            ),
            ProbeConstraint(
                probe_id="probe2",
                source_peak_layer=1,  # Different source layer
                target_peak_layer=2,
                source_activation=1.0,
                target_activation=1.0,
                agreement_score=1.0,
            ),
        ]
        result = aligner.align_from_constraints(constraints)

        assert len(result.layer_mappings) == 2
        assert result.unanimous_mappings == 2


class TestDiagnoseProbeConflict:
    """Tests for diagnose_probe_conflict function."""

    def test_returns_dict(self):
        """Should return a diagnostic dictionary."""
        source = {0: [1.0, 2.0], 1: [0.5, 0.5]}
        target = {0: [1.0, 2.0], 1: [0.5, 0.5]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert isinstance(result, dict)
        assert "probe_id" in result
        assert result["probe_id"] == "probe1"

    def test_includes_peak_info(self):
        """Diagnosis should include peak layer information."""
        source = {0: [1.0, 2.0], 1: [0.1, 0.1]}
        target = {0: [0.1, 0.1], 1: [1.0, 2.0]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert "source_peak_layer" in result
        assert "target_peak_layer" in result
        assert result["source_peak_layer"] == 0
        assert result["target_peak_layer"] == 1

    def test_includes_activation_strength(self):
        """Diagnosis should include activation strengths."""
        source = {0: [1.0, 2.0]}
        target = {0: [1.0, 2.0]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert "source_activation_strength" in result
        assert "target_activation_strength" in result
        assert result["source_activation_strength"] > 0
        assert result["target_activation_strength"] > 0

    def test_includes_cka_at_peak(self):
        """Diagnosis should include CKA at peak layers."""
        source = {0: [1.0, 2.0, 3.0]}
        target = {0: [1.0, 2.0, 3.0]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert "peak_cka" in result
        assert 0.0 <= result["peak_cka"] <= 1.0

    def test_includes_issues_list(self):
        """Diagnosis should include issues list."""
        source = {0: [1.0, 2.0], 1: [0.9, 1.9]}  # Two similar peaks
        target = {0: [1.0, 2.0], 1: [0.9, 1.9]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_secondary_peak_detection(self):
        """Should detect secondary peaks."""
        source = {0: [10.0, 10.0], 1: [9.0, 9.0], 2: [1.0, 1.0]}
        target = {0: [1.0, 1.0], 1: [10.0, 10.0], 2: [9.0, 9.0]}

        result = diagnose_probe_conflict("probe1", source, target)

        assert "source_secondary_layer" in result
        assert "target_secondary_layer" in result


class TestComputePerLayerCKA:
    """Tests for compute_per_layer_cka method."""

    def test_returns_dict(self):
        """Should return dictionary of CKA values."""
        aligner = ConstraintAligner()
        source = {0: [1.0, 2.0, 3.0]}
        target = {0: [1.0, 2.0, 3.0]}

        result = aligner.compute_per_layer_cka(source, target)

        assert isinstance(result, dict)
        assert (0, 0) in result

    def test_cka_bounded(self):
        """CKA values should be in [0, 1]."""
        aligner = ConstraintAligner()
        source = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
        target = {0: [1.0, 2.0, 3.0], 1: [7.0, 8.0, 9.0]}

        result = aligner.compute_per_layer_cka(source, target)

        for cka_value in result.values():
            assert 0.0 <= cka_value <= 1.0

    def test_handles_insufficient_data(self):
        """Should handle cases with insufficient data."""
        aligner = ConstraintAligner()
        source = {0: [1.0]}  # Only 1 element
        target = {0: [2.0]}

        result = aligner.compute_per_layer_cka(source, target)

        assert (0, 0) in result
        assert result[(0, 0)] == 0.0  # Insufficient data


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_layers_empty(self):
        """Handle all empty layers."""
        aligner = ConstraintAligner()
        activations = {0: [], 1: [], 2: []}
        peak_layer, peak_activation = aligner.find_peak_layer(activations)
        assert peak_layer == -1
        assert peak_activation == 0.0

    def test_constraint_with_negative_layers(self):
        """Constraints can have negative layer values (from empty input)."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id="probe1",
                source_peak_layer=-1,
                target_peak_layer=-1,
                source_activation=0.0,
                target_activation=0.0,
                agreement_score=0.0,
            )
        ]
        result = aligner.align_from_constraints(constraints)
        assert isinstance(result, ConstraintAlignmentResult)

    def test_large_number_of_constraints(self):
        """Should handle many constraints efficiently."""
        aligner = ConstraintAligner()
        constraints = [
            ProbeConstraint(
                probe_id=f"probe{i}",
                source_peak_layer=i % 5,
                target_peak_layer=i % 5,
                source_activation=1.0,
                target_activation=1.0,
                agreement_score=1.0,
            )
            for i in range(50)
        ]
        result = aligner.align_from_constraints(constraints)
        assert isinstance(result, ConstraintAlignmentResult)
        assert len(result.layer_mappings) == 5  # 5 unique source layers

    def test_high_dimensional_activations(self):
        """Should handle high-dimensional activations."""
        aligner = ConstraintAligner()
        source = {0: [float(i) for i in range(100)]}
        target = {0: [float(i) for i in range(100)]}

        constraint = aligner.extract_constraints("probe1", source, target)
        assert isinstance(constraint, ProbeConstraint)
