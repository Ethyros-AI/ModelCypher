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

"""Comprehensive tests for concept_response_matrix.py module."""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorCategory,
    AnchorMetadata,
    ComparisonReport,
    ConceptResponseMatrix,
    ConsistencyProfile,
    LayerStatistics,
    LayerTransitionResult,
    TransitionExperiment,
    _cosine_similarity_matrix,
    _decode_datetime,
    _encode_datetime,
    _interpolate_layer_alignment,
    _mean_absolute_difference,
    _mean_pool_state,
    _sample_layer_indices,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


# =============================================================================
# Fixtures
# =============================================================================


def _build_crm() -> ConceptResponseMatrix:
    """Build a simple CRM with 3 anchors and 2 layers."""
    anchor_ids = ["prime:A", "prime:B", "prime:C"]
    metadata = AnchorMetadata(
        total_count=3,
        semantic_prime_count=3,
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier="test-model",
        layer_count=2,
        hidden_dim=2,
        anchor_metadata=metadata,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    crm.record_activations("prime:A", {0: [1.0, 0.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:B", {0: [0.0, 1.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:C", {0: [1.0, 1.0], 1: [0.0, 1.0]})
    return crm


def _build_crm4() -> ConceptResponseMatrix:
    """Build a CRM with 4 anchors and 2 layers."""
    anchor_ids = ["prime:A", "prime:B", "prime:C", "prime:D"]
    metadata = AnchorMetadata(
        total_count=4,
        semantic_prime_count=4,
        computational_gate_count=0,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier="test-model",
        layer_count=2,
        hidden_dim=2,
        anchor_metadata=metadata,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    crm.record_activations("prime:A", {0: [1.0, 0.0], 1: [1.0, 0.0]})
    crm.record_activations("prime:B", {0: [0.0, 1.0], 1: [0.0, 1.0]})
    crm.record_activations("prime:C", {0: [1.0, 1.0], 1: [1.0, 1.0]})
    crm.record_activations("prime:D", {0: [1.0, -1.0], 1: [1.0, -1.0]})
    return crm


def _build_crm_mixed() -> ConceptResponseMatrix:
    """Build a CRM with both semantic primes and computational gates."""
    anchor_ids = ["prime:A", "prime:B", "gate:X", "gate:Y"]
    metadata = AnchorMetadata(
        total_count=4,
        semantic_prime_count=2,
        computational_gate_count=2,
        anchor_ids=anchor_ids,
    )
    crm = ConceptResponseMatrix(
        model_identifier="mixed-model",
        layer_count=3,
        hidden_dim=4,
        anchor_metadata=metadata,
        created_at=datetime(2025, 6, 15, tzinfo=timezone.utc),
    )
    crm.record_activations("prime:A", {0: [1.0, 0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]})
    crm.record_activations("prime:B", {0: [0.0, 1.0, 0.0, 0.0], 1: [0.0, 0.5, 0.5, 0.0], 2: [0.0, 0.0, 1.0, 0.0]})
    crm.record_activations("gate:X", {0: [0.0, 0.0, 1.0, 0.0], 1: [0.0, 0.0, 0.5, 0.5], 2: [0.0, 0.0, 0.0, 1.0]})
    crm.record_activations("gate:Y", {0: [0.0, 0.0, 0.0, 1.0], 1: [0.5, 0.0, 0.0, 0.5], 2: [1.0, 0.0, 0.0, 0.0]})
    return crm


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


# =============================================================================
# AnchorActivation Tests
# =============================================================================


class TestAnchorActivation:
    """Tests for the AnchorActivation dataclass."""

    def test_norm_computed_from_activation(self) -> None:
        """Norm should be computed from activation vector."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[3.0, 4.0])
        assert abs(act.norm - 5.0) < _div_eps()

    def test_norm_unit_vector(self) -> None:
        """Unit vector should have norm 1."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[1.0, 0.0, 0.0])
        assert abs(act.norm - 1.0) < _div_eps()

    def test_norm_zero_vector(self) -> None:
        """Zero vector should have norm 0."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[0.0, 0.0])
        assert act.norm == 0.0

    def test_norm_empty_activation(self) -> None:
        """Empty activation should have norm 0."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[])
        assert act.norm == 0.0

    def test_frozen_dataclass(self) -> None:
        """AnchorActivation should be immutable."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[1.0, 2.0])
        with pytest.raises(Exception):  # FrozenInstanceError
            act.anchor_id = "modified"  # type: ignore

    def test_negative_values_in_activation(self) -> None:
        """Norm should work with negative values."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[-3.0, -4.0])
        assert abs(act.norm - 5.0) < _div_eps()

    def test_single_element_activation(self) -> None:
        """Single element activation norm."""
        act = AnchorActivation(anchor_id="test", layer=0, activation=[7.0])
        assert abs(act.norm - 7.0) < _div_eps()

    def test_high_dimensional_activation(self) -> None:
        """High dimensional activation norm."""
        # Vector of 100 ones has norm sqrt(100) = 10
        act = AnchorActivation(anchor_id="test", layer=0, activation=[1.0] * 100)
        assert abs(act.norm - 10.0) < _div_eps()


# =============================================================================
# LayerStatistics Tests
# =============================================================================


class TestLayerStatistics:
    """Tests for the LayerStatistics dataclass."""

    def test_frozen_dataclass(self) -> None:
        """LayerStatistics should be immutable."""
        stats = LayerStatistics(
            layer=0,
            anchor_count=10,
            mean_activation_norm=1.5,
            std_activation_norm=0.5,
            hidden_dim=768,
        )
        with pytest.raises(Exception):
            stats.layer = 1  # type: ignore

    def test_all_fields_accessible(self) -> None:
        """All fields should be accessible."""
        stats = LayerStatistics(
            layer=5,
            anchor_count=100,
            mean_activation_norm=2.5,
            std_activation_norm=0.8,
            hidden_dim=1024,
        )
        assert stats.layer == 5
        assert stats.anchor_count == 100
        assert abs(stats.mean_activation_norm - 2.5) < _div_eps()
        assert abs(stats.std_activation_norm - 0.8) < _div_eps()
        assert stats.hidden_dim == 1024


# =============================================================================
# AnchorCategory Tests
# =============================================================================


class TestAnchorCategory:
    """Tests for the AnchorCategory enum."""

    def test_semantic_prime_value(self) -> None:
        """Semantic prime should have value 'prime'."""
        assert AnchorCategory.semantic_prime.value == "prime"

    def test_computational_gate_value(self) -> None:
        """Computational gate should have value 'gate'."""
        assert AnchorCategory.computational_gate.value == "gate"

    def test_semantic_prime_prefix(self) -> None:
        """Semantic prime prefix should be 'prime:'."""
        assert AnchorCategory.semantic_prime.prefix == "prime:"

    def test_computational_gate_prefix(self) -> None:
        """Computational gate prefix should be 'gate:'."""
        assert AnchorCategory.computational_gate.prefix == "gate:"

    def test_enum_is_str(self) -> None:
        """AnchorCategory should be a string enum."""
        assert isinstance(AnchorCategory.semantic_prime, str)
        assert AnchorCategory.semantic_prime == "prime"


# =============================================================================
# AnchorMetadata Tests
# =============================================================================


class TestAnchorMetadata:
    """Tests for the AnchorMetadata dataclass."""

    def test_frozen_dataclass(self) -> None:
        """AnchorMetadata should be immutable."""
        meta = AnchorMetadata(
            total_count=10,
            semantic_prime_count=6,
            computational_gate_count=4,
            anchor_ids=["a", "b"],
        )
        with pytest.raises(Exception):
            meta.total_count = 20  # type: ignore

    def test_all_fields_accessible(self) -> None:
        """All fields should be accessible."""
        ids = ["prime:A", "prime:B", "gate:X"]
        meta = AnchorMetadata(
            total_count=3,
            semantic_prime_count=2,
            computational_gate_count=1,
            anchor_ids=ids,
        )
        assert meta.total_count == 3
        assert meta.semantic_prime_count == 2
        assert meta.computational_gate_count == 1
        assert meta.anchor_ids == ids


# =============================================================================
# ConceptResponseMatrix Tests - Basic Operations
# =============================================================================


class TestConceptResponseMatrixBasic:
    """Tests for basic ConceptResponseMatrix operations."""

    def test_activation_matrix_order(self) -> None:
        """Activation matrix should preserve anchor order."""
        crm = _build_crm()
        matrix = crm.activation_matrix(0)
        assert matrix == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    def test_activation_matrix_layer_1(self) -> None:
        """Activation matrix for layer 1."""
        crm = _build_crm()
        matrix = crm.activation_matrix(1)
        assert matrix == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    def test_activation_matrix_invalid_layer(self) -> None:
        """Invalid layer should return None."""
        crm = _build_crm()
        assert crm.activation_matrix(99) is None

    def test_activation_matrix_negative_layer(self) -> None:
        """Negative layer should return None."""
        crm = _build_crm()
        assert crm.activation_matrix(-1) is None

    def test_common_anchor_ids(self) -> None:
        """Should find common anchors between two CRMs."""
        crm = _build_crm()
        crm_alt = _build_crm4()
        common = crm.common_anchor_ids(crm_alt)
        assert common == ["prime:A", "prime:B", "prime:C"]

    def test_common_anchor_ids_same_crm(self) -> None:
        """CRM compared to itself should return all anchors."""
        crm = _build_crm()
        common = crm.common_anchor_ids(crm)
        assert sorted(common) == sorted(crm.anchor_metadata.anchor_ids)

    def test_common_anchor_ids_no_overlap(self) -> None:
        """No overlapping anchors should return empty list."""
        crm1 = _build_crm()
        anchor_ids = ["prime:X", "prime:Y"]
        metadata = AnchorMetadata(
            total_count=2,
            semantic_prime_count=2,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm2 = ConceptResponseMatrix(
            model_identifier="other",
            layer_count=2,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        common = crm1.common_anchor_ids(crm2)
        assert common == []


# =============================================================================
# ConceptResponseMatrix Tests - Layer Statistics
# =============================================================================


class TestConceptResponseMatrixStatistics:
    """Tests for compute_layer_statistics."""

    def test_compute_layer_statistics_basic(self) -> None:
        """Should compute statistics for each layer."""
        crm = _build_crm()
        stats = crm.compute_layer_statistics()
        assert len(stats) == 2
        assert stats[0].layer == 0
        assert stats[1].layer == 1

    def test_compute_layer_statistics_anchor_count(self) -> None:
        """Anchor count should match recorded activations."""
        crm = _build_crm()
        stats = crm.compute_layer_statistics()
        assert stats[0].anchor_count == 3
        assert stats[1].anchor_count == 3

    def test_compute_layer_statistics_hidden_dim(self) -> None:
        """Hidden dim should match activation dimension."""
        crm = _build_crm()
        stats = crm.compute_layer_statistics()
        assert stats[0].hidden_dim == 2
        assert stats[1].hidden_dim == 2

    def test_compute_layer_statistics_mean_norm(self) -> None:
        """Mean norm should be computed correctly."""
        crm = _build_crm()
        stats = crm.compute_layer_statistics()
        # Layer 0: norms are 1.0, 1.0, sqrt(2) ≈ 1.414
        expected = (2.0 + math.sqrt(2.0)) / 3.0
        assert abs(stats[0].mean_activation_norm - expected) < _div_eps()

    def test_compute_layer_statistics_empty_crm(self) -> None:
        """CRM with no activations should return empty stats."""
        metadata = AnchorMetadata(
            total_count=0,
            semantic_prime_count=0,
            computational_gate_count=0,
            anchor_ids=[],
        )
        crm = ConceptResponseMatrix(
            model_identifier="empty",
            layer_count=2,
            hidden_dim=4,
            anchor_metadata=metadata,
        )
        stats = crm.compute_layer_statistics()
        assert stats == []


# =============================================================================
# ConceptResponseMatrix Tests - Category Filtering
# =============================================================================


class TestConceptResponseMatrixCategory:
    """Tests for activation_matrix_for_category."""

    def test_filter_semantic_primes(self) -> None:
        """Should filter to only semantic primes."""
        crm = _build_crm_mixed()
        matrix = crm.activation_matrix_for_category(AnchorCategory.semantic_prime, 0)
        assert matrix is not None
        assert len(matrix) == 2  # Only prime:A and prime:B

    def test_filter_computational_gates(self) -> None:
        """Should filter to only computational gates."""
        crm = _build_crm_mixed()
        matrix = crm.activation_matrix_for_category(AnchorCategory.computational_gate, 0)
        assert matrix is not None
        assert len(matrix) == 2  # Only gate:X and gate:Y

    def test_filter_category_invalid_layer(self) -> None:
        """Invalid layer should return None."""
        crm = _build_crm_mixed()
        matrix = crm.activation_matrix_for_category(AnchorCategory.semantic_prime, 99)
        assert matrix is None

    def test_filter_category_no_matches(self) -> None:
        """No matching category should return None."""
        crm = _build_crm()  # Only has primes
        matrix = crm.activation_matrix_for_category(AnchorCategory.computational_gate, 0)
        assert matrix is None


# =============================================================================
# ConceptResponseMatrix Tests - CKA Computation
# =============================================================================


class TestConceptResponseMatrixCKA:
    """Tests for CKA matrix computation."""

    def test_cka_matrix_values(self) -> None:
        """CKA matrix should have correct self-similarity."""
        crm = _build_crm()
        cka = crm.compute_cka_matrix(crm)
        assert len(cka) == 2
        # Self-comparison should be 1.0
        assert abs(cka[0][0] - 1.0) < _div_eps()
        assert abs(cka[1][1] - 1.0) < _div_eps()
        # Cross-layer CKA values - verify they are symmetric and bounded
        assert abs(cka[0][1] - cka[1][0]) < _div_eps()  # Symmetric
        assert 0.0 <= cka[0][1] <= 1.0  # Bounded

    def test_cka_matrix_symmetric_for_self(self) -> None:
        """CKA matrix comparing CRM to itself should be symmetric."""
        crm = _build_crm4()
        cka = crm.compute_cka_matrix(crm)
        for i in range(len(cka)):
            for j in range(len(cka[0])):
                assert abs(cka[i][j] - cka[j][i]) < _div_eps()

    def test_cka_matrix_bounded_zero_one(self) -> None:
        """CKA values should be in [0, 1]."""
        crm = _build_crm()
        cka = crm.compute_cka_matrix(crm)
        for row in cka:
            for value in row:
                assert 0.0 <= value <= 1.0

    def test_cka_matrix_no_common_anchors(self) -> None:
        """No common anchors should return zero matrix."""
        crm1 = _build_crm()
        anchor_ids = ["prime:X", "prime:Y"]
        metadata = AnchorMetadata(
            total_count=2,
            semantic_prime_count=2,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm2 = ConceptResponseMatrix(
            model_identifier="other",
            layer_count=2,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        crm2.record_activations("prime:X", {0: [1.0, 0.0], 1: [0.0, 1.0]})
        crm2.record_activations("prime:Y", {0: [0.0, 1.0], 1: [1.0, 0.0]})

        cka = crm1.compute_cka_matrix(crm2)
        for row in cka:
            for value in row:
                assert value == 0.0

    def test_compute_layer_cka_valid(self) -> None:
        """compute_layer_cka should return valid CKA value."""
        crm = _build_crm()
        cka = crm.compute_layer_cka(0, crm, 0)
        assert cka is not None
        assert abs(cka - 1.0) < _div_eps()  # Self-comparison

    def test_compute_layer_cka_cross_layer(self) -> None:
        """Cross-layer CKA should be computed."""
        crm = _build_crm()
        cka = crm.compute_layer_cka(0, crm, 1)
        assert cka is not None
        assert 0.0 <= cka <= 1.0

    def test_compute_layer_cka_no_common_anchors(self) -> None:
        """No common anchors should return None."""
        crm1 = _build_crm()
        anchor_ids = ["prime:X"]
        metadata = AnchorMetadata(
            total_count=1,
            semantic_prime_count=1,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm2 = ConceptResponseMatrix(
            model_identifier="other",
            layer_count=1,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        cka = crm1.compute_layer_cka(0, crm2, 0)
        assert cka is None


# =============================================================================
# ConceptResponseMatrix Tests - Comparison
# =============================================================================


class TestConceptResponseMatrixComparison:
    """Tests for compare method."""

    def test_compare_report(self) -> None:
        """Compare should return a valid report."""
        crm = _build_crm()
        report = crm.compare(crm)
        assert report.common_anchor_count == 3
        assert len(report.layer_correspondence) == 2
        assert report.layer_correspondence[0].source_layer == 0
        assert report.layer_correspondence[0].target_layer == 0

    def test_compare_report_model_identifiers(self) -> None:
        """Report should contain correct model identifiers."""
        crm1 = _build_crm()
        crm2 = _build_crm4()
        crm2.model_identifier = "other-model"
        report = crm1.compare(crm2)
        assert report.source_model == "test-model"
        assert report.target_model == "other-model"

    def test_compare_overall_alignment_self(self) -> None:
        """Overall alignment with self should be 1.0."""
        crm = _build_crm()
        report = crm.compare(crm)
        expected = sum(match.cka for match in report.layer_correspondence) / float(
            len(report.layer_correspondence)
        )
        assert abs(report.overall_alignment - expected) < _div_eps()

    def test_compare_cka_matrix_in_report(self) -> None:
        """CKA matrix should be included in report."""
        crm = _build_crm()
        report = crm.compare(crm)
        assert len(report.cka_matrix) == 2
        assert len(report.cka_matrix[0]) == 2


# =============================================================================
# ConceptResponseMatrix Tests - Transition Alignment
# =============================================================================


class TestConceptResponseMatrixTransition:
    """Tests for compute_transition_alignment."""

    def test_transition_alignment_self(self) -> None:
        """Transition alignment with self should have perfect scores."""
        crm = _build_crm()
        experiment = crm.compute_transition_alignment(crm)
        assert experiment is not None
        assert experiment.anchor_count == 3
        assert experiment.layer_transition_count == 1
        assert abs(experiment.mean_transition_cka - 1.0) < _div_eps()
        assert abs(experiment.mean_state_cka - 1.0) < _div_eps()
        assert abs(experiment.transition_advantage - 1.0) < _div_eps()
        assert experiment.transition_better_than_state is False
        assert abs(experiment.transitions[0].delta_alignment - 1.0) < _div_eps()

    def test_transition_alignment_model_identifiers(self) -> None:
        """Experiment should have correct model identifiers."""
        crm1 = _build_crm()
        crm2 = _build_crm4()
        crm2.model_identifier = "other-model"
        experiment = crm1.compute_transition_alignment(crm2)
        assert experiment is not None
        assert experiment.source_model == "test-model"
        assert experiment.target_model == "other-model"

    def test_transition_alignment_too_few_anchors(self) -> None:
        """Less than 3 common anchors should return None."""
        anchor_ids = ["prime:A", "prime:B"]
        metadata = AnchorMetadata(
            total_count=2,
            semantic_prime_count=2,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm = ConceptResponseMatrix(
            model_identifier="small",
            layer_count=2,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        crm.record_activations("prime:A", {0: [1.0, 0.0], 1: [0.0, 1.0]})
        crm.record_activations("prime:B", {0: [0.0, 1.0], 1: [1.0, 0.0]})

        experiment = crm.compute_transition_alignment(crm)
        assert experiment is None

    def test_transition_alignment_single_layer(self) -> None:
        """Single layer CRM should return None (no transitions)."""
        anchor_ids = ["prime:A", "prime:B", "prime:C"]
        metadata = AnchorMetadata(
            total_count=3,
            semantic_prime_count=3,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm = ConceptResponseMatrix(
            model_identifier="single",
            layer_count=1,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        crm.record_activations("prime:A", {0: [1.0, 0.0]})
        crm.record_activations("prime:B", {0: [0.0, 1.0]})
        crm.record_activations("prime:C", {0: [1.0, 1.0]})

        experiment = crm.compute_transition_alignment(crm)
        assert experiment is None

    def test_transition_alignment_timestamp(self) -> None:
        """Experiment should have a timestamp."""
        crm = _build_crm()
        experiment = crm.compute_transition_alignment(crm)
        assert experiment is not None
        assert experiment.timestamp is not None
        assert experiment.timestamp.tzinfo is not None


# =============================================================================
# ConceptResponseMatrix Tests - Consistency Profile
# =============================================================================


class TestConceptResponseMatrixConsistency:
    """Tests for compute_consistency_profile."""

    def test_consistency_profile_alignment_centered(self) -> None:
        """Alignment values should be centered when comparing CRM to itself."""
        crm = _build_crm4()
        profile = crm.compute_consistency_profile(crm, layer_sample_count=2)
        assert profile is not None
        assert profile.anchor_count == 4
        assert profile.sample_layer_count == 2
        for alignment in profile.target_alignment_by_layer.values():
            assert abs(alignment - 0.5) < _div_eps()

    def test_consistency_profile_too_few_anchors(self) -> None:
        """Less than 4 common anchors should return None."""
        crm = _build_crm()  # Only 3 anchors
        profile = crm.compute_consistency_profile(crm)
        assert profile is None

    def test_consistency_profile_alignment_bounded(self) -> None:
        """Alignment values should be in [0, 1]."""
        crm = _build_crm4()
        profile = crm.compute_consistency_profile(crm, layer_sample_count=2)
        assert profile is not None
        for alignment in profile.target_alignment_by_layer.values():
            assert 0.0 <= alignment <= 1.0

    def test_consistency_profile_distances_non_negative(self) -> None:
        """Distances should be non-negative."""
        crm = _build_crm4()
        profile = crm.compute_consistency_profile(crm, layer_sample_count=2)
        assert profile is not None
        assert profile.mean_source_distance >= 0.0
        assert profile.mean_target_distance >= 0.0


# =============================================================================
# ConceptResponseMatrix Tests - Serialization
# =============================================================================


class TestConceptResponseMatrixSerialization:
    """Tests for save/load and to_dict/from_dict."""

    def test_to_dict_round_trip(self) -> None:
        """to_dict/from_dict should preserve data."""
        crm = _build_crm()
        payload = crm.to_dict()
        restored = ConceptResponseMatrix.from_dict(payload)

        assert restored.model_identifier == crm.model_identifier
        assert restored.layer_count == crm.layer_count
        assert restored.hidden_dim == crm.hidden_dim
        assert restored.anchor_metadata.total_count == crm.anchor_metadata.total_count

    def test_to_dict_activations_preserved(self) -> None:
        """Activations should be preserved in serialization."""
        crm = _build_crm()
        payload = crm.to_dict()
        restored = ConceptResponseMatrix.from_dict(payload)

        for layer in range(crm.layer_count):
            orig_matrix = crm.activation_matrix(layer)
            rest_matrix = restored.activation_matrix(layer)
            assert orig_matrix == rest_matrix

    def test_save_load_file(self) -> None:
        """save/load should preserve data in file."""
        crm = _build_crm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crm.json"
            crm.save(str(path))

            assert path.exists()
            restored = ConceptResponseMatrix.load(str(path))

            assert restored.model_identifier == crm.model_identifier
            assert restored.layer_count == crm.layer_count

    def test_save_creates_valid_json(self) -> None:
        """Saved file should be valid JSON."""
        crm = _build_crm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "crm.json"
            crm.save(str(path))

            with open(path) as f:
                data = json.load(f)

            assert "modelIdentifier" in data
            assert "layerCount" in data
            assert "activations" in data

    def test_to_dict_datetime_encoding(self) -> None:
        """Datetime should be properly encoded."""
        crm = _build_crm()
        payload = crm.to_dict()

        # Should be ISO format with Z suffix
        assert payload["createdAt"].endswith("Z")

    def test_from_dict_datetime_decoding(self) -> None:
        """Datetime should be properly decoded."""
        crm = _build_crm()
        payload = crm.to_dict()
        restored = ConceptResponseMatrix.from_dict(payload)

        assert restored.created_at.tzinfo is not None
        assert restored.created_at.year == 2025


# =============================================================================
# ConceptResponseMatrix Tests - Private Methods
# =============================================================================


class TestConceptResponseMatrixPrivate:
    """Tests for private methods."""

    def test_extract_activations_valid(self) -> None:
        """Should extract activations for given anchors."""
        crm = _build_crm()
        result = crm._extract_activations(0, ["prime:A", "prime:B"])
        assert result is not None
        assert len(result) == 2
        assert result[0] == [1.0, 0.0]
        assert result[1] == [0.0, 1.0]

    def test_extract_activations_missing_anchor(self) -> None:
        """Missing anchor should return None."""
        crm = _build_crm()
        result = crm._extract_activations(0, ["prime:A", "prime:MISSING"])
        assert result is None

    def test_extract_activations_invalid_layer(self) -> None:
        """Invalid layer should return None."""
        crm = _build_crm()
        result = crm._extract_activations(99, ["prime:A"])
        assert result is None

    def test_compute_layer_delta_basic(self) -> None:
        """Should compute layer delta correctly."""
        current = [[1.0, 0.0], [0.0, 1.0]]
        next_layer = [[2.0, 1.0], [1.0, 2.0]]

        delta, norm = ConceptResponseMatrix._compute_layer_delta(current, next_layer)

        assert len(delta) == 2
        assert delta[0] == [1.0, 1.0]
        assert delta[1] == [1.0, 1.0]
        # Each delta has norm sqrt(2), mean is sqrt(2)
        assert abs(norm - math.sqrt(2)) < _div_eps()

    def test_compute_layer_delta_different_lengths(self) -> None:
        """Different length lists should return empty."""
        current = [[1.0, 0.0]]
        next_layer = [[2.0, 1.0], [1.0, 2.0]]

        delta, norm = ConceptResponseMatrix._compute_layer_delta(current, next_layer)

        assert delta == []
        assert norm == 0.0

    def test_compute_layer_delta_empty(self) -> None:
        """Empty lists should return empty."""
        delta, norm = ConceptResponseMatrix._compute_layer_delta([], [])

        assert delta == []
        assert norm == 0.0

    def test_compute_linear_cka_identical(self) -> None:
        """Identical matrices should have CKA = 1."""
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        cka = ConceptResponseMatrix.compute_linear_cka(x, x)
        assert abs(cka - 1.0) < _div_eps()

    def test_compute_linear_cka_bounded(self) -> None:
        """CKA should be in [0, 1]."""
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        cka = ConceptResponseMatrix.compute_linear_cka(x, y)
        assert 0.0 <= cka <= 1.0


# =============================================================================
# ComparisonReport Tests
# =============================================================================


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_frozen_dataclass(self) -> None:
        """ComparisonReport should be immutable."""
        report = ComparisonReport(
            source_model="src",
            target_model="tgt",
            common_anchor_count=10,
            cka_matrix=[[1.0]],
            layer_correspondence=[],
            overall_alignment=0.9,
        )
        with pytest.raises(Exception):
            report.source_model = "modified"  # type: ignore

    def test_layer_match_frozen(self) -> None:
        """LayerMatch should be immutable."""
        match = ComparisonReport.LayerMatch(
            source_layer=0,
            target_layer=1,
            cka=0.95,
        )
        with pytest.raises(Exception):
            match.cka = 0.5  # type: ignore


# =============================================================================
# LayerTransitionResult Tests
# =============================================================================


class TestLayerTransitionResult:
    """Tests for LayerTransitionResult dataclass."""

    def test_delta_alignment_computed(self) -> None:
        """delta_alignment should be computed from transition/state CKA."""
        result = LayerTransitionResult(
            from_layer=0,
            to_layer=1,
            transition_cka=0.8,
            state_cka=0.4,
            source_delta_norm=1.0,
            target_delta_norm=1.0,
        )
        # delta_alignment = transition_cka / state_cka = 0.8 / 0.4 = 2.0
        assert abs(result.delta_alignment - 2.0) < _div_eps()

    def test_delta_alignment_zero_state_cka(self) -> None:
        """delta_alignment should be 0 when state_cka is very small."""
        eps = _div_eps()
        result = LayerTransitionResult(
            from_layer=0,
            to_layer=1,
            transition_cka=0.8,
            state_cka=eps * 0.5,
            source_delta_norm=1.0,
            target_delta_norm=1.0,
        )
        assert result.delta_alignment == 0.0

    def test_frozen_dataclass(self) -> None:
        """LayerTransitionResult should be immutable."""
        result = LayerTransitionResult(
            from_layer=0,
            to_layer=1,
            transition_cka=0.8,
            state_cka=0.4,
            source_delta_norm=1.0,
            target_delta_norm=1.0,
        )
        with pytest.raises(Exception):
            result.from_layer = 5  # type: ignore


# =============================================================================
# TransitionExperiment Tests
# =============================================================================


class TestTransitionExperiment:
    """Tests for TransitionExperiment dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TransitionExperiment should be immutable."""
        experiment = TransitionExperiment(
            source_model="src",
            target_model="tgt",
            timestamp=datetime.now(timezone.utc),
            transitions=[],
            mean_transition_cka=0.9,
            mean_state_cka=0.8,
            transition_better_than_state=True,
            transition_advantage=1.125,
            anchor_count=10,
            layer_transition_count=5,
        )
        with pytest.raises(Exception):
            experiment.source_model = "modified"  # type: ignore


# =============================================================================
# ConsistencyProfile Tests
# =============================================================================


class TestConsistencyProfile:
    """Tests for ConsistencyProfile dataclass."""

    def test_frozen_dataclass(self) -> None:
        """ConsistencyProfile should be immutable."""
        profile = ConsistencyProfile(
            anchor_count=10,
            sample_layer_count=5,
            mean_source_distance=0.1,
            mean_target_distance=0.2,
            target_alignment_by_layer={0: 0.5, 1: 0.6},
        )
        with pytest.raises(Exception):
            profile.anchor_count = 20  # type: ignore


# =============================================================================
# Helper Function Tests - _mean_pool_state
# =============================================================================


class TestMeanPoolState:
    """Tests for _mean_pool_state helper function."""

    def test_3d_tensor_pooled(self) -> None:
        """3D tensor should be mean-pooled to 1D."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        # Shape (2, 3, 4) - batch, seq, hidden
        state = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                 [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]

        result = _mean_pool_state(state, backend)
        assert result.shape == (4,)

    def test_2d_tensor_pooled(self) -> None:
        """2D tensor should be mean-pooled over first axis."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        # Shape (3, 4) - seq, hidden
        state = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]

        result = _mean_pool_state(state, backend)
        assert result.shape == (4,)
        # Mean of [1,5,9], [2,6,10], [3,7,11], [4,8,12] = [5, 6, 7, 8]
        expected = [5.0, 6.0, 7.0, 8.0]
        for i, val in enumerate(expected):
            assert abs(float(backend.to_numpy(result)[i]) - val) < _div_eps()

    def test_1d_tensor_unchanged(self) -> None:
        """1D tensor should be unchanged."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        state = [1.0, 2.0, 3.0, 4.0]
        result = _mean_pool_state(state, backend)
        assert result.shape == (4,)


# =============================================================================
# Helper Function Tests - _cosine_similarity_matrix
# =============================================================================


class TestCosineSimilarityMatrix:
    """Tests for _cosine_similarity_matrix helper function."""

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have zero similarity."""
        activations = [[1.0, 0.0], [0.0, 1.0]]
        result = _cosine_similarity_matrix(activations)

        assert result is not None
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()
        np_result = backend.to_numpy(result)

        # Diagonal should be 1, off-diagonal should be 0
        assert abs(np_result[0, 0] - 1.0) < _div_eps()
        assert abs(np_result[1, 1] - 1.0) < _div_eps()
        assert abs(np_result[0, 1]) < _div_eps()
        assert abs(np_result[1, 0]) < _div_eps()

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1."""
        activations = [[1.0, 1.0], [1.0, 1.0]]
        result = _cosine_similarity_matrix(activations)

        assert result is not None
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()
        np_result = backend.to_numpy(result)

        # All entries should be 1
        for i in range(2):
            for j in range(2):
                assert abs(np_result[i, j] - 1.0) < _div_eps()

    def test_empty_activations(self) -> None:
        """Empty activations should return None."""
        result = _cosine_similarity_matrix([])
        assert result is None

    def test_single_activation(self) -> None:
        """Single activation should return 1x1 matrix."""
        activations = [[1.0, 2.0, 3.0]]
        result = _cosine_similarity_matrix(activations)

        assert result is not None
        assert result.shape == (1, 1)


# =============================================================================
# Helper Function Tests - _mean_absolute_difference
# =============================================================================


class TestMeanAbsoluteDifference:
    """Tests for _mean_absolute_difference helper function."""

    def test_identical_arrays(self) -> None:
        """Identical arrays should have zero difference."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        a = backend.array([[1.0, 2.0], [3.0, 4.0]])
        b = backend.array([[1.0, 2.0], [3.0, 4.0]])

        diff = _mean_absolute_difference(a, b)
        assert abs(diff) < _div_eps()

    def test_constant_difference(self) -> None:
        """Constant difference should be reflected in mean."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        a = backend.array([[1.0, 2.0], [3.0, 4.0]])
        b = backend.array([[2.0, 3.0], [4.0, 5.0]])

        diff = _mean_absolute_difference(a, b)
        assert abs(diff - 1.0) < _div_eps()

    def test_different_shapes(self) -> None:
        """Different shapes should return 0."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        a = backend.array([[1.0, 2.0]])
        b = backend.array([[1.0, 2.0], [3.0, 4.0]])

        diff = _mean_absolute_difference(a, b)
        assert diff == 0.0

    def test_empty_arrays(self) -> None:
        """Empty arrays should return 0."""
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()

        a = backend.array([])
        b = backend.array([])

        diff = _mean_absolute_difference(a, b)
        assert diff == 0.0


# =============================================================================
# Helper Function Tests - _sample_layer_indices
# =============================================================================


class TestSampleLayerIndices:
    """Tests for _sample_layer_indices helper function."""

    def test_single_sample(self) -> None:
        """Single sample should return middle layer."""
        result = _sample_layer_indices(10, 1)
        assert result == [5]

    def test_sample_more_than_layers(self) -> None:
        """Requesting more samples than layers should return all layers."""
        result = _sample_layer_indices(5, 10)
        assert result == [0, 1, 2, 3, 4]

    def test_sample_equals_layers(self) -> None:
        """Requesting same as layers should return all layers."""
        result = _sample_layer_indices(5, 5)
        assert result == [0, 1, 2, 3, 4]

    def test_zero_layers(self) -> None:
        """Zero layers should return empty list."""
        result = _sample_layer_indices(0, 5)
        assert result == []

    def test_includes_first_and_last(self) -> None:
        """Should always include first and last layers."""
        result = _sample_layer_indices(10, 3)
        assert 0 in result
        assert 9 in result

    def test_evenly_spaced(self) -> None:
        """Samples should be approximately evenly spaced."""
        result = _sample_layer_indices(10, 4)
        assert 0 in result
        assert 9 in result
        assert len(result) >= 4

    def test_no_duplicates(self) -> None:
        """Should not have duplicate indices."""
        result = _sample_layer_indices(100, 10)
        assert len(result) == len(set(result))


# =============================================================================
# Helper Function Tests - _interpolate_layer_alignment
# =============================================================================


class TestInterpolateLayerAlignment:
    """Tests for _interpolate_layer_alignment helper function."""

    def test_basic_interpolation(self) -> None:
        """Should interpolate between sample weights."""
        result = _interpolate_layer_alignment(
            sample_layers=[0, 4],
            sample_alignment={0: 0.0, 4: 1.0},
            layer_count=5,
        )

        assert result[0] == 0.0
        assert result[4] == 1.0
        assert abs(result[2] - 0.5) < _div_eps()

    def test_extrapolation_before_first(self) -> None:
        """Should extrapolate before first sample."""
        result = _interpolate_layer_alignment(
            sample_layers=[2, 4],
            sample_alignment={2: 0.3, 4: 0.7},
            layer_count=5,
        )

        # Layers 0 and 1 should have weight of first sample
        assert result[0] == 0.3
        assert result[1] == 0.3

    def test_extrapolation_after_last(self) -> None:
        """Should extrapolate after last sample."""
        result = _interpolate_layer_alignment(
            sample_layers=[0, 2],
            sample_alignment={0: 0.3, 2: 0.7},
            layer_count=5,
        )

        # Layers 3 and 4 should have weight of last sample
        assert result[3] == 0.7
        assert result[4] == 0.7

    def test_empty_sample_layers(self) -> None:
        """Empty sample layers should return empty dict."""
        result = _interpolate_layer_alignment(
            sample_layers=[],
            sample_alignment={},
            layer_count=5,
        )
        assert result == {}

    def test_zero_layer_count(self) -> None:
        """Zero layer count should return empty dict."""
        result = _interpolate_layer_alignment(
            sample_layers=[0, 1],
            sample_alignment={0: 0.5, 1: 0.5},
            layer_count=0,
        )
        assert result == {}

    def test_all_layers_covered(self) -> None:
        """All layers should be in result."""
        result = _interpolate_layer_alignment(
            sample_layers=[0, 9],
            sample_alignment={0: 0.2, 9: 0.8},
            layer_count=10,
        )

        for layer in range(10):
            assert layer in result


# =============================================================================
# Helper Function Tests - DateTime Encoding/Decoding
# =============================================================================


class TestDateTimeEncoding:
    """Tests for _encode_datetime and _decode_datetime."""

    def test_encode_utc_datetime(self) -> None:
        """UTC datetime should encode with Z suffix."""
        dt = datetime(2025, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        encoded = _encode_datetime(dt)
        assert encoded.endswith("Z")
        assert "2025-06-15" in encoded
        assert "12:30:45" in encoded

    def test_encode_naive_datetime(self) -> None:
        """Naive datetime should be treated as UTC."""
        dt = datetime(2025, 6, 15, 12, 30, 45)
        encoded = _encode_datetime(dt)
        assert encoded.endswith("Z")

    def test_decode_z_suffix(self) -> None:
        """Should decode Z suffix correctly."""
        decoded = _decode_datetime("2025-06-15T12:30:45Z")
        assert decoded.year == 2025
        assert decoded.month == 6
        assert decoded.day == 15
        assert decoded.hour == 12
        assert decoded.tzinfo is not None

    def test_decode_offset_format(self) -> None:
        """Should decode +00:00 format correctly."""
        decoded = _decode_datetime("2025-06-15T12:30:45+00:00")
        assert decoded.year == 2025
        assert decoded.tzinfo is not None

    def test_round_trip(self) -> None:
        """Encode/decode should preserve datetime."""
        original = datetime(2025, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        encoded = _encode_datetime(original)
        decoded = _decode_datetime(encoded)

        assert decoded.year == original.year
        assert decoded.month == original.month
        assert decoded.day == original.day
        assert decoded.hour == original.hour
        assert decoded.minute == original.minute
        assert decoded.second == original.second


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and stress tests."""

    def test_large_crm(self) -> None:
        """Should handle CRM with many layers and anchors."""
        num_anchors = 50
        num_layers = 20
        hidden_dim = 16

        anchor_ids = [f"prime:{i}" for i in range(num_anchors)]
        metadata = AnchorMetadata(
            total_count=num_anchors,
            semantic_prime_count=num_anchors,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm = ConceptResponseMatrix(
            model_identifier="large-model",
            layer_count=num_layers,
            hidden_dim=hidden_dim,
            anchor_metadata=metadata,
        )

        # Record activations for all anchors and layers
        for anchor in anchor_ids:
            layer_states = {
                layer: [float(i + layer) for i in range(hidden_dim)]
                for layer in range(num_layers)
            }
            crm.record_activations(anchor, layer_states)

        # Should be able to compute statistics
        stats = crm.compute_layer_statistics()
        assert len(stats) == num_layers

        # Should be able to compute CKA
        cka = crm.compute_cka_matrix(crm)
        assert len(cka) == num_layers
        assert len(cka[0]) == num_layers

    def test_high_dimensional_activations(self) -> None:
        """Should handle high-dimensional activations."""
        hidden_dim = 4096
        anchor_ids = ["prime:A", "prime:B", "prime:C", "prime:D"]
        metadata = AnchorMetadata(
            total_count=4,
            semantic_prime_count=4,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm = ConceptResponseMatrix(
            model_identifier="high-dim",
            layer_count=2,
            hidden_dim=hidden_dim,
            anchor_metadata=metadata,
        )

        for anchor in anchor_ids:
            crm.record_activations(anchor, {
                0: [1.0] * hidden_dim,
                1: [2.0] * hidden_dim,
            })

        stats = crm.compute_layer_statistics()
        assert len(stats) == 2
        assert stats[0].hidden_dim == hidden_dim

    def test_crm_with_sparse_activations(self) -> None:
        """Should handle CRM where some layers have no activations."""
        anchor_ids = ["prime:A", "prime:B"]
        metadata = AnchorMetadata(
            total_count=2,
            semantic_prime_count=2,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm = ConceptResponseMatrix(
            model_identifier="sparse",
            layer_count=5,
            hidden_dim=4,
            anchor_metadata=metadata,
        )

        # Only record for layers 0 and 4
        crm.record_activations("prime:A", {0: [1.0, 0.0, 0.0, 0.0], 4: [0.0, 0.0, 0.0, 1.0]})
        crm.record_activations("prime:B", {0: [0.0, 1.0, 0.0, 0.0], 4: [0.0, 0.0, 1.0, 0.0]})

        stats = crm.compute_layer_statistics()
        # Only layers 0 and 4 should have stats
        layer_indices = [s.layer for s in stats]
        assert 0 in layer_indices
        assert 4 in layer_indices
        assert len(stats) == 2

    def test_crm_default_created_at(self) -> None:
        """CRM should have default created_at timestamp."""
        metadata = AnchorMetadata(
            total_count=0,
            semantic_prime_count=0,
            computational_gate_count=0,
            anchor_ids=[],
        )
        crm = ConceptResponseMatrix(
            model_identifier="test",
            layer_count=1,
            hidden_dim=4,
            anchor_metadata=metadata,
        )

        assert crm.created_at is not None
        assert crm.created_at.tzinfo is not None

    def test_cka_matrix_different_layer_counts(self) -> None:
        """CKA matrix should handle CRMs with different layer counts."""
        crm1 = _build_crm()  # 2 layers

        anchor_ids = ["prime:A", "prime:B", "prime:C"]
        metadata = AnchorMetadata(
            total_count=3,
            semantic_prime_count=3,
            computational_gate_count=0,
            anchor_ids=anchor_ids,
        )
        crm2 = ConceptResponseMatrix(
            model_identifier="deeper",
            layer_count=4,
            hidden_dim=2,
            anchor_metadata=metadata,
        )
        for anchor in anchor_ids:
            crm2.record_activations(anchor, {
                0: [1.0, 0.0], 1: [0.5, 0.5], 2: [0.0, 1.0], 3: [0.5, 0.5]
            })

        cka = crm1.compute_cka_matrix(crm2)

        # Should be 2x4 matrix
        assert len(cka) == 2
        assert len(cka[0]) == 4
