# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Comprehensive tests for geometry domain type definitions."""

from __future__ import annotations

import dataclasses
import time

import pytest

import modelcypher.core.domain.geometry.types as mod


# =============================================================================
# PermutationAlignmentResult (mutable dataclass)
# =============================================================================


class TestPermutationAlignmentResult:
    def test_instantiation_with_required_fields(self) -> None:
        result = mod.PermutationAlignmentResult(
            permutation=[0, 2, 1],
            signs=[1, -1, 1],
            match_quality=0.95,
            match_confidences=[0.9, 0.8, 0.95],
            sign_flip_count=1,
        )
        assert result.permutation == [0, 2, 1]
        assert result.signs == [1, -1, 1]
        assert result.match_quality == 0.95
        assert result.match_confidences == [0.9, 0.8, 0.95]
        assert result.sign_flip_count == 1

    def test_defaults(self) -> None:
        result = mod.PermutationAlignmentResult(
            permutation=[0, 1],
            signs=[1, 1],
            match_quality=1.0,
            match_confidences=[1.0, 1.0],
            sign_flip_count=0,
        )
        assert result.is_sparse_permutation is False
        assert result.assignment_indices is None

    def test_optional_fields_can_be_set(self) -> None:
        result = mod.PermutationAlignmentResult(
            permutation=[1, 0],
            signs=[-1, 1],
            match_quality=0.8,
            match_confidences=[0.7, 0.9],
            sign_flip_count=1,
            is_sparse_permutation=True,
            assignment_indices=[1, 0],
        )
        assert result.is_sparse_permutation is True
        assert result.assignment_indices == [1, 0]

    def test_is_mutable(self) -> None:
        result = mod.PermutationAlignmentResult(
            permutation=[0, 1],
            signs=[1, 1],
            match_quality=0.5,
            match_confidences=[0.5, 0.5],
            sign_flip_count=0,
        )
        result.match_quality = 0.99
        assert result.match_quality == 0.99


# =============================================================================
# RebasinResult (mutable dataclass)
# =============================================================================


class TestRebasinResult:
    def test_instantiation(self) -> None:
        weights = {"layer.0": [1.0, 2.0], "layer.1": [3.0, 4.0]}
        result = mod.RebasinResult(
            aligned_weights=weights,
            quality=0.98,
            sign_flip_count=2,
        )
        assert result.aligned_weights == weights
        assert result.quality == 0.98
        assert result.sign_flip_count == 2

    def test_is_mutable(self) -> None:
        result = mod.RebasinResult(
            aligned_weights={},
            quality=0.5,
            sign_flip_count=0,
        )
        result.quality = 0.75
        assert result.quality == 0.75


# =============================================================================
# DetectedConcept (frozen dataclass)
# =============================================================================


class TestDetectedConcept:
    def test_instantiation(self) -> None:
        concept = mod.DetectedConcept(
            concept_id="emotion_joy",
            category="emotion",
            similarity=0.85,
            character_span=slice(10, 25),
            trigger_text="very happy",
        )
        assert concept.concept_id == "emotion_joy"
        assert concept.category == "emotion"
        assert concept.similarity == 0.85
        assert concept.character_span == slice(10, 25)
        assert concept.trigger_text == "very happy"
        assert concept.cross_modal_similarity is None

    def test_cross_modal_similarity_default_none(self) -> None:
        concept = mod.DetectedConcept(
            concept_id="c1",
            category="cat",
            similarity=0.5,
            character_span=slice(0, 5),
            trigger_text="hello",
        )
        assert concept.cross_modal_similarity is None

    def test_cross_modal_similarity_set(self) -> None:
        concept = mod.DetectedConcept(
            concept_id="c1",
            category="cat",
            similarity=0.5,
            character_span=slice(0, 5),
            trigger_text="hello",
            cross_modal_similarity=0.72,
        )
        assert concept.cross_modal_similarity == 0.72

    def test_frozen_enforcement(self) -> None:
        concept = mod.DetectedConcept(
            concept_id="c1",
            category="cat",
            similarity=0.5,
            character_span=slice(0, 5),
            trigger_text="hello",
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            concept.similarity = 0.9  # type: ignore[misc]

    def test_character_span_is_slice(self) -> None:
        concept = mod.DetectedConcept(
            concept_id="c1",
            category="cat",
            similarity=0.5,
            character_span=slice(3, 10),
            trigger_text="example",
        )
        text = "an example of text"
        assert text[concept.character_span] == text[3:10]


# =============================================================================
# DetectionResult (frozen dataclass)
# =============================================================================


class TestDetectionResult:
    def _make_concept(self, concept_id: str = "c1") -> mod.DetectedConcept:
        return mod.DetectedConcept(
            concept_id=concept_id,
            category="test",
            similarity=0.9,
            character_span=slice(0, 5),
            trigger_text="test",
        )

    def test_instantiation(self) -> None:
        c1 = self._make_concept("alpha")
        c2 = self._make_concept("beta")
        result = mod.DetectionResult(
            model_id="model-1",
            prompt_id="prompt-1",
            response_text="some response",
            detected_concepts=(c1, c2),
            mean_similarity=0.9,
            mean_cross_modal_similarity=None,
        )
        assert result.model_id == "model-1"
        assert result.prompt_id == "prompt-1"
        assert result.response_text == "some response"
        assert len(result.detected_concepts) == 2
        assert result.mean_similarity == 0.9
        assert result.mean_cross_modal_similarity is None
        assert isinstance(result.timestamp, float)

    def test_timestamp_default_is_recent(self) -> None:
        before = time.time()
        result = mod.DetectionResult(
            model_id="m",
            prompt_id="p",
            response_text="r",
            detected_concepts=(),
            mean_similarity=0.0,
            mean_cross_modal_similarity=None,
        )
        after = time.time()
        assert before <= result.timestamp <= after

    def test_timestamp_can_be_overridden(self) -> None:
        result = mod.DetectionResult(
            model_id="m",
            prompt_id="p",
            response_text="r",
            detected_concepts=(),
            mean_similarity=0.0,
            mean_cross_modal_similarity=None,
            timestamp=1234567890.0,
        )
        assert result.timestamp == 1234567890.0

    def test_concept_sequence_property(self) -> None:
        c1 = self._make_concept("alpha")
        c2 = self._make_concept("beta")
        c3 = self._make_concept("gamma")
        result = mod.DetectionResult(
            model_id="m",
            prompt_id="p",
            response_text="r",
            detected_concepts=(c1, c2, c3),
            mean_similarity=0.9,
            mean_cross_modal_similarity=None,
        )
        assert result.concept_sequence == ("alpha", "beta", "gamma")

    def test_concept_sequence_empty(self) -> None:
        result = mod.DetectionResult(
            model_id="m",
            prompt_id="p",
            response_text="r",
            detected_concepts=(),
            mean_similarity=0.0,
            mean_cross_modal_similarity=None,
        )
        assert result.concept_sequence == ()

    def test_frozen_enforcement(self) -> None:
        result = mod.DetectionResult(
            model_id="m",
            prompt_id="p",
            response_text="r",
            detected_concepts=(),
            mean_similarity=0.0,
            mean_cross_modal_similarity=None,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.model_id = "other"  # type: ignore[misc]


# =============================================================================
# ConceptComparisonResult (frozen dataclass)
# =============================================================================


class TestConceptComparisonResult:
    def test_instantiation(self) -> None:
        result = mod.ConceptComparisonResult(
            model_a="model-a",
            model_b="model-b",
            concept_path_a=("c1", "c2", "c3"),
            concept_path_b=("c2", "c3", "c4"),
            cka=0.85,
            cosine_similarity=0.92,
            aligned_concepts=("c2", "c3"),
            unique_to_a=("c1",),
            unique_to_b=("c4",),
        )
        assert result.model_a == "model-a"
        assert result.model_b == "model-b"
        assert result.cka == 0.85
        assert result.cosine_similarity == 0.92

    def test_alignment_ratio_with_overlap(self) -> None:
        # Union of {c1,c2,c3} | {c2,c3,c4} = {c1,c2,c3,c4} = 4 elements
        # Aligned = (c2, c3) = 2 elements
        # Ratio = 2/4 = 0.5
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=("c1", "c2", "c3"),
            concept_path_b=("c2", "c3", "c4"),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=("c2", "c3"),
            unique_to_a=("c1",),
            unique_to_b=("c4",),
        )
        assert result.alignment_ratio == pytest.approx(0.5)

    def test_alignment_ratio_perfect(self) -> None:
        # All concepts aligned: union = {c1,c2}, aligned = (c1,c2)
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=("c1", "c2"),
            concept_path_b=("c1", "c2"),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=("c1", "c2"),
            unique_to_a=(),
            unique_to_b=(),
        )
        assert result.alignment_ratio == pytest.approx(1.0)

    def test_alignment_ratio_no_overlap(self) -> None:
        # Union = {c1,c2,c3,c4}, aligned = ()
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=("c1", "c2"),
            concept_path_b=("c3", "c4"),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=(),
            unique_to_a=("c1", "c2"),
            unique_to_b=("c3", "c4"),
        )
        assert result.alignment_ratio == pytest.approx(0.0)

    def test_alignment_ratio_empty_paths(self) -> None:
        # Both paths empty, union is empty, should return 1.0
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=(),
            concept_path_b=(),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=(),
            unique_to_a=(),
            unique_to_b=(),
        )
        assert result.alignment_ratio == 1.0

    def test_alignment_ratio_with_duplicates_in_paths(self) -> None:
        # Duplicates in paths: path_a has c1 twice, union is set-based
        # Union of {c1} | {c1} = {c1}, aligned = (c1,) -> ratio = 1/1 = 1.0
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=("c1", "c1"),
            concept_path_b=("c1",),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=("c1",),
            unique_to_a=(),
            unique_to_b=(),
        )
        assert result.alignment_ratio == pytest.approx(1.0)

    def test_optional_fields_none(self) -> None:
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=(),
            concept_path_b=(),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=(),
            unique_to_a=(),
            unique_to_b=(),
        )
        assert result.cka is None
        assert result.cosine_similarity is None

    def test_frozen_enforcement(self) -> None:
        result = mod.ConceptComparisonResult(
            model_a="a",
            model_b="b",
            concept_path_a=(),
            concept_path_b=(),
            cka=None,
            cosine_similarity=None,
            aligned_concepts=(),
            unique_to_a=(),
            unique_to_b=(),
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.model_a = "other"  # type: ignore[misc]


# =============================================================================
# CompositionCategory (Enum)
# =============================================================================


class TestCompositionCategory:
    def test_all_seven_values_accessible(self) -> None:
        expected = {
            "MENTAL_PREDICATE": "mentalPredicate",
            "ACTION": "action",
            "EVALUATIVE": "evaluative",
            "TEMPORAL": "temporal",
            "SPATIAL": "spatial",
            "QUANTIFIED": "quantified",
            "RELATIONAL": "relational",
        }
        for name, value in expected.items():
            member = mod.CompositionCategory[name]
            assert member.value == value

    def test_exactly_seven_members(self) -> None:
        assert len(mod.CompositionCategory) == 7

    def test_members_are_enum_instances(self) -> None:
        for member in mod.CompositionCategory:
            assert isinstance(member, mod.CompositionCategory)

    def test_not_str_subclass(self) -> None:
        # CompositionCategory is a plain Enum, not str subclass
        assert not isinstance(mod.CompositionCategory.ACTION, str)


# =============================================================================
# CompositionProbe (frozen dataclass)
# =============================================================================


class TestCompositionProbe:
    def test_instantiation(self) -> None:
        probe = mod.CompositionProbe(
            phrase="big red ball",
            components=("big", "red", "ball"),
            category=mod.CompositionCategory.EVALUATIVE,
        )
        assert probe.phrase == "big red ball"
        assert probe.components == ("big", "red", "ball")
        assert probe.category == mod.CompositionCategory.EVALUATIVE

    def test_frozen_enforcement(self) -> None:
        probe = mod.CompositionProbe(
            phrase="test",
            components=("a",),
            category=mod.CompositionCategory.ACTION,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            probe.phrase = "other"  # type: ignore[misc]


# =============================================================================
# CompositionAnalysis (frozen dataclass with is_compositional property)
# =============================================================================


class TestCompositionAnalysis:
    def _make_probe(self) -> mod.CompositionProbe:
        return mod.CompositionProbe(
            phrase="quickly ran",
            components=("quickly", "ran"),
            category=mod.CompositionCategory.ACTION,
        )

    def test_instantiation(self) -> None:
        probe = self._make_probe()
        analysis = mod.CompositionAnalysis(
            probe=probe,
            barycentric_weights=(0.6, 0.4),
            residual_norm=0.3,
            centroid_similarity=0.85,
            component_angles=(0.1, 0.2),
        )
        assert analysis.probe is probe
        assert analysis.barycentric_weights == (0.6, 0.4)
        assert analysis.residual_norm == 0.3
        assert analysis.centroid_similarity == 0.85
        assert analysis.component_angles == (0.1, 0.2)

    def test_is_compositional_true(self) -> None:
        # residual_norm < 1.0 AND centroid_similarity > 0.0 -> True
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=0.5,
            centroid_similarity=0.7,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is True

    def test_is_compositional_false_high_residual(self) -> None:
        # residual_norm >= 1.0 -> False
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=1.0,
            centroid_similarity=0.8,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is False

    def test_is_compositional_false_zero_similarity(self) -> None:
        # centroid_similarity <= 0.0 -> False
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=0.3,
            centroid_similarity=0.0,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is False

    def test_is_compositional_false_negative_similarity(self) -> None:
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=0.3,
            centroid_similarity=-0.1,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is False

    def test_is_compositional_boundary_residual_just_below(self) -> None:
        # residual_norm = 0.999 < 1.0, similarity > 0 -> True
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=0.999,
            centroid_similarity=0.001,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is True

    def test_is_compositional_boundary_residual_above(self) -> None:
        # residual_norm = 1.001 >= 1.0 -> False
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=1.001,
            centroid_similarity=0.5,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is False

    def test_is_compositional_both_bad(self) -> None:
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=2.0,
            centroid_similarity=0.0,
            component_angles=(0.1,),
        )
        assert analysis.is_compositional is False

    def test_frozen_enforcement(self) -> None:
        analysis = mod.CompositionAnalysis(
            probe=self._make_probe(),
            barycentric_weights=(0.5, 0.5),
            residual_norm=0.3,
            centroid_similarity=0.8,
            component_angles=(0.1,),
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            analysis.residual_norm = 0.0  # type: ignore[misc]


# =============================================================================
# GeometryConsistencyResult (frozen dataclass)
# =============================================================================


class TestGeometryConsistencyResult:
    def _make_analysis(self, residual: float = 0.3) -> mod.CompositionAnalysis:
        probe = mod.CompositionProbe(
            phrase="test",
            components=("t",),
            category=mod.CompositionCategory.SPATIAL,
        )
        return mod.CompositionAnalysis(
            probe=probe,
            barycentric_weights=(1.0,),
            residual_norm=residual,
            centroid_similarity=0.9,
            component_angles=(0.1,),
        )

    def test_instantiation(self) -> None:
        a1 = self._make_analysis(0.2)
        b1 = self._make_analysis(0.4)
        result = mod.GeometryConsistencyResult(
            probe_count=1,
            analyses_a=(a1,),
            analyses_b=(b1,),
            barycentric_correlation=0.95,
            angular_correlation=0.88,
            consistency_score=0.91,
        )
        assert result.probe_count == 1
        assert len(result.analyses_a) == 1
        assert len(result.analyses_b) == 1
        assert result.barycentric_correlation == 0.95
        assert result.angular_correlation == 0.88
        assert result.consistency_score == 0.91

    def test_frozen_enforcement(self) -> None:
        result = mod.GeometryConsistencyResult(
            probe_count=0,
            analyses_a=(),
            analyses_b=(),
            barycentric_correlation=0.0,
            angular_correlation=0.0,
            consistency_score=0.0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.probe_count = 5  # type: ignore[misc]


# =============================================================================
# ProcrustesResult (frozen dataclass)
# =============================================================================


class TestProcrustesResult:
    def test_instantiation(self) -> None:
        # Build minimal nested tuples
        consensus = ((1.0, 0.0), (0.0, 1.0))
        rotations = (
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, -1.0), (1.0, 0.0)),
        )
        residuals = (
            ((0.01, 0.02), (0.03, 0.04)),
        )
        result = mod.ProcrustesResult(
            consensus=consensus,
            rotations=rotations,
            scales=(1.0, 0.99),
            residuals=residuals,
            converged=True,
            iterations=5,
            alignment_error=0.01,
            per_model_errors=(0.005, 0.015),
            consensus_variance_ratio=0.98,
            sample_count=50,
            dimension=2,
            model_count=2,
        )
        assert result.converged is True
        assert result.iterations == 5
        assert result.alignment_error == 0.01
        assert result.sample_count == 50
        assert result.dimension == 2
        assert result.model_count == 2
        assert result.scales == (1.0, 0.99)
        assert result.per_model_errors == (0.005, 0.015)
        assert result.consensus_variance_ratio == 0.98

    def test_frozen_enforcement(self) -> None:
        result = mod.ProcrustesResult(
            consensus=(),
            rotations=(),
            scales=(),
            residuals=(),
            converged=False,
            iterations=0,
            alignment_error=0.0,
            per_model_errors=(),
            consensus_variance_ratio=0.0,
            sample_count=0,
            dimension=0,
            model_count=0,
        )
        with pytest.raises(
            (AttributeError, dataclasses.FrozenInstanceError),
        ):
            result.converged = True  # type: ignore[misc]


# =============================================================================
# PairwiseProcrustesResult (mutable, generic dataclass)
# =============================================================================


class TestPairwiseProcrustesResult:
    def test_instantiation_with_plain_lists(self) -> None:
        result = mod.PairwiseProcrustesResult(
            rotation=[[1.0, 0.0], [0.0, 1.0]],
            scale=1.0,
            translation=[0.0, 0.0],
            residual=0.001,
        )
        assert result.rotation == [[1.0, 0.0], [0.0, 1.0]]
        assert result.scale == 1.0
        assert result.translation == [0.0, 0.0]
        assert result.residual == 0.001

    def test_is_mutable(self) -> None:
        result = mod.PairwiseProcrustesResult(
            rotation=[[1.0]],
            scale=1.0,
            translation=[0.0],
            residual=0.5,
        )
        result.scale = 2.0
        assert result.scale == 2.0
        result.residual = 0.0
        assert result.residual == 0.0

    def test_generic_type_accepts_any_array_type(self) -> None:
        # Plain Python lists as Array stand-in
        result = mod.PairwiseProcrustesResult(
            rotation=[1, 2, 3],
            scale=0.5,
            translation=[0, 0],
            residual=0.1,
        )
        assert result.rotation == [1, 2, 3]

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(mod.PairwiseProcrustesResult)
