"""Tests for EmotionConceptAtlas."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.agents.emotion_concept_atlas import (
    EmotionCategory,
    EmotionIntensity,
    EmotionConcept,
    EmotionDyad,
    EmotionConceptInventory,
    EmotionConceptSignature,
    EmotionAtlasConfiguration,
    EmotionConceptAtlas,
    OppositionPreservationScorer,
    OPPOSITION_PAIRS,
)


class TestEmotionConceptInventory:
    """Tests for EmotionConceptInventory."""

    def test_primary_emotions_count(self) -> None:
        """Should have 8 primary emotions."""
        primaries = EmotionConceptInventory.primary_emotions()
        assert len(primaries) == 8

    def test_mild_emotions_count(self) -> None:
        """Should have 8 mild intensity emotions."""
        mild = EmotionConceptInventory.mild_emotions()
        assert len(mild) == 8

    def test_intense_emotions_count(self) -> None:
        """Should have 8 intense emotions."""
        intense = EmotionConceptInventory.intense_emotions()
        assert len(intense) == 8

    def test_all_emotions_count(self) -> None:
        """Should have 24 total emotions (8 x 3 intensities)."""
        all_emotions = EmotionConceptInventory.all_emotions()
        assert len(all_emotions) == 24

    def test_primary_dyads_count(self) -> None:
        """Should have 8 primary dyads."""
        dyads = EmotionConceptInventory.primary_dyads()
        assert len(dyads) == 8

    def test_all_emotion_ids_unique(self) -> None:
        """All emotion IDs should be unique."""
        all_emotions = EmotionConceptInventory.all_emotions()
        ids = [e.id for e in all_emotions]
        assert len(ids) == len(set(ids))

    def test_all_dyad_ids_unique(self) -> None:
        """All dyad IDs should be unique."""
        dyads = EmotionConceptInventory.primary_dyads()
        ids = [d.id for d in dyads]
        assert len(ids) == len(set(ids))

    def test_primary_emotions_have_opposites(self) -> None:
        """All primary emotions should have an opposite_id."""
        primaries = EmotionConceptInventory.primary_emotions()
        for emotion in primaries:
            assert emotion.opposite_id is not None, f"{emotion.id} missing opposite"

    def test_opposite_pairs_symmetric(self) -> None:
        """Opposition should be symmetric: if A's opposite is B, B's opposite is A."""
        primaries = EmotionConceptInventory.primary_emotions()
        id_to_emotion = {e.id: e for e in primaries}

        for emotion in primaries:
            opp_id = emotion.opposite_id
            assert opp_id in id_to_emotion, f"Opposite {opp_id} not found"
            opposite = id_to_emotion[opp_id]
            assert opposite.opposite_id == emotion.id, \
                f"{emotion.id}'s opposite is {opp_id}, but {opp_id}'s opposite is {opposite.opposite_id}"

    def test_by_category_returns_all_intensities(self) -> None:
        """by_category should return emotions of all intensities for that category."""
        joy_emotions = EmotionConceptInventory.by_category(EmotionCategory.JOY)
        assert len(joy_emotions) == 3  # mild, primary, intense
        intensities = {e.intensity for e in joy_emotions}
        assert intensities == {
            EmotionIntensity.MILD,
            EmotionIntensity.PRIMARY,
            EmotionIntensity.INTENSE,
        }

    def test_get_opposite_valid(self) -> None:
        """get_opposite should return correct opposite for known emotion."""
        assert EmotionConceptInventory.get_opposite("joy") == "sadness"
        assert EmotionConceptInventory.get_opposite("sadness") == "joy"
        assert EmotionConceptInventory.get_opposite("fear") == "anger"

    def test_get_opposite_unknown(self) -> None:
        """get_opposite should return None for unknown emotion."""
        assert EmotionConceptInventory.get_opposite("unknown") is None


class TestEmotionConceptVAD:
    """Tests for VAD coordinate validity."""

    def test_valence_in_range(self) -> None:
        """All valence values should be in [-1, 1]."""
        all_emotions = EmotionConceptInventory.all_emotions()
        for emotion in all_emotions:
            assert -1.0 <= emotion.valence <= 1.0, \
                f"{emotion.id} valence {emotion.valence} out of range"

    def test_arousal_in_range(self) -> None:
        """All arousal values should be in [0, 1]."""
        all_emotions = EmotionConceptInventory.all_emotions()
        for emotion in all_emotions:
            assert 0.0 <= emotion.arousal <= 1.0, \
                f"{emotion.id} arousal {emotion.arousal} out of range"

    def test_dominance_in_range(self) -> None:
        """All dominance values should be in [0, 1]."""
        all_emotions = EmotionConceptInventory.all_emotions()
        for emotion in all_emotions:
            assert 0.0 <= emotion.dominance <= 1.0, \
                f"{emotion.id} dominance {emotion.dominance} out of range"

    def test_opposites_have_different_valence(self) -> None:
        """Opposite emotions should have contrasting valence."""
        primaries = EmotionConceptInventory.primary_emotions()
        id_to_emotion = {e.id: e for e in primaries}

        # Joy should be positive, sadness negative
        assert id_to_emotion["joy"].valence > 0
        assert id_to_emotion["sadness"].valence < 0

        # Trust should be positive, disgust negative
        assert id_to_emotion["trust"].valence > 0
        assert id_to_emotion["disgust"].valence < 0

    def test_intense_has_more_extreme_values(self) -> None:
        """Intense emotions should have more extreme arousal than mild."""
        all_emotions = EmotionConceptInventory.all_emotions()

        for category in EmotionCategory:
            by_intensity = {}
            for e in all_emotions:
                if e.category == category:
                    by_intensity[e.intensity] = e

            if EmotionIntensity.MILD in by_intensity and EmotionIntensity.INTENSE in by_intensity:
                mild = by_intensity[EmotionIntensity.MILD]
                intense = by_intensity[EmotionIntensity.INTENSE]
                # Intense should have >= arousal than mild
                assert intense.arousal >= mild.arousal, \
                    f"{category}: intense arousal {intense.arousal} < mild arousal {mild.arousal}"

    def test_vad_property_returns_tuple(self) -> None:
        """vad property should return (valence, arousal, dominance) tuple."""
        emotion = EmotionConceptInventory.primary_emotions()[0]
        vad = emotion.vad
        assert isinstance(vad, tuple)
        assert len(vad) == 3
        assert vad == (emotion.valence, emotion.arousal, emotion.dominance)


class TestEmotionDyad:
    """Tests for EmotionDyad."""

    def test_dyads_reference_valid_primaries(self) -> None:
        """Dyad primary_ids should reference existing primary emotions."""
        dyads = EmotionConceptInventory.primary_dyads()
        primary_ids = {e.id for e in EmotionConceptInventory.primary_emotions()}

        for dyad in dyads:
            assert dyad.primary_ids[0] in primary_ids, \
                f"Dyad {dyad.id} references unknown primary {dyad.primary_ids[0]}"
            assert dyad.primary_ids[1] in primary_ids, \
                f"Dyad {dyad.id} references unknown primary {dyad.primary_ids[1]}"

    def test_dyad_vad_in_range(self) -> None:
        """Dyad VAD values should be in valid ranges."""
        dyads = EmotionConceptInventory.primary_dyads()
        for dyad in dyads:
            assert -1.0 <= dyad.valence <= 1.0
            assert 0.0 <= dyad.arousal <= 1.0
            assert 0.0 <= dyad.dominance <= 1.0

    def test_dyad_has_support_texts(self) -> None:
        """All dyads should have at least one support text."""
        dyads = EmotionConceptInventory.primary_dyads()
        for dyad in dyads:
            assert len(dyad.support_texts) > 0, f"Dyad {dyad.id} has no support texts"


class TestEmotionConceptSignature:
    """Tests for EmotionConceptSignature."""

    def test_cosine_similarity_identical(self) -> None:
        """Identical signatures should have cosine similarity of 1.0."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[1.0, 0.0],
        )
        assert sig.cosine_similarity(sig) == 1.0

    def test_cosine_similarity_orthogonal(self) -> None:
        """Orthogonal signatures should have cosine similarity of 0.0."""
        sig_a = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[1.0, 0.0],
        )
        sig_b = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[0.0, 1.0],
        )
        assert sig_a.cosine_similarity(sig_b) == 0.0

    def test_cosine_similarity_mismatched_ids(self) -> None:
        """Mismatched emotion_ids should return None."""
        sig_a = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[1.0, 0.0],
        )
        sig_b = EmotionConceptSignature(
            emotion_ids=["fear", "anger"],
            values=[1.0, 0.0],
        )
        assert sig_a.cosine_similarity(sig_b) is None

    def test_l2_normalized(self) -> None:
        """l2_normalized should produce unit vector."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[3.0, 4.0],  # 3-4-5 triangle
        )
        normalized = sig.l2_normalized()
        # Should be [0.6, 0.8]
        assert abs(normalized.values[0] - 0.6) < 1e-6
        assert abs(normalized.values[1] - 0.8) < 1e-6

    def test_dominant_emotion(self) -> None:
        """dominant_emotion should return highest activation."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "anger"],
            values=[0.3, 0.7, 0.5],
        )
        dominant_id, dominant_val = sig.dominant_emotion()
        assert dominant_id == "sadness"
        assert dominant_val == 0.7

    def test_top_emotions(self) -> None:
        """top_emotions should return k highest by activation."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "anger", "fear"],
            values=[0.3, 0.9, 0.7, 0.1],
        )
        top_2 = sig.top_emotions(k=2)
        assert len(top_2) == 2
        assert top_2[0] == ("sadness", 0.9)
        assert top_2[1] == ("anger", 0.7)

    def test_opposition_balance(self) -> None:
        """opposition_balance should compute difference for each pair."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"],
            values=[0.8, 0.2, 0.5, 0.5, 0.3, 0.7, 0.6, 0.4],
        )
        balances = sig.opposition_balance()

        # joy (0.8) vs sadness (0.2) = 0.6
        assert abs(balances["joy_vs_sadness"] - 0.6) < 1e-6
        # trust (0.5) vs disgust (0.5) = 0.0
        assert abs(balances["trust_vs_disgust"]) < 1e-6
        # fear (0.3) vs anger (0.7) = -0.4
        assert abs(balances["fear_vs_anger"] - (-0.4)) < 1e-6

    def test_vad_projection_with_inventory(self) -> None:
        """vad_projection should compute weighted average of VAD coordinates."""
        emotions = EmotionConceptInventory.primary_emotions()
        sig = EmotionConceptSignature(
            emotion_ids=[e.id for e in emotions],
            values=[1.0 if e.id == "joy" else 0.0 for e in emotions],
            _inventory=emotions,
        )
        vad = sig.vad_projection()
        # Should match joy's VAD exactly
        joy = next(e for e in emotions if e.id == "joy")
        assert abs(vad[0] - joy.valence) < 1e-6
        assert abs(vad[1] - joy.arousal) < 1e-6
        assert abs(vad[2] - joy.dominance) < 1e-6

    def test_vad_projection_without_inventory(self) -> None:
        """vad_projection without inventory should return (0, 0, 0)."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy"],
            values=[1.0],
        )
        assert sig.vad_projection() == (0.0, 0.0, 0.0)


class TestOppositionPreservationScorer:
    """Tests for OppositionPreservationScorer."""

    def test_identical_signatures_perfect_preservation(self) -> None:
        """Identical signatures should have perfect preservation."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"],
            values=[0.8, 0.2, 0.6, 0.4, 0.3, 0.7, 0.5, 0.5],
        )
        score = OppositionPreservationScorer.compute_score(sig, sig)
        assert score.mean_preservation == 1.0
        assert len(score.violated_pairs) == 0

    def test_flipped_opposition_violation(self) -> None:
        """Flipped opposition should be detected as violation."""
        sig_a = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[0.8, 0.2],  # joy dominant
        )
        sig_b = EmotionConceptSignature(
            emotion_ids=["joy", "sadness"],
            values=[0.2, 0.8],  # sadness dominant (flipped)
        )
        score = OppositionPreservationScorer.compute_score(sig_a, sig_b)
        assert score.mean_preservation < 1.0
        assert "joy_vs_sadness" in score.violated_pairs

    def test_co_activation_penalty_no_coactivation(self) -> None:
        """No co-activation should have zero penalty."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"],
            values=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        )
        penalty = OppositionPreservationScorer.co_activation_penalty(sig)
        assert penalty == 0.0

    def test_co_activation_penalty_full_coactivation(self) -> None:
        """Full co-activation should have high penalty."""
        sig = EmotionConceptSignature(
            emotion_ids=["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"],
            values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        penalty = OppositionPreservationScorer.co_activation_penalty(sig)
        assert penalty == 1.0


class TestEmotionConceptAtlas:
    """Tests for EmotionConceptAtlas."""

    def test_default_configuration(self) -> None:
        """Default config should include all emotion types."""
        atlas = EmotionConceptAtlas()
        # 8 primary + 8 mild + 8 intense = 24
        assert len(atlas.inventory) == 24
        # 8 dyads
        assert len(atlas.dyads) == 8

    def test_configuration_excludes_mild(self) -> None:
        """Configuration can exclude mild emotions."""
        config = EmotionAtlasConfiguration(include_mild=False)
        atlas = EmotionConceptAtlas(configuration=config)
        # 8 primary + 8 intense = 16
        assert len(atlas.inventory) == 16

    def test_configuration_excludes_intense(self) -> None:
        """Configuration can exclude intense emotions."""
        config = EmotionAtlasConfiguration(include_intense=False)
        atlas = EmotionConceptAtlas(configuration=config)
        # 8 primary + 8 mild = 16
        assert len(atlas.inventory) == 16

    def test_configuration_excludes_dyads(self) -> None:
        """Configuration can exclude dyads."""
        config = EmotionAtlasConfiguration(include_dyads=False)
        atlas = EmotionConceptAtlas(configuration=config)
        assert len(atlas.dyads) == 0

    def test_normalized_entropy_uniform(self) -> None:
        """Uniform distribution should have entropy of 1.0."""
        atlas = EmotionConceptAtlas()
        entropy = atlas._normalized_entropy([1.0, 1.0, 1.0])
        assert entropy is not None
        assert abs(entropy - 1.0) < 1e-6

    def test_normalized_entropy_concentrated(self) -> None:
        """Concentrated distribution should have low entropy."""
        atlas = EmotionConceptAtlas()
        entropy = atlas._normalized_entropy([1.0, 0.0, 0.0])
        assert entropy is not None
        assert entropy == 0.0

    def test_vad_distance_identical(self) -> None:
        """Identical VAD projections should have distance 0."""
        atlas = EmotionConceptAtlas()
        emotions = EmotionConceptInventory.primary_emotions()
        sig = EmotionConceptSignature(
            emotion_ids=[e.id for e in emotions],
            values=[1.0 if e.id == "joy" else 0.0 for e in emotions],
            _inventory=emotions,
        )
        distance = atlas.vad_distance(sig, sig)
        assert distance == 0.0

    def test_vad_distance_opposites(self) -> None:
        """Opposite emotions should have large VAD distance."""
        atlas = EmotionConceptAtlas()
        emotions = EmotionConceptInventory.primary_emotions()

        sig_joy = EmotionConceptSignature(
            emotion_ids=[e.id for e in emotions],
            values=[1.0 if e.id == "joy" else 0.0 for e in emotions],
            _inventory=emotions,
        )
        sig_sadness = EmotionConceptSignature(
            emotion_ids=[e.id for e in emotions],
            values=[1.0 if e.id == "sadness" else 0.0 for e in emotions],
            _inventory=emotions,
        )

        distance = atlas.vad_distance(sig_joy, sig_sadness)
        # Joy and sadness have opposite valence, different arousal/dominance
        # Should be a significant distance
        assert distance > 1.0


class TestOppositionPairs:
    """Tests for OPPOSITION_PAIRS constant."""

    def test_opposition_pairs_count(self) -> None:
        """Should have 4 opposition pairs."""
        assert len(OPPOSITION_PAIRS) == 4

    def test_opposition_pairs_cover_all_categories(self) -> None:
        """Opposition pairs should cover all 8 categories."""
        covered = set()
        for cat_a, cat_b in OPPOSITION_PAIRS:
            covered.add(cat_a)
            covered.add(cat_b)
        assert covered == set(EmotionCategory)

    def test_each_category_in_exactly_one_pair(self) -> None:
        """Each category should appear in exactly one pair."""
        all_categories = []
        for cat_a, cat_b in OPPOSITION_PAIRS:
            all_categories.extend([cat_a, cat_b])
        assert len(all_categories) == 8
        assert len(set(all_categories)) == 8
