# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import modelcypher.core.domain.atlas.semantic_primes as semantic_primes
import modelcypher.core.domain.geometry.signature_base as signature_base


# ---------------------------------------------------------------------------
# SemanticPrimeCategory enum
# ---------------------------------------------------------------------------


class TestSemanticPrimeCategory:
    """Tests for the SemanticPrimeCategory enum."""

    _EXPECTED_VALUES = {
        "substantives": "substantives",
        "relational_substantives": "relationalSubstantives",
        "determiners": "determiners",
        "quantifiers": "quantifiers",
        "evaluators": "evaluators",
        "descriptors": "descriptors",
        "mental_predicates": "mentalPredicates",
        "speech": "speech",
        "actions_events_movement": "actionsEventsMovement",
        "location_existence_specification": "locationExistenceSpecification",
        "possession": "possession",
        "life_and_death": "lifeAndDeath",
        "time": "time",
        "place": "place",
        "logical_concepts": "logicalConcepts",
        "augmentor_intensifier": "augmentorIntensifier",
        "similarity": "similarity",
    }

    def test_all_17_members(self) -> None:
        assert len(semantic_primes.SemanticPrimeCategory) == 17

    def test_all_values_accessible(self) -> None:
        for name, value in self._EXPECTED_VALUES.items():
            member = semantic_primes.SemanticPrimeCategory[name]
            assert member.value == value

    def test_str_subclass(self) -> None:
        for cat in semantic_primes.SemanticPrimeCategory:
            assert isinstance(cat, str)

    def test_from_value(self) -> None:
        cat = semantic_primes.SemanticPrimeCategory("substantives")
        assert cat is semantic_primes.SemanticPrimeCategory.substantives

    def test_all_values_are_non_empty_strings(self) -> None:
        for cat in semantic_primes.SemanticPrimeCategory:
            assert cat.value
            assert isinstance(cat.value, str)


# ---------------------------------------------------------------------------
# SemanticPrimeActivationMethod enum
# ---------------------------------------------------------------------------


class TestSemanticPrimeActivationMethod:
    """Tests for the SemanticPrimeActivationMethod enum."""

    def test_embeddings(self) -> None:
        assert semantic_primes.SemanticPrimeActivationMethod.embeddings == "embeddings"
        assert semantic_primes.SemanticPrimeActivationMethod.embeddings.value == "embeddings"

    def test_skipped(self) -> None:
        assert semantic_primes.SemanticPrimeActivationMethod.skipped == "skipped"
        assert semantic_primes.SemanticPrimeActivationMethod.skipped.value == "skipped"

    def test_member_count(self) -> None:
        assert len(semantic_primes.SemanticPrimeActivationMethod) == 2

    def test_str_subclass(self) -> None:
        for m in semantic_primes.SemanticPrimeActivationMethod:
            assert isinstance(m, str)


# ---------------------------------------------------------------------------
# SemanticPrime frozen dataclass
# ---------------------------------------------------------------------------


class TestSemanticPrime:
    """Tests for the SemanticPrime frozen dataclass."""

    def test_instantiation(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="I",
            category=semantic_primes.SemanticPrimeCategory.substantives,
            english_exponents=["I", "me"],
        )
        assert prime.id == "I"
        assert prime.category is semantic_primes.SemanticPrimeCategory.substantives
        assert prime.english_exponents == ["I", "me"]

    def test_field_access(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="WANT",
            category=semantic_primes.SemanticPrimeCategory.mental_predicates,
            english_exponents=["want"],
        )
        assert prime.id == "WANT"
        assert prime.category.value == "mentalPredicates"
        assert isinstance(prime.english_exponents, list)

    def test_frozen_enforcement(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="TEST",
            category=semantic_primes.SemanticPrimeCategory.substantives,
            english_exponents=["test"],
        )
        with pytest.raises(FrozenInstanceError):
            prime.id = "CHANGED"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.category = semantic_primes.SemanticPrimeCategory.time  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.english_exponents = []  # type: ignore[misc]

    def test_canonical_english_returns_first_exponent(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="GOOD",
            category=semantic_primes.SemanticPrimeCategory.evaluators,
            english_exponents=["good", "well"],
        )
        assert prime.canonical_english == "good"

    def test_canonical_english_with_empty_exponents(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="ORPHAN",
            category=semantic_primes.SemanticPrimeCategory.substantives,
            english_exponents=[],
        )
        assert prime.canonical_english == "ORPHAN"

    def test_canonical_english_single_exponent(self) -> None:
        prime = semantic_primes.SemanticPrime(
            id="DO",
            category=semantic_primes.SemanticPrimeCategory.actions_events_movement,
            english_exponents=["do"],
        )
        assert prime.canonical_english == "do"


# ---------------------------------------------------------------------------
# SemanticPrimeInventory
# ---------------------------------------------------------------------------


class TestSemanticPrimeInventory:
    """Tests for the SemanticPrimeInventory class methods."""

    def setup_method(self) -> None:
        semantic_primes.SemanticPrimeInventory._english_2014 = None

    def teardown_method(self) -> None:
        semantic_primes.SemanticPrimeInventory._english_2014 = None

    def test_english2014_returns_list(self) -> None:
        result = semantic_primes.SemanticPrimeInventory.english2014()
        assert isinstance(result, list)

    def test_english2014_non_empty(self) -> None:
        result = semantic_primes.SemanticPrimeInventory.english2014()
        assert len(result) > 0, "Expected at least one semantic prime"

    def test_english2014_returns_semantic_prime_instances(self) -> None:
        result = semantic_primes.SemanticPrimeInventory.english2014()
        for p in result:
            assert isinstance(p, semantic_primes.SemanticPrime)

    def test_english2014_caching(self) -> None:
        first = semantic_primes.SemanticPrimeInventory.english2014()
        second = semantic_primes.SemanticPrimeInventory.english2014()
        assert len(first) == len(second)
        assert first is not second  # returns copy

    def test_each_prime_has_valid_category(self) -> None:
        for p in semantic_primes.SemanticPrimeInventory.english2014():
            assert isinstance(p.category, semantic_primes.SemanticPrimeCategory)

    def test_each_prime_has_non_empty_english_exponents(self) -> None:
        for p in semantic_primes.SemanticPrimeInventory.english2014():
            assert isinstance(p.english_exponents, list)
            assert len(p.english_exponents) > 0, (
                f"Prime {p.id} has empty english_exponents"
            )
            for exp in p.english_exponents:
                assert isinstance(exp, str)
                assert exp, f"Empty exponent in prime {p.id}"

    def test_all_ids_unique(self) -> None:
        primes = semantic_primes.SemanticPrimeInventory.english2014()
        ids = [p.id for p in primes]
        assert len(ids) == len(set(ids)), "Duplicate prime ids found"


# ---------------------------------------------------------------------------
# AtlasSemanticPrimeSignature
# ---------------------------------------------------------------------------


class TestAtlasSemanticPrimeSignature:
    """Tests for the AtlasSemanticPrimeSignature frozen dataclass."""

    def test_instantiation(self) -> None:
        sig = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["I", "YOU", "WANT"],
            values=[0.5, 0.3, 0.2],
        )
        assert sig.prime_ids == ["I", "YOU", "WANT"]
        assert sig.values == [0.5, 0.3, 0.2]

    def test_inherits_labeled_signature_mixin(self) -> None:
        sig = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A"],
            values=[1.0],
        )
        assert isinstance(sig, signature_base.LabeledSignatureMixin)
        assert isinstance(sig, signature_base.SignatureMixin)

    def test_frozen_enforcement(self) -> None:
        sig = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A"],
            values=[1.0],
        )
        with pytest.raises(FrozenInstanceError):
            sig.prime_ids = ["B"]  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            sig.values = [2.0]  # type: ignore[misc]

    def test_mean_empty_list(self) -> None:
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([])
        assert result is None

    def test_mean_incompatible_prime_ids(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            signature_base, "get_default_backend", lambda: any_backend
        )
        sig_a = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[1.0, 2.0],
        )
        sig_b = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["X", "Y"],
            values=[3.0, 4.0],
        )
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([sig_a, sig_b])
        assert result is None

    def test_mean_incompatible_lengths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            signature_base, "get_default_backend", lambda: any_backend
        )
        sig_a = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[1.0, 2.0],
        )
        sig_b = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[3.0, 4.0, 5.0],
        )
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([sig_a, sig_b])
        assert result is None

    def test_mean_single_signature(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            signature_base, "get_default_backend", lambda: any_backend
        )
        sig = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[3.0, 4.0],
        )
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([sig])
        assert result is not None
        assert isinstance(result, semantic_primes.AtlasSemanticPrimeSignature)
        assert result.prime_ids == ["A", "B"]
        # Result should be l2-normalized
        assert len(result.values) == 2

    def test_mean_compatible_signatures(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            signature_base, "get_default_backend", lambda: any_backend
        )
        sig_a = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[1.0, 0.0],
        )
        sig_b = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=["A", "B"],
            values=[0.0, 1.0],
        )
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([sig_a, sig_b])
        assert result is not None
        assert isinstance(result, semantic_primes.AtlasSemanticPrimeSignature)
        assert result.prime_ids == ["A", "B"]
        # Mean of [1,0] and [0,1] is [0.5, 0.5], then l2-normalized
        assert len(result.values) == 2

    def test_mean_preserves_prime_ids(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            signature_base, "get_default_backend", lambda: any_backend
        )
        ids = ["GOOD", "BAD", "IF"]
        sig_a = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=ids, values=[1.0, 2.0, 3.0]
        )
        sig_b = semantic_primes.AtlasSemanticPrimeSignature(
            prime_ids=ids, values=[4.0, 5.0, 6.0]
        )
        result = semantic_primes.AtlasSemanticPrimeSignature.mean([sig_a, sig_b])
        assert result is not None
        assert result.prime_ids == ids


# ---------------------------------------------------------------------------
# SemanticPrimeActivationSummary and PrimeScore
# ---------------------------------------------------------------------------


class TestSemanticPrimeActivationSummary:
    """Tests for SemanticPrimeActivationSummary and its nested PrimeScore."""

    def test_prime_score_instantiation(self) -> None:
        score = semantic_primes.SemanticPrimeActivationSummary.PrimeScore(
            prime_id="I",
            english="I",
            similarity=0.95,
        )
        assert score.prime_id == "I"
        assert score.english == "I"
        assert score.similarity == 0.95

    def test_prime_score_frozen(self) -> None:
        score = semantic_primes.SemanticPrimeActivationSummary.PrimeScore(
            prime_id="I",
            english="I",
            similarity=0.95,
        )
        with pytest.raises(FrozenInstanceError):
            score.prime_id = "YOU"  # type: ignore[misc]

    def test_summary_instantiation(self) -> None:
        scores = [
            semantic_primes.SemanticPrimeActivationSummary.PrimeScore(
                prime_id="I", english="I", similarity=0.9
            ),
            semantic_primes.SemanticPrimeActivationSummary.PrimeScore(
                prime_id="YOU", english="you", similarity=0.8
            ),
        ]
        summary = semantic_primes.SemanticPrimeActivationSummary(
            method=semantic_primes.SemanticPrimeActivationMethod.embeddings,
            top_primes=scores,
            normalized_activation_entropy=0.5,
            mean_top_k_similarity=0.85,
            note=None,
        )
        assert summary.method is semantic_primes.SemanticPrimeActivationMethod.embeddings
        assert len(summary.top_primes) == 2
        assert summary.normalized_activation_entropy == 0.5
        assert summary.mean_top_k_similarity == 0.85
        assert summary.note is None

    def test_summary_with_none_fields(self) -> None:
        summary = semantic_primes.SemanticPrimeActivationSummary(
            method=semantic_primes.SemanticPrimeActivationMethod.skipped,
            top_primes=[],
            normalized_activation_entropy=None,
            mean_top_k_similarity=None,
            note="Skipped due to missing embeddings",
        )
        assert summary.normalized_activation_entropy is None
        assert summary.mean_top_k_similarity is None
        assert summary.note == "Skipped due to missing embeddings"

    def test_summary_frozen(self) -> None:
        summary = semantic_primes.SemanticPrimeActivationSummary(
            method=semantic_primes.SemanticPrimeActivationMethod.embeddings,
            top_primes=[],
            normalized_activation_entropy=None,
            mean_top_k_similarity=None,
            note=None,
        )
        with pytest.raises(FrozenInstanceError):
            summary.method = semantic_primes.SemanticPrimeActivationMethod.skipped  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            summary.top_primes = []  # type: ignore[misc]
