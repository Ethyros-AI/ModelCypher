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
from pathlib import Path

import pytest

import modelcypher.core.domain.atlas.semantic_prime_frames as spf

# ---------------------------------------------------------------------------
# EnrichedPrime frozen dataclass
# ---------------------------------------------------------------------------


class TestEnrichedPrime:
    """Tests for the EnrichedPrime frozen dataclass."""

    def test_instantiation(self) -> None:
        prime = spf.EnrichedPrime(
            id="I",
            word="I",
            frames=["I am here.", "I exist."],
            contrast="You are not I.",
            exemplars=["myself", "me"],
            category="substantives",
        )
        assert prime.id == "I"
        assert prime.word == "I"
        assert prime.frames == ["I am here.", "I exist."]
        assert prime.contrast == "You are not I."
        assert prime.exemplars == ["myself", "me"]
        assert prime.category == "substantives"

    def test_field_access(self) -> None:
        prime = spf.EnrichedPrime(
            id="WANT",
            word="want",
            frames=["I want something."],
            contrast=None,
            exemplars=["desire", "wish"],
            category="mentalPredicates",
        )
        assert prime.id == "WANT"
        assert prime.word == "want"
        assert prime.contrast is None
        assert isinstance(prime.frames, list)
        assert isinstance(prime.exemplars, list)

    def test_frozen_enforcement(self) -> None:
        prime = spf.EnrichedPrime(
            id="DO",
            word="do",
            frames=["do it"],
            contrast="don't",
            exemplars=["perform"],
            category="actionsEventsMovement",
        )
        with pytest.raises(FrozenInstanceError):
            prime.id = "CHANGED"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.word = "changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.frames = []  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.contrast = "new"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.exemplars = []  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            prime.category = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EnrichedPrime.all_directional_texts property
# ---------------------------------------------------------------------------


class TestEnrichedPrimeAllDirectionalTexts:
    """Tests for the all_directional_texts property."""

    def test_combines_frames_contrast_exemplars(self) -> None:
        prime = spf.EnrichedPrime(
            id="TEST",
            word="test",
            frames=["frame1", "frame2"],
            contrast="contrast1",
            exemplars=["ex1", "ex2"],
            category="cat",
        )
        texts = prime.all_directional_texts
        assert texts == ["frame1", "frame2", "contrast1", "ex1", "ex2"]

    def test_excludes_none_contrast(self) -> None:
        prime = spf.EnrichedPrime(
            id="TEST",
            word="test",
            frames=["frame1"],
            contrast=None,
            exemplars=["ex1"],
            category="cat",
        )
        texts = prime.all_directional_texts
        assert texts == ["frame1", "ex1"]
        assert None not in texts

    def test_empty_frames_and_exemplars(self) -> None:
        prime = spf.EnrichedPrime(
            id="TEST",
            word="test",
            frames=[],
            contrast="only_contrast",
            exemplars=[],
            category="cat",
        )
        texts = prime.all_directional_texts
        assert texts == ["only_contrast"]

    def test_all_empty(self) -> None:
        prime = spf.EnrichedPrime(
            id="TEST",
            word="test",
            frames=[],
            contrast=None,
            exemplars=[],
            category="cat",
        )
        texts = prime.all_directional_texts
        assert texts == []

    def test_returns_list_of_str(self) -> None:
        prime = spf.EnrichedPrime(
            id="TEST",
            word="test",
            frames=["f1"],
            contrast="c1",
            exemplars=["e1"],
            category="cat",
        )
        texts = prime.all_directional_texts
        assert isinstance(texts, list)
        for t in texts:
            assert isinstance(t, str)


# ---------------------------------------------------------------------------
# SemanticPrimeFrames
# ---------------------------------------------------------------------------


class TestSemanticPrimeFrames:
    """Tests for the SemanticPrimeFrames class methods."""

    def setup_method(self) -> None:
        spf.SemanticPrimeFrames._enriched = None

    def teardown_method(self) -> None:
        spf.SemanticPrimeFrames._enriched = None

    def test_enriched_returns_list(self) -> None:
        result = spf.SemanticPrimeFrames.enriched()
        assert isinstance(result, list)

    def test_enriched_non_empty(self) -> None:
        result = spf.SemanticPrimeFrames.enriched()
        assert len(result) > 0, "Expected at least one enriched prime"

    def test_enriched_returns_enriched_prime_instances(self) -> None:
        result = spf.SemanticPrimeFrames.enriched()
        for item in result:
            assert isinstance(item, spf.EnrichedPrime)

    def test_enriched_caching(self) -> None:
        """Verify caching works -- second call returns same data."""
        first = spf.SemanticPrimeFrames.enriched()
        second = spf.SemanticPrimeFrames.enriched()
        assert len(first) == len(second)
        # Returns a copy, not the same list object
        assert first is not second

    def test_enriched_returns_fresh_list(self) -> None:
        a = spf.SemanticPrimeFrames.enriched()
        b = spf.SemanticPrimeFrames.enriched()
        assert a is not b
        assert a == b

    def test_each_prime_has_non_empty_id(self) -> None:
        for prime in spf.SemanticPrimeFrames.enriched():
            assert prime.id, f"Prime has empty id: {prime}"
            assert isinstance(prime.id, str)

    def test_each_prime_has_non_empty_word(self) -> None:
        for prime in spf.SemanticPrimeFrames.enriched():
            assert prime.word, f"Prime has empty word: {prime}"
            assert isinstance(prime.word, str)

    def test_each_prime_has_non_empty_category(self) -> None:
        for prime in spf.SemanticPrimeFrames.enriched():
            assert prime.category, f"Prime has empty category: {prime}"
            assert isinstance(prime.category, str)

    def test_each_prime_has_non_empty_frames(self) -> None:
        for prime in spf.SemanticPrimeFrames.enriched():
            assert isinstance(prime.frames, list)
            assert len(prime.frames) > 0, f"Prime {prime.id} has empty frames"

    def test_directional_texts_grouped(self) -> None:
        grouped = spf.SemanticPrimeFrames.directional_texts_grouped()
        assert isinstance(grouped, list)
        assert len(grouped) > 0
        for item in grouped:
            assert isinstance(item, tuple)
            assert len(item) == 2
            prime_id, texts = item
            assert isinstance(prime_id, str)
            assert isinstance(texts, list)
            for t in texts:
                assert isinstance(t, str)

    def test_directional_texts_grouped_ids_match_enriched(self) -> None:
        enriched = spf.SemanticPrimeFrames.enriched()
        grouped = spf.SemanticPrimeFrames.directional_texts_grouped()
        enriched_ids = [p.id for p in enriched]
        grouped_ids = [item[0] for item in grouped]
        assert enriched_ids == grouped_ids

    def test_all_directional_texts(self) -> None:
        all_texts = spf.SemanticPrimeFrames.all_directional_texts()
        assert isinstance(all_texts, list)
        assert len(all_texts) > 0
        for t in all_texts:
            assert isinstance(t, str)

    def test_all_directional_texts_is_flat(self) -> None:
        """Verify all_directional_texts is a flat list, not nested."""
        all_texts = spf.SemanticPrimeFrames.all_directional_texts()
        for t in all_texts:
            assert not isinstance(t, (list, tuple))

    def test_all_directional_texts_count_matches_grouped(self) -> None:
        all_texts = spf.SemanticPrimeFrames.all_directional_texts()
        grouped = spf.SemanticPrimeFrames.directional_texts_grouped()
        total_from_grouped = sum(len(texts) for _, texts in grouped)
        assert len(all_texts) == total_from_grouped


# ---------------------------------------------------------------------------
# SemanticPrimeFrames.standard_paths
# ---------------------------------------------------------------------------


class TestSemanticPrimeFramesStandardPaths:
    """Tests for the standard_paths static method."""

    def test_returns_8_paths(self) -> None:
        paths = spf.SemanticPrimeFrames.standard_paths()
        assert len(paths) == 8

    def test_returns_path_objects(self) -> None:
        from modelcypher.core.domain.geometry.traversal_coherence import (
            Path as TraversalPath,
        )

        paths = spf.SemanticPrimeFrames.standard_paths()
        for p in paths:
            assert isinstance(p, TraversalPath)

    def test_each_path_has_anchor_ids(self) -> None:
        paths = spf.SemanticPrimeFrames.standard_paths()
        for p in paths:
            assert hasattr(p, "anchor_ids")
            assert isinstance(p.anchor_ids, list)
            assert len(p.anchor_ids) >= 2, (
                f"Path should have at least 2 anchor_ids, got {len(p.anchor_ids)}"
            )

    def test_all_anchor_ids_are_strings(self) -> None:
        paths = spf.SemanticPrimeFrames.standard_paths()
        for p in paths:
            for aid in p.anchor_ids:
                assert isinstance(aid, str)
                assert aid, "anchor_id should be non-empty"

    def test_known_path_contents(self) -> None:
        """Verify a few known standard paths."""
        paths = spf.SemanticPrimeFrames.standard_paths()
        first_path = paths[0]
        assert first_path.anchor_ids == ["I", "WANT", "SOMETHING", "DO"]
        second_path = paths[1]
        assert second_path.anchor_ids == ["KNOW", "THINK", "TRUE"]
