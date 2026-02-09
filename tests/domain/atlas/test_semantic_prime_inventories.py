# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.atlas.semantic_prime_frames import (
    EnrichedPrime,
    SemanticPrimeFrames,
)
from modelcypher.core.domain.atlas.semantic_primes import (
    AtlasSemanticPrimeSignature,
    SemanticPrime,
    SemanticPrimeCategory,
    SemanticPrimeInventory,
)


@pytest.fixture(autouse=True)
def _clear_inventory_caches() -> None:
    SemanticPrimeFrames._enriched = None
    SemanticPrimeInventory._english_2014 = None
    yield
    SemanticPrimeFrames._enriched = None
    SemanticPrimeInventory._english_2014 = None


def test_enriched_prime_all_directional_texts_with_and_without_contrast() -> None:
    with_contrast = EnrichedPrime(
        id="TEST",
        word="test",
        frames=["frame-a"],
        contrast="contrast-a",
        exemplars=["example-a", "example-b"],
        category="demo",
    )
    without_contrast = EnrichedPrime(
        id="TEST2",
        word="test2",
        frames=["frame-b"],
        contrast=None,
        exemplars=["example-c"],
        category="demo",
    )

    assert with_contrast.all_directional_texts == [
        "frame-a",
        "contrast-a",
        "example-a",
        "example-b",
    ]
    assert without_contrast.all_directional_texts == ["frame-b", "example-c"]


def test_semantic_prime_frames_inventory_is_non_empty_and_returns_copy() -> None:
    first = SemanticPrimeFrames.enriched()
    second = SemanticPrimeFrames.enriched()

    assert len(first) > 0
    assert first is not second
    assert first[0].id == second[0].id

    first.append(
        EnrichedPrime(
            id="TEMP",
            word="temp",
            frames=[],
            contrast=None,
            exemplars=[],
            category="temp",
        )
    )
    third = SemanticPrimeFrames.enriched()
    assert all(prime.id != "TEMP" for prime in third)


def test_semantic_prime_frames_grouped_and_flat_directional_texts_are_consistent() -> None:
    grouped = SemanticPrimeFrames.directional_texts_grouped()
    flat = SemanticPrimeFrames.all_directional_texts()

    assert len(grouped) > 0
    assert all(group_id for group_id, _ in grouped)
    assert all(isinstance(texts, list) for _, texts in grouped)
    assert sum(len(texts) for _, texts in grouped) == len(flat)
    assert len(flat) > 0


def test_semantic_prime_standard_paths_have_anchor_ids() -> None:
    paths = SemanticPrimeFrames.standard_paths()

    assert len(paths) == 8
    assert all(path.anchor_ids for path in paths)


def test_semantic_prime_canonical_english_fallback() -> None:
    prime_with_exponents = SemanticPrime(
        id="I",
        category=SemanticPrimeCategory.substantives,
        english_exponents=["I", "me"],
    )
    prime_without_exponents = SemanticPrime(
        id="SOMETHING",
        category=SemanticPrimeCategory.substantives,
        english_exponents=[],
    )

    assert prime_with_exponents.canonical_english == "I"
    assert prime_without_exponents.canonical_english == "SOMETHING"


def test_semantic_prime_inventory_loads_known_data() -> None:
    primes = SemanticPrimeInventory.english2014()

    assert len(primes) > 0
    assert all(isinstance(prime.category, SemanticPrimeCategory) for prime in primes)
    assert all(prime.id for prime in primes)


def test_semantic_prime_signature_mean_requires_compatible_signatures() -> None:
    a = AtlasSemanticPrimeSignature(prime_ids=["I", "YOU"], values=[1.0, 0.0])
    b = AtlasSemanticPrimeSignature(prime_ids=["I", "YOU"], values=[0.0, 1.0])
    incompatible = AtlasSemanticPrimeSignature(prime_ids=["YOU", "I"], values=[1.0, 0.0])

    mean_signature = AtlasSemanticPrimeSignature.mean([a, b])
    incompatible_mean = AtlasSemanticPrimeSignature.mean([a, incompatible])
    empty_mean = AtlasSemanticPrimeSignature.mean([])

    assert mean_signature is not None
    l2_norm = math.sqrt(sum(v * v for v in mean_signature.values))
    assert l2_norm == pytest.approx(1.0)
    assert incompatible_mean is None
    assert empty_mean is None

