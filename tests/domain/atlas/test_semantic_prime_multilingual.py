# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.atlas.semantic_prime_multilingual import (
    DuplicatePrimeIDsError,
    EmptyTextsError,
    LanguageTexts,
    MissingLanguagesError,
    MissingPrimeIDsError,
    MultilingualPrime,
    SemanticPrimeMultilingualInventory,
    SemanticPrimeMultilingualInventoryLoader,
    UnsupportedVersionError,
)


def _sample_inventory() -> SemanticPrimeMultilingualInventory:
    return SemanticPrimeMultilingualInventory(
        version=1,
        inventory_id="demo",
        source="test",
        notes=None,
        primes=[
            MultilingualPrime(
                id="I",
                category="substantives",
                languages=[
                    LanguageTexts(language="en", texts=["I", " me ", "I"]),
                    LanguageTexts(language="es", texts=["yo"]),
                ],
            ),
            MultilingualPrime(
                id="YOU",
                category="substantives",
                languages=[LanguageTexts(language="en", texts=["you"])],
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _clear_loader_cache() -> None:
    SemanticPrimeMultilingualInventoryLoader._core_european = None
    SemanticPrimeMultilingualInventoryLoader._global_diverse = None
    yield
    SemanticPrimeMultilingualInventoryLoader._core_european = None
    SemanticPrimeMultilingualInventoryLoader._global_diverse = None


def test_ordered_texts_filters_languages_deduplicates_and_trims() -> None:
    inventory = _sample_inventory()

    grouped = inventory.ordered_texts(["I", "YOU"], languages=[" en ", "es"])

    assert grouped[0][0] == "I"
    assert grouped[0][1] == ["I", "me", "yo"]
    assert grouped[1] == ("YOU", ["you"])


def test_ordered_texts_missing_and_invalid_inputs_raise() -> None:
    inventory = _sample_inventory()

    with pytest.raises(MissingPrimeIDsError):
        inventory.ordered_texts(["I", "NOT_PRESENT"])

    with pytest.raises(MissingLanguagesError):
        inventory.ordered_texts(["YOU"], languages=["es"], strict_languages=True)

    with pytest.raises(UnsupportedVersionError):
        inventory.ordered_texts(["I"], supported_versions={2})


def test_ordered_texts_duplicate_ids_and_empty_texts_raise() -> None:
    duplicate = SemanticPrimeMultilingualInventory(
        version=1,
        inventory_id="dup",
        source=None,
        notes=None,
        primes=[
            MultilingualPrime(id="I", category=None, languages=[LanguageTexts("en", ["I"])]),
            MultilingualPrime(id="I", category=None, languages=[LanguageTexts("en", ["me"])]),
        ],
    )
    with pytest.raises(DuplicatePrimeIDsError):
        duplicate.ordered_texts(["I"])

    empty_texts = SemanticPrimeMultilingualInventory(
        version=1,
        inventory_id="empty",
        source=None,
        notes=None,
        primes=[
            MultilingualPrime(
                id="I",
                category=None,
                languages=[LanguageTexts("en", ["   ", "\n\t"])],
            )
        ],
    )
    with pytest.raises(EmptyTextsError):
        empty_texts.ordered_texts(["I"])


def test_loader_returns_cached_inventories() -> None:
    core_1 = SemanticPrimeMultilingualInventoryLoader.core_european()
    core_2 = SemanticPrimeMultilingualInventoryLoader.core_european()
    global_1 = SemanticPrimeMultilingualInventoryLoader.global_diverse()
    global_2 = SemanticPrimeMultilingualInventoryLoader.global_diverse()

    assert core_1 is core_2
    assert global_1 is global_2
    assert core_1.inventory_id
    assert len(core_1.primes) > 0
    assert len(global_1.primes) > 0

