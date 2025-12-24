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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from modelcypher.data import load_json


@dataclass(frozen=True)
class LanguageTexts:
    language: str
    texts: list[str]


@dataclass(frozen=True)
class MultilingualPrime:
    id: str
    category: str | None
    languages: list[LanguageTexts]


class MultilingualInventoryError(ValueError):
    pass


class UnsupportedVersionError(MultilingualInventoryError):
    pass


class DuplicatePrimeIDsError(MultilingualInventoryError):
    def __init__(self, ids: list[str]) -> None:
        super().__init__(f"Duplicate prime IDs in multilingual inventory: {', '.join(ids)}.")


class MissingPrimeIDsError(MultilingualInventoryError):
    def __init__(self, ids: list[str]) -> None:
        super().__init__(f"Missing prime IDs in multilingual inventory: {', '.join(ids)}.")


class MissingLanguagesError(MultilingualInventoryError):
    def __init__(self, prime_id: str, requested: list[str], available: list[str]) -> None:
        super().__init__(
            f"Prime '{prime_id}' is missing requested languages ({', '.join(requested)}). "
            f"Available: {', '.join(available)}."
        )


class EmptyTextsError(MultilingualInventoryError):
    def __init__(self, prime_id: str) -> None:
        super().__init__(f"Prime '{prime_id}' has no usable texts after normalization.")


@dataclass(frozen=True)
class SemanticPrimeMultilingualInventory:
    version: int
    inventory_id: str
    source: str | None
    notes: str | None
    primes: list[MultilingualPrime]

    def ordered_texts(
        self,
        prime_ids_in_order: list[str],
        languages: Iterable[str] | None = None,
        strict_languages: bool = True,
        supported_versions: set[int] | None = None,
    ) -> list[tuple[str, list[str]]]:
        supported = supported_versions or {1}
        if self.version not in supported:
            raise UnsupportedVersionError(f"Unsupported inventory version: {self.version}.")

        grouped: dict[str, list[MultilingualPrime]] = {}
        for prime in self.primes:
            grouped.setdefault(prime.id, []).append(prime)

        duplicate_ids = sorted([pid for pid, values in grouped.items() if len(values) > 1])
        if duplicate_ids:
            raise DuplicatePrimeIDsError(duplicate_ids)

        normalized_languages: list[str] | None = None
        if languages is not None:
            normalized_languages = sorted({lang.strip() for lang in languages if lang.strip()})

        missing_ids: list[str] = []
        result: list[tuple[str, list[str]]] = []

        for prime_id in prime_ids_in_order:
            prime_list = grouped.get(prime_id)
            if not prime_list:
                missing_ids.append(prime_id)
                continue
            prime = prime_list[0]
            available_languages = [bucket.language for bucket in prime.languages]
            if normalized_languages is None:
                selected = prime.languages
            else:
                selected = [bucket for bucket in prime.languages if bucket.language in normalized_languages]
                if strict_languages and normalized_languages and not selected:
                    raise MissingLanguagesError(prime_id, normalized_languages, available_languages)

            flattened: list[str] = []
            for bucket in selected:
                for raw in bucket.texts:
                    trimmed = raw.strip()
                    if trimmed:
                        flattened.append(trimmed)

            seen: set[str] = set()
            unique: list[str] = []
            for text in flattened:
                if text not in seen:
                    seen.add(text)
                    unique.append(text)

            if not unique:
                raise EmptyTextsError(prime_id)

            result.append((prime_id, unique))

        if missing_ids:
            raise MissingPrimeIDsError(sorted(missing_ids))

        return result


class SemanticPrimeMultilingualInventoryLoader:
    _core_european: SemanticPrimeMultilingualInventory | None = None
    _global_diverse: SemanticPrimeMultilingualInventory | None = None

    @classmethod
    def core_european(cls) -> SemanticPrimeMultilingualInventory:
        if cls._core_european is None:
            cls._core_european = _load_inventory("coreEuropean")
        return cls._core_european

    @classmethod
    def global_diverse(cls) -> SemanticPrimeMultilingualInventory:
        if cls._global_diverse is None:
            cls._global_diverse = _load_inventory("globalDiverse")
        return cls._global_diverse


def _load_inventory(key: str) -> SemanticPrimeMultilingualInventory:
    raw = load_json("semantic_prime_multilingual.json")
    payload = raw[key]
    primes: list[MultilingualPrime] = []
    for prime in payload.get("primes", []):
        languages: list[LanguageTexts] = []
        for bucket in prime.get("languages", []):
            languages.append(
                LanguageTexts(
                    language=str(bucket["language"]),
                    texts=[str(text) for text in bucket.get("texts", [])],
                )
            )
        primes.append(
            MultilingualPrime(
                id=str(prime["id"]),
                category=str(prime.get("category")) if prime.get("category") is not None else None,
                languages=languages,
            )
        )
    return SemanticPrimeMultilingualInventory(
        version=int(payload.get("version", 1)),
        inventory_id=str(payload.get("inventoryID") or payload.get("inventoryId")),
        source=payload.get("source"),
        notes=payload.get("notes"),
        primes=primes,
    )
