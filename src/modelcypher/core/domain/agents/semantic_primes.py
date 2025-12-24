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
from enum import Enum


from modelcypher.core.domain.geometry import VectorMath
from modelcypher.core.domain.geometry.signature_base import LabeledSignatureMixin
from modelcypher.data import load_json


class SemanticPrimeCategory(str, Enum):
    substantives = "substantives"
    relational_substantives = "relationalSubstantives"
    determiners = "determiners"
    quantifiers = "quantifiers"
    evaluators = "evaluators"
    descriptors = "descriptors"
    mental_predicates = "mentalPredicates"
    speech = "speech"
    actions_events_movement = "actionsEventsMovement"
    location_existence_specification = "locationExistenceSpecification"
    possession = "possession"
    life_and_death = "lifeAndDeath"
    time = "time"
    place = "place"
    logical_concepts = "logicalConcepts"
    augmentor_intensifier = "augmentorIntensifier"
    similarity = "similarity"


@dataclass(frozen=True)
class SemanticPrime:
    id: str
    category: SemanticPrimeCategory
    english_exponents: list[str]

    @property
    def canonical_english(self) -> str:
        return self.english_exponents[0] if self.english_exponents else self.id


class SemanticPrimeInventory:
    _english_2014: list[SemanticPrime] | None = None

    @classmethod
    def english2014(cls) -> list[SemanticPrime]:
        if cls._english_2014 is None:
            data = load_json("semantic_primes.json")
            primes: list[SemanticPrime] = []
            for item in data.get("english2014", []):
                primes.append(
                    SemanticPrime(
                        id=str(item["id"]),
                        category=SemanticPrimeCategory(str(item["category"])),
                        english_exponents=[str(value) for value in item.get("englishExponents", [])],
                    )
                )
            cls._english_2014 = primes
        return list(cls._english_2014)


@dataclass(frozen=True)
class SemanticPrimeSignature(LabeledSignatureMixin):
    """Frozen signature for semantic primes.

    Inherits l2_normalized() and cosine_similarity() from LabeledSignatureMixin.
    """
    prime_ids: list[str]
    values: list[float]

    @staticmethod
    def mean(signatures: list[SemanticPrimeSignature]) -> SemanticPrimeSignature | None:
        if not signatures:
            return None
        first = signatures[0]
        if not all(sig.prime_ids == first.prime_ids and len(sig.values) == len(first.values) for sig in signatures):
            return None
        summed = [0.0] * len(first.values)
        for signature in signatures:
            for idx, value in enumerate(signature.values):
                summed[idx] += float(value)
        inv_count = 1.0 / float(len(signatures))
        mean_vals = [value * inv_count for value in summed]
        return SemanticPrimeSignature(prime_ids=first.prime_ids, values=mean_vals).l2_normalized()


class SemanticPrimeActivationMethod(str, Enum):
    embeddings = "embeddings"
    skipped = "skipped"


@dataclass(frozen=True)
class SemanticPrimeActivationSummary:
    @dataclass(frozen=True)
    class PrimeScore:
        prime_id: str
        english: str
        similarity: float

    method: SemanticPrimeActivationMethod
    top_primes: list[PrimeScore]
    normalized_activation_entropy: float | None
    mean_top_k_similarity: float | None
    note: str | None
