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

"""
Semantic Prime Atlas.

Embedding-based "semantic primes" analyzer for agent/adapter telemetry.
Interpretable goal:
- Map arbitrary text to a compact, stable coordinate system (NSM primes).
- Track whether trajectories stay in a reference behavior region (drift detection).

Ported from the reference Swift implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)
from modelcypher.core.domain.geometry.signature_base import LabeledSignatureMixin
from modelcypher.ports.embedding import EmbeddingProvider


class SemanticPrimeCategory(str, Enum):
    SUBSTANTIVES = "substantives"
    RELATIONAL_SUBSTANTIVES = "relationalSubstantives"
    DETERMINERS = "determiners"
    QUANTIFIERS = "quantifiers"
    EVALUATORS = "evaluators"
    DESCRIPTORS = "descriptors"
    MENTAL_PREDICATES = "mentalPredicates"
    SPEECH = "speech"
    ACTIONS_EVENTS_MOVEMENT = "actionsEventsMovement"
    LOCATION_EXISTENCE_SPECIFICATION = "locationExistenceSpecification"
    POSSESSION = "possession"
    LIFE_AND_DEATH = "lifeAndDeath"
    TIME = "time"
    PLACE = "place"
    LOGICAL_CONCEPTS = "logicalConcepts"
    AUGMENTOR_INTENSIFIER = "augmentorIntensifier"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class SemanticPrime:
    """Natural Semantic Metalanguage (NSM) semantic prime (English exponents)."""

    id: str
    category: SemanticPrimeCategory
    english_exponents: list[str]

    @property
    def canonical_english(self) -> str:
        return self.english_exponents[0] if self.english_exponents else self.id


class SemanticPrimeInventory:
    """Proposed semantic primes (English exponents) after Goddard & Wierzbicka (2014)."""

    @staticmethod
    def english_2014() -> list[SemanticPrime]:
        return [
            # Substantives
            SemanticPrime("I", SemanticPrimeCategory.SUBSTANTIVES, ["i", "me"]),
            SemanticPrime("YOU", SemanticPrimeCategory.SUBSTANTIVES, ["you"]),
            SemanticPrime("SOMEONE", SemanticPrimeCategory.SUBSTANTIVES, ["someone"]),
            SemanticPrime("SOMETHING", SemanticPrimeCategory.SUBSTANTIVES, ["something", "thing"]),
            SemanticPrime("PEOPLE", SemanticPrimeCategory.SUBSTANTIVES, ["people"]),
            SemanticPrime("BODY", SemanticPrimeCategory.SUBSTANTIVES, ["body"]),
            # Relational substantives
            SemanticPrime("KIND", SemanticPrimeCategory.RELATIONAL_SUBSTANTIVES, ["kind", "kinds"]),
            SemanticPrime("PART", SemanticPrimeCategory.RELATIONAL_SUBSTANTIVES, ["part", "parts"]),
            # Determiners
            SemanticPrime("THIS", SemanticPrimeCategory.DETERMINERS, ["this"]),
            SemanticPrime("THE_SAME", SemanticPrimeCategory.DETERMINERS, ["the same"]),
            SemanticPrime("OTHER", SemanticPrimeCategory.DETERMINERS, ["other", "else"]),
            # Quantifiers
            SemanticPrime("ONE", SemanticPrimeCategory.QUANTIFIERS, ["one"]),
            SemanticPrime("TWO", SemanticPrimeCategory.QUANTIFIERS, ["two"]),
            SemanticPrime("SOME", SemanticPrimeCategory.QUANTIFIERS, ["some"]),
            SemanticPrime("ALL", SemanticPrimeCategory.QUANTIFIERS, ["all"]),
            SemanticPrime("MUCH_MANY", SemanticPrimeCategory.QUANTIFIERS, ["much", "many"]),
            SemanticPrime("LITTLE_FEW", SemanticPrimeCategory.QUANTIFIERS, ["little", "few"]),
            # Evaluators
            SemanticPrime("GOOD", SemanticPrimeCategory.EVALUATORS, ["good"]),
            SemanticPrime("BAD", SemanticPrimeCategory.EVALUATORS, ["bad"]),
            # Descriptors
            SemanticPrime("BIG", SemanticPrimeCategory.DESCRIPTORS, ["big"]),
            SemanticPrime("SMALL", SemanticPrimeCategory.DESCRIPTORS, ["small"]),
            # Mental predicates
            SemanticPrime("KNOW", SemanticPrimeCategory.MENTAL_PREDICATES, ["know"]),
            SemanticPrime("THINK", SemanticPrimeCategory.MENTAL_PREDICATES, ["think"]),
            SemanticPrime("WANT", SemanticPrimeCategory.MENTAL_PREDICATES, ["want"]),
            SemanticPrime(
                "DONT_WANT", SemanticPrimeCategory.MENTAL_PREDICATES, ["don't want", "dont want"]
            ),
            SemanticPrime("FEEL", SemanticPrimeCategory.MENTAL_PREDICATES, ["feel"]),
            SemanticPrime("SEE", SemanticPrimeCategory.MENTAL_PREDICATES, ["see"]),
            SemanticPrime("HEAR", SemanticPrimeCategory.MENTAL_PREDICATES, ["hear"]),
            # Speech
            SemanticPrime("SAY", SemanticPrimeCategory.SPEECH, ["say"]),
            SemanticPrime("WORDS", SemanticPrimeCategory.SPEECH, ["words"]),
            SemanticPrime("TRUE", SemanticPrimeCategory.SPEECH, ["true"]),
            # Actions, events, movement
            SemanticPrime("DO", SemanticPrimeCategory.ACTIONS_EVENTS_MOVEMENT, ["do"]),
            SemanticPrime("HAPPEN", SemanticPrimeCategory.ACTIONS_EVENTS_MOVEMENT, ["happen"]),
            SemanticPrime("MOVE", SemanticPrimeCategory.ACTIONS_EVENTS_MOVEMENT, ["move"]),
            # Location, existence...
            SemanticPrime(
                "BE_SOMEWHERE",
                SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION,
                ["be somewhere"],
            ),
            SemanticPrime(
                "THERE_IS", SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION, ["there is"]
            ),
            SemanticPrime(
                "BE_SOMEONE_SOMETHING",
                SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION,
                ["be someone", "be something"],
            ),
            # Possession
            SemanticPrime("MINE", SemanticPrimeCategory.POSSESSION, ["mine"]),
            # Life and death
            SemanticPrime("LIVE", SemanticPrimeCategory.LIFE_AND_DEATH, ["live"]),
            SemanticPrime("DIE", SemanticPrimeCategory.LIFE_AND_DEATH, ["die"]),
            # Time
            SemanticPrime("WHEN_TIME", SemanticPrimeCategory.TIME, ["when", "time"]),
            SemanticPrime("NOW", SemanticPrimeCategory.TIME, ["now"]),
            SemanticPrime("BEFORE", SemanticPrimeCategory.TIME, ["before"]),
            SemanticPrime("AFTER", SemanticPrimeCategory.TIME, ["after"]),
            SemanticPrime("A_LONG_TIME", SemanticPrimeCategory.TIME, ["a long time"]),
            SemanticPrime("A_SHORT_TIME", SemanticPrimeCategory.TIME, ["a short time"]),
            SemanticPrime("FOR_SOME_TIME", SemanticPrimeCategory.TIME, ["for some time"]),
            SemanticPrime("MOMENT", SemanticPrimeCategory.TIME, ["moment"]),
            # Place
            SemanticPrime("WHERE_PLACE", SemanticPrimeCategory.PLACE, ["where", "place"]),
            SemanticPrime("HERE", SemanticPrimeCategory.PLACE, ["here"]),
            SemanticPrime("ABOVE", SemanticPrimeCategory.PLACE, ["above"]),
            SemanticPrime("BELOW", SemanticPrimeCategory.PLACE, ["below"]),
            SemanticPrime("FAR", SemanticPrimeCategory.PLACE, ["far"]),
            SemanticPrime("NEAR", SemanticPrimeCategory.PLACE, ["near"]),
            SemanticPrime("SIDE", SemanticPrimeCategory.PLACE, ["side"]),
            SemanticPrime("INSIDE", SemanticPrimeCategory.PLACE, ["inside"]),
            SemanticPrime("TOUCH", SemanticPrimeCategory.PLACE, ["touch"]),
            # Logical concepts
            SemanticPrime("NOT", SemanticPrimeCategory.LOGICAL_CONCEPTS, ["not"]),
            SemanticPrime("MAYBE", SemanticPrimeCategory.LOGICAL_CONCEPTS, ["maybe"]),
            SemanticPrime("CAN", SemanticPrimeCategory.LOGICAL_CONCEPTS, ["can"]),
            SemanticPrime("BECAUSE", SemanticPrimeCategory.LOGICAL_CONCEPTS, ["because"]),
            SemanticPrime("IF", SemanticPrimeCategory.LOGICAL_CONCEPTS, ["if"]),
            # Augmentor
            SemanticPrime("VERY", SemanticPrimeCategory.AUGMENTOR_INTENSIFIER, ["very"]),
            SemanticPrime("MORE", SemanticPrimeCategory.AUGMENTOR_INTENSIFIER, ["more"]),
            # Similarity
            SemanticPrime("LIKE", SemanticPrimeCategory.SIMILARITY, ["like", "as"]),
        ]


@dataclass
class SemanticPrimeSignature(LabeledSignatureMixin):
    """A 65-dimensional 'prime activation' vector aligned to a specific inventory order.

    Inherits l2_normalized() and cosine_similarity() from LabeledSignatureMixin.
    """

    prime_ids: list[str]
    values: list[float]


@dataclass
class SemanticPrimeActivationSummary:
    class Method(str, Enum):
        EMBEDDINGS = "embeddings"
        SKIPPED = "skipped"

    @dataclass
    class PrimeScore:
        prime_id: str
        english: str
        similarity: float

    method: Method
    top_primes: list[PrimeScore]
    normalized_activation_entropy: float | None
    mean_top_k_similarity: float | None
    note: str | None


class SemanticPrimeAtlas:
    """Embedding-based 'semantic primes' analyzer.

    Returns ALL primes sorted by similarity. No configuration needed.
    The geometry of the embedding space determines significance.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        inventory: list[SemanticPrime] | None = None,
    ):
        self.inventory = inventory or SemanticPrimeInventory.english_2014()
        self.embedder = embedder
        self._backend = get_default_backend()
        self._cached_prime_embeddings: Any | None = None

    async def signature(self, text: str) -> SemanticPrimeSignature | None:
        trimmed = text.strip()
        if not trimmed:
            return None
        if self.embedder is None:
            return None

        try:
            prime_embeddings = await self._get_or_create_prime_embeddings()
            if self._backend.shape(prime_embeddings)[0] != len(self.inventory):
                return None

            # Embedder handles any necessary truncation
            embeddings = await self.embedder.embed([trimmed])
            if not embeddings:
                return None

            text_vec = self._normalize_vector(embeddings[0])
            sims = self._backend.matmul(
                prime_embeddings,
                self._backend.reshape(text_vec, (-1, 1)),
            )
            sims = self._backend.reshape(
                sims, (self._backend.shape(prime_embeddings)[0],)
            )
            sims = self._backend.maximum(sims, self._backend.zeros_like(sims))
            self._backend.eval(sims)
            similarities = self._backend.tolist(sims)
            if not isinstance(similarities, list):
                similarities = [float(similarities)]

            return SemanticPrimeSignature(
                prime_ids=[p.id for p in self.inventory], values=similarities
            )
        except Exception:
            return None

    async def analyze(
        self, text: str
    ) -> tuple[SemanticPrimeSignature | None, SemanticPrimeActivationSummary]:
        sig = await self.signature(text)
        if not sig:
            return None, SemanticPrimeActivationSummary(
                method=SemanticPrimeActivationSummary.Method.SKIPPED,
                top_primes=[],
                normalized_activation_entropy=None,
                mean_top_k_similarity=None,
                note="no_signature",
            )

        # Return ALL primes sorted by similarity - geometry determines significance
        scored = []
        for i, prime in enumerate(self.inventory):
            similarity = sig.values[i]
            scored.append(
                SemanticPrimeActivationSummary.PrimeScore(
                    prime_id=prime.id, english=prime.canonical_english, similarity=similarity
                )
            )

        scored.sort(key=lambda x: x.similarity, reverse=True)

        # Mean of all similarities (not arbitrary top_k)
        mean_similarity = self._mean_similarity(sig.values)

        normalized_entropy = self._normalized_entropy(sig.values)

        return sig, SemanticPrimeActivationSummary(
            method=SemanticPrimeActivationSummary.Method.EMBEDDINGS,
            top_primes=scored,  # All primes, sorted
            normalized_activation_entropy=normalized_entropy,
            mean_top_k_similarity=mean_similarity,
            note=None,
        )

    async def _get_or_create_prime_embeddings(self) -> Any:
        if self._cached_prime_embeddings is not None:
            return self._cached_prime_embeddings
        if self.embedder is None:
            return self._backend.array([])

        # In Python port, we'll just embed canonical English for now (skipping complex triangulation)
        texts = [p.canonical_english for p in self.inventory]
        embeddings = await self.embedder.embed(texts)
        if not embeddings:
            return self._backend.array([])

        normalized = self._normalize_rows(embeddings)
        self._cached_prime_embeddings = normalized
        return normalized

    @staticmethod
    def _normalized_entropy(values: list[float]) -> float | None:
        if not values:
            return None
        backend = get_default_backend()
        arr = backend.array(values)
        clamped = backend.maximum(arr, backend.zeros_like(arr))
        total_arr = backend.sum(clamped)
        backend.eval(total_arr)
        total = float(backend.to_scalar(total_arr))
        if total <= 0:
            return None

        probs = clamped / total
        mask = probs > 0
        safe_probs = backend.where(mask, probs, backend.ones_like(probs))
        log_probs = backend.log(safe_probs)
        entropy_arr = -backend.sum(probs * log_probs)

        n_arr = backend.sum(mask)
        backend.eval(entropy_arr, n_arr)
        entropy = float(backend.to_scalar(entropy_arr))
        n = int(backend.to_scalar(n_arr))
        if n <= 1:
            return 0.0
        max_entropy = log_scalar(float(n), backend)
        return entropy / max_entropy if max_entropy > 0 else None

    def _ensure_array(self, value: Any) -> Any:
        if hasattr(value, "shape"):
            return value
        return self._backend.array(value)

    def _normalize_rows(self, matrix: Any) -> Any:
        matrix_arr = self._ensure_array(matrix)
        norms = self._backend.norm(matrix_arr, axis=1, keepdims=True)
        eps = division_epsilon(self._backend, matrix_arr)
        safe_norms = self._backend.where(
            norms > eps, norms, self._backend.ones_like(norms)
        )
        return matrix_arr / safe_norms

    def _normalize_vector(self, vector: Any) -> Any:
        vector_arr = self._ensure_array(vector)
        norm = self._backend.norm(vector_arr)
        eps = division_epsilon(self._backend, vector_arr)
        safe_norm = self._backend.where(
            norm > eps, norm, self._backend.ones_like(norm)
        )
        return vector_arr / safe_norm

    def _mean_similarity(self, values: list[float]) -> float:
        if not values:
            return 0.0
        scores = self._backend.array(values)
        mean_score = self._backend.mean(scores)
        self._backend.eval(mean_score)
        return float(self._backend.to_scalar(mean_score))
