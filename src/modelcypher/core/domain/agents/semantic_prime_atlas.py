"""
Semantic Prime Atlas.

Embedding-based "semantic primes" analyzer for agent/adapter telemetry.
Interpretable goal:
- Map arbitrary text to a compact, stable coordinate system (NSM primes).
- Track whether trajectories stay in a reference behavior region (drift detection).

Ported from the reference Swift implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import asyncio
import numpy as np

# Assuming VectorMath utility exists or we implement simple helpers
from modelcypher.core.domain.geometry.vector_math import VectorMath
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
            SemanticPrime("DONT_WANT", SemanticPrimeCategory.MENTAL_PREDICATES, ["don't want", "dont want"]),
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
            SemanticPrime("BE_SOMEWHERE", SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION, ["be somewhere"]),
            SemanticPrime("THERE_IS", SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION, ["there is"]),
            SemanticPrime("BE_SOMEONE_SOMETHING", SemanticPrimeCategory.LOCATION_EXISTENCE_SPECIFICATION, ["be someone", "be something"]),

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
class SemanticPrimeSignature:
    """A 65-dimensional 'prime activation' vector aligned to a specific inventory order.

    Inherits l2_normalized() and cosine_similarity() from VectorMath.
    """
    prime_ids: list[str]
    values: list[float]

    def cosine_similarity(self, other: "SemanticPrimeSignature") -> float | None:
        """Compute cosine similarity, checking label compatibility."""
        if self.prime_ids != other.prime_ids or len(self.values) != len(other.values):
            return None
        return VectorMath.cosine_similarity(self.values, other.values)

    def l2_normalized(self) -> "SemanticPrimeSignature":
        """Return L2-normalized copy of this signature."""
        normalized = VectorMath.l2_normalized(self.values)
        return SemanticPrimeSignature(self.prime_ids, normalized)


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


@dataclass
class AtlasConfiguration:
    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8


class SemanticPrimeAtlas:
    """Embedding-based 'semantic primes' analyzer."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        configuration: AtlasConfiguration = AtlasConfiguration(),
        inventory: list[SemanticPrime] | None = None
    ):
        self.config = configuration
        self.inventory = inventory or SemanticPrimeInventory.english_2014()
        self.embedder = embedder
        self._cached_prime_embeddings: list[list[float]] | None = None

    async def signature(self, text: str) -> SemanticPrimeSignature | None:
        if not self.config.enabled:
            return None

        trimmed = text.strip()
        if not trimmed:
            return None
        if self.embedder is None:
            return None

        try:
            prime_embeddings = await self._get_or_create_prime_embeddings()
            if len(prime_embeddings) != len(self.inventory):
                return None

            capped = trimmed[:self.config.max_characters_per_text]
            embeddings = await self.embedder.embed([capped])
            if not embeddings:
                return None

            text_vec = VectorMath.l2_normalized(embeddings[0])
            
            # Compute similarities
            similarities = []
            for prime_vec in prime_embeddings:
                dot = VectorMath.dot(prime_vec, text_vec)
                similarities.append(max(0.0, dot))

            return SemanticPrimeSignature(
                prime_ids=[p.id for p in self.inventory],
                values=similarities
            )
        except Exception as e:
            # print(f"Atlas signature failed: {e}")
            return None

    async def analyze(self, text: str) -> tuple[SemanticPrimeSignature | None, SemanticPrimeActivationSummary]:
        sig = await self.signature(text)
        if not sig:
            return None, SemanticPrimeActivationSummary(
                method=SemanticPrimeActivationSummary.Method.SKIPPED,
                top_primes=[],
                normalized_activation_entropy=None,
                mean_top_k_similarity=None,
                note="no_signature" if self.config.enabled else "disabled"
            )

        # Summarize
        scored = []
        for i, prime in enumerate(self.inventory):
            similarity = sig.values[i]
            scored.append(SemanticPrimeActivationSummary.PrimeScore(
                prime_id=prime.id,
                english=prime.canonical_english,
                similarity=similarity
            ))
        
        scored.sort(key=lambda x: x.similarity, reverse=True)
        top_k = scored[:self.config.top_k]
        
        mean_top_k = 0.0
        if top_k:
            mean_top_k = sum(p.similarity for p in top_k) / len(top_k)

        normalized_entropy = self._normalized_entropy(sig.values)

        return sig, SemanticPrimeActivationSummary(
            method=SemanticPrimeActivationSummary.Method.EMBEDDINGS,
            top_primes=top_k,
            normalized_activation_entropy=normalized_entropy,
            mean_top_k_similarity=mean_top_k,
            note=None
        )

    async def _get_or_create_prime_embeddings(self) -> list[list[float]]:
        if self._cached_prime_embeddings:
            return self._cached_prime_embeddings
        if self.embedder is None:
            return []

        # In Python port, we'll just embed canonical English for now (skipping complex triangulation)
        texts = [p.canonical_english for p in self.inventory]
        embeddings = await self.embedder.embed(texts)
        
        normalized = [VectorMath.l2_normalized(vec) for vec in embeddings]
        self._cached_prime_embeddings = normalized
        return normalized

    @staticmethod
    def _normalized_entropy(values: list[float]) -> float | None:
        clamped = [max(0.0, v) for v in values]
        total = sum(clamped)
        if total <= 0:
            return None
        
        probs = [v / total for v in clamped]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        
        n = max(1, len(probs))
        max_entropy = math.log(n)
        return entropy / max_entropy if max_entropy > 0 else None
