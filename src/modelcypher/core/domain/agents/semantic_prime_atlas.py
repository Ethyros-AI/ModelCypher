from __future__ import annotations

from dataclasses import dataclass

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import (
    SemanticPrimeActivationMethod,
    SemanticPrimeActivationSummary,
    SemanticPrimeInventory,
    SemanticPrimeSignature,
)
from modelcypher.core.domain.geometry import VectorMath
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.utils.text import truncate


@dataclass(frozen=True)
class SemanticPrimeAtlasConfig:
    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8


class SemanticPrimeAtlas:
    def __init__(
        self,
        configuration: SemanticPrimeAtlasConfig | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._config = configuration or SemanticPrimeAtlasConfig()
        self._inventory = SemanticPrimeInventory.english2014()
        self._embedder = embedder if embedder is not None else EmbeddingDefaults.make_default_embedder()
        self._cached_prime_embeddings: list[list[float]] | None = None

    def signature(self, text: str) -> SemanticPrimeSignature | None:
        if not self._config.enabled:
            return None
        trimmed = text.strip()
        if not trimmed or self._embedder is None:
            return None
        try:
            prime_embeddings = self._get_or_create_prime_embeddings()
            if len(prime_embeddings) != len(self._inventory):
                return None
            capped = truncate(trimmed, self._config.max_characters_per_text)
            embedded = self._embedder.embed([capped])
            if not embedded:
                return None
            text_embedding = VectorMath.l2_normalized([float(v) for v in embedded[0]])
            similarities = [
                max(0.0, VectorMath.dot(prime_vector, text_embedding) or 0.0)
                for prime_vector in prime_embeddings
            ]
            return SemanticPrimeSignature(
                prime_ids=[prime.id for prime in self._inventory],
                values=similarities,
            )
        except Exception:
            return None

    def summarize(self, text: str) -> SemanticPrimeActivationSummary:
        if not self._config.enabled:
            return SemanticPrimeActivationSummary(
                method=SemanticPrimeActivationMethod.skipped,
                top_primes=[],
                normalized_activation_entropy=None,
                mean_top_k_similarity=None,
                note="disabled",
            )
        result = self.analyze(text)
        return result[1]

    def analyze(self, text: str) -> tuple[SemanticPrimeSignature | None, SemanticPrimeActivationSummary]:
        if not self._config.enabled:
            return (
                None,
                SemanticPrimeActivationSummary(
                    method=SemanticPrimeActivationMethod.skipped,
                    top_primes=[],
                    normalized_activation_entropy=None,
                    mean_top_k_similarity=None,
                    note="disabled",
                ),
            )
        signature = self.signature(text)
        if signature is None:
            return (
                None,
                SemanticPrimeActivationSummary(
                    method=SemanticPrimeActivationMethod.skipped,
                    top_primes=[],
                    normalized_activation_entropy=None,
                    mean_top_k_similarity=None,
                    note="no_signature",
                ),
            )

        scores = [
            SemanticPrimeActivationSummary.PrimeScore(
                prime_id=prime.id,
                english=prime.canonical_english,
                similarity=float(similarity),
            )
            for prime, similarity in zip(self._inventory, signature.values)
        ]
        scores.sort(key=lambda item: item.similarity, reverse=True)
        k = max(0, min(self._config.top_k, len(scores)))
        top_k = scores[:k]
        mean_top_k = sum(score.similarity for score in top_k) / float(len(top_k)) if top_k else None
        normalized_entropy = self._normalized_entropy(signature.values)

        summary = SemanticPrimeActivationSummary(
            method=SemanticPrimeActivationMethod.embeddings,
            top_primes=top_k,
            normalized_activation_entropy=normalized_entropy,
            mean_top_k_similarity=mean_top_k,
            note=None,
        )
        return signature, summary

    def _get_or_create_prime_embeddings(self) -> list[list[float]]:
        if self._cached_prime_embeddings is not None:
            return self._cached_prime_embeddings
        if self._embedder is None:
            return []

        multilingual = SemanticPrimeMultilingualInventoryLoader.global_diverse()
        prime_embeddings: list[list[float]] = []

        for prime in self._inventory:
            texts: list[str] = []
            matching = next((p for p in multilingual.primes if p.id == prime.id), None)
            if matching is not None:
                for language in matching.languages:
                    texts.extend(language.texts)
            texts = [text.strip() for text in texts if text.strip()]

            if texts:
                embeddings = self._embedder.embed(texts)
                if embeddings:
                    prime_embeddings.append(self._centroid(embeddings))
                    continue

            fallback_embeddings = self._embedder.embed([prime.canonical_english])
            if fallback_embeddings:
                prime_embeddings.append(VectorMath.l2_normalized(fallback_embeddings[0]))
                continue

            if prime_embeddings and self._embedder is not None:
                prime_embeddings.append([0.0 for _ in range(len(prime_embeddings[0]))])
            else:
                prime_embeddings.append([])

        self._cached_prime_embeddings = prime_embeddings
        return prime_embeddings

    @staticmethod
    def _centroid(embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        dimension = len(embeddings[0])
        summed = [0.0] * dimension
        for vec in embeddings:
            for idx, value in enumerate(vec):
                summed[idx] += float(value)
        return VectorMath.l2_normalized(summed)

    @staticmethod
    def _normalized_entropy(values: list[float]) -> float | None:
        clamped = [max(0.0, float(value)) for value in values]
        total = sum(clamped)
        if total <= 0:
            return None
        probabilities = [value / total for value in clamped]
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                entropy -= prob * _safe_log(prob)
        max_entropy = _safe_log(float(len(probabilities)))
        if max_entropy <= 0:
            return None
        return entropy / max_entropy


def _safe_log(value: float) -> float:
    import math

    if value <= 0:
        return 0.0
    return math.log(value)
