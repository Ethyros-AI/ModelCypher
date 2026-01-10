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

"""Base classes for concept atlases.

This module provides shared infrastructure for embedding-based concept atlases
including EmotionConceptAtlas, ComputationalGateAtlas, SemanticPrimeAtlas, and
SequenceInvariantAtlas.

Common patterns extracted:
- Configuration dataclass with enabled flag
- Concept protocol with id, canonical_name
- Signature computation (text → embedding → cosine similarities)
- Embedding caching
- Normalized entropy calculation

Atlas implementations extend these base classes and provide:
- Concept inventory (static or loaded)
- Concept embedding text extraction (varies by domain)
- Signature class with domain-specific labels
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

from modelcypher.core.domain.agents.embedding_cache import get_or_compute_embeddings
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import log_scalar
from modelcypher.core.domain.geometry.signature_base import LabeledSignatureMixin
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_batch

if TYPE_CHECKING:
    from modelcypher.ports.embedding import EmbeddingProvider


@runtime_checkable
class AtlasConcept(Protocol):
    """Protocol for concept entries in an atlas inventory.

    All atlas concepts must provide:
    - id: Unique identifier within the atlas
    - canonical_name: Display name for the concept

    Implementations may add domain-specific fields (e.g., VAD coordinates for
    emotions, category for gates, family for sequences).
    """

    @property
    def id(self) -> str:
        """Unique concept identifier."""
        ...

    @property
    def canonical_name(self) -> str:
        """Human-readable display name."""
        ...


# Type variable for concept type (allows generic atlas)
C = TypeVar("C", bound=AtlasConcept)

# Type variable for signature type
S = TypeVar("S", bound="BaseAtlasSignature")


@dataclass(frozen=True)
class BaseAtlasSignature(LabeledSignatureMixin):
    """Base class for atlas signatures.

    A signature represents the activation pattern of text across atlas concepts.
    Each value corresponds to the cosine similarity between the text embedding
    and the concept embedding.

    Subclasses should define domain-specific label attribute (e.g., emotion_ids,
    gate_ids, prime_ids) and override _get_labels() if not using standard names.

    Attributes:
        concept_ids: List of concept IDs in signature order.
        values: Cosine similarities to each concept (0 to 1 for normalized embeddings).
    """

    concept_ids: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def _get_labels(self) -> list[str] | None:
        """Return concept_ids as the dimension labels."""
        return self.concept_ids

    def _with_values(self, new_values: list[float]) -> "BaseAtlasSignature":
        """Create a copy with new values."""
        return BaseAtlasSignature(concept_ids=self.concept_ids, values=new_values)

    def top_k(self, k: int | None = None) -> list[tuple[str, float]]:
        """Get top k concepts by activation strength.

        Args:
            k: Number of concepts to return. If None, returns all.

        Returns:
            List of (concept_id, similarity) pairs sorted by similarity descending.
        """
        if not self.values:
            return []
        backend = get_default_backend()
        values_arr = backend.array(self.values)
        n = int(backend.shape(values_arr)[0])
        if k is None or k >= n:
            indices = backend.argsort(values_arr)
            select = backend.take(indices, backend.arange(0, n))
            rev = backend.take(
                select, backend.arange(backend.shape(select)[0] - 1, -1, -1)
            )
            backend.eval(rev)
            order = backend.tolist(rev)
        else:
            neg_vals = -values_arr
            kth = max(0, k - 1)
            part = backend.argpartition(neg_vals, kth)
            select = backend.take(part, backend.arange(k))
            vals = backend.take(values_arr, select)
            order_idx = backend.argsort(-vals)
            select = backend.take(select, order_idx)
            backend.eval(select)
            order = backend.tolist(select)
        if not isinstance(order, list):
            order = [int(order)]
        return [(self.concept_ids[i], self.values[i]) for i in order]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary mapping concept_id → similarity."""
        return dict(zip(self.concept_ids, self.values))


class BaseAtlas(ABC, Generic[C, S]):
    """Abstract base class for embedding-based concept atlases.

    Provides shared implementation of:
    - Embedding caching
    - Signature computation (text → embeddings → cosine similarities)
    - Normalized entropy calculation

    Subclasses must implement:
    - inventory property: Returns list of concepts
    - _get_concept_text(concept): Returns text to embed for a concept
    - _create_signature(ids, values): Creates domain-specific signature

    Example usage:
        class MyAtlas(BaseAtlas[MyConcept, MySignature]):
            def __init__(self, embedder):
                super().__init__(embedder)

            @property
            def inventory(self) -> list[MyConcept]:
                return MY_CONCEPTS

            def _get_concept_text(self, concept: MyConcept) -> str:
                return f"{concept.name}: {concept.description}"

            def _create_signature(self, ids, values) -> MySignature:
                return MySignature(concept_ids=ids, values=values)
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
    ):
        """Initialize atlas with embedder.

        Args:
            embedder: Embedding provider for computing text embeddings.
                If None, signature() will return None.
        """
        self.embedder = embedder
        self._backend = get_default_backend()
        self._cached_concept_embeddings: Any | None = None

    @property
    @abstractmethod
    def inventory(self) -> list[C]:
        """Return the list of concepts in this atlas.

        The order determines the signature dimension order.
        """
        ...

    @abstractmethod
    def _get_concept_text(self, concept: C) -> str:
        """Extract text to embed for a concept.

        Different atlases use different text representations:
        - SemanticPrimeAtlas: concept.canonical_english
        - EmotionConceptAtlas: f"{concept.name}: {concept.description}"
        - ComputationalGateAtlas: f"{concept.name}: {concept.description}"

        Args:
            concept: A concept from the inventory.

        Returns:
            Text string to embed as the concept's representation.
        """
        ...

    @abstractmethod
    def _create_signature(self, concept_ids: list[str], values: list[float]) -> S:
        """Create a domain-specific signature instance.

        Args:
            concept_ids: List of concept IDs in signature order.
            values: Cosine similarities to each concept.

        Returns:
            Signature instance of the appropriate type.
        """
        ...

    async def signature(self, text: str) -> S | None:
        """Compute activation signature for input text.

        The signature measures how strongly the text activates each concept
        in the atlas, based on cosine similarity in embedding space.

        Args:
            text: Input text to analyze.

        Returns:
            Signature with activation values for each concept, or None if:
            - Text is empty after stripping whitespace
            - No embedder is configured
            - Embedding operation fails
        """
        trimmed = text.strip()
        if not trimmed:
            return None
        if self.embedder is None:
            return None

        try:
            concept_embeddings = await self._get_or_create_concept_embeddings()
            if self._backend.shape(concept_embeddings)[0] != len(self.inventory):
                return None

            embeddings = await self.embedder.embed([trimmed])
            if not embeddings:
                return None

            text_vec = self._ensure_array(embeddings[0])
            sims = geodesic_cosine_batch(text_vec, concept_embeddings, self._backend)
            sims = self._backend.maximum(sims, self._backend.zeros_like(sims))
            self._backend.eval(sims)
            similarities = self._backend.tolist(sims)
            if not isinstance(similarities, list):
                similarities = [float(similarities)]

            return self._create_signature(
                concept_ids=[c.id for c in self.inventory],
                values=similarities,
            )
        except Exception:
            return None

    async def _get_or_create_concept_embeddings(self) -> Any:
        """Get or create cached concept embeddings.

        Embeddings are computed once and cached for the lifetime of the atlas
        instance. This avoids redundant embedding API calls.

        Returns:
            List of geodesic-normalized embedding vectors, one per concept in inventory order.
            Returns empty list if no embedder is configured.
        """
        if self._cached_concept_embeddings is not None:
            return self._cached_concept_embeddings
        if self.embedder is None:
            return self._backend.array([])

        texts = [self._get_concept_text(c) for c in self.inventory]
        concept_embeddings = await get_or_compute_embeddings(
            self.embedder,
            self._backend,
            f"{type(self).__name__}_concepts",
            texts,
        )
        if self._backend.shape(concept_embeddings)[0] == 0:
            return self._backend.array([])
        self._cached_concept_embeddings = concept_embeddings
        return concept_embeddings

    def _ensure_array(self, value: Any) -> Any:
        if hasattr(value, "shape"):
            return value
        return self._backend.array(value)

    def clear_cache(self) -> None:
        """Clear cached concept embeddings.

        Call this if the embedder changes or you want to force re-computation.
        """
        self._cached_concept_embeddings = None

    @staticmethod
    def normalized_entropy(values: list[float]) -> float | None:
        """Compute normalized entropy of activation distribution.

        Measures how spread out activations are across concepts:
        - Low entropy: activations concentrated on few concepts (specific)
        - High entropy: activations spread across many concepts (diffuse)

        Normalized to [0, 1] where 1.0 = maximum entropy (uniform distribution).

        Args:
            values: Activation values (will be clamped to non-negative and normalized).

        Returns:
            Normalized entropy in [0, 1], or None if values sum to zero.
        """
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

        # Normalize by maximum entropy (uniform distribution)
        n_arr = backend.sum(mask)
        backend.eval(entropy_arr, n_arr)
        entropy = float(backend.to_scalar(entropy_arr))
        n = int(backend.to_scalar(n_arr))
        if n <= 1:
            return 0.0
        max_entropy = log_scalar(float(n), backend)
        return entropy / max_entropy if max_entropy > 0 else 0.0
