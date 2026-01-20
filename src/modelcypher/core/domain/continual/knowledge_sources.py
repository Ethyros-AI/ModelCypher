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

"""Knowledge Sources - Adapters for external knowledge retrieval.

This module provides adapters for querying external knowledge sources during
manifold completion. When the model encounters sparse regions (fog banks),
these adapters can fetch "ground truth" attractors from external sources.

The architecture is source-agnostic: all adapters implement the same interface,
allowing the completion system to blend local geometry with external knowledge
regardless of where that knowledge comes from.

Supported sources:
    - Web search (embed search results)
    - RAG / vector stores
    - Aligned models (via Universal Translator)
    - Knowledge graphs (Wikidata, etc.)
    - Domain-specific APIs

The retrieval function signature:
    (sparse_embedding, neighbor_indices) -> (attractor_vector, confidence) | None

Where:
    - sparse_embedding: The embedding of the sparse point (WHERE the confusion is)
    - neighbor_indices: Indices of dense neighbors (context)
    - attractor_vector: Target embedding to pull toward (same dimension as sparse_embedding)
    - confidence: How much to trust this attractor [0, 1]
    - None: No knowledge available for this query

Example usage:
    from modelcypher.core.domain.continual.knowledge_sources import (
        create_composite_source,
        RAGKnowledgeSource,
        WebSearchKnowledgeSource,
    )

    # Create composite source that tries multiple backends
    source = create_composite_source([
        RAGKnowledgeSource(vector_store=my_store),
        WebSearchKnowledgeSource(search_fn=my_search),
    ])

    # Use with ManifoldCompletion
    completion = ManifoldCompletion(
        model=model,
        null_space_tracker=tracker,
        knowledge_encoder=encoder,
        knowledge_retrieval_fn=source.retrieve,
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.knowledge_sources")


@dataclass
class RetrievalResult:
    """Result from a knowledge source query.

    Attributes
    ----------
    attractor : Array
        Target embedding to pull toward.
    confidence : float
        How much to trust this attractor [0, 1].
    source_name : str
        Name of the source that provided this result.
    metadata : dict
        Additional metadata (e.g., search query, document ID).
    """

    attractor: Any  # Array
    confidence: float
    source_name: str
    metadata: dict[str, Any]


class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources.

    All knowledge sources implement this interface, enabling the completion
    system to query any external source uniformly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this knowledge source."""
        ...

    @abstractmethod
    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        """Retrieve an attractor for a sparse point.

        Parameters
        ----------
        sparse_embedding : Array
            The embedding of the sparse point (WHERE the confusion is).
        neighbor_indices : list[int]
            Indices of dense neighbors (context for the query).

        Returns
        -------
        tuple[Array, float] | None
            (attractor_vector, confidence) or None if no knowledge available.
            attractor_vector has the same shape as sparse_embedding.
            confidence is in [0, 1].
        """
        ...


class CompositeKnowledgeSource(KnowledgeSource):
    """Composite source that tries multiple sources in order.

    Returns the first successful result with confidence above threshold.
    """

    def __init__(
        self,
        sources: list[KnowledgeSource],
        min_confidence: float = 0.1,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize composite source.

        Parameters
        ----------
        sources : list[KnowledgeSource]
            Sources to try, in order of preference.
        min_confidence : float
            Minimum confidence to accept a result.
        backend : Backend, optional
            Compute backend.
        """
        self._sources = sources
        self._min_confidence = min_confidence
        self._backend = backend or get_default_backend()

    @property
    def name(self) -> str:
        source_names = ", ".join(s.name for s in self._sources)
        return f"Composite({source_names})"

    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        """Try each source in order, return first successful result."""
        for source in self._sources:
            try:
                result = source.retrieve(sparse_embedding, neighbor_indices)
                if result is not None:
                    attractor, confidence = result
                    if confidence >= self._min_confidence:
                        logger.debug(
                            "Got attractor from %s with confidence %.3f",
                            source.name,
                            confidence,
                        )
                        return result
            except Exception as e:
                logger.warning("Source %s failed: %s", source.name, e)
                continue

        return None


class NullKnowledgeSource(KnowledgeSource):
    """Placeholder source that always returns None.

    Useful for testing and as a default when no external sources are configured.
    """

    @property
    def name(self) -> str:
        return "Null"

    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        return None


class CallableKnowledgeSource(KnowledgeSource):
    """Wrap any callable as a knowledge source.

    Useful for one-off functions or lambdas.
    """

    def __init__(
        self,
        fn: Callable[["Array", list[int]], tuple["Array", float] | None],
        name: str = "Callable",
    ) -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        return self._fn(sparse_embedding, neighbor_indices)


class RAGKnowledgeSource(KnowledgeSource):
    """Knowledge source backed by a RAG / vector store.

    Queries the vector store using the sparse embedding as the query vector,
    retrieves relevant documents, and returns an attractor based on the
    retrieved content.

    This is a template implementation - subclass or configure with your
    specific vector store API.
    """

    def __init__(
        self,
        query_fn: Callable[["Array", int], list[tuple["Array", float, dict[str, Any]]]],
        top_k: int = 3,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize RAG source.

        Parameters
        ----------
        query_fn : Callable
            Function to query the vector store.
            Signature: (query_embedding, top_k) -> [(embedding, score, metadata), ...]
        top_k : int
            Number of results to retrieve.
        backend : Backend, optional
            Compute backend.
        """
        self._query_fn = query_fn
        self._top_k = top_k
        self._backend = backend or get_default_backend()

    @property
    def name(self) -> str:
        return "RAG"

    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        """Query vector store and compute attractor from results."""
        b = self._backend

        # Query the vector store
        results = self._query_fn(sparse_embedding, self._top_k)

        if not results:
            return None

        # Compute weighted average of retrieved embeddings
        # Weight by similarity score
        embeddings = [r[0] for r in results]
        scores = [r[1] for r in results]

        if not scores or max(scores) <= 0:
            return None

        # Stack embeddings and compute weighted average
        stacked = b.stack(embeddings, axis=0)
        weights = b.array(scores)
        weights = weights / b.sum(weights)  # Normalize
        b.eval(weights)

        attractor = b.sum(stacked * weights[:, None], axis=0)
        b.eval(attractor)

        # Confidence = mean similarity score
        confidence = sum(scores) / len(scores)

        return attractor, confidence


class AlignedModelKnowledgeSource(KnowledgeSource):
    """Knowledge source backed by an aligned "god model".

    Uses the Universal Translator to query a larger, more capable model
    and map its response back to the target model's activation space.

    This implements the "god model" pattern - when the small model is
    confused, ask the big model what should live at those coordinates.
    """

    def __init__(
        self,
        source_model: Any,
        alignment_fn: Callable[["Array"], "Array"],
        forward_fn: Callable[["Array"], "Array"],
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize aligned model source.

        Parameters
        ----------
        source_model : Any
            The aligned source model.
        alignment_fn : Callable
            Function to map from target space to source space.
            Signature: (target_embedding) -> source_embedding
        forward_fn : Callable
            Function to get source model's output for an embedding.
            Signature: (source_embedding) -> source_output
        backend : Backend, optional
            Compute backend.
        """
        self._source_model = source_model
        self._alignment_fn = alignment_fn
        self._forward_fn = forward_fn
        self._backend = backend or get_default_backend()

    @property
    def name(self) -> str:
        return "AlignedModel"

    def retrieve(
        self,
        sparse_embedding: "Array",
        neighbor_indices: list[int],
    ) -> tuple["Array", float] | None:
        """Query aligned model and map response back to target space."""
        b = self._backend

        try:
            # Map sparse embedding to source space
            source_query = self._alignment_fn(sparse_embedding)

            # Get source model's response
            source_response = self._forward_fn(source_query)

            # The alignment function is bidirectional - use inverse to map back
            # For now, assume the response is already in target space
            # (the alignment_fn handles the inverse mapping internally)
            attractor = source_response

            # Confidence based on activation magnitude (proxy for model certainty)
            magnitude = float(b.to_scalar(b.sqrt(b.sum(attractor * attractor))))
            # Normalize to [0, 1] using softplus-like function
            confidence = 1.0 - 1.0 / (1.0 + magnitude)

            return attractor, confidence

        except Exception as e:
            logger.warning("Aligned model query failed: %s", e)
            return None


def create_composite_source(
    sources: list[KnowledgeSource],
    min_confidence: float = 0.1,
) -> CompositeKnowledgeSource:
    """Create a composite knowledge source from multiple sources.

    Parameters
    ----------
    sources : list[KnowledgeSource]
        Sources to try, in order of preference.
    min_confidence : float
        Minimum confidence to accept a result.

    Returns
    -------
    CompositeKnowledgeSource
        Configured composite source.
    """
    return CompositeKnowledgeSource(sources=sources, min_confidence=min_confidence)


def create_null_source() -> NullKnowledgeSource:
    """Create a null knowledge source (no external knowledge).

    Returns
    -------
    NullKnowledgeSource
        Source that always returns None.
    """
    return NullKnowledgeSource()
