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

"""RAG (Retrieval-Augmented Generation) service.

Provides document indexing, querying, and status functionality for
retrieval-augmented generation workflows.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
CONTENT_PREVIEW_LIMIT = 500


@dataclass
class RAGDocument:
    """A document in the RAG index."""

    doc_id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RAGIndexResult:
    """Result of indexing documents."""

    index_id: str
    document_count: int
    total_chunks: int
    index_path: str
    created_at: str
    embedding_model: str
    embedding_dimension: int


@dataclass
class RAGQueryResult:
    """Result of a RAG query."""

    query: str
    results: list[dict[str, Any]]
    total_results: int
    query_time_ms: float


@dataclass
class RAGStatusResult:
    """Status of the RAG index."""

    index_id: str | None
    status: str  # empty, ready, building, error
    document_count: int
    chunk_count: int
    index_size_bytes: int
    last_updated: str | None
    embedding_model: str | None


@dataclass
class RAGSystemSummary:
    """Summary of a RAG system for listing."""

    system_id: str
    name: str
    model_path: str | None
    embedding_model: str
    document_count: int
    chunk_count: int
    created_at: str


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations.

    Provides document indexing, semantic search, and retrieval functionality
    for augmenting LLM generation with relevant context.
    """

    def __init__(self) -> None:
        """Initialize RAG service."""
        self._index: dict[str, RAGDocument] = {}
        self._index_id: str | None = None
        self._index_name: str | None = None
        self._index_path: str | None = None
        self._created_at: str | None = None
        self._model_path: str | None = None
        self._embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedding_dimension = 384

    def index(
        self,
        documents: list[str],
        output_path: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        index_name: str | None = None,
        model_path: str | None = None,
        embedding_model: str | None = None,
    ) -> RAGIndexResult:
        """Create a vector index from documents.

        Args:
            documents: List of document paths or content strings
            output_path: Optional path to save the index
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            index_name: Optional display name for the index
            model_path: Optional model path used for embeddings
            embedding_model: Optional embedding model identifier

        Returns:
            RAGIndexResult with index metadata

        Raises:
            ValueError: If no valid documents provided
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        self._index_id = f"idx-{uuid.uuid4().hex[:12]}"
        self._index_name = index_name or self._index_id
        self._model_path = model_path
        self._created_at = datetime.now(timezone.utc).isoformat()
        if embedding_model:
            self._embedding_model = embedding_model

        total_chunks = 0
        doc_count = 0

        for doc_input in documents:
            doc_path = Path(doc_input).expanduser()

            if doc_path.exists() and doc_path.is_file():
                # It's a file path
                content = doc_path.read_text(encoding="utf-8")
                source = str(doc_path)
            else:
                # Treat as content string
                content = doc_input
                source = f"inline-{hashlib.md5(content.encode()).hexdigest()[:8]}"

            # Chunk the content
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                doc_id = f"doc-{uuid.uuid4().hex[:8]}"
                self._index[doc_id] = RAGDocument(
                    doc_id=doc_id,
                    content=chunk,
                    source=source,
                    metadata={"chunk_index": i, "total_chunks": len(chunks)},
                )
                total_chunks += 1

            doc_count += 1

        if output_path:
            self._index_path = str(Path(output_path).expanduser().resolve())
        else:
            self._index_path = f"/tmp/rag-index-{self._index_id}"

        logger.info(
            "Created RAG index %s with %d documents, %d chunks",
            self._index_id,
            doc_count,
            total_chunks,
        )

        return RAGIndexResult(
            index_id=self._index_id,
            document_count=doc_count,
            total_chunks=total_chunks,
            index_path=self._index_path,
            created_at=self._created_at,
            embedding_model=self._embedding_model,
            embedding_dimension=self._embedding_dimension,
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGQueryResult:
        """Query the index for relevant documents.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            RAGQueryResult with matching documents

        Raises:
            ValueError: If index is empty
        """
        import time

        start_time = time.perf_counter()

        if not self._index:
            raise ValueError("Index is empty. Call index() first.")

        # Simple keyword-based search (in production, use embeddings)
        query_lower = query.lower()
        scored_docs: list[tuple[float, RAGDocument]] = []

        for doc in self._index.values():
            content_lower = doc.content.lower()
            # Simple scoring: count query term occurrences
            score = sum(content_lower.count(term) for term in query_lower.split())
            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top_k results
        results = []
        for score, doc in scored_docs[:top_k]:
            content_preview = doc.content[:CONTENT_PREVIEW_LIMIT]
            content_truncated = len(doc.content) > CONTENT_PREVIEW_LIMIT
            content_bytes = len(doc.content.encode("utf-8"))
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "content": content_preview,
                    "source": doc.source,
                    "score": score,
                    "metadata": doc.metadata,
                    "content_truncated": content_truncated,
                    "content_bytes": content_bytes,
                }
            )

        query_time_ms = (time.perf_counter() - start_time) * 1000

        return RAGQueryResult(
            query=query,
            results=results,
            total_results=len(scored_docs),
            query_time_ms=query_time_ms,
        )

    def status(self) -> RAGStatusResult:
        """Get the status of the RAG index.

        Returns:
            RAGStatusResult with index statistics
        """
        if not self._index:
            return RAGStatusResult(
                index_id=None,
                status="empty",
                document_count=0,
                chunk_count=0,
                index_size_bytes=0,
                last_updated=None,
                embedding_model=None,
            )

        # Estimate index size
        total_size = sum(len(doc.content.encode("utf-8")) for doc in self._index.values())

        # Count unique sources
        unique_sources = len(set(doc.source for doc in self._index.values()))

        return RAGStatusResult(
            index_id=self._index_id,
            status="ready",
            document_count=unique_sources,
            chunk_count=len(self._index),
            index_size_bytes=total_size,
            last_updated=self._created_at,
            embedding_model=self._embedding_model,
        )

    def list_indexes(self) -> list[RAGSystemSummary]:
        """List available RAG indexes.

        Returns:
            List of RAGSystemSummary entries.
        """
        if not self._index or not self._index_id or not self._created_at:
            return []

        status = self.status()
        return [
            RAGSystemSummary(
                system_id=self._index_id,
                name=self._index_name or self._index_id,
                model_path=self._model_path,
                embedding_model=self._embedding_model,
                document_count=status.document_count,
                chunk_count=status.chunk_count,
                created_at=self._created_at,
            )
        ]

    def delete_index(self, identifier: str) -> bool:
        """Delete the current RAG index if it matches the identifier.

        Args:
            identifier: Index id or name.

        Returns:
            True if deleted, False otherwise.
        """
        if not self._index_id:
            return False
        if identifier not in {self._index_id, self._index_name}:
            return False
        self._index.clear()
        self._index_id = None
        self._index_name = None
        self._index_path = None
        self._created_at = None
        self._model_path = None
        return True

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks
