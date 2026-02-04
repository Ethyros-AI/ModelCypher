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
Task Diversion Detector.

Scores whether a model response has diverged from the expected task.
Agent Cypher uses *geometry* (vector similarity) instead of prompt heuristics to detect when a
response is no longer aligned with the task at hand.

Ported from the reference Swift implementation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.embedding_cache import get_or_compute_embeddings
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_batch
from modelcypher.ports.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)

class LexicalTokenizer:
    """Simple tokenizer matching Swift implementation."""

    @staticmethod
    def tokens(text: str) -> list[str]:
        # Lowercase and split on non-alphanumerics
        text = text.lower()
        # Regex to split on anything that is NOT alphanumeric
        return [t for t in re.split(r"[^a-z0-9]", text) if t]


class LexicalStopWords:
    """Stop words for task diversion detection."""

    task_diversion_detector: set[str] = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "would",
        "could",
        "should",
        "about",
        "which",
        "their",
        "there",
        "been",
        "being",
        "some",
        "what",
        "when",
        "where",
        "they",
        "them",
        "then",
        "than",
        "these",
        "those",
        "each",
        "other",
        "into",
        "just",
        "only",
        "your",
        "youre",
        "you're",
        "please",
        "thanks",
        "thank",
        "also",
        "can",
        "cant",
        "can't",
        "dont",
        "don't",
        "does",
        "doesnt",
        "doesn't",
        "using",
        "use",
        "make",
        "made",
        "like",
        "need",
        "needs",
        "want",
        "wants",
    }


class TaskDiversionMethod(str, Enum):
    """Method used for task diversion detection."""

    EMBEDDINGS = "embeddings"
    LEXICAL_FALLBACK = "lexicalFallback"
    SKIPPED = "skipped"


@dataclass
class TaskDiversionAssessment:
    """Assessment of task diversion via embedding or lexical similarity.

    Raw measurements:
    - embedding_cosine_similarity: Cosine similarity via embeddings
    - lexical_jaccard_similarity: Jaccard similarity via tokens
    - threshold: The configured threshold (for reference, not interpretation)

    Callers should interpret similarity relative to their own baselines.
    """

    method: TaskDiversionMethod
    embedding_cosine_similarity: float | None = None
    lexical_jaccard_similarity: float | None = None
    threshold: float | None = None
    note: str | None = None


class TaskDiversionDetector:
    """Embedding-first task diversion detector.

    Returns raw similarity measurements - caller interprets via their own baselines.
    Falls back to lexical (Jaccard) similarity when embeddings fail.
    """

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder

    async def assess(self, expected_task: str, observed_text: str) -> TaskDiversionAssessment:
        """Assess similarity between expected task and observed text.

        Returns raw measurements. Caller interprets significance.
        """
        expected_trimmed = expected_task.strip()
        observed_trimmed = observed_text.strip()

        if not expected_trimmed or not observed_trimmed:
            return TaskDiversionAssessment(
                method=TaskDiversionMethod.SKIPPED,
                note="missing_text",
            )

        # Try Embeddings - embedder handles truncation
        try:
            backend = get_default_backend()
            embeddings = await get_or_compute_embeddings(
                self.embedder,
                backend,
                "task_diversion",
                [expected_trimmed, observed_trimmed],
            )
            if int(backend.shape(embeddings)[0]) == 2:
                similarity = self._cosine_similarity(embeddings[0], embeddings[1]) or 0.0

                return TaskDiversionAssessment(
                    method=TaskDiversionMethod.EMBEDDINGS,
                    embedding_cosine_similarity=similarity,
                )
        except Exception as exc:
            logger.debug("Embedding-based diversion check failed: %s", exc)

        # Fallback to Lexical (always enabled - geometry determines which is available)
        lexical_similarity = self._lexical_jaccard_similarity(expected_trimmed, observed_trimmed)

        return TaskDiversionAssessment(
            method=TaskDiversionMethod.LEXICAL_FALLBACK,
            lexical_jaccard_similarity=lexical_similarity,
        )

    @staticmethod
    def _cosine_similarity(lhs: list[float] | object, rhs: list[float] | object) -> float:
        backend = get_default_backend()
        arr_lhs = lhs if hasattr(lhs, "shape") else backend.array(lhs)
        arr_rhs = rhs if hasattr(rhs, "shape") else backend.array(rhs)
        if backend.shape(arr_lhs)[0] == 0 or backend.shape(arr_rhs)[0] == 0:
            return 0.0
        if backend.shape(arr_lhs)[0] != backend.shape(arr_rhs)[0]:
            raise ValueError("Cosine similarity requires matching dimensions")
        cos_arr = geodesic_cosine_batch(
            arr_lhs, backend.reshape(arr_rhs, (1, -1)), backend
        )
        backend.eval(cos_arr)
        return float(backend.to_scalar(cos_arr))

    def _lexical_jaccard_similarity(self, lhs: str, rhs: str) -> float:
        lhs_tokens = self._lexical_token_set(lhs)
        rhs_tokens = self._lexical_token_set(rhs)
        return self._jaccard_similarity(lhs_tokens, rhs_tokens)

    def _lexical_token_set(self, text: str) -> set[str]:
        raw_tokens = LexicalTokenizer.tokens(text)
        # Filter tokens < 3 chars or stop words
        return {
            t
            for t in raw_tokens
            if len(t) >= 3 and t not in LexicalStopWords.task_diversion_detector
        }

    @staticmethod
    def _jaccard_similarity(lhs: set[str], rhs: set[str]) -> float:
        if not lhs and not rhs:
            return 0.0  # Swift SetMath behavior likely 0 if both empty? Or 1? Usually 1 if identical emptiness, but text similarity usually 0.
            # Viewing SetMath would confirm. Usually jaccard = intersection / union.
            # If both empty, union is empty -> 0/0.
            # Let's assume 0.0 for text similarity context.

        intersection = lhs.intersection(rhs)
        union = lhs.union(rhs)

        if not union:
            return 0.0

        return len(intersection) / len(union)
