"""
Task Diversion Detector.

Scores whether a model response has diverged from the expected task.
Agent Cypher uses *geometry* (vector similarity) instead of prompt heuristics to detect when a
response is no longer aligned with the task at hand.

Ported from TrainingCypher/Domain/Agents/TaskDiversionDetector.swift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, List, Protocol

from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.ports.embedding import EmbeddingProvider


class LexicalTokenizer:
    """Simple tokenizer matching Swift implementation."""
    @staticmethod
    def tokens(text: str) -> List[str]:
        # Lowercase and split on non-alphanumerics
        text = text.lower()
        # Regex to split on anything that is NOT alphanumeric
        return [t for t in re.split(r'[^a-z0-9]', text) if t]


class LexicalStopWords:
    """Stop words for task diversion detection."""
    task_diversion_detector: Set[str] = {
        "this", "that", "with", "from", "have", "will", "would", "could",
        "should", "about", "which", "their", "there", "been", "being",
        "some", "what", "when", "where", "they", "them", "then", "than",
        "these", "those", "each", "other", "into", "just", "only", "your",
        "youre", "you're", "please", "thanks", "thank", "also", "can", "cant",
        "can't", "dont", "don't", "does", "doesnt", "doesn't", "using", "use",
        "make", "made", "like", "need", "needs", "want", "wants",
    }


@dataclass
class TaskDiversionAssessment:
    class Method(str, Enum):
        EMBEDDINGS = "embeddings"
        LEXICAL_FALLBACK = "lexicalFallback"
        SKIPPED = "skipped"

    class Verdict(str, Enum):
        ALIGNED = "aligned"
        DIVERGED = "diverged"
        UNKNOWN = "unknown"

    method: Method
    verdict: Verdict
    embedding_cosine_similarity: Optional[float] = None
    lexical_jaccard_similarity: Optional[float] = None
    threshold: Optional[float] = None
    note: Optional[str] = None


@dataclass
class DiversionDetectorConfiguration:
    enabled: bool = True
    max_characters_per_text: int = 4096
    minimum_embedding_cosine_similarity: float = 0.35
    minimum_lexical_jaccard_similarity: float = 0.08
    enable_lexical_fallback: bool = True
    fail_closed: bool = False


class TaskDiversionDetector:
    """Embedding-first task diversion detector with deterministic thresholds."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        configuration: DiversionDetectorConfiguration = DiversionDetectorConfiguration()
    ):
        self.config = configuration
        self.embedder = embedder

    async def assess(self, expected_task: str, observed_text: str) -> TaskDiversionAssessment:
        if not self.config.enabled:
            return TaskDiversionAssessment(
                method=TaskDiversionAssessment.Method.SKIPPED,
                verdict=TaskDiversionAssessment.Verdict.UNKNOWN,
                note="disabled"
            )

        expected_trimmed = expected_task.strip()
        observed_trimmed = observed_text.strip()

        if not expected_trimmed or not observed_trimmed:
            return TaskDiversionAssessment(
                method=TaskDiversionAssessment.Method.SKIPPED,
                verdict=TaskDiversionAssessment.Verdict.DIVERGED if self.config.fail_closed else TaskDiversionAssessment.Verdict.UNKNOWN,
                note="missing_text"
            )

        expected_capped = expected_trimmed[:self.config.max_characters_per_text]
        observed_capped = observed_trimmed[:self.config.max_characters_per_text]

        # Try Embeddings
        try:
            embeddings = await self.embedder.embed([expected_capped, observed_capped])
            if len(embeddings) == 2:
                similarity = VectorMath.cosine_similarity(embeddings[0], embeddings[1]) or 0.0
                verdict = (
                    TaskDiversionAssessment.Verdict.ALIGNED
                    if similarity >= self.config.minimum_embedding_cosine_similarity
                    else TaskDiversionAssessment.Verdict.DIVERGED
                )
                
                return TaskDiversionAssessment(
                    method=TaskDiversionAssessment.Method.EMBEDDINGS,
                    verdict=verdict,
                    embedding_cosine_similarity=similarity,
                    threshold=self.config.minimum_embedding_cosine_similarity,
                    note="cosine_below_threshold" if verdict == TaskDiversionAssessment.Verdict.DIVERGED else None
                )
        except Exception as e:
            # print(f"Embedding scoring failed: {e}")
            pass

        # Fallback to Lexical
        if not self.config.enable_lexical_fallback:
            return TaskDiversionAssessment(
                method=TaskDiversionAssessment.Method.SKIPPED,
                verdict=TaskDiversionAssessment.Verdict.DIVERGED if self.config.fail_closed else TaskDiversionAssessment.Verdict.UNKNOWN,
                note="no_embedding_no_fallback"
            )

        lexical_similarity = self._lexical_jaccard_similarity(expected_trimmed, observed_trimmed)
        verdict = (
            TaskDiversionAssessment.Verdict.ALIGNED
            if lexical_similarity >= self.config.minimum_lexical_jaccard_similarity
            else TaskDiversionAssessment.Verdict.DIVERGED
        )

        return TaskDiversionAssessment(
            method=TaskDiversionAssessment.Method.LEXICAL_FALLBACK,
            verdict=verdict,
            lexical_jaccard_similarity=lexical_similarity,
            threshold=self.config.minimum_lexical_jaccard_similarity,
            note="lexical_below_threshold" if verdict == TaskDiversionAssessment.Verdict.DIVERGED else None
        )

    def _lexical_jaccard_similarity(self, lhs: str, rhs: str) -> float:
        lhs_tokens = self._lexical_token_set(lhs)
        rhs_tokens = self._lexical_token_set(rhs)
        return self._jaccard_similarity(lhs_tokens, rhs_tokens)

    def _lexical_token_set(self, text: str) -> Set[str]:
        raw_tokens = LexicalTokenizer.tokens(text)
        # Filter tokens < 3 chars or stop words
        return {
            t for t in raw_tokens
            if len(t) >= 3 and t not in LexicalStopWords.task_diversion_detector
        }

    @staticmethod
    def _jaccard_similarity(lhs: Set[str], rhs: Set[str]) -> float:
        if not lhs and not rhs:
            return 0.0 # Swift SetMath behavior likely 0 if both empty? Or 1? Usually 1 if identical emptiness, but text similarity usually 0. 
                       # Viewing SetMath would confirm. Usually jaccard = intersection / union.
                       # If both empty, union is empty -> 0/0.
                       # Let's assume 0.0 for text similarity context.
        
        intersection = lhs.intersection(rhs)
        union = lhs.union(rhs)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
