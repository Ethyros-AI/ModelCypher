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

"""Hierarchical document chunker.

Splits text at paragraph, sentence, and word boundaries to fit memory limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class TextTokenizer(Protocol):
    """Protocol for text tokenizers."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...


@dataclass(frozen=True)
class TextChunk:
    """A single text chunk with metadata."""

    text: str
    """The chunk's text content."""

    tokens: list[int]
    """Token IDs for this chunk."""

    start_offset: int
    """Character offset where chunk starts in original document."""

    end_offset: int
    """Character offset where chunk ends in original document."""

    @property
    def token_count(self) -> int:
        """Number of tokens in this chunk."""
        return len(self.tokens)


class DocumentChunker:
    """Hierarchical document chunker.

    Splits text at paragraph → sentence → word boundaries to fit memory limits.
    """

    # Paragraph separator pattern (2+ newlines)
    PARAGRAPH_PATTERN = re.compile(r"(\r\n|\r|\n){2,}")

    # Sentence ending pattern
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        tokenizer: TextTokenizer | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """Initialize chunker.

        Args:
            tokenizer: Optional tokenizer for accurate token counting.
            token_estimator: Optional function to estimate tokens (default: len/4).
        """
        self._tokenizer = tokenizer
        self._token_estimator = token_estimator or (lambda s: len(s) // 4)

    def _encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        if self._tokenizer:
            return self._tokenizer.encode(text)
        # Return dummy tokens based on estimation
        count = self._token_estimator(text)
        return list(range(count))

    def _decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        if self._tokenizer:
            return self._tokenizer.decode(tokens)
        # Can't decode without tokenizer
        return ""

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return self._token_estimator(text)

    def chunk(
        self,
        text: str,
        target_tokens: int,
        preserve_boundaries: bool = True,
    ) -> list[TextChunk]:
        """Chunk text into memory-appropriate pieces.

        Args:
            text: The document text to chunk.
            target_tokens: Target token count per chunk.
            preserve_boundaries: If True, respect paragraph/sentence boundaries.

        Returns:
            Array of text chunks, each with metadata.
        """
        if not text or target_tokens <= 0:
            return []

        if preserve_boundaries:
            return self._chunk_with_boundaries(text, target_tokens)
        else:
            return self._chunk_fixed_size(text, target_tokens)

    def _chunk_with_boundaries(self, text: str, target_tokens: int) -> list[TextChunk]:
        """Chunk respecting paragraph/sentence boundaries."""
        paragraphs = self._split_into_paragraphs(text)
        chunks: list[TextChunk] = []
        current_offset = 0

        for i, (para_text, separator) in enumerate(paragraphs):
            para_tokens = self._encode(para_text)
            has_more = i < len(paragraphs) - 1

            # If paragraph exceeds target, split at sentence/word level
            if len(para_tokens) > target_tokens:
                sentence_chunks = self._chunk_large_paragraph(
                    para_text, target_tokens, current_offset
                )
                chunks.extend(sentence_chunks)
            else:
                chunks.append(
                    TextChunk(
                        text=para_text,
                        tokens=para_tokens,
                        start_offset=current_offset,
                        end_offset=current_offset + len(para_text),
                    )
                )

            current_offset += len(para_text) + (len(separator) if has_more else 0)

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[tuple[str, str]]:
        """Split text into paragraphs with separators.

        Returns:
            List of (paragraph_text, trailing_separator) tuples.
        """
        slices: list[tuple[str, str]] = []
        last_end = 0

        for match in self.PARAGRAPH_PATTERN.finditer(text):
            para_text = text[last_end : match.start()].strip()
            if para_text:
                slices.append((para_text, match.group()))
            last_end = match.end()

        # Handle trailing text
        trailing = text[last_end:].strip()
        if trailing:
            slices.append((trailing, ""))

        # If no paragraphs found, return whole text
        if not slices:
            trimmed = text.strip()
            if trimmed:
                slices.append((trimmed, ""))

        return slices

    def _chunk_large_paragraph(
        self, paragraph: str, target_tokens: int, start_offset: int
    ) -> list[TextChunk]:
        """Chunk a paragraph that exceeds target at sentence level."""
        sentences = self._split_into_sentences(paragraph)

        chunks: list[TextChunk] = []
        buffer_texts: list[str] = []
        buffer_tokens: list[int] = []
        buffer_token_count = 0
        current_offset = start_offset
        chunk_start_offset = start_offset

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._encode(sentence)
            has_more = i < len(sentences) - 1

            # If single sentence exceeds target
            if len(sentence_tokens) > target_tokens:
                # Flush buffer first
                if buffer_texts:
                    chunk_text = " ".join(buffer_texts)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            tokens=buffer_tokens.copy(),
                            start_offset=chunk_start_offset,
                            end_offset=chunk_start_offset + len(chunk_text),
                        )
                    )
                    buffer_texts = []
                    buffer_tokens = []
                    buffer_token_count = 0

                # Split sentence at word level
                word_chunks = self._chunk_large_sentence(sentence, target_tokens, current_offset)
                chunks.extend(word_chunks)
                current_offset += len(sentence) + (1 if has_more else 0)
                chunk_start_offset = current_offset
                continue

            # Check if adding sentence would exceed target
            separator_cost = 1 if buffer_texts else 0  # Space between sentences
            if buffer_token_count + separator_cost + len(sentence_tokens) > target_tokens:
                if buffer_texts:
                    # Flush buffer
                    chunk_text = " ".join(buffer_texts)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            tokens=buffer_tokens.copy(),
                            start_offset=chunk_start_offset,
                            end_offset=chunk_start_offset + len(chunk_text),
                        )
                    )
                    buffer_texts = []
                    buffer_tokens = []
                    buffer_token_count = 0
                    chunk_start_offset = current_offset

            # Add sentence to buffer
            buffer_texts.append(sentence)
            buffer_tokens.extend(sentence_tokens)
            buffer_token_count += len(sentence_tokens)
            current_offset += len(sentence) + (1 if has_more else 0)

        # Flush remaining buffer
        if buffer_texts:
            chunk_text = " ".join(buffer_texts)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    tokens=buffer_tokens,
                    start_offset=chunk_start_offset,
                    end_offset=chunk_start_offset + len(chunk_text),
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if not text:
            return []

        # Simple sentence splitting
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_large_sentence(
        self, sentence: str, target_tokens: int, start_offset: int
    ) -> list[TextChunk]:
        """Chunk a sentence that exceeds target at word level."""
        words = sentence.split()

        chunks: list[TextChunk] = []
        buffer_words: list[str] = []
        buffer_tokens: list[int] = []
        buffer_token_count = 0
        current_offset = start_offset

        for i, word in enumerate(words):
            word_tokens = self._encode(word)
            has_more = i < len(words) - 1

            # If single word exceeds target (very rare)
            if len(word_tokens) > target_tokens:
                # Flush buffer first
                if buffer_words:
                    chunk_text = " ".join(buffer_words)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            tokens=buffer_tokens.copy(),
                            start_offset=current_offset - len(chunk_text),
                            end_offset=current_offset,
                        )
                    )
                    buffer_words = []
                    buffer_tokens = []
                    buffer_token_count = 0

                # Add word as its own chunk
                chunks.append(
                    TextChunk(
                        text=word,
                        tokens=word_tokens,
                        start_offset=current_offset,
                        end_offset=current_offset + len(word),
                    )
                )
                current_offset += len(word) + (1 if has_more else 0)
                continue

            # Check if adding word would exceed target
            separator_cost = 1 if buffer_words else 0
            if buffer_token_count + separator_cost + len(word_tokens) > target_tokens:
                if buffer_words:
                    chunk_text = " ".join(buffer_words)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            tokens=buffer_tokens.copy(),
                            start_offset=current_offset - len(chunk_text),
                            end_offset=current_offset,
                        )
                    )
                    buffer_words = []
                    buffer_tokens = []
                    buffer_token_count = 0

            # Add word to buffer
            buffer_words.append(word)
            buffer_tokens.extend(word_tokens)
            buffer_token_count += len(word_tokens)
            current_offset += len(word) + (1 if has_more else 0)

        # Flush remaining buffer
        if buffer_words:
            chunk_text = " ".join(buffer_words)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    tokens=buffer_tokens,
                    start_offset=current_offset - len(chunk_text),
                    end_offset=current_offset,
                )
            )

        return chunks

    def _chunk_fixed_size(self, text: str, target_tokens: int) -> list[TextChunk]:
        """Chunk at fixed token boundaries (ignores natural boundaries)."""
        all_tokens = self._encode(text)

        chunks: list[TextChunk] = []
        offset = 0

        for chunk_start in range(0, len(all_tokens), target_tokens):
            chunk_end = min(chunk_start + target_tokens, len(all_tokens))
            chunk_tokens = all_tokens[chunk_start:chunk_end]

            # Try to decode back to text
            if self._tokenizer:
                chunk_text = self._tokenizer.decode(chunk_tokens)
            else:
                # Approximate: take proportional substring
                char_start = int(chunk_start / len(all_tokens) * len(text))
                char_end = int(chunk_end / len(all_tokens) * len(text))
                chunk_text = text[char_start:char_end]

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    tokens=chunk_tokens,
                    start_offset=offset,
                    end_offset=offset + len(chunk_text),
                )
            )

            offset += len(chunk_text)

        return chunks
