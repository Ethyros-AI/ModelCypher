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

"""Tests for ByteFrequencyEmbeddingProvider (embedding stub)."""

from __future__ import annotations

import math
import pytest

from modelcypher.adapters.embedding_stub import ByteFrequencyEmbeddingProvider
from modelcypher.ports.embedding import EmbeddingProvider


class TestByteFrequencyEmbeddingProviderProtocol:
    """Verify ByteFrequencyEmbeddingProvider implements EmbeddingProvider."""

    def test_implements_embedding_provider(self):
        """ByteFrequencyEmbeddingProvider should implement EmbeddingProvider."""
        provider = ByteFrequencyEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_has_dimension_property(self):
        """Provider should have dimension property."""
        provider = ByteFrequencyEmbeddingProvider()
        assert hasattr(provider, "dimension")
        assert isinstance(provider.dimension, int)

    def test_has_embed_method(self):
        """Provider should have embed method."""
        provider = ByteFrequencyEmbeddingProvider()
        assert hasattr(provider, "embed")
        assert callable(provider.embed)


class TestByteFrequencyEmbeddingProviderDimension:
    """Test dimension property."""

    def test_dimension_is_256(self):
        """Dimension should be 256 (one per byte value)."""
        provider = ByteFrequencyEmbeddingProvider()
        assert provider.dimension == 256


class TestByteFrequencyEmbeddingProviderEmbed:
    """Test embed() method."""

    def test_embed_single_text(self):
        """embed() should handle single text."""
        provider = ByteFrequencyEmbeddingProvider()
        result = provider.embed(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 256

    def test_embed_multiple_texts(self):
        """embed() should handle multiple texts."""
        provider = ByteFrequencyEmbeddingProvider()
        texts = ["hello", "world", "test"]
        result = provider.embed(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 256

    def test_embed_empty_list(self):
        """embed() should handle empty list."""
        provider = ByteFrequencyEmbeddingProvider()
        result = provider.embed([])

        assert result == []

    def test_embed_empty_string(self):
        """embed() should handle empty string."""
        provider = ByteFrequencyEmbeddingProvider()
        result = provider.embed([""])

        assert len(result) == 1
        assert len(result[0]) == 256
        # All values should be 0 for empty string
        assert all(v == 0.0 for v in result[0])

    def test_embed_returns_normalized_frequencies(self):
        """embed() should return normalized byte frequencies."""
        provider = ByteFrequencyEmbeddingProvider()
        # "aaa" = 3 'a' bytes (ASCII 97)
        result = provider.embed(["aaa"])

        # Index 97 ('a') should be 1.0, all others 0.0
        assert result[0][97] == 1.0
        assert sum(result[0]) == pytest.approx(1.0)

    def test_embed_frequency_distribution(self):
        """embed() should correctly distribute byte frequencies."""
        provider = ByteFrequencyEmbeddingProvider()
        # "ab" = 1 'a' (97) + 1 'b' (98)
        result = provider.embed(["ab"])

        assert result[0][97] == pytest.approx(0.5)
        assert result[0][98] == pytest.approx(0.5)
        assert sum(result[0]) == pytest.approx(1.0)

    def test_embed_handles_unicode(self):
        """embed() should handle unicode by encoding to UTF-8."""
        provider = ByteFrequencyEmbeddingProvider()
        # Unicode emoji encodes to multiple bytes
        result = provider.embed(["🎉"])

        assert len(result[0]) == 256
        # Sum should be 1.0 (normalized)
        assert sum(result[0]) == pytest.approx(1.0)

    def test_embed_handles_invalid_unicode(self):
        """embed() should handle invalid unicode gracefully."""
        provider = ByteFrequencyEmbeddingProvider()
        # Mix of valid and potentially problematic characters
        result = provider.embed(["test\x00\xff"])

        assert len(result[0]) == 256

    def test_embed_deterministic(self):
        """embed() should be deterministic for same input."""
        provider = ByteFrequencyEmbeddingProvider()
        text = "The quick brown fox"

        result1 = provider.embed([text])
        result2 = provider.embed([text])

        assert result1 == result2

    def test_embed_different_texts_different_results(self):
        """embed() should produce different results for different texts."""
        provider = ByteFrequencyEmbeddingProvider()

        result1 = provider.embed(["hello"])
        result2 = provider.embed(["world"])

        assert result1 != result2

    def test_embed_preserves_character_structure(self):
        """embed() should preserve character-level structure."""
        provider = ByteFrequencyEmbeddingProvider()

        # "aa" and "bb" should have peaks at different indices
        result_aa = provider.embed(["aa"])
        result_bb = provider.embed(["bb"])

        # 'a' = 97, 'b' = 98
        assert result_aa[0][97] == 1.0
        assert result_aa[0][98] == 0.0
        assert result_bb[0][97] == 0.0
        assert result_bb[0][98] == 1.0

    def test_embed_all_bytes_representable(self):
        """embed() should be able to represent all 256 byte values."""
        provider = ByteFrequencyEmbeddingProvider()

        # Create text with all byte values
        all_bytes = bytes(range(256))
        # Some bytes may not be valid UTF-8, so we use latin-1
        text = all_bytes.decode("latin-1")
        result = provider.embed([text])

        # Each byte appears once, so all frequencies should be equal
        expected_freq = 1.0 / 256
        for i in range(256):
            assert result[0][i] == pytest.approx(expected_freq, abs=math.ulp(expected_freq))
