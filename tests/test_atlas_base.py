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

"""Tests for atlas base classes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.atlas_base import (
    AtlasConcept,
    BaseAtlas,
    BaseAtlasConfiguration,
    BaseAtlasSignature,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.embedding import EmbeddingProvider


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


# Test implementation of AtlasConcept
@dataclass(frozen=True)
class MockConcept:
    """Mock concept for testing."""

    id: str
    name: str
    description: str

    @property
    def canonical_name(self) -> str:
        return self.name


# Test implementation of signature
@dataclass(frozen=True)
class MockSignature(BaseAtlasSignature):
    """Mock signature class."""

    pass


# Mock embedding provider
class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        self._embeddings = embeddings or {}
        self._default_dim = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            if text in self._embeddings:
                result.append(self._embeddings[text])
            else:
                # Generate deterministic embedding based on text hash
                h = hash(text) % 1000 / 1000.0
                result.append([h, 1 - h, h * 0.5, (1 - h) * 0.5])
        return result


# Mock implementation of BaseAtlas
class MockAtlas(BaseAtlas[MockConcept, MockSignature]):
    """Mock atlas implementation for testing."""

    def __init__(
        self,
        embedder: "EmbeddingProvider | None" = None,
        configuration: BaseAtlasConfiguration | None = None,
        concepts: list[MockConcept] | None = None,
    ):
        super().__init__(embedder, configuration)
        self._concepts = concepts or [
            MockConcept(id="c1", name="Concept One", description="First concept"),
            MockConcept(id="c2", name="Concept Two", description="Second concept"),
            MockConcept(id="c3", name="Concept Three", description="Third concept"),
        ]

    @property
    def inventory(self) -> list[MockConcept]:
        return self._concepts

    def _get_concept_text(self, concept: MockConcept) -> str:
        return f"{concept.name}: {concept.description}"

    def _create_signature(
        self, concept_ids: list[str], values: list[float]
    ) -> MockSignature:
        return MockSignature(concept_ids=concept_ids, values=values)


class TestBaseAtlasConfiguration:
    """Tests for BaseAtlasConfiguration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BaseAtlasConfiguration()

        assert config.enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BaseAtlasConfiguration(enabled=False)

        assert config.enabled is False


class TestBaseAtlasSignature:
    """Tests for BaseAtlasSignature."""

    def test_to_dict(self):
        """Test signature to dictionary conversion."""
        sig = BaseAtlasSignature(
            concept_ids=["a", "b", "c"],
            values=[0.8, 0.5, 0.2],
        )

        d = sig.to_dict()

        assert d == {"a": 0.8, "b": 0.5, "c": 0.2}

    def test_top_k(self):
        """Test top-k retrieval."""
        sig = BaseAtlasSignature(
            concept_ids=["a", "b", "c", "d"],
            values=[0.3, 0.9, 0.1, 0.7],
        )

        top2 = sig.top_k(2)

        assert top2 == [("b", 0.9), ("d", 0.7)]

    def test_top_k_all(self):
        """Test top-k with None returns all sorted."""
        sig = BaseAtlasSignature(
            concept_ids=["a", "b", "c"],
            values=[0.3, 0.9, 0.1],
        )

        all_sorted = sig.top_k(None)

        assert all_sorted == [("b", 0.9), ("a", 0.3), ("c", 0.1)]

    def test_l2_normalized(self):
        """Test L2 normalization."""
        eps = _eps()
        sig = BaseAtlasSignature(
            concept_ids=["a", "b"],
            values=[3.0, 4.0],
        )

        normalized = sig.l2_normalized()

        assert normalized.concept_ids == ["a", "b"]
        assert abs(normalized.values[0] - 0.6) <= eps
        assert abs(normalized.values[1] - 0.8) <= eps

    def test_cosine_similarity(self):
        """Test cosine similarity between signatures."""
        sig1 = BaseAtlasSignature(
            concept_ids=["a", "b"],
            values=[1.0, 0.0],
        )
        sig2 = BaseAtlasSignature(
            concept_ids=["a", "b"],
            values=[1.0, 0.0],
        )

        similarity = sig1.cosine_similarity(sig2)

        assert abs(similarity - 1.0) <= _eps()

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        sig1 = BaseAtlasSignature(
            concept_ids=["a", "b"],
            values=[1.0, 0.0],
        )
        sig2 = BaseAtlasSignature(
            concept_ids=["a", "b"],
            values=[0.0, 1.0],
        )

        similarity = sig1.cosine_similarity(sig2)

        assert abs(similarity) <= _eps()


class TestAtlasConcept:
    """Tests for AtlasConcept protocol."""

    def test_concept_implements_protocol(self):
        """Test that MockConcept implements AtlasConcept protocol."""
        concept = MockConcept(id="test", name="Test", description="A test concept")

        assert isinstance(concept, AtlasConcept)
        assert concept.id == "test"
        assert concept.canonical_name == "Test"


class TestBaseAtlas:
    """Tests for BaseAtlas."""

    @pytest.fixture
    def atlas(self):
        """Create test atlas with mock embedder."""
        return MockAtlas(embedder=MockEmbedder())

    @pytest.fixture
    def disabled_atlas(self):
        """Create disabled test atlas."""
        config = BaseAtlasConfiguration(enabled=False)
        return MockAtlas(embedder=MockEmbedder(), configuration=config)

    @pytest.fixture
    def no_embedder_atlas(self):
        """Create test atlas without embedder."""
        return MockAtlas(embedder=None)

    @pytest.mark.asyncio
    async def test_signature_returns_values(self, atlas):
        """Test signature computation returns values."""
        sig = await atlas.signature("Hello world")

        assert sig is not None
        assert len(sig.concept_ids) == 3
        assert len(sig.values) == 3
        assert sig.concept_ids == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_signature_disabled_returns_none(self, disabled_atlas):
        """Test disabled atlas returns None."""
        sig = await disabled_atlas.signature("Hello world")

        assert sig is None

    @pytest.mark.asyncio
    async def test_signature_empty_text_returns_none(self, atlas):
        """Test empty text returns None."""
        sig = await atlas.signature("")

        assert sig is None

    @pytest.mark.asyncio
    async def test_signature_whitespace_text_returns_none(self, atlas):
        """Test whitespace-only text returns None."""
        sig = await atlas.signature("   \n\t   ")

        assert sig is None

    @pytest.mark.asyncio
    async def test_signature_no_embedder_returns_none(self, no_embedder_atlas):
        """Test atlas without embedder returns None."""
        sig = await no_embedder_atlas.signature("Hello world")

        assert sig is None

    @pytest.mark.asyncio
    async def test_signature_text_trimmed(self):
        """Test that text is trimmed before embedding."""
        config = BaseAtlasConfiguration()
        class RecordingEmbedder:
            def __init__(self):
                self.texts: list[str] = []

            async def embed(self, texts: list[str]) -> list[list[float]]:
                self.texts.extend(texts)
                return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        embedder = RecordingEmbedder()
        atlas = MockAtlas(embedder=embedder, configuration=config)
        text = "  This is a very long text that should be preserved.  "

        await atlas.signature(text)

        # First 3 are concept texts, then the input.
        input_text = embedder.texts[-1]
        assert input_text == text.strip()

    @pytest.mark.asyncio
    async def test_embeddings_cached(self, atlas):
        """Test that concept embeddings are cached."""
        # First call creates cache
        await atlas.signature("First text")
        cached = atlas._cached_concept_embeddings

        # Second call uses cache
        await atlas.signature("Second text")

        assert atlas._cached_concept_embeddings is cached

    @pytest.mark.asyncio
    async def test_clear_cache(self, atlas):
        """Test cache clearing."""
        await atlas.signature("Hello")
        assert atlas._cached_concept_embeddings is not None

        atlas.clear_cache()

        assert atlas._cached_concept_embeddings is None

    @pytest.mark.asyncio
    async def test_signature_values_nonnegative(self, atlas):
        """Test that signature values are clamped to non-negative."""
        sig = await atlas.signature("Test text")

        assert sig is not None
        for v in sig.values:
            assert v >= -_eps()


class TestNormalizedEntropy:
    """Tests for normalized entropy calculation."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution is 1.0."""
        values = [1.0, 1.0, 1.0, 1.0]
        entropy = BaseAtlas.normalized_entropy(values)

        assert abs(entropy - 1.0) <= _eps()

    def test_single_peak(self):
        """Test entropy of single-peak distribution is 0.0."""
        values = [1.0, 0.0, 0.0, 0.0]
        entropy = BaseAtlas.normalized_entropy(values)

        assert abs(entropy) <= _eps()

    def test_moderate_entropy(self):
        """Test entropy of moderate distribution."""
        values = [0.5, 0.5, 0.0, 0.0]
        entropy = BaseAtlas.normalized_entropy(values)

        # Entropy of [0.5, 0.5] = -0.5*log(0.5) - 0.5*log(0.5) = log(2)
        # Normalized by log(2) = 1.0
        assert abs(entropy - 1.0) <= _eps()

    def test_zero_values_returns_none(self):
        """Test all-zero values returns None."""
        values = [0.0, 0.0, 0.0, 0.0]
        entropy = BaseAtlas.normalized_entropy(values)

        assert entropy is None

    def test_negative_values_clamped(self):
        """Test negative values are clamped to 0."""
        values = [-1.0, 1.0, 0.0, 0.0]
        entropy = BaseAtlas.normalized_entropy(values)

        # After clamping: [0.0, 1.0, 0.0, 0.0] -> single peak -> 0
        assert abs(entropy) <= _eps()

    def test_single_nonzero_value(self):
        """Test single non-zero value has 0 entropy."""
        values = [0.0, 0.0, 5.0]
        entropy = BaseAtlas.normalized_entropy(values)

        assert abs(entropy) <= _eps()
