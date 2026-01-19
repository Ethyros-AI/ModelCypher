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

"""Hypothesis property tests for deterministic embedding stub."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.adapters.embedding_stub import ByteFrequencyEmbeddingProvider


@settings(max_examples=20, deadline=None)
@given(texts=st.lists(st.text(min_size=0, max_size=64), min_size=1, max_size=4))
def test_byte_frequency_embeddings_properties(texts: list[str]) -> None:
    provider = ByteFrequencyEmbeddingProvider()
    embeddings = provider.embed(texts)

    assert len(embeddings) == len(texts)
    for text, embedding in zip(texts, embeddings):
        assert len(embedding) == provider.dimension
        total = sum(embedding)
        assert all(value >= 0.0 for value in embedding)
        if text:
            tol = math.ulp(1.0) * len(embedding)
            assert abs(total - 1.0) <= tol
        else:
            assert total <= 1.0
