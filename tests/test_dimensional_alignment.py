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

"""Tests for dimensional_alignment.py - Per-dimension alignment measurement.

Tests cover:
- DimensionalAlignment dataclass and summary() method
- measure_1d_alignment() for tokenization
- measure_dimensional_alignment() full report
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from modelcypher.core.domain.geometry.dimensional_alignment import (
    DimensionalAlignment,
    measure_1d_alignment,
    measure_dimensional_alignment,
)


# =============================================================================
# DimensionalAlignment Tests
# =============================================================================


class TestDimensionalAlignment:
    """Tests for DimensionalAlignment dataclass."""

    def test_fields_stored(self):
        """DimensionalAlignment stores all fields."""
        alignment = DimensionalAlignment(
            vocab_overlap=0.8,
            vocab_jaccard=0.7,
            sequence_agreement=0.9,
            shared_token_count=5000,
            source_vocab_size=10000,
            target_vocab_size=8000,
            embedding_cka=0.95,
            layernorm_cka=0.98,
            hidden_cka_mean=0.92,
            hidden_cka_min=0.88,
            intermediate_cka_mean=0.90,
        )
        assert alignment.vocab_overlap == 0.8
        assert alignment.embedding_cka == 0.95

    def test_summary_returns_string(self):
        """summary() returns formatted string."""
        alignment = DimensionalAlignment(
            vocab_overlap=0.8,
            vocab_jaccard=0.7,
            sequence_agreement=0.9,
            shared_token_count=5000,
            source_vocab_size=10000,
            target_vocab_size=8000,
            embedding_cka=None,
            layernorm_cka=None,
            hidden_cka_mean=None,
            hidden_cka_min=None,
            intermediate_cka_mean=None,
        )
        summary = alignment.summary()
        
        assert isinstance(summary, str)
        assert "overlap" in summary.lower() or "vocab" in summary.lower()


# =============================================================================
# measure_1d_alignment Tests
# =============================================================================


class TestMeasure1DAlignment:
    """Tests for measure_1d_alignment function."""

    def test_measure_1d_with_mock_tokenizers(self):
        """measure_1d_alignment works with mock tokenizers."""
        # Create mock tokenizers with vocab
        source_tok = Mock()
        source_tok.get_vocab.return_value = {"hello": 0, "world": 1, "test": 2}
        
        target_tok = Mock()
        target_tok.get_vocab.return_value = {"hello": 0, "there": 1, "world": 2}
        
        result = measure_1d_alignment(source_tok, target_tok)
        
        assert "vocab_overlap" in result
        assert "vocab_jaccard" in result
        assert 0.0 <= result["vocab_overlap"] <= 1.0

    def test_measure_1d_identical_vocabs(self):
        """measure_1d_alignment with identical vocabs returns 1.0."""
        tok = Mock()
        tok.get_vocab.return_value = {"a": 0, "b": 1, "c": 2}
        
        result = measure_1d_alignment(tok, tok)
        
        assert result["vocab_overlap"] == 1.0
        assert result["vocab_jaccard"] == 1.0


# =============================================================================
# measure_dimensional_alignment Tests
# =============================================================================


class TestMeasureDimensionalAlignment:
    """Tests for measure_dimensional_alignment function."""

    def test_measure_dimensional_alignment_returns_result(self):
        """measure_dimensional_alignment returns DimensionalAlignment."""
        source_tok = Mock()
        source_tok.get_vocab.return_value = {"a": 0, "b": 1}
        
        target_tok = Mock()
        target_tok.get_vocab.return_value = {"a": 0, "c": 1}
        
        probe_metrics = {
            "embedding_cka": 0.9,
            "hidden_cka_mean": 0.85,
            "hidden_cka_min": 0.80,
        }
        
        result = measure_dimensional_alignment(
            source_tok, target_tok, probe_metrics
        )
        
        assert isinstance(result, DimensionalAlignment)
