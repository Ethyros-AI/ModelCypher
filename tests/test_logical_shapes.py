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

"""Unit tests for logical shapes training module."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.logical_shapes import (
    LogicalShape,
    LogicalShapeExample,
    get_shape_description,
)
from modelcypher.core.domain.training.logical_shapes_patterns import (
    get_logical_shape_examples,
)


class TestLogicalShape:
    """Tests for LogicalShape enum."""

    def test_all_shapes_defined(self):
        """All 6 fundamental shapes should be defined."""
        shapes = list(LogicalShape)
        assert len(shapes) == 6

        # Verify specific shapes exist
        assert LogicalShape.PERCENTAGE_INCREASE in shapes
        assert LogicalShape.AVERAGE_RATE in shapes
        assert LogicalShape.THRESHOLD_CROSSING in shapes
        assert LogicalShape.INVERSE_CHAIN in shapes
        assert LogicalShape.SEQUENTIAL_OPERATIONS in shapes
        assert LogicalShape.REMAINING_FIRST in shapes

    def test_shape_values_are_strings(self):
        """Shape values should be lowercase snake_case strings."""
        for shape in LogicalShape:
            assert isinstance(shape.value, str)
            assert shape.value == shape.value.lower()
            assert "_" in shape.value or shape.value.isalpha()


class TestLogicalShapeExample:
    """Tests for LogicalShapeExample dataclass."""

    def test_frozen_dataclass(self):
        """LogicalShapeExample should be immutable."""
        example = LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A house worth $100 increases by 50%. New value?",
            reasoning="Add 50% to original: $100 + $50 = $150",
            answer="150",
            common_error="Forgetting to add to original",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            example.answer = "200"

    def test_full_text_format(self):
        """full_text should format as training example."""
        example = LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="Travel 60 miles at 30 mph, 60 miles at 60 mph. Average speed?",
            reasoning="Total distance / Total time = 120 / 3 = 40 mph",
            answer="40",
            common_error="Arithmetic mean of speeds",
        )

        text = example.full_text

        assert "Question:" in text
        assert "Travel 60 miles" in text
        assert "AVERAGE RATE" in text
        assert "Total distance / Total time" in text
        assert "#### 40" in text

    def test_training_dict_format(self):
        """training_dict should return dict with 'text' key."""
        example = LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="Tree costs $100, yields $10/year. When profit?",
            reasoning="Breakeven at year 10, first profit year 11",
            answer="11",
            common_error="Reporting breakeven year",
        )

        d = example.training_dict

        assert "text" in d
        assert isinstance(d["text"], str)
        assert "#### 11" in d["text"]

    def test_as_dict_serialization(self):
        """as_dict should include all fields for JSON serialization."""
        example = LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After giving half, Sarah has 5. Started with?",
            reasoning="Work backwards: 5 × 2 = 10",
            answer="10",
            common_error="Working forwards with unknowns",
        )

        d = example.as_dict()

        assert d["shape"] == "inverse_chain"
        assert d["question"] == "After giving half, Sarah has 5. Started with?"
        assert d["reasoning"] == "Work backwards: 5 × 2 = 10"
        assert d["answer"] == "10"
        assert d["common_error"] == "Working forwards with unknowns"


class TestGetShapeDescription:
    """Tests for get_shape_description function."""

    def test_all_shapes_have_descriptions(self):
        """Every LogicalShape should have a description."""
        for shape in LogicalShape:
            desc = get_shape_description(shape)
            assert desc is not None
            assert len(desc) > 50  # Descriptions should be substantial

    def test_descriptions_mention_common_errors(self):
        """Descriptions should mention common errors."""
        for shape in LogicalShape:
            desc = get_shape_description(shape)
            assert "error" in desc.lower() or "wrong" in desc.lower()


class TestLogicalShapePatterns:
    """Tests for logical shape pattern examples."""

    def test_total_example_count(self):
        """Should have ~90 examples (15 per shape)."""
        examples = get_logical_shape_examples()
        # Allow some variance
        assert 80 <= len(examples) <= 100

    def test_all_shapes_covered(self):
        """Each shape should have multiple examples."""
        examples = get_logical_shape_examples()

        shape_counts = {}
        for ex in examples:
            shape_counts[ex.shape] = shape_counts.get(ex.shape, 0) + 1

        # Each shape should have at least 10 examples
        for shape in LogicalShape:
            assert shape in shape_counts, f"Missing examples for {shape}"
            assert shape_counts[shape] >= 10, f"Too few examples for {shape}"

    def test_examples_have_required_fields(self):
        """All examples should have non-empty required fields."""
        examples = get_logical_shape_examples()

        for ex in examples:
            assert ex.shape is not None
            assert len(ex.question) > 10, f"Question too short: {ex.question}"
            assert len(ex.reasoning) > 20, f"Reasoning too short: {ex.reasoning}"
            assert len(ex.answer) > 0, f"Answer empty: {ex.question}"
            assert len(ex.common_error) > 5, f"Common error too short: {ex.common_error}"

    def test_answers_are_numeric_or_simple(self):
        """Answers should be numeric or simple text."""
        examples = get_logical_shape_examples()

        for ex in examples:
            # Answer should be parseable as int/float or short text
            try:
                int(ex.answer)  # Most answers should be integers
            except ValueError:
                # Allow short text answers (e.g., "5 mph", "Yes")
                assert len(ex.answer) <= 20, f"Answer too long: {ex.answer}"

    def test_full_text_contains_gsm8k_format(self):
        """Full text should use GSM8K-compatible format with ####."""
        examples = get_logical_shape_examples()

        for ex in examples:
            text = ex.full_text
            assert "####" in text, f"Missing #### in: {ex.question[:50]}"
            assert "Question:" in text

    def test_no_duplicate_questions(self):
        """Questions should not be duplicated."""
        examples = get_logical_shape_examples()
        questions = [ex.question for ex in examples]

        # Normalize for comparison
        normalized = [q.strip().lower() for q in questions]
        unique = set(normalized)

        assert len(unique) == len(normalized), "Duplicate questions found"
