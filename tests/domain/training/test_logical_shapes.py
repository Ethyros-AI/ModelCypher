# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
    def test_exactly_six_shapes(self):
        """6 fundamental shapes — adding one requires updating all curricula."""
        assert len(LogicalShape) == 6

    def test_all_values_distinct(self):
        """No duplicate enum values (prevents silent shadowing)."""
        values = [s.value for s in LogicalShape]
        assert len(values) == len(set(values))


class TestGetShapeDescription:
    def test_all_shapes_have_descriptions(self):
        """Every shape has a non-empty description, not the fallback."""
        for shape in LogicalShape:
            desc = get_shape_description(shape)
            assert len(desc) > 0
            assert "Unknown" not in desc

    def test_descriptions_mention_common_error(self):
        """Every description warns about the common mistake."""
        for shape in LogicalShape:
            desc = get_shape_description(shape)
            assert "error" in desc.lower() or "wrong" in desc.lower()


class TestLogicalShapeExample:
    def _make_example(self) -> LogicalShapeExample:
        return LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="Price $100 increases by 50%. New price?",
            reasoning="50% of 100 = 50. New = 100 + 50 = 150.",
            answer="150",
            common_error="Multiplying by 0.5 gives $50 (wrong)",
        )

    def test_full_text_contains_all_parts(self):
        """full_text includes question, shape name, reasoning, and answer marker."""
        ex = self._make_example()
        text = ex.full_text
        assert "Price $100" in text
        assert "PERCENTAGE INCREASE" in text
        assert "50% of 100" in text
        assert "#### 150" in text

    def test_training_dict_has_text_key(self):
        """training_dict produces JSONL-compatible format."""
        ex = self._make_example()
        d = ex.training_dict
        assert "text" in d
        assert d["text"] == ex.full_text

    def test_as_dict_has_all_keys(self):
        """as_dict serializes all fields for JSON export."""
        ex = self._make_example()
        d = ex.as_dict()
        assert d["shape"] == "percentage_increase"
        assert d["question"] == ex.question
        assert d["reasoning"] == ex.reasoning
        assert d["answer"] == "150"
        assert d["common_error"] == ex.common_error

    def test_frozen(self):
        """Examples are immutable — prevents accidental mutation during training."""
        ex = self._make_example()
        with pytest.raises(AttributeError):
            ex.answer = "999"


class TestGetLogicalShapeExamples:
    def test_total_count(self):
        """90 examples total (15 per shape x 6 shapes)."""
        examples = get_logical_shape_examples()
        assert len(examples) == 90

    def test_fifteen_per_shape(self):
        """Each shape has exactly 15 examples — balanced curriculum."""
        examples = get_logical_shape_examples()
        counts: dict[LogicalShape, int] = {}
        for ex in examples:
            counts[ex.shape] = counts.get(ex.shape, 0) + 1
        for shape in LogicalShape:
            assert counts.get(shape, 0) == 15, (
                f"{shape.value} has {counts.get(shape, 0)} examples, expected 15"
            )

    def test_all_examples_have_nonempty_fields(self):
        """No blank questions, reasoning, answers, or errors."""
        examples = get_logical_shape_examples()
        for i, ex in enumerate(examples):
            assert len(ex.question) > 0, f"Example {i} has empty question"
            assert len(ex.reasoning) > 0, f"Example {i} has empty reasoning"
            assert len(ex.answer) > 0, f"Example {i} has empty answer"
            assert len(ex.common_error) > 0, f"Example {i} has empty common_error"

    def test_full_text_identifies_shape(self):
        """Every example's full_text explicitly names its logical shape."""
        examples = get_logical_shape_examples()
        for ex in examples:
            shape_label = ex.shape.value.upper().replace("_", " ")
            assert shape_label in ex.full_text, (
                f"full_text missing shape label '{shape_label}' for {ex.shape.value}"
            )
