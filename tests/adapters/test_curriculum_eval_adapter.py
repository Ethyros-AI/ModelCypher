# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

"""Non-GPU tests for the strict answer-span measurement operator.

Covers _extract_answer_span and the exact-mode scoring contract:
  - Primary metric scores on the answer span (first line) only
  - Explanation-only matches are tracked but NOT counted as correct
"""

from __future__ import annotations

import pytest

from modelcypher.adapters.curriculum_eval_adapter import _extract_answer_span


class TestExtractAnswerSpan:
    """First-line extraction from model output."""

    def test_single_line(self):
        assert _extract_answer_span("The fuse blows.") == "The fuse blows."

    def test_answer_then_explanation(self):
        predicted = "The fuse blows.\n\nExplanation: Overloading causes..."
        assert _extract_answer_span(predicted) == "The fuse blows."

    def test_answer_then_step_reasoning(self):
        predicted = "hypotension is diagnosed.\nStep 1: Analyze the premise..."
        assert _extract_answer_span(predicted) == "hypotension is diagnosed."

    def test_strips_whitespace(self):
        predicted = "  Caves develop.  \nMore text here"
        assert _extract_answer_span(predicted) == "Caves develop."

    def test_empty_string(self):
        assert _extract_answer_span("") == ""

    def test_only_newlines(self):
        assert _extract_answer_span("\n\n\n") == ""


class TestExactModeAnswerSpanScoring:
    """Verify that exact-mode scoring matches only the answer span.

    These tests exercise the scoring logic indirectly through the pure
    functions. The full evaluate_skill_mastery integration test requires
    GPU and runs separately.
    """

    def test_match_in_answer_span_is_correct(self):
        """Expected appears in first line -> correct."""
        predicted = "The fuse blows.\n\nExplanation: When overloaded..."
        span = _extract_answer_span(predicted)
        expected = "the fuse blows."
        assert expected in span.lower()

    def test_match_only_in_explanation_is_not_correct(self):
        """Expected appears in explanation but not first line -> not correct."""
        # Item 70 pattern: model says "cheaper" then explains "more expensive"
        predicted = (
            "borrowing becomes cheaper.\n"
            "The reasoning is based on the hypothetical statement that "
            "if the central bank raises interest rates, then borrowing "
            "becomes more expensive."
        )
        span = _extract_answer_span(predicted)
        expected = "borrowing becomes more expensive."
        assert expected not in span.lower()
        assert expected in predicted.lower()

    def test_negation_in_answer_span_does_not_match(self):
        """Item 23 pattern: adapter says 'not diagnosed' vs expected 'diagnosed'.

        The substring operator correctly rejects this: 'hypotension is diagnosed.'
        is NOT a contiguous substring of 'hypotension is not diagnosed.' because
        the inserted 'not ' breaks the match.
        """
        predicted = "hypotension is not diagnosed.\nStep 1: ..."
        span = _extract_answer_span(predicted)
        expected = "hypotension is diagnosed."
        assert expected not in span.lower()

    def test_explanation_substring_items(self):
        """Items 1, 54, 97 pattern: expected in explanation, not first line."""
        # Item 1: expected "it emits pulsar beams" only in explanation
        predicted = (
            "The neutron star must be spinning very fast.\n"
            "Explanation: When a neutron star spins rapidly, "
            "it emits pulsar beams."
        )
        span = _extract_answer_span(predicted)
        expected = "it emits pulsar beams."
        assert expected not in span.lower()
        assert expected in predicted.lower()
