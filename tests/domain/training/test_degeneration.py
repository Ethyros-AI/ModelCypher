"""Tests for degeneration measurement (n-gram repetition rate)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.degeneration import fourgram_repetition_rate


def test_no_repetition_returns_zero():
    """Fully unique text has repetition rate 0."""
    text = "the quick brown fox jumps over the lazy dog today"
    rate = fourgram_repetition_rate(text)
    assert rate == 0.0


def test_fully_repetitive_returns_high():
    """Highly repetitive text has rate close to 1."""
    text = " ".join(["the same four words"] * 50)
    rate = fourgram_repetition_rate(text)
    assert rate > 0.9


def test_short_text_returns_zero():
    """Texts shorter than 4 words cannot have 4-gram repetition."""
    assert fourgram_repetition_rate("one two three") == 0.0
    assert fourgram_repetition_rate("one") == 0.0
    assert fourgram_repetition_rate("") == 0.0


def test_exactly_four_words_no_repetition():
    """Exactly 4 words = exactly 1 4-gram, rate = 0."""
    assert fourgram_repetition_rate("one two three four") == 0.0


def test_case_insensitive():
    """Repetition detection is case-insensitive."""
    text = "The quick brown fox The Quick Brown Fox"
    rate = fourgram_repetition_rate(text)
    assert rate > 0.0


def test_rate_in_unit_interval():
    """Rate is always in [0, 1]."""
    for text in [
        "a b c d e f g h i j",
        "a b c d a b c d a b c d",
        "x " * 100,
    ]:
        rate = fourgram_repetition_rate(text)
        assert 0.0 <= rate <= 1.0
