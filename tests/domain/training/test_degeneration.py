"""Tests for degeneration measurement (n-gram repetition rate)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.degeneration import (
    derive_ngram_order,
    ngram_repetition_rate,
    sequence_ngram_repetition_rate,
)


# --- derive_ngram_order tests ---


def test_derive_ngram_order_small_erank():
    """r_eff=10 → n=6 (severely constrained model)."""
    assert derive_ngram_order(10.0, 400) == 6


def test_derive_ngram_order_medium_erank():
    """r_eff=50 → n=4 (matches historical hardcoded value for small models)."""
    assert derive_ngram_order(50.0, 400) == 4


def test_derive_ngram_order_large_erank():
    """r_eff=200 → n=3 (larger model: more sensitive detection)."""
    assert derive_ngram_order(200.0, 400) == 3


def test_derive_ngram_order_degenerate():
    """r_eff < 2 clamps to n=2."""
    assert derive_ngram_order(1.5, 400) == 2


def test_derive_ngram_order_floor_at_two():
    """Even very large r_eff never drops below n=2."""
    assert derive_ngram_order(1e6, 400) >= 2


def test_derive_ngram_order_monotone_in_erank():
    """Higher effective rank → smaller or equal n-gram order."""
    n_low = derive_ngram_order(10.0, 400)
    n_high = derive_ngram_order(200.0, 400)
    assert n_low >= n_high


def test_derive_ngram_order_monotone_in_length():
    """Longer generation → larger or equal n-gram order."""
    n_short = derive_ngram_order(50.0, 100)
    n_long = derive_ngram_order(50.0, 1000)
    assert n_long >= n_short


# --- ngram_repetition_rate tests ---


def test_ngram_repetition_rate_no_repeats():
    """Fully unique text has repetition rate 0."""
    assert ngram_repetition_rate("a b c d e f g h", 4) == 0.0


def test_ngram_repetition_rate_all_repeats():
    """Highly repetitive text has rate > 0.5."""
    rate = ngram_repetition_rate("a b c d a b c d a b c d", 4)
    assert rate > 0.5


def test_ngram_repetition_rate_short_text():
    """Text shorter than n words returns 0."""
    assert ngram_repetition_rate("a b c", 4) == 0.0


def test_ngram_repetition_rate_variable_n():
    """n=2 detects bigram repetition."""
    rate = ngram_repetition_rate("a b a b a b", 2)
    assert rate > 0.5


def test_ngram_repetition_rate_in_unit_interval():
    """Rate is always in [0, 1]."""
    for text in [
        "a b c d e f g h i j",
        "a b c d a b c d a b c d",
        "x " * 100,
    ]:
        for n in [2, 3, 4, 5]:
            rate = ngram_repetition_rate(text, n)
            assert 0.0 <= rate <= 1.0


# --- n=4 specific tests ---


def test_no_repetition_returns_zero():
    """Fully unique text has repetition rate 0."""
    text = "the quick brown fox jumps over the lazy dog today"
    rate = ngram_repetition_rate(text, 4)
    assert rate == 0.0


def test_fully_repetitive_returns_high():
    """Highly repetitive text has rate close to 1."""
    text = " ".join(["the same four words"] * 50)
    rate = ngram_repetition_rate(text, 4)
    assert rate > 0.9


def test_short_text_returns_zero():
    """Texts shorter than 4 words cannot have 4-gram repetition."""
    assert ngram_repetition_rate("one two three", 4) == 0.0
    assert ngram_repetition_rate("one", 4) == 0.0
    assert ngram_repetition_rate("", 4) == 0.0


def test_exactly_four_words_no_repetition():
    """Exactly 4 words = exactly 1 4-gram, rate = 0."""
    assert ngram_repetition_rate("one two three four", 4) == 0.0


def test_case_insensitive():
    """Repetition detection is case-insensitive."""
    text = "The quick brown fox The Quick Brown Fox"
    rate = ngram_repetition_rate(text, 4)
    assert rate > 0.0


def test_rate_in_unit_interval():
    """Rate is always in [0, 1]."""
    for text in [
        "a b c d e f g h i j",
        "a b c d a b c d a b c d",
        "x " * 100,
    ]:
        rate = ngram_repetition_rate(text, 4)
        assert 0.0 <= rate <= 1.0


# --- sequence_ngram_repetition_rate tests ---


def test_sequence_ngram_no_repeats():
    """Fully unique token sequence has repetition rate 0."""
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    assert sequence_ngram_repetition_rate(tokens, 4) == 0.0


def test_sequence_ngram_all_repeats():
    """Repeating token pattern has high repetition rate."""
    tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    rate = sequence_ngram_repetition_rate(tokens, 4)
    assert rate > 0.5


def test_sequence_ngram_short():
    """Sequence shorter than n returns 0."""
    assert sequence_ngram_repetition_rate([1, 2, 3], 4) == 0.0
    assert sequence_ngram_repetition_rate([], 4) == 0.0


def test_sequence_ngram_matches_text():
    """sequence_ngram_repetition_rate on word list equals ngram_repetition_rate on text."""
    text = "a b c d a b c d a b c d"
    words = text.lower().split()
    assert sequence_ngram_repetition_rate(words, 4) == ngram_repetition_rate(text, 4)


def test_sequence_ngram_unit_interval():
    """Rate is always in [0, 1]."""
    for tokens in [
        list(range(20)),
        [1, 2, 3, 4] * 10,
        [42] * 50,
    ]:
        for n in [2, 3, 4, 5]:
            rate = sequence_ngram_repetition_rate(tokens, n)
            assert 0.0 <= rate <= 1.0
