"""Tests for StreamingTokenBuffer sliding window behavior."""

import pytest

from modelcypher.core.domain.safety.streaming_token_buffer import StreamingTokenBuffer


class TestStreamingTokenBufferInit:
    """Validation on construction."""

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            StreamingTokenBuffer(window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            StreamingTokenBuffer(window_size=-1)

    def test_window_size_negative_large_raises(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            StreamingTokenBuffer(window_size=-100)

    def test_valid_window_size_creates_buffer(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.window_size == 10


class TestStreamingTokenBufferEmpty:
    """Fresh buffer state."""

    def test_is_empty(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.is_empty is True

    def test_count_zero(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.count == 0

    def test_processed_count_zero(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.processed_count == 0

    def test_window_empty_string(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.window == ""

    def test_suffix_on_empty_returns_empty(self):
        buf = StreamingTokenBuffer(window_size=10)
        assert buf.suffix(5) == ""


class TestStreamingTokenBufferAppend:
    """Append behavior and return value."""

    def test_append_returns_window_content(self):
        buf = StreamingTokenBuffer(window_size=20)
        result = buf.append("hello")
        assert result == "hello"

    def test_append_accumulates(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("hello")
        result = buf.append(" world")
        assert result == "hello world"

    def test_append_updates_count(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("abc")
        assert buf.count == 3

    def test_append_updates_processed_count(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("abc")
        assert buf.processed_count == 3

    def test_append_updates_is_empty(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("x")
        assert buf.is_empty is False

    def test_append_single_chars_build_up(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("a")
        buf.append("b")
        buf.append("c")
        assert buf.window == "abc"
        assert buf.count == 3
        assert buf.processed_count == 3

    def test_append_empty_string_is_noop(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("hi")
        result = buf.append("")
        assert result == "hi"
        assert buf.count == 2
        assert buf.processed_count == 2


class TestStreamingTokenBufferSlidingWindow:
    """Truncation when content exceeds window_size."""

    def test_truncates_when_exceeding_window(self):
        buf = StreamingTokenBuffer(window_size=5)
        buf.append("abcde")
        result = buf.append("fg")
        # "abcdefg" -> trim first 2 -> "cdefg"
        assert result == "cdefg"
        assert buf.count == 5

    def test_processed_count_tracks_lifetime_after_truncation(self):
        buf = StreamingTokenBuffer(window_size=5)
        buf.append("abcde")  # 5 chars, no truncation
        buf.append("fgh")  # 8 total processed, window = "defgh"
        assert buf.processed_count == 8
        assert buf.count == 5

    def test_window_size_one(self):
        buf = StreamingTokenBuffer(window_size=1)
        buf.append("a")
        assert buf.window == "a"
        buf.append("b")
        assert buf.window == "b"
        buf.append("cd")
        assert buf.window == "d"
        assert buf.processed_count == 4

    def test_large_token_exceeding_window(self):
        buf = StreamingTokenBuffer(window_size=3)
        result = buf.append("abcdefghij")
        assert result == "hij"
        assert buf.count == 3
        assert buf.processed_count == 10

    def test_repeated_truncations(self):
        buf = StreamingTokenBuffer(window_size=4)
        buf.append("ab")
        buf.append("cd")  # "abcd", exactly at limit
        assert buf.window == "abcd"
        buf.append("ef")  # "abcdef" -> "cdef"
        assert buf.window == "cdef"
        buf.append("gh")  # "cdefgh" -> "efgh"
        assert buf.window == "efgh"
        assert buf.processed_count == 8

    def test_exactly_at_window_size_no_truncation(self):
        buf = StreamingTokenBuffer(window_size=5)
        buf.append("abcde")
        assert buf.window == "abcde"
        assert buf.count == 5


class TestStreamingTokenBufferSuffix:
    """suffix(count) returns trailing characters."""

    def test_suffix_returns_last_n_chars(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("hello world")
        assert buf.suffix(5) == "world"

    def test_suffix_returns_all_when_count_exceeds_length(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("hi")
        assert buf.suffix(100) == "hi"

    def test_suffix_equal_to_length(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("abc")
        assert buf.suffix(3) == "abc"

    def test_suffix_zero_returns_all(self):
        """suffix(0) returns full content because Python s[-0:] == s[0:] == s."""
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("abc")
        assert buf.suffix(0) == "abc"

    def test_suffix_one(self):
        buf = StreamingTokenBuffer(window_size=20)
        buf.append("abc")
        assert buf.suffix(1) == "c"

    def test_suffix_after_truncation(self):
        buf = StreamingTokenBuffer(window_size=5)
        buf.append("abcdefgh")
        # window = "defgh"
        assert buf.suffix(3) == "fgh"


class TestStreamingTokenBufferReset:
    """reset() clears all state."""

    def test_reset_clears_content(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("hello")
        buf.reset()
        assert buf.window == ""

    def test_reset_clears_count(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("hello")
        buf.reset()
        assert buf.count == 0

    def test_reset_clears_processed_count(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("hello")
        buf.reset()
        assert buf.processed_count == 0

    def test_reset_sets_is_empty(self):
        buf = StreamingTokenBuffer(window_size=10)
        buf.append("hello")
        buf.reset()
        assert buf.is_empty is True

    def test_usable_after_reset(self):
        buf = StreamingTokenBuffer(window_size=5)
        buf.append("abcde")
        buf.reset()
        result = buf.append("xy")
        assert result == "xy"
        assert buf.count == 2
        assert buf.processed_count == 2
