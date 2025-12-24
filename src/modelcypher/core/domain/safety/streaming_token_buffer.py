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

"""Circular buffer for cross-token pattern detection in streaming output.

When model outputs arrive token-by-token, dangerous patterns can span multiple
tokens (e.g., "ki" + "ll" = "kill"). This buffer maintains a sliding window
of recent content to detect such patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_WINDOW_SIZE = 200
"""Default window size in characters."""

MIN_WINDOW_SIZE = 10
"""Minimum allowed window size."""


@dataclass
class StreamingTokenBuffer:
    """Circular buffer that accumulates streaming tokens for cross-token pattern detection.

    Maintains a sliding window of recent content to detect patterns that span
    multiple tokens.

    This is a mutable class - callers must manage synchronization if used
    concurrently.
    """

    window_size: int = DEFAULT_WINDOW_SIZE
    """Maximum characters to retain."""

    _content: str = field(default="", init=False, repr=False)
    """Accumulated content."""

    _total_processed: int = field(default=0, init=False, repr=False)
    """Total characters processed (before truncation)."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_size < MIN_WINDOW_SIZE:
            object.__setattr__(self, "window_size", MIN_WINDOW_SIZE)

    def append(self, token: str) -> str:
        """Append a token to the buffer, maintaining the sliding window.

        Args:
            token: The token string to append.

        Returns:
            The content window after appending.
        """
        self._content += token
        self._total_processed += len(token)

        # Trim from front if over capacity
        if len(self._content) > self.window_size:
            excess = len(self._content) - self.window_size
            self._content = self._content[excess:]

        return self._content

    @property
    def window(self) -> str:
        """Current content window."""
        return self._content

    @property
    def count(self) -> int:
        """Number of characters in the current window."""
        return len(self._content)

    @property
    def processed_count(self) -> int:
        """Total characters processed since creation."""
        return self._total_processed

    @property
    def is_empty(self) -> bool:
        """Whether the buffer is empty."""
        return len(self._content) == 0

    def reset(self) -> None:
        """Clear all content and reset counters."""
        self._content = ""
        self._total_processed = 0

    def suffix(self, count: int) -> str:
        """Return the last N characters from the buffer.

        Args:
            count: Number of characters to return.

        Returns:
            The suffix of the content, or all content if count exceeds length.
        """
        if count >= len(self._content):
            return self._content
        return self._content[-count:]
