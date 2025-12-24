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

"""Deterministic redaction helpers for agent traces.

This is intentionally conservative: it aims to remove common high-risk tokens
(usernames, emails, API keys) while retaining enough surface structure to debug
"policy" behavior locally.
"""

from __future__ import annotations

import re
from typing import Any


# Pre-compiled regex patterns for sensitive content
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_HUGGINGFACE_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-\._~\+\/]+=*\b")
_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_USER_PATH_PATTERN = re.compile(r"/Users/[^/\s]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


class AgentTraceSanitizer:
    """Deterministic redaction helpers for agent traces."""

    @staticmethod
    def sanitized_preview(text: str, max_length: int = 256) -> str:
        """Create a sanitized preview of text.

        Args:
            text: The text to sanitize and preview.
            max_length: Maximum length of preview.

        Returns:
            Sanitized preview string.
        """
        # Collapse whitespace
        collapsed = _WHITESPACE_PATTERN.sub(" ", text.strip())

        sanitized = AgentTraceSanitizer.sanitize(collapsed)

        if len(sanitized) <= max_length:
            return sanitized

        return sanitized[:max_length] + "…"

    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize text by redacting sensitive content.

        Args:
            text: The text to sanitize.

        Returns:
            Sanitized text with sensitive content redacted.
        """
        if not text:
            return text

        result = text

        # Redact common secret-like tokens
        result = _OPENAI_KEY_PATTERN.sub("sk-<redacted>", result)
        result = _HUGGINGFACE_TOKEN_PATTERN.sub("hf_<redacted>", result)
        result = _BEARER_TOKEN_PATTERN.sub("Bearer <redacted>", result)

        # Redact email addresses
        result = _EMAIL_PATTERN.sub("<email:redacted>", result)

        # Redact local usernames in common macOS paths
        result = _USER_PATH_PATTERN.sub("/Users/<user>", result)

        return result

    @staticmethod
    def sanitize_json_value(value: Any) -> Any:
        """Recursively sanitize a JSON-like value.

        Args:
            value: The value to sanitize (dict, list, str, or primitive).

        Returns:
            Sanitized value with strings redacted.
        """
        if isinstance(value, str):
            return AgentTraceSanitizer.sanitize(value)
        elif isinstance(value, dict):
            return {k: AgentTraceSanitizer.sanitize_json_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [AgentTraceSanitizer.sanitize_json_value(item) for item in value]
        else:
            # Numbers, bools, None pass through unchanged
            return value
