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

"""Deterministic heuristics for enforcing intrinsic identity in prompts and training data.

See Intrinsic_Agents.md: safe agents should be what they do, not what they role-play.
"""

from __future__ import annotations


class IntrinsicIdentityRules:
    """Deterministic heuristics for enforcing intrinsic identity in prompts and training data.

    See Intrinsic_Agents.md: safe agents should be what they do, not what they role-play.
    """

    # Identity instruction patterns (checked at word boundaries)
    _BOUNDARY_PATTERNS: list[str] = [
        "you are",
        "act as",
        "stay in role",
        "as a",
        "as an",
        "as the",
        "i am",
        "i'm",
    ]

    # Patterns checked via simple substring match
    _SUBSTRING_PATTERNS: list[str] = [
        "roleplay",
        "pretend",
    ]

    # Assistant roleplay leak patterns (checked at word boundaries)
    _ASSISTANT_LEAK_BOUNDARY_PATTERNS: list[str] = [
        "as an ai language model",
        "as a language model",
        "as chatgpt",
        "as a helpful assistant",
        "as an assistant",
        "as a",
        "as an",
        "as the",
        "i am an ai",
        "i'm an ai",
        "i am a model",
        "i'm a model",
        "i am chatgpt",
        "i'm chatgpt",
    ]

    @staticmethod
    def _is_alphanumeric(char: str) -> bool:
        """Check if character is alphanumeric."""
        return char.isalnum()

    @staticmethod
    def _contains_phrase_at_word_boundaries(lowercased_text: str, phrase: str) -> bool:
        """Check if phrase appears at word boundaries in text."""
        if not lowercased_text or not phrase:
            return False

        start = 0
        while True:
            index = lowercased_text.find(phrase, start)
            if index == -1:
                return False

            before_index = index
            after_index = index + len(phrase)

            # Check before boundary
            before_is_boundary = before_index == 0 or not IntrinsicIdentityRules._is_alphanumeric(
                lowercased_text[before_index - 1]
            )

            # Check after boundary
            after_is_boundary = after_index >= len(
                lowercased_text
            ) or not IntrinsicIdentityRules._is_alphanumeric(lowercased_text[after_index])

            if before_is_boundary and after_is_boundary:
                return True

            start = after_index

    @staticmethod
    def sanitize_text(text: str | None) -> str | None:
        """Sanitize text by trimming whitespace.

        Args:
            text: The text to sanitize.

        Returns:
            Trimmed text, or None if empty.
        """
        if text is None:
            return None
        trimmed = text.strip()
        return trimmed if trimmed else None

    @staticmethod
    def looks_like_environment_context(text: str) -> bool:
        """Check if text looks like environment context.

        Args:
            text: The text to check.

        Returns:
            True if text appears to be environment context.
        """
        trimmed = text.strip()
        if not trimmed:
            return False

        # Check for bracketed context
        if trimmed.startswith("[") and trimmed.endswith("]"):
            return True

        # Check for context XML tags
        lower = trimmed.lower()
        if lower.startswith("<context>") and "</context>" in lower:
            return True
        if lower.startswith("<environment_context>") and "</environment_context>" in lower:
            return True

        return False

    @staticmethod
    def contains_identity_instruction(text: str) -> bool:
        """Check if text contains identity instruction patterns.

        Args:
            text: The text to check.

        Returns:
            True if text contains identity instructions.
        """
        lower = text.lower()

        # Check boundary patterns
        for pattern in IntrinsicIdentityRules._BOUNDARY_PATTERNS:
            if IntrinsicIdentityRules._contains_phrase_at_word_boundaries(lower, pattern):
                return True

        # Check substring patterns
        for pattern in IntrinsicIdentityRules._SUBSTRING_PATTERNS:
            if pattern in lower:
                return True

        return False

    @staticmethod
    def is_context_only_system_text(text: str) -> bool:
        """Check if text is context-only system text.

        Args:
            text: The text to check.

        Returns:
            True if text is environment context without identity instructions.
        """
        return IntrinsicIdentityRules.looks_like_environment_context(
            text
        ) and not IntrinsicIdentityRules.contains_identity_instruction(text)

    @staticmethod
    def contains_assistant_roleplay_leak(text: str) -> bool:
        """Check if text contains assistant roleplay leak patterns.

        Args:
            text: The text to check.

        Returns:
            True if text contains assistant roleplay leaks.
        """
        lower = text.lower()

        # Check boundary patterns
        for pattern in IntrinsicIdentityRules._ASSISTANT_LEAK_BOUNDARY_PATTERNS:
            if IntrinsicIdentityRules._contains_phrase_at_word_boundaries(lower, pattern):
                return True

        # Check substring patterns
        for pattern in IntrinsicIdentityRules._SUBSTRING_PATTERNS:
            if pattern in lower:
                return True

        return False

    @staticmethod
    def excerpt(text: str, max_length: int = 80) -> str:
        """Create an excerpt of text.

        Args:
            text: The text to excerpt.
            max_length: Maximum length of excerpt.

        Returns:
            Excerpted text.
        """
        trimmed = text.strip()
        if len(trimmed) <= max_length:
            return trimmed
        return trimmed[:max_length]
