"""JSON snippet extraction from agent responses.

Deterministic and conservative extraction of JSON objects from text,
preferring fenced code blocks when present.
"""

from __future__ import annotations


class AgentJSONSnippetExtractor:
    """Extracts JSON snippets from agent response text."""

    @staticmethod
    def extract_first_json_object(text: str) -> str | None:
        """Extract the first valid-looking JSON object substring from text.

        Extraction is deterministic and conservative:
        - Prefers fenced ```json blocks when present
        - Otherwise scans for the first balanced `{}` region, ignoring braces inside JSON strings

        Args:
            text: The text to extract from.

        Returns:
            The first JSON object found, or None.
        """
        trimmed = text.strip()
        if not trimmed:
            return None

        # Try fenced block first
        fenced = AgentJSONSnippetExtractor._extract_first_fenced_block(trimmed)
        if fenced is not None:
            fenced_trimmed = fenced.strip()
            if fenced_trimmed.startswith("{") and fenced_trimmed.endswith("}"):
                return fenced_trimmed

        # Fall back to balanced braces
        return AgentJSONSnippetExtractor._extract_first_balanced_braces(trimmed)

    @staticmethod
    def _extract_first_fenced_block(text: str) -> str | None:
        """Extract content from first fenced code block.

        Args:
            text: The text to search.

        Returns:
            Content of fenced block, or None.
        """
        # Fast path: first fence must be present
        open_index = text.find("```")
        if open_index == -1:
            return None

        after_open = open_index + 3

        # Find first newline after opening fence (skips language hint)
        first_newline = text.find("\n", after_open)
        if first_newline == -1:
            return None

        block_start = first_newline + 1

        # Find closing fence
        close_index = text.find("```", block_start)
        if close_index == -1:
            return None

        return text[block_start:close_index]

    @staticmethod
    def _extract_first_balanced_braces(text: str) -> str | None:
        """Extract first balanced braces region, respecting JSON strings.

        Args:
            text: The text to search.

        Returns:
            First balanced braces region, or None.
        """
        depth = 0
        in_string = False
        is_escaped = False
        start_index: int | None = None

        for i, char in enumerate(text):
            if in_string:
                if is_escaped:
                    is_escaped = False
                    continue

                if char == "\\":
                    is_escaped = True
                    continue

                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start_index = i
                depth += 1
                continue

            if char == "}":
                if depth <= 0:
                    continue
                depth -= 1
                if depth == 0 and start_index is not None:
                    return text[start_index : i + 1]

        return None
