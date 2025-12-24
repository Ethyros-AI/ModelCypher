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

"""Chat message types for conversational datasets.

Provides flexible deserialization from various field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """Chat message for conversational datasets.

    Supports multiple content field names: content, text, value, message.
    """

    role: str
    """Message role (system, user, assistant, tool)."""

    content: str
    """Message content."""

    name: str | None = None
    """Optional name for tool messages."""

    tool_call_id: str | None = None
    """Tool call ID for tool response messages."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Create from dictionary with flexible field names.

        Accepts content under: content, text, value, or message.

        Args:
            data: Dictionary with role and content.

        Returns:
            ChatMessage instance.

        Raises:
            ValueError: If role or content is missing.
        """
        role = data.get("role")
        if not role:
            raise ValueError("ChatMessage requires 'role' field")

        # Try multiple content field names
        content = (
            data.get("content")
            or data.get("text")
            or data.get("value")
            or data.get("message")
        )

        if content is None:
            raise ValueError(
                "ChatMessage requires content under one of: content, text, value, message"
            )

        return cls(
            role=str(role),
            content=str(content),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary with role and content.
        """
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result

    @property
    def is_system(self) -> bool:
        """Whether this is a system message."""
        return self.role == "system"

    @property
    def is_user(self) -> bool:
        """Whether this is a user message."""
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        """Whether this is an assistant message."""
        return self.role == "assistant"

    @property
    def is_tool(self) -> bool:
        """Whether this is a tool message."""
        return self.role == "tool"


def parse_messages(data: list[dict[str, Any]]) -> list[ChatMessage]:
    """Parse a list of message dictionaries.

    Args:
        data: List of message dictionaries.

    Returns:
        List of ChatMessage instances.

    Raises:
        ValueError: If any message is invalid.
    """
    return [ChatMessage.from_dict(msg) for msg in data]
