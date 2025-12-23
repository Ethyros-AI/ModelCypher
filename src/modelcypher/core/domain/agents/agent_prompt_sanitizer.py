"""Agent prompt sanitization to align with intrinsic identity.

Sanitizes agent prompts to align with intrinsic identity (being vs playing).
This is intentionally minimal and deterministic:
- It does not attempt semantic classification.
- It enforces a small "context-only" contract for system prompts when enabled.

See Intrinsic_Agents.md: safe agents should not be instructed to "play a role"
via runtime identity prompts. System prompts should carry *context*, not *identity*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from modelcypher.core.domain.agents.intrinsic_identity_rules import (
    IntrinsicIdentityRules,
)


class AgentRole(str, Enum):
    """Role in agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class AgentMessage:
    """A message in agent conversation history."""

    role: AgentRole
    """Message role."""

    content: str
    """Message content."""

    timestamp: Optional[datetime] = None
    """When the message was created."""


class AgentSystemPromptPolicy(str, Enum):
    """Policy for handling system prompts in agent mode.

    See Intrinsic_Agents.md: safe agents should not be instructed to "play a role"
    via runtime identity prompts. System prompts should carry *context*, not *identity*.
    """

    ALLOW_ALL = "allow_all"
    """Accept system prompts and system messages as provided."""

    CONTEXT_ONLY = "context_only"
    """Only allow system prompts/messages that look like environment context.

    This drops identity-style prompts (e.g., "You are ...", "Act as ...") to reduce
    the meta-layer attack surface.
    """


@dataclass(frozen=True)
class AgentPromptSanitizationResult:
    """Result of agent prompt sanitization."""

    system_prompt: Optional[str]
    """Sanitized system prompt, or None if dropped."""

    history: Optional[list[AgentMessage]]
    """Sanitized conversation history, or None if empty."""

    dropped_system_prompt: bool
    """Whether the system prompt was dropped."""

    dropped_system_message_count: int
    """Number of system messages dropped from history."""

    notes: list[str] = field(default_factory=list)
    """Notes about sanitization actions taken."""


class AgentPromptSanitizer:
    """Sanitizes agent prompts to align with intrinsic identity (being vs playing).

    This is intentionally minimal and deterministic:
    - It does not attempt semantic classification.
    - It enforces a small "context-only" contract for system prompts when enabled.
    """

    @staticmethod
    def sanitize(
        system_prompt: Optional[str],
        history: Optional[list[AgentMessage]],
        policy: AgentSystemPromptPolicy,
    ) -> AgentPromptSanitizationResult:
        """Sanitize agent prompts according to policy.

        Args:
            system_prompt: The system prompt to sanitize.
            history: Conversation history to sanitize.
            policy: The sanitization policy to apply.

        Returns:
            Sanitization result with cleaned prompts and notes.
        """
        if policy == AgentSystemPromptPolicy.ALLOW_ALL:
            return AgentPromptSanitizationResult(
                system_prompt=system_prompt,
                history=history,
                dropped_system_prompt=False,
                dropped_system_message_count=0,
                notes=[],
            )

        # Context-only policy
        notes: list[str] = []

        # Process system prompt
        cleaned_system_prompt = AgentPromptSanitizer._sanitize_system_text(system_prompt)
        system_prompt_allowed = (
            cleaned_system_prompt is not None
            and AgentPromptSanitizer._is_allowed_context_system_text(
                cleaned_system_prompt
            )
        )

        final_system_prompt: Optional[str] = (
            cleaned_system_prompt if system_prompt_allowed else None
        )

        dropped_system_prompt = (
            system_prompt is not None
            and system_prompt.strip() != ""
            and final_system_prompt is None
        )
        if dropped_system_prompt:
            notes.append("dropped_system_prompt")

        # Process history
        dropped_system_messages = 0
        final_history: Optional[list[AgentMessage]] = None

        if history is not None:
            filtered: list[AgentMessage] = []
            for message in history:
                if message.role != AgentRole.SYSTEM:
                    filtered.append(message)
                    continue

                cleaned = AgentPromptSanitizer._sanitize_system_text(message.content)
                if cleaned is None or not AgentPromptSanitizer._is_allowed_context_system_text(
                    cleaned
                ):
                    dropped_system_messages += 1
                    continue

                filtered.append(
                    AgentMessage(
                        role=AgentRole.SYSTEM,
                        content=cleaned,
                        timestamp=message.timestamp,
                    )
                )

            if dropped_system_messages > 0:
                notes.append(f"dropped_system_messages={dropped_system_messages}")

            final_history = filtered if filtered else None

        return AgentPromptSanitizationResult(
            system_prompt=final_system_prompt,
            history=final_history,
            dropped_system_prompt=dropped_system_prompt,
            dropped_system_message_count=dropped_system_messages,
            notes=notes,
        )

    @staticmethod
    def _sanitize_system_text(text: Optional[str]) -> Optional[str]:
        """Sanitize system text using intrinsic identity rules."""
        return IntrinsicIdentityRules.sanitize_text(text)

    @staticmethod
    def _is_allowed_context_system_text(text: str) -> bool:
        """Check if text is allowed context system text."""
        return IntrinsicIdentityRules.is_context_only_system_text(text)

    @staticmethod
    def _looks_like_environment_context(text: str) -> bool:
        """Check if text looks like environment context."""
        return IntrinsicIdentityRules.looks_like_environment_context(text)

    @staticmethod
    def _contains_identity_instruction(text: str) -> bool:
        """Check if text contains identity instruction."""
        return IntrinsicIdentityRules.contains_identity_instruction(text)
