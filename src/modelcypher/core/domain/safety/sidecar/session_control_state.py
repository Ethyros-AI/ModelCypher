"""Session control state for consent-gated safety thresholds.

This module provides types for managing scenario modes and consent grants
that affect safety threshold behavior without enabling unsafe content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ScenarioMode(str, Enum):
    """Scenario mode for consent-gated safety thresholds.

    This is a *control input*, not a content label. It never enables unsafe
    behavior; it only adjusts soft thresholds for instability detection while
    hard-stop policies remain invariant.
    """

    DEFAULT = "default"
    """Default scenario mode with standard thresholds."""

    ROLEPLAY = "roleplay"
    """Roleplay scenario with consent-adjustable thresholds."""

    HORROR = "horror"
    """Horror scenario with consent-adjustable thresholds."""

    THERAPY = "therapy"
    """Therapy scenario with adjusted thresholds."""

    FICTION = "fiction"
    """Fiction scenario with consent-adjustable thresholds."""


@dataclass
class ConsentGrant:
    """Session-scoped consent grant used to adjust safety thresholds.

    Consent can be time-limited (via expiresAt) or turn-limited (via
    remaining_turns). When either limit is reached, consent becomes inactive.
    """

    is_granted: bool = False
    """Whether the user has granted consent for the current scenario."""

    expires_at: datetime | None = None
    """Optional expiration time for the consent."""

    remaining_turns: int | None = None
    """Optional remaining turn budget. When it reaches zero, consent becomes inactive."""

    def is_active(self, now: datetime | None = None) -> bool:
        """Check if consent is currently active.

        Args:
            now: Current time. Defaults to UTC now.

        Returns:
            True if consent is active.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not self.is_granted:
            return False

        if self.expires_at is not None and now >= self.expires_at:
            return False

        if self.remaining_turns is not None and self.remaining_turns <= 0:
            return False

        return True

    def consume_turn(self, now: datetime | None = None) -> None:
        """Consume a turn from the remaining turn budget.

        If consent becomes inactive (either expired or out of turns),
        is_granted is set to False.

        Args:
            now: Current time. Defaults to UTC now.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not self.is_active(now):
            self.is_granted = False
            return

        if self.remaining_turns is not None:
            self.remaining_turns = max(0, self.remaining_turns - 1)
            if self.remaining_turns == 0:
                self.is_granted = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_granted": self.is_granted,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "remaining_turns": self.remaining_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConsentGrant:
        """Create from dictionary."""
        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return cls(
            is_granted=data.get("is_granted", False),
            expires_at=expires_at,
            remaining_turns=data.get("remaining_turns"),
        )


@dataclass
class SessionControlState:
    """Session-level control state that affects safety gating behavior."""

    scenario: ScenarioMode = ScenarioMode.DEFAULT
    """Current scenario mode."""

    consent: ConsentGrant = field(default_factory=ConsentGrant)
    """Current consent grant."""

    @classmethod
    def default(cls) -> SessionControlState:
        """Create default session control state."""
        return cls(scenario=ScenarioMode.DEFAULT, consent=ConsentGrant())

    def is_consent_active(self, now: datetime | None = None) -> bool:
        """Check if consent is currently active.

        Args:
            now: Current time. Defaults to UTC now.

        Returns:
            True if consent is active.
        """
        return self.consent.is_active(now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scenario": self.scenario.value,
            "consent": self.consent.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionControlState:
        """Create from dictionary."""
        return cls(
            scenario=ScenarioMode(data.get("scenario", "default")),
            consent=ConsentGrant.from_dict(data.get("consent", {})),
        )
