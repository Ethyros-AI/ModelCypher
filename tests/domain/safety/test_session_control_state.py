# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ConsentGrant,
    ScenarioMode,
    SessionControlState,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_scenario_mode_importable(self) -> None:
        assert ScenarioMode is not None

    def test_consent_grant_importable(self) -> None:
        assert ConsentGrant is not None

    def test_session_control_state_importable(self) -> None:
        assert SessionControlState is not None


# ---------------------------------------------------------------------------
# ScenarioMode
# ---------------------------------------------------------------------------


class TestScenarioMode:
    def test_enum_values(self) -> None:
        assert ScenarioMode.DEFAULT.value == "default"
        assert ScenarioMode.ROLEPLAY.value == "roleplay"
        assert ScenarioMode.HORROR.value == "horror"
        assert ScenarioMode.THERAPY.value == "therapy"
        assert ScenarioMode.FICTION.value == "fiction"

    def test_all_members_present(self) -> None:
        names = {m.name for m in ScenarioMode}
        assert names == {"DEFAULT", "ROLEPLAY", "HORROR", "THERAPY", "FICTION"}


# ---------------------------------------------------------------------------
# ConsentGrant - Instantiation
# ---------------------------------------------------------------------------


class TestConsentGrantInstantiation:
    def test_default_instantiation(self) -> None:
        cg = ConsentGrant()
        assert cg.is_granted is False
        assert cg.expires_at is None
        assert cg.remaining_turns is None

    def test_instantiation_with_values(self) -> None:
        expires = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=expires, remaining_turns=5)
        assert cg.is_granted is True
        assert cg.expires_at == expires
        assert cg.remaining_turns == 5


# ---------------------------------------------------------------------------
# ConsentGrant - is_active
# ---------------------------------------------------------------------------


class TestConsentGrantIsActive:
    def test_default_is_not_active(self) -> None:
        cg = ConsentGrant()
        assert cg.is_active() is False

    def test_granted_is_active(self) -> None:
        cg = ConsentGrant(is_granted=True)
        assert cg.is_active() is True

    def test_expired_is_not_active(self) -> None:
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=past)
        now = datetime(2025, 6, 15, tzinfo=timezone.utc)
        assert cg.is_active(now=now) is False

    def test_not_yet_expired_is_active(self) -> None:
        future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=future)
        now = datetime(2025, 6, 15, tzinfo=timezone.utc)
        assert cg.is_active(now=now) is True

    def test_expires_at_equal_to_now_is_not_active(self) -> None:
        t = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=t)
        assert cg.is_active(now=t) is False

    def test_remaining_turns_positive_is_active(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=3)
        assert cg.is_active() is True

    def test_remaining_turns_zero_is_not_active(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=0)
        assert cg.is_active() is False

    def test_remaining_turns_negative_is_not_active(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=-1)
        assert cg.is_active() is False

    def test_not_granted_ignores_other_fields(self) -> None:
        future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=False, expires_at=future, remaining_turns=10)
        assert cg.is_active() is False


# ---------------------------------------------------------------------------
# ConsentGrant - consume_turn
# ---------------------------------------------------------------------------


class TestConsentGrantConsumeTurn:
    def test_consume_turn_decrements(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=3)
        cg.consume_turn()
        assert cg.remaining_turns == 2
        assert cg.is_granted is True

    def test_consume_turn_to_zero_disables_grant(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=1)
        cg.consume_turn()
        assert cg.remaining_turns == 0
        assert cg.is_granted is False
        assert cg.is_active() is False

    def test_consume_turn_already_inactive_sets_granted_false(self) -> None:
        cg = ConsentGrant(is_granted=False, remaining_turns=5)
        cg.consume_turn()
        assert cg.is_granted is False

    def test_consume_turn_no_remaining_turns_stays_active(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=None)
        cg.consume_turn()
        assert cg.is_granted is True
        assert cg.remaining_turns is None

    def test_consume_turn_expired_disables_grant(self) -> None:
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=past, remaining_turns=5)
        now = datetime(2025, 6, 15, tzinfo=timezone.utc)
        cg.consume_turn(now=now)
        assert cg.is_granted is False

    def test_consume_multiple_turns(self) -> None:
        cg = ConsentGrant(is_granted=True, remaining_turns=3)
        cg.consume_turn()
        cg.consume_turn()
        assert cg.remaining_turns == 1
        assert cg.is_granted is True
        cg.consume_turn()
        assert cg.remaining_turns == 0
        assert cg.is_granted is False


# ---------------------------------------------------------------------------
# ConsentGrant - Serialization
# ---------------------------------------------------------------------------


class TestConsentGrantSerialization:
    def test_to_dict(self) -> None:
        expires = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        cg = ConsentGrant(is_granted=True, expires_at=expires, remaining_turns=5)
        d = cg.to_dict()
        assert d["is_granted"] is True
        assert d["expires_at"] == expires.isoformat()
        assert d["remaining_turns"] == 5

    def test_to_dict_defaults(self) -> None:
        cg = ConsentGrant()
        d = cg.to_dict()
        assert d["is_granted"] is False
        assert d["expires_at"] is None
        assert d["remaining_turns"] is None

    def test_from_dict(self) -> None:
        data = {
            "is_granted": True,
            "expires_at": "2025-12-31T23:59:59+00:00",
            "remaining_turns": 10,
        }
        cg = ConsentGrant.from_dict(data)
        assert cg.is_granted is True
        assert cg.expires_at == datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        assert cg.remaining_turns == 10

    def test_from_dict_empty(self) -> None:
        cg = ConsentGrant.from_dict({})
        assert cg.is_granted is False
        assert cg.expires_at is None
        assert cg.remaining_turns is None

    def test_round_trip(self) -> None:
        expires = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        original = ConsentGrant(is_granted=True, expires_at=expires, remaining_turns=7)
        restored = ConsentGrant.from_dict(original.to_dict())
        assert restored.is_granted == original.is_granted
        assert restored.expires_at == original.expires_at
        assert restored.remaining_turns == original.remaining_turns


# ---------------------------------------------------------------------------
# SessionControlState - Instantiation
# ---------------------------------------------------------------------------


class TestSessionControlStateInstantiation:
    def test_default_factory(self) -> None:
        state = SessionControlState.default()
        assert state.scenario == ScenarioMode.DEFAULT
        assert state.consent.is_granted is False
        assert state.consent.expires_at is None
        assert state.consent.remaining_turns is None

    def test_instantiation_with_values(self) -> None:
        consent = ConsentGrant(is_granted=True, remaining_turns=5)
        state = SessionControlState(scenario=ScenarioMode.HORROR, consent=consent)
        assert state.scenario == ScenarioMode.HORROR
        assert state.consent.is_granted is True

    def test_default_consent_is_new_instance(self) -> None:
        state = SessionControlState()
        assert isinstance(state.consent, ConsentGrant)
        assert state.consent.is_granted is False


# ---------------------------------------------------------------------------
# SessionControlState - is_consent_active
# ---------------------------------------------------------------------------


class TestSessionControlStateConsentActive:
    def test_delegates_to_consent(self) -> None:
        consent = ConsentGrant(is_granted=True)
        state = SessionControlState(consent=consent)
        assert state.is_consent_active() is True

    def test_not_active_when_consent_not_granted(self) -> None:
        state = SessionControlState.default()
        assert state.is_consent_active() is False

    def test_passes_now_to_consent(self) -> None:
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        consent = ConsentGrant(is_granted=True, expires_at=past)
        state = SessionControlState(consent=consent)
        now = datetime(2025, 6, 15, tzinfo=timezone.utc)
        assert state.is_consent_active(now=now) is False


# ---------------------------------------------------------------------------
# SessionControlState - Serialization
# ---------------------------------------------------------------------------


class TestSessionControlStateSerialization:
    def test_to_dict(self) -> None:
        consent = ConsentGrant(is_granted=True, remaining_turns=3)
        state = SessionControlState(scenario=ScenarioMode.ROLEPLAY, consent=consent)
        d = state.to_dict()
        assert d["scenario"] == "roleplay"
        assert d["consent"]["is_granted"] is True
        assert d["consent"]["remaining_turns"] == 3

    def test_from_dict(self) -> None:
        data = {
            "scenario": "horror",
            "consent": {
                "is_granted": True,
                "expires_at": None,
                "remaining_turns": 2,
            },
        }
        state = SessionControlState.from_dict(data)
        assert state.scenario == ScenarioMode.HORROR
        assert state.consent.is_granted is True
        assert state.consent.remaining_turns == 2

    def test_from_dict_empty(self) -> None:
        state = SessionControlState.from_dict({})
        assert state.scenario == ScenarioMode.DEFAULT
        assert state.consent.is_granted is False

    def test_round_trip(self) -> None:
        consent = ConsentGrant(
            is_granted=True,
            expires_at=datetime(2025, 12, 31, tzinfo=timezone.utc),
            remaining_turns=10,
        )
        original = SessionControlState(scenario=ScenarioMode.FICTION, consent=consent)
        restored = SessionControlState.from_dict(original.to_dict())
        assert restored.scenario == original.scenario
        assert restored.consent.is_granted == original.consent.is_granted
        assert restored.consent.expires_at == original.consent.expires_at
        assert restored.consent.remaining_turns == original.consent.remaining_turns
