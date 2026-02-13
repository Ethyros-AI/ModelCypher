# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from modelcypher.core.domain.safety.security_event import (
    SecurityEvent,
    SecurityEventType,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_event_importable(self) -> None:
        assert SecurityEvent is not None

    def test_security_event_type_importable(self) -> None:
        assert SecurityEventType is not None


# ---------------------------------------------------------------------------
# SecurityEventType enum
# ---------------------------------------------------------------------------


class TestSecurityEventType:
    def test_all_values(self) -> None:
        assert SecurityEventType.ADAPTER_EVALUATED.value == "adapterEvaluated"
        assert SecurityEventType.ACTIVATION_BLOCKED.value == "activationBlocked"
        assert SecurityEventType.RISK_OVERRIDE_RECORDED.value == "riskOverrideRecorded"
        assert SecurityEventType.CANARY_TRIPPED.value == "canaryTripped"

    def test_is_str_enum(self) -> None:
        assert isinstance(SecurityEventType.ADAPTER_EVALUATED, str)


# ---------------------------------------------------------------------------
# Direct instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_defaults(self) -> None:
        event = SecurityEvent()
        assert event.event_id is None
        assert event.severity_score is None
        assert event.source is None
        assert event.message is None
        assert event.metadata is None
        assert event.event_type is None
        assert event.adapter_id is None
        assert event.status is None
        assert event.tier is None
        assert event.reason is None
        assert event.probe is None
        # timestamp should be auto-populated
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None

    def test_timestamp_is_utc(self) -> None:
        before = datetime.now(timezone.utc)
        event = SecurityEvent()
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_direct_construction_with_generic_fields(self) -> None:
        event = SecurityEvent(
            event_id="evt-001",
            severity_score=0.8,
            source="capability_guard",
            message="Undeclared file access",
            metadata={"path": "/etc/passwd"},
        )
        assert event.event_id == "evt-001"
        assert event.severity_score == 0.8
        assert event.source == "capability_guard"
        assert event.message == "Undeclared file access"
        assert event.metadata == {"path": "/etc/passwd"}

    def test_frozen(self) -> None:
        event = SecurityEvent()
        try:
            event.status = "changed"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    def _make_adapter_id(self) -> UUID:
        return uuid4()

    def test_adapter_evaluated(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="approved", tier="standard"
        )
        assert event.event_type == SecurityEventType.ADAPTER_EVALUATED
        assert event.adapter_id == aid
        assert event.status == "approved"
        assert event.tier == "standard"
        assert event.reason is None
        assert event.probe is None

    def test_activation_blocked(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.activation_blocked(
            adapter_id=aid, reason="spectral bound exceeded"
        )
        assert event.event_type == SecurityEventType.ACTIVATION_BLOCKED
        assert event.adapter_id == aid
        assert event.reason == "spectral bound exceeded"
        assert event.status is None
        assert event.tier is None

    def test_risk_override_recorded(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.risk_override_recorded(
            adapter_id=aid, reason="admin override"
        )
        assert event.event_type == SecurityEventType.RISK_OVERRIDE_RECORDED
        assert event.adapter_id == aid
        assert event.reason == "admin override"

    def test_canary_tripped(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.canary_tripped(
            adapter_id=aid, probe="hidden_prompt_extraction"
        )
        assert event.event_type == SecurityEventType.CANARY_TRIPPED
        assert event.adapter_id == aid
        assert event.probe == "hidden_prompt_extraction"
        assert event.reason is None

    def test_factory_methods_set_timestamp(self) -> None:
        aid = self._make_adapter_id()
        before = datetime.now(timezone.utc)
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="ok", tier="low"
        )
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


class TestToDict:
    def _make_adapter_id(self) -> UUID:
        return uuid4()

    def test_adapter_evaluated_to_dict(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="approved", tier="standard"
        )
        d = event.to_dict()
        assert d["eventType"] == "adapterEvaluated"
        assert d["adapterId"] == str(aid)
        assert d["status"] == "approved"
        assert d["tier"] == "standard"
        assert "reason" not in d
        assert "probe" not in d

    def test_activation_blocked_to_dict(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.activation_blocked(
            adapter_id=aid, reason="out of bounds"
        )
        d = event.to_dict()
        assert d["eventType"] == "activationBlocked"
        assert d["reason"] == "out of bounds"
        assert "status" not in d
        assert "tier" not in d

    def test_canary_tripped_to_dict(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.canary_tripped(adapter_id=aid, probe="test_probe")
        d = event.to_dict()
        assert d["eventType"] == "canaryTripped"
        assert d["probe"] == "test_probe"

    def test_timestamp_is_iso_format(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="ok", tier="low"
        )
        d = event.to_dict()
        ts = d["timestamp"]
        assert isinstance(ts, str)
        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_uuid_serialized_as_string(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="ok", tier="low"
        )
        d = event.to_dict()
        # adapterId should be a string, not UUID object
        assert isinstance(d["adapterId"], str)
        # Should be parseable back to UUID
        assert UUID(d["adapterId"]) == aid

    def test_to_dict_contains_all_set_fields(self) -> None:
        aid = self._make_adapter_id()
        event = SecurityEvent.risk_override_recorded(
            adapter_id=aid, reason="manual approval"
        )
        d = event.to_dict()
        assert "timestamp" in d
        assert "eventType" in d
        assert "adapterId" in d
        assert "reason" in d


# ---------------------------------------------------------------------------
# emit() does not crash
# ---------------------------------------------------------------------------


class TestEmit:
    def test_emit_does_not_raise(self) -> None:
        aid = uuid4()
        event = SecurityEvent.adapter_evaluated(
            adapter_id=aid, status="ok", tier="low"
        )
        # Should not raise
        event.emit()

    def test_emit_activation_blocked(self) -> None:
        aid = uuid4()
        event = SecurityEvent.activation_blocked(
            adapter_id=aid, reason="blocked"
        )
        event.emit()

    def test_emit_canary_tripped(self) -> None:
        aid = uuid4()
        event = SecurityEvent.canary_tripped(adapter_id=aid, probe="probe_1")
        event.emit()

    def test_emit_risk_override(self) -> None:
        aid = uuid4()
        event = SecurityEvent.risk_override_recorded(
            adapter_id=aid, reason="override"
        )
        event.emit()
