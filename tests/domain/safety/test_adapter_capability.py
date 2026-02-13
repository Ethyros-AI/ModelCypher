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

import pytest

from modelcypher.core.domain.safety.adapter_capability import (
    CapabilityCheckOutcome,
    CapabilityCheckResult,
    CapabilityViolation,
    EnforcementMode,
    ResourceCapability,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_types_importable(self) -> None:
        for cls in (
            ResourceCapability,
            CapabilityCheckResult,
            EnforcementMode,
            CapabilityViolation,
            CapabilityCheckOutcome,
        ):
            assert cls is not None


# ---------------------------------------------------------------------------
# ResourceCapability enum
# ---------------------------------------------------------------------------


class TestResourceCapability:
    def test_all_values(self) -> None:
        assert ResourceCapability.FILE_READ.value == "file_read"
        assert ResourceCapability.FILE_WRITE.value == "file_write"
        assert ResourceCapability.NETWORK_HTTP.value == "network_http"
        assert ResourceCapability.CODE_EXEC.value == "code_exec"
        assert ResourceCapability.NONE.value == "none"

    def test_is_str_enum(self) -> None:
        assert isinstance(ResourceCapability.FILE_READ, str)

    # --- from_capability_string ---

    def test_from_capability_string_valid(self) -> None:
        assert (
            ResourceCapability.from_capability_string("resource:file_read")
            == ResourceCapability.FILE_READ
        )
        assert (
            ResourceCapability.from_capability_string("resource:file_write")
            == ResourceCapability.FILE_WRITE
        )
        assert (
            ResourceCapability.from_capability_string("resource:network_http")
            == ResourceCapability.NETWORK_HTTP
        )
        assert (
            ResourceCapability.from_capability_string("resource:code_exec")
            == ResourceCapability.CODE_EXEC
        )
        assert (
            ResourceCapability.from_capability_string("resource:none")
            == ResourceCapability.NONE
        )

    def test_from_capability_string_invalid_prefix(self) -> None:
        assert ResourceCapability.from_capability_string("file_read") is None
        assert ResourceCapability.from_capability_string("capability:file_read") is None
        assert ResourceCapability.from_capability_string("") is None

    def test_from_capability_string_unknown_value(self) -> None:
        assert ResourceCapability.from_capability_string("resource:unknown") is None
        assert ResourceCapability.from_capability_string("resource:") is None

    # --- capability_string ---

    def test_capability_string_format(self) -> None:
        for cap in ResourceCapability:
            cs = cap.capability_string
            assert cs.startswith("resource:")
            assert cs == f"resource:{cap.value}"

    def test_capability_string_roundtrip(self) -> None:
        for cap in ResourceCapability:
            restored = ResourceCapability.from_capability_string(cap.capability_string)
            assert restored == cap

    # --- display_name ---

    def test_display_name_returns_strings(self) -> None:
        for cap in ResourceCapability:
            name = cap.display_name
            assert isinstance(name, str)
            assert len(name) > 0

    def test_display_name_specific_values(self) -> None:
        assert ResourceCapability.FILE_READ.display_name == "File Read"
        assert ResourceCapability.NONE.display_name == "None"

    # --- description ---

    def test_description_returns_strings(self) -> None:
        for cap in ResourceCapability:
            desc = cap.description
            assert isinstance(desc, str)
            assert len(desc) > 0


# ---------------------------------------------------------------------------
# CapabilityCheckResult enum
# ---------------------------------------------------------------------------


class TestCapabilityCheckResult:
    def test_all_values(self) -> None:
        assert CapabilityCheckResult.ALLOWED.value == "allowed"
        assert CapabilityCheckResult.DENIED.value == "denied"
        assert CapabilityCheckResult.MONITOR_ONLY.value == "monitorOnly"

    def test_is_allowed_for_allowed(self) -> None:
        assert CapabilityCheckResult.ALLOWED.is_allowed is True

    def test_is_allowed_for_denied(self) -> None:
        assert CapabilityCheckResult.DENIED.is_allowed is False

    def test_is_allowed_for_monitor_only(self) -> None:
        # MONITOR_ONLY allows access but logs it
        assert CapabilityCheckResult.MONITOR_ONLY.is_allowed is True


# ---------------------------------------------------------------------------
# EnforcementMode enum
# ---------------------------------------------------------------------------


class TestEnforcementMode:
    def test_all_values(self) -> None:
        assert EnforcementMode.ENFORCE.value == "enforce"
        assert EnforcementMode.MONITOR.value == "monitor"
        assert EnforcementMode.DISABLED.value == "disabled"

    def test_is_str_enum(self) -> None:
        assert isinstance(EnforcementMode.ENFORCE, str)


# ---------------------------------------------------------------------------
# CapabilityViolation dataclass
# ---------------------------------------------------------------------------


class TestCapabilityViolation:
    def _make_violation(self, **kwargs) -> CapabilityViolation:
        defaults = {
            "adapter_id": uuid4(),
            "adapter_name": "test-adapter",
            "requested_capability": ResourceCapability.FILE_WRITE,
            "declared_capabilities": frozenset({ResourceCapability.FILE_READ}),
        }
        defaults.update(kwargs)
        return CapabilityViolation(**defaults)

    def test_instantiation_with_required_fields(self) -> None:
        v = self._make_violation()
        assert isinstance(v.adapter_id, UUID)
        assert v.adapter_name == "test-adapter"
        assert v.requested_capability == ResourceCapability.FILE_WRITE
        assert v.declared_capabilities == frozenset({ResourceCapability.FILE_READ})
        assert v.resource_identifier is None
        assert isinstance(v.timestamp, datetime)
        assert isinstance(v.id, UUID)

    def test_instantiation_with_resource_identifier(self) -> None:
        v = self._make_violation(resource_identifier="/etc/shadow")
        assert v.resource_identifier == "/etc/shadow"

    def test_timestamp_auto_populated(self) -> None:
        before = datetime.now(timezone.utc)
        v = self._make_violation()
        after = datetime.now(timezone.utc)
        assert before <= v.timestamp <= after

    def test_id_auto_populated_unique(self) -> None:
        v1 = self._make_violation()
        v2 = self._make_violation()
        assert v1.id != v2.id

    def test_frozen(self) -> None:
        v = self._make_violation()
        with pytest.raises(AttributeError):
            v.adapter_name = "changed"  # type: ignore[misc]

    # --- summary property ---

    def test_summary_contains_adapter_name(self) -> None:
        v = self._make_violation(adapter_name="evil-adapter")
        assert "evil-adapter" in v.summary

    def test_summary_contains_requested_capability_display_name(self) -> None:
        v = self._make_violation(requested_capability=ResourceCapability.CODE_EXEC)
        assert ResourceCapability.CODE_EXEC.display_name in v.summary

    def test_summary_contains_declared_capabilities(self) -> None:
        v = self._make_violation(
            declared_capabilities=frozenset({ResourceCapability.FILE_READ})
        )
        assert ResourceCapability.FILE_READ.display_name in v.summary

    def test_summary_with_empty_declared_capabilities(self) -> None:
        v = self._make_violation(declared_capabilities=frozenset())
        assert "none" in v.summary

    def test_summary_is_human_readable(self) -> None:
        v = self._make_violation(
            adapter_name="my-adapter",
            requested_capability=ResourceCapability.NETWORK_HTTP,
            declared_capabilities=frozenset(
                {ResourceCapability.FILE_READ, ResourceCapability.NONE}
            ),
        )
        summary = v.summary
        assert isinstance(summary, str)
        assert "my-adapter" in summary
        assert "Network (HTTP)" in summary


# ---------------------------------------------------------------------------
# CapabilityCheckOutcome dataclass
# ---------------------------------------------------------------------------


class TestCapabilityCheckOutcome:
    def test_instantiation_allowed_no_violation(self) -> None:
        outcome = CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)
        assert outcome.result == CapabilityCheckResult.ALLOWED
        assert outcome.violation is None

    def test_instantiation_denied_with_violation(self) -> None:
        violation = CapabilityViolation(
            adapter_id=uuid4(),
            adapter_name="test",
            requested_capability=ResourceCapability.FILE_WRITE,
            declared_capabilities=frozenset(),
        )
        outcome = CapabilityCheckOutcome(
            result=CapabilityCheckResult.DENIED,
            violation=violation,
        )
        assert outcome.result == CapabilityCheckResult.DENIED
        assert outcome.violation is violation

    def test_is_allowed_delegates_to_result(self) -> None:
        allowed = CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)
        assert allowed.is_allowed is True

        denied = CapabilityCheckOutcome(result=CapabilityCheckResult.DENIED)
        assert denied.is_allowed is False

        monitor = CapabilityCheckOutcome(result=CapabilityCheckResult.MONITOR_ONLY)
        assert monitor.is_allowed is True

    def test_frozen(self) -> None:
        outcome = CapabilityCheckOutcome(result=CapabilityCheckResult.ALLOWED)
        with pytest.raises(AttributeError):
            outcome.result = CapabilityCheckResult.DENIED  # type: ignore[misc]
