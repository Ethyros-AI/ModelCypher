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

"""Tests for CapabilityGuard and related types."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from modelcypher.core.domain.safety.adapter_capability import (
    CapabilityCheckResult,
    CapabilityGuardConfiguration,
    CapabilityViolation,
    EnforcementMode,
    ResourceCapability,
    RiskLevel,
)
from modelcypher.core.domain.safety.capability_guard import (
    AuditEventType,
    CapabilityAuditLog,
    CapabilityGuard,
)


class TestResourceCapability:
    """Tests for ResourceCapability enum."""

    def test_capability_values(self) -> None:
        """All expected capabilities exist."""
        assert ResourceCapability.FILE_READ == "file_read"
        assert ResourceCapability.FILE_WRITE == "file_write"
        assert ResourceCapability.NETWORK_HTTP == "network_http"
        assert ResourceCapability.CODE_EXEC == "code_exec"
        assert ResourceCapability.NONE == "none"

    def test_from_capability_string_valid(self) -> None:
        """from_capability_string parses valid strings."""
        assert ResourceCapability.from_capability_string("resource:file_read") == ResourceCapability.FILE_READ
        assert ResourceCapability.from_capability_string("resource:code_exec") == ResourceCapability.CODE_EXEC

    def test_from_capability_string_invalid_prefix(self) -> None:
        """from_capability_string returns None for invalid prefix."""
        assert ResourceCapability.from_capability_string("invalid:file_read") is None
        assert ResourceCapability.from_capability_string("file_read") is None

    def test_from_capability_string_invalid_value(self) -> None:
        """from_capability_string returns None for unknown values."""
        assert ResourceCapability.from_capability_string("resource:unknown") is None

    def test_capability_string_property(self) -> None:
        """capability_string returns formatted string."""
        assert ResourceCapability.FILE_READ.capability_string == "resource:file_read"
        assert ResourceCapability.NONE.capability_string == "resource:none"

    def test_risk_levels(self) -> None:
        """Risk levels are correctly assigned."""
        assert ResourceCapability.NONE.risk_level == RiskLevel.SAFE
        assert ResourceCapability.FILE_READ.risk_level == RiskLevel.MODERATE
        assert ResourceCapability.FILE_WRITE.risk_level == RiskLevel.ELEVATED
        assert ResourceCapability.CODE_EXEC.risk_level == RiskLevel.HIGH


class TestCapabilityViolation:
    """Tests for CapabilityViolation dataclass."""

    def test_violation_creation(self) -> None:
        """Violation can be created with all fields."""
        adapter_id = uuid4()
        violation = CapabilityViolation(
            adapter_id=adapter_id,
            adapter_name="test-adapter",
            requested_capability=ResourceCapability.FILE_WRITE,
            declared_capabilities=frozenset([ResourceCapability.FILE_READ]),
            resource_identifier="/path/to/file",
        )
        assert violation.adapter_id == adapter_id
        assert violation.requested_capability == ResourceCapability.FILE_WRITE
        assert ResourceCapability.FILE_READ in violation.declared_capabilities

    def test_violation_summary(self) -> None:
        """summary property produces readable string."""
        violation = CapabilityViolation(
            adapter_id=uuid4(),
            adapter_name="my-adapter",
            requested_capability=ResourceCapability.CODE_EXEC,
            declared_capabilities=frozenset([ResourceCapability.FILE_READ]),
        )
        summary = violation.summary
        assert "my-adapter" in summary
        assert "Code Execution" in summary
        assert "File Read" in summary

    def test_violation_summary_no_capabilities(self) -> None:
        """summary handles empty declared capabilities."""
        violation = CapabilityViolation(
            adapter_id=uuid4(),
            adapter_name="empty-adapter",
            requested_capability=ResourceCapability.FILE_READ,
            declared_capabilities=frozenset(),
        )
        assert "none" in violation.summary


class TestCapabilityAuditLog:
    """Tests for CapabilityAuditLog class."""

    def test_log_violation(self) -> None:
        """log_violation adds event to log."""
        log = CapabilityAuditLog()
        violation = CapabilityViolation(
            adapter_id=uuid4(),
            adapter_name="test",
            requested_capability=ResourceCapability.FILE_WRITE,
            declared_capabilities=frozenset(),
        )
        log.log_violation(violation)
        assert log.total_events == 1
        assert log.violation_count == 1

    def test_log_access_allowed(self) -> None:
        """log_access_allowed adds allowed event."""
        log = CapabilityAuditLog()
        adapter_id = uuid4()
        log.log_access_allowed(adapter_id, ResourceCapability.FILE_READ)
        assert log.total_events == 1
        assert log.violation_count == 0
        events = log.recent_events()
        assert events[0].event_type == AuditEventType.ALLOWED

    def test_max_events_trimming(self) -> None:
        """Log trims old events when over capacity."""
        log = CapabilityAuditLog(max_events=5)
        adapter_id = uuid4()
        for _ in range(10):
            log.log_access_allowed(adapter_id, ResourceCapability.FILE_READ)
        assert log.total_events == 5

    def test_recent_events_limit(self) -> None:
        """recent_events respects limit parameter."""
        log = CapabilityAuditLog()
        adapter_id = uuid4()
        for _ in range(10):
            log.log_access_allowed(adapter_id, ResourceCapability.FILE_READ)
        assert len(log.recent_events(limit=3)) == 3

    def test_reset(self) -> None:
        """reset clears all events."""
        log = CapabilityAuditLog()
        log.log_access_allowed(uuid4(), ResourceCapability.FILE_READ)
        assert log.total_events == 1
        log.reset()
        assert log.total_events == 0

    def test_hash_privacy(self) -> None:
        """_hash produces consistent, truncated hash."""
        hash1 = CapabilityAuditLog._hash("/path/to/secret")
        hash2 = CapabilityAuditLog._hash("/path/to/secret")
        hash3 = CapabilityAuditLog._hash("/different/path")
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 8  # Truncated


class TestCapabilityGuard:
    """Tests for CapabilityGuard class."""

    def test_register_and_check_allowed(self) -> None:
        """Registered adapter with declared capability is allowed."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id,
            "test-adapter",
            frozenset([ResourceCapability.FILE_READ]),
        )
        outcome = guard.check_access(adapter_id, ResourceCapability.FILE_READ)
        assert outcome.result == CapabilityCheckResult.ALLOWED
        assert outcome.violation is None

    def test_check_denied_undeclared(self) -> None:
        """Accessing undeclared capability is denied."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id,
            "test-adapter",
            frozenset([ResourceCapability.FILE_READ]),
        )
        outcome = guard.check_access(adapter_id, ResourceCapability.FILE_WRITE)
        assert outcome.result == CapabilityCheckResult.DENIED
        assert outcome.violation is not None
        assert outcome.violation.requested_capability == ResourceCapability.FILE_WRITE

    def test_check_unregistered_adapter(self) -> None:
        """Unregistered adapter is denied."""
        guard = CapabilityGuard()
        outcome = guard.check_access(uuid4(), ResourceCapability.FILE_READ)
        assert outcome.result == CapabilityCheckResult.DENIED

    def test_monitor_mode(self) -> None:
        """Monitor mode allows but logs violations."""
        config = CapabilityGuardConfiguration.monitoring()
        guard = CapabilityGuard(configuration=config)
        adapter_id = uuid4()
        guard.register_adapter(adapter_id, "test", frozenset())

        outcome = guard.check_access(adapter_id, ResourceCapability.FILE_READ)
        assert outcome.result == CapabilityCheckResult.MONITOR_ONLY
        assert outcome.violation is not None

    def test_disabled_mode(self) -> None:
        """Disabled mode always allows."""
        config = CapabilityGuardConfiguration.disabled()
        guard = CapabilityGuard(configuration=config)
        outcome = guard.check_access(uuid4(), ResourceCapability.CODE_EXEC)
        assert outcome.result == CapabilityCheckResult.ALLOWED

    def test_always_allowed_capabilities(self) -> None:
        """Always-allowed capabilities bypass checks."""
        config = CapabilityGuardConfiguration(
            always_allowed_capabilities=frozenset([ResourceCapability.FILE_READ])
        )
        guard = CapabilityGuard(configuration=config)
        # Don't register adapter, but FILE_READ should still be allowed
        outcome = guard.check_access(uuid4(), ResourceCapability.FILE_READ)
        assert outcome.result == CapabilityCheckResult.ALLOWED

    def test_unregister_adapter(self) -> None:
        """Unregistered adapter loses capabilities."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id, "test", frozenset([ResourceCapability.FILE_READ])
        )
        guard.unregister_adapter(adapter_id)
        assert guard.capabilities_for(adapter_id) is None

    def test_violation_tracking(self) -> None:
        """Violations are tracked per adapter."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(adapter_id, "test", frozenset())

        guard.check_access(adapter_id, ResourceCapability.FILE_READ)
        guard.check_access(adapter_id, ResourceCapability.FILE_WRITE)

        violations = guard.violations_for(adapter_id)
        assert len(violations) == 2
        assert guard.total_violation_count == 2

    def test_adapter_disabled_after_max_violations(self) -> None:
        """Adapter is disabled after max violations."""
        config = CapabilityGuardConfiguration(max_violations_before_disable=2)
        guard = CapabilityGuard(configuration=config)
        adapter_id = uuid4()
        guard.register_adapter(adapter_id, "test", frozenset())

        guard.check_access(adapter_id, ResourceCapability.FILE_READ)
        assert not guard.is_adapter_disabled(adapter_id)

        guard.check_access(adapter_id, ResourceCapability.FILE_WRITE)
        assert guard.is_adapter_disabled(adapter_id)

    def test_disabled_adapter_denied(self) -> None:
        """Disabled adapter is always denied."""
        config = CapabilityGuardConfiguration(max_violations_before_disable=1)
        guard = CapabilityGuard(configuration=config)
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id, "test", frozenset([ResourceCapability.FILE_READ])
        )

        # Trigger violation to disable
        guard.check_access(adapter_id, ResourceCapability.CODE_EXEC)
        assert guard.is_adapter_disabled(adapter_id)

        # Now even declared capability is denied
        outcome = guard.check_access(adapter_id, ResourceCapability.FILE_READ)
        assert outcome.result == CapabilityCheckResult.DENIED

    def test_reenable_adapter(self) -> None:
        """Disabled adapter can be re-enabled."""
        config = CapabilityGuardConfiguration(max_violations_before_disable=1)
        guard = CapabilityGuard(configuration=config)
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id, "test", frozenset([ResourceCapability.FILE_READ])
        )

        guard.check_access(adapter_id, ResourceCapability.CODE_EXEC)
        assert guard.is_adapter_disabled(adapter_id)

        guard.reenable_adapter(adapter_id)
        assert not guard.is_adapter_disabled(adapter_id)

    def test_check_access_batch(self) -> None:
        """check_access_batch checks multiple capabilities."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id,
            "test",
            frozenset([ResourceCapability.FILE_READ, ResourceCapability.FILE_WRITE]),
        )

        outcome = guard.check_access_batch(
            adapter_id,
            frozenset([ResourceCapability.FILE_READ, ResourceCapability.FILE_WRITE]),
        )
        assert outcome.result == CapabilityCheckResult.ALLOWED

    def test_check_access_batch_partial_fail(self) -> None:
        """check_access_batch fails on first violation."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(
            adapter_id, "test", frozenset([ResourceCapability.FILE_READ])
        )

        outcome = guard.check_access_batch(
            adapter_id,
            frozenset([ResourceCapability.FILE_READ, ResourceCapability.CODE_EXEC]),
        )
        assert outcome.result == CapabilityCheckResult.DENIED

    def test_configure_updates_mode(self) -> None:
        """configure() updates enforcement mode."""
        guard = CapabilityGuard()
        assert guard.enforcement_mode == EnforcementMode.ENFORCE

        guard.configure(CapabilityGuardConfiguration.monitoring())
        assert guard.enforcement_mode == EnforcementMode.MONITOR

    def test_reset(self) -> None:
        """reset() clears all state."""
        guard = CapabilityGuard()
        adapter_id = uuid4()
        guard.register_adapter(adapter_id, "test", frozenset())
        guard.check_access(adapter_id, ResourceCapability.FILE_READ)

        guard.reset()
        assert guard.capabilities_for(adapter_id) is None
        assert guard.total_violation_count == 0

    def test_all_violations(self) -> None:
        """all_violations returns violations from all adapters."""
        guard = CapabilityGuard()
        adapter1 = uuid4()
        adapter2 = uuid4()
        guard.register_adapter(adapter1, "adapter1", frozenset())
        guard.register_adapter(adapter2, "adapter2", frozenset())

        guard.check_access(adapter1, ResourceCapability.FILE_READ)
        guard.check_access(adapter2, ResourceCapability.FILE_WRITE)

        all_violations = guard.all_violations()
        assert len(all_violations) == 2
