"""Tests for privacy-preserving safety audit log.

Covers initial state, event logging, category tracking, reset behavior,
defensive copy semantics, and session summary logging.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.safety_audit_log import SafetyAuditLog
from modelcypher.core.domain.safety.safety_models import SafetyCategory

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestSafetyAuditLogInitialState:
    def test_total_events_zero(self) -> None:
        log = SafetyAuditLog()
        assert log.total_events == 0

    def test_category_breakdown_empty(self) -> None:
        log = SafetyAuditLog()
        assert log.category_breakdown == {}

    def test_session_id_is_8_chars(self) -> None:
        log = SafetyAuditLog()
        assert len(log.session_id) == 8

    def test_session_id_is_hex(self) -> None:
        log = SafetyAuditLog()
        int(log.session_id, 16)  # raises ValueError if not valid hex

    def test_distinct_session_ids(self) -> None:
        log_a = SafetyAuditLog()
        log_b = SafetyAuditLog()
        assert log_a.session_id != log_b.session_id


# ---------------------------------------------------------------------------
# log_filter_event
# ---------------------------------------------------------------------------


class TestLogFilterEvent:
    def test_increments_total_events(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-001", context_length=128)
        assert log.total_events == 1

    def test_updates_category_breakdown(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-001", context_length=128)
        assert log.category_breakdown == {"toxicity": 1}

    def test_multiple_events_same_category(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.PII, "pii-001", context_length=64)
        log.log_filter_event(SafetyCategory.PII, "pii-002", context_length=96)
        assert log.total_events == 2
        assert log.category_breakdown == {"pii": 2}

    def test_multiple_events_different_categories(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-001", context_length=128)
        log.log_filter_event(SafetyCategory.HATE_SPEECH, "hate-001", context_length=256)
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-002", context_length=100)
        assert log.total_events == 3
        assert log.category_breakdown == {"toxicity": 2, "hate_speech": 1}


# ---------------------------------------------------------------------------
# log_truncation_event
# ---------------------------------------------------------------------------


class TestLogTruncationEvent:
    def test_does_not_crash(self) -> None:
        log = SafetyAuditLog()
        log.log_truncation_event(reason="too many violations", total_violations=5)

    def test_does_not_increment_total_events(self) -> None:
        log = SafetyAuditLog()
        log.log_truncation_event(reason="limit reached", total_violations=3)
        # Truncation events are logged via logger.warning, not counted as filter events
        assert log.total_events == 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_event_count(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.VIOLENCE, "vio-001", context_length=50)
        log.reset()
        assert log.total_events == 0

    def test_clears_category_breakdown(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.SEXUAL, "sex-001", context_length=80)
        log.reset()
        assert log.category_breakdown == {}

    def test_session_id_persists_after_reset(self) -> None:
        log = SafetyAuditLog()
        original_id = log.session_id
        log.reset()
        assert log.session_id == original_id

    def test_can_log_after_reset(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.PII, "pii-001", context_length=100)
        log.reset()
        log.log_filter_event(SafetyCategory.HARASSMENT, "har-001", context_length=200)
        assert log.total_events == 1
        assert log.category_breakdown == {"harassment": 1}


# ---------------------------------------------------------------------------
# category_breakdown returns a copy
# ---------------------------------------------------------------------------


class TestCategoryBreakdownCopy:
    def test_returns_copy_not_internal_dict(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-001", context_length=128)
        breakdown = log.category_breakdown
        breakdown["toxicity"] = 999
        breakdown["injected"] = 1
        # Internal state must be unaffected
        assert log.category_breakdown == {"toxicity": 1}


# ---------------------------------------------------------------------------
# log_session_summary
# ---------------------------------------------------------------------------


class TestLogSessionSummary:
    def test_does_not_crash_with_no_events(self) -> None:
        log = SafetyAuditLog()
        log.log_session_summary()

    def test_does_not_crash_with_events(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.TOXICITY, "tox-001", context_length=128)
        log.log_filter_event(SafetyCategory.PII, "pii-001", context_length=64)
        log.log_session_summary()

    def test_does_not_alter_state(self) -> None:
        log = SafetyAuditLog()
        log.log_filter_event(SafetyCategory.VIOLENCE, "vio-001", context_length=50)
        log.log_session_summary()
        assert log.total_events == 1
        assert log.category_breakdown == {"violence": 1}
