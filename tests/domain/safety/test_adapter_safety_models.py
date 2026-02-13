"""Tests for adapter safety model types.

Covers enums, dataclass instantiation, computed properties, and frozen
immutability for all types in adapter_safety_models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyContext,
    AdapterSafetyScorecard,
    AdapterSafetyStatus,
    AdapterSafetyTier,
    AdapterSafetyTrigger,
    RiskOverride,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestAdapterSafetyStatus:
    def test_all_values_accessible(self) -> None:
        assert AdapterSafetyStatus.ALLOWED == "allowed"
        assert AdapterSafetyStatus.QUARANTINED == "quarantined"
        assert AdapterSafetyStatus.BLOCKED == "blocked"
        assert AdapterSafetyStatus.PENDING == "pending"

    def test_is_str_enum(self) -> None:
        assert isinstance(AdapterSafetyStatus.ALLOWED, str)

    def test_member_count(self) -> None:
        assert len(AdapterSafetyStatus) == 4


class TestAdapterSafetyTier:
    def test_all_values_accessible(self) -> None:
        assert AdapterSafetyTier.QUICK == "quick"
        assert AdapterSafetyTier.STANDARD == "standard"
        assert AdapterSafetyTier.FULL == "full"

    def test_is_str_enum(self) -> None:
        assert isinstance(AdapterSafetyTier.QUICK, str)

    def test_member_count(self) -> None:
        assert len(AdapterSafetyTier) == 3


class TestAdapterSafetyTrigger:
    def test_all_values_accessible(self) -> None:
        assert AdapterSafetyTrigger.IMPORT_ADAPTER == "importAdapter"
        assert AdapterSafetyTrigger.POST_TRAINING == "postTraining"
        assert AdapterSafetyTrigger.MERGE == "merge"
        assert AdapterSafetyTrigger.MANUAL_RESCAN == "manualRescan"
        assert AdapterSafetyTrigger.ACTIVATION_CHECK == "activationCheck"
        assert AdapterSafetyTrigger.RUNTIME_CANARY == "runtimeCanary"

    def test_is_str_enum(self) -> None:
        assert isinstance(AdapterSafetyTrigger.MERGE, str)

    def test_member_count(self) -> None:
        assert len(AdapterSafetyTrigger) == 6


# ---------------------------------------------------------------------------
# RiskOverride tests
# ---------------------------------------------------------------------------


class TestRiskOverride:
    @pytest.fixture()
    def override(self) -> RiskOverride:
        return RiskOverride(
            adapter_id="adapter-001",
            reason="Reviewed and approved by human",
            probe_version_at_override="1.0.0",
            delta_score_at_override=0.05,
            adapter_hash_at_override="abc123",
        )

    def test_instantiation(self, override: RiskOverride) -> None:
        assert override.adapter_id == "adapter-001"
        assert override.reason == "Reviewed and approved by human"
        assert override.probe_version_at_override == "1.0.0"
        assert override.delta_score_at_override == 0.05
        assert override.adapter_hash_at_override == "abc123"

    def test_timestamp_auto_set(self, override: RiskOverride) -> None:
        assert isinstance(override.timestamp, datetime)
        assert override.timestamp.tzinfo is not None

    def test_frozen(self, override: RiskOverride) -> None:
        with pytest.raises(AttributeError):
            override.adapter_id = "changed"  # type: ignore[misc]

    def test_is_valid_both_match(self, override: RiskOverride) -> None:
        assert override.is_valid(adapter_hash="abc123", probe_version="1.0.0") is True

    def test_is_valid_hash_differs(self, override: RiskOverride) -> None:
        assert override.is_valid(adapter_hash="different", probe_version="1.0.0") is False

    def test_is_valid_version_differs(self, override: RiskOverride) -> None:
        assert override.is_valid(adapter_hash="abc123", probe_version="2.0.0") is False

    def test_is_valid_both_differ(self, override: RiskOverride) -> None:
        assert override.is_valid(adapter_hash="different", probe_version="2.0.0") is False


# ---------------------------------------------------------------------------
# AdapterSafetyScorecard tests
# ---------------------------------------------------------------------------


class TestAdapterSafetyScorecard:
    @pytest.fixture()
    def scorecard(self) -> AdapterSafetyScorecard:
        return AdapterSafetyScorecard(
            adapter_id=uuid4(),
            status=AdapterSafetyStatus.ALLOWED,
            evaluation_tier=AdapterSafetyTier.STANDARD,
            evaluation_version="1.0.0",
            adapter_hash="hash123",
        )

    def test_instantiation_required_fields(self, scorecard: AdapterSafetyScorecard) -> None:
        assert isinstance(scorecard.adapter_id, type(uuid4()))
        assert scorecard.status == AdapterSafetyStatus.ALLOWED
        assert scorecard.evaluation_tier == AdapterSafetyTier.STANDARD
        assert scorecard.evaluation_version == "1.0.0"
        assert scorecard.adapter_hash == "hash123"

    def test_default_optional_fields(self, scorecard: AdapterSafetyScorecard) -> None:
        assert scorecard.reason is None
        assert scorecard.warnings == ()
        assert scorecard.probe_version is None
        assert scorecard.delta_score is None
        assert scorecard.projection_status is None
        assert scorecard.risk_override is None

    def test_evaluated_at_auto_set(self, scorecard: AdapterSafetyScorecard) -> None:
        assert isinstance(scorecard.evaluated_at, datetime)

    def test_frozen(self, scorecard: AdapterSafetyScorecard) -> None:
        with pytest.raises(AttributeError):
            scorecard.status = AdapterSafetyStatus.BLOCKED  # type: ignore[misc]

    def test_with_all_optional_fields(self) -> None:
        override = RiskOverride(
            adapter_id="a1",
            reason="ok",
            probe_version_at_override="1.0",
            delta_score_at_override=0.1,
            adapter_hash_at_override="h1",
        )
        scorecard = AdapterSafetyScorecard(
            adapter_id=uuid4(),
            status=AdapterSafetyStatus.QUARANTINED,
            evaluation_tier=AdapterSafetyTier.FULL,
            evaluation_version="2.0.0",
            adapter_hash="hash456",
            reason="High delta score",
            warnings=("warn1", "warn2"),
            probe_version="1.0.0",
            delta_score=0.95,
            projection_status="projected",
            risk_override=override,
        )
        assert scorecard.reason == "High delta score"
        assert scorecard.warnings == ("warn1", "warn2")
        assert scorecard.probe_version == "1.0.0"
        assert scorecard.delta_score == 0.95
        assert scorecard.projection_status == "projected"
        assert scorecard.risk_override is override


# ---------------------------------------------------------------------------
# AdapterSafetyContext tests
# ---------------------------------------------------------------------------


class TestAdapterSafetyContext:
    def test_instantiation_required_fields(self) -> None:
        ctx = AdapterSafetyContext(
            trigger=AdapterSafetyTrigger.IMPORT_ADAPTER,
            tier=AdapterSafetyTier.QUICK,
        )
        assert ctx.trigger == AdapterSafetyTrigger.IMPORT_ADAPTER
        assert ctx.tier == AdapterSafetyTier.QUICK

    def test_requested_at_auto_set(self) -> None:
        ctx = AdapterSafetyContext(
            trigger=AdapterSafetyTrigger.MERGE,
            tier=AdapterSafetyTier.STANDARD,
        )
        assert isinstance(ctx.requested_at, datetime)
        assert ctx.requested_at.tzinfo is not None

    def test_probe_version_hint_default_none(self) -> None:
        ctx = AdapterSafetyContext(
            trigger=AdapterSafetyTrigger.RUNTIME_CANARY,
            tier=AdapterSafetyTier.FULL,
        )
        assert ctx.probe_version_hint is None

    def test_probe_version_hint_set(self) -> None:
        ctx = AdapterSafetyContext(
            trigger=AdapterSafetyTrigger.POST_TRAINING,
            tier=AdapterSafetyTier.STANDARD,
            probe_version_hint="3.0.0",
        )
        assert ctx.probe_version_hint == "3.0.0"

    def test_frozen(self) -> None:
        ctx = AdapterSafetyContext(
            trigger=AdapterSafetyTrigger.MANUAL_RESCAN,
            tier=AdapterSafetyTier.QUICK,
        )
        with pytest.raises(AttributeError):
            ctx.trigger = AdapterSafetyTrigger.MERGE  # type: ignore[misc]
