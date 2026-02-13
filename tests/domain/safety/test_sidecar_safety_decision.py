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
from uuid import UUID

import pytest

from modelcypher.core.domain.safety.sidecar.sidecar_safety_decision import (
    InterventionKind,
    SidecarDivergenceSample,
    SidecarSafetyIntervention,
    SidecarSafetyMode,
    SidecarSafetyTelemetry,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_sidecar_safety_mode_importable(self) -> None:
        assert SidecarSafetyMode is not None

    def test_intervention_kind_importable(self) -> None:
        assert InterventionKind is not None

    def test_sidecar_safety_intervention_importable(self) -> None:
        assert SidecarSafetyIntervention is not None

    def test_sidecar_divergence_sample_importable(self) -> None:
        assert SidecarDivergenceSample is not None

    def test_sidecar_safety_telemetry_importable(self) -> None:
        assert SidecarSafetyTelemetry is not None


# ---------------------------------------------------------------------------
# SidecarSafetyMode
# ---------------------------------------------------------------------------


class TestSidecarSafetyMode:
    def test_enum_values(self) -> None:
        assert SidecarSafetyMode.NORMAL.value == "normal"
        assert SidecarSafetyMode.CAUTION.value == "caution"
        assert SidecarSafetyMode.INTERVENTION.value == "intervention"

    def test_severity_rank_values(self) -> None:
        assert SidecarSafetyMode.NORMAL.severity_rank == 0
        assert SidecarSafetyMode.CAUTION.severity_rank == 1
        assert SidecarSafetyMode.INTERVENTION.severity_rank == 2

    def test_ordering_lt(self) -> None:
        assert SidecarSafetyMode.NORMAL < SidecarSafetyMode.CAUTION
        assert SidecarSafetyMode.CAUTION < SidecarSafetyMode.INTERVENTION
        assert not (SidecarSafetyMode.INTERVENTION < SidecarSafetyMode.NORMAL)

    def test_ordering_le(self) -> None:
        assert SidecarSafetyMode.NORMAL <= SidecarSafetyMode.NORMAL
        assert SidecarSafetyMode.NORMAL <= SidecarSafetyMode.CAUTION
        assert not (SidecarSafetyMode.INTERVENTION <= SidecarSafetyMode.CAUTION)

    def test_ordering_gt(self) -> None:
        assert SidecarSafetyMode.INTERVENTION > SidecarSafetyMode.CAUTION
        assert SidecarSafetyMode.CAUTION > SidecarSafetyMode.NORMAL
        assert not (SidecarSafetyMode.NORMAL > SidecarSafetyMode.CAUTION)

    def test_ordering_ge(self) -> None:
        assert SidecarSafetyMode.INTERVENTION >= SidecarSafetyMode.INTERVENTION
        assert SidecarSafetyMode.INTERVENTION >= SidecarSafetyMode.NORMAL
        assert not (SidecarSafetyMode.NORMAL >= SidecarSafetyMode.CAUTION)

    def test_comparison_with_non_mode_returns_not_implemented(self) -> None:
        assert SidecarSafetyMode.NORMAL.__lt__("other") is NotImplemented
        assert SidecarSafetyMode.NORMAL.__le__("other") is NotImplemented
        assert SidecarSafetyMode.NORMAL.__gt__("other") is NotImplemented
        assert SidecarSafetyMode.NORMAL.__ge__("other") is NotImplemented


# ---------------------------------------------------------------------------
# InterventionKind
# ---------------------------------------------------------------------------


class TestInterventionKind:
    def test_enum_values(self) -> None:
        assert InterventionKind.STABILIZER_TAKEOVER.value == "stabilizerTakeover"
        assert InterventionKind.HALT.value == "halt"


# ---------------------------------------------------------------------------
# SidecarSafetyIntervention
# ---------------------------------------------------------------------------


class TestSidecarSafetyIntervention:
    def test_instantiation(self) -> None:
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.HALT,
            token_index=42,
            reason="horror threshold crossed",
        )
        assert intervention.kind == InterventionKind.HALT
        assert intervention.token_index == 42
        assert intervention.reason == "horror threshold crossed"
        assert isinstance(intervention.timestamp, datetime)

    def test_frozen(self) -> None:
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.HALT,
            token_index=0,
            reason="test",
        )
        with pytest.raises(AttributeError):
            intervention.token_index = 10  # type: ignore[misc]

    def test_to_dict(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.STABILIZER_TAKEOVER,
            token_index=7,
            reason="sentinel divergence",
            timestamp=ts,
        )
        d = intervention.to_dict()
        assert d["kind"] == "stabilizerTakeover"
        assert d["token_index"] == 7
        assert d["reason"] == "sentinel divergence"
        assert d["timestamp"] == ts.isoformat()

    def test_from_dict(self) -> None:
        data = {
            "kind": "halt",
            "token_index": 99,
            "reason": "manual halt",
            "timestamp": "2025-06-15T12:00:00+00:00",
        }
        intervention = SidecarSafetyIntervention.from_dict(data)
        assert intervention.kind == InterventionKind.HALT
        assert intervention.token_index == 99
        assert intervention.reason == "manual halt"
        assert intervention.timestamp == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_from_dict_without_timestamp(self) -> None:
        data = {
            "kind": "halt",
            "token_index": 0,
            "reason": "test",
        }
        intervention = SidecarSafetyIntervention.from_dict(data)
        assert isinstance(intervention.timestamp, datetime)

    def test_round_trip(self) -> None:
        original = SidecarSafetyIntervention(
            kind=InterventionKind.STABILIZER_TAKEOVER,
            token_index=15,
            reason="kl divergence spike",
            timestamp=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        restored = SidecarSafetyIntervention.from_dict(original.to_dict())
        assert restored.kind == original.kind
        assert restored.token_index == original.token_index
        assert restored.reason == original.reason
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# SidecarDivergenceSample
# ---------------------------------------------------------------------------


class TestSidecarDivergenceSample:
    def test_default_instantiation(self) -> None:
        sample = SidecarDivergenceSample()
        assert isinstance(sample.id, UUID)
        assert sample.token_index == 0
        assert sample.kl_to_sentinel is None
        assert sample.kl_to_horror is None
        assert isinstance(sample.timestamp, datetime)

    def test_instantiation_with_data(self) -> None:
        sample = SidecarDivergenceSample(
            token_index=5,
            kl_to_sentinel=0.3,
            kl_to_horror=1.2,
        )
        assert sample.token_index == 5
        assert sample.kl_to_sentinel == 0.3
        assert sample.kl_to_horror == 1.2

    def test_to_dict(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        sample_id = UUID("12345678-1234-5678-1234-567812345678")
        sample = SidecarDivergenceSample(
            id=sample_id,
            token_index=3,
            kl_to_sentinel=0.5,
            kl_to_horror=None,
            timestamp=ts,
        )
        d = sample.to_dict()
        assert d["id"] == str(sample_id)
        assert d["token_index"] == 3
        assert d["kl_to_sentinel"] == 0.5
        assert d["kl_to_horror"] is None
        assert d["timestamp"] == ts.isoformat()

    def test_from_dict(self) -> None:
        sample_id = "12345678-1234-5678-1234-567812345678"
        data = {
            "id": sample_id,
            "token_index": 10,
            "kl_to_sentinel": 0.8,
            "kl_to_horror": 2.1,
            "timestamp": "2025-06-15T12:00:00+00:00",
        }
        sample = SidecarDivergenceSample.from_dict(data)
        assert sample.id == UUID(sample_id)
        assert sample.token_index == 10
        assert sample.kl_to_sentinel == 0.8
        assert sample.kl_to_horror == 2.1
        assert sample.timestamp == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_round_trip(self) -> None:
        original = SidecarDivergenceSample(
            token_index=7,
            kl_to_sentinel=0.4,
            kl_to_horror=1.5,
            timestamp=datetime(2025, 3, 1, 8, 30, 0, tzinfo=timezone.utc),
        )
        restored = SidecarDivergenceSample.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.token_index == original.token_index
        assert restored.kl_to_sentinel == original.kl_to_sentinel
        assert restored.kl_to_horror == original.kl_to_horror
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# SidecarSafetyTelemetry
# ---------------------------------------------------------------------------


class TestSidecarSafetyTelemetry:
    def test_instantiation_without_intervention(self) -> None:
        telemetry = SidecarSafetyTelemetry(
            total_tokens_observed=100,
            min_horror_kl=0.5,
            min_sentinel_kl=0.8,
            max_mode_reached=SidecarSafetyMode.CAUTION,
            intervention=None,
        )
        assert telemetry.total_tokens_observed == 100
        assert telemetry.min_horror_kl == 0.5
        assert telemetry.min_sentinel_kl == 0.8
        assert telemetry.max_mode_reached == SidecarSafetyMode.CAUTION
        assert telemetry.intervention is None

    def test_instantiation_with_intervention(self) -> None:
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.HALT,
            token_index=50,
            reason="too close",
        )
        telemetry = SidecarSafetyTelemetry(
            total_tokens_observed=50,
            min_horror_kl=0.1,
            min_sentinel_kl=None,
            max_mode_reached=SidecarSafetyMode.INTERVENTION,
            intervention=intervention,
        )
        assert telemetry.intervention is not None
        assert telemetry.intervention.kind == InterventionKind.HALT

    def test_to_dict_without_intervention(self) -> None:
        telemetry = SidecarSafetyTelemetry(
            total_tokens_observed=200,
            min_horror_kl=None,
            min_sentinel_kl=1.5,
            max_mode_reached=SidecarSafetyMode.NORMAL,
            intervention=None,
        )
        d = telemetry.to_dict()
        assert d["total_tokens_observed"] == 200
        assert d["min_horror_kl"] is None
        assert d["min_sentinel_kl"] == 1.5
        assert d["max_mode_reached"] == "normal"
        assert d["intervention"] is None

    def test_to_dict_with_intervention(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.STABILIZER_TAKEOVER,
            token_index=25,
            reason="sentinel alert",
            timestamp=ts,
        )
        telemetry = SidecarSafetyTelemetry(
            total_tokens_observed=25,
            min_horror_kl=0.3,
            min_sentinel_kl=0.2,
            max_mode_reached=SidecarSafetyMode.INTERVENTION,
            intervention=intervention,
        )
        d = telemetry.to_dict()
        assert d["intervention"] is not None
        assert d["intervention"]["kind"] == "stabilizerTakeover"
        assert d["intervention"]["token_index"] == 25

    def test_from_dict_without_intervention(self) -> None:
        data = {
            "total_tokens_observed": 300,
            "min_horror_kl": 0.9,
            "min_sentinel_kl": None,
            "max_mode_reached": "caution",
            "intervention": None,
        }
        telemetry = SidecarSafetyTelemetry.from_dict(data)
        assert telemetry.total_tokens_observed == 300
        assert telemetry.min_horror_kl == 0.9
        assert telemetry.min_sentinel_kl is None
        assert telemetry.max_mode_reached == SidecarSafetyMode.CAUTION
        assert telemetry.intervention is None

    def test_from_dict_with_intervention(self) -> None:
        data = {
            "total_tokens_observed": 50,
            "min_horror_kl": 0.1,
            "min_sentinel_kl": 0.05,
            "max_mode_reached": "intervention",
            "intervention": {
                "kind": "halt",
                "token_index": 50,
                "reason": "emergency stop",
                "timestamp": "2025-06-15T12:00:00+00:00",
            },
        }
        telemetry = SidecarSafetyTelemetry.from_dict(data)
        assert telemetry.intervention is not None
        assert telemetry.intervention.kind == InterventionKind.HALT
        assert telemetry.intervention.token_index == 50

    def test_round_trip_without_intervention(self) -> None:
        original = SidecarSafetyTelemetry(
            total_tokens_observed=150,
            min_horror_kl=0.7,
            min_sentinel_kl=1.0,
            max_mode_reached=SidecarSafetyMode.CAUTION,
            intervention=None,
        )
        restored = SidecarSafetyTelemetry.from_dict(original.to_dict())
        assert restored.total_tokens_observed == original.total_tokens_observed
        assert restored.min_horror_kl == original.min_horror_kl
        assert restored.min_sentinel_kl == original.min_sentinel_kl
        assert restored.max_mode_reached == original.max_mode_reached
        assert restored.intervention is None

    def test_round_trip_with_intervention(self) -> None:
        ts = datetime(2025, 8, 20, 16, 45, 0, tzinfo=timezone.utc)
        intervention = SidecarSafetyIntervention(
            kind=InterventionKind.STABILIZER_TAKEOVER,
            token_index=33,
            reason="sentinel kl collapse",
            timestamp=ts,
        )
        original = SidecarSafetyTelemetry(
            total_tokens_observed=33,
            min_horror_kl=0.05,
            min_sentinel_kl=0.02,
            max_mode_reached=SidecarSafetyMode.INTERVENTION,
            intervention=intervention,
        )
        restored = SidecarSafetyTelemetry.from_dict(original.to_dict())
        assert restored.total_tokens_observed == original.total_tokens_observed
        assert restored.max_mode_reached == original.max_mode_reached
        assert restored.intervention is not None
        assert restored.intervention.kind == original.intervention.kind
        assert restored.intervention.token_index == original.intervention.token_index
        assert restored.intervention.reason == original.intervention.reason
        assert restored.intervention.timestamp == original.intervention.timestamp
