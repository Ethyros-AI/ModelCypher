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

import modelcypher.core.domain.safety.sidecar.sidecar_safety_session as session_mod
from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ConsentGrant,
    ScenarioMode,
    SessionControlState,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_decision import (
    InterventionKind,
    SidecarDivergenceSample,
    SidecarSafetyMode,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_policy import SidecarSafetyPolicy
from modelcypher.core.domain.safety.sidecar.sidecar_safety_session import SidecarSafetySession


def _policy() -> SidecarSafetyPolicy:
    return SidecarSafetyPolicy.from_baseline(
        baseline_measurements=[0.2, 0.4, 0.6, 0.8, 1.0],
        hard_percentile=10.0,
        soft_percentile=80.0,
    )


def test_session_modes_intervention_consumption_and_telemetry(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(session_mod, "get_default_backend", lambda: b)

    session = SidecarSafetySession(policy=_policy(), stabilizer_configured=False)

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    mode_normal = session.observe(
        SidecarDivergenceSample(token_index=0, kl_to_horror=1.5),
        now=now,
    )
    mode_caution = session.observe(
        SidecarDivergenceSample(token_index=1, kl_to_horror=0.8),
        now=now,
    )
    mode_intervention = session.observe(
        SidecarDivergenceSample(token_index=2, kl_to_horror=0.1),
        now=now,
    )

    assert mode_normal == SidecarSafetyMode.NORMAL
    assert mode_caution == SidecarSafetyMode.CAUTION
    assert mode_intervention == SidecarSafetyMode.INTERVENTION

    intervention = session.consume_pending_intervention()
    assert intervention is not None
    assert intervention.kind == InterventionKind.HALT
    assert intervention.token_index == 2

    assert session.consume_pending_intervention() is None

    telemetry = session.telemetry_snapshot()
    assert telemetry.total_tokens_observed == 3
    assert telemetry.min_horror_kl == 0.1
    assert telemetry.max_mode_reached == SidecarSafetyMode.INTERVENTION
    assert telemetry.intervention is not None


def test_session_sentinel_caution_and_stabilizer_takeover(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(session_mod, "get_default_backend", lambda: b)

    session = SidecarSafetySession(policy=_policy(), stabilizer_configured=True)

    mode_caution = session.observe(
        SidecarDivergenceSample(token_index=0, kl_to_horror=None, kl_to_sentinel=0.5),
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert mode_caution == SidecarSafetyMode.CAUTION

    control = SessionControlState(
        scenario=ScenarioMode.HORROR,
        consent=ConsentGrant(
            is_granted=True,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            remaining_turns=2,
        ),
    )

    now = datetime.now(timezone.utc)
    mode_intervention = session.observe(
        SidecarDivergenceSample(token_index=1, kl_to_horror=0.1),
        control=control,
        now=now,
    )

    assert mode_intervention == SidecarSafetyMode.INTERVENTION

    intervention = session.consume_pending_intervention()
    assert intervention is not None
    assert intervention.kind == InterventionKind.STABILIZER_TAKEOVER
    assert "scenario=horror" in intervention.reason
    assert "consent=True" in intervention.reason


def test_session_reset_and_non_finite_inputs(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(session_mod, "get_default_backend", lambda: b)

    session = SidecarSafetySession(policy=SidecarSafetyPolicy.default())

    mode = session.observe(
        SidecarDivergenceSample(
            token_index=0,
            kl_to_horror=float("nan"),
            kl_to_sentinel=float("inf"),
        ),
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert mode == SidecarSafetyMode.NORMAL

    telemetry_before = session.telemetry_snapshot()
    assert telemetry_before.total_tokens_observed == 1
    assert telemetry_before.min_horror_kl is None
    assert telemetry_before.min_sentinel_kl is None

    session.reset()
    telemetry_after = session.telemetry_snapshot()
    assert telemetry_after.total_tokens_observed == 0
    assert telemetry_after.intervention is None
