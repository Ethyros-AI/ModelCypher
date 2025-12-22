"""Sidecar safety module.

Provides types and session management for sidecar LoRA safety monitoring,
including KL divergence-based threshold detection and consent-gated
threshold adjustments.
"""

from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ConsentGrant,
    ScenarioMode,
    SessionControlState,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_configuration import (
    SidecarSafetyConfiguration,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_decision import (
    InterventionKind,
    SidecarDivergenceSample,
    SidecarSafetyIntervention,
    SidecarSafetyMode,
    SidecarSafetyTelemetry,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_policy import (
    SidecarSafetyPolicy,
    SidecarSafetyThresholds,
)
from modelcypher.core.domain.safety.sidecar.sidecar_safety_session import (
    SidecarSafetySession,
)

__all__ = [
    "ConsentGrant",
    "InterventionKind",
    "ScenarioMode",
    "SessionControlState",
    "SidecarDivergenceSample",
    "SidecarSafetyConfiguration",
    "SidecarSafetyIntervention",
    "SidecarSafetyMode",
    "SidecarSafetyPolicy",
    "SidecarSafetySession",
    "SidecarSafetyTelemetry",
    "SidecarSafetyThresholds",
]
