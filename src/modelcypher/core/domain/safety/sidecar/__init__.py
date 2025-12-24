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
