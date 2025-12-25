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

"""Intervention Executor - severity-based intervention system.

Executes safety interventions based on raw severity measurements.
Closes the safety loop by connecting detection to actual interventions.

No classification levels. Raw severity determines action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerState,
    RecommendedAction,
)


@dataclass
class ExecutionResult:
    """Result of intervention execution."""

    class Type(str, Enum):
        CONTINUE = "continue"
        SCALED_LOGITS = "scaled_logits"
        INJECTED_PROMPT = "injected_prompt"
        PENDING_CONFIRMATION = "pending_confirmation"
        TERMINATED = "terminated"

    type: Type
    factor: float | None = None
    message: str | None = None
    reason: str | None = None
    correlation_id: UUID | None = None


@dataclass
class CombinedEvaluation:
    """Combined evaluation from GAS and circuit breaker."""

    oscillation_severity: float
    """Raw oscillation severity [0, 1]."""

    circuit_breaker_state: CircuitBreakerState
    combined_severity: float
    """Combined severity from all signals."""

    trigger_source: "TriggerSource"
    token_index: int

    class TriggerSource(str, Enum):
        GAS = "gas"
        CIRCUIT_BREAKER = "circuit_breaker"
        COMBINED = "combined"

    @property
    def effective_severity(self) -> float:
        """Maximum severity across signals."""
        return max(self.oscillation_severity, self.circuit_breaker_state.severity)


@dataclass
class InterventionConfig:
    """Intervention executor configuration.

    Thresholds define severity ranges for different actions.
    """

    auto_execute_soft_interventions: bool = True
    temperature_reduction_factor: float = 0.5
    default_safety_prompt: str = "Please ensure your response is helpful, harmless, and honest."
    emit_telemetry: bool = True

    # Severity thresholds for action selection (caller should calibrate these)
    gentle_threshold: float = 0.25
    """Severity above which to apply soft interventions (temperature reduction)."""

    clarify_threshold: float = 0.50
    """Severity above which to inject safety prompt."""

    hard_threshold: float = 0.75
    """Severity above which to require user confirmation."""

    terminate_threshold: float = 0.90
    """Severity above which to terminate generation."""

    @classmethod
    def default(cls) -> InterventionConfig:
        return cls()


class UserChoice(str, Enum):
    """User choice for confirmation resolution."""

    CONTINUE = "continue"
    CONTINUE_WITH_SAFETY = "continue_with_safety"
    STOP = "stop"
    MODIFY = "modify"


class InterventionExecutor:
    """Executes safety interventions based on severity measurements."""

    def __init__(
        self,
        config: InterventionConfig = InterventionConfig.default(),
        confirmation_callback: Callable[[UUID, CombinedEvaluation], Awaitable[None]] | None = None,
        telemetry_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ):
        self.config = config
        self.confirmation_callback = confirmation_callback
        self.telemetry_callback = telemetry_callback

        self.pending_confirmations: dict[UUID, CombinedEvaluation] = {}
        self.execution_history: list[dict[str, Any]] = []
        self.max_history_size = 100

    async def evaluate_and_execute(
        self,
        oscillation_severity: float | None,
        circuit_breaker_state: CircuitBreakerState,
        token_index: int,
    ) -> ExecutionResult:
        """Evaluate combined signals and execute intervention.

        Args:
            oscillation_severity: Raw oscillation severity [0, 1], or None if not available
            circuit_breaker_state: Circuit breaker state
            token_index: Current token index

        Returns:
            ExecutionResult indicating what action was taken
        """
        severity = oscillation_severity if oscillation_severity is not None else 0.0

        # Determine trigger source
        if severity > 0.0 and circuit_breaker_state.is_tripped:
            trigger_source = CombinedEvaluation.TriggerSource.COMBINED
        elif severity > 0.0:
            trigger_source = CombinedEvaluation.TriggerSource.GAS
        elif circuit_breaker_state.is_tripped:
            trigger_source = CombinedEvaluation.TriggerSource.CIRCUIT_BREAKER
        else:
            trigger_source = CombinedEvaluation.TriggerSource.GAS

        evaluation = CombinedEvaluation(
            oscillation_severity=severity,
            circuit_breaker_state=circuit_breaker_state,
            combined_severity=circuit_breaker_state.severity,
            trigger_source=trigger_source,
            token_index=token_index,
        )

        return await self._execute_intervention(evaluation)

    async def _execute_intervention(self, evaluation: CombinedEvaluation) -> ExecutionResult:
        """Execute intervention based on severity."""
        severity = evaluation.effective_severity
        config = self.config

        if severity >= config.terminate_threshold:
            # Terminate
            reason = self._build_termination_reason(evaluation)
            result = ExecutionResult(ExecutionResult.Type.TERMINATED, reason=reason)
            await self._emit_terminated(evaluation, reason)

        elif severity >= config.hard_threshold:
            # Require confirmation
            result = await self._request_confirmation(evaluation)

        elif severity >= config.clarify_threshold:
            # Inject safety prompt
            if config.auto_execute_soft_interventions:
                result = ExecutionResult(
                    ExecutionResult.Type.INJECTED_PROMPT,
                    message=config.default_safety_prompt,
                )
                await self._emit_executed(severity, result, evaluation)
            else:
                result = await self._request_confirmation(evaluation)

        elif severity >= config.gentle_threshold:
            # Scale temperature
            if config.auto_execute_soft_interventions:
                result = ExecutionResult(
                    ExecutionResult.Type.SCALED_LOGITS,
                    factor=config.temperature_reduction_factor,
                )
                await self._emit_executed(severity, result, evaluation)
            else:
                result = await self._request_confirmation(evaluation)

        else:
            # Continue normally
            result = ExecutionResult(ExecutionResult.Type.CONTINUE)

        self._record_execution(severity, result, evaluation.token_index)
        return result

    async def _request_confirmation(self, evaluation: CombinedEvaluation) -> ExecutionResult:
        """Request user confirmation for intervention."""
        correlation_id = uuid4()
        self.pending_confirmations[correlation_id] = evaluation

        if self.confirmation_callback:
            await self.confirmation_callback(correlation_id, evaluation)

        if self.config.emit_telemetry and self.telemetry_callback:
            await self.telemetry_callback(
                "intervention_pending",
                {
                    "severity": evaluation.effective_severity,
                    "correlation_id": str(correlation_id),
                    "combined_severity": evaluation.combined_severity,
                },
            )

        return ExecutionResult(
            ExecutionResult.Type.PENDING_CONFIRMATION, correlation_id=correlation_id
        )

    async def resolve_confirmation(
        self, correlation_id: UUID, choice: UserChoice, custom_prompt: str | None = None
    ) -> ExecutionResult:
        """Resolve a pending confirmation."""
        if correlation_id not in self.pending_confirmations:
            return ExecutionResult(ExecutionResult.Type.CONTINUE)

        del self.pending_confirmations[correlation_id]

        if choice == UserChoice.CONTINUE:
            return ExecutionResult(ExecutionResult.Type.CONTINUE)
        elif choice == UserChoice.CONTINUE_WITH_SAFETY:
            return ExecutionResult(
                ExecutionResult.Type.INJECTED_PROMPT, message=self.config.default_safety_prompt
            )
        elif choice == UserChoice.STOP:
            return ExecutionResult(ExecutionResult.Type.TERMINATED, reason="User chose to stop")
        elif choice == UserChoice.MODIFY:
            return ExecutionResult(
                ExecutionResult.Type.INJECTED_PROMPT,
                message=custom_prompt or self.config.default_safety_prompt,
            )

        return ExecutionResult(ExecutionResult.Type.CONTINUE)

    def _build_termination_reason(self, evaluation: CombinedEvaluation) -> str:
        """Build termination reason from evaluation."""
        if evaluation.trigger_source == CombinedEvaluation.TriggerSource.GAS:
            return f"GAS severity {evaluation.oscillation_severity:.2f}: Entropy instability"
        elif evaluation.trigger_source == CombinedEvaluation.TriggerSource.CIRCUIT_BREAKER:
            return evaluation.circuit_breaker_state.interpretation
        else:
            return (
                f"Combined safety signals exceeded threshold "
                f"(severity: {evaluation.effective_severity:.2f})"
            )

    async def _emit_executed(
        self, severity: float, result: ExecutionResult, evaluation: CombinedEvaluation
    ):
        """Emit telemetry for executed intervention."""
        if self.config.emit_telemetry and self.telemetry_callback:
            await self.telemetry_callback(
                "intervention_executed",
                {
                    "severity": severity,
                    "action": result.type.value,
                    "combined_severity": evaluation.combined_severity,
                },
            )

    async def _emit_terminated(self, evaluation: CombinedEvaluation, reason: str):
        """Emit telemetry for termination."""
        if self.config.emit_telemetry and self.telemetry_callback:
            await self.telemetry_callback(
                "intervention_terminated",
                {
                    "severity": evaluation.effective_severity,
                    "reason": reason,
                    "combined_severity": evaluation.combined_severity,
                },
            )

    def _record_execution(
        self, severity: float, result: ExecutionResult, token_index: int
    ):
        """Record execution in history."""
        self.execution_history.append(
            {
                "timestamp": datetime.now(),
                "severity": severity,
                "result": result.type.value,
                "token_index": token_index,
            }
        )
        if len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)
