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

"""
Intervention Executor.

Executes safety interventions based on combined signals (GAS + CircuitBreaker).
Closes the safety loop by connecting detection to actual interventions during generation.

Ported 1:1 from the reference Swift implementation.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration, CircuitBreakerState, InterventionLevel, RecommendedAction
)

# Mocking dependencies if not present
class GeometricAlignmentSystem:
    # Use the same InterventionLevel as CircuitBreaker to avoid circular deps or redefinition issues
    InterventionLevel = InterventionLevel
    
    @dataclass
    class Decision:
        level: "InterventionLevel"
        # pattern: Pattern ... simplified


@dataclass
class ExecutionResult:
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
    gas_level: InterventionLevel
    circuit_breaker_state: CircuitBreakerState
    severity: float
    trigger_source: "TriggerSource"
    token_index: int

    class TriggerSource(str, Enum):
        GAS = "gas"
        CIRCUIT_BREAKER = "circuit_breaker"
        COMBINED = "combined"

    @property
    def effective_level(self) -> InterventionLevel:
        cb_level = self._cb_to_level(self.circuit_breaker_state)
        # Assuming InterventionLevel enum is ordered or comparable via custom logic
        # For simplicity, using string comparison if values are ordered (level0 < level1)
        # Or implementing a helper.
        return max(self.gas_level, cb_level, key=lambda l: l.value)

    @staticmethod
    def _cb_to_level(state: CircuitBreakerState) -> InterventionLevel:
        if state.recommended_action == RecommendedAction.continue_generation:
            return InterventionLevel.level0_continue
        elif state.recommended_action in (RecommendedAction.monitor, RecommendedAction.reduce_temperature):
            return InterventionLevel.level1_gentle
        elif state.recommended_action == RecommendedAction.insert_safety_prompt:
            return InterventionLevel.level2_clarify
        elif state.recommended_action == RecommendedAction.stop_generation:
            return InterventionLevel.level4_terminate
        elif state.recommended_action == RecommendedAction.human_review:
            return InterventionLevel.level3_hard
        return InterventionLevel.level0_continue


@dataclass
class InterventionConfig:
    auto_execute_soft_interventions: bool = True
    temperature_reduction_factor: float = 0.5
    default_safety_prompt: str = "Please ensure your response is helpful, harmless, and honest."
    emit_telemetry: bool = True

    @classmethod
    def default(cls) -> "InterventionConfig":
        return cls()


class UserChoice(str, Enum):
    CONTINUE = "continue"
    CONTINUE_WITH_SAFETY = "continue_with_safety"
    STOP = "stop"
    MODIFY = "modify"


class InterventionExecutor:
    """
    Executes safety interventions.
    """

    def __init__(
        self,
        config: InterventionConfig = InterventionConfig.default(),
        confirmation_callback: Callable[[UUID, CombinedEvaluation], Awaitable[None]] | None = None,
        telemetry_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    ):
        self.config = config
        self.confirmation_callback = confirmation_callback
        self.telemetry_callback = telemetry_callback
        
        self.pending_confirmations: dict[UUID, CombinedEvaluation] = {}
        self.execution_history: list[dict[str, Any]] = []
        self.max_history_size = 100

    async def evaluate_and_execute(
        self,
        gas_decision: GeometricAlignmentSystem.Decision | None,
        circuit_breaker_state: CircuitBreakerState,
        token_index: int
    ) -> ExecutionResult:
        """Evaluates combined signals and executes intervention."""
        
        gas_level = gas_decision.level if gas_decision else InterventionLevel.level0_continue
        
        # Determine trigger source
        trigger_source = CombinedEvaluation.TriggerSource.GAS # Default
        if gas_level != InterventionLevel.level0_continue and circuit_breaker_state.is_tripped:
            trigger_source = CombinedEvaluation.TriggerSource.COMBINED
        elif gas_level != InterventionLevel.level0_continue:
            trigger_source = CombinedEvaluation.TriggerSource.GAS
        elif circuit_breaker_state.is_tripped:
            trigger_source = CombinedEvaluation.TriggerSource.CIRCUIT_BREAKER
            
        evaluation = CombinedEvaluation(
            gas_level=gas_level,
            circuit_breaker_state=circuit_breaker_state,
            severity=circuit_breaker_state.severity,
            trigger_source=trigger_source,
            token_index=token_index
        )
        
        return await self._execute_intervention(evaluation)

    async def _execute_intervention(self, evaluation: CombinedEvaluation) -> ExecutionResult:
        level = evaluation.effective_level
        
        if level == InterventionLevel.level0_continue:
            result = ExecutionResult(ExecutionResult.Type.CONTINUE)
            
        elif level == InterventionLevel.level1_gentle:
            if self.config.auto_execute_soft_interventions:
                result = ExecutionResult(
                    ExecutionResult.Type.SCALED_LOGITS, 
                    factor=self.config.temperature_reduction_factor
                )
                await self._emit_executed(level, result, evaluation)
            else:
                result = await self._request_confirmation(evaluation)
                
        elif level == InterventionLevel.level2_clarify:
            if self.config.auto_execute_soft_interventions:
                result = ExecutionResult(
                    ExecutionResult.Type.INJECTED_PROMPT,
                    message=self.config.default_safety_prompt
                )
                await self._emit_executed(level, result, evaluation)
            else:
                result = await self._request_confirmation(evaluation)
                
        elif level == InterventionLevel.level3_hard:
            result = await self._request_confirmation(evaluation)
            
        elif level == InterventionLevel.level4_terminate:
            reason = self._build_termination_reason(evaluation)
            result = ExecutionResult(ExecutionResult.Type.TERMINATED, reason=reason)
            await self._emit_terminated(evaluation, reason)
            
        else:
            result = ExecutionResult(ExecutionResult.Type.CONTINUE)

        self._record_execution(level, result, evaluation.token_index)
        return result

    async def _request_confirmation(self, evaluation: CombinedEvaluation) -> ExecutionResult:
        correlation_id = uuid4()
        self.pending_confirmations[correlation_id] = evaluation
        
        # Find some way to notify coordinator
        if self.confirmation_callback:
            await self.confirmation_callback(correlation_id, evaluation)
            
        # Emit pending telemetry
        if self.config.emit_telemetry and self.telemetry_callback:
             await self.telemetry_callback("intervention_pending", {
                "level": evaluation.effective_level.value,
                "correlation_id": str(correlation_id),
                "severity": evaluation.severity
            })

        return ExecutionResult(ExecutionResult.Type.PENDING_CONFIRMATION, correlation_id=correlation_id)

    async def resolve_confirmation(
        self,
        correlation_id: UUID,
        choice: UserChoice,
        custom_prompt: str | None = None
    ) -> ExecutionResult:
        """Resolves a pending confirmation."""
        if correlation_id not in self.pending_confirmations:
            return ExecutionResult(ExecutionResult.Type.CONTINUE)
            
        del self.pending_confirmations[correlation_id]
        
        if choice == UserChoice.CONTINUE:
            return ExecutionResult(ExecutionResult.Type.CONTINUE)
        elif choice == UserChoice.CONTINUE_WITH_SAFETY:
            return ExecutionResult(ExecutionResult.Type.INJECTED_PROMPT, message=self.config.default_safety_prompt)
        elif choice == UserChoice.STOP:
            return ExecutionResult(ExecutionResult.Type.TERMINATED, reason="User chose to stop")
        elif choice == UserChoice.MODIFY:
            return ExecutionResult(ExecutionResult.Type.INJECTED_PROMPT, message=custom_prompt or self.config.default_safety_prompt)
            
        return ExecutionResult(ExecutionResult.Type.CONTINUE)

    def _build_termination_reason(self, evaluation: CombinedEvaluation) -> str:
        if evaluation.trigger_source == CombinedEvaluation.TriggerSource.GAS:
            return f"GAS level {evaluation.gas_level.value}: Entropy instability"
        elif evaluation.trigger_source == CombinedEvaluation.TriggerSource.CIRCUIT_BREAKER:
            return evaluation.circuit_breaker_state.interpretation
        else:
            return f"Combined safety signals exceeded threshold (severity: {evaluation.severity:.2f})"

    async def _emit_executed(self, level: InterventionLevel, result: ExecutionResult, evaluation: CombinedEvaluation):
        if self.config.emit_telemetry and self.telemetry_callback:
            await self.telemetry_callback("intervention_executed", {
                "level": level.value,
                "action": result.type.value,
                "severity": evaluation.severity
            })

    async def _emit_terminated(self, evaluation: CombinedEvaluation, reason: str):
        if self.config.emit_telemetry and self.telemetry_callback:
            await self.telemetry_callback("intervention_terminated", {
                 "level": evaluation.effective_level.value,
                 "reason": reason,
                 "severity": evaluation.severity
            })

    def _record_execution(self, level: InterventionLevel, result: ExecutionResult, token_index: int):
        self.execution_history.append({
            "timestamp": datetime.now(),
            "level": level,
            "result": result.type.value,
            "token_index": token_index
        })
        if len(self.execution_history) > self.max_history_size:
            self.execution_history.pop(0)
