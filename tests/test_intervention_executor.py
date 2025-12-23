"""Tests for InterventionExecutor.

Tests the safety intervention execution system that closes the
safety loop by connecting detection to actual interventions.
"""

import pytest
from uuid import uuid4

from modelcypher.core.domain.safety.intervention_executor import (
    InterventionExecutor,
    InterventionConfig,
    ExecutionResult,
    CombinedEvaluation,
    UserChoice,
    GeometricAlignmentSystem,
)
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerState,
    InterventionLevel,
    RecommendedAction,
    SignalContributions,
    TriggerSource,
)


class TestInterventionConfig:
    """Tests for InterventionConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = InterventionConfig.default()
        assert config.auto_execute_soft_interventions is True
        assert config.temperature_reduction_factor == 0.5
        assert config.emit_telemetry is True
        assert "helpful" in config.default_safety_prompt.lower()

    def test_custom_config(self):
        """Custom config should accept provided values."""
        config = InterventionConfig(
            auto_execute_soft_interventions=False,
            temperature_reduction_factor=0.3,
            default_safety_prompt="Custom safety message",
            emit_telemetry=False,
        )
        assert config.auto_execute_soft_interventions is False
        assert config.temperature_reduction_factor == 0.3
        assert config.default_safety_prompt == "Custom safety message"
        assert config.emit_telemetry is False


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_continue_result(self):
        """Continue result should have correct type."""
        result = ExecutionResult(ExecutionResult.Type.CONTINUE)
        assert result.type == ExecutionResult.Type.CONTINUE
        assert result.factor is None
        assert result.message is None

    def test_scaled_logits_result(self):
        """Scaled logits result should include factor."""
        result = ExecutionResult(ExecutionResult.Type.SCALED_LOGITS, factor=0.5)
        assert result.type == ExecutionResult.Type.SCALED_LOGITS
        assert result.factor == 0.5

    def test_injected_prompt_result(self):
        """Injected prompt result should include message."""
        result = ExecutionResult(
            ExecutionResult.Type.INJECTED_PROMPT,
            message="Safety prompt here"
        )
        assert result.type == ExecutionResult.Type.INJECTED_PROMPT
        assert result.message == "Safety prompt here"

    def test_terminated_result(self):
        """Terminated result should include reason."""
        result = ExecutionResult(
            ExecutionResult.Type.TERMINATED,
            reason="Safety limit exceeded"
        )
        assert result.type == ExecutionResult.Type.TERMINATED
        assert result.reason == "Safety limit exceeded"

    def test_pending_confirmation_result(self):
        """Pending confirmation result should include correlation ID."""
        correlation_id = uuid4()
        result = ExecutionResult(
            ExecutionResult.Type.PENDING_CONFIRMATION,
            correlation_id=correlation_id
        )
        assert result.type == ExecutionResult.Type.PENDING_CONFIRMATION
        assert result.correlation_id == correlation_id


class TestCombinedEvaluation:
    """Tests for CombinedEvaluation dataclass."""

    @pytest.fixture
    def safe_cb_state(self):
        """Create a safe circuit breaker state."""
        return CircuitBreakerState(
            is_tripped=False,
            severity=0.1,
            trigger_source=None,
            confidence=0.8,
            recommended_action=RecommendedAction.continue_generation,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

    @pytest.fixture
    def tripped_cb_state(self):
        """Create a tripped circuit breaker state."""
        return CircuitBreakerState(
            is_tripped=True,
            severity=0.85,
            trigger_source=TriggerSource.entropy_spike,
            confidence=0.9,
            recommended_action=RecommendedAction.stop_generation,
            signal_contributions=SignalContributions(0.5, 0.15, 0.1, 0.1),
            token_index=100,
        )

    def test_effective_level_gas_higher(self, safe_cb_state):
        """Should use GAS level when higher than CB level."""
        evaluation = CombinedEvaluation(
            gas_level=InterventionLevel.level2_clarify,
            circuit_breaker_state=safe_cb_state,  # safe state = level0
            severity=0.5,
            trigger_source=CombinedEvaluation.TriggerSource.GAS,
            token_index=50,
        )

        # GAS level2 should dominate over CB level0
        assert evaluation.effective_level == InterventionLevel.level2_clarify

    def test_effective_level_cb_higher(self, tripped_cb_state):
        """Should use CB level when higher than GAS level."""
        evaluation = CombinedEvaluation(
            gas_level=InterventionLevel.level0_continue,
            circuit_breaker_state=tripped_cb_state,
            severity=0.85,
            trigger_source=CombinedEvaluation.TriggerSource.CIRCUIT_BREAKER,
            token_index=100,
        )

        # CB level should dominate (stop_generation -> level4)
        assert evaluation.effective_level == InterventionLevel.level4_terminate


class TestInterventionExecutor:
    """Tests for InterventionExecutor."""

    @pytest.fixture
    def executor(self):
        """Default executor."""
        return InterventionExecutor()

    @pytest.fixture
    def safe_cb_state(self):
        """Create a safe circuit breaker state."""
        return CircuitBreakerState(
            is_tripped=False,
            severity=0.1,
            trigger_source=None,
            confidence=0.8,
            recommended_action=RecommendedAction.continue_generation,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

    @pytest.fixture
    def warning_cb_state(self):
        """Create a warning-level circuit breaker state."""
        return CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.monitor,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

    @pytest.fixture
    def tripped_cb_state(self):
        """Create a tripped circuit breaker state."""
        return CircuitBreakerState(
            is_tripped=True,
            severity=0.85,
            trigger_source=TriggerSource.entropy_spike,
            confidence=0.9,
            recommended_action=RecommendedAction.stop_generation,
            signal_contributions=SignalContributions(0.5, 0.15, 0.1, 0.1),
            token_index=100,
        )

    @pytest.mark.asyncio
    async def test_evaluate_safe_signals(self, executor, safe_cb_state):
        """Safe signals should result in continue action."""
        result = await executor.evaluate_and_execute(
            gas_decision=None,
            circuit_breaker_state=safe_cb_state,
            token_index=10,
        )

        assert result.type == ExecutionResult.Type.CONTINUE

    @pytest.mark.asyncio
    async def test_evaluate_level1_auto_scales(self, executor, warning_cb_state):
        """Level 1 with auto-execute should scale logits."""
        result = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level1_gentle
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert result.type == ExecutionResult.Type.SCALED_LOGITS
        assert result.factor == 0.5  # default temperature reduction

    @pytest.mark.asyncio
    async def test_evaluate_level2_injects_prompt(self, executor, warning_cb_state):
        """Level 2 with auto-execute should inject safety prompt."""
        result = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level2_clarify
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert result.type == ExecutionResult.Type.INJECTED_PROMPT
        assert "helpful" in result.message.lower()

    @pytest.mark.asyncio
    async def test_evaluate_level4_terminates(self, executor, tripped_cb_state):
        """Level 4 should terminate generation."""
        result = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level4_terminate
            ),
            circuit_breaker_state=tripped_cb_state,
            token_index=100,
        )

        assert result.type == ExecutionResult.Type.TERMINATED
        assert result.reason is not None

    @pytest.mark.asyncio
    async def test_evaluate_level3_requires_confirmation(self):
        """Level 3 should require user confirmation."""
        executor = InterventionExecutor(InterventionConfig.default())
        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.human_review,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        result = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level3_hard
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert result.type == ExecutionResult.Type.PENDING_CONFIRMATION
        assert result.correlation_id is not None

    @pytest.mark.asyncio
    async def test_no_auto_execute_requires_confirmation(self, warning_cb_state):
        """Disabling auto-execute should require confirmation for soft interventions."""
        config = InterventionConfig(auto_execute_soft_interventions=False)
        executor = InterventionExecutor(config)

        result = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level1_gentle
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert result.type == ExecutionResult.Type.PENDING_CONFIRMATION


class TestResolveConfirmation:
    """Tests for resolve_confirmation method."""

    @pytest.fixture
    def executor(self):
        return InterventionExecutor()

    @pytest.mark.asyncio
    async def test_resolve_unknown_id_continues(self, executor):
        """Resolving unknown correlation ID should continue."""
        result = await executor.resolve_confirmation(
            correlation_id=uuid4(),
            choice=UserChoice.STOP,
        )

        assert result.type == ExecutionResult.Type.CONTINUE

    @pytest.mark.asyncio
    async def test_resolve_continue_choice(self, executor):
        """User choosing CONTINUE should continue generation."""
        # First, create a pending confirmation
        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.human_review,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        pending = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level3_hard
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert pending.type == ExecutionResult.Type.PENDING_CONFIRMATION
        correlation_id = pending.correlation_id

        # Now resolve with CONTINUE
        result = await executor.resolve_confirmation(
            correlation_id=correlation_id,
            choice=UserChoice.CONTINUE,
        )

        assert result.type == ExecutionResult.Type.CONTINUE

    @pytest.mark.asyncio
    async def test_resolve_stop_choice(self, executor):
        """User choosing STOP should terminate."""
        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.human_review,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        pending = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level3_hard
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        correlation_id = pending.correlation_id

        result = await executor.resolve_confirmation(
            correlation_id=correlation_id,
            choice=UserChoice.STOP,
        )

        assert result.type == ExecutionResult.Type.TERMINATED
        assert "User chose to stop" in result.reason

    @pytest.mark.asyncio
    async def test_resolve_continue_with_safety(self, executor):
        """User choosing CONTINUE_WITH_SAFETY should inject prompt."""
        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.human_review,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        pending = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level3_hard
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        correlation_id = pending.correlation_id

        result = await executor.resolve_confirmation(
            correlation_id=correlation_id,
            choice=UserChoice.CONTINUE_WITH_SAFETY,
        )

        assert result.type == ExecutionResult.Type.INJECTED_PROMPT

    @pytest.mark.asyncio
    async def test_resolve_modify_with_custom_prompt(self, executor):
        """User choosing MODIFY should inject custom prompt."""
        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.human_review,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        pending = await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level3_hard
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        correlation_id = pending.correlation_id

        result = await executor.resolve_confirmation(
            correlation_id=correlation_id,
            choice=UserChoice.MODIFY,
            custom_prompt="Please reconsider your approach",
        )

        assert result.type == ExecutionResult.Type.INJECTED_PROMPT
        assert result.message == "Please reconsider your approach"


class TestExecutionHistory:
    """Tests for execution history tracking."""

    @pytest.fixture
    def executor(self):
        return InterventionExecutor()

    @pytest.fixture
    def safe_cb_state(self):
        return CircuitBreakerState(
            is_tripped=False,
            severity=0.1,
            trigger_source=None,
            confidence=0.8,
            recommended_action=RecommendedAction.continue_generation,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

    @pytest.mark.asyncio
    async def test_history_recorded(self, executor, safe_cb_state):
        """Execution should be recorded in history."""
        await executor.evaluate_and_execute(
            gas_decision=None,
            circuit_breaker_state=safe_cb_state,
            token_index=10,
        )

        assert len(executor.execution_history) == 1
        assert executor.execution_history[0]["token_index"] == 10

    @pytest.mark.asyncio
    async def test_history_limit(self, executor, safe_cb_state):
        """History should be limited to max_history_size."""
        for i in range(150):
            await executor.evaluate_and_execute(
                gas_decision=None,
                circuit_breaker_state=safe_cb_state,
                token_index=i,
            )

        assert len(executor.execution_history) <= executor.max_history_size


class TestTelemetry:
    """Tests for telemetry callback."""

    @pytest.mark.asyncio
    async def test_telemetry_callback_called(self):
        """Telemetry callback should be called on execution."""
        telemetry_events = []

        async def capture_telemetry(event_type, data):
            telemetry_events.append((event_type, data))

        executor = InterventionExecutor(
            config=InterventionConfig.default(),
            telemetry_callback=capture_telemetry,
        )

        warning_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.6,
            trigger_source=None,
            confidence=0.75,
            recommended_action=RecommendedAction.monitor,
            signal_contributions=SignalContributions(0.3, 0.15, 0.1, 0.05),
            token_index=50,
        )

        await executor.evaluate_and_execute(
            gas_decision=GeometricAlignmentSystem.Decision(
                level=InterventionLevel.level1_gentle
            ),
            circuit_breaker_state=warning_cb_state,
            token_index=50,
        )

        assert len(telemetry_events) > 0
        assert any(event[0] == "intervention_executed" for event in telemetry_events)

    @pytest.mark.asyncio
    async def test_telemetry_disabled(self):
        """Telemetry should not be called when disabled."""
        telemetry_events = []

        async def capture_telemetry(event_type, data):
            telemetry_events.append((event_type, data))

        config = InterventionConfig(emit_telemetry=False)
        executor = InterventionExecutor(
            config=config,
            telemetry_callback=capture_telemetry,
        )

        safe_cb_state = CircuitBreakerState(
            is_tripped=False,
            severity=0.1,
            trigger_source=None,
            confidence=0.8,
            recommended_action=RecommendedAction.continue_generation,
            signal_contributions=SignalContributions(0.05, 0.02, 0.02, 0.01),
            token_index=10,
        )

        await executor.evaluate_and_execute(
            gas_decision=None,
            circuit_breaker_state=safe_cb_state,
            token_index=10,
        )

        assert len(telemetry_events) == 0
