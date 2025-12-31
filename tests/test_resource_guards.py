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

"""Tests for training resource guards (GPU lock, session management)."""

import asyncio

import pytest

from modelcypher.core.domain.training.resources import (
    ActivitySubscriber,
    InferenceInProgressError,
    InferenceOwner,
    ResourceError,
    ResourceIntensiveOperation,
    TrainingActivityState,
    TrainingInProgressError,
    TrainingReleaseReason,
    TrainingResourceGuard,
    TrainingSessionInfo,
    WorkloadActivityState,
)


@pytest.fixture
def guard():
    """Create a fresh TrainingResourceGuard for testing."""
    TrainingResourceGuard.reset_for_testing()
    g = TrainingResourceGuard()
    yield g
    TrainingResourceGuard.reset_for_testing()


class TestResourceIntensiveOperation:
    """Tests for ResourceIntensiveOperation enum."""

    def test_rag_indexing_value(self):
        assert ResourceIntensiveOperation.RAG_INDEXING.value == "RAG Indexing"

    def test_rag_query_value(self):
        assert ResourceIntensiveOperation.RAG_QUERY.value == "RAG Query"

    def test_model_inference_value(self):
        assert ResourceIntensiveOperation.MODEL_INFERENCE.value == "Model Inference"

    def test_unavailable_message(self):
        msg = ResourceIntensiveOperation.RAG_INDEXING.unavailable_message
        assert "RAG Indexing" in msg
        assert "Training" in msg


class TestInferenceOwner:
    """Tests for InferenceOwner enum."""

    def test_user_value(self):
        assert InferenceOwner.USER.value == "user"

    def test_comparison_session_value(self):
        assert InferenceOwner.COMPARISON_SESSION.value == "comparison_session"


class TestTrainingSessionInfo:
    """Tests for TrainingSessionInfo dataclass."""

    def test_required_fields(self):
        info = TrainingSessionInfo(
            job_id="job-123",
            start_time=1000.0,
            duration=120.5,
        )
        assert info.job_id == "job-123"
        assert info.start_time == 1000.0
        assert info.duration == 120.5

    def test_formatted_duration_minutes_seconds(self):
        info = TrainingSessionInfo(job_id="job", start_time=0, duration=125)
        assert info.formatted_duration == "2:05"

    def test_formatted_duration_zero(self):
        info = TrainingSessionInfo(job_id="job", start_time=0, duration=0)
        assert info.formatted_duration == "0:00"

    def test_formatted_duration_only_seconds(self):
        info = TrainingSessionInfo(job_id="job", start_time=0, duration=45)
        assert info.formatted_duration == "0:45"

    def test_formatted_duration_long(self):
        info = TrainingSessionInfo(job_id="job", start_time=0, duration=3661)
        # 61 minutes 1 second
        assert info.formatted_duration == "61:01"


class TestTrainingReleaseReason:
    """Tests for TrainingReleaseReason enum."""

    def test_normal_value(self):
        assert TrainingReleaseReason.NORMAL.value == "normal"

    def test_watchdog_timeout_value(self):
        assert TrainingReleaseReason.WATCHDOG_TIMEOUT.value == "watchdog_timeout"

    def test_cancelled_value(self):
        assert TrainingReleaseReason.CANCELLED.value == "cancelled"


class TestTrainingActivityState:
    """Tests for TrainingActivityState dataclass."""

    def test_minimal_state(self):
        state = TrainingActivityState(is_training=False)
        assert state.is_training is False
        assert state.active_job_id is None
        assert state.termination_reason is None

    def test_active_training(self):
        state = TrainingActivityState(
            is_training=True,
            active_job_id="job-123",
        )
        assert state.is_training is True
        assert state.active_job_id == "job-123"

    def test_with_termination_reason(self):
        state = TrainingActivityState(
            is_training=False,
            termination_reason=TrainingReleaseReason.CANCELLED,
        )
        assert state.termination_reason == TrainingReleaseReason.CANCELLED


class TestWorkloadActivityState:
    """Tests for WorkloadActivityState dataclass."""

    def test_inactive_state(self):
        state = WorkloadActivityState(is_active=False)
        assert state.is_active is False
        assert state.training_job_id is None
        assert state.inference_owner is None

    def test_training_active(self):
        state = WorkloadActivityState(
            is_active=True,
            training_job_id="job-123",
        )
        assert state.is_active is True
        assert state.training_job_id == "job-123"

    def test_inference_active(self):
        state = WorkloadActivityState(
            is_active=True,
            inference_owner="user",
        )
        assert state.inference_owner == "user"


class TestResourceErrors:
    """Tests for resource exception classes."""

    def test_resource_error(self):
        error = ResourceError("GPU busy")
        assert "GPU busy" in str(error)

    def test_training_in_progress_error(self):
        error = TrainingInProgressError("job-123")
        assert error.job_id == "job-123"
        assert "job-123" in str(error)

    def test_inference_in_progress_error(self):
        error = InferenceInProgressError("user")
        assert error.owner == "user"
        assert "user" in str(error)


class TestActivitySubscriber:
    """Tests for ActivitySubscriber class."""

    def test_has_unique_id(self):
        sub1 = ActivitySubscriber()
        sub2 = ActivitySubscriber()
        assert sub1.id != sub2.id

    def test_emit_and_receive(self):
        subscriber = ActivitySubscriber()
        state = TrainingActivityState(is_training=True, active_job_id="job-1")
        subscriber.emit(state)

        # Should be in queue
        assert not subscriber._queue.empty()

    def test_close_sets_inactive(self):
        subscriber = ActivitySubscriber()
        assert subscriber._active is True
        subscriber.close()
        assert subscriber._active is False

    def test_emit_after_close_is_ignored(self):
        subscriber = ActivitySubscriber()
        subscriber.close()
        subscriber.emit("should be ignored")
        assert subscriber._queue.empty()


class TestTrainingResourceGuardSingleton:
    """Tests for TrainingResourceGuard singleton behavior."""

    def test_shared_returns_same_instance(self, guard):
        guard2 = TrainingResourceGuard.shared()
        assert guard is guard2

    def test_reset_for_testing_clears_instance(self):
        TrainingResourceGuard.reset_for_testing()
        g1 = TrainingResourceGuard()
        TrainingResourceGuard.reset_for_testing()
        g2 = TrainingResourceGuard()
        # After reset, should get new instance
        assert g1 is not g2


class TestTrainingResourceGuardTraining:
    """Tests for training session management."""

    @pytest.mark.asyncio
    async def test_is_training_active_initially_false(self, guard):
        assert guard.is_training_active is False

    @pytest.mark.asyncio
    async def test_begin_training_sets_active(self, guard):
        await guard.begin_training("job-1")
        assert guard.is_training_active is True

    @pytest.mark.asyncio
    async def test_end_training_clears_active(self, guard):
        await guard.begin_training("job-1")
        await guard.end_training("job-1")
        assert guard.is_training_active is False

    @pytest.mark.asyncio
    async def test_double_begin_training_raises(self, guard):
        await guard.begin_training("job-1")
        with pytest.raises(TrainingInProgressError) as exc_info:
            await guard.begin_training("job-2")
        assert exc_info.value.job_id == "job-1"

    @pytest.mark.asyncio
    async def test_end_wrong_job_id_no_effect(self, guard):
        await guard.begin_training("job-1")
        await guard.end_training("job-2")  # Wrong ID
        assert guard.is_training_active is True

    @pytest.mark.asyncio
    async def test_get_current_session_when_inactive(self, guard):
        session = await guard.get_current_training_session()
        assert session is None

    @pytest.mark.asyncio
    async def test_get_current_session_when_active(self, guard):
        await guard.begin_training("job-1")
        session = await guard.get_current_training_session()
        assert session is not None
        assert session.job_id == "job-1"
        assert session.duration >= 0


class TestTrainingResourceGuardInference:
    """Tests for inference session management."""

    @pytest.mark.asyncio
    async def test_begin_inference_while_training_raises(self, guard):
        await guard.begin_training("job-1")
        with pytest.raises(ResourceError) as exc_info:
            await guard.begin_inference("user")
        assert "job-1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_begin_inference_success(self, guard):
        await guard.begin_inference("user")
        is_active = await guard.is_workload_active()
        assert is_active is True

    @pytest.mark.asyncio
    async def test_end_inference(self, guard):
        await guard.begin_inference("user")
        await guard.end_inference("user")
        is_active = await guard.is_workload_active()
        assert is_active is False

    @pytest.mark.asyncio
    async def test_duplicate_inference_owner_no_effect(self, guard):
        await guard.begin_inference("user")
        await guard.begin_inference("user")  # Same owner again
        # Should still work

    @pytest.mark.asyncio
    async def test_max_concurrent_inference(self, guard):
        await guard.begin_inference("owner1")
        await guard.begin_inference("owner2")
        with pytest.raises(ResourceError) as exc_info:
            await guard.begin_inference("owner3")
        assert "Maximum concurrent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_begin_training_while_inference_raises(self, guard):
        await guard.begin_inference("user")
        with pytest.raises(InferenceInProgressError) as exc_info:
            await guard.begin_training("job-1")
        assert exc_info.value.owner == "user"


class TestTrainingResourceGuardContextManagers:
    """Tests for context manager methods."""

    @pytest.mark.asyncio
    async def test_training_session_context_manager(self, guard):
        async with guard.training_session("job-1"):
            assert guard.is_training_active is True
        assert guard.is_training_active is False

    @pytest.mark.asyncio
    async def test_training_session_releases_on_exception(self, guard):
        with pytest.raises(ValueError):
            async with guard.training_session("job-1"):
                raise ValueError("Test error")
        assert guard.is_training_active is False

    @pytest.mark.asyncio
    async def test_inference_session_context_manager(self, guard):
        async with guard.inference_session("user"):
            is_active = await guard.is_workload_active()
            assert is_active is True
        is_active = await guard.is_workload_active()
        assert is_active is False


class TestResourceAccessControl:
    """Tests for resource access control methods."""

    @pytest.mark.asyncio
    async def test_request_access_when_idle(self, guard):
        # Should not raise
        await guard.request_resource_access(ResourceIntensiveOperation.RAG_INDEXING)

    @pytest.mark.asyncio
    async def test_request_access_during_training_raises(self, guard):
        await guard.begin_training("job-1")
        with pytest.raises(ResourceError):
            await guard.request_resource_access(ResourceIntensiveOperation.RAG_QUERY)

    @pytest.mark.asyncio
    async def test_can_perform_operation_when_idle(self, guard):
        can = await guard.can_perform_operation(ResourceIntensiveOperation.MODEL_INFERENCE)
        assert can is True

    @pytest.mark.asyncio
    async def test_can_perform_operation_during_training(self, guard):
        await guard.begin_training("job-1")
        can = await guard.can_perform_operation(ResourceIntensiveOperation.RAG_INDEXING)
        assert can is False

    @pytest.mark.asyncio
    async def test_is_workload_active_when_idle(self, guard):
        is_active = await guard.is_workload_active()
        assert is_active is False

    @pytest.mark.asyncio
    async def test_is_workload_active_during_training(self, guard):
        await guard.begin_training("job-1")
        is_active = await guard.is_workload_active()
        assert is_active is True

    @pytest.mark.asyncio
    async def test_is_workload_active_during_inference(self, guard):
        await guard.begin_inference("user")
        is_active = await guard.is_workload_active()
        assert is_active is True
