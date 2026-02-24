# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Integration tests for the checkpoint comparison pipeline.

Tests the flow: checkpoints → async generation → event streaming → comparison results.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import pytest

from modelcypher.core.use_cases.inference.comparison import (
    CheckpointComparisonCoordinator,
    ComparisonEvent,
    ComparisonResult,
    EventType,
)

# =============================================================================
# Test Data Structures
# =============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_creation(self) -> None:
        """ComparisonResult should store checkpoint, response, and metrics."""
        result = ComparisonResult(
            checkpoint_path="/path/to/checkpoint",
            response="Hello world",
            metrics={"tokens_per_second": 50.0},
        )

        assert result.checkpoint_path == "/path/to/checkpoint"
        assert result.response == "Hello world"
        assert result.metrics["tokens_per_second"] == 50.0


class TestComparisonEvent:
    """Tests for ComparisonEvent dataclass."""

    def test_prefetch_event(self) -> None:
        """Prefetch events should have path and index."""
        event = ComparisonEvent(
            type=EventType.PREFETCH_STARTED,
            index=0,
            path="/path/to/model",
        )

        assert event.type == EventType.PREFETCH_STARTED
        assert event.index == 0
        assert event.path == "/path/to/model"

    def test_token_event(self) -> None:
        """Token events should have text content."""
        event = ComparisonEvent(
            type=EventType.TOKEN,
            index=1,
            text="Hello",
        )

        assert event.type == EventType.TOKEN
        assert event.index == 1
        assert event.text == "Hello"

    def test_finished_event_with_result(self) -> None:
        """Finished events should have ComparisonResult."""
        result = ComparisonResult("/path", "response", None)
        event = ComparisonEvent(
            type=EventType.CHECKPOINT_FINISHED,
            index=0,
            result=result,
        )

        assert event.type == EventType.CHECKPOINT_FINISHED
        assert event.result is not None
        assert event.result.response == "response"

    def test_failed_event_with_error(self) -> None:
        """Failed events should have error message."""
        event = ComparisonEvent(
            type=EventType.CHECKPOINT_FAILED,
            index=0,
            path="/path/to/model",
            error="Model load failed",
        )

        assert event.type == EventType.CHECKPOINT_FAILED
        assert event.error == "Model load failed"


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_defined(self) -> None:
        """All required event types should be defined."""
        expected_types = [
            "PREFETCH_STARTED",
            "PREFETCH_FINISHED",
            "PREFETCH_FAILED",
            "CHECKPOINT_STARTED",
            "TOKEN",
            "CHECKPOINT_FINISHED",
            "CHECKPOINT_FAILED",
        ]

        for type_name in expected_types:
            assert hasattr(EventType, type_name)

    def test_event_type_values(self) -> None:
        """Event types should have string values."""
        assert EventType.PREFETCH_STARTED.value == "prefetch_started"
        assert EventType.TOKEN.value == "token"
        assert EventType.CHECKPOINT_FINISHED.value == "checkpoint_finished"


# =============================================================================
# Mock Generator for Testing
# =============================================================================


class MockDualPathGenerator:
    """Mock generator that produces test tokens."""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None = None,
    ) -> None:
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self._tokens = ["Hello", " world", "!", " This", " is", " a", " test", "."]

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """Generate mock tokens."""
        for token in self._tokens:
            yield {"type": "token", "text": token}
            await asyncio.sleep(0.001)  # Simulate async generation

        # Yield metrics at end
        yield {
            "type": "metrics",
            "metrics": {
                "tokens_per_second": 100.0,
                "total_tokens": len(self._tokens),
            },
        }


class MockFailingGenerator:
    """Mock generator that fails."""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None = None,
    ) -> None:
        self.base_model_path = base_model_path

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """Fail immediately."""
        raise RuntimeError("Model load failed")
        yield  # Make it a generator


# =============================================================================
# Integration Tests
# =============================================================================


class TestCheckpointComparisonCoordinatorFlow:
    """Tests for the full comparison flow."""

    @pytest.mark.asyncio
    async def test_compare_single_checkpoint(self) -> None:
        """Single checkpoint comparison should yield expected events."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Should have: prefetch_started, prefetch_finished, checkpoint_started, tokens..., checkpoint_finished
        event_types = [e.type for e in events]

        assert EventType.PREFETCH_STARTED in event_types
        assert EventType.PREFETCH_FINISHED in event_types
        assert EventType.CHECKPOINT_STARTED in event_types
        assert EventType.TOKEN in event_types
        assert EventType.CHECKPOINT_FINISHED in event_types

    @pytest.mark.asyncio
    async def test_compare_multiple_checkpoints(self) -> None:
        """Multiple checkpoint comparison should process all checkpoints."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        checkpoints = ["/test/model1", "/test/model2"]

        events = []
        async for event in coordinator.compare(
            checkpoints=checkpoints,
            prompt="Hello",
        ):
            events.append(event)

        # Count finished events - should have 2 (one per checkpoint)
        finished_events = [e for e in events if e.type == EventType.CHECKPOINT_FINISHED]
        assert len(finished_events) == 2

        # Each finished event should have a result
        for event in finished_events:
            assert event.result is not None
            assert event.result.response != ""

    @pytest.mark.asyncio
    async def test_compare_collects_tokens(self) -> None:
        """Comparison should collect tokens into response."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Find finished event
        finished = next(e for e in events if e.type == EventType.CHECKPOINT_FINISHED)

        # Response should be concatenation of all tokens
        assert finished.result is not None
        assert finished.result.response == "Hello world! This is a test."

    @pytest.mark.asyncio
    async def test_compare_collects_metrics(self) -> None:
        """Comparison should collect metrics from generator."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Find finished event
        finished = next(e for e in events if e.type == EventType.CHECKPOINT_FINISHED)

        # Metrics should be captured
        assert finished.result is not None
        assert finished.result.metrics is not None
        assert finished.result.metrics["tokens_per_second"] == 100.0

    @pytest.mark.asyncio
    async def test_compare_handles_failures(self) -> None:
        """Comparison should yield failure events on errors."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockFailingGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Should have a failure event
        failed_events = [e for e in events if e.type == EventType.CHECKPOINT_FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].error is not None
        assert "Model load failed" in failed_events[0].error


class TestCheckpointComparisonEventOrder:
    """Tests for event ordering guarantees."""

    @pytest.mark.asyncio
    async def test_prefetch_before_generation(self) -> None:
        """Prefetch events should come before generation events."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Find indices
        prefetch_finished_idx = next(
            i for i, e in enumerate(events) if e.type == EventType.PREFETCH_FINISHED
        )
        checkpoint_started_idx = next(
            i for i, e in enumerate(events) if e.type == EventType.CHECKPOINT_STARTED
        )

        assert prefetch_finished_idx < checkpoint_started_idx

    @pytest.mark.asyncio
    async def test_tokens_between_start_and_finish(self) -> None:
        """Token events should come between checkpoint start and finish."""
        coordinator = CheckpointComparisonCoordinator(
            generator_cls=MockDualPathGenerator
        )

        events = []
        async for event in coordinator.compare(
            checkpoints=["/test/model"],
            prompt="Hello",
        ):
            events.append(event)

        # Find indices
        started_idx = next(
            i for i, e in enumerate(events) if e.type == EventType.CHECKPOINT_STARTED
        )
        finished_idx = next(
            i for i, e in enumerate(events) if e.type == EventType.CHECKPOINT_FINISHED
        )
        token_indices = [
            i for i, e in enumerate(events) if e.type == EventType.TOKEN
        ]

        # All tokens should be between start and finish
        for token_idx in token_indices:
            assert started_idx < token_idx < finished_idx
