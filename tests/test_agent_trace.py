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
Tests for AgentTrace observability module.

This tests the privacy-preserving trace functionality for agentic workflows.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from uuid import uuid4

from modelcypher.core.domain.agents.agent_trace import (
    AgentTrace,
    InferenceMetrics,
    PayloadDigest,
    SchemaValidation,
    TraceKind,
    TraceSource,
    TraceSpan,
    TraceStatus,
    TraceStore,
    TraceSummary,
)

# =============================================================================
# PayloadDigest Tests
# =============================================================================


class TestPayloadDigest:
    """Tests for PayloadDigest."""

    def test_hashing_basic(self) -> None:
        """Test basic hashing functionality."""
        text = "Hello, World!"
        digest = PayloadDigest.hashing(text)

        assert len(digest.sha256) == 64  # SHA256 hex length
        assert digest.character_count == 13
        assert digest.byte_count == 13
        assert digest.preview is None

    def test_hashing_with_preview(self) -> None:
        """Test hashing with preview."""
        text = "This is a long text that should be truncated"
        digest = PayloadDigest.hashing(text, preview_length=10)

        assert digest.preview == "This is a "
        assert digest.character_count == len(text)

    def test_hashing_unicode(self) -> None:
        """Test hashing with unicode characters."""
        text = "Hello 🌍 World! αβγ"
        digest = PayloadDigest.hashing(text)

        assert digest.character_count == len(text)  # Python len counts code points
        assert digest.byte_count > digest.character_count  # UTF-8 encoding expands

    def test_hashing_empty(self) -> None:
        """Test hashing empty string."""
        digest = PayloadDigest.hashing("")
        assert digest.character_count == 0
        assert digest.byte_count == 0
        # SHA256 of empty string
        expected_hash = hashlib.sha256(b"").hexdigest()
        assert digest.sha256 == expected_hash

    def test_hashing_deterministic(self) -> None:
        """Test that hashing is deterministic."""
        text = "Same input"
        digest1 = PayloadDigest.hashing(text)
        digest2 = PayloadDigest.hashing(text)
        assert digest1.sha256 == digest2.sha256

    def test_hashing_with_preview_explicit(self) -> None:
        """Test hashing with explicit preview."""
        text = "Full text content"
        preview = "Custom preview..."
        digest = PayloadDigest.hashing_with_preview(text, preview)

        assert digest.preview == preview
        assert digest.character_count == len(text)


# =============================================================================
# SchemaValidation Tests
# =============================================================================


class TestSchemaValidation:
    """Tests for SchemaValidation."""

    def test_valid_schema(self) -> None:
        """Test valid schema validation result."""
        validation = SchemaValidation(
            schema_id="action_v1",
            schema_version=1,
            is_valid=True,
            errors=[],
            warnings=["Consider adding description"],
        )
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 1

    def test_invalid_schema(self) -> None:
        """Test invalid schema validation result."""
        validation = SchemaValidation(
            schema_id="action_v1",
            schema_version=1,
            is_valid=False,
            errors=["Missing required field: action_type"],
            warnings=[],
        )
        assert validation.is_valid is False
        assert len(validation.errors) == 1


# =============================================================================
# InferenceMetrics Tests
# =============================================================================


class TestInferenceMetrics:
    """Tests for InferenceMetrics."""

    def test_create(self) -> None:
        """Test creating inference metrics."""
        metrics = InferenceMetrics(
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            time_to_first_token_ms=250.5,
            tokens_per_second=45.2,
            latency_ms=2200.0,
        )
        assert metrics.input_tokens == 50
        assert metrics.output_tokens == 100
        assert metrics.total_tokens == 150
        assert metrics.tokens_per_second == 45.2


# =============================================================================
# TraceSpan Tests
# =============================================================================


class TestTraceSpan:
    """Tests for TraceSpan."""

    def test_create_span(self) -> None:
        """Test creating a span."""
        start = datetime.utcnow()
        span = TraceSpan(
            span_id="abc123",
            parent_span_id=None,
            operation_name="model_forward",
            start_time=start,
            attributes={"layer_count": 32},
        )
        assert span.span_id == "abc123"
        assert span.operation_name == "model_forward"
        assert span.duration_ms is None  # Not ended

    def test_span_duration(self) -> None:
        """Test span duration calculation."""
        start = datetime.utcnow()
        end = start + timedelta(milliseconds=500)
        span = TraceSpan(
            span_id="abc123",
            parent_span_id=None,
            operation_name="tokenize",
            start_time=start,
            end_time=end,
        )
        assert span.duration_ms == 500.0


# =============================================================================
# TraceSource Tests
# =============================================================================


class TestTraceSource:
    """Tests for TraceSource."""

    def test_create_source(self) -> None:
        """Test creating a trace source."""
        source = TraceSource(
            provider="monocle",
            trace_id="trace-123",
            project_id="my-project",
            original_format="otel",
        )
        assert source.provider == "monocle"
        assert source.trace_id == "trace-123"


# =============================================================================
# AgentTrace Tests
# =============================================================================


class TestAgentTrace:
    """Tests for AgentTrace."""

    def test_start_trace(self) -> None:
        """Test starting a new trace."""
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="What is the weather today?",
            base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        )

        assert trace.kind == TraceKind.inference
        assert trace.status == TraceStatus.success
        assert trace.completed_at is None
        assert trace.input_digest.character_count == 26
        assert trace.base_model_id == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_complete_trace(self) -> None:
        """Test completing a trace."""
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Hello",
        )

        time.sleep(0.01)  # Small delay for duration

        metrics = InferenceMetrics(
            input_tokens=5,
            output_tokens=20,
            total_tokens=25,
            tokens_per_second=50.0,
        )

        trace.complete(
            output_text="Hello! How can I help you today?",
            status=TraceStatus.success,
            inference_metrics=metrics,
            average_entropy=1.8,
        )

        assert trace.status == TraceStatus.success
        assert trace.completed_at is not None
        assert trace.output_digest is not None
        assert trace.average_entropy == 1.8
        assert trace.duration_ms is not None
        assert trace.duration_ms > 0

    def test_fail_trace(self) -> None:
        """Test failing a trace."""
        trace = AgentTrace.start(
            kind=TraceKind.agent_pipeline,
            input_text="Complex query",
        )

        trace.fail(error_message="Model timeout")

        assert trace.status == TraceStatus.failed
        assert trace.completed_at is not None
        assert trace.metadata["error"] == "Model timeout"

    def test_cancel_trace(self) -> None:
        """Test cancelling a trace."""
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Query",
        )

        trace.cancel()

        assert trace.status == TraceStatus.cancelled
        assert trace.completed_at is not None

    def test_add_span(self) -> None:
        """Test adding spans to a trace."""
        trace = AgentTrace.start(
            kind=TraceKind.agent_pipeline,
            input_text="Query",
        )

        span1 = trace.add_span("tokenize", attributes={"vocab_size": 32000})
        span2 = trace.add_span("forward_pass", parent_span_id=span1.span_id)

        assert len(trace.spans) == 2
        assert span2.parent_span_id == span1.span_id

    def test_summary(self) -> None:
        """Test getting trace summary."""
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Query",
            base_model_id="model-123",
        )

        summary = trace.summary

        assert isinstance(summary, TraceSummary)
        assert summary.id == trace.id
        assert summary.kind == TraceKind.inference
        assert summary.base_model_id == "model-123"

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Hello",
        )
        trace.complete(output_text="Hi there!")

        data = trace.to_dict()

        assert data["kind"] == "inference"
        assert data["status"] == "success"
        assert "input_digest" in data
        assert "output_digest" in data
        assert data["output_digest"]["character_count"] == 9

    def test_trace_with_adapter(self) -> None:
        """Test trace with adapter ID."""
        adapter_id = uuid4()
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Query",
            adapter_id=adapter_id,
        )

        assert trace.adapter_id == adapter_id


# =============================================================================
# TraceStore Tests
# =============================================================================


class TestTraceStore:
    """Tests for TraceStore."""

    def test_add_and_get(self) -> None:
        """Test adding and retrieving traces."""
        store = TraceStore(capacity=10)

        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Test",
        )
        store.add(trace)

        retrieved = store.get(trace.id)
        assert retrieved is not None
        assert retrieved.id == trace.id

    def test_capacity_eviction(self) -> None:
        """Test FIFO eviction when at capacity."""
        store = TraceStore(capacity=3)

        traces = []
        for i in range(5):
            trace = AgentTrace.start(
                kind=TraceKind.inference,
                input_text=f"Query {i}",
            )
            traces.append(trace)
            store.add(trace)

        # Oldest 2 should be evicted
        assert store.count == 3
        assert store.get(traces[0].id) is None
        assert store.get(traces[1].id) is None
        assert store.get(traces[2].id) is not None
        assert store.get(traces[4].id) is not None

    def test_list_summaries(self) -> None:
        """Test listing trace summaries."""
        store = TraceStore()

        for i in range(5):
            kind = TraceKind.inference if i % 2 == 0 else TraceKind.agent_pipeline
            trace = AgentTrace.start(
                kind=kind,
                input_text=f"Query {i}",
            )
            store.add(trace)

        all_summaries = store.list_summaries()
        assert len(all_summaries) == 5

        inference_only = store.list_summaries(kind=TraceKind.inference)
        assert len(inference_only) == 3

    def test_list_summaries_by_status(self) -> None:
        """Test filtering summaries by status."""
        store = TraceStore()

        # Add some successful traces
        for _ in range(3):
            trace = AgentTrace.start(kind=TraceKind.inference, input_text="Query")
            trace.complete(output_text="Response")
            store.add(trace)

        # Add some failed traces
        for _ in range(2):
            trace = AgentTrace.start(kind=TraceKind.inference, input_text="Query")
            trace.fail()
            store.add(trace)

        failures = store.query_failures()
        assert len(failures) == 2

        successes = store.list_summaries(status=TraceStatus.success)
        assert len(successes) == 3

    def test_query_by_model(self) -> None:
        """Test querying by model ID."""
        store = TraceStore()

        for i in range(5):
            model_id = "model-a" if i % 2 == 0 else "model-b"
            trace = AgentTrace.start(
                kind=TraceKind.inference,
                input_text=f"Query {i}",
                base_model_id=model_id,
            )
            store.add(trace)

        model_a_traces = store.query_by_model("model-a")
        assert len(model_a_traces) == 3

    def test_query_by_adapter(self) -> None:
        """Test querying by adapter ID."""
        store = TraceStore()

        adapter_a = uuid4()
        adapter_b = uuid4()

        for i in range(4):
            adapter_id = adapter_a if i < 2 else adapter_b
            trace = AgentTrace.start(
                kind=TraceKind.inference,
                input_text=f"Query {i}",
                adapter_id=adapter_id,
            )
            store.add(trace)

        adapter_a_traces = store.query_by_adapter(adapter_a)
        assert len(adapter_a_traces) == 2

    def test_clear(self) -> None:
        """Test clearing the store."""
        store = TraceStore()

        for i in range(5):
            trace = AgentTrace.start(kind=TraceKind.inference, input_text=f"Query {i}")
            store.add(trace)

        assert store.count == 5

        store.clear()

        assert store.count == 0

    def test_compute_statistics(self) -> None:
        """Test computing aggregate statistics."""
        store = TraceStore()

        # Add varied traces
        for i in range(5):
            trace = AgentTrace.start(
                kind=TraceKind.inference,
                input_text=f"Query {i}",
            )
            metrics = InferenceMetrics(
                input_tokens=10,
                output_tokens=20 + i,
                total_tokens=30 + i,
            )
            trace.complete(
                output_text=f"Response {i}",
                inference_metrics=metrics,
                average_entropy=1.5 + (i * 0.1),
            )
            store.add(trace)

        stats = store.compute_statistics()

        assert stats["total"] == 5
        assert stats["by_kind"]["inference"] == 5
        assert stats["by_status"]["success"] == 5
        assert stats["avg_tokens"] is not None
        assert stats["avg_entropy"] is not None

    def test_statistics_empty_store(self) -> None:
        """Test statistics on empty store."""
        store = TraceStore()
        stats = store.compute_statistics()

        assert stats["total"] == 0
        assert stats["avg_duration_ms"] is None
        assert stats["avg_tokens"] is None

    def test_update_existing_trace(self) -> None:
        """Test updating an existing trace in store."""
        store = TraceStore(capacity=10)

        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="Query",
        )
        store.add(trace)

        # Update the trace
        trace.complete(output_text="Response")
        store.add(trace)  # Re-add updated trace

        # Should still be 1 trace
        assert store.count == 1

        retrieved = store.get(trace.id)
        assert retrieved is not None
        assert retrieved.output_digest is not None


# =============================================================================
# TraceKind Tests
# =============================================================================


class TestTraceKind:
    """Tests for TraceKind enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert TraceKind.inference.value == "inference"
        assert TraceKind.agent_pipeline.value == "agent_pipeline"


# =============================================================================
# TraceStatus Tests
# =============================================================================


class TestTraceStatus:
    """Tests for TraceStatus enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert TraceStatus.success.value == "success"
        assert TraceStatus.cancelled.value == "cancelled"
        assert TraceStatus.failed.value == "failed"
