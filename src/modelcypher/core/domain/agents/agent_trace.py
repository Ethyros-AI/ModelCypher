"""
Agent Trace for Agentic Workflow Observability.

Ported 1:1 from the reference Swift implementation.

A persisted, privacy-preserving trace of a single agent-related execution.
Treats traces as structured trajectories with hashed payload summaries 
(not raw prompts/responses) so traces can be retained locally for debugging, 
evaluation, and distillation without leaking sensitive content.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


# =============================================================================
# Trace Kind
# =============================================================================


class TraceKind(str, Enum):
    """Type of trace."""

    inference = "inference"
    agent_pipeline = "agent_pipeline"


# =============================================================================
# Trace Status
# =============================================================================


class TraceStatus(str, Enum):
    """Execution status."""

    success = "success"
    cancelled = "cancelled"
    failed = "failed"


# =============================================================================
# Payload Digest
# =============================================================================


@dataclass(frozen=True)
class PayloadDigest:
    """
    Privacy-preserving digest of payload content.

    Stores hash instead of raw content for security.
    """

    sha256: str
    character_count: int
    byte_count: int
    preview: str | None = None

    @staticmethod
    def hashing(
        text: str,
        preview_length: int | None = None,
    ) -> "PayloadDigest":
        """
        Create a digest by hashing the text.

        Args:
            text: The text to hash.
            preview_length: Optional length of preview to include.

        Returns:
            PayloadDigest with SHA256 hash and metadata.
        """
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        preview = text[:preview_length] if preview_length else None

        return PayloadDigest(
            sha256=digest,
            character_count=len(text),
            byte_count=len(data),
            preview=preview,
        )

    @staticmethod
    def hashing_with_preview(text: str, preview: str | None) -> "PayloadDigest":
        """Create a digest with explicit preview."""
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()

        return PayloadDigest(
            sha256=digest,
            character_count=len(text),
            byte_count=len(data),
            preview=preview,
        )


# =============================================================================
# Schema Validation
# =============================================================================


@dataclass(frozen=True)
class SchemaValidation:
    """Result of validating output against an action schema."""

    schema_id: str
    schema_version: int
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Inference Metrics
# =============================================================================


@dataclass(frozen=True)
class InferenceMetrics:
    """Performance metrics from inference."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    time_to_first_token_ms: float | None = None
    tokens_per_second: float | None = None
    latency_ms: float | None = None


# =============================================================================
# Trace Span
# =============================================================================


@dataclass(frozen=True)
class TraceSpan:
    """
    Individual span within a trace (OpenTelemetry-compatible).

    Represents a unit of work within the agent execution.
    """

    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: datetime
    end_time: datetime | None = None
    status: TraceStatus = TraceStatus.success
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000


# =============================================================================
# Trace Source
# =============================================================================


@dataclass(frozen=True)
class TraceSource:
    """Provenance metadata for imported traces."""

    provider: str  # e.g., "monocle", "opentelemetry", "langfuse"
    trace_id: str | None = None
    project_id: str | None = None
    imported_at: datetime = field(default_factory=datetime.utcnow)
    original_format: str | None = None


# =============================================================================
# Trace Summary
# =============================================================================


@dataclass(frozen=True)
class TraceSummary:
    """Lightweight summary of a trace for listing."""

    id: UUID
    kind: TraceKind
    status: TraceStatus
    started_at: datetime
    duration_ms: int | None
    base_model_id: str | None
    adapter_id: UUID | None
    average_entropy: float | None
    tokens_generated: int | None


# =============================================================================
# Agent Trace
# =============================================================================


@dataclass
class AgentTrace:
    """
    A persisted, privacy-preserving trace of a single agent-related execution.

    Traces are structured trajectories with hashed payload summaries for 
    privacy-preserving debugging, evaluation, and distillation.

    Usage:
        trace = AgentTrace.start(
            kind=TraceKind.inference,
            input_text="What is the weather?",
        )
        # ... perform inference ...
        trace.complete(
            output_text="The weather is sunny.",
            status=TraceStatus.success,
        )
    """

    id: UUID
    kind: TraceKind
    started_at: datetime
    input_digest: PayloadDigest
    status: TraceStatus = TraceStatus.success
    completed_at: datetime | None = None
    output_digest: PayloadDigest | None = None

    # Model context
    base_model_id: str | None = None
    adapter_id: UUID | None = None

    # Performance metrics
    inference_metrics: InferenceMetrics | None = None

    # Entropy monitoring
    average_entropy: float | None = None
    max_entropy: float | None = None

    # Schema validation (if output is structured)
    action_schema_validation: SchemaValidation | None = None

    # Observability spans
    spans: list[TraceSpan] = field(default_factory=list)

    # Import provenance
    source: TraceSource | None = None

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def start(
        kind: TraceKind,
        input_text: str,
        preview_length: int = 100,
        base_model_id: str | None = None,
        adapter_id: UUID | None = None,
    ) -> "AgentTrace":
        """
        Start a new trace.

        Args:
            kind: Type of trace (inference or agent_pipeline).
            input_text: The input prompt/text.
            preview_length: Length of preview to store.
            base_model_id: Optional model identifier.
            adapter_id: Optional adapter identifier.

        Returns:
            A new AgentTrace in started state.
        """
        return AgentTrace(
            id=uuid4(),
            kind=kind,
            started_at=datetime.utcnow(),
            input_digest=PayloadDigest.hashing(input_text, preview_length),
            base_model_id=base_model_id,
            adapter_id=adapter_id,
        )

    def complete(
        self,
        output_text: str | None = None,
        status: TraceStatus = TraceStatus.success,
        preview_length: int = 100,
        inference_metrics: InferenceMetrics | None = None,
        average_entropy: float | None = None,
    ) -> None:
        """
        Complete the trace with output.

        Args:
            output_text: The generated output text.
            status: Completion status.
            preview_length: Length of output preview to store.
            inference_metrics: Performance metrics.
            average_entropy: Average entropy during generation.
        """
        self.completed_at = datetime.utcnow()
        self.status = status

        if output_text is not None:
            self.output_digest = PayloadDigest.hashing(output_text, preview_length)

        if inference_metrics is not None:
            self.inference_metrics = inference_metrics

        if average_entropy is not None:
            self.average_entropy = average_entropy

    def fail(self, error_message: str | None = None) -> None:
        """Mark the trace as failed."""
        self.completed_at = datetime.utcnow()
        self.status = TraceStatus.failed
        if error_message:
            self.metadata["error"] = error_message

    def cancel(self) -> None:
        """Mark the trace as cancelled."""
        self.completed_at = datetime.utcnow()
        self.status = TraceStatus.cancelled

    def add_span(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """
        Add a new span to the trace.

        Args:
            operation_name: Name of the operation.
            parent_span_id: Optional parent span ID.
            attributes: Optional span attributes.

        Returns:
            The created span.
        """
        span = TraceSpan(
            span_id=uuid4().hex[:16],
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    @property
    def duration_ms(self) -> int | None:
        """Duration in milliseconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return int(max(0, delta.total_seconds() * 1000))

    @property
    def summary(self) -> TraceSummary:
        """Get a lightweight summary of this trace."""
        return TraceSummary(
            id=self.id,
            kind=self.kind,
            status=self.status,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            base_model_id=self.base_model_id,
            adapter_id=self.adapter_id,
            average_entropy=self.average_entropy,
            tokens_generated=(
                self.inference_metrics.output_tokens
                if self.inference_metrics
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/export."""
        return {
            "id": str(self.id),
            "kind": self.kind.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "input_digest": {
                "sha256": self.input_digest.sha256,
                "character_count": self.input_digest.character_count,
                "byte_count": self.input_digest.byte_count,
                "preview": self.input_digest.preview,
            },
            "output_digest": (
                {
                    "sha256": self.output_digest.sha256,
                    "character_count": self.output_digest.character_count,
                    "byte_count": self.output_digest.byte_count,
                    "preview": self.output_digest.preview,
                }
                if self.output_digest
                else None
            ),
            "base_model_id": self.base_model_id,
            "adapter_id": str(self.adapter_id) if self.adapter_id else None,
            "average_entropy": self.average_entropy,
            "inference_metrics": (
                {
                    "input_tokens": self.inference_metrics.input_tokens,
                    "output_tokens": self.inference_metrics.output_tokens,
                    "total_tokens": self.inference_metrics.total_tokens,
                    "tokens_per_second": self.inference_metrics.tokens_per_second,
                    "latency_ms": self.inference_metrics.latency_ms,
                }
                if self.inference_metrics
                else None
            ),
            "span_count": len(self.spans),
            "metadata": self.metadata,
        }


# =============================================================================
# Trace Store
# =============================================================================


class TraceStore:
    """
    In-memory trace store with capacity limits.

    Provides storage and querying for agent traces.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._traces: dict[UUID, AgentTrace] = {}
        self._order: list[UUID] = []  # For FIFO eviction

    def add(self, trace: AgentTrace) -> None:
        """Add a trace to the store."""
        if trace.id in self._traces:
            # Update existing
            self._traces[trace.id] = trace
            return

        # Evict oldest if at capacity
        while len(self._traces) >= self.capacity and self._order:
            oldest_id = self._order.pop(0)
            self._traces.pop(oldest_id, None)

        self._traces[trace.id] = trace
        self._order.append(trace.id)

    def get(self, trace_id: UUID) -> AgentTrace | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def list_summaries(
        self,
        kind: TraceKind | None = None,
        status: TraceStatus | None = None,
        limit: int = 100,
    ) -> list[TraceSummary]:
        """
        List trace summaries with optional filtering.

        Args:
            kind: Filter by trace kind.
            status: Filter by status.
            limit: Maximum results.

        Returns:
            List of trace summaries.
        """
        traces = list(self._traces.values())

        if kind is not None:
            traces = [t for t in traces if t.kind == kind]
        if status is not None:
            traces = [t for t in traces if t.status == status]

        # Sort by started_at descending
        traces.sort(key=lambda t: t.started_at, reverse=True)

        return [t.summary for t in traces[:limit]]

    def query_by_model(self, model_id: str, limit: int = 100) -> list[TraceSummary]:
        """Query traces by model ID."""
        traces = [t for t in self._traces.values() if t.base_model_id == model_id]
        traces.sort(key=lambda t: t.started_at, reverse=True)
        return [t.summary for t in traces[:limit]]

    def query_by_adapter(self, adapter_id: UUID, limit: int = 100) -> list[TraceSummary]:
        """Query traces by adapter ID."""
        traces = [t for t in self._traces.values() if t.adapter_id == adapter_id]
        traces.sort(key=lambda t: t.started_at, reverse=True)
        return [t.summary for t in traces[:limit]]

    def query_failures(self, limit: int = 100) -> list[TraceSummary]:
        """Query failed traces."""
        return self.list_summaries(status=TraceStatus.failed, limit=limit)

    def clear(self) -> None:
        """Clear all traces."""
        self._traces.clear()
        self._order.clear()

    @property
    def count(self) -> int:
        """Number of traces in store."""
        return len(self._traces)

    def compute_statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics over traces."""
        if not self._traces:
            return {
                "total": 0,
                "by_kind": {},
                "by_status": {},
                "avg_duration_ms": None,
                "avg_tokens": None,
                "avg_entropy": None,
            }

        traces = list(self._traces.values())

        by_kind = {}
        for kind in TraceKind:
            by_kind[kind.value] = sum(1 for t in traces if t.kind == kind)

        by_status = {}
        for status in TraceStatus:
            by_status[status.value] = sum(1 for t in traces if t.status == status)

        durations = [t.duration_ms for t in traces if t.duration_ms is not None]
        tokens = [
            t.inference_metrics.output_tokens
            for t in traces
            if t.inference_metrics and t.inference_metrics.output_tokens
        ]
        entropies = [t.average_entropy for t in traces if t.average_entropy is not None]

        return {
            "total": len(traces),
            "by_kind": by_kind,
            "by_status": by_status,
            "avg_duration_ms": sum(durations) / len(durations) if durations else None,
            "avg_tokens": sum(tokens) / len(tokens) if tokens else None,
            "avg_entropy": sum(entropies) / len(entropies) if entropies else None,
        }
