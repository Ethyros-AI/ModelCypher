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

"""Aggregate, locally-derived analytics over a window of AgentTrace records.

Agent Cypher dashboards use this type to surface only metrics that are
directly derivable from persisted traces (no cloud calls, no "magic" scoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


from modelcypher.core.domain.agents.agent_trace import TraceKind, TraceStatus
from modelcypher.core.domain.agents.agent_eval_suite_engine import AgentActionKind


@dataclass(frozen=True)
class MessageCount:
    """A message with its occurrence count."""

    message: str
    """The message text."""

    count: int
    """Number of occurrences."""

    @property
    def id(self) -> str:
        """Identifier for the message (same as message)."""
        return self.message


@dataclass(frozen=True)
class EntropyBucket:
    """Statistics for entropy values in a bucket."""

    count: int
    """Number of samples in bucket."""

    average: float | None = None
    """Average entropy value."""

    min: float | None = None
    """Minimum entropy value."""

    max: float | None = None
    """Maximum entropy value."""

    @property
    def is_empty(self) -> bool:
        """Check if bucket is empty."""
        return self.count == 0

    @classmethod
    def empty(cls) -> EntropyBucket:
        """Create an empty bucket."""
        return cls(count=0)


@dataclass(frozen=True)
class EntropyBuckets:
    """Entropy statistics grouped by action compliance."""

    valid_action: EntropyBucket
    """Entropy for valid actions."""

    invalid_action: EntropyBucket
    """Entropy for invalid actions."""

    unvalidated_action: EntropyBucket
    """Entropy for unvalidated actions."""

    no_action: EntropyBucket
    """Entropy for responses with no action."""

    @classmethod
    def empty(cls) -> EntropyBuckets:
        """Create empty buckets."""
        return cls(
            valid_action=EntropyBucket.empty(),
            invalid_action=EntropyBucket.empty(),
            unvalidated_action=EntropyBucket.empty(),
            no_action=EntropyBucket.empty(),
        )


@dataclass(frozen=True)
class ActionCompliance:
    """Action compliance statistics."""

    decoded_actions: int
    """Total number of decoded actions."""

    no_action: int
    """Count of responses with no action."""

    validations_recorded: int
    """Number of validations recorded."""

    valid_actions: int
    """Number of valid actions."""

    invalid_actions: int
    """Number of invalid actions."""

    unvalidated_actions: int
    """Number of unvalidated actions."""

    actions_with_warnings: int
    """Number of actions with warnings."""

    kinds: dict[AgentActionKind, int] = field(default_factory=dict)
    """Action counts by kind."""

    top_errors: list[MessageCount] = field(default_factory=list)
    """Top error messages."""

    top_warnings: list[MessageCount] = field(default_factory=list)
    """Top warning messages."""

    @classmethod
    def empty(cls) -> ActionCompliance:
        """Create empty compliance stats."""
        return cls(
            decoded_actions=0,
            no_action=0,
            validations_recorded=0,
            valid_actions=0,
            invalid_actions=0,
            unvalidated_actions=0,
            actions_with_warnings=0,
            kinds={},
            top_errors=[],
            top_warnings=[],
        )


@dataclass(frozen=True)
class AgentTraceAnalytics:
    """Aggregate, locally-derived analytics over a window of AgentTrace records.

    Agent Cypher dashboards use this type to surface only metrics that are
    directly derivable from persisted traces (no cloud calls, no "magic" scoring).
    """

    computed_at: datetime
    """When these analytics were computed."""

    requested_trace_count: int
    """Number of traces requested."""

    loaded_trace_count: int
    """Number of traces actually loaded."""

    oldest_started_at: datetime | None = None
    """Oldest trace start time."""

    newest_started_at: datetime | None = None
    """Newest trace start time."""

    kinds: dict[TraceKind, int] = field(default_factory=dict)
    """Trace counts by kind."""

    statuses: dict[TraceStatus, int] = field(default_factory=dict)
    """Trace counts by status."""

    intervention_count: int = 0
    """Number of safety interventions."""

    action_compliance: ActionCompliance = field(default_factory=ActionCompliance.empty)
    """Action compliance statistics."""

    entropy_by_compliance: EntropyBuckets = field(default_factory=EntropyBuckets.empty)
    """Entropy statistics grouped by compliance."""

    issues: list[str] = field(default_factory=list)
    """Any issues encountered during analysis."""

    @classmethod
    def empty(cls, requested_count: int = 0) -> AgentTraceAnalytics:
        """Create empty analytics.

        Args:
            requested_count: Number of traces that were requested.

        Returns:
            Empty analytics instance.
        """
        return cls(
            computed_at=datetime.now(),
            requested_trace_count=requested_count,
            loaded_trace_count=0,
        )

    @classmethod
    def from_traces(cls, traces: list, requested_count: int | None = None) -> "AgentTraceAnalytics":
        """Compute analytics from a list of traces.

        Args:
            traces: List of AgentTrace objects to analyze.
            requested_count: Number of traces that were requested.
                            Defaults to len(traces).

        Returns:
            Analytics computed from the traces.
        """
        from modelcypher.core.domain.agents.agent_trace import AgentTrace
        
        if not traces:
            return cls.empty(requested_count or 0)
        
        req_count = requested_count if requested_count is not None else len(traces)
        
        # Compute date range
        oldest = None
        newest = None
        kinds: dict[TraceKind, int] = {}
        statuses: dict[TraceStatus, int] = {}
        intervention_count = 0
        
        for trace in traces:
            if not isinstance(trace, AgentTrace):
                continue
            
            # Date range
            if trace.started_at:
                if oldest is None or trace.started_at < oldest:
                    oldest = trace.started_at
                if newest is None or trace.started_at > newest:
                    newest = trace.started_at
            
            # Kinds
            if trace.kind:
                kinds[trace.kind] = kinds.get(trace.kind, 0) + 1
            
            # Statuses
            if trace.status:
                statuses[trace.status] = statuses.get(trace.status, 0) + 1
            
            # Interventions
            if trace.intervention_required:
                intervention_count += 1
        
        return cls(
            computed_at=datetime.now(),
            requested_trace_count=req_count,
            loaded_trace_count=len(traces),
            oldest_started_at=oldest,
            newest_started_at=newest,
            kinds=kinds,
            statuses=statuses,
            intervention_count=intervention_count,
            action_compliance=ActionCompliance.empty(),
            entropy_by_compliance=EntropyBuckets.empty(),
        )

