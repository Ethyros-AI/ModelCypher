from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class QuerySubtype(str, Enum):
    infer = "infer"
    generate = "generate"
    classify = "classify"
    extract = "extract"
    summarize = "summarize"
    translate = "translate"
    chat = "chat"


class InputFormat(str, Enum):
    text = "text"
    html = "html"
    markdown = "markdown"
    json = "json"
    csv = "csv"
    code = "code"
    image = "image"
    audio = "audio"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"
    json_schema = "json_schema"
    markdown = "markdown"
    code = "code"
    structured = "structured"


class TaskType(str, Enum):
    scrape = "scrape"
    parse = "parse"
    transform = "transform"
    analyze = "analyze"
    generate = "generate"
    validate = "validate"
    classify = "classify"


class SystemEvent(str, Enum):
    memory_pressure = "memoryPressure"
    thermal_throttle = "thermalThrottle"
    gpu_available = "gpuAvailable"
    gpu_unavailable = "gpuUnavailable"
    model_loaded = "modelLoaded"
    model_unloaded = "modelUnloaded"
    adapter_registered = "adapterRegistered"
    adapter_unregistered = "adapterUnregistered"
    batch_start = "batchStart"
    batch_complete = "batchComplete"
    volatility_update = "volatilityUpdate"
    circuit_breaker_tripped = "circuitBreakerTripped"
    circuit_breaker_reset = "circuitBreakerReset"
    model_state_changed = "modelStateChanged"
    distress_detected = "distressDetected"
    routing_decision = "routingDecision"
    skill_execution_complete = "skillExecutionComplete"
    dpo_event = "dpoEvent"
    dpo_batch_ready = "dpoBatchReady"
    failure_captured = "failureCaptured"
    synthesis_started = "synthesisStarted"
    synthesis_completed = "synthesisCompleted"
    self_improvement_job_started = "selfImprovementJobStarted"
    self_improvement_job_completed = "selfImprovementJobCompleted"
    quality_gate_evaluated = "qualityGateEvaluated"
    adapter_promoted = "adapterPromoted"
    adapter_rejected = "adapterRejected"
    adapter_rolled_back = "adapterRolledBack"
    adapter_anomaly_detected = "adapterAnomalyDetected"
    security_circuit_breaker_tripped = "securityCircuitBreakerTripped"
    security_scan_complete = "securityScanComplete"
    metric_sample = "metricSample"
    entropy_drop_detected = "entropyDropDetected"
    intervention_executed = "interventionExecuted"
    intervention_pending = "interventionPending"
    intervention_resolved = "interventionResolved"
    intervention_terminated = "interventionTerminated"
    error = "error"


class Priority(int, Enum):
    low = 0
    normal = 1
    high = 2
    critical = 3


@dataclass(frozen=True)
class PayloadValue:
    kind: str
    value: object | None = None

    @staticmethod
    def string(value: str) -> "PayloadValue":
        return PayloadValue(kind="string", value=value)

    @staticmethod
    def int(value: int) -> "PayloadValue":
        return PayloadValue(kind="int", value=value)

    @staticmethod
    def double(value: float) -> "PayloadValue":
        return PayloadValue(kind="double", value=value)

    @staticmethod
    def bool(value: bool) -> "PayloadValue":
        return PayloadValue(kind="bool", value=value)

    @staticmethod
    def array(value: list["PayloadValue"]) -> "PayloadValue":
        return PayloadValue(kind="array", value=value)

    @staticmethod
    def dict(value: dict[str, "PayloadValue"]) -> "PayloadValue":
        return PayloadValue(kind="dict", value=value)

    @staticmethod
    def null() -> "PayloadValue":
        return PayloadValue(kind="null", value=None)

    @property
    def string_value(self) -> Optional[str]:
        return self.value if self.kind == "string" else None

    @property
    def int_value(self) -> Optional[int]:
        return self.value if self.kind == "int" else None

    @property
    def double_value(self) -> Optional[float]:
        if self.kind == "double":
            return float(self.value) if self.value is not None else None
        if self.kind == "int":
            return float(self.value) if self.value is not None else None
        return None

    @property
    def bool_value(self) -> Optional[bool]:
        return self.value if self.kind == "bool" else None


@dataclass(frozen=True)
class SignalType:
    namespace: str
    value: str
    custom_name: Optional[str] = None

    @property
    def capability_string(self) -> str:
        if self.namespace == "custom" and self.custom_name:
            return f"{self.value}:{self.custom_name}"
        return f"{self.namespace}:{self.value}"

    @staticmethod
    def query(subtype: QuerySubtype) -> "SignalType":
        return SignalType(namespace="query", value=subtype.value)

    @staticmethod
    def input_format(fmt: InputFormat) -> "SignalType":
        return SignalType(namespace="input", value=fmt.value)

    @staticmethod
    def output_format(fmt: OutputFormat) -> "SignalType":
        return SignalType(namespace="output", value=fmt.value)

    @staticmethod
    def domain(name: str) -> "SignalType":
        return SignalType(namespace="domain", value=name)

    @staticmethod
    def task(task_type: TaskType) -> "SignalType":
        return SignalType(namespace="task", value=task_type.value)

    @staticmethod
    def system_event(event: SystemEvent) -> "SignalType":
        return SignalType(namespace="system", value=event.value)

    @staticmethod
    def custom(namespace: str, name: str) -> "SignalType":
        return SignalType(namespace=namespace, value=namespace, custom_name=name)


@dataclass(frozen=True)
class Signal:
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    type: SignalType = field(default_factory=lambda: SignalType.custom("custom", "unknown"))
    payload: dict[str, PayloadValue] = field(default_factory=dict)
    correlation_id: Optional[UUID] = None
    priority: Priority = Priority.normal
    ttl: Optional[float] = None
    source: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl

    @staticmethod
    def query(
        subtype: QuerySubtype,
        text: str,
        correlation_id: Optional[UUID] = None,
    ) -> "Signal":
        return Signal(
            type=SignalType.query(subtype),
            payload={"text": PayloadValue.string(text)},
            correlation_id=correlation_id,
        )

    @staticmethod
    def domain(
        name: str,
        action: Optional[str] = None,
        payload: Optional[dict[str, PayloadValue]] = None,
    ) -> "Signal":
        final_payload = dict(payload or {})
        if action:
            final_payload["action"] = PayloadValue.string(action)
        return Signal(type=SignalType.domain(name), payload=final_payload)

    @staticmethod
    def system(
        event: SystemEvent,
        payload: Optional[dict[str, PayloadValue]] = None,
        priority: Priority = Priority.normal,
    ) -> "Signal":
        return Signal(
            type=SignalType.system_event(event),
            payload=dict(payload or {}),
            priority=priority,
        )

    @staticmethod
    def task(
        task_type: TaskType,
        input_format: Optional[InputFormat] = None,
        output_format: Optional[OutputFormat] = None,
        payload: Optional[dict[str, PayloadValue]] = None,
    ) -> "Signal":
        final_payload = dict(payload or {})
        if input_format:
            final_payload["input_format"] = PayloadValue.string(input_format.value)
        if output_format:
            final_payload["output_format"] = PayloadValue.string(output_format.value)
        return Signal(type=SignalType.task(task_type), payload=final_payload)


def normalize_capability_value(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def matches_capability(signal: Signal, capability_pattern: str) -> bool:
    parts = capability_pattern.split(":", 1)
    if len(parts) != 2:
        return False
    namespace = normalize_capability_value(parts[0])
    value = normalize_capability_value(parts[1])

    signal_parts = signal.type.capability_string.split(":", 1)
    if len(signal_parts) != 2:
        return False
    signal_namespace = normalize_capability_value(signal_parts[0])
    signal_value = normalize_capability_value(signal_parts[1])

    if namespace != signal_namespace:
        return False
    if value == "*" or value == signal_value:
        return True
    return False
