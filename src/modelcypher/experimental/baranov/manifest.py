"""Replication manifest schema and validation for Baranov tracks A/B/C.

EXPERIMENTAL: Not validated for production use.

The schema matches Section 9 of ``baranov_replication_protocol_2026_02.md``.
All dataclasses are frozen for immutability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    id: str
    quantization: str
    backend: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "quantization": self.quantization, "backend": self.backend}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> ModelInfo:
        return cls(id=data["id"], quantization=data["quantization"], backend=data["backend"])


@dataclass(frozen=True)
class CodeInfo:
    modelcypher_commit: str
    experiment_module_commit: str

    def as_dict(self) -> dict[str, str]:
        return {
            "modelcypher_commit": self.modelcypher_commit,
            "experiment_module_commit": self.experiment_module_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> CodeInfo:
        return cls(
            modelcypher_commit=data["modelcypher_commit"],
            experiment_module_commit=data["experiment_module_commit"],
        )


@dataclass(frozen=True)
class DataHashes:
    fact_pool_hash: str
    split_manifest_hash: str
    reference_corpus_hash: str

    def as_dict(self) -> dict[str, str]:
        return {
            "fact_pool_hash": self.fact_pool_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "reference_corpus_hash": self.reference_corpus_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> DataHashes:
        return cls(
            fact_pool_hash=data["fact_pool_hash"],
            split_manifest_hash=data["split_manifest_hash"],
            reference_corpus_hash=data["reference_corpus_hash"],
        )


@dataclass(frozen=True)
class ControlFlags:
    base_control: bool
    lora_only_control: bool
    edit_only_control: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "base_control": self.base_control,
            "lora_only_control": self.lora_only_control,
            "edit_only_control": self.edit_only_control,
        }

    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> ControlFlags:
        return cls(
            base_control=data["base_control"],
            lora_only_control=data["lora_only_control"],
            edit_only_control=data["edit_only_control"],
        )


@dataclass(frozen=True)
class PreRegisteredDecision:
    criteria_version: str
    outcome: str  # "pass" | "fail" | "inconclusive"
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "criteria_version": self.criteria_version,
            "outcome": self.outcome,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PreRegisteredDecision:
        return cls(
            criteria_version=data["criteria_version"],
            outcome=data["outcome"],
            reason=data["reason"],
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationManifest:
    """Schema matching Section 9 of the replication protocol.

    Captures full provenance for a single replication run.
    """

    run_id: str
    track: str  # "A" | "B" | "C"
    timestamp_utc: str  # ISO-8601
    model: ModelInfo
    code: CodeInfo
    data: DataHashes
    controls: ControlFlags
    metrics: tuple[tuple[str, float], ...]
    pre_registered_decision: PreRegisteredDecision

    @property
    def metrics_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict (read-only convenience)."""
        return dict(self.metrics)

    @classmethod
    def from_metrics_dict(
        cls,
        *,
        run_id: str,
        track: str,
        timestamp_utc: str,
        model: ModelInfo,
        code: CodeInfo,
        data: DataHashes,
        controls: ControlFlags,
        metrics_dict: dict[str, float],
        pre_registered_decision: PreRegisteredDecision,
    ) -> ReplicationManifest:
        """Construct from a dict of metrics (convenience)."""
        return cls(
            run_id=run_id,
            track=track,
            timestamp_utc=timestamp_utc,
            model=model,
            code=code,
            data=data,
            controls=controls,
            metrics=tuple(sorted(metrics_dict.items())),
            pre_registered_decision=pre_registered_decision,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "track": self.track,
            "timestamp_utc": self.timestamp_utc,
            "model": self.model.as_dict(),
            "code": self.code.as_dict(),
            "data": self.data.as_dict(),
            "controls": self.controls.as_dict(),
            "metrics": dict(self.metrics),
            "pre_registered_decision": self.pre_registered_decision.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplicationManifest:
        metrics_raw = data.get("metrics", {})
        return cls(
            run_id=data["run_id"],
            track=data["track"],
            timestamp_utc=data["timestamp_utc"],
            model=ModelInfo.from_dict(data["model"]),
            code=CodeInfo.from_dict(data["code"]),
            data=DataHashes.from_dict(data["data"]),
            controls=ControlFlags.from_dict(data["controls"]),
            metrics=tuple(sorted(metrics_raw.items())),
            pre_registered_decision=PreRegisteredDecision.from_dict(
                data["pre_registered_decision"],
            ),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_TRACKS = frozenset({"A", "B", "C"})
_VALID_OUTCOMES = frozenset({"pass", "fail", "inconclusive"})

REQUIRED_METRIC_KEYS = frozenset(
    {
        "cka_drift",
        "preserved_fraction",
        "perplexity_drift_identity",
        "perplexity_drift_general",
        "recall_raw_completion",
        "recall_chat_template",
        "null_rank",
        "condition_number",
        "spectral_gap",
    },
)

_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "run_id",
        "track",
        "timestamp_utc",
        "model",
        "code",
        "data",
        "controls",
        "metrics",
        "pre_registered_decision",
    },
)

_REQUIRED_STRING_PATHS: list[tuple[str, ...]] = [
    ("run_id",),
    ("track",),
    ("timestamp_utc",),
    ("model", "id"),
    ("model", "quantization"),
    ("model", "backend"),
    ("code", "modelcypher_commit"),
    ("code", "experiment_module_commit"),
    ("data", "fact_pool_hash"),
    ("data", "split_manifest_hash"),
    ("data", "reference_corpus_hash"),
    ("pre_registered_decision", "criteria_version"),
    ("pre_registered_decision", "outcome"),
    ("pre_registered_decision", "reason"),
]


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Traverse a nested dict by key path."""
    obj: Any = data
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def validate_manifest(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate manifest JSON data against the replication protocol schema.

    Returns ``(is_valid, errors)`` where *errors* is a list of human-readable
    messages describing each validation failure.
    """
    errors: list[str] = []

    # Required top-level keys
    missing_top = _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")

    # Track value
    track = data.get("track")
    if track is not None and track not in _VALID_TRACKS:
        errors.append(f"Invalid track: {track!r} (must be one of {sorted(_VALID_TRACKS)})")

    # Outcome value
    decision = data.get("pre_registered_decision")
    if isinstance(decision, dict):
        outcome = decision.get("outcome")
        if outcome is not None and outcome not in _VALID_OUTCOMES:
            errors.append(
                f"Invalid outcome: {outcome!r} "
                f"(must be one of {sorted(_VALID_OUTCOMES)})",
            )

    # Required string fields must be non-empty
    for path in _REQUIRED_STRING_PATHS:
        val = _get_nested(data, path)
        dotted = ".".join(path)
        if val is None:
            errors.append(f"Missing required field: {dotted}")
        elif not isinstance(val, str):
            errors.append(f"Field {dotted} must be a string, got {type(val).__name__}")
        elif not val.strip():
            errors.append(f"Field {dotted} must be non-empty")

    # Metrics: required keys present, values finite
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        missing_metrics = REQUIRED_METRIC_KEYS - set(metrics.keys())
        if missing_metrics:
            errors.append(f"Missing metric keys: {sorted(missing_metrics)}")
        for key, value in metrics.items():
            if key in REQUIRED_METRIC_KEYS:
                if not isinstance(value, (int, float)):
                    errors.append(
                        f"Metric {key!r} must be a number, got {type(value).__name__}",
                    )
                elif not math.isfinite(value):
                    errors.append(f"Metric {key!r} must be finite, got {value}")
    elif metrics is not None:
        errors.append(f"'metrics' must be a dict, got {type(metrics).__name__}")

    return (len(errors) == 0, errors)


__all__ = [
    "CodeInfo",
    "ControlFlags",
    "DataHashes",
    "ModelInfo",
    "PreRegisteredDecision",
    "REQUIRED_METRIC_KEYS",
    "ReplicationManifest",
    "validate_manifest",
]
