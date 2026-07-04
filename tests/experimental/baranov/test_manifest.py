"""Unit tests for replication manifest schema and validation."""

from __future__ import annotations

import json
import math

import pytest

from modelcypher.experimental.baranov.manifest import (
    REQUIRED_METRIC_KEYS,
    CodeInfo,
    ControlFlags,
    DataHashes,
    ModelInfo,
    PreRegisteredDecision,
    ReplicationManifest,
    validate_manifest,
)


def _sample_metrics() -> dict[str, float]:
    """Return a complete set of required metrics."""
    return {
        "cka_drift": 0.02,
        "preserved_fraction": 0.97,
        "perplexity_drift_identity": 0.03,
        "perplexity_drift_general": 0.05,
        "recall_raw_completion": 0.8,
        "recall_chat_template": 0.75,
        "null_rank": 42.0,
        "condition_number": 1234.5,
        "spectral_gap": 0.12,
    }


def _sample_manifest_dict() -> dict:
    """Return a complete, valid manifest dict."""
    return {
        "run_id": "run-001",
        "track": "A",
        "timestamp_utc": "2026-02-26T12:00:00Z",
        "model": {"id": "test-model", "quantization": "bf16", "backend": "mlx"},
        "code": {"modelcypher_commit": "abc123", "experiment_module_commit": "def456"},
        "data": {
            "fact_pool_hash": "sha256:aaa",
            "split_manifest_hash": "sha256:bbb",
            "reference_corpus_hash": "sha256:ccc",
        },
        "controls": {"base_control": True, "lora_only_control": True, "edit_only_control": False},
        "metrics": _sample_metrics(),
        "pre_registered_decision": {
            "criteria_version": "v1",
            "outcome": "pass",
            "reason": "All criteria met.",
        },
    }


def _sample_manifest() -> ReplicationManifest:
    """Return a typed ReplicationManifest instance."""
    return ReplicationManifest.from_dict(_sample_manifest_dict())


class TestReplicationManifestRoundTrip:
    def test_round_trip_json(self) -> None:
        """as_dict -> from_dict preserves all fields."""
        original = _sample_manifest()
        d = original.as_dict()
        restored = ReplicationManifest.from_dict(d)
        assert restored.run_id == original.run_id
        assert restored.track == original.track
        assert restored.model == original.model
        assert restored.code == original.code
        assert restored.data == original.data
        assert restored.controls == original.controls
        assert dict(restored.metrics) == dict(original.metrics)
        assert restored.pre_registered_decision == original.pre_registered_decision

    def test_json_serializable(self) -> None:
        """as_dict output is JSON-serializable."""
        m = _sample_manifest()
        serialized = json.dumps(m.as_dict())
        assert isinstance(serialized, str)
        restored = json.loads(serialized)
        assert restored["run_id"] == m.run_id

    def test_metrics_dict_property(self) -> None:
        """metrics_dict returns the metrics as a dict."""
        m = _sample_manifest()
        assert isinstance(m.metrics_dict, dict)
        assert "cka_drift" in m.metrics_dict

    def test_from_metrics_dict_classmethod(self) -> None:
        """from_metrics_dict converts dict -> sorted tuples."""
        m = ReplicationManifest.from_metrics_dict(
            run_id="r1",
            track="B",
            timestamp_utc="2026-01-01T00:00:00Z",
            model=ModelInfo(id="m", quantization="fp16", backend="jax"),
            code=CodeInfo(modelcypher_commit="a", experiment_module_commit="b"),
            data=DataHashes(fact_pool_hash="h1", split_manifest_hash="h2", reference_corpus_hash="h3"),
            controls=ControlFlags(base_control=True, lora_only_control=False, edit_only_control=False),
            metrics_dict=_sample_metrics(),
            pre_registered_decision=PreRegisteredDecision(criteria_version="v1", outcome="fail", reason="no"),
        )
        assert m.track == "B"
        assert m.metrics_dict["cka_drift"] == 0.02


class TestValidateManifest:
    def test_valid_manifest_passes(self) -> None:
        """A complete manifest passes validation."""
        valid, errors = validate_manifest(_sample_manifest_dict())
        assert valid, f"Unexpected errors: {errors}"
        assert errors == []

    def test_missing_top_level_key(self) -> None:
        """Missing run_id fails validation."""
        d = _sample_manifest_dict()
        del d["run_id"]
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("run_id" in e for e in errors)

    def test_invalid_track(self) -> None:
        """Track 'D' is not valid."""
        d = _sample_manifest_dict()
        d["track"] = "D"
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("Invalid track" in e for e in errors)

    def test_invalid_outcome(self) -> None:
        """Outcome 'maybe' is not valid."""
        d = _sample_manifest_dict()
        d["pre_registered_decision"]["outcome"] = "maybe"
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("Invalid outcome" in e for e in errors)

    def test_missing_metric_key(self) -> None:
        """Missing a required metric key fails."""
        d = _sample_manifest_dict()
        del d["metrics"]["spectral_gap"]
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("spectral_gap" in e for e in errors)

    def test_nan_metric_fails(self) -> None:
        """NaN metric value fails."""
        d = _sample_manifest_dict()
        d["metrics"]["cka_drift"] = float("nan")
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("finite" in e for e in errors)

    def test_inf_metric_fails(self) -> None:
        """Inf metric value fails."""
        d = _sample_manifest_dict()
        d["metrics"]["null_rank"] = float("inf")
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("finite" in e for e in errors)

    def test_non_numeric_metric_fails(self) -> None:
        """String metric value fails."""
        d = _sample_manifest_dict()
        d["metrics"]["cka_drift"] = "not_a_number"
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("number" in e for e in errors)

    def test_empty_string_fields_fail(self) -> None:
        """Empty run_id fails."""
        d = _sample_manifest_dict()
        d["run_id"] = ""
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("non-empty" in e for e in errors)

    def test_whitespace_only_string_fails(self) -> None:
        """Whitespace-only string fails the non-empty check."""
        d = _sample_manifest_dict()
        d["run_id"] = "   "
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("non-empty" in e for e in errors)

    def test_nested_empty_string_fails(self) -> None:
        """Empty model.id fails."""
        d = _sample_manifest_dict()
        d["model"]["id"] = ""
        valid, errors = validate_manifest(d)
        assert not valid
        assert any("model.id" in e for e in errors)

    def test_all_tracks_valid(self) -> None:
        """Tracks A, B, C all pass."""
        for track in ("A", "B", "C"):
            d = _sample_manifest_dict()
            d["track"] = track
            valid, errors = validate_manifest(d)
            assert valid, f"Track {track} failed: {errors}"

    def test_all_outcomes_valid(self) -> None:
        """Outcomes pass, fail, inconclusive all pass."""
        for outcome in ("pass", "fail", "inconclusive"):
            d = _sample_manifest_dict()
            d["pre_registered_decision"]["outcome"] = outcome
            valid, errors = validate_manifest(d)
            assert valid, f"Outcome {outcome} failed: {errors}"
