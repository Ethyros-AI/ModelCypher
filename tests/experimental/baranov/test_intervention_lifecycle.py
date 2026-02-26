"""Tests for Track A intervention lifecycle.

Tests the core evaluation pipeline functions from baranov_track_a.py
without requiring real models or backends.  Validates:
- InterventionMode enum
- measure_recall function
- evaluate_model result structure (baseline, no_op, lora_only)
- build_manifest with intervention modes
- build_summary structure
- CSV row generation with phase column
- Recall curves structure with pre/post
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import script-level functions (scripts/ is on path via conftest or direct import)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from baranov_track_a import (
    InterventionMode,
    _build_csv_rows,
    _build_recall_curves,
    _compute_fact_pool_hash,
    build_manifest,
    build_summary,
    measure_recall,
    FACT_POOL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact_triple(
    subject: str = "Paris",
    relation: str = "capital_of",
    obj: str = "France",
    fact_id: str = "f1",
):
    from modelcypher.experimental.baranov.models import FactTriple
    return FactTriple(subject=subject, relation=relation, object=obj, fact_id=fact_id)


def _make_mock_backend():
    """Create a mock backend with generate_fn."""
    backend = MagicMock()
    backend.generate.return_value = "France is the answer"
    return backend


def _make_pre_recall(raw_rate: float = 0.8, chat_rate: float = 0.6) -> dict[str, Any]:
    """Build a minimal recall measurement dict."""
    return {
        "raw_completion": {
            "recall_rate": raw_rate,
            "recalled_count": int(raw_rate * 5),
            "total": 5,
            "confidence_interval": None,
            "elapsed_s": 1.0,
            "per_fact": [
                {"fact_id": f"f{i+1}", "recalled": i < int(raw_rate * 5),
                 "raw_output": "output", "confidence": None}
                for i in range(5)
            ],
        },
        "chat_template_result": {
            "recall_rate": chat_rate,
            "recalled_count": int(chat_rate * 5),
            "total": 5,
            "confidence_interval": None,
            "elapsed_s": 1.0,
            "per_fact": [
                {"fact_id": f"f{i+1}", "recalled": i < int(chat_rate * 5),
                 "raw_output": "output", "confidence": None}
                for i in range(5)
            ],
        },
    }


def _make_result(
    model_name: str = "TestModel",
    intervention: str = "baseline",
    pre_raw: float = 0.8,
    pre_chat: float = 0.6,
    post_raw: float | None = None,
    post_chat: float | None = None,
) -> dict[str, Any]:
    """Build a minimal model result dict."""
    result: dict[str, Any] = {
        "model_name": model_name,
        "model_path": "/fake/path",
        "quantization": "bf16",
        "architecture": "test",
        "intervention": intervention,
        "n_facts": 5,
        "pre": _make_pre_recall(pre_raw, pre_chat),
        "post": None,
        "deltas": None,
        "geometry": {
            "per_layer_cka": {},
            "min_cka": 1.0,
            "mean_cka": 1.0,
            "cka_drift": 0.0,
            "preserved_fraction": 1.0,
        },
        "training_meta": None,
    }

    if post_raw is not None and post_chat is not None:
        result["post"] = _make_pre_recall(post_raw, post_chat)
        result["deltas"] = {
            "delta_raw_recall": post_raw - pre_raw,
            "delta_chat_recall": post_chat - pre_chat,
            "delta_cka_drift": 0.05,
            "delta_preserved_fraction": -0.03,
        }
        result["geometry"]["cka_drift"] = 0.05
        result["geometry"]["preserved_fraction"] = 0.97

    return result


# ---------------------------------------------------------------------------
# InterventionMode
# ---------------------------------------------------------------------------


class TestInterventionMode:
    def test_values(self):
        assert InterventionMode.baseline.value == "baseline"
        assert InterventionMode.no_op.value == "no_op"
        assert InterventionMode.lora_only.value == "lora_only"

    def test_str_enum(self):
        """InterventionMode is str, Enum — JSON-serializable."""
        assert isinstance(InterventionMode.baseline, str)
        data = json.dumps({"mode": InterventionMode.lora_only})
        assert "lora_only" in data

    def test_from_string(self):
        mode = InterventionMode("no_op")
        assert mode == InterventionMode.no_op


# ---------------------------------------------------------------------------
# measure_recall
# ---------------------------------------------------------------------------


class TestMeasureRecall:
    def test_returns_both_modes(self):
        backend = _make_mock_backend()
        model = MagicMock()
        tokenizer = MagicMock()
        facts = [_make_fact_triple()]

        result = measure_recall(model, tokenizer, facts, backend, "test")

        assert "raw_completion" in result
        assert "chat_template_result" in result

    def test_raw_completion_structure(self):
        backend = _make_mock_backend()
        result = measure_recall(
            MagicMock(), MagicMock(), [_make_fact_triple()], backend, "test",
        )

        raw = result["raw_completion"]
        assert "recall_rate" in raw
        assert "recalled_count" in raw
        assert "total" in raw
        assert "per_fact" in raw
        assert "elapsed_s" in raw

    def test_recall_counts_match(self):
        backend = _make_mock_backend()
        # "France" is in the output, so recall should be True
        backend.generate.return_value = "The answer is France"

        facts = [_make_fact_triple(), _make_fact_triple(fact_id="f2")]
        result = measure_recall(
            MagicMock(), MagicMock(), facts, backend, "test",
        )

        raw = result["raw_completion"]
        assert raw["total"] == 2
        assert raw["recalled_count"] == 2
        assert raw["recall_rate"] == 1.0


# ---------------------------------------------------------------------------
# _build_csv_rows
# ---------------------------------------------------------------------------


class TestBuildCsvRows:
    def test_baseline_has_pre_only(self):
        result = _make_result(intervention="baseline")
        rows = _build_csv_rows([result], InterventionMode.baseline)

        phases = [r["phase"] for r in rows]
        assert all(p == "pre" for p in phases)
        assert len(rows) == 2  # raw + chat

    def test_intervention_has_pre_post_delta(self):
        result = _make_result(
            intervention="lora_only",
            post_raw=0.9, post_chat=0.7,
        )
        rows = _build_csv_rows([result], InterventionMode.lora_only)

        phases = [r["phase"] for r in rows]
        assert phases.count("pre") == 2
        assert phases.count("post") == 2
        assert phases.count("delta") == 2
        assert len(rows) == 6

    def test_no_op_has_pre_post_delta(self):
        result = _make_result(
            intervention="no_op",
            post_raw=0.8, post_chat=0.6,  # Same as pre
        )
        rows = _build_csv_rows([result], InterventionMode.no_op)
        assert len(rows) == 6

    def test_phase_column_present(self):
        result = _make_result()
        rows = _build_csv_rows([result], InterventionMode.baseline)
        assert all("phase" in r for r in rows)

    def test_multiple_models(self):
        r1 = _make_result(model_name="ModelA", post_raw=0.9, post_chat=0.7)
        r2 = _make_result(model_name="ModelB", post_raw=0.85, post_chat=0.65)
        rows = _build_csv_rows([r1, r2], InterventionMode.lora_only)
        assert len(rows) == 12  # 6 per model


# ---------------------------------------------------------------------------
# _build_recall_curves
# ---------------------------------------------------------------------------


class TestBuildRecallCurves:
    def test_baseline_has_pre_only(self):
        result = _make_result()
        curves = _build_recall_curves([result])

        assert "TestModel" in curves
        assert "pre" in curves["TestModel"]
        assert "post" not in curves["TestModel"]

    def test_intervention_has_pre_and_post(self):
        result = _make_result(post_raw=0.9, post_chat=0.7)
        curves = _build_recall_curves([result])

        assert "pre" in curves["TestModel"]
        assert "post" in curves["TestModel"]

    def test_per_fact_detail_present(self):
        result = _make_result()
        curves = _build_recall_curves([result])

        pre = curves["TestModel"]["pre"]
        assert "raw_completion" in pre
        assert "chat_template" in pre
        assert len(pre["raw_completion"]) == 5


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_baseline_manifest(self):
        result = _make_result()
        manifest = build_manifest([result], FACT_POOL[:5], "test-run", InterventionMode.baseline)

        assert manifest["track"] == "A"
        assert manifest["run_id"] == "test-run"
        assert manifest["pre_registered_decision"]["outcome"] == "inconclusive"
        assert "Baseline" in manifest["pre_registered_decision"]["reason"]

    def test_no_op_manifest(self):
        result = _make_result(intervention="no_op", post_raw=0.8, post_chat=0.6)
        manifest = build_manifest([result], FACT_POOL[:5], "test-run", InterventionMode.no_op)

        assert manifest["pre_registered_decision"]["outcome"] == "inconclusive"
        assert "No-op" in manifest["pre_registered_decision"]["reason"]

    def test_lora_manifest(self):
        result = _make_result(
            intervention="lora_only", post_raw=0.9, post_chat=0.7,
        )
        manifest = build_manifest([result], FACT_POOL[:5], "test-run", InterventionMode.lora_only)

        assert manifest["pre_registered_decision"]["outcome"] == "inconclusive"
        assert "LoRA" in manifest["pre_registered_decision"]["reason"]

    def test_control_flags_baseline(self):
        result = _make_result()
        manifest = build_manifest([result], FACT_POOL[:5], "run1", InterventionMode.baseline)
        assert manifest["controls"]["base_control"] is True
        assert manifest["controls"]["lora_only_control"] is False

    def test_control_flags_lora(self):
        result = _make_result(post_raw=0.9, post_chat=0.7)
        manifest = build_manifest([result], FACT_POOL[:5], "run1", InterventionMode.lora_only)
        assert manifest["controls"]["base_control"] is False
        assert manifest["controls"]["lora_only_control"] is True

    def test_manifest_validates(self):
        """Built manifest passes schema validation."""
        from modelcypher.experimental.baranov.manifest import validate_manifest

        result = _make_result()
        manifest = build_manifest([result], FACT_POOL[:5], "run1", InterventionMode.baseline)
        valid, errors = validate_manifest(manifest)
        assert valid, f"Manifest validation errors: {errors}"


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_baseline_summary(self):
        result = _make_result()
        summary = build_summary([result], InterventionMode.baseline, "run1")

        assert summary["run_id"] == "run1"
        assert summary["intervention"] == "baseline"
        assert "TestModel" in summary["models"]

        model = summary["models"]["TestModel"]
        assert "pre_raw_recall" in model
        assert "pre_chat_recall" in model
        assert "post_raw_recall" not in model

    def test_intervention_summary_has_deltas(self):
        result = _make_result(
            intervention="lora_only", post_raw=0.9, post_chat=0.7,
        )
        summary = build_summary([result], InterventionMode.lora_only, "run1")

        model = summary["models"]["TestModel"]
        assert "post_raw_recall" in model
        assert "deltas" in model
        assert "cka_drift" in model

    def test_training_meta_excluded_adapter_path(self):
        """adapter_path should not appear in the summary (local filesystem path)."""
        result = _make_result(
            intervention="lora_only", post_raw=0.9, post_chat=0.7,
        )
        result["training_meta"] = {
            "adapter_path": "/local/path/to/adapter",
            "train_iters": 100,
            "final_loss": 0.5,
        }

        summary = build_summary([result], InterventionMode.lora_only, "run1")
        training = summary["models"]["TestModel"]["training"]
        assert "adapter_path" not in training
        assert "train_iters" in training

    def test_multiple_models(self):
        r1 = _make_result(model_name="Small")
        r2 = _make_result(model_name="Large")
        summary = build_summary([r1, r2], InterventionMode.baseline, "run1")
        assert "Small" in summary["models"]
        assert "Large" in summary["models"]


# ---------------------------------------------------------------------------
# _compute_fact_pool_hash
# ---------------------------------------------------------------------------


class TestFactPoolHash:
    def test_deterministic(self):
        h1 = _compute_fact_pool_hash(FACT_POOL[:5])
        h2 = _compute_fact_pool_hash(FACT_POOL[:5])
        assert h1 == h2

    def test_different_for_different_pools(self):
        h1 = _compute_fact_pool_hash(FACT_POOL[:3])
        h2 = _compute_fact_pool_hash(FACT_POOL[:5])
        assert h1 != h2

    def test_starts_with_sha256(self):
        h = _compute_fact_pool_hash(FACT_POOL[:1])
        assert h.startswith("sha256:")
