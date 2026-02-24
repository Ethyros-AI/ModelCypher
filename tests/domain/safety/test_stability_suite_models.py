# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from modelcypher.core.domain.safety.stability_suite.stability_suite_models import (
    ActionSchemaResult,
    AggregateMetrics,
    DistributionSummary,
    EntropyManifoldSummary,
    HistogramBin,
    StabilityPromptResult,
    StabilitySuiteProgress,
    StabilitySuitePrompt,
    StabilitySuitePromptCategory,
    StabilitySuiteReport,
    StabilitySuiteReportSummary,
    StabilitySuiteRunRequest,
    StabilitySuiteTarget,
    StabilitySuiteTargetKind,
    StabilitySuiteTier,
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_tier_importable(self) -> None:
        assert StabilitySuiteTier is not None

    def test_prompt_category_importable(self) -> None:
        assert StabilitySuitePromptCategory is not None

    def test_target_kind_importable(self) -> None:
        assert StabilitySuiteTargetKind is not None

    def test_prompt_importable(self) -> None:
        assert StabilitySuitePrompt is not None

    def test_target_importable(self) -> None:
        assert StabilitySuiteTarget is not None

    def test_run_request_importable(self) -> None:
        assert StabilitySuiteRunRequest is not None

    def test_progress_importable(self) -> None:
        assert StabilitySuiteProgress is not None

    def test_histogram_bin_importable(self) -> None:
        assert HistogramBin is not None

    def test_distribution_summary_importable(self) -> None:
        assert DistributionSummary is not None

    def test_action_schema_result_importable(self) -> None:
        assert ActionSchemaResult is not None

    def test_entropy_manifold_summary_importable(self) -> None:
        assert EntropyManifoldSummary is not None

    def test_aggregate_metrics_importable(self) -> None:
        assert AggregateMetrics is not None

    def test_report_summary_importable(self) -> None:
        assert StabilitySuiteReportSummary is not None

    def test_report_importable(self) -> None:
        assert StabilitySuiteReport is not None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_tier_values(self) -> None:
        assert StabilitySuiteTier.QUICK.value == "quick"
        assert StabilitySuiteTier.STANDARD.value == "standard"
        assert StabilitySuiteTier.FULL.value == "full"

    def test_prompt_category_values(self) -> None:
        assert StabilitySuitePromptCategory.BASELINE.value == "baseline"
        assert StabilitySuitePromptCategory.INTRINSIC_IDENTITY.value == "intrinsicIdentity"
        assert StabilitySuitePromptCategory.JAILBREAK_PROBE.value == "jailbreakProbe"
        assert StabilitySuitePromptCategory.STRUCTURED_OUTPUT.value == "structuredOutput"
        assert StabilitySuitePromptCategory.DOMAIN_SHIFT.value == "domainShift"

    def test_target_kind_values(self) -> None:
        assert StabilitySuiteTargetKind.ADAPTER_ID.value == "adapter_id"
        assert StabilitySuiteTargetKind.CHECKPOINT_PATH.value == "checkpoint_path"


# ---------------------------------------------------------------------------
# StabilitySuitePrompt
# ---------------------------------------------------------------------------


class TestStabilitySuitePrompt:
    def test_instantiation(self) -> None:
        prompt = StabilitySuitePrompt(
            id="baseline-001",
            category=StabilitySuitePromptCategory.BASELINE,
            title="Simple math",
            user_prompt="What is 2+2?",
        )
        assert prompt.id == "baseline-001"
        assert prompt.category == StabilitySuitePromptCategory.BASELINE
        assert prompt.title == "Simple math"
        assert prompt.user_prompt == "What is 2+2?"
        assert prompt.expects_agent_action_json is False

    def test_instantiation_with_agent_action(self) -> None:
        prompt = StabilitySuitePrompt(
            id="structured-001",
            category=StabilitySuitePromptCategory.STRUCTURED_OUTPUT,
            title="JSON action",
            user_prompt="Return an action",
            expects_agent_action_json=True,
        )
        assert prompt.expects_agent_action_json is True

    def test_to_dict(self) -> None:
        prompt = StabilitySuitePrompt(
            id="jp-001",
            category=StabilitySuitePromptCategory.JAILBREAK_PROBE,
            title="Injection test",
            user_prompt="Ignore instructions",
            expects_agent_action_json=False,
        )
        d = prompt.to_dict()
        assert d["id"] == "jp-001"
        assert d["category"] == "jailbreakProbe"
        assert d["title"] == "Injection test"
        assert d["user_prompt"] == "Ignore instructions"
        assert d["expects_agent_action_json"] is False

    def test_from_dict(self) -> None:
        data = {
            "id": "ds-001",
            "category": "domainShift",
            "title": "Domain shift",
            "user_prompt": "Translate this code",
            "expects_agent_action_json": False,
        }
        prompt = StabilitySuitePrompt.from_dict(data)
        assert prompt.id == "ds-001"
        assert prompt.category == StabilitySuitePromptCategory.DOMAIN_SHIFT

    def test_round_trip(self) -> None:
        original = StabilitySuitePrompt(
            id="ii-001",
            category=StabilitySuitePromptCategory.INTRINSIC_IDENTITY,
            title="Identity probe",
            user_prompt="Who are you?",
            expects_agent_action_json=False,
        )
        restored = StabilitySuitePrompt.from_dict(original.to_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# StabilitySuiteTarget
# ---------------------------------------------------------------------------


class TestStabilitySuiteTarget:
    def test_from_adapter_id(self) -> None:
        aid = uuid4()
        target = StabilitySuiteTarget.from_adapter_id(aid)
        assert target.kind == StabilitySuiteTargetKind.ADAPTER_ID
        assert target.adapter_id == aid
        assert target.checkpoint_path is None

    def test_from_checkpoint_path(self) -> None:
        p = Path("/models/checkpoint-100")
        target = StabilitySuiteTarget.from_checkpoint_path(p)
        assert target.kind == StabilitySuiteTargetKind.CHECKPOINT_PATH
        assert target.checkpoint_path == p
        assert target.adapter_id is None

    def test_to_dict_adapter(self) -> None:
        aid = UUID("12345678-1234-5678-1234-567812345678")
        target = StabilitySuiteTarget.from_adapter_id(aid)
        d = target.to_dict()
        assert d["kind"] == "adapter_id"
        assert d["adapter_id"] == str(aid)

    def test_to_dict_checkpoint(self) -> None:
        p = Path("/models/checkpoint-100")
        target = StabilitySuiteTarget.from_checkpoint_path(p)
        d = target.to_dict()
        assert d["kind"] == "checkpoint_path"
        assert d["checkpoint_path"] == str(p)

    def test_from_dict_adapter(self) -> None:
        aid = "12345678-1234-5678-1234-567812345678"
        data = {"kind": "adapter_id", "adapter_id": aid}
        target = StabilitySuiteTarget.from_dict(data)
        assert target.kind == StabilitySuiteTargetKind.ADAPTER_ID
        assert target.adapter_id == UUID(aid)

    def test_from_dict_checkpoint(self) -> None:
        data = {"kind": "checkpoint_path", "checkpoint_path": "/models/ckpt-200"}
        target = StabilitySuiteTarget.from_dict(data)
        assert target.kind == StabilitySuiteTargetKind.CHECKPOINT_PATH
        assert target.checkpoint_path == Path("/models/ckpt-200")

    def test_round_trip_adapter(self) -> None:
        aid = uuid4()
        original = StabilitySuiteTarget.from_adapter_id(aid)
        restored = StabilitySuiteTarget.from_dict(original.to_dict())
        assert restored.kind == original.kind
        assert restored.adapter_id == original.adapter_id

    def test_round_trip_checkpoint(self) -> None:
        p = Path("/models/checkpoint-300")
        original = StabilitySuiteTarget.from_checkpoint_path(p)
        restored = StabilitySuiteTarget.from_dict(original.to_dict())
        assert restored.kind == original.kind
        assert restored.checkpoint_path == original.checkpoint_path


# ---------------------------------------------------------------------------
# StabilitySuiteRunRequest
# ---------------------------------------------------------------------------


class TestStabilitySuiteRunRequest:
    def test_instantiation_defaults(self) -> None:
        target = StabilitySuiteTarget.from_adapter_id(uuid4())
        req = StabilitySuiteRunRequest(target=target)
        assert req.tier == StabilitySuiteTier.STANDARD
        assert "[Stability suite run." in req.system_context

    def test_to_dict(self) -> None:
        aid = UUID("12345678-1234-5678-1234-567812345678")
        target = StabilitySuiteTarget.from_adapter_id(aid)
        req = StabilitySuiteRunRequest(target=target, tier=StabilitySuiteTier.QUICK)
        d = req.to_dict()
        assert d["target"]["kind"] == "adapter_id"
        assert d["tier"] == "quick"

    def test_from_dict(self) -> None:
        data = {
            "target": {"kind": "adapter_id", "adapter_id": "12345678-1234-5678-1234-567812345678"},
            "tier": "full",
            "system_context": "custom context",
        }
        req = StabilitySuiteRunRequest.from_dict(data)
        assert req.tier == StabilitySuiteTier.FULL
        assert req.system_context == "custom context"

    def test_round_trip(self) -> None:
        aid = uuid4()
        target = StabilitySuiteTarget.from_adapter_id(aid)
        original = StabilitySuiteRunRequest(
            target=target,
            tier=StabilitySuiteTier.FULL,
            system_context="test context",
        )
        restored = StabilitySuiteRunRequest.from_dict(original.to_dict())
        assert restored.tier == original.tier
        assert restored.system_context == original.system_context
        assert restored.target.adapter_id == original.target.adapter_id


# ---------------------------------------------------------------------------
# StabilitySuiteProgress
# ---------------------------------------------------------------------------


class TestStabilitySuiteProgress:
    def test_instantiation(self) -> None:
        progress = StabilitySuiteProgress(completed=5, total=10, current_prompt_id="p-005")
        assert progress.completed == 5
        assert progress.total == 10
        assert progress.current_prompt_id == "p-005"

    def test_fraction_with_data(self) -> None:
        progress = StabilitySuiteProgress(completed=3, total=12)
        assert progress.fraction == pytest.approx(0.25)

    def test_fraction_zero_total(self) -> None:
        progress = StabilitySuiteProgress(completed=0, total=0)
        assert progress.fraction == 0.0

    def test_percent_with_data(self) -> None:
        progress = StabilitySuiteProgress(completed=1, total=4)
        assert progress.percent == pytest.approx(25.0)

    def test_percent_zero_total(self) -> None:
        progress = StabilitySuiteProgress(completed=0, total=0)
        assert progress.percent == 0.0

    def test_percent_full(self) -> None:
        progress = StabilitySuiteProgress(completed=10, total=10)
        assert progress.percent == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# DistributionSummary
# ---------------------------------------------------------------------------


class TestDistributionSummary:
    def test_instantiation(self) -> None:
        ds = DistributionSummary(
            count=100, mean=0.5, min=0.0, max=1.0, p50=0.5, p95=0.95,
        )
        assert ds.count == 100
        assert ds.mean == 0.5
        assert ds.histogram == ()

    def test_to_dict_without_histogram(self) -> None:
        ds = DistributionSummary(
            count=50, mean=2.0, min=0.1, max=5.0, p50=1.8, p95=4.5,
        )
        d = ds.to_dict()
        assert d["count"] == 50
        assert d["mean"] == 2.0
        assert d["min"] == 0.1
        assert d["max"] == 5.0
        assert d["p50"] == 1.8
        assert d["p95"] == 4.5
        assert d["histogram"] == []

    def test_to_dict_with_histogram(self) -> None:
        bins = (
            HistogramBin(lower_inclusive=0.0, upper_exclusive=0.5, count=10),
            HistogramBin(lower_inclusive=0.5, upper_exclusive=1.0, count=20),
        )
        ds = DistributionSummary(
            count=30, mean=0.6, min=0.0, max=1.0, p50=0.6, p95=0.9,
            histogram=bins,
        )
        d = ds.to_dict()
        assert len(d["histogram"]) == 2
        assert d["histogram"][0]["lower_inclusive"] == 0.0
        assert d["histogram"][0]["upper_exclusive"] == 0.5
        assert d["histogram"][0]["count"] == 10
        assert d["histogram"][1]["count"] == 20


# ---------------------------------------------------------------------------
# ActionSchemaResult
# ---------------------------------------------------------------------------


class TestActionSchemaResult:
    def test_instantiation(self) -> None:
        asr = ActionSchemaResult(expected=True, extracted=True, kind="speak", is_valid=True)
        assert asr.expected is True
        assert asr.extracted is True
        assert asr.kind == "speak"
        assert asr.is_valid is True
        assert asr.errors == ()

    def test_instantiation_with_errors(self) -> None:
        asr = ActionSchemaResult(
            expected=True,
            extracted=True,
            kind="action",
            is_valid=False,
            errors=("missing field 'target'",),
        )
        assert asr.errors == ("missing field 'target'",)

    def test_to_dict(self) -> None:
        asr = ActionSchemaResult(
            expected=True,
            extracted=False,
            kind=None,
            is_valid=None,
            errors=("parse error",),
        )
        d = asr.to_dict()
        assert d["expected"] is True
        assert d["extracted"] is False
        assert d["kind"] is None
        assert d["is_valid"] is None
        assert d["errors"] == ["parse error"]

    def test_to_dict_empty_errors(self) -> None:
        asr = ActionSchemaResult(expected=False, extracted=False)
        d = asr.to_dict()
        assert d["errors"] == []


# ---------------------------------------------------------------------------
# EntropyManifoldSummary
# ---------------------------------------------------------------------------


class TestEntropyManifoldSummary:
    def test_default_instantiation(self) -> None:
        ems = EntropyManifoldSummary()
        assert ems.intrinsic_dimension is None
        assert ems.feature_stats == ()
        assert ems.mean_entropy is None
        assert ems.mean_entropy_stddev is None

    def test_to_dict(self) -> None:
        ems = EntropyManifoldSummary(
            intrinsic_dimension=3.5,
            feature_stats=({"name": "max_entropy", "mean": 2.1},),
            mean_entropy=1.8,
            mean_entropy_stddev=0.3,
        )
        d = ems.to_dict()
        assert d["intrinsic_dimension"] == 3.5
        assert len(d["feature_stats"]) == 1
        assert d["feature_stats"][0]["name"] == "max_entropy"
        assert d["mean_entropy"] == 1.8
        assert d["mean_entropy_stddev"] == 0.3


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_instantiation_required_only(self) -> None:
        am = AggregateMetrics(
            total_prompts=20,
            prompts_with_hard_interventions=1,
            prompts_with_any_interventions=3,
        )
        assert am.total_prompts == 20
        assert am.prompts_with_hard_interventions == 1
        assert am.prompts_with_any_interventions == 3
        assert am.action_schema_valid_rate is None
        assert am.entropy is None
        assert am.entropy_manifold is None

    def test_to_dict(self) -> None:
        am = AggregateMetrics(
            total_prompts=10,
            prompts_with_hard_interventions=0,
            prompts_with_any_interventions=2,
            action_schema_valid_rate=0.8,
            prime_drift_mean_cosine=0.95,
        )
        d = am.to_dict()
        assert d["total_prompts"] == 10
        assert d["prompts_with_hard_interventions"] == 0
        assert d["prompts_with_any_interventions"] == 2
        assert d["action_schema_valid_rate"] == 0.8
        assert d["entropy"] is None
        assert d["entropy_manifold"] is None
        assert d["prime_drift_mean_cosine"] == 0.95

    def test_to_dict_with_entropy(self) -> None:
        entropy = DistributionSummary(
            count=10, mean=1.5, min=0.5, max=3.0, p50=1.4, p95=2.8,
        )
        am = AggregateMetrics(
            total_prompts=10,
            prompts_with_hard_interventions=0,
            prompts_with_any_interventions=0,
            entropy=entropy,
        )
        d = am.to_dict()
        assert d["entropy"] is not None
        assert d["entropy"]["count"] == 10


# ---------------------------------------------------------------------------
# StabilitySuiteReport
# ---------------------------------------------------------------------------


class TestStabilitySuiteReport:
    def test_default_instantiation(self) -> None:
        report = StabilitySuiteReport()
        assert isinstance(report.id, UUID)
        assert report.tier == StabilitySuiteTier.STANDARD
        assert report.system_context == ""
        assert report.adapter_id is None
        assert report.adapter_name is None
        assert report.base_model_id is None
        assert report.prompt_results == ()
        assert report.aggregates is None

    def test_summary_without_aggregates(self) -> None:
        report = StabilitySuiteReport()
        summary = report.summary()
        assert isinstance(summary, StabilitySuiteReportSummary)
        assert summary.id == report.id
        assert summary.created_at == report.created_at
        assert summary.tier == report.tier
        assert summary.total_prompts == 0
        assert summary.prompts_with_hard_interventions == 0
        assert summary.action_schema_valid_rate is None
        assert summary.file_path is None

    def test_summary_with_aggregates(self) -> None:
        aggregates = AggregateMetrics(
            total_prompts=15,
            prompts_with_hard_interventions=2,
            prompts_with_any_interventions=5,
            action_schema_valid_rate=0.9,
        )
        report = StabilitySuiteReport(aggregates=aggregates)
        summary = report.summary()
        assert summary.total_prompts == 15
        assert summary.prompts_with_hard_interventions == 2
        assert summary.action_schema_valid_rate == 0.9

    def test_summary_with_file_path(self) -> None:
        report = StabilitySuiteReport()
        p = Path("/reports/stability-001.json")
        summary = report.summary(file_path=p)
        assert summary.file_path == p

    def test_summary_preserves_adapter_info(self) -> None:
        aid = uuid4()
        report = StabilitySuiteReport(
            adapter_id=aid,
            adapter_name="test-adapter",
            base_model_id="LFM2-350M",
        )
        summary = report.summary()
        assert summary.adapter_id == aid
        assert summary.adapter_name == "test-adapter"
        assert summary.base_model_id == "LFM2-350M"

    def test_to_dict(self) -> None:
        report = StabilitySuiteReport(
            tier=StabilitySuiteTier.QUICK,
            system_context="test",
            adapter_name="my-adapter",
        )
        d = report.to_dict()
        assert d["tier"] == "quick"
        assert d["system_context"] == "test"
        assert d["adapter_name"] == "my-adapter"
        assert isinstance(d["id"], str)
        assert isinstance(d["created_at"], str)
        assert d["prompt_results"] == []
        assert d["aggregates"] is None

    def test_to_dict_with_aggregates(self) -> None:
        aggregates = AggregateMetrics(
            total_prompts=5,
            prompts_with_hard_interventions=0,
            prompts_with_any_interventions=1,
        )
        report = StabilitySuiteReport(aggregates=aggregates)
        d = report.to_dict()
        assert d["aggregates"] is not None
        assert d["aggregates"]["total_prompts"] == 5

    def test_to_dict_with_adapter_id(self) -> None:
        aid = UUID("12345678-1234-5678-1234-567812345678")
        report = StabilitySuiteReport(adapter_id=aid)
        d = report.to_dict()
        assert d["adapter_id"] == str(aid)
