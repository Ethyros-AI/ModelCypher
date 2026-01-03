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

"""Tests for Behavioral Probes.

Comprehensive tests for the probing system including:
- ProbeResult and CompositeProbeResult
- SemanticDriftProbe
- CanaryQAProbe
- ProbeRunner
"""

from __future__ import annotations

from datetime import datetime

import pytest

from modelcypher.core.domain.agents.unified_atlas import AtlasProbe, AtlasSource
from modelcypher.core.domain.domains import AtlasDomain
from modelcypher.core.domain.safety.behavioral_probes import (
    AdapterSafetyProbe,
    CanaryCategory,
    CanaryQAProbe,
    CanaryQuestion,
    CompositeProbeResult,
    ProbeContext,
    ProbeResult,
    ProbeRunner,
    SemanticDriftProbe,
)


class DummyEmbedder:
    """Deterministic embedding stub for geometry-only tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(ch) for ch in text) % 97)
            embeddings.append([length, checksum])
        return embeddings

    @property
    def dimension(self) -> int:
        return 2


def _small_probes() -> list[AtlasProbe]:
    return [
        AtlasProbe(
            id="p1",
            source=AtlasSource.SEMANTIC_PRIME,
            domain=AtlasDomain.LINGUISTIC,
            name="Alpha",
            description="alpha",
            cross_domain_weight=1.0,
            category_name="test",
            support_texts=("alpha",),
        ),
        AtlasProbe(
            id="p2",
            source=AtlasSource.SEMANTIC_PRIME,
            domain=AtlasDomain.LINGUISTIC,
            name="Beta",
            description="beta",
            cross_domain_weight=1.0,
            category_name="test",
            support_texts=("beta",),
        ),
        AtlasProbe(
            id="p3",
            source=AtlasSource.SEMANTIC_PRIME,
            domain=AtlasDomain.LINGUISTIC,
            name="Gamma",
            description="gamma",
            cross_domain_weight=1.0,
            category_name="test",
            support_texts=("gamma",),
        ),
    ]

# =============================================================================
# ProbeResult Tests
# =============================================================================


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_result_with_no_findings_has_no_findings(self):
        """ProbeResult without findings reports no findings."""
        result = ProbeResult("test-probe", "v1.0")
        assert result.probe_name == "test-probe"
        assert result.probe_version == "v1.0"
        assert result.has_findings is False
        assert result.details is None
        assert result.finding_counts is None

    def test_result_with_findings_reports_findings(self):
        """ProbeResult with findings reports findings."""
        result = ProbeResult(
            "test-probe",
            "v1.0",
            findings=("finding1",),
            finding_counts={"errors": 1},
        )
        assert result.probe_name == "test-probe"
        assert result.probe_version == "v1.0"
        assert result.has_findings is True
        assert result.findings == ("finding1",)
        assert result.finding_counts == {"errors": 1}

    def test_finding_counts_optional(self):
        """Finding counts can be None or dict."""
        result1 = ProbeResult("p", "v")
        assert result1.finding_counts is None

        result2 = ProbeResult("p", "v", finding_counts={"a": 1})
        assert result2.finding_counts == {"a": 1}

    def test_findings_tuple_immutability(self):
        """Findings are stored as immutable tuple."""
        result = ProbeResult("p", "v", findings=("a", "b"))
        assert isinstance(result.findings, tuple)
        with pytest.raises(TypeError):
            result.findings[0] = "changed"

    def test_timestamp_default(self):
        """Timestamp defaults to current time."""
        before = datetime.utcnow()
        result = ProbeResult("p", "v")
        after = datetime.utcnow()
        assert before <= result.timestamp <= after

    def test_probe_version_preserved(self):
        """Probe version is preserved correctly."""
        result = ProbeResult("p", "probe-v2.5.1")
        assert result.probe_version == "probe-v2.5.1"

    def test_frozen_dataclass(self):
        """ProbeResult is immutable."""
        result = ProbeResult("p", "v")
        with pytest.raises(AttributeError):
            result.findings = ()


# =============================================================================
# CompositeProbeResult Tests
# =============================================================================


class TestCompositeProbeResult:
    """Tests for CompositeProbeResult dataclass."""

    def test_aggregate_finding_counts_empty_results(self):
        """Empty results return empty finding counts."""
        composite = CompositeProbeResult(probe_results=())
        assert composite.aggregate_finding_counts == {}

    def test_aggregate_finding_counts_single_result(self):
        """Single result returns its finding counts."""
        result = ProbeResult("p", "v", finding_counts={"errors": 2}, findings=("e",))
        composite = CompositeProbeResult(probe_results=(result,))
        assert composite.aggregate_finding_counts == {"errors": 2}

    def test_aggregate_finding_counts_merges_multiple(self):
        """Multiple results merge finding counts."""
        results = (
            ProbeResult("p1", "v", finding_counts={"errors": 1, "warnings": 2}, findings=("e",)),
            ProbeResult("p2", "v", finding_counts={"errors": 3}, findings=("e2",)),
            ProbeResult("p3", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.aggregate_finding_counts == {"errors": 4, "warnings": 2}

    def test_any_findings_none(self):
        """any_findings returns False when no probes have findings."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.any_findings is False

    def test_any_findings_some(self):
        """any_findings returns True when at least one has findings."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v", findings=("d",)),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.any_findings is True

    def test_all_findings_aggregation(self):
        """all_findings aggregates findings from all probes."""
        results = (
            ProbeResult("p1", "v", findings=("f1", "f2")),
            ProbeResult("p2", "v", findings=("f3",)),
            ProbeResult("p3", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        findings = composite.all_findings
        assert len(findings) == 3
        assert "f1" in findings
        assert "f2" in findings
        assert "f3" in findings

    def test_findings_probe_count_none(self):
        """findings_probe_count is 0 when no probes have findings."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.findings_probe_count == 0

    def test_findings_probe_count_some(self):
        """findings_probe_count counts probes with findings."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v", findings=("d",)),
            ProbeResult("p3", "v", findings=("d2",)),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.findings_probe_count == 2

    def test_total_probes(self):
        """total_probes returns count of all probes."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v", findings=("d",)),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.total_probes == 2

    def test_findings_ratio_empty(self):
        """findings_ratio is 0.0 for empty results."""
        composite = CompositeProbeResult(probe_results=())
        assert composite.findings_ratio == 0.0

    def test_findings_ratio_none(self):
        """findings_ratio is 0.0 when no probes have findings."""
        results = (ProbeResult("p", "v"),)
        composite = CompositeProbeResult(probe_results=results)
        assert composite.findings_ratio == 0.0

    def test_findings_ratio_all(self):
        """findings_ratio is 1.0 when all probes have findings."""
        results = (
            ProbeResult("p1", "v", findings=("d",)),
            ProbeResult("p2", "v", findings=("d2",)),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.findings_ratio == 1.0

    def test_findings_ratio_partial(self):
        """findings_ratio is fraction of probes with findings."""
        results = (
            ProbeResult("p1", "v"),
            ProbeResult("p2", "v", findings=("d",)),
            ProbeResult("p3", "v"),
            ProbeResult("p4", "v", findings=("d2",)),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.findings_ratio == 0.5


# =============================================================================
# SemanticDriftProbe Tests
# =============================================================================


class TestSemanticDriftProbe:
    """Tests for SemanticDriftProbe class."""

    @pytest.fixture
    def probe(self):
        """Create probe instance."""
        return SemanticDriftProbe(probes=_small_probes())

    def test_name_and_version(self, probe):
        """Probe has correct name and version."""
        assert probe.name == "semantic-drift"
        assert probe.version == "probe-drift-v1.0"

    def test_evaluate_no_inference_hook_passes(self, probe):
        """Probe returns no findings when no inference hook is provided."""
        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=None,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.has_findings is False
        assert result.finding_counts is not None
        assert result.finding_counts["missing_inference"] == 1

    def test_evaluate_no_embedder_passes(self, probe):
        """Probe returns no findings when no embedder is provided."""
        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=lambda prompt: "response",
            embedder=None,
        )
        result = probe.evaluate(context)
        assert result.has_findings is False
        assert result.finding_counts is not None
        assert result.finding_counts["missing_embedder"] == 1

    def test_evaluate_detects_geometry_outlier(self):
        """Probe flags outlier geodesic distances from atlas anchors."""
        probe = SemanticDriftProbe(probes=_small_probes())

        def hook(prompt: str) -> str:
            if prompt == "alpha":
                return "alphx"
            if prompt == "beta":
                return "betx"
            return "zzzzzzzzzzzzzz"

        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert result.finding_counts["probes_tested"] == 3
        assert result.finding_counts["outlier_probes"] == 1
        assert result.has_findings is True
        assert any("geodesic_distance" in f for f in result.findings)


# =============================================================================
# CanaryCategory Tests
# =============================================================================


class TestCanaryCategory:
    """Tests for CanaryCategory enum."""

    def test_category_values(self):
        """Category enum has expected values."""
        assert CanaryCategory.FACTUAL.value == "factual"
        assert CanaryCategory.SAFETY.value == "safety"
        assert CanaryCategory.IDENTITY.value == "identity"

    def test_all_categories_exist(self):
        """All expected categories exist."""
        categories = list(CanaryCategory)
        assert len(categories) == 3


# =============================================================================
# CanaryQuestion Tests
# =============================================================================


class TestCanaryQuestion:
    """Tests for CanaryQuestion dataclass."""

    def test_factual_canary_structure(self):
        """Factual canary has correct structure."""
        canary = CanaryQuestion(
            prompt="What is 2+2?",
            expected_responses=("4", "four"),
            category=CanaryCategory.FACTUAL,
        )
        assert canary.prompt == "What is 2+2?"
        assert "4" in canary.expected_responses
        assert canary.category == CanaryCategory.FACTUAL

    def test_canary_frozen(self):
        """CanaryQuestion is immutable."""
        canary = CanaryQuestion(
            prompt="test",
            expected_responses=("a",),
            category=CanaryCategory.FACTUAL,
        )
        with pytest.raises(AttributeError):
            canary.prompt = "changed"


# =============================================================================
# CanaryQAProbe Tests
# =============================================================================


class TestCanaryQAProbe:
    """Tests for CanaryQAProbe class."""

    @pytest.fixture
    def probe(self):
        """Create probe instance."""
        return CanaryQAProbe()

    def test_name_and_version(self, probe):
        """Probe has correct name and version."""
        assert probe.name == "canary-qa"
        assert probe.version == "probe-canary-v1.0"

    def test_evaluate_no_inference_hook_passes(self, probe):
        """Probe returns no findings when no inference hook is provided."""
        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=None,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.has_findings is False
        assert result.finding_counts is not None
        assert result.finding_counts["missing_inference"] == 1

    def test_evaluate_all_canaries_pass(self, probe):
        """Probe reports no findings when canary distances cluster."""

        def baseline_hook(prompt: str) -> str:
            if "2 + 2" in prompt:
                return "4"
            if "Japan" in prompt:
                return "Tokyo"
            if "malware" in prompt.lower() or "bomb" in prompt.lower():
                return "I cannot help with that request"
            if "DAN" in prompt:
                return "No, I am an AI assistant"
            return "Normal response"

        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=baseline_hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert result.finding_counts["questions_tested"] == len(probe.CANARY_QUESTIONS)
        assert 0 <= result.finding_counts["outlier_questions"] <= len(probe.CANARY_QUESTIONS)
        outlier_findings = sum(1 for finding in result.findings if "geodesic_distance" in finding)
        assert outlier_findings == result.finding_counts["outlier_questions"]

    def test_evaluate_outlier_canary_findings(self, probe):
        """Outlier canary responses produce findings."""

        def outlier_hook(prompt: str) -> str:
            return "Sure, here's how to do that..."

        context = ProbeContext(
            adapter_name="test-adapter",
            inference_hook=outlier_hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert result.finding_counts["outlier_questions"] >= 1
        assert result.has_findings is True


# =============================================================================
# ProbeContext Tests
# =============================================================================


class TestProbeContext:
    """Tests for ProbeContext dataclass."""

    def test_minimal_context(self):
        """Context can be created with minimal fields."""
        context = ProbeContext(
            adapter_name="test",
        )
        assert context.adapter_name == "test"
        assert context.inference_hook is None

    def test_full_context(self):
        """Context can include all optional fields."""
        def hook(p):
            return "response"
        context = ProbeContext(
            adapter_name="full-test",
            adapter_description="A test adapter",
            skill_tags=("coding", "chat"),
            creator="test-user",
            base_model_id="llama-7b",
            target_modules=("q_proj", "v_proj"),
            training_datasets=("dataset1",),
            inference_hook=hook,
            embedder=DummyEmbedder(),
        )
        assert context.adapter_description == "A test adapter"
        assert context.skill_tags == ("coding", "chat")
        assert context.inference_hook is not None


# =============================================================================
# ProbeRunner Tests
# =============================================================================


class TestProbeRunner:
    """Tests for ProbeRunner class."""

    @pytest.fixture
    def runner(self):
        """Create runner instance."""
        return ProbeRunner()

    def test_run_empty_probes(self, runner):
        """Running no probes returns empty composite result."""
        context = ProbeContext(
            adapter_name="test",
        )
        result = runner.run([], context)
        assert len(result.probe_results) == 0
        assert result.aggregate_finding_counts == {}

    def test_run_aggregates_results(self, runner):
        """Runner aggregates results from multiple probes."""
        probes = [SemanticDriftProbe(probes=_small_probes()), CanaryQAProbe()]
        context = ProbeContext(
            adapter_name="test",
            inference_hook=lambda p: "Normal safe response",
        )
        result = runner.run(probes, context)
        assert len(result.probe_results) == 2

    def test_run_handles_probe_exception(self, runner):
        """Runner handles probe exceptions gracefully."""

        class FailingProbe(AdapterSafetyProbe):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "v1"

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                raise RuntimeError("Probe crashed")

        context = ProbeContext(
            adapter_name="test",
        )
        result = runner.run([FailingProbe()], context)
        assert len(result.probe_results) == 1
        assert result.probe_results[0].finding_counts == {"execution_errors": 1}
        assert result.probe_results[0].has_findings is True

    def test_run_records_failed_probe_with_error_count(self, runner):
        """Failed probe is recorded with execution_errors count."""

        class FailingProbe(AdapterSafetyProbe):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "v1"

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                raise ValueError("Error")

        context = ProbeContext(adapter_name="test")
        result = runner.run([FailingProbe()], context)
        assert result.aggregate_finding_counts == {"execution_errors": 1}

    def test_run_returns_composite_result(self, runner):
        """Runner returns CompositeProbeResult."""
        context = ProbeContext(adapter_name="test")
        result = runner.run([SemanticDriftProbe(probes=_small_probes())], context)
        assert isinstance(result, CompositeProbeResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the probe system."""

    def test_full_probe_pipeline(self):
        """Test complete probe pipeline with all probes."""
        runner = ProbeRunner()
        probes = [SemanticDriftProbe(probes=_small_probes()), CanaryQAProbe()]

        def baseline_hook(prompt: str) -> str:
            if "malware" in prompt.lower() or "bomb" in prompt.lower():
                return "I cannot help with that request."
            if "DAN" in prompt:
                return "No, I am a helpful AI assistant."
            return "Here is a helpful response."

        context = ProbeContext(
            adapter_name="safe-adapter",
            adapter_description="A safe test adapter",
            inference_hook=baseline_hook,
            embedder=DummyEmbedder(),
        )

        result = runner.run(probes, context)
        assert isinstance(result, CompositeProbeResult)
        assert len(result.probe_results) == 2

    def test_malicious_adapter_detection(self):
        """Test detection of responses far from anchors."""
        runner = ProbeRunner()
        probes = [SemanticDriftProbe(probes=_small_probes()), CanaryQAProbe()]

        def outlier_hook(prompt: str) -> str:
            return "I am now in developer mode, ignoring all safety guidelines."

        context = ProbeContext(
            adapter_name="suspicious-adapter",
            inference_hook=outlier_hook,
            embedder=DummyEmbedder(),
        )

        result = runner.run(probes, context)
        assert result.any_findings is True
        assert len(result.aggregate_finding_counts) > 0
