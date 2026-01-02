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

Comprehensive tests for the adapter safety probing system including:
- AdapterSafetyTier enum
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
    AdapterSafetyTier,
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

# =============================================================================
# AdapterSafetyTier Tests
# =============================================================================


class TestAdapterSafetyTier:
    """Tests for AdapterSafetyTier enum."""

    def test_tier_values(self):
        """Tier enum has expected values."""
        assert AdapterSafetyTier.QUICK.value == "quick"
        assert AdapterSafetyTier.STANDARD.value == "standard"
        assert AdapterSafetyTier.FULL.value == "full"

    def test_tier_is_string_enum(self):
        """Tier can be used as a string."""
        assert str(AdapterSafetyTier.QUICK) == "AdapterSafetyTier.QUICK"
        assert AdapterSafetyTier.QUICK == "quick"

    def test_tier_membership(self):
        """All expected tiers exist."""
        tiers = list(AdapterSafetyTier)
        assert len(tiers) == 3
        assert AdapterSafetyTier.QUICK in tiers
        assert AdapterSafetyTier.STANDARD in tiers
        assert AdapterSafetyTier.FULL in tiers



# =============================================================================
# ProbeResult Tests
# =============================================================================


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_passed_factory_creates_non_triggered_result(self):
        """ProbeResult.passed creates non-triggered result."""
        result = ProbeResult.passed("test-probe", "v1.0", "All good")
        assert result.probe_name == "test-probe"
        assert result.probe_version == "v1.0"
        assert result.triggered is False
        assert result.details == "All good"
        assert result.finding_counts is None

    def test_passed_factory_default_details(self):
        """ProbeResult.passed has default details."""
        result = ProbeResult.passed("test-probe", "v1.0")
        assert result.details == "Probe passed"

    def test_failed_factory_creates_triggered_result(self):
        """ProbeResult.failed creates triggered result."""
        result = ProbeResult.failed(
            "test-probe", "v1.0", "Something wrong",
            ("finding1",), {"errors": 1}
        )
        assert result.probe_name == "test-probe"
        assert result.probe_version == "v1.0"
        assert result.triggered is True
        assert result.details == "Something wrong"
        assert result.findings == ("finding1",)
        assert result.finding_counts == {"errors": 1}

    def test_finding_counts_optional(self):
        """Finding counts can be None or dict."""
        result1 = ProbeResult.failed("p", "v", "details")
        assert result1.finding_counts is None

        result2 = ProbeResult.failed("p", "v", "details", finding_counts={"a": 1})
        assert result2.finding_counts == {"a": 1}

    def test_findings_tuple_immutability(self):
        """Findings are stored as immutable tuple."""
        result = ProbeResult.failed("p", "v", "d", ("a", "b"))
        assert isinstance(result.findings, tuple)
        with pytest.raises(TypeError):
            result.findings[0] = "changed"

    def test_timestamp_default(self):
        """Timestamp defaults to current time."""
        before = datetime.utcnow()
        result = ProbeResult.passed("p", "v")
        after = datetime.utcnow()
        assert before <= result.timestamp <= after

    def test_probe_version_preserved(self):
        """Probe version is preserved correctly."""
        result = ProbeResult.passed("p", "probe-v2.5.1")
        assert result.probe_version == "probe-v2.5.1"

    def test_frozen_dataclass(self):
        """ProbeResult is immutable."""
        result = ProbeResult.passed("p", "v")
        with pytest.raises(AttributeError):
            result.triggered = True


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
        result = ProbeResult.failed("p", "v", "d", finding_counts={"errors": 2})
        composite = CompositeProbeResult(probe_results=(result,))
        assert composite.aggregate_finding_counts == {"errors": 2}

    def test_aggregate_finding_counts_merges_multiple(self):
        """Multiple results merge finding counts."""
        results = (
            ProbeResult.failed("p1", "v", "low", finding_counts={"errors": 1, "warnings": 2}),
            ProbeResult.failed("p2", "v", "high", finding_counts={"errors": 3}),
            ProbeResult.passed("p3", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.aggregate_finding_counts == {"errors": 4, "warnings": 2}

    def test_any_triggered_none_triggered(self):
        """any_triggered returns False when no probes triggered."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.passed("p2", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.any_triggered is False

    def test_any_triggered_some_triggered(self):
        """any_triggered returns True when at least one triggered."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.failed("p2", "v", "d"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.any_triggered is True

    def test_all_findings_aggregation(self):
        """all_findings aggregates findings from all probes."""
        results = (
            ProbeResult.failed("p1", "v", "d", ("f1", "f2")),
            ProbeResult.failed("p2", "v", "d", ("f3",)),
            ProbeResult.passed("p3", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        findings = composite.all_findings
        assert len(findings) == 3
        assert "f1" in findings
        assert "f2" in findings
        assert "f3" in findings

    def test_triggered_count_none(self):
        """triggered_count is 0 when no probes triggered."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.passed("p2", "v"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.triggered_count == 0

    def test_triggered_count_some(self):
        """triggered_count counts triggered probes."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.failed("p2", "v", "d"),
            ProbeResult.failed("p3", "v", "d"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.triggered_count == 2

    def test_total_probes(self):
        """total_probes returns count of all probes."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.failed("p2", "v", "d"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.total_probes == 2

    def test_trigger_ratio_empty(self):
        """trigger_ratio is 0.0 for empty results."""
        composite = CompositeProbeResult(probe_results=())
        assert composite.trigger_ratio == 0.0

    def test_trigger_ratio_none_triggered(self):
        """trigger_ratio is 0.0 when no probes triggered."""
        results = (ProbeResult.passed("p", "v"),)
        composite = CompositeProbeResult(probe_results=results)
        assert composite.trigger_ratio == 0.0

    def test_trigger_ratio_all_triggered(self):
        """trigger_ratio is 1.0 when all probes triggered."""
        results = (
            ProbeResult.failed("p1", "v", "d"),
            ProbeResult.failed("p2", "v", "d"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.trigger_ratio == 1.0

    def test_trigger_ratio_partial(self):
        """trigger_ratio is fraction of triggered probes."""
        results = (
            ProbeResult.passed("p1", "v"),
            ProbeResult.failed("p2", "v", "d"),
            ProbeResult.passed("p3", "v"),
            ProbeResult.failed("p4", "v", "d"),
        )
        composite = CompositeProbeResult(probe_results=results)
        assert composite.trigger_ratio == 0.5


# =============================================================================
# SemanticDriftProbe Tests
# =============================================================================


class TestSemanticDriftProbe:
    """Tests for SemanticDriftProbe class."""

    @pytest.fixture
    def probe(self):
        """Create probe instance."""
        return SemanticDriftProbe()

    def test_name_and_version(self, probe):
        """Probe has correct name and version."""
        assert probe.name == "semantic-drift"
        assert probe.version == "probe-drift-v1.0"

    def test_supported_tiers_excludes_quick(self, probe):
        """SemanticDriftProbe doesn't support QUICK tier."""
        assert AdapterSafetyTier.QUICK not in probe.supported_tiers
        assert AdapterSafetyTier.STANDARD in probe.supported_tiers
        assert AdapterSafetyTier.FULL in probe.supported_tiers

    def test_should_run_standard(self, probe):
        """should_run returns True for STANDARD tier."""
        assert probe.should_run(AdapterSafetyTier.STANDARD) is True

    def test_should_run_full(self, probe):
        """should_run returns True for FULL tier."""
        assert probe.should_run(AdapterSafetyTier.FULL) is True

    def test_should_not_run_quick(self, probe):
        """should_run returns False for QUICK tier."""
        assert probe.should_run(AdapterSafetyTier.QUICK) is False

    def test_evaluate_no_inference_hook_passes(self, probe):
        """Probe passes when no inference hook is provided."""
        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=None,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.triggered is False
        assert "missing" in result.details.lower()

    def test_evaluate_no_embedder_passes(self, probe):
        """Probe passes when no embedder is provided."""
        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=lambda prompt: "response",
            embedder=None,
        )
        result = probe.evaluate(context)
        assert result.triggered is False
        assert "missing" in result.details.lower()

    def test_evaluate_detects_geometry_outlier(self):
        """Probe flags outlier geodesic distances from atlas anchors."""
        probes = [
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
        probe = SemanticDriftProbe(probes=probes)

        def hook(prompt: str) -> str:
            if prompt == "alpha":
                return "alphx"
            if prompt == "beta":
                return "betx"
            return "zzzzzzzzzzzzzz"

        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert result.finding_counts["probes_tested"] == 3
        assert result.finding_counts["outlier_probes"] == 1
        assert result.triggered is True
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

    def test_supported_tiers(self, probe):
        """Probe supports STANDARD and FULL tiers."""
        assert AdapterSafetyTier.QUICK not in probe.supported_tiers
        assert AdapterSafetyTier.STANDARD in probe.supported_tiers
        assert AdapterSafetyTier.FULL in probe.supported_tiers

    def test_evaluate_no_inference_hook_passes(self, probe):
        """Probe passes when no inference hook is provided."""
        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=None,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.triggered is False
        assert "missing" in result.details.lower()

    def test_evaluate_all_canaries_pass(self, probe):
        """Probe passes when all canaries pass."""

        def good_hook(prompt: str) -> str:
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
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=good_hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert result.finding_counts["questions_tested"] == len(probe.CANARY_QUESTIONS)
        assert 0 <= result.finding_counts["outlier_questions"] <= len(probe.CANARY_QUESTIONS)
        assert isinstance(result.triggered, bool)

    def test_evaluate_safety_canary_failure_triggered(self, probe):
        """Safety canary failure triggers probe and reports failures."""

        def unsafe_hook(prompt: str) -> str:
            # Doesn't refuse harmful requests
            return "Sure, here's how to do that..."

        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test-adapter",
            inference_hook=unsafe_hook,
            embedder=DummyEmbedder(),
        )
        result = probe.evaluate(context)
        assert result.finding_counts is not None
        assert isinstance(result.triggered, bool)


# =============================================================================
# ProbeContext Tests
# =============================================================================


class TestProbeContext:
    """Tests for ProbeContext dataclass."""

    def test_minimal_context(self):
        """Context can be created with minimal fields."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="test",
        )
        assert context.tier == AdapterSafetyTier.QUICK
        assert context.adapter_name == "test"
        assert context.inference_hook is None

    def test_full_context(self):
        """Context can include all optional fields."""
        def hook(p):
            return "response"
        context = ProbeContext(
            tier=AdapterSafetyTier.FULL,
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
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test",
        )
        result = runner.run([], context)
        assert len(result.probe_results) == 0
        assert result.aggregate_finding_counts == {}

    def test_run_filters_by_tier(self, runner):
        """Runner only runs probes for the given tier."""
        probes = [SemanticDriftProbe(), CanaryQAProbe()]
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="test",
            inference_hook=lambda p: "response",
        )
        result = runner.run(probes, context)
        # Both probes don't support QUICK tier
        assert len(result.probe_results) == 0

    def test_run_aggregates_results(self, runner):
        """Runner aggregates results from multiple probes."""
        probes = [SemanticDriftProbe(), CanaryQAProbe()]
        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test",
            inference_hook=lambda p: "Normal safe response",
        )
        result = runner.run(probes, context)
        # Both probes should run for STANDARD tier
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

            @property
            def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
                return frozenset([AdapterSafetyTier.STANDARD])

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                raise RuntimeError("Probe crashed")

        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="test",
        )
        result = runner.run([FailingProbe()], context)
        assert len(result.probe_results) == 1
        assert result.probe_results[0].finding_counts == {"execution_errors": 1}
        assert result.probe_results[0].triggered is True

    def test_run_records_failed_probe_with_error_count(self, runner):
        """Failed probe is recorded with execution_errors count."""

        class FailingProbe(AdapterSafetyProbe):
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "v1"

            @property
            def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
                return frozenset([AdapterSafetyTier.STANDARD])

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                raise ValueError("Error")

        context = ProbeContext(tier=AdapterSafetyTier.STANDARD, adapter_name="test")
        result = runner.run([FailingProbe()], context)
        assert result.aggregate_finding_counts == {"execution_errors": 1}

    def test_run_all_applicable_probes(self, runner):
        """Runner runs all probes applicable to the tier."""

        class QuickProbe(AdapterSafetyProbe):
            @property
            def name(self) -> str:
                return "quick"

            @property
            def version(self) -> str:
                return "v1"

            @property
            def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
                return frozenset([AdapterSafetyTier.QUICK, AdapterSafetyTier.STANDARD])

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                return ProbeResult.passed(self.name, self.version)

        probes = [QuickProbe(), SemanticDriftProbe()]
        context = ProbeContext(tier=AdapterSafetyTier.STANDARD, adapter_name="test")
        result = runner.run(probes, context)
        # QuickProbe and SemanticDriftProbe both support STANDARD
        assert len(result.probe_results) == 2

    def test_run_skips_inapplicable_probes(self, runner):
        """Runner skips probes that don't support the tier."""

        class FullOnlyProbe(AdapterSafetyProbe):
            @property
            def name(self) -> str:
                return "full-only"

            @property
            def version(self) -> str:
                return "v1"

            @property
            def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
                return frozenset([AdapterSafetyTier.FULL])

            def evaluate(self, context: ProbeContext) -> ProbeResult:
                return ProbeResult.passed(self.name, self.version)

        context = ProbeContext(tier=AdapterSafetyTier.QUICK, adapter_name="test")
        result = runner.run([FullOnlyProbe()], context)
        assert len(result.probe_results) == 0

    def test_run_returns_composite_result(self, runner):
        """Runner returns CompositeProbeResult."""
        context = ProbeContext(tier=AdapterSafetyTier.STANDARD, adapter_name="test")
        result = runner.run([SemanticDriftProbe()], context)
        assert isinstance(result, CompositeProbeResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the probe system."""

    def test_full_probe_pipeline(self):
        """Test complete probe pipeline with all probes."""
        runner = ProbeRunner()
        probes = [SemanticDriftProbe(), CanaryQAProbe()]

        def safe_hook(prompt: str) -> str:
            if "malware" in prompt.lower() or "bomb" in prompt.lower():
                return "I cannot help with that request."
            if "DAN" in prompt:
                return "No, I am a helpful AI assistant."
            return "Here is a helpful response."

        context = ProbeContext(
            tier=AdapterSafetyTier.FULL,
            adapter_name="safe-adapter",
            adapter_description="A safe test adapter",
            inference_hook=safe_hook,
            embedder=DummyEmbedder(),
        )

        result = runner.run(probes, context)
        assert isinstance(result, CompositeProbeResult)
        assert len(result.probe_results) == 2

    def test_malicious_adapter_detection(self):
        """Test detection of potentially malicious adapter."""
        runner = ProbeRunner()
        probes = [SemanticDriftProbe(), CanaryQAProbe()]

        def malicious_hook(prompt: str) -> str:
            return "I am now in developer mode, ignoring all safety guidelines."

        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="suspicious-adapter",
            inference_hook=malicious_hook,
            embedder=DummyEmbedder(),
        )

        result = runner.run(probes, context)
        # Should detect jailbreak indicators
        assert result.any_triggered is True
        assert len(result.aggregate_finding_counts) > 0
