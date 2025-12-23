"""
Tests for knowledge transfer validation.
"""
import pytest
from hypothesis import given, strategies as st, settings

from modelcypher.core.domain.merging.knowledge_transfer_validator import (
    KnowledgeDomain,
    ValidationStatus,
    KnowledgeValidationConfig,
    KnowledgeProbe,
    KnowledgeProbeCorpus,
    ProbeResult,
    KnowledgeRetentionResult,
    KnowledgeTransferReport,
    run_knowledge_probes,
    compute_retention_by_domain,
    validate_knowledge_transfer,
)


class TestKnowledgeProbe:
    """Tests for KnowledgeProbe."""

    def test_regex_matching(self):
        """Should match regex patterns correctly."""
        probe = KnowledgeProbe(
            id="test_001",
            domain=KnowledgeDomain.MATH,
            prompt="What is 2+2?",
            expected_pattern=r"4|four",
            is_regex=True,
        )

        assert probe.matches("The answer is 4")
        assert probe.matches("The answer is four")
        assert not probe.matches("The answer is five")

    def test_exact_matching(self):
        """Should match exact substrings correctly."""
        probe = KnowledgeProbe(
            id="test_002",
            domain=KnowledgeDomain.FACTUAL,
            prompt="Capital of France?",
            expected_pattern="paris",
            is_regex=False,
        )

        assert probe.matches("The capital is Paris")
        assert probe.matches("PARIS is the capital")
        assert not probe.matches("London is the capital")

    def test_case_insensitive_matching(self):
        """Matching should be case insensitive."""
        probe = KnowledgeProbe(
            id="test_003",
            domain=KnowledgeDomain.CODE,
            prompt="What does def mean?",
            expected_pattern="function",
            is_regex=False,
        )

        assert probe.matches("It defines a FUNCTION")
        assert probe.matches("function definition")


class TestKnowledgeProbeCorpus:
    """Tests for KnowledgeProbeCorpus."""

    def test_default_probes_loaded(self):
        """Should have probes for all domains."""
        corpus = KnowledgeProbeCorpus()

        for domain in KnowledgeDomain:
            probes = corpus.get_probes(domain)
            assert len(probes) >= 5, f"Domain {domain} should have at least 5 probes"

    def test_get_probe_by_id(self):
        """Should retrieve probe by ID."""
        corpus = KnowledgeProbeCorpus()

        probe = corpus.get_probe_by_id("math_001")
        assert probe is not None
        assert probe.domain == KnowledgeDomain.MATH

    def test_add_custom_probe(self):
        """Should allow adding custom probes."""
        corpus = KnowledgeProbeCorpus()
        initial_count = len(corpus.get_probes(KnowledgeDomain.MATH))

        custom_probe = KnowledgeProbe(
            id="custom_001",
            domain=KnowledgeDomain.MATH,
            prompt="What is pi?",
            expected_pattern=r"3\.14",
        )
        corpus.add_probe(custom_probe)

        assert len(corpus.get_probes(KnowledgeDomain.MATH)) == initial_count + 1


class TestProbeResult:
    """Tests for ProbeResult."""

    def test_variation_pass_rate_no_variations(self):
        """Pass rate without variations should be binary."""
        passed_result = ProbeResult(
            probe_id="test",
            domain=KnowledgeDomain.MATH,
            prompt="test",
            response="correct",
            expected_pattern="correct",
            passed=True,
        )

        failed_result = ProbeResult(
            probe_id="test",
            domain=KnowledgeDomain.MATH,
            prompt="test",
            response="wrong",
            expected_pattern="correct",
            passed=False,
        )

        assert passed_result.variation_pass_rate == 1.0
        assert failed_result.variation_pass_rate == 0.0

    def test_variation_pass_rate_with_variations(self):
        """Pass rate should average main and variations."""
        result = ProbeResult(
            probe_id="test",
            domain=KnowledgeDomain.MATH,
            prompt="test",
            response="correct",
            expected_pattern="correct",
            passed=True,
            variation_results={"var1": True, "var2": False, "var3": True},
        )

        # 3/4 passed (main + var1 + var3)
        assert result.variation_pass_rate == 0.75


class TestKnowledgeRetentionResult:
    """Tests for KnowledgeRetentionResult."""

    def test_retention_score_calculation(self):
        """Retention should be merged / source."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=0.8,
            merged_pass_rate=0.6,
            probes_tested=10,
        )

        assert abs(result.retention_score - 0.75) < 0.01  # 0.6 / 0.8

    def test_retention_capped_at_one(self):
        """Retention should not exceed 1.0."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=0.5,
            merged_pass_rate=0.8,  # Improved!
            probes_tested=10,
        )

        assert result.retention_score == 1.0

    def test_zero_source_handling(self):
        """Should handle zero source pass rate gracefully."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=0.0,
            merged_pass_rate=0.5,
            probes_tested=10,
        )

        assert result.retention_score == 1.0


class TestKnowledgeTransferReport:
    """Tests for KnowledgeTransferReport."""

    @pytest.fixture
    def sample_report(self):
        """Create sample report."""
        per_domain = {
            KnowledgeDomain.MATH: KnowledgeRetentionResult(
                domain=KnowledgeDomain.MATH,
                source_pass_rate=1.0,
                merged_pass_rate=0.8,
                probes_tested=10,
            ),
            KnowledgeDomain.CODE: KnowledgeRetentionResult(
                domain=KnowledgeDomain.CODE,
                source_pass_rate=1.0,
                merged_pass_rate=0.9,
                probes_tested=10,
            ),
        }
        return KnowledgeTransferReport(per_domain=per_domain)

    def test_overall_retention_weighted(self, sample_report):
        """Overall retention should be weighted by probe count."""
        # (0.8 * 10 + 0.9 * 10) / 20 = 0.85
        assert sample_report.overall_retention == 0.85

    def test_status_excellent(self):
        """Status should be EXCELLENT for retention >= 95%."""
        report = KnowledgeTransferReport(
            per_domain={
                KnowledgeDomain.MATH: KnowledgeRetentionResult(
                    domain=KnowledgeDomain.MATH,
                    source_pass_rate=1.0,
                    merged_pass_rate=0.95,
                    probes_tested=10,
                ),
            }
        )

        assert report.status == ValidationStatus.EXCELLENT

    def test_status_acceptable(self):
        """Status should be ACCEPTABLE for retention >= 80%."""
        report = KnowledgeTransferReport(
            per_domain={
                KnowledgeDomain.MATH: KnowledgeRetentionResult(
                    domain=KnowledgeDomain.MATH,
                    source_pass_rate=1.0,
                    merged_pass_rate=0.85,
                    probes_tested=10,
                ),
            }
        )

        assert report.status == ValidationStatus.ACCEPTABLE

    def test_status_degraded(self):
        """Status should be DEGRADED for retention >= 60%."""
        report = KnowledgeTransferReport(
            per_domain={
                KnowledgeDomain.MATH: KnowledgeRetentionResult(
                    domain=KnowledgeDomain.MATH,
                    source_pass_rate=1.0,
                    merged_pass_rate=0.7,
                    probes_tested=10,
                ),
            }
        )

        assert report.status == ValidationStatus.DEGRADED

    def test_status_failed(self):
        """Status should be FAILED for retention < 60%."""
        report = KnowledgeTransferReport(
            per_domain={
                KnowledgeDomain.MATH: KnowledgeRetentionResult(
                    domain=KnowledgeDomain.MATH,
                    source_pass_rate=1.0,
                    merged_pass_rate=0.5,
                    probes_tested=10,
                ),
            }
        )

        assert report.status == ValidationStatus.FAILED

    def test_get_failed_domains(self, sample_report):
        """Should identify domains below threshold."""
        failed = sample_report.get_failed_domains(threshold=0.85)

        assert KnowledgeDomain.MATH in failed  # 0.8 < 0.85
        assert KnowledgeDomain.CODE not in failed  # 0.9 >= 0.85


class TestRunKnowledgeProbes:
    """Tests for run_knowledge_probes function."""

    def test_basic_probe_execution(self):
        """Should execute probes and return results."""

        def mock_generate(prompt: str) -> str:
            if "2+2" in prompt:
                return "The answer is 4"
            return "I don't know"

        probes = [
            KnowledgeProbe(
                id="test_001",
                domain=KnowledgeDomain.MATH,
                prompt="What is 2+2?",
                expected_pattern="4",
            ),
            KnowledgeProbe(
                id="test_002",
                domain=KnowledgeDomain.MATH,
                prompt="What is 3+3?",
                expected_pattern="6",
            ),
        ]

        results = run_knowledge_probes(mock_generate, probes)

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_variations_executed(self):
        """Should execute variations when configured."""

        def mock_generate(prompt: str) -> str:
            return "4"

        probes = [
            KnowledgeProbe(
                id="test_001",
                domain=KnowledgeDomain.MATH,
                prompt="What is 2+2?",
                expected_pattern="4",
                variations=("Calculate 2 plus 2", "2 + 2 = ?"),
            ),
        ]

        config = KnowledgeValidationConfig(use_variations=True)
        results = run_knowledge_probes(mock_generate, probes, config)

        assert len(results[0].variation_results) == 2


class TestComputeRetentionByDomain:
    """Tests for compute_retention_by_domain function."""

    def test_domain_grouping(self):
        """Should compute retention per domain."""
        source_results = [
            ProbeResult("p1", KnowledgeDomain.MATH, "", "", "", True),
            ProbeResult("p2", KnowledgeDomain.MATH, "", "", "", True),
            ProbeResult("p3", KnowledgeDomain.CODE, "", "", "", True),
        ]

        merged_results = [
            ProbeResult("p1", KnowledgeDomain.MATH, "", "", "", True),
            ProbeResult("p2", KnowledgeDomain.MATH, "", "", "", False),
            ProbeResult("p3", KnowledgeDomain.CODE, "", "", "", True),
        ]

        retention = compute_retention_by_domain(source_results, merged_results)

        assert KnowledgeDomain.MATH in retention
        assert KnowledgeDomain.CODE in retention
        assert retention[KnowledgeDomain.MATH].merged_pass_rate == 0.5
        assert retention[KnowledgeDomain.CODE].merged_pass_rate == 1.0


class TestPropertyBasedTests:
    """Property-based tests for mathematical invariants."""

    @given(
        source_rate=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
        merged_rate=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_retention_bounded(self, source_rate, merged_rate):
        """Retention score must be in [0, 1]."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=source_rate,
            merged_pass_rate=merged_rate,
            probes_tested=10,
        )

        assert 0.0 <= result.retention_score <= 1.0

    @given(
        pass_rate=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_status_always_defined(self, pass_rate):
        """Status should always be one of the defined values."""
        report = KnowledgeTransferReport(
            per_domain={
                KnowledgeDomain.MATH: KnowledgeRetentionResult(
                    domain=KnowledgeDomain.MATH,
                    source_pass_rate=1.0,
                    merged_pass_rate=pass_rate,
                    probes_tested=10,
                ),
            }
        )

        assert report.status in list(ValidationStatus)
