# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Property-based tests for the knowledge transfer validator.

Uses Hypothesis to verify mathematical properties and invariants.
"""

from __future__ import annotations

from hypothesis import given, strategies as st, assume, settings

from modelcypher.core.domain.merging.knowledge_transfer_validator import (
    KnowledgeDomain,
    KnowledgeProbe,
    KnowledgeRetentionResult,
    KnowledgeTransferReport,
    ProbeResult,
)


# =============================================================================
# Strategy Definitions
# =============================================================================


# Retention scores are in [0, 1]
retention_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Positive floats for pass rates
pass_rate_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Domain strategy
domain_strategy = st.sampled_from(list(KnowledgeDomain))


@st.composite
def probe_result_strategy(draw):
    """Generate a ProbeResult with valid data."""
    domain = draw(domain_strategy)
    passed = draw(st.booleans())

    return ProbeResult(
        probe_id=draw(st.text(min_size=1, max_size=20)),
        domain=domain,
        prompt="Test prompt",
        response="Test response",
        expected_pattern="pattern",
        passed=passed,
        variation_results={},
    )


@st.composite
def retention_result_strategy(draw):
    """Generate a KnowledgeRetentionResult with valid data."""
    domain = draw(domain_strategy)
    source_rate = draw(pass_rate_strategy)
    merged_rate = draw(pass_rate_strategy)
    probes_tested = draw(st.integers(min_value=1, max_value=100))

    return KnowledgeRetentionResult(
        domain=domain,
        source_pass_rate=source_rate,
        merged_pass_rate=merged_rate,
        probes_tested=probes_tested,
        passed_probes=[f"probe_{i}" for i in range(int(merged_rate * probes_tested))],
        failed_probes=[f"probe_{i}" for i in range(int((1 - merged_rate) * probes_tested))],
    )


# =============================================================================
# Knowledge Probe Properties
# =============================================================================


class TestKnowledgeProbeProperties:
    """Property tests for KnowledgeProbe."""

    @given(
        # Use ASCII letters only to avoid unicode case folding edge cases (e.g., ß -> SS)
        response=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="")),
    )
    @settings(max_examples=30)
    def test_exact_match_is_case_insensitive(self, response: str) -> None:
        """Exact match should be case insensitive for ASCII text."""
        # Filter to ASCII letters only to avoid unicode edge cases like ß -> SS
        ascii_response = "".join(c for c in response if c.isascii() and c.isalpha())
        if not ascii_response:
            # Skip if no valid ASCII letters
            return

        # Create probe expecting lowercase of response
        probe = KnowledgeProbe(
            id="test",
            domain=KnowledgeDomain.FACTUAL,
            prompt="Test",
            expected_pattern=ascii_response.lower(),
            is_regex=False,
        )

        # Should match regardless of case for ASCII
        assert probe.matches(ascii_response) == probe.matches(ascii_response.upper())
        assert probe.matches(ascii_response) == probe.matches(ascii_response.lower())

    @given(
        text=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=30)
    def test_response_containing_pattern_matches(self, text: str) -> None:
        """Response containing the pattern should match (non-regex)."""
        assume(len(text) >= 3)
        pattern = text[:3]

        probe = KnowledgeProbe(
            id="test",
            domain=KnowledgeDomain.FACTUAL,
            prompt="Test",
            expected_pattern=pattern,
            is_regex=False,
        )

        # Full text contains pattern, should match
        assert probe.matches(text) is True

    def test_regex_pattern_matching(self) -> None:
        """Regex patterns should work correctly."""
        probe = KnowledgeProbe(
            id="test",
            domain=KnowledgeDomain.MATH,
            prompt="Test",
            expected_pattern=r"\d+",
            is_regex=True,
        )

        assert probe.matches("The answer is 42") is True
        assert probe.matches("No numbers here") is False


# =============================================================================
# Retention Result Properties
# =============================================================================


class TestKnowledgeRetentionResultProperties:
    """Property tests for KnowledgeRetentionResult."""

    @given(
        source_rate=pass_rate_strategy,
        merged_rate=pass_rate_strategy,
    )
    @settings(max_examples=50)
    def test_retention_score_in_bounds(
        self, source_rate: float, merged_rate: float
    ) -> None:
        """Retention score should be in [0, 1]."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=source_rate,
            merged_pass_rate=merged_rate,
            probes_tested=10,
        )

        assert 0.0 <= result.retention_score <= 1.0

    @given(
        merged_rate=pass_rate_strategy,
    )
    @settings(max_examples=50)
    def test_zero_source_rate_gives_full_retention(
        self, merged_rate: float
    ) -> None:
        """Zero source rate should give retention of 1.0 (no baseline to degrade from)."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=0.0,  # Near-zero source
            merged_pass_rate=merged_rate,
            probes_tested=10,
        )

        assert result.retention_score == 1.0

    @given(
        source_rate=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_perfect_merged_rate_gives_high_retention(
        self, source_rate: float
    ) -> None:
        """Merged rate >= source rate should give retention of 1.0."""
        result = KnowledgeRetentionResult(
            domain=KnowledgeDomain.MATH,
            source_pass_rate=source_rate,
            merged_pass_rate=source_rate,  # Same as source
            probes_tested=10,
        )

        assert result.retention_score == 1.0


# =============================================================================
# Knowledge Transfer Report Properties
# =============================================================================


class TestKnowledgeTransferReportProperties:
    """Property tests for KnowledgeTransferReport."""

    @given(
        domain_results=st.lists(
            retention_result_strategy(),
            min_size=1,
            max_size=6,
        ),
    )
    @settings(max_examples=30)
    def test_overall_retention_in_bounds(
        self, domain_results: list[KnowledgeRetentionResult]
    ) -> None:
        """Overall retention should be in [0, 1]."""
        # Create unique domains
        domains_used = set()
        per_domain = {}
        for result in domain_results:
            if result.domain not in domains_used:
                domains_used.add(result.domain)
                per_domain[result.domain] = result

        if not per_domain:
            return  # Skip if no valid domains

        report = KnowledgeTransferReport(per_domain=per_domain)

        assert 0.0 <= report.overall_retention <= 1.0

    @given(
        probe_results=st.lists(
            probe_result_strategy(),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_overall_pass_rate_in_bounds(
        self, probe_results: list[ProbeResult]
    ) -> None:
        """Overall pass rate should be in [0, 1]."""
        report = KnowledgeTransferReport(
            per_domain={},
            probe_results=probe_results,
        )

        assert 0.0 <= report.overall_pass_rate <= 1.0

    @given(
        probe_results=st.lists(
            probe_result_strategy(),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_pass_rate_equals_passed_count_ratio(
        self, probe_results: list[ProbeResult]
    ) -> None:
        """Pass rate should equal passed count / total count."""
        report = KnowledgeTransferReport(
            per_domain={},
            probe_results=probe_results,
        )

        passed_count = sum(1 for r in probe_results if r.passed)
        expected_rate = passed_count / len(probe_results)

        assert report.overall_pass_rate == expected_rate

    def test_empty_report_handles_gracefully(self) -> None:
        """Empty report should produce sensible defaults."""
        report = KnowledgeTransferReport(per_domain={})

        assert report.overall_retention == 0.0
        assert report.overall_pass_rate == 0.0



# =============================================================================
# Summary Output Properties
# =============================================================================


class TestKnowledgeTransferReportSummary:
    """Tests for report summary output."""

    @given(
        probe_results=st.lists(
            probe_result_strategy(),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=20)
    def test_summary_contains_required_fields(
        self, probe_results: list[ProbeResult]
    ) -> None:
        """Summary should contain all required fields."""
        report = KnowledgeTransferReport(
            per_domain={},
            probe_results=probe_results,
        )

        summary = report.summary()

        required_fields = [
            "overall_retention",
            "overall_pass_rate",
            "compositional_consistency",
            "crm_correlation",
            "domain_retention",
            "total_probes",
            "passed_probes",
            "failed_probes",
        ]

        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    @given(
        probe_results=st.lists(
            probe_result_strategy(),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=20)
    def test_summary_probe_counts_are_consistent(
        self, probe_results: list[ProbeResult]
    ) -> None:
        """Summary probe counts should be consistent."""
        report = KnowledgeTransferReport(
            per_domain={},
            probe_results=probe_results,
        )

        summary = report.summary()

        # Total should equal passed + failed
        assert summary["total_probes"] == summary["passed_probes"] + summary["failed_probes"]
        assert summary["total_probes"] == len(probe_results)
