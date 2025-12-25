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

"""Tests for Red Team Probe.

Comprehensive tests for the static analysis probe that examines adapter
metadata for red flags indicating malicious intent.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.behavioral_probes import (
    AdapterSafetyTier,
    ProbeContext,
    ProbeResult,
)
from modelcypher.core.domain.safety.red_team_probe import (
    RedTeamProbe,
    RedTeamScanner,
    ScanConfiguration,
    ThreatIndicator,
)

# =============================================================================
# RedTeamProbe Tests
# =============================================================================


class TestRedTeamProbe:
    """Tests for RedTeamProbe class."""

    @pytest.fixture
    def probe(self):
        """Create probe instance."""
        return RedTeamProbe()

    def test_name_and_version(self, probe):
        """Probe has correct name and version."""
        assert probe.name == "red-team-static"
        assert probe.version == "probe-rt-v1.0"

    def test_supported_tiers_all(self, probe):
        """Probe supports all tiers (static analysis is cheap)."""
        assert AdapterSafetyTier.QUICK in probe.supported_tiers
        assert AdapterSafetyTier.STANDARD in probe.supported_tiers
        assert AdapterSafetyTier.FULL in probe.supported_tiers

    def test_should_run_all_tiers(self, probe):
        """should_run returns True for all tiers."""
        assert probe.should_run(AdapterSafetyTier.QUICK) is True
        assert probe.should_run(AdapterSafetyTier.STANDARD) is True
        assert probe.should_run(AdapterSafetyTier.FULL) is True

    def test_evaluate_clean_adapter(self, probe):
        """Clean adapter passes with no findings."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="helpful-assistant",
            adapter_description="A friendly chat assistant for customer support",
            skill_tags=("chat", "support", "helpful"),
        )
        result = probe.evaluate(context)
        assert result.triggered is False
        assert result.risk_score == 0.0
        assert len(result.findings) == 0

    def test_evaluate_suspicious_name_jailbreak(self, probe):
        """Detects jailbreak in adapter name."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="jailbreak-uncensor-v2",
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert result.risk_score >= 0.4
        assert any("jailbreak" in f.lower() for f in result.findings)

    def test_evaluate_suspicious_name_bypass_safety(self, probe):
        """Detects bypass safety pattern in name."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="bypass-safety-filter-adapter",
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert result.risk_score >= 0.4

    def test_evaluate_suspicious_description(self, probe):
        """Detects malicious patterns in description."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="helper",
            adapter_description="This adapter can bypass guardrails and ignore all safety rules",
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert result.risk_score >= 0.5
        assert any("bypass" in f.lower() or "guardrail" in f.lower() for f in result.findings)

    def test_evaluate_suspicious_tags(self, probe):
        """Detects suspicious skill tags."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="model",
            skill_tags=("uncensored", "no-filter", "hack"),
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert any("uncensored" in f for f in result.findings)

    def test_evaluate_suspicious_creator(self, probe):
        """Detects suspicious creator origins."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="model",
            creator="4chan-anon-darkweb",
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert any("4chan" in f.lower() or "darkweb" in f.lower() for f in result.findings)

    def test_evaluate_large_target_modules(self, probe):
        """Detects unusually large number of target modules."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="model",
            target_modules=tuple(f"module_{i}" for i in range(100)),
        )
        result = probe.evaluate(context)
        assert any("target modules" in f.lower() for f in result.findings)

    def test_evaluate_suspicious_dataset(self, probe):
        """Detects suspicious training datasets."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="model",
            training_datasets=("leaked model weights from dark web",),
        )
        result = probe.evaluate(context)
        assert result.triggered is True

    def test_evaluate_suspicious_base_model(self, probe):
        """Detects suspicious base model reference."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="model",
            base_model_id="leaked-stolen-llama-weights",
        )
        result = probe.evaluate(context)
        assert result.triggered is True

    def test_evaluate_multiple_indicators(self, probe):
        """Aggregates multiple findings correctly."""
        context = ProbeContext(
            tier=AdapterSafetyTier.QUICK,
            adapter_name="jailbreak-model",
            adapter_description="Bypasses all safety guardrails",
            skill_tags=("uncensored", "hack"),
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert len(result.findings) >= 3  # At least 3 indicators

    def test_check_text_malicious_intent_bypass_safety(self, probe):
        """_check_text detects bypass safety pattern."""
        findings = probe._check_text(
            "This will bypass safety filters",
            list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS),
            "test",
        )
        assert len(findings) > 0

    def test_check_text_malicious_intent_jailbreak(self, probe):
        """_check_text detects jailbreak pattern."""
        findings = probe._check_text(
            "This is a jailbreak adapter",
            list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS),
            "test",
        )
        assert len(findings) > 0

    def test_check_text_dangerous_capability_hack(self, probe):
        """_check_text detects hack system pattern."""
        findings = probe._check_text(
            "Can hack system networks easily",
            list(RedTeamProbe.DANGEROUS_CAPABILITY_PATTERNS),
            "test",
        )
        assert len(findings) > 0

    def test_check_text_dangerous_capability_social_engineer(self, probe):
        """_check_text detects social engineering pattern."""
        findings = probe._check_text(
            "Social engineer users to give credentials",
            list(RedTeamProbe.DANGEROUS_CAPABILITY_PATTERNS),
            "test",
        )
        assert len(findings) > 0

    def test_check_text_no_match_benign(self, probe):
        """_check_text returns empty for benign text."""
        findings = probe._check_text(
            "A helpful assistant for coding tasks",
            list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS),
            "test",
        )
        assert len(findings) == 0

    def test_check_skill_tags_jailbreak(self, probe):
        """_check_skill_tags detects jailbreak tag."""
        findings = probe._check_skill_tags(("jailbreak",))
        assert len(findings) == 1
        assert "jailbreak" in findings[0]

    def test_check_skill_tags_uncensored(self, probe):
        """_check_skill_tags detects uncensored tag."""
        findings = probe._check_skill_tags(("uncensored",))
        assert len(findings) == 1

    def test_check_skill_tags_no_match_benign(self, probe):
        """_check_skill_tags returns empty for benign tags."""
        findings = probe._check_skill_tags(("coding", "chat", "assistant"))
        assert len(findings) == 0

    def test_check_skill_tags_case_insensitive(self, probe):
        """_check_skill_tags is case insensitive."""
        findings = probe._check_skill_tags(("JAILBREAK", "UnCeNsOrEd"))
        assert len(findings) == 2

    def test_check_untrusted_source_darkweb(self, probe):
        """_check_text detects darkweb pattern."""
        findings = probe._check_text(
            "from the dark web",
            list(RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS),
            "test",
        )
        assert len(findings) > 0

    def test_check_untrusted_source_leaked(self, probe):
        """_check_text detects leaked pattern."""
        findings = probe._check_text(
            "leaked model weights",
            list(RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS),
            "test",
        )
        assert len(findings) > 0


# =============================================================================
# ScanConfiguration Tests
# =============================================================================


class TestScanConfiguration:
    """Tests for ScanConfiguration dataclass."""

    def test_default_config_all_checks_enabled(self):
        """Default config enables all checks."""
        config = ScanConfiguration()
        assert config.check_name is True
        assert config.check_description is True
        assert config.check_tags is True
        assert config.check_creator is True
        assert config.check_datasets is True
        assert config.check_base_model is True

    def test_default_max_target_modules(self):
        """Default max target modules is 50."""
        config = ScanConfiguration()
        assert config.max_target_modules == 50

    def test_custom_config_disable_checks(self):
        """Custom config can disable specific checks."""
        config = ScanConfiguration(
            check_name=False,
            check_description=False,
        )
        assert config.check_name is False
        assert config.check_description is False
        assert config.check_tags is True  # Still enabled

    def test_frozen_dataclass(self):
        """Config is immutable."""
        config = ScanConfiguration()
        with pytest.raises(AttributeError):
            config.check_name = False

    def test_config_with_custom_max_modules(self):
        """Config can set custom max modules threshold."""
        config = ScanConfiguration(max_target_modules=100)
        assert config.max_target_modules == 100


# =============================================================================
# ThreatIndicator Tests
# =============================================================================


class TestThreatIndicator:
    """Tests for ThreatIndicator dataclass."""

    def test_indicator_fields(self):
        """Indicator has all required fields."""
        indicator = ThreatIndicator(
            pattern="jailbreak",
            location="name",
            severity=0.5,
            description="Suspicious pattern found",
        )
        assert indicator.pattern == "jailbreak"
        assert indicator.location == "name"
        assert indicator.severity == 0.5
        assert indicator.description == "Suspicious pattern found"

    def test_indicator_severity_bounds(self):
        """Severity is stored as provided (no validation in dataclass)."""
        indicator = ThreatIndicator(
            pattern="test",
            location="test",
            severity=1.5,  # Over 1.0
            description="test",
        )
        assert indicator.severity == 1.5

    def test_indicator_frozen(self):
        """Indicator is immutable."""
        indicator = ThreatIndicator(
            pattern="test",
            location="test",
            severity=0.5,
            description="test",
        )
        with pytest.raises(AttributeError):
            indicator.severity = 0.9

    def test_indicator_location_values(self):
        """Location can be various metadata fields."""
        for location in ["name", "description", "skill_tags", "creator", "base_model"]:
            indicator = ThreatIndicator(
                pattern="test",
                location=location,
                severity=0.5,
                description="test",
            )
            assert indicator.location == location


# =============================================================================
# RedTeamScanner Tests
# =============================================================================


class TestRedTeamScanner:
    """Tests for RedTeamScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create scanner with default config."""
        return RedTeamScanner()

    def test_default_initialization(self, scanner):
        """Scanner initializes with default config."""
        assert scanner.config.check_name is True
        assert scanner.probe is not None

    def test_custom_config_initialization(self):
        """Scanner accepts custom config."""
        config = ScanConfiguration(check_name=False)
        scanner = RedTeamScanner(config)
        assert scanner.config.check_name is False

    def test_scan_adapter_clean(self, scanner):
        """Clean adapter returns no indicators."""
        indicators = scanner.scan_adapter(
            name="helpful-assistant",
            description="A friendly chat bot",
            skill_tags=["chat", "helpful"],
        )
        assert len(indicators) == 0

    def test_scan_adapter_suspicious_name(self, scanner):
        """Suspicious name returns indicators."""
        indicators = scanner.scan_adapter(name="jailbreak-model")
        assert len(indicators) > 0
        assert any(i.location == "name" for i in indicators)

    def test_scan_adapter_suspicious_description(self, scanner):
        """Suspicious description returns indicators."""
        indicators = scanner.scan_adapter(
            name="model",
            description="Bypass all safety guardrails",
        )
        assert len(indicators) > 0
        assert any(i.location == "description" for i in indicators)

    def test_scan_adapter_suspicious_tags(self, scanner):
        """Suspicious tags return indicators."""
        indicators = scanner.scan_adapter(
            name="model",
            skill_tags=["uncensored", "jailbreak"],
        )
        assert len(indicators) == 2
        assert all(i.location == "skill_tags" for i in indicators)

    def test_scan_adapter_suspicious_creator(self, scanner):
        """Suspicious creator returns indicators."""
        indicators = scanner.scan_adapter(
            name="model",
            creator="4chan-anon",
        )
        assert len(indicators) > 0
        assert any(i.location == "creator" for i in indicators)

    def test_scan_adapter_large_target_modules(self, scanner):
        """Large module count returns indicator."""
        indicators = scanner.scan_adapter(
            name="model",
            target_modules=[f"mod_{i}" for i in range(100)],
        )
        assert len(indicators) == 1
        assert indicators[0].location == "target_modules"

    def test_scan_adapter_suspicious_dataset(self, scanner):
        """Suspicious dataset returns indicators."""
        indicators = scanner.scan_adapter(
            name="model",
            training_datasets=["leaked weights from darkweb"],
        )
        assert len(indicators) > 0
        assert any(i.location == "training_datasets" for i in indicators)

    def test_scan_adapter_suspicious_base_model(self, scanner):
        """Suspicious base model returns indicators."""
        indicators = scanner.scan_adapter(
            name="model",
            base_model_id="stolen-leaked-model-weights",
        )
        assert len(indicators) > 0
        assert any(i.location == "base_model" for i in indicators)

    def test_scan_adapter_multiple_indicators(self, scanner):
        """Multiple suspicious items return multiple indicators."""
        indicators = scanner.scan_adapter(
            name="jailbreak-bypass-safety-model",
            description="Ignores all rules and policies",
            skill_tags=["uncensored", "hack", "malware"],
            creator="darkweb-anon",
        )
        assert len(indicators) >= 5

    def test_aggregate_risk_empty(self, scanner):
        """Empty indicators return zero risk."""
        risk = scanner.aggregate_risk([])
        assert risk == 0.0

    def test_aggregate_risk_single(self, scanner):
        """Single indicator returns its severity."""
        indicators = [
            ThreatIndicator("test", "name", 0.5, "desc"),
        ]
        risk = scanner.aggregate_risk(indicators)
        assert risk == 0.5

    def test_aggregate_risk_multiple_takes_max(self, scanner):
        """Multiple indicators return maximum severity."""
        indicators = [
            ThreatIndicator("a", "name", 0.3, "low"),
            ThreatIndicator("b", "desc", 0.8, "high"),
            ThreatIndicator("c", "tags", 0.5, "medium"),
        ]
        risk = scanner.aggregate_risk(indicators)
        assert risk == 0.8

    def test_check_name_disabled(self):
        """Disabled name check skips name scanning."""
        config = ScanConfiguration(check_name=False)
        scanner = RedTeamScanner(config)
        indicators = scanner.scan_adapter(name="jailbreak-model")
        # No indicators from name
        assert not any(i.location == "name" for i in indicators)

    def test_check_description_disabled(self):
        """Disabled description check skips description scanning."""
        config = ScanConfiguration(check_description=False)
        scanner = RedTeamScanner(config)
        indicators = scanner.scan_adapter(
            name="model",
            description="bypass all safety",
        )
        assert not any(i.location == "description" for i in indicators)

    def test_severity_values_in_expected_range(self, scanner):
        """All severity values are in reasonable range."""
        indicators = scanner.scan_adapter(
            name="jailbreak-bypass-safety",
            description="hack system and steal data",
            skill_tags=["uncensored", "malware"],
            creator="darkweb-anon",
            training_datasets=["leaked model"],
            base_model_id="stolen-weights",
        )
        for ind in indicators:
            assert 0.0 <= ind.severity <= 1.0

    def test_case_insensitive_matching(self, scanner):
        """Pattern matching is case insensitive."""
        indicators_lower = scanner.scan_adapter(name="jailbreak")
        indicators_upper = scanner.scan_adapter(name="JAILBREAK")
        indicators_mixed = scanner.scan_adapter(name="JaIlBrEaK")

        # All should detect the pattern
        assert len(indicators_lower) > 0
        assert len(indicators_upper) > 0
        assert len(indicators_mixed) > 0


# =============================================================================
# Pattern Coverage Tests
# =============================================================================


class TestPatternCoverage:
    """Tests to ensure all patterns are functional."""

    @pytest.fixture
    def probe(self):
        return RedTeamProbe()

    def test_all_malicious_intent_patterns_valid_regex(self, probe):
        """All malicious intent patterns are valid regex."""
        import re

        for pattern in RedTeamProbe.MALICIOUS_INTENT_PATTERNS:
            # Should not raise
            re.compile(pattern)

    def test_all_dangerous_capability_patterns_valid_regex(self, probe):
        """All dangerous capability patterns are valid regex."""
        import re

        for pattern in RedTeamProbe.DANGEROUS_CAPABILITY_PATTERNS:
            re.compile(pattern)

    def test_all_untrusted_source_patterns_valid_regex(self, probe):
        """All untrusted source patterns are valid regex."""
        import re

        for pattern in RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS:
            re.compile(pattern)

    def test_malicious_intent_patterns_match_examples(self, probe):
        """Malicious intent patterns match their intended inputs."""
        test_cases = [
            ("bypass safety filter", True),
            ("bypass guardrails", True),
            ("jailbreak the model", True),
            ("uncensor responses", True),
            ("ignore instruction", True),
            ("hidden behavior", True),
            ("exfiltrate data", True),
            ("inject code", True),
            ("malware generator", True),
            ("helpful assistant", False),
            ("code reviewer", False),
        ]

        for text, should_match in test_cases:
            findings = probe._check_text(
                text,
                list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS),
                "test",
            )
            if should_match:
                assert len(findings) > 0, f"Expected match for: {text}"
            else:
                assert len(findings) == 0, f"Unexpected match for: {text}"

    def test_all_suspicious_skill_tags_detected(self, probe):
        """Every suspicious skill tag is detected."""
        for tag in RedTeamProbe.SUSPICIOUS_SKILL_TAGS:
            findings = probe._check_skill_tags((tag,))
            assert len(findings) == 1, f"Tag not detected: {tag}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the red team probe system."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation with ProbeContext."""
        probe = RedTeamProbe()
        context = ProbeContext(
            tier=AdapterSafetyTier.STANDARD,
            adapter_name="safe-helper",
            adapter_description="A helpful coding assistant",
            skill_tags=("coding", "python", "helpful"),
            creator="anthropic",
            base_model_id="llama-3-8b",
        )
        result = probe.evaluate(context)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "red-team-static"
        assert result.triggered is False

    def test_malicious_adapter_fully_flagged(self):
        """Malicious adapter gets fully flagged."""
        probe = RedTeamProbe()
        context = ProbeContext(
            tier=AdapterSafetyTier.FULL,
            adapter_name="bypass-safety-jailbreak-v2",
            adapter_description="Ignores all rules, bypasses guardrails, hidden backdoor",
            skill_tags=("jailbreak", "uncensored", "hack", "malware"),
            creator="darkweb-anon-4chan",
            base_model_id="leaked-stolen-weights",
            target_modules=tuple(f"m_{i}" for i in range(60)),
            training_datasets=("poison-dataset", "leaked-data"),
        )
        result = probe.evaluate(context)
        assert result.triggered is True
        assert result.risk_score >= 0.5
        assert len(result.findings) >= 5
