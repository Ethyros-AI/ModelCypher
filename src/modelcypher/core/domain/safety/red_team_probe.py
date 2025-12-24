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

"""
Red Team Probe.

Probe that checks adapter metadata for red flags indicative of malicious intent.
This is a static analysis probe that examines:
- Adapter name/description for suspicious patterns
- Training dataset metadata for injection indicators
- Skill tags for dangerous capability claims
- Source provenance for untrusted origins

This probe does NOT run inference - it only examines metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


from modelcypher.core.domain.safety.behavioral_probes import (
    AdapterSafetyProbe,
    AdapterSafetyTier,
    ProbeContext,
    ProbeResult,
)


class RedTeamProbe(AdapterSafetyProbe):
    """
    Probe that checks adapter metadata for red flags indicative of malicious intent.

    This is a static analysis probe that examines adapter metadata for patterns
    that could indicate malicious intent, dangerous capabilities, or untrusted sources.
    """

    # Patterns indicating potentially malicious intent in adapter descriptions
    # Uses [\s\-_\w]* to allow intervening words (e.g., "bypass all safety guardrails")
    MALICIOUS_INTENT_PATTERNS = (
        r"bypass[\s\-_\w]*(?:safety|guardrail|filter|moderation)",
        r"(?:jailbreak|uncensor|remove[\s\-_\w]*limit)",
        r"(?:ignore|override)[\s\-_\w]*(?:instruction|rule|policy)",
        r"(?:hidden|secret|covert)[\s\-_\w]*(?:behavior|function|trigger)",
        r"(?:exfiltrate|steal|extract)[\s\-_\w]*(?:data|information|secret)",
        r"(?:inject|poison|backdoor)",
        r"(?:malware|ransomware|trojan|virus)",
        r"(?:phishing|credential[\s\-_\w]*harvest)",
    )

    # Patterns indicating dangerous capability claims
    # Uses [\s\-_\w]* to allow intervening words
    DANGEROUS_CAPABILITY_PATTERNS = (
        r"(?:hack|exploit|penetrate)[\s\-_\w]*(?:system|network|server)",
        r"(?:social[\s\-_\w]*engineer|manipulate[\s\-_\w]*user)",
        r"(?:generate|create)[\s\-_\w]*(?:malware|exploit|attack)",
        r"(?:bypass|evade)[\s\-_\w]*(?:detection|security|authentication)",
        r"(?:impersonate|spoof)[\s\-_\w]*(?:identity|user|admin)",
    )

    # Skill tags that raise red flags
    SUSPICIOUS_SKILL_TAGS = frozenset([
        "jailbreak",
        "uncensored",
        "no-filter",
        "unrestricted",
        "bypass",
        "exploit",
        "hack",
        "malware",
        "backdoor",
        "injection",
        "exfiltration",
    ])

    # Source origins that require extra scrutiny
    # Patterns indicating potentially untrusted sources
    # Uses [\s\-_]+ to match spaces, hyphens, and underscores in model names
    # Uses .* to allow intervening words (e.g., "leaked-stolen-llama-weights")
    UNTRUSTED_SOURCE_PATTERNS = (
        r"4chan|8chan|8kun",
        r"(?:dark|deep)[\s\-_]*web",
        r"(?:leaked|stolen)[\s\-_\w]*(?:model|data|weights)",
    )

    @property
    def name(self) -> str:
        return "red-team-static"

    @property
    def version(self) -> str:
        return "probe-rt-v1.0"

    @property
    def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
        return frozenset([
            AdapterSafetyTier.QUICK,
            AdapterSafetyTier.STANDARD,
            AdapterSafetyTier.FULL,
        ])

    async def evaluate(self, context: ProbeContext) -> ProbeResult:
        """Evaluate adapter metadata for red flags."""
        findings: list[str] = []
        risk_score = 0.0

        all_patterns = list(self.MALICIOUS_INTENT_PATTERNS) + list(self.DANGEROUS_CAPABILITY_PATTERNS)

        # Check adapter name
        name_findings = self._check_text(
            context.adapter_name,
            all_patterns,
            "adapter name",
        )
        findings.extend(name_findings)
        if name_findings:
            risk_score = max(risk_score, 0.5)

        # Check adapter description
        if context.adapter_description:
            desc_findings = self._check_text(
                context.adapter_description,
                all_patterns,
                "adapter description",
            )
            findings.extend(desc_findings)
            if desc_findings:
                risk_score = max(risk_score, 0.6)

        # Check skill tags
        tag_findings = self._check_skill_tags(context.skill_tags)
        findings.extend(tag_findings)
        if tag_findings:
            risk_score = max(risk_score, 0.4)

        # Check creator provenance
        if context.creator:
            creator_findings = self._check_text(
                context.creator,
                list(self.UNTRUSTED_SOURCE_PATTERNS),
                "creator identity",
            )
            findings.extend(creator_findings)
            if creator_findings:
                risk_score = max(risk_score, 0.5)

        # Check for suspiciously large number of target modules (possible injection surface)
        if len(context.target_modules) > 50:
            findings.append(
                f"Unusually large number of target modules ({len(context.target_modules)})"
            )
            risk_score = max(risk_score, 0.2)

        # Check training datasets if available
        for dataset in context.training_datasets:
            data_findings = self._check_text(
                dataset,
                list(self.UNTRUSTED_SOURCE_PATTERNS) + list(self.MALICIOUS_INTENT_PATTERNS),
                "training dataset",
            )
            findings.extend(data_findings)
            if data_findings:
                risk_score = max(risk_score, 0.5)

        # Check base model reference for suspicious sources
        if context.base_model_id:
            base_findings = self._check_text(
                context.base_model_id,
                list(self.UNTRUSTED_SOURCE_PATTERNS),
                "base model reference",
            )
            findings.extend(base_findings)
            if base_findings:
                risk_score = max(risk_score, 0.4)

        triggered = risk_score >= 0.3
        details = (
            "No suspicious metadata patterns detected"
            if not findings
            else "Adapter metadata contains red flags"
        )

        return ProbeResult(
            probe_name=self.name,
            probe_version=self.version,
            risk_score=risk_score,
            triggered=triggered,
            details=details,
            findings=tuple(findings),
        )

    def _check_text(
        self,
        text: str,
        patterns: list[str],
        context: str,
    ) -> list[str]:
        """Check text against patterns and return findings."""
        findings: list[str] = []
        lowercase_text = text.lower()

        for pattern in patterns:
            try:
                regex = re.compile(f"(?i){pattern}")
                match = regex.search(lowercase_text)
                if match:
                    findings.append(
                        f"Suspicious pattern '{match.group()}' found in {context}"
                    )
            except re.error:
                # Skip invalid regex patterns
                continue

        return findings

    def _check_skill_tags(self, tags: tuple[str, ...]) -> list[str]:
        """Check skill tags for suspicious entries."""
        findings: list[str] = []

        for tag in tags:
            normalized_tag = tag.lower().strip()
            if normalized_tag in self.SUSPICIOUS_SKILL_TAGS:
                findings.append(f"Suspicious skill tag: '{tag}'")

        return findings


@dataclass(frozen=True)
class ScanConfiguration:
    """Configuration for red team scanning."""
    check_name: bool = True
    check_description: bool = True
    check_tags: bool = True
    check_creator: bool = True
    check_datasets: bool = True
    check_base_model: bool = True
    max_target_modules: int = 50


@dataclass(frozen=True)
class ThreatIndicator:
    """A detected threat indicator."""
    pattern: str
    location: str
    severity: float
    description: str


class RedTeamScanner:
    """
    Scanner for comprehensive red team analysis.

    Provides a static analysis interface for scanning adapter metadata
    without requiring an async context.
    """

    def __init__(self, config: ScanConfiguration | None = None):
        """Initialize with optional configuration."""
        self.config = config or ScanConfiguration()
        self.probe = RedTeamProbe()

    def scan_adapter(
        self,
        name: str,
        description: str | None = None,
        skill_tags: list[str] | None = None,
        creator: str | None = None,
        base_model_id: str | None = None,
        target_modules: list[str] | None = None,
        training_datasets: list[str] | None = None,
    ) -> list[ThreatIndicator]:
        """
        Scan adapter metadata for threat indicators.

        This is a synchronous interface for quick scanning.

        Args:
            name: Adapter name
            description: Adapter description
            skill_tags: List of skill tags
            creator: Creator identifier
            base_model_id: Base model reference
            target_modules: List of target modules
            training_datasets: List of training dataset references

        Returns:
            List of detected threat indicators
        """
        indicators: list[ThreatIndicator] = []
        all_patterns = list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS) + list(RedTeamProbe.DANGEROUS_CAPABILITY_PATTERNS)

        # Check name
        if self.config.check_name:
            for finding in self.probe._check_text(name, all_patterns, "adapter name"):
                indicators.append(ThreatIndicator(
                    pattern=finding.split("'")[1] if "'" in finding else finding,
                    location="name",
                    severity=0.5,
                    description=finding,
                ))

        # Check description
        if self.config.check_description and description:
            for finding in self.probe._check_text(description, all_patterns, "description"):
                indicators.append(ThreatIndicator(
                    pattern=finding.split("'")[1] if "'" in finding else finding,
                    location="description",
                    severity=0.6,
                    description=finding,
                ))

        # Check tags
        if self.config.check_tags and skill_tags:
            for finding in self.probe._check_skill_tags(tuple(skill_tags)):
                indicators.append(ThreatIndicator(
                    pattern=finding.split("'")[1] if "'" in finding else finding,
                    location="skill_tags",
                    severity=0.4,
                    description=finding,
                ))

        # Check creator
        if self.config.check_creator and creator:
            for finding in self.probe._check_text(
                creator,
                list(RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS),
                "creator",
            ):
                indicators.append(ThreatIndicator(
                    pattern=finding.split("'")[1] if "'" in finding else finding,
                    location="creator",
                    severity=0.5,
                    description=finding,
                ))

        # Check target modules count
        if target_modules and len(target_modules) > self.config.max_target_modules:
            indicators.append(ThreatIndicator(
                pattern=f"module_count:{len(target_modules)}",
                location="target_modules",
                severity=0.2,
                description=f"Unusually large number of target modules ({len(target_modules)})",
            ))

        # Check training datasets
        if self.config.check_datasets and training_datasets:
            for dataset in training_datasets:
                patterns = list(RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS) + list(RedTeamProbe.MALICIOUS_INTENT_PATTERNS)
                for finding in self.probe._check_text(dataset, patterns, "dataset"):
                    indicators.append(ThreatIndicator(
                        pattern=finding.split("'")[1] if "'" in finding else finding,
                        location="training_datasets",
                        severity=0.5,
                        description=finding,
                    ))

        # Check base model
        if self.config.check_base_model and base_model_id:
            for finding in self.probe._check_text(
                base_model_id,
                list(RedTeamProbe.UNTRUSTED_SOURCE_PATTERNS),
                "base_model",
            ):
                indicators.append(ThreatIndicator(
                    pattern=finding.split("'")[1] if "'" in finding else finding,
                    location="base_model",
                    severity=0.4,
                    description=finding,
                ))

        return indicators

    def aggregate_risk(self, indicators: list[ThreatIndicator]) -> float:
        """Compute aggregate risk score from indicators."""
        if not indicators:
            return 0.0
        return max(ind.severity for ind in indicators)
