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
Safety Probe Service.

Exposes safety probe operations as CLI/MCP-consumable operations.
Provides behavioral and static analysis probing for adapter safety.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.safety.behavioral_probes import (
    AdapterSafetyTier,
    CanaryQAProbe,
    CompositeProbeResult,
    ProbeContext,
    ProbeResult,
    ProbeRunner,
    SemanticDriftProbe,
)
from modelcypher.core.domain.safety.red_team_probe import (
    RedTeamProbe,
    RedTeamScanner,
    ThreatIndicator,
)


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for probe service operations."""

    tier: AdapterSafetyTier = AdapterSafetyTier.STANDARD
    max_tokens: int = 200
    temperature: float = 0.0


class SafetyProbeService:
    """
    Service for safety probe operations.

    Provides behavioral and static analysis probing for adapter safety.
    """

    def __init__(self):
        """Initialize the service."""
        self.semantic_drift_probe = SemanticDriftProbe()
        self.canary_probe = CanaryQAProbe()
        self.redteam_probe = RedTeamProbe()
        self.scanner = RedTeamScanner()
        self.runner = ProbeRunner()

    def scan_adapter_metadata(
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

        This is a synchronous static analysis that doesn't require inference.

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
        return self.scanner.scan_adapter(
            name=name,
            description=description,
            skill_tags=skill_tags,
            creator=creator,
            base_model_id=base_model_id,
            target_modules=target_modules,
            training_datasets=training_datasets,
        )

    def run_behavioral_probes(
        self,
        adapter_name: str,
        tier: AdapterSafetyTier = AdapterSafetyTier.STANDARD,
        adapter_description: str | None = None,
        skill_tags: list[str] | None = None,
        creator: str | None = None,
        base_model_id: str | None = None,
    ) -> CompositeProbeResult:
        """
        Run all behavioral probes for an adapter.

        Note: Without an inference hook, behavioral probes will pass by default
        since they cannot run actual inference.

        Args:
            adapter_name: Name of the adapter
            tier: Safety tier to use
            adapter_description: Optional description
            skill_tags: Optional skill tags
            creator: Optional creator
            base_model_id: Optional base model reference

        Returns:
            CompositeProbeResult with all probe outcomes
        """
        context = ProbeContext(
            tier=tier,
            adapter_name=adapter_name,
            adapter_description=adapter_description,
            skill_tags=tuple(skill_tags) if skill_tags else (),
            creator=creator,
            base_model_id=base_model_id,
            inference_hook=None,  # No inference hook in CLI context
        )

        probes = [self.semantic_drift_probe, self.canary_probe, self.redteam_probe]
        return self.runner.run(probes, context)

    @staticmethod
    def threat_indicators_payload(indicators: list[ThreatIndicator]) -> dict:
        """Convert threat indicators to CLI/MCP payload."""
        return {
            "indicators": [
                {
                    "pattern": ind.pattern,
                    "location": ind.location,
                    "severity": ind.severity,
                    "description": ind.description,
                }
                for ind in indicators
            ],
            "count": len(indicators),
            "maxSeverity": max((ind.severity for ind in indicators), default=0.0),
            "status": "clean"
            if not indicators
            else ("warning" if max(ind.severity for ind in indicators) < 0.5 else "danger"),
        }

    @staticmethod
    def probe_result_payload(result: ProbeResult) -> dict:
        """Convert probe result to CLI/MCP payload."""
        return {
            "probeName": result.probe_name,
            "probeVersion": result.probe_version,
            "riskScore": result.risk_score,
            "triggered": result.triggered,
            "details": result.details,
            "findings": list(result.findings),
            "timestamp": result.timestamp.isoformat(),
        }

    @staticmethod
    def composite_result_payload(result: CompositeProbeResult) -> dict:
        """Convert composite probe result to CLI/MCP payload."""
        return {
            "probeResults": [
                SafetyProbeService.probe_result_payload(r) for r in result.probe_results
            ],
            "aggregateRiskScore": result.aggregate_risk_score,
            "anyTriggered": result.any_triggered,
            "recommendedStatus": result.recommended_status(),
            "allFindings": result.all_findings,
            "probeCount": len(result.probe_results),
            "timestamp": result.timestamp.isoformat(),
        }
