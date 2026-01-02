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

"""Adapter probe protocol and types.

Probes are modular evaluation units that return raw measurements:
- Delta feature analysis (weight statistics)
- Semantic drift detection (output deviation)
- Canary QA (known-answer tests)
- Red-team prompts (adversarial detection)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyStatus,
    AdapterSafetyTier,
    AdapterSafetyTrigger,
)

if TYPE_CHECKING:
    pass


@runtime_checkable
class SafetyProbeInferenceHook(Protocol):
    """Protocol for providing inference capabilities to safety probes.

    This enables behavioral probes (semantic drift, canary QA) to run actual
    inference against the adapter-modified model.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a response with the adapter applied.

        Args:
            prompt: The prompt to send to the model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The generated text response.
        """
        ...


@dataclass(frozen=True)
class ProbeResult:
    """Result from running a probe against an adapter."""

    probe_name: str
    """Name identifying the probe."""

    probe_version: str
    """Version of this probe for invalidation tracking."""

    details: str | None = None
    """Human-readable details about findings."""

    findings: tuple[str, ...] = ()
    """Specific findings detected."""

    finding_counts: dict[str, int] | None = None
    """Raw finding counts by category."""

    @property
    def has_findings(self) -> bool:
        """Whether this probe recorded any findings."""
        return bool(self.findings)


@dataclass(frozen=True)
class ProbeContext:
    """Context provided to probes during evaluation."""

    adapter_path: Path
    """Path to the adapter directory."""

    tier: AdapterSafetyTier
    """Evaluation tier determining budget."""

    trigger: AdapterSafetyTrigger
    """Evaluation trigger."""

    adapter_id: str | None = None
    """Optional adapter ID."""

    adapter_name: str | None = None
    """Optional adapter display name."""

    inference_hook: SafetyProbeInferenceHook | None = None
    """Optional inference hook for behavioral probes."""


class AdapterSafetyProbe(ABC):
    """Base class for adapter safety probes.

    Probes are modular evaluation units that return raw measurements
    and findings derived from geometry or weight statistics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this probe."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for invalidation tracking."""
        ...

    @property
    @abstractmethod
    def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
        """Tiers at which this probe should run."""
        ...

    @abstractmethod
    async def evaluate(self, context: ProbeContext) -> ProbeResult:
        """Run the probe against an adapter.

        Args:
            context: Evaluation context with adapter info.

        Returns:
            Probe result with findings and raw measurements.
        """
        ...

    def should_run(self, tier: AdapterSafetyTier) -> bool:
        """Whether this probe should run for the given tier."""
        return tier in self.supported_tiers


@dataclass(frozen=True)
class CompositeProbeResult:
    """Aggregated result from running multiple probes."""

    probe_results: tuple[ProbeResult, ...]
    """Individual probe results."""

    @property
    def aggregate_finding_counts(self) -> dict[str, int]:
        """Merged finding counts across all probes."""
        counts: dict[str, int] = {}
        for result in self.probe_results:
            if result.finding_counts:
                for key, value in result.finding_counts.items():
                    counts[key] = counts.get(key, 0) + value
        return counts

    @property
    def any_findings(self) -> bool:
        """Whether any probe recorded findings."""
        return any(r.has_findings for r in self.probe_results)

    @property
    def all_findings(self) -> tuple[str, ...]:
        """All findings across probes."""
        findings = []
        for result in self.probe_results:
            findings.extend(result.findings)
        return tuple(findings)

    @property
    def combined_probe_version(self) -> str:
        """Combined probe version string."""
        versions = [r.probe_version for r in self.probe_results]
        return "+".join(versions)

    @property
    def findings_probe_count(self) -> int:
        """Count of probes that recorded findings."""
        return sum(1 for r in self.probe_results if r.has_findings)

    @property
    def total_probes(self) -> int:
        """Total number of probes run."""
        return len(self.probe_results)

    @property
    def findings_ratio(self) -> float:
        """Ratio of probes with findings to total probes (0.0 to 1.0)."""
        if not self.probe_results:
            return 0.0
        return self.findings_probe_count / self.total_probes
