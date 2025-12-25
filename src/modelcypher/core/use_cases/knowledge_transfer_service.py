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
Knowledge Transfer Validation Service.

Orchestrates post-merge knowledge validation using targeted probes
to verify that merged models retain knowledge from source models.

Integrates with:
- KnowledgeTransferValidator domain module
- MergeValidationService for perplexity/coherence
- InferenceEngine port for probe execution
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modelcypher.ports import InferenceEngine

from modelcypher.core.domain.merging.knowledge_transfer_validator import (
    KnowledgeDomain,
    KnowledgeProbe,
    KnowledgeProbeCorpus,
    KnowledgeTransferReport,
    ValidationStatus,
    compute_retention_by_domain,
    run_knowledge_probes,
)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeTransferConfig:
    """Configuration for knowledge transfer validation."""

    # Domains to test
    domains: list[KnowledgeDomain] = field(default_factory=lambda: list(KnowledgeDomain))

    # Probe execution
    max_tokens: int = 200
    temperature: float = 0.0
    timeout_per_probe: float = 30.0

    # Thresholds
    retention_threshold: float = 0.8
    domain_failure_threshold: float = 0.6

    # Options
    include_variations: bool = True
    parallel_execution: bool = False
    custom_probes: list[KnowledgeProbe] | None = None

    # Comparison models
    compare_to_source: bool = True
    compare_to_target: bool = False


@dataclass
class KnowledgeTransferValidationResult:
    """Complete result of knowledge transfer validation."""

    validation_id: str
    merged_model: str
    source_model: str | None
    validated_at: datetime

    # Core report from domain module
    report: KnowledgeTransferReport

    # Comparison data (if source/target tested)
    source_report: KnowledgeTransferReport | None = None
    target_report: KnowledgeTransferReport | None = None

    # Execution metadata
    probes_executed: int = 0
    execution_time_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> ValidationStatus:
        """Overall validation status."""
        return self.report.status

    @property
    def overall_retention(self) -> float:
        """Overall knowledge retention score."""
        return self.report.overall_retention

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "validationId": self.validation_id,
            "mergedModel": self.merged_model,
            "sourceModel": self.source_model,
            "validatedAt": self.validated_at.isoformat(),
            "status": self.status.value,
            "overallRetention": round(self.overall_retention, 4),
            "perDomainRetention": {
                domain.value: {
                    "retention": round(result.retention_score, 4),
                    "sourcePassRate": round(result.source_pass_rate, 4),
                    "mergedPassRate": round(result.merged_pass_rate, 4),
                    "probesTested": result.probes_tested,
                }
                for domain, result in self.report.per_domain.items()
            },
            "probeResults": [
                {
                    "probeId": r.probe_id,
                    "domain": r.domain.value,
                    "passed": r.passed,
                    "variationPassRate": round(r.variation_pass_rate, 4)
                    if r.variation_results
                    else None,
                }
                for r in self.report.probe_results
            ],
            "probesExecuted": self.probes_executed,
            "executionTimeSeconds": round(self.execution_time_seconds, 2),
            "warnings": self.warnings,
            "recommendation": self.report.recommendation,
        }


class KnowledgeTransferService:
    """
    Service for validating knowledge transfer in merged models.

    Executes domain-specific probes against merged models and computes
    retention scores comparing against source model baselines.

    Usage:
        service = KnowledgeTransferService(inference_engine=engine)
        result = service.validate(
            merged_model="/path/to/merged",
            source_model="/path/to/source",  # For baseline comparison
        )
        print(f"Retention: {result.overall_retention:.1%}")
        print(f"Status: {result.status}")
    """

    def __init__(self, inference_engine: "InferenceEngine") -> None:
        """Initialize KnowledgeTransferService with required dependencies.

        Args:
            inference_engine: Inference engine port implementation (REQUIRED).
        """
        self._inference_engine = inference_engine
        self._corpus = KnowledgeProbeCorpus.default()

    def validate(
        self,
        merged_model: str,
        source_model: str | None = None,
        target_model: str | None = None,
        config: KnowledgeTransferConfig | None = None,
    ) -> KnowledgeTransferValidationResult:
        """
        Execute full knowledge transfer validation.

        Args:
            merged_model: Path to merged model directory.
            source_model: Path to source model for baseline comparison.
            target_model: Path to target model for comparison (optional).
            config: Validation configuration.

        Returns:
            KnowledgeTransferValidationResult with retention metrics.
        """
        import time

        start_time = time.time()
        config = config or KnowledgeTransferConfig()
        validation_id = f"ktv-{uuid.uuid4().hex[:8]}"

        warnings: list[str] = []

        # Build probe corpus
        corpus = self._build_corpus(config)
        probes = corpus.get_probes_by_domains(config.domains)

        if not probes:
            warnings.append("No probes found for specified domains")
            return KnowledgeTransferValidationResult(
                validation_id=validation_id,
                merged_model=merged_model,
                source_model=source_model,
                validated_at=datetime.utcnow(),
                report=KnowledgeTransferReport(per_domain={}, probe_results=[]),
                warnings=warnings,
            )

        # Create inference function
        infer_fn = self._create_inference_fn(config)

        # Run probes on source model first (for baseline)
        source_results: dict[str, bool] = {}
        if source_model and config.compare_to_source:
            logger.info(f"Running baseline probes on source model: {source_model}")
            try:
                source_probe_results = run_knowledge_probes(
                    probes,
                    self._create_model_infer_fn(source_model, config),
                    include_variations=config.include_variations,
                )
                source_results = {r.probe_id: r.passed for r in source_probe_results}
            except Exception as e:
                logger.warning(f"Source model probing failed: {e}")
                warnings.append(f"Source model probing failed: {e}")

        # Run probes on merged model
        logger.info(f"Running knowledge probes on merged model: {merged_model}")
        try:
            merged_probe_results = run_knowledge_probes(
                probes,
                self._create_model_infer_fn(merged_model, config),
                include_variations=config.include_variations,
            )
        except Exception as e:
            logger.error(f"Merged model probing failed: {e}")
            return KnowledgeTransferValidationResult(
                validation_id=validation_id,
                merged_model=merged_model,
                source_model=source_model,
                validated_at=datetime.utcnow(),
                report=KnowledgeTransferReport(per_domain={}, probe_results=[]),
                warnings=[f"Merged model probing failed: {e}"],
            )

        # Compute retention by domain
        per_domain = compute_retention_by_domain(
            merged_probe_results,
            source_pass_rates=self._compute_source_pass_rates(source_results, probes)
            if source_results
            else None,
        )

        # Build report
        report = KnowledgeTransferReport(
            per_domain=per_domain,
            probe_results=merged_probe_results,
        )

        execution_time = time.time() - start_time

        return KnowledgeTransferValidationResult(
            validation_id=validation_id,
            merged_model=merged_model,
            source_model=source_model,
            validated_at=datetime.utcnow(),
            report=report,
            probes_executed=len(merged_probe_results),
            execution_time_seconds=execution_time,
            warnings=warnings,
        )

    def validate_specific_domains(
        self,
        merged_model: str,
        domains: list[KnowledgeDomain],
        source_model: str | None = None,
    ) -> KnowledgeTransferValidationResult:
        """Validate specific knowledge domains only."""
        config = KnowledgeTransferConfig(domains=domains)
        return self.validate(merged_model, source_model, config=config)

    def quick_validate(
        self,
        merged_model: str,
        source_model: str | None = None,
    ) -> KnowledgeTransferValidationResult:
        """Quick validation with subset of probes."""
        config = KnowledgeTransferConfig(
            include_variations=False,  # Skip variations for speed
        )
        return self.validate(merged_model, source_model, config=config)

    def add_custom_probes(self, probes: list[KnowledgeProbe]) -> None:
        """Add custom probes to the corpus."""
        for probe in probes:
            self._corpus.add_probe(probe)

    def get_available_probes(self) -> list[KnowledgeProbe]:
        """Get list of all available probes."""
        return self._corpus.probes.copy()

    def _build_corpus(self, config: KnowledgeTransferConfig) -> KnowledgeProbeCorpus:
        """Build probe corpus with any custom probes."""
        corpus = KnowledgeProbeCorpus.default()
        if config.custom_probes:
            for probe in config.custom_probes:
                corpus.add_probe(probe)
        return corpus

    def _create_inference_fn(self, config: KnowledgeTransferConfig) -> Callable[[str], str]:
        """Create generic inference function."""

        def infer(prompt: str) -> str:
            # This shouldn't be called directly - use model-specific fn
            raise NotImplementedError("Use model-specific inference function")

        return infer

    def _create_model_infer_fn(
        self, model_path: str, config: KnowledgeTransferConfig
    ) -> Callable[[str], str]:
        """Create inference function bound to a specific model."""

        def infer(prompt: str) -> str:
            try:
                result = self._inference_engine.infer(
                    model_path,
                    prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=1.0,
                )
                return result.get("response", "")
            except Exception as e:
                logger.warning(f"Inference failed for prompt: {e}")
                return ""

        return infer

    def _compute_source_pass_rates(
        self,
        source_results: dict[str, bool],
        probes: list[KnowledgeProbe],
    ) -> dict[KnowledgeDomain, float]:
        """Compute per-domain pass rates from source model results."""
        domain_counts: dict[KnowledgeDomain, int] = {}
        domain_passes: dict[KnowledgeDomain, int] = {}

        for probe in probes:
            domain = probe.domain
            if domain not in domain_counts:
                domain_counts[domain] = 0
                domain_passes[domain] = 0

            domain_counts[domain] += 1
            if source_results.get(probe.probe_id, False):
                domain_passes[domain] += 1

        return {
            domain: domain_passes[domain] / domain_counts[domain]
            for domain in domain_counts
            if domain_counts[domain] > 0
        }
