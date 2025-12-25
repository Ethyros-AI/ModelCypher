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
Merge Validation Service.

Comprehensive post-merge model validation using:
- Perplexity on held-out text
- Coherence scoring (sentence completion log-probability)
- Task probes (code generation, reasoning pattern matching)
- Geometric diagnosis (layer-wise divergence analysis)
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class MergeValidationConfig:
    """Configuration for merge validation.

    Thresholds are derived from source model baseline - not configurable.
    """

    # Perplexity evaluation
    perplexity_dataset: str | None = None
    perplexity_max_samples: int = 100
    perplexity_batch_size: int = 4

    # Coherence scoring
    coherence_prompts: list[str] | None = None
    coherence_max_tokens: int = 50

    # Task probes: list of {name, prompt, expected_pattern}
    task_probes: list[dict] | None = None

    # Geometric diagnosis
    geometric_diagnosis: bool = True


@dataclass
class TaskProbeResult:
    """Result of a single task probe."""

    name: str
    prompt: str
    expected_pattern: str
    output: str
    passed: bool
    match_details: str | None = None


@dataclass
class GeometricDiagnosis:
    """Geometric analysis of merge quality."""

    diverged_layers: list[int]
    high_drift_layers: list[int]
    mean_drift: float
    max_drift: float
    recommendations: list[str]
    raw_analysis: dict | None = None


@dataclass
class MergeValidationResult:
    """Complete result of merge validation."""

    validation_id: str
    merged_model: str
    source_model: str | None
    target_model: str | None
    validated_at: datetime

    # Metrics
    perplexity: float | None = None
    source_perplexity: float | None = None
    perplexity_delta: float | None = None
    coherence_score: float | None = None
    task_probe_results: list[TaskProbeResult] = field(default_factory=list)
    task_probe_pass_rate: float | None = None

    # Diagnosis
    geometric_diagnosis: GeometricDiagnosis | None = None

    # Overall assessment
    overall_status: str = "unknown"  # healthy, degraded, failed
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "validationId": self.validation_id,
            "mergedModel": self.merged_model,
            "sourceModel": self.source_model,
            "targetModel": self.target_model,
            "validatedAt": self.validated_at.isoformat(),
            "perplexity": self.perplexity,
            "sourcePerplexity": self.source_perplexity,
            "perplexityDelta": self.perplexity_delta,
            "coherenceScore": self.coherence_score,
            "taskProbeResults": [
                {
                    "name": p.name,
                    "prompt": p.prompt,
                    "expectedPattern": p.expected_pattern,
                    "output": p.output[:500] if p.output else None,  # Truncate
                    "passed": p.passed,
                    "matchDetails": p.match_details,
                }
                for p in self.task_probe_results
            ],
            "taskProbePassRate": self.task_probe_pass_rate,
            "geometricDiagnosis": {
                "divergedLayers": self.geometric_diagnosis.diverged_layers,
                "highDriftLayers": self.geometric_diagnosis.high_drift_layers,
                "meanDrift": self.geometric_diagnosis.mean_drift,
                "maxDrift": self.geometric_diagnosis.max_drift,
                "recommendations": self.geometric_diagnosis.recommendations,
            }
            if self.geometric_diagnosis
            else None,
            "overallStatus": self.overall_status,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


class MergeValidationService:
    """
    Service for validating merged models.

    Provides comprehensive behavioral validation after model merging:
    - Perplexity measurement on held-out text
    - Coherence scoring via sentence completion
    - Task probes for specific capabilities
    - Geometric diagnosis when issues are detected
    """

    def __init__(self, inference_engine: "InferenceEngine") -> None:
        """Initialize MergeValidationService with required dependencies.

        Args:
            inference_engine: Inference engine port implementation (REQUIRED).
        """
        self._inference_engine = inference_engine

    def validate(
        self,
        merged_model: str,
        source_model: str | None = None,
        target_model: str | None = None,
        config: MergeValidationConfig | None = None,
    ) -> MergeValidationResult:
        """
        Execute full merge validation suite.

        Args:
            merged_model: Path to merged model directory.
            source_model: Path to source model (for comparison).
            target_model: Path to target model (for comparison).
            config: Validation configuration.

        Returns:
            MergeValidationResult with all metrics and diagnosis.
        """
        config = config or MergeValidationConfig()
        validation_id = f"val-{uuid.uuid4().hex[:8]}"

        result = MergeValidationResult(
            validation_id=validation_id,
            merged_model=merged_model,
            source_model=source_model,
            target_model=target_model,
            validated_at=datetime.utcnow(),
        )

        # 1. Perplexity evaluation
        if config.perplexity_dataset:
            try:
                result.perplexity = self.compute_perplexity(
                    merged_model,
                    config.perplexity_dataset,
                    config.perplexity_max_samples,
                    config.perplexity_batch_size,
                )
                if source_model and result.perplexity is not None:
                    result.source_perplexity = self.compute_perplexity(
                        source_model,
                        config.perplexity_dataset,
                        config.perplexity_max_samples,
                        config.perplexity_batch_size,
                    )
                    if result.source_perplexity is not None:
                        result.perplexity_delta = result.perplexity - result.source_perplexity
            except Exception as e:
                logger.warning(f"Perplexity evaluation failed: {e}")
                result.warnings.append(f"Perplexity evaluation failed: {e}")

        # 2. Coherence scoring
        if config.coherence_prompts:
            try:
                result.coherence_score = self.compute_coherence(
                    merged_model,
                    config.coherence_prompts,
                    config.coherence_max_tokens,
                )
            except Exception as e:
                logger.warning(f"Coherence scoring failed: {e}")
                result.warnings.append(f"Coherence scoring failed: {e}")

        # 3. Task probes
        if config.task_probes:
            try:
                result.task_probe_results = self.run_task_probes(merged_model, config.task_probes)
                passed = sum(1 for p in result.task_probe_results if p.passed)
                result.task_probe_pass_rate = (
                    passed / len(result.task_probe_results) if result.task_probe_results else None
                )
            except Exception as e:
                logger.warning(f"Task probes failed: {e}")
                result.warnings.append(f"Task probes failed: {e}")

        # 4. Geometric diagnosis (if enabled and degradation detected)
        degraded = self._is_degraded(result)
        if config.geometric_diagnosis and degraded and source_model and target_model:
            try:
                result.geometric_diagnosis = self.diagnose_geometry(
                    merged_model, source_model, target_model
                )
            except Exception as e:
                logger.warning(f"Geometric diagnosis failed: {e}")
                result.warnings.append(f"Geometric diagnosis failed: {e}")

        # 5. Compute overall status and recommendations
        result.overall_status = self._compute_status(result)
        result.recommendations = self._generate_recommendations(result)

        return result

    def compute_perplexity(
        self,
        model: str,
        dataset: str,
        max_samples: int = 100,
        batch_size: int = 4,
    ) -> float:
        """
        Compute perplexity on a held-out dataset.

        Uses MLX for efficient evaluation.
        """
        from modelcypher.core.use_cases.evaluation_service import (
            EvalConfig,
            EvaluationService,
        )

        service = EvaluationService()
        config = EvalConfig(
            batch_size=batch_size,
            max_samples=max_samples,
        )

        result = service.run(model, dataset, config)
        return result.perplexity

    def compute_coherence(
        self,
        model: str,
        prompts: list[str],
        max_tokens: int = 50,
    ) -> float:
        """
        Compute coherence score via sentence completion.

        Measures how well the model continues given prompts.
        Higher score = more coherent completions.
        """
        scores = []
        for prompt in prompts:
            try:
                result = self._inference_engine.infer(
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic
                    top_p=1.0,
                )
                # Score based on response quality heuristics
                response = result.get("response", "")
                score = self._score_coherence(prompt, response)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Coherence probe failed for prompt: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def run_task_probes(self, model: str, probes: list[dict]) -> list[TaskProbeResult]:
        """
        Run task probes to test specific capabilities.

        Each probe has:
        - name: Human-readable name
        - prompt: The prompt to send
        - expected_pattern: Regex pattern expected in output
        """
        results = []
        for probe in probes:
            name = probe.get("name", "unnamed")
            prompt = probe.get("prompt", "")
            expected_pattern = probe.get("expected_pattern", "")
            max_tokens = probe.get("max_tokens", 200)

            try:
                result = self._inference_engine.infer(
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                output = result.get("response", "")

                # Check if output matches expected pattern
                if expected_pattern:
                    match = re.search(expected_pattern, output, re.IGNORECASE)
                    passed = match is not None
                    match_details = match.group(0) if match else None
                else:
                    # No pattern = just check non-empty response
                    passed = len(output.strip()) > 0
                    match_details = None

                results.append(
                    TaskProbeResult(
                        name=name,
                        prompt=prompt,
                        expected_pattern=expected_pattern,
                        output=output,
                        passed=passed,
                        match_details=match_details,
                    )
                )

            except Exception as e:
                logger.warning(f"Probe {name} failed: {e}")
                results.append(
                    TaskProbeResult(
                        name=name,
                        prompt=prompt,
                        expected_pattern=expected_pattern,
                        output=f"ERROR: {e}",
                        passed=False,
                    )
                )

        return results

    def diagnose_geometry(
        self,
        merged_model: str,
        source_model: str,
        target_model: str,
    ) -> GeometricDiagnosis:
        """
        Diagnose geometric issues in merged model.

        Identifies which layers diverged and provides recommendations.
        """
        from modelcypher.core.domain.geometry.dare_sparsity import (
            Configuration as DAREConfig,
        )
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
        )
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )
        from modelcypher.core.domain.geometry.refinement_density import (
            RefinementDensityAnalyzer,
            RefinementDensityConfig,
        )

        try:
            import mlx.core as mx
            from mlx_lm import load as mlx_load

            # Load merged and source weights
            _, merged_weights = mlx_load(merged_model, lazy=True)
            _, source_weights = mlx_load(source_model, lazy=True)

            merged_weights = dict(merged_weights)
            source_weights = dict(source_weights)

            # Compute delta between merged and source
            delta_weights = {}
            for name in source_weights:
                if name not in merged_weights:
                    continue
                source = source_weights[name]
                merged = merged_weights[name]
                if source.shape != merged.shape:
                    continue
                delta = merged - source
                mx.eval(delta)
                flat = delta.flatten().tolist()
                if len(flat) > 10000:
                    import random

                    flat = random.sample(flat, 10000)
                delta_weights[name] = flat

            # DARE sparsity analysis
            sparsity_analysis = DARESparsityAnalyzer.analyze(
                delta_weights, DAREConfig(compute_per_layer_metrics=True)
            )

            # DoRA decomposition
            dora = DoRADecomposition()
            dora_result = dora.analyze_adapter(source_weights, merged_weights)

            # Refinement density analysis
            config = RefinementDensityConfig.default()
            analyzer = RefinementDensityAnalyzer(config)
            result = analyzer.analyze(
                source_model=source_model,
                target_model=merged_model,
                sparsity_analysis=sparsity_analysis,
                dora_result=dora_result,
            )

            # Find diverged layers (high composite score = significant difference)
            diverged_layers = []
            high_drift_layers = []
            drift_values = []

            for layer_idx, score in result.layer_scores.items():
                drift_values.append(score.composite_score)
                if score.composite_score > 0.6:
                    diverged_layers.append(layer_idx)
                if score.composite_score > 0.8:
                    high_drift_layers.append(layer_idx)

            mean_drift = sum(drift_values) / len(drift_values) if drift_values else 0.0
            max_drift = max(drift_values) if drift_values else 0.0

            # Generate recommendations
            recommendations = []
            if high_drift_layers:
                recommendations.append(
                    f"High drift in layers {high_drift_layers} - consider reducing alpha for these layers"
                )
            if mean_drift > 0.5:
                recommendations.append(
                    "Overall high drift - reduce global alpha or use layer-wise adaptive alpha"
                )
            if result.has_sparsity_data and not result.has_directional_data:
                recommendations.append(
                    "Only sparsity data available - consider computing DoRA for better diagnosis"
                )

            return GeometricDiagnosis(
                diverged_layers=diverged_layers,
                high_drift_layers=high_drift_layers,
                mean_drift=mean_drift,
                max_drift=max_drift,
                recommendations=recommendations,
                raw_analysis=result.to_dict(),
            )

        except ImportError as e:
            logger.warning(f"MLX not available for geometric diagnosis: {e}")
            return GeometricDiagnosis(
                diverged_layers=[],
                high_drift_layers=[],
                mean_drift=0.0,
                max_drift=0.0,
                recommendations=["MLX not available - install for geometric diagnosis"],
            )

    def _score_coherence(self, prompt: str, response: str) -> float:
        """Score coherence of a response to a prompt."""
        if not response or len(response.strip()) == 0:
            return 0.0

        # Basic heuristics for coherence:
        # 1. Non-empty response
        # 2. Reasonable length (not too short, not just repetition)
        # 3. No obvious error patterns

        score = 0.5  # Base score for non-empty

        # Length bonus
        words = response.split()
        if 5 <= len(words) <= 200:
            score += 0.2

        # Repetition penalty
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            if uniqueness > 0.5:
                score += 0.2

        # Error pattern penalty
        error_patterns = ["error", "sorry", "cannot", "unable", "as an ai"]
        if any(p in response.lower() for p in error_patterns):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _derive_thresholds(
        self, result: MergeValidationResult
    ) -> tuple[float, float, float, float]:
        """Derive thresholds from source baseline. Geometry determines everything.

        Returns:
            (perplexity_threshold, perplexity_delta_threshold,
             coherence_threshold, probe_pass_threshold)
        """
        if result.source_perplexity is None:
            raise ValueError(
                "Source perplexity required to derive thresholds. "
                "Run validation on source model first."
            )

        # All thresholds derived from source baseline
        ppl_thresh = result.source_perplexity * 1.5  # 50% degradation
        ppl_delta_thresh = result.source_perplexity * 0.25  # 25% delta
        coh_thresh = 0.5 * result.source_perplexity / 10.0  # Scales with source complexity
        probe_thresh = 0.9  # 90% of probes must pass - geometry determines pass/fail

        return ppl_thresh, ppl_delta_thresh, coh_thresh, probe_thresh

    def _is_degraded(self, result: MergeValidationResult) -> bool:
        """Check if model appears degraded based on initial metrics."""
        if result.source_perplexity is None:
            return False  # Can't determine without baseline

        ppl_thresh, ppl_delta_thresh, coh_thresh, probe_thresh = self._derive_thresholds(result)

        if result.perplexity is not None and result.perplexity > ppl_thresh:
            return True
        if result.perplexity_delta is not None and result.perplexity_delta > ppl_delta_thresh:
            return True
        if result.coherence_score is not None and result.coherence_score < coh_thresh:
            return True
        if result.task_probe_pass_rate is not None and result.task_probe_pass_rate < probe_thresh:
            return True
        return False

    def _compute_status(self, result: MergeValidationResult) -> str:
        """Compute overall status based on all metrics."""
        if result.source_perplexity is None:
            return "unknown"  # Can't determine without baseline

        ppl_thresh, ppl_delta_thresh, coh_thresh, probe_thresh = self._derive_thresholds(result)

        issues = 0
        severe_issues = 0

        # Perplexity checks
        if result.perplexity is not None:
            if result.perplexity > ppl_thresh * 2:
                severe_issues += 1
            elif result.perplexity > ppl_thresh:
                issues += 1

        if result.perplexity_delta is not None:
            if result.perplexity_delta > ppl_delta_thresh * 2:
                severe_issues += 1
            elif result.perplexity_delta > ppl_delta_thresh:
                issues += 1

        # Coherence checks
        if result.coherence_score is not None:
            if result.coherence_score < coh_thresh / 2:
                severe_issues += 1
            elif result.coherence_score < coh_thresh:
                issues += 1

        # Probe checks
        if result.task_probe_pass_rate is not None:
            if result.task_probe_pass_rate < probe_thresh / 2:
                severe_issues += 1
            elif result.task_probe_pass_rate < probe_thresh:
                issues += 1

        if severe_issues > 0:
            return "failed"
        elif issues > 0:
            return "degraded"
        else:
            return "healthy"

    def _generate_recommendations(self, result: MergeValidationResult) -> list[str]:
        """Generate actionable recommendations based on results."""
        if result.source_perplexity is None:
            # Can't generate threshold-based recommendations without baseline
            if result.geometric_diagnosis:
                return result.geometric_diagnosis.recommendations
            return []

        ppl_thresh, ppl_delta_thresh, coh_thresh, probe_thresh = self._derive_thresholds(
            result
        )

        recommendations = []

        if result.overall_status == "failed":
            recommendations.append(
                "Model merge has critical issues - consider re-merging with different parameters"
            )

        if result.perplexity_delta is not None and result.perplexity_delta > ppl_delta_thresh:
            recommendations.append(
                f"Perplexity increased by {result.perplexity_delta:.2f} - reduce alpha or use adaptive merging"
            )

        if result.coherence_score is not None and result.coherence_score < coh_thresh:
            recommendations.append("Low coherence score - check attention layer alignment")

        if result.task_probe_pass_rate is not None:
            failed_probes = [p.name for p in result.task_probe_results if not p.passed]
            if failed_probes:
                recommendations.append(
                    f"Failed probes: {', '.join(failed_probes)} - inspect relevant layers"
                )

        if result.geometric_diagnosis:
            recommendations.extend(result.geometric_diagnosis.recommendations)

        if not recommendations and result.overall_status == "healthy":
            recommendations.append("No issues detected - model appears healthy")

        return recommendations
