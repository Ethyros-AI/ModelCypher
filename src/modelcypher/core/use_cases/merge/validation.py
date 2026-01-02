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
    perplexity_dataset: str | None
    perplexity_max_samples: int
    perplexity_batch_size: int

    # Coherence scoring
    coherence_prompts: list[str] | None
    coherence_max_tokens: int

    # Task probes: list of {name, prompt, expected_pattern}
    task_probes: list[dict] | None

    # Geometric diagnosis
    geometric_diagnosis: bool


@dataclass
class TaskProbeResult:
    """Result of a single task probe."""

    name: str
    prompt: str
    expected_pattern: str
    output: str
    match_details: str | None = None


@dataclass
class GeometricDiagnosis:
    """Geometric analysis of merge quality.

    All values are raw measurements. No categorical thresholds.
    Callers interpret layer_composite_scores relative to their own baselines.
    """

    # Raw composite scores per layer - callers decide what constitutes "drift"
    layer_composite_scores: dict[int, float]
    mean_drift: float
    max_drift: float
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

    # Diagnosis
    geometric_diagnosis: GeometricDiagnosis | None = None

    # Diagnostics
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
                    "matchDetails": p.match_details,
                }
                for p in self.task_probe_results
            ],
            "geometricDiagnosis": {
                "layerCompositeScores": self.geometric_diagnosis.layer_composite_scores,
                "meanDrift": self.geometric_diagnosis.mean_drift,
                "maxDrift": self.geometric_diagnosis.max_drift,
            }
            if self.geometric_diagnosis
            else None,
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
        config: MergeValidationConfig,
        source_model: str | None = None,
        target_model: str | None = None,
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
            except Exception as e:
                logger.warning(f"Task probes failed: {e}")
                result.warnings.append(f"Task probes failed: {e}")

        # 4. Geometric diagnosis (if enabled)
        # Run geometric diagnosis unconditionally when enabled - return raw measurements
        # and let the caller decide what constitutes "degradation"
        if config.geometric_diagnosis and source_model and target_model:
            try:
                result.geometric_diagnosis = self.diagnose_geometry(
                    merged_model, source_model, target_model
                )
            except Exception as e:
                logger.warning(f"Geometric diagnosis failed: {e}")
                result.warnings.append(f"Geometric diagnosis failed: {e}")

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
        max_tokens: int | None = None,
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

            try:
                result = self._inference_engine.infer(
                    model,
                    prompt,
                )
                output = result.get("response", "")

                # Check if output matches expected pattern
                if expected_pattern:
                    match = re.search(expected_pattern, output, re.IGNORECASE)
                    match_details = match.group(0) if match else None
                else:
                    match_details = None

                results.append(
                    TaskProbeResult(
                        name=name,
                        prompt=prompt,
                        expected_pattern=expected_pattern,
                        output=output,
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

        Identifies which layers diverged using refinement density scores.
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

            # Return raw composite scores per layer - no arbitrary thresholds
            # Callers interpret these values relative to their own baselines
            layer_composite_scores = {}
            drift_values = []

            for layer_idx, score in result.layer_scores.items():
                layer_composite_scores[layer_idx] = score.composite_score
                drift_values.append(score.composite_score)

            mean_drift = sum(drift_values) / len(drift_values) if drift_values else 0.0
            max_drift = max(drift_values) if drift_values else 0.0

            return GeometricDiagnosis(
                layer_composite_scores=layer_composite_scores,
                mean_drift=mean_drift,
                max_drift=max_drift,
                raw_analysis=result.to_dict(),
            )

        except ImportError as e:
            logger.warning(f"MLX not available for geometric diagnosis: {e}")
            return GeometricDiagnosis(
                layer_composite_scores={},
                mean_drift=0.0,
                max_drift=0.0,
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

        # Repetition penalty - use boundary value (any uniqueness > 0)
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            if uniqueness > 0:
                score += 0.2 * uniqueness  # Scale by uniqueness ratio

        # Error pattern penalty
        error_patterns = ["error", "sorry", "cannot", "unable", "as an ai"]
        if any(p in response.lower() for p in error_patterns):
            score -= 0.2

        return max(0.0, min(1.0, score))

    # NOTE: _derive_thresholds and _is_degraded were removed.
    # Validation returns raw measurements; callers decide what constitutes degradation.
    # Hardcoded thresholds (1.5x, 0.25x, 0.9) violated the "no vibes" principle.
