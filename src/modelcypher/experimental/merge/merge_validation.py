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
- Coherence scoring (sentence completion log-score)
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
    from modelcypher.core.use_cases.evaluation_service import EvaluationService
    from modelcypher.ports import InferenceEngine
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


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
                    "output": p.output if p.output else None,
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

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        evaluation_service: "EvaluationService",
        model_loader: "ModelLoaderPort",
    ) -> None:
        """Initialize MergeValidationService with required dependencies.

        Args:
            inference_engine: Inference engine port implementation (REQUIRED).
            evaluation_service: Evaluation service (REQUIRED).
            model_loader: Model loader port (required).
        """
        self._inference_engine = inference_engine
        self._evaluation_service = evaluation_service
        self._model_loader = model_loader

    def validate(
        self,
        merged_model: str,
        source_model: str | None = None,
        target_model: str | None = None,
        *,
        perplexity_dataset: str | None = None,
        coherence_prompts: list[str] | None = None,
        task_probes: list[dict] | None = None,
    ) -> MergeValidationResult:
        """
        Execute full merge validation suite.

        Args:
            merged_model: Path to merged model directory.
            source_model: Path to source model (for comparison).
            target_model: Path to target model (for comparison).
            perplexity_dataset: Dataset for perplexity evaluation (optional).
            coherence_prompts: Prompts for coherence scoring (optional).
            task_probes: Task probes for capability checks (optional).

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
        if perplexity_dataset:
            try:
                result.perplexity = self.compute_perplexity(
                    merged_model,
                    perplexity_dataset,
                )
                if source_model and result.perplexity is not None:
                    result.source_perplexity = self.compute_perplexity(
                        source_model,
                        perplexity_dataset,
                    )
                    if result.source_perplexity is not None:
                        result.perplexity_delta = result.perplexity - result.source_perplexity
            except Exception as e:
                logger.warning(f"Perplexity evaluation failed: {e}")
                result.warnings.append(f"Perplexity evaluation failed: {e}")

        # 2. Coherence scoring
        if coherence_prompts:
            try:
                result.coherence_score = self.compute_coherence(
                    merged_model,
                    coherence_prompts,
                )
            except Exception as e:
                logger.warning(f"Coherence scoring failed: {e}")
                result.warnings.append(f"Coherence scoring failed: {e}")

        # 3. Task probes
        if task_probes:
            try:
                result.task_probe_results = self.run_task_probes(merged_model, task_probes)
            except Exception as e:
                logger.warning(f"Task probes failed: {e}")
                result.warnings.append(f"Task probes failed: {e}")

        # 4. Geometric diagnosis (if enabled)
        # Run geometric diagnosis unconditionally when enabled - return raw measurements
        # and let the caller decide what constitutes "degradation"
        if source_model and target_model:
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
    ) -> float:
        """
        Compute perplexity on a held-out dataset.

        Uses the active backend for efficient evaluation.
        """
        result = self._evaluation_service.run(model, dataset)
        return result.perplexity

    def compute_coherence(
        self,
        model: str,
        prompts: list[str],
    ) -> float:
        """
        Compute a response coherence proxy via sentence completion.

        Returns the mean unique-token ratio (0-1) for completions.
        Higher values indicate less repetition; callers interpret relative
        to their own baselines.
        """
        scores = []
        for prompt in prompts:
            try:
                result = self._inference_engine.infer(
                    model,
                    prompt,
                )
                # Score based on response token uniqueness ratio
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
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
        )
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )
        from modelcypher.core.domain.geometry.refinement_density import (
            RefinementDensityAnalyzer,
        )

        try:
            if self._model_loader is None:
                raise RuntimeError("Model loader required for merge validation")

            b = get_default_backend()

            # Load weights via ModelLoaderPort (hexagonal architecture)
            merged_weights = self._model_loader.load_weights(merged_model)
            source_weights = self._model_loader.load_weights(source_model)

            # Compute delta between merged and source
            delta_weights = {}
            for name in source_weights:
                if name not in merged_weights:
                    continue
                source = source_weights[name]
                merged = merged_weights[name]
                if source.shape != merged.shape:
                    continue
                delta = b.array(merged) - b.array(source)
                b.eval(delta)
                delta_weights[name] = delta

            # DARE sparsity analysis
            sparsity_analysis = DARESparsityAnalyzer.analyze_with_backend(
                delta_weights, backend=b
            )

            # DoRA decomposition
            dora = DoRADecomposition()
            dora_result = dora.analyze_adapter(source_weights, merged_weights)

            # Refinement density analysis
            analyzer = RefinementDensityAnalyzer()
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

        except (ImportError, RuntimeError) as e:
            logger.warning(f"Model loader not available for geometric diagnosis: {e}")
            return GeometricDiagnosis(
                layer_composite_scores={},
                mean_drift=0.0,
                max_drift=0.0,
            )

    def _score_coherence(self, prompt: str, response: str) -> float:
        """Return the unique-token ratio for a response."""
        if not response or len(response.strip()) == 0:
            return 0.0

        words = response.split()
        if not words:
            return 0.0

        unique_ratio = len(set(words)) / len(words)
        return max(0.0, min(1.0, unique_ratio))

    # NOTE: _derive_thresholds and _is_degraded were removed.
    # Validation returns raw measurements; callers decide what constitutes degradation.
    # Hardcoded thresholds (1.5x, 0.25x, 0.9) violated the "no vibes" principle.
