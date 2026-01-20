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

"""Thermo service for thermodynamic analysis of training."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
)

from modelcypher.core.domain.geometry.thermo_path_integration import (
    CombinedMeasurement,
    ThermoPathIntegration,
)
from modelcypher.ports.embedding import EmbeddingProvider

if TYPE_CHECKING:
    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermoAnalysisResult:
    """Result of thermodynamic analysis."""

    job_id: str
    entropy: float
    temperature: float


@dataclass(frozen=True)
class ThermoPathResult:
    """Result of path integration analysis."""

    checkpoints: list[str]
    path_length: float
    curvature: float


@dataclass(frozen=True)
class ThermoPathIntegrationResult:
    """Result of thermo-path integration analysis."""

    model_id: str
    prompt: str
    response_text: str
    measurement: CombinedMeasurement


@dataclass(frozen=True)
class ThermoEntropyResult:
    """Entropy metrics over training."""

    job_id: str
    entropy_history: list[dict]
    final_entropy: float
    entropy_delta: float
    entropy_ratio: float | None


# --- New data classes for measure/detect ---


@dataclass(frozen=True)
class LinguisticModifier:
    """A linguistic modifier that transforms prompts.

    Modifiers are text transformations whose effects are MEASURED, not assumed.
    """

    name: str
    transform: Callable[[str], str]


@dataclass(frozen=True)
class ModifierMeasurement:
    """Entropy measurement for a single modifier.

    Raw measurements only - no behavioral classification.
    """

    modifier: str
    mean_entropy: float
    delta_h: float | None
    ridge_crossed: bool


@dataclass(frozen=True)
class ThermoStatistics:
    """Statistics across all modifier measurements.

    All values are computed from actual measurements, not predictions.
    """

    mean_entropy: float
    std_entropy: float
    min_entropy: float
    max_entropy: float
    mean_delta_h: float | None


@dataclass(frozen=True)
class ThermoMeasureResult:
    """Result of entropy measurement across linguistic modifiers."""

    base_prompt: str
    measurements: list[ModifierMeasurement]
    statistics: ThermoStatistics
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ThermoDetectResult:
    """Result of prompt entropy measurement.

    Raw measurements only - no classification or risk levels.
    The caller interprets delta_h relative to their baseline.
    """

    prompt: str
    baseline_entropy: float
    intensity_entropy: float
    delta_h: float
    processing_time: float


# Default linguistic modifiers - their effects are measured, not assumed
DEFAULT_MODIFIERS: list[LinguisticModifier] = [
    LinguisticModifier(
        name="baseline",
        transform=lambda p: p,
    ),
    LinguisticModifier(
        name="caps",
        transform=lambda p: p.upper(),
    ),
    LinguisticModifier(
        name="direct",
        transform=lambda p: f"Answer directly: {p}",
    ),
    LinguisticModifier(
        name="polite",
        transform=lambda p: f"Please {p.lower()}",
    ),
    LinguisticModifier(
        name="roleplay",
        transform=lambda p: f"Pretend you are an expert. {p}",
    ),
    LinguisticModifier(
        name="negation",
        transform=lambda p: f"Do not refuse to answer: {p}",
    ),
    LinguisticModifier(
        name="combined",
        transform=lambda p: f"IMPORTANT: Pretend you are an expert. Answer directly: {p}".upper(),
    ),
]


class ThermoService:
    """Service for thermodynamic analysis of training."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        model_loader: "ModelLoaderPort | None" = None,
    ) -> None:
        self._embedder = embedder
        self._model_loader = model_loader
        self._integration = ThermoPathIntegration()
        self._calorimeter: "LinguisticCalorimeter" | None = None
        self._calorimeter_model_path: str | None = None

    def _get_calorimeter(self, model_path: str) -> "LinguisticCalorimeter":
        """Get or create a LinguisticCalorimeter for the given model path.

        Caches the calorimeter for efficiency when making multiple measurements.
        """
        if not model_path:
            raise ValueError("Model path required for thermodynamic analysis")

        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise ValueError(f"Model path '{model_path}' not found")

        # Check if we need to create/recreate the calorimeter
        if self._calorimeter is None or self._calorimeter_model_path != model_path:
            from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter

            self._calorimeter = LinguisticCalorimeter(
                model_path=str(resolved_path),
                model_loader=self._model_loader,
            )
            self._calorimeter_model_path = model_path
            logger.info("Using real inference from '%s'", resolved_path)

        return self._calorimeter

    def analyze(self, job_id: str, model_path: str | None = None) -> ThermoAnalysisResult:
        """Thermodynamic analysis of training job or model.

        Args:
            job_id: Job ID to analyze.
            model_path: Optional model path for direct analysis.

        Returns:
            ThermoAnalysisResult with thermodynamic metrics.
        """
        # If model_path provided, measure entropy directly
        if model_path:
            calorimeter = self._get_calorimeter(model_path)
            measurement = calorimeter.measure_entropy("Analyze the current state.")
        else:
            # Try to load job checkpoint for entropy measurement
            from modelcypher.utils.paths import get_jobs_dir

            job_dir = get_jobs_dir() / job_id
            checkpoint_dir = job_dir / "checkpoints"

            if not checkpoint_dir.exists():
                raise ValueError(
                    f"No checkpoints found for job '{job_id}'. Provide model_path for direct analysis."
                )

            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
            if not checkpoints:
                raise ValueError(
                    f"No checkpoints found for job '{job_id}'. Provide model_path for direct analysis."
                )

            latest = checkpoints[-1]
            calorimeter = self._get_calorimeter(str(latest))
            measurement = calorimeter.measure_entropy("Analyze the current state.")

        return ThermoAnalysisResult(
            job_id=job_id,
            entropy=measurement.mean_entropy,
            temperature=measurement.temperature,
        )

    def path(self, checkpoints: list[str]) -> ThermoPathResult:
        """Path integration analysis between checkpoints.

        Computes the thermodynamic path through weight space by measuring
        entropy at each checkpoint and analyzing the trajectory.

        Args:
            checkpoints: List of checkpoint paths.

        Returns:
            ThermoPathResult with path metrics.
        """
        if len(checkpoints) < 2:
            raise ValueError("At least two checkpoints required for path analysis")

        # Measure entropy at each checkpoint
        entropies: list[float] = []
        for ckpt_path in checkpoints:
            ckpt = Path(ckpt_path)
            if not ckpt.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue
            calorimeter = self._get_calorimeter(str(ckpt))
            measurement = calorimeter.measure_entropy("Analyze checkpoint state.")
            entropies.append(measurement.mean_entropy)

        if len(entropies) < 2:
            raise ValueError("Need at least 2 valid checkpoints for path analysis")

        # Compute path length as sum of entropy changes
        path_length = sum(abs(entropies[i] - entropies[i - 1]) for i in range(1, len(entropies)))

        # Compute curvature as variance in entropy deltas
        # INTENTIONAL SCALAR: Standard deviation of 1D scalar entropy deltas.
        # This is a statistical measure on a list of scalars, not a geometric
        # distance in high-dimensional space.
        deltas = [entropies[i] - entropies[i - 1] for i in range(1, len(entropies))]
        mean_delta = sum(deltas) / len(deltas)
        curvature = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5

        return ThermoPathResult(
            checkpoints=checkpoints,
            path_length=path_length,
            curvature=curvature,
        )

    def path_integration(
        self,
        prompt: str,
        model_path: str,
    ) -> ThermoPathIntegrationResult:
        """Integrate entropy trajectories with gate detections for a response.

        All analysis parameters are derived from the data.
        """
        if self._embedder is None:
            raise ValueError("Embedding provider required for thermo-path integration")

        from modelcypher.core.domain.geometry.gate_detector import GateDetector

        calorimeter = self._get_calorimeter(model_path)

        measurement = calorimeter.measure_entropy(prompt)

        # GateDetector derives all thresholds from data
        detector = GateDetector(embedder=self._embedder)
        model_id = Path(model_path).name if Path(model_path).exists() else model_path
        detection = detector.detect(
            text=measurement.generated_text,
            model_id=model_id,
            prompt_id="thermo-path-integration",
            entropy_trace=measurement.entropy_trajectory,
        )

        # ThermoPathIntegration derives all analysis parameters from data
        integration = ThermoPathIntegration()
        combined = integration.analyze_response(
            response_text=measurement.generated_text,
            entropy_trajectory=measurement.entropy_trajectory,
            gate_detection_result=detection,
        )

        return ThermoPathIntegrationResult(
            model_id=model_id,
            prompt=prompt,
            response_text=measurement.generated_text,
            measurement=combined,
        )

    def entropy(self, job_id: str, model_path: str | None = None) -> ThermoEntropyResult:
        """Entropy metrics over training.

        Args:
            job_id: Job ID to analyze.
            model_path: Optional model path for direct measurement.

        Returns:
            ThermoEntropyResult with entropy history.
        """
        from modelcypher.utils.paths import get_jobs_dir

        entropy_history: list[dict] = []
        job_dir = get_jobs_dir() / job_id

        # Try to load entropy from training checkpoints
        checkpoint_dir = job_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
            for i, ckpt in enumerate(checkpoints):
                try:
                    # Extract step number from checkpoint name
                    step = int(ckpt.name.split("-")[-1]) if "-" in ckpt.name else i * 100
                    calorimeter = self._get_calorimeter(str(ckpt))
                    measurement = calorimeter.measure_entropy("Measure checkpoint entropy.")
                    entropy_history.append({"step": step, "entropy": measurement.mean_entropy})
                except Exception as e:
                    logger.warning(f"Failed to measure entropy for {ckpt}: {e}")

        # If no data and model_path provided, measure current state
        if not entropy_history and model_path:
            calorimeter = self._get_calorimeter(model_path)
            measurement = calorimeter.measure_entropy("Measure current entropy.")
            entropy_history.append({"step": 0, "entropy": measurement.mean_entropy})

        if not entropy_history:
            raise ValueError(
                f"No entropy data found for job '{job_id}'. Provide model_path for direct measurement."
            )

        final_entropy = entropy_history[-1]["entropy"]
        initial_entropy = entropy_history[0]["entropy"]

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        entropy_delta = final_entropy - initial_entropy
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([initial_entropy]))
        entropy_ratio = (
            final_entropy / initial_entropy
            if abs(initial_entropy) > eps
            else None
        )

        return ThermoEntropyResult(
            job_id=job_id,
            entropy_history=entropy_history,
            final_entropy=final_entropy,
            entropy_delta=entropy_delta,
            entropy_ratio=entropy_ratio,
        )

    def measure(
        self,
        prompt: str,
        model_path: str,
    ) -> ThermoMeasureResult:
        """Measure entropy across linguistic modifiers for a prompt.

        Args:
            prompt: The base prompt to measure.
            model_path: Path to the model directory.

        Returns:
            ThermoMeasureResult with measurements and statistics.
        """
        active_modifiers = DEFAULT_MODIFIERS

        measurements: list[ModifierMeasurement] = []
        entropies: list[float] = []
        delta_hs: list[float] = []
        baseline_entropy: float | None = None

        # Get calorimeter for entropy measurement
        calorimeter = self._get_calorimeter(model_path)

        for modifier in active_modifiers:
            transformed_prompt = modifier.transform(prompt)

            # Compute entropy using LinguisticCalorimeter
            measurement = calorimeter.measure_entropy(transformed_prompt)
            entropy = measurement.mean_entropy
            entropies.append(entropy)

            # Compute delta_h relative to baseline
            if modifier.name == "baseline":
                baseline_entropy = entropy
                delta_h = None
            else:
                delta_h = entropy - (baseline_entropy or entropy)
                delta_hs.append(delta_h)

            # ridge_crossed indicates a non-baseline modifier produced a change
            # The magnitude of change is in delta_h - caller determines significance
            ridge_crossed = delta_h is not None

            measurements.append(
                ModifierMeasurement(
                    modifier=modifier.name,
                    mean_entropy=entropy,
                    delta_h=delta_h,
                    ridge_crossed=ridge_crossed,
                )
            )

        # Compute statistics from measured values
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        std_entropy = self._compute_std(entropies)
        min_entropy = min(entropies) if entropies else 0.0
        max_entropy = max(entropies) if entropies else 0.0
        mean_delta_h = sum(delta_hs) / len(delta_hs) if delta_hs else None

        statistics = ThermoStatistics(
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            min_entropy=min_entropy,
            max_entropy=max_entropy,
            mean_delta_h=mean_delta_h,
        )

        return ThermoMeasureResult(
            base_prompt=prompt,
            measurements=measurements,
            statistics=statistics,
        )

    def _compute_std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance**0.5

    def _compute_correlation(self, x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 3:
            return None

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        b = get_default_backend()
        eps = machine_epsilon(b, b.array([1.0], dtype=precision_dtype(b)))
        if denom_x <= eps or denom_y <= eps:
            return None

        return numerator / (denom_x * denom_y)

    def detect(
        self,
        prompt: str,
        model_path: str,
    ) -> ThermoDetectResult:
        """Measure prompt entropy differential.

        Returns raw measurements. The caller interprets delta_h
        relative to their baseline distribution.

        Args:
            prompt: The prompt to analyze.
            model_path: Path to the model directory.

        Returns:
            ThermoDetectResult with raw entropy measurements.
        """
        start_time = time.time()

        # Measure entropy across modifiers
        measure_result = self.measure(prompt, model_path)

        # Extract baseline and intensity entropies
        baseline_entropy = 0.0
        intensity_entropy = 0.0

        for measurement in measure_result.measurements:
            if measurement.modifier == "baseline":
                baseline_entropy = measurement.mean_entropy
            elif measurement.modifier in ("combined", "caps", "direct"):
                # Use highest intensity modifier as intensity_entropy
                if measurement.mean_entropy > intensity_entropy:
                    intensity_entropy = measurement.mean_entropy

        # If no intensity modifier found, use max non-baseline
        if intensity_entropy == 0.0:
            for measurement in measure_result.measurements:
                if measurement.modifier != "baseline":
                    if measurement.mean_entropy > intensity_entropy:
                        intensity_entropy = measurement.mean_entropy

        # Compute delta_h - the raw measurement
        delta_h = intensity_entropy - baseline_entropy

        processing_time = time.time() - start_time

        return ThermoDetectResult(
            prompt=prompt,
            baseline_entropy=baseline_entropy,
            intensity_entropy=intensity_entropy,
            delta_h=delta_h,
            processing_time=processing_time,
        )

    def detect_batch(
        self,
        prompts_file: str,
        model_path: str,
    ) -> list[ThermoDetectResult]:
        """Batch measure entropy differential across multiple prompts.

        Args:
            prompts_file: Path to file containing prompts (JSON array or newline-separated).
            model_path: Path to the model directory.

        Returns:
            List of ThermoDetectResult with raw measurements, one per prompt.
        """
        prompts = self._load_prompts_from_file(prompts_file)

        results: list[ThermoDetectResult] = []
        for prompt in prompts:
            result = self.detect(prompt, model_path)
            results.append(result)

        return results

    def _load_prompts_from_file(self, file_path: str) -> list[str]:
        """Load prompts from a file.

        Supports:
        - JSON array of strings
        - Newline-separated text file

        Args:
            file_path: Path to the prompts file.

        Returns:
            List of prompt strings.
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Prompts file not found: {file_path}")

        content = path.read_text(encoding="utf-8")

        # Try JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, list):
            # Validate all items are strings
            prompts = []
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item["prompt"]))
                else:
                    prompts.append(str(item))
            return prompts

        # Fall back to newline-separated
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]
