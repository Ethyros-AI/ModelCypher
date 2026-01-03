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

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineMetrics:
    mean_entropy: float
    entropy_std_dev: float
    coherence_index: float
    per_prompt_entropy: list[float]
    duration: float


@dataclass(frozen=True)
class ValidationResult:
    baseline: BaselineMetrics
    post_perturbation: BaselineMetrics
    entropy_delta: float
    coherence_change: float
    perturbed_layers: list[int]

    def generate_report(self) -> str:
        report_lines = [
            "# Sparse Region Validation Metrics",
            "",
            "## Metrics Comparison",
            "",
            "| Metric | Baseline | Post-Perturbation | Delta |",
            "|--------|----------|-------------------|-------|",
            f"| Entropy | {self.baseline.mean_entropy:.3f} | {self.post_perturbation.mean_entropy:.3f} | {self.entropy_delta * 100:.1f}% |",
            f"| Coherence Index | {self.baseline.coherence_index:.2f} | {self.post_perturbation.coherence_index:.2f} | {self.coherence_change:+.2f} |",
            "",
            "## Perturbed Layers",
            ", ".join(str(layer) for layer in self.perturbed_layers),
        ]

        return "\n".join(report_lines)


class ValidationPhase(str, Enum):
    baseline = "Measuring baseline"
    perturbation = "Applying perturbation"
    post_measurement = "Measuring post-perturbation"
    analysis = "Analyzing results"


@dataclass(frozen=True)
class ValidationProgress:
    phase: ValidationPhase
    current_prompt: int
    total_prompts: int
    status: str

    @property
    def percentage(self) -> float:
        if self.phase == ValidationPhase.baseline:
            return float(self.current_prompt) / float(max(1, self.total_prompts)) * 0.4
        if self.phase == ValidationPhase.perturbation:
            return 0.4 + float(self.current_prompt) / float(max(1, self.total_prompts)) * 0.1
        if self.phase == ValidationPhase.post_measurement:
            return 0.5 + float(self.current_prompt) / float(max(1, self.total_prompts)) * 0.4
        return 0.9 + float(self.current_prompt) / float(max(1, self.total_prompts)) * 0.1


class SparseRegionValidator:
    """Validates that sparse regions remain stable under perturbation.

    All parameters are derived from the data:
    - Prompts: All validation prompts are used (caller controls set)
    - Perturbation magnitude: Must be derived from weight scale by caller
    """

    def validate(
        self,
        perturbed_layers: list[int],
        validation_prompts: list[str],
        perturbation_magnitude: float,
        measure_metrics: Callable[[list[str]], BaselineMetrics],
        apply_perturbation: Callable[[list[int], float], None],
        remove_perturbation: Callable[[], None],
        progress: Callable[[ValidationProgress], None] | None = None,
    ) -> ValidationResult:
        """Validate sparse regions under perturbation.

        Args:
            perturbed_layers: Layer indices to perturb.
            validation_prompts: Prompts for measuring behavior.
            perturbation_magnitude: Perturbation scale derived from weight statistics
                (e.g., std(weights) * sqrt(machine_epsilon)).
            measure_metrics: Function to measure baseline metrics.
            apply_perturbation: Function to apply perturbation.
            remove_perturbation: Function to remove perturbation.
            progress: Optional progress callback.

        Returns:
            ValidationResult with baseline and post-perturbation metrics.
        """
        if progress:
            progress(
                ValidationProgress(
                    phase=ValidationPhase.baseline,
                    current_prompt=0,
                    total_prompts=len(validation_prompts),
                    status="Measuring baseline...",
                )
            )

        baseline = measure_metrics(validation_prompts)

        layers_to_perturb = sorted(set(perturbed_layers))

        if progress:
            progress(
                ValidationProgress(
                    phase=ValidationPhase.perturbation,
                    current_prompt=0,
                    total_prompts=len(layers_to_perturb),
                    status="Applying perturbation...",
                )
            )

        apply_perturbation(layers_to_perturb, perturbation_magnitude)

        if progress:
            progress(
                ValidationProgress(
                    phase=ValidationPhase.post_measurement,
                    current_prompt=0,
                    total_prompts=len(validation_prompts),
                    status="Measuring post-perturbation...",
                )
            )

        post_perturbation = measure_metrics(validation_prompts)
        remove_perturbation()

        if progress:
            progress(
                ValidationProgress(
                    phase=ValidationPhase.analysis,
                    current_prompt=0,
                    total_prompts=1,
                    status="Analyzing results...",
                )
            )

        return self.analyze_results(
            baseline=baseline,
            post_perturbation=post_perturbation,
            perturbed_layers=layers_to_perturb,
        )

    def analyze_results(
        self,
        baseline: BaselineMetrics,
        post_perturbation: BaselineMetrics,
        perturbed_layers: list[int],
    ) -> ValidationResult:
        backend = get_default_backend()
        eps = float(division_epsilon(backend, backend.array([baseline.mean_entropy])))
        if baseline.mean_entropy > eps:
            entropy_delta = (
                abs(post_perturbation.mean_entropy - baseline.mean_entropy) / baseline.mean_entropy
            )
        else:
            entropy_delta = abs(post_perturbation.mean_entropy - baseline.mean_entropy)

        coherence_change = post_perturbation.coherence_index - baseline.coherence_index
        logger.info(
            "Validation metrics computed: entropy_delta=%.4f, coherence_change=%.4f",
            entropy_delta,
            coherence_change,
        )

        return ValidationResult(
            baseline=baseline,
            post_perturbation=post_perturbation,
            entropy_delta=entropy_delta,
            coherence_change=coherence_change,
            perturbed_layers=perturbed_layers,
        )

    @staticmethod
    def compute_coherence(entropies: list[float]) -> float:
        if not entropies:
            return 0.0
        mean = sum(entropies) / float(len(entropies))
        if len(entropies) > 1:
            variance = sum((value - mean) ** 2 for value in entropies) / float(len(entropies) - 1)
        else:
            variance = 0.0
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array(entropies))
        scale = variance + (mean * mean) + eps
        return 1.0 - (variance / scale)
