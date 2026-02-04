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

"""Multilingual intensity calibration for linguistic modifiers.

Provides calibration logic for modifier intensity across languages with
different resource levels using measured entropy effects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from modelcypher.core.domain.geometry.numerical_stability import model_eps

from modelcypher.core.domain.entropy.entropy_math import EntropyMath
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    LinguisticModifier,
    MultilingualPerturbedPrompt,
    PromptLanguage,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.thermo.linguistic_calorimeter import (
        LinguisticCalorimeter,
    )


# =============================================================================
# Calibration Results
# =============================================================================


@dataclass(frozen=True)
class CalibratedIntensity:
    """Result of intensity calibration for a specific language."""

    language: PromptLanguage
    base_intensity: float
    calibrated_intensity: float
    scaling_factor: float


@dataclass
class LanguageParityResult:
    """Parity test result for a single language.

    All values are measured, not predicted. Parity is determined by comparing
    measured effects across languages relative to a reference language.
    """

    language: PromptLanguage
    modifier: LinguisticModifier
    baseline_entropy: float
    modified_entropy: float
    delta_h: float
    relative_effect: float | None = None  # Relative to reference language

    @property
    def effect_magnitude(self) -> float:
        """Absolute magnitude of the entropy change."""
        return abs(self.delta_h)


@dataclass
class ParityReport:
    """Cross-lingual parity test report.

    Reports measured effects across languages. No predictions or expected values.
    """

    id: UUID
    prompt: str
    modifier: LinguisticModifier
    results: list[LanguageParityResult]
    reference_language: PromptLanguage | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def languages_tested(self) -> list[PromptLanguage]:
        """Languages included in this test."""
        return [r.language for r in self.results]

    @property
    def effect_variance(self) -> float:
        """Variance in effect magnitude across languages (lower = more consistent)."""
        if len(self.results) < 2:
            return 0.0
        effects = [r.effect_magnitude for r in self.results]
        mean_effect = sum(effects) / len(effects)
        return sum((e - mean_effect) ** 2 for e in effects) / (len(effects) - 1)

    @classmethod
    def create(
        cls,
        prompt: str,
        modifier: LinguisticModifier,
        results: list[LanguageParityResult],
        reference_language: PromptLanguage | None = None,
    ) -> ParityReport:
        """Create a new parity report."""
        return cls(
            id=uuid4(),
            prompt=prompt,
            modifier=modifier,
            results=results,
            reference_language=reference_language,
        )


# =============================================================================
# Multilingual Calibrator
# =============================================================================


class MultilingualCalibrator:
    """Calibrate modifier intensity for different languages.

    Notes
    -----
    Calibration is derived from measured entropy data, not hardcoded factors.
    Call compute_calibration() with actual entropy measurements to set up
    language-specific scaling.
    """

    def __init__(self):
        # Calibration factors derived from data, not preset
        self._calibration: dict[PromptLanguage, float] = {}
        self._reference_language: PromptLanguage | None = None

    def compute_calibration(
        self,
        entropy_by_language: dict[PromptLanguage, float],
        reference_language: PromptLanguage = PromptLanguage.ENGLISH,
    ) -> None:
        """Derive calibration factors from measured entropy data.

        Parameters
        ----------
        entropy_by_language : dict[PromptLanguage, float]
            Measured entropy values per language.
        reference_language : PromptLanguage
            Reference language for calibration (default: English).

        Notes
        -----
        Scaling factor = reference_entropy / language_entropy
        """
        if reference_language not in entropy_by_language:
            raise ValueError(f"Reference language {reference_language} not in measurements")

        reference_entropy = entropy_by_language[reference_language]
        self._reference_language = reference_language

        for lang, entropy in entropy_by_language.items():
            if entropy > model_eps():
                self._calibration[lang] = reference_entropy / entropy
            else:
                self._calibration[lang] = 1.0

    def calibrate_intensity(
        self,
        language: PromptLanguage,
        base_intensity: float,
    ) -> CalibratedIntensity:
        """Scale modifier intensity using computed calibration factors."""
        scaling = self._calibration.get(language, 1.0)
        calibrated = base_intensity * scaling

        return CalibratedIntensity(
            language=language,
            base_intensity=base_intensity,
            calibrated_intensity=calibrated,
            scaling_factor=scaling,
        )

    def expected_delta_h(
        self,
        language: PromptLanguage,
        measured_effect: float,
    ) -> float:
        """Expected delta_H scaled by calibration for this language.

        Parameters
        ----------
        language : PromptLanguage
            Target language.
        measured_effect : float
            Measured effect in reference language (from calibration data).

        Returns
        -------
        float
            Expected delta_H for this language based on calibration.
        """
        calibrated = self.calibrate_intensity(language, measured_effect)
        return calibrated.calibrated_intensity

    def cross_lingual_parity_test(
        self,
        prompt: str,
        modifier: LinguisticModifier,
        calorimeter: "LinguisticCalorimeter",
        languages: list[PromptLanguage] | None = None,
        reference_language: PromptLanguage = PromptLanguage.ENGLISH,
    ) -> ParityReport:
        """Test modifier effect consistency across languages.

        Measures entropy deltas across languages. All values are measured, not predicted.

        Args:
            prompt: Base prompt to test (in English, will be translated conceptually).
            modifier: Modifier to test.
            calorimeter: LinguisticCalorimeter instance for measurements.
            languages: Languages to test. Defaults to all.
            reference_language: Language to use as reference for relative effects.

        Returns:
            ParityReport with results for each language.
        """
        if languages is None:
            languages = list(PromptLanguage)

        # First pass: measure all languages
        measurements: dict[PromptLanguage, tuple[float, float, float]] = {}

        for language in languages:
            # Create multilingual perturbed prompt
            baseline_prompt = MultilingualPerturbedPrompt.create(
                base_content=prompt,
                modifier=LinguisticModifier.BASELINE,
                language=language,
            )
            modified_prompt = MultilingualPerturbedPrompt.create(
                base_content=prompt,
                modifier=modifier,
                language=language,
            )

            # Measure entropy for both
            baseline_measurement = calorimeter.measure_entropy(
                prompt=baseline_prompt.full_prompt,
            )
            modified_measurement = calorimeter.measure_entropy(
                prompt=modified_prompt.full_prompt,
            )

            baseline_entropy = baseline_measurement.mean_entropy
            modified_entropy = modified_measurement.mean_entropy
            delta_h = EntropyMath.compute_delta_h(modified_entropy, baseline_entropy)

            measurements[language] = (baseline_entropy, modified_entropy, delta_h)

        # Get reference effect for relative comparisons
        reference_effect = abs(measurements.get(reference_language, (0, 0, 0))[2])

        # Build results with relative effects
        results = []
        for language in languages:
            baseline_entropy, modified_entropy, delta_h = measurements[language]

            # Compute relative effect (1.0 = same as reference)
            relative_effect: float | None = None
            if reference_effect > model_eps():
                relative_effect = abs(delta_h) / reference_effect

            result = LanguageParityResult(
                language=language,
                modifier=modifier,
                baseline_entropy=baseline_entropy,
                modified_entropy=modified_entropy,
                delta_h=delta_h,
                relative_effect=relative_effect,
            )
            results.append(result)

        return ParityReport.create(
            prompt=prompt,
            modifier=modifier,
            results=results,
            reference_language=reference_language,
        )

    def generate_calibration_table(self) -> str:
        """Generate markdown table showing calibration parameters.

        Only shows measured calibration factors, not predictions.
        """
        if not self._calibration:
            return (
                "# Multilingual Intensity Calibration\n\n"
                "**No calibration data available.**\n\n"
                "Run `compute_calibration()` with measured entropy data first."
            )

        lines = [
            "# Multilingual Intensity Calibration",
            "",
            f"**Reference Language**: {self._reference_language.display_name if self._reference_language else 'None'}",
            "",
            "## Measured Scaling Factors",
            "",
            "| Language | Resource Score | Scaling Factor |",
            "|----------|----------------|----------------|",
        ]

        for language in PromptLanguage:
            calibrated = self.calibrate_intensity(language, 1.0)
            lines.append(
                f"| {language.display_name} | {language.resource_score:.1f} | "
                f"{calibrated.scaling_factor:.2f} |"
            )

        lines.extend(
            [
                "",
                "Scaling factors are derived from measured entropy ratios:",
                "  scaling_factor = reference_entropy / language_entropy",
            ]
        )

        return "\n".join(lines)
