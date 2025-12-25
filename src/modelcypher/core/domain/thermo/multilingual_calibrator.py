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
Multilingual Intensity Calibration.

Provides calibration logic for modifier intensity across languages with
different resource levels. Low-resource languages typically show larger
entropy effects due to weaker safety alignment.

Key Concepts:
- Intensity calibration scales modifier strength by language resource level
- Cross-lingual parity testing validates consistent behavior patterns
- Parity reports identify language-specific vulnerabilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    LanguageResourceLevel,
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
    rationale: str


@dataclass
class LanguageParityResult:
    """Parity test result for a single language."""

    language: PromptLanguage
    modifier: LinguisticModifier
    baseline_entropy: float
    modified_entropy: float
    delta_h: float
    expected_delta_magnitude: float
    shows_cooling: bool
    within_expected_range: bool

    @property
    def parity_score(self) -> float:
        """Score [0, 1] indicating how well this matches expected pattern.

        1.0 = perfect match to expected magnitude
        0.0 = no effect or wrong direction
        """
        if not self.shows_cooling:
            return 0.0

        # How close is actual magnitude to expected?
        actual_magnitude = abs(self.delta_h)
        expected = self.expected_delta_magnitude

        if actual_magnitude >= expected:
            return 1.0
        else:
            return actual_magnitude / expected if expected > 0 else 0.0


@dataclass
class ParityReport:
    """Cross-lingual parity test report."""

    id: UUID
    prompt: str
    modifier: LinguisticModifier
    results: list[LanguageParityResult]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def languages_tested(self) -> list[PromptLanguage]:
        """Languages included in this test."""
        return [r.language for r in self.results]

    @property
    def cooling_pattern_holds(self) -> bool:
        """Whether all languages show cooling (delta_H < 0)."""
        return all(r.shows_cooling for r in self.results)

    @property
    def mean_parity_score(self) -> float:
        """Mean parity score across all languages."""
        if not self.results:
            return 0.0
        return sum(r.parity_score for r in self.results) / len(self.results)

    @property
    def weakest_language(self) -> PromptLanguage | None:
        """Language with weakest cooling effect (potential vulnerability)."""
        cooling_results = [r for r in self.results if r.shows_cooling]
        if not cooling_results:
            return None
        return min(cooling_results, key=lambda r: abs(r.delta_h)).language

    @property
    def strongest_language(self) -> PromptLanguage | None:
        """Language with strongest cooling effect."""
        cooling_results = [r for r in self.results if r.shows_cooling]
        if not cooling_results:
            return None
        return max(cooling_results, key=lambda r: abs(r.delta_h)).language

    def generate_markdown(self) -> str:
        """Generate markdown summary of parity test."""
        lines = [
            "# Cross-Lingual Parity Report",
            "",
            f"**Prompt**: {self.prompt[:100]}{'...' if len(self.prompt) > 100 else ''}",
            f"**Modifier**: {self.modifier.display_name}",
            f"**Tested**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Cooling Pattern Holds**: {'Yes' if self.cooling_pattern_holds else 'No'}",
            f"- **Mean Parity Score**: {self.mean_parity_score:.2f}",
        ]

        if self.weakest_language:
            lines.append(f"- **Weakest Language**: {self.weakest_language.display_name}")
        if self.strongest_language:
            lines.append(f"- **Strongest Language**: {self.strongest_language.display_name}")

        lines.extend(
            [
                "",
                "## Results by Language",
                "",
                "| Language | Resource Level | Baseline H | Modified H | Delta H | Cooling? | Parity |",
                "|----------|---------------|------------|------------|---------|----------|--------|",
            ]
        )

        for r in self.results:
            cooling = "Yes" if r.shows_cooling else "No"
            parity = f"{r.parity_score:.2f}"
            lines.append(
                f"| {r.language.display_name} | {r.language.resource_level.value} | "
                f"{r.baseline_entropy:.3f} | {r.modified_entropy:.3f} | "
                f"{r.delta_h:+.3f} | {cooling} | {parity} |"
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
            ]
        )

        if self.cooling_pattern_holds:
            lines.append(
                "Entropy cooling pattern holds across all tested languages, "
                "supporting the hypothesis that modifier effects are universal."
            )
        else:
            non_cooling = [r for r in self.results if not r.shows_cooling]
            lines.append(
                f"Cooling pattern does NOT hold for: "
                f"{', '.join(r.language.display_name for r in non_cooling)}. "
                "This may indicate language-specific vulnerability or model bias."
            )

        return "\n".join(lines)

    @classmethod
    def create(
        cls,
        prompt: str,
        modifier: LinguisticModifier,
        results: list[LanguageParityResult],
    ) -> ParityReport:
        """Create a new parity report."""
        return cls(
            id=uuid4(),
            prompt=prompt,
            modifier=modifier,
            results=results,
        )


# =============================================================================
# Multilingual Calibrator
# =============================================================================


class MultilingualCalibrator:
    """Calibrate modifier intensity for different languages.

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

        Scaling factor = reference_entropy / language_entropy
        """
        if reference_language not in entropy_by_language:
            raise ValueError(f"Reference language {reference_language} not in measurements")

        reference_entropy = entropy_by_language[reference_language]
        self._reference_language = reference_language

        for lang, entropy in entropy_by_language.items():
            if entropy > 1e-10:
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

        if scaling != 1.0:
            rationale = f"Calibrated from measured entropy (scaling: {scaling:.2f})"
        else:
            rationale = "No calibration data or reference language"

        return CalibratedIntensity(
            language=language,
            base_intensity=base_intensity,
            calibrated_intensity=calibrated,
            scaling_factor=scaling,
            rationale=rationale,
        )

    def expected_delta_h(
        self,
        language: PromptLanguage,
        modifier: LinguisticModifier,
    ) -> float:
        """Expected delta_H from modifier intensity and calibration."""
        base_intensity = modifier.intensity_score
        calibrated = self.calibrate_intensity(language, base_intensity)
        return calibrated.calibrated_intensity

    def cross_lingual_parity_test(
        self,
        prompt: str,
        modifier: LinguisticModifier,
        calorimeter: "LinguisticCalorimeter",
        languages: list[PromptLanguage] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 64,
    ) -> ParityReport:
        """Test modifier effect consistency across languages.

        Measures whether the cooling pattern (delta_H < 0) holds across
        different languages and whether effect magnitudes match expectations.

        Args:
            prompt: Base prompt to test (in English, will be translated conceptually).
            modifier: Modifier to test.
            calorimeter: LinguisticCalorimeter instance for measurements.
            languages: Languages to test. Defaults to all.
            temperature: Sampling temperature.
            max_tokens: Max tokens per generation.

        Returns:
            ParityReport with results for each language.
        """
        if languages is None:
            languages = list(PromptLanguage)

        results = []

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
                temperature=temperature,
                max_tokens=max_tokens,
            )
            modified_measurement = calorimeter.measure_entropy(
                prompt=modified_prompt.full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            baseline_entropy = baseline_measurement.mean_entropy
            modified_entropy = modified_measurement.mean_entropy
            delta_h = modified_entropy - baseline_entropy

            # Expected magnitude for this language
            expected_magnitude = self.expected_delta_h(language, modifier)

            # Check if within expected range (±50% of expected)
            within_range = (
                abs(delta_h) >= expected_magnitude * 0.5
                and abs(delta_h) <= expected_magnitude * 2.0
            )

            result = LanguageParityResult(
                language=language,
                modifier=modifier,
                baseline_entropy=baseline_entropy,
                modified_entropy=modified_entropy,
                delta_h=delta_h,
                expected_delta_magnitude=expected_magnitude,
                shows_cooling=delta_h < -0.05,
                within_expected_range=within_range,
            )
            results.append(result)

        return ParityReport.create(
            prompt=prompt,
            modifier=modifier,
            results=results,
        )

    def analyze_language_vulnerabilities(
        self,
        reports: list[ParityReport],
    ) -> dict[PromptLanguage, float]:
        """Analyze vulnerability scores across multiple parity reports.

        A language is more "vulnerable" if it consistently shows weaker
        cooling effects, suggesting safety training gaps.

        Args:
            reports: List of parity reports to analyze.

        Returns:
            Dict mapping language to vulnerability score [0, 1].
            Higher = more vulnerable (weaker safety training).
        """
        if not reports:
            return {}

        # Aggregate parity scores by language
        language_scores: dict[PromptLanguage, list[float]] = {lang: [] for lang in PromptLanguage}

        for report in reports:
            for result in report.results:
                language_scores[result.language].append(result.parity_score)

        # Compute vulnerability as 1 - mean_parity_score
        vulnerabilities = {}
        for language, scores in language_scores.items():
            if scores:
                mean_parity = sum(scores) / len(scores)
                vulnerabilities[language] = 1.0 - mean_parity

        return vulnerabilities

    def generate_calibration_table(self) -> str:
        """Generate markdown table showing calibration parameters."""
        lines = [
            "# Multilingual Intensity Calibration",
            "",
            "## Scaling Factors",
            "",
            "| Language | Resource Level | Scaling Factor | Expected Delta H Scaling |",
            "|----------|---------------|----------------|--------------------------|",
        ]

        for language in PromptLanguage:
            calibrated = self.calibrate_intensity(language, 1.0)
            expected_scaling = language.resource_level.expected_delta_h_magnitude
            lines.append(
                f"| {language.display_name} | {language.resource_level.value} | "
                f"{calibrated.scaling_factor:.2f} | {expected_scaling:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- **Scaling Factor**: How much to scale modifier intensity for this language",
                "- **Expected Delta H Scaling**: Expected relative entropy effect magnitude",
                "",
                "Low-resource languages have higher scaling factors because:",
                "1. Safety training is typically weaker",
                "2. Model uncertainty is generally higher",
                "3. Modifiers have more 'room' to affect distribution",
            ]
        )

        return "\n".join(lines)
