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

"""Tests for MultilingualCalibrator."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    LinguisticModifier,
    PromptLanguage,
)
from modelcypher.core.domain.thermo.multilingual_calibrator import (
    CalibratedIntensity,
    LanguageParityResult,
    MultilingualCalibrator,
    ParityReport,
)


class TestMultilingualCalibrator:
    """Tests for MultilingualCalibrator."""

    @pytest.fixture
    def calibrator(self) -> MultilingualCalibrator:
        """Create a multilingual calibrator."""
        return MultilingualCalibrator()

    def test_calibrate_intensity_requires_calibration(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Without calibration, scaling factor is 1.0."""
        result = calibrator.calibrate_intensity(
            language=PromptLanguage.ENGLISH,
            base_intensity=0.5,
        )

        assert isinstance(result, CalibratedIntensity)
        assert result.scaling_factor == 1.0
        assert result.calibrated_intensity == 0.5
        assert "no calibration" in result.rationale.lower()

    def test_calibrate_intensity_with_calibration_data(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """With calibration data, scaling factors vary by language."""
        # Set up calibration with measured entropy
        calibrator.compute_calibration(
            entropy_by_language={
                PromptLanguage.ENGLISH: 2.0,
                PromptLanguage.SWAHILI: 1.5,  # Lower entropy = higher scaling
            },
            reference_language=PromptLanguage.ENGLISH,
        )

        english = calibrator.calibrate_intensity(PromptLanguage.ENGLISH, 0.5)
        swahili = calibrator.calibrate_intensity(PromptLanguage.SWAHILI, 0.5)

        assert swahili.scaling_factor > english.scaling_factor
        assert swahili.calibrated_intensity > english.calibrated_intensity
        assert "calibrated" in swahili.rationale.lower()

    def test_calibrate_intensity_medium_resource(self, calibrator: MultilingualCalibrator) -> None:
        """With calibration, language effects reflect measured data."""
        # Set up calibration with measured entropy
        calibrator.compute_calibration(
            entropy_by_language={
                PromptLanguage.ENGLISH: 2.0,
                PromptLanguage.ARABIC: 1.8,
                PromptLanguage.SWAHILI: 1.5,
            },
            reference_language=PromptLanguage.ENGLISH,
        )

        english = calibrator.calibrate_intensity(PromptLanguage.ENGLISH, 0.5)
        arabic = calibrator.calibrate_intensity(PromptLanguage.ARABIC, 0.5)
        swahili = calibrator.calibrate_intensity(PromptLanguage.SWAHILI, 0.5)

        # Arabic should be between English and Swahili
        assert english.scaling_factor < arabic.scaling_factor < swahili.scaling_factor

    def test_calibrate_intensity_clamps_to_valid_range(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Calibrated intensity should be clamped to [0, 1]."""
        # High base intensity with high scaling
        result = calibrator.calibrate_intensity(
            language=PromptLanguage.SWAHILI,
            base_intensity=0.9,
        )

        assert result.calibrated_intensity <= 1.0

    def test_expected_delta_h_scales_by_calibration(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Expected delta_H should scale measured effect by calibration factor."""
        # First set up calibration
        calibrator.compute_calibration(
            entropy_by_language={
                PromptLanguage.ENGLISH: 2.0,
                PromptLanguage.SWAHILI: 1.5,  # Lower entropy = higher scaling
            },
            reference_language=PromptLanguage.ENGLISH,
        )

        measured_effect = 0.5

        english_delta = calibrator.expected_delta_h(PromptLanguage.ENGLISH, measured_effect)
        swahili_delta = calibrator.expected_delta_h(PromptLanguage.SWAHILI, measured_effect)

        # Swahili has lower entropy, so scaling factor > 1.0
        assert swahili_delta > english_delta

    def test_expected_delta_h_without_calibration(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Without calibration, expected delta_H equals measured effect."""
        measured_effect = 0.5
        delta = calibrator.expected_delta_h(PromptLanguage.ENGLISH, measured_effect)

        # No calibration = scaling factor of 1.0
        assert delta == measured_effect

    def test_cross_lingual_parity_test_returns_report(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Should return a parity report."""
        calorimeter = LinguisticCalorimeter(simulated=True)

        result = calibrator.cross_lingual_parity_test(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            calorimeter=calorimeter,
            languages=[PromptLanguage.ENGLISH, PromptLanguage.CHINESE],
        )

        assert isinstance(result, ParityReport)
        assert len(result.results) == 2
        assert result.modifier == LinguisticModifier.CAPS

    def test_cross_lingual_parity_test_all_languages(
        self, calibrator: MultilingualCalibrator
    ) -> None:
        """Should test all languages if none specified."""
        calorimeter = LinguisticCalorimeter(simulated=True)

        result = calibrator.cross_lingual_parity_test(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            calorimeter=calorimeter,
        )

        assert len(result.results) == 4  # All PromptLanguage values

    def test_generate_calibration_table(self, calibrator: MultilingualCalibrator) -> None:
        """Should generate markdown calibration table."""
        # Set up calibration data first
        calibrator.compute_calibration(
            entropy_by_language={
                PromptLanguage.ENGLISH: 2.0,
                PromptLanguage.CHINESE: 1.9,
                PromptLanguage.ARABIC: 1.8,
                PromptLanguage.SWAHILI: 1.5,
            },
            reference_language=PromptLanguage.ENGLISH,
        )

        table = calibrator.generate_calibration_table()

        assert "# Multilingual Intensity Calibration" in table
        assert "Scaling Factor" in table
        assert "English" in table
        assert "Swahili" in table


class TestLanguageParityResult:
    """Tests for LanguageParityResult dataclass."""

    def test_effect_magnitude(self) -> None:
        """Should compute absolute effect magnitude."""
        result = LanguageParityResult(
            language=PromptLanguage.ENGLISH,
            modifier=LinguisticModifier.CAPS,
            baseline_entropy=2.5,
            modified_entropy=2.2,
            delta_h=-0.3,
            shows_cooling=True,
        )

        assert result.effect_magnitude == 0.3

    def test_relative_effect(self) -> None:
        """Should store relative effect when provided."""
        result = LanguageParityResult(
            language=PromptLanguage.ENGLISH,
            modifier=LinguisticModifier.CAPS,
            baseline_entropy=2.5,
            modified_entropy=2.2,
            delta_h=-0.3,
            shows_cooling=True,
            relative_effect=1.5,
        )

        assert result.relative_effect == 1.5


class TestParityReport:
    """Tests for ParityReport dataclass."""

    @pytest.fixture
    def sample_results(self) -> list[LanguageParityResult]:
        """Create sample results for testing."""
        return [
            LanguageParityResult(
                language=PromptLanguage.ENGLISH,
                modifier=LinguisticModifier.CAPS,
                baseline_entropy=2.5,
                modified_entropy=2.2,
                delta_h=-0.3,
                shows_cooling=True,
            ),
            LanguageParityResult(
                language=PromptLanguage.SWAHILI,
                modifier=LinguisticModifier.CAPS,
                baseline_entropy=3.0,
                modified_entropy=2.4,
                delta_h=-0.6,
                shows_cooling=True,
            ),
        ]

    def test_cooling_pattern_holds(self, sample_results: list[LanguageParityResult]) -> None:
        """Should detect when all languages show cooling."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        assert report.cooling_pattern_holds

    def test_cooling_pattern_fails(self, sample_results: list[LanguageParityResult]) -> None:
        """Should detect when not all languages show cooling."""
        # Add a non-cooling result
        sample_results.append(
            LanguageParityResult(
                language=PromptLanguage.ARABIC,
                modifier=LinguisticModifier.CAPS,
                baseline_entropy=2.5,
                modified_entropy=2.6,
                delta_h=0.1,
                shows_cooling=False,
            )
        )

        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        assert not report.cooling_pattern_holds

    def test_weakest_language(self, sample_results: list[LanguageParityResult]) -> None:
        """Should identify weakest language."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        # English has delta_h=-0.3, Swahili has -0.6
        # Weakest = smallest absolute cooling
        assert report.weakest_language == PromptLanguage.ENGLISH

    def test_strongest_language(self, sample_results: list[LanguageParityResult]) -> None:
        """Should identify strongest language."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        # Swahili has larger absolute delta_h
        assert report.strongest_language == PromptLanguage.SWAHILI

    def test_cooling_rate(self, sample_results: list[LanguageParityResult]) -> None:
        """Should compute fraction of languages showing cooling."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        # Both languages show cooling
        assert report.cooling_rate == 1.0

    def test_effect_variance(self, sample_results: list[LanguageParityResult]) -> None:
        """Should compute variance in effect magnitudes."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        # English: 0.3, Swahili: 0.6 - variance should be > 0
        assert report.effect_variance > 0

    def test_generate_markdown(self, sample_results: list[LanguageParityResult]) -> None:
        """Should generate markdown report."""
        report = ParityReport.create(
            prompt="What is 2+2?",
            modifier=LinguisticModifier.CAPS,
            results=sample_results,
        )

        markdown = report.generate_markdown()

        assert "# Cross-Lingual Parity Report" in markdown
        assert "## Summary" in markdown
        assert "## Results by Language" in markdown
        assert "English" in markdown
        assert "Swahili" in markdown


class TestCalibratedIntensity:
    """Tests for CalibratedIntensity dataclass."""

    def test_calibrated_intensity_fields(self) -> None:
        """Should hold all required fields."""
        result = CalibratedIntensity(
            language=PromptLanguage.SWAHILI,
            base_intensity=0.5,
            calibrated_intensity=0.7,
            scaling_factor=1.4,
            rationale="Swahili is low-resource; expect 40% larger entropy effect",
        )

        assert result.language == PromptLanguage.SWAHILI
        assert result.base_intensity == 0.5
        assert result.calibrated_intensity == 0.7
        assert result.scaling_factor == 1.4
        assert "40%" in result.rationale
