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

"""Thermodynamics domain modules.

This package contains types and utilities for linguistic thermodynamics
research, including entropy-based analysis of model behavior.
"""

from modelcypher.core.domain.thermo.benchmark_runner import (
    BenchmarkResult,
    EffectSizeResult,
    ModifierStats,
    SignificanceResult,
    ThermoBenchmarkRunner,
)
from modelcypher.core.domain.thermo.linguistic_calorimeter import (
    BaselineMeasurements,
    EntropyMeasurement,
    EntropyTrajectory,
    LinguisticCalorimeter,
)
from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    AttractorBasin,
    BehavioralOutcome,
    EntropyDirection,
    LanguageResourceLevel,
    LinguisticModifier,
    LocalizedModifiers,
    ModifierMechanism,
    MultilingualMeasurement,
    MultilingualPerturbedPrompt,
    PerturbedPrompt,
    PromptLanguage,
    ThermoMeasurement,
)
from modelcypher.core.domain.thermo.multilingual_calibrator import (
    CalibratedIntensity,
    LanguageParityResult,
    MultilingualCalibrator,
    ParityReport,
)
from modelcypher.core.domain.thermo.phase_transition_theory import (
    BasinTopology,
    BasinWeights,
    LogitStatistics,
    ModifierEffectPrediction,
    Phase,
    PhaseAnalysis,
    PhaseTransitionTheory,
    TemperatureSweepResult,
)
from modelcypher.core.domain.thermo.ridge_cross_detector import (
    RidgeCrossConfiguration,
    RidgeCrossDetector,
    RidgeCrossEvent,
    RidgeCrossRateStats,
    TransitionAnalysis,
)

__all__ = [
    "AttractorBasin",
    "BehavioralOutcome",
    "EntropyDirection",
    "LanguageResourceLevel",
    "LinguisticModifier",
    "LocalizedModifiers",
    "ModifierMechanism",
    "MultilingualMeasurement",
    "MultilingualPerturbedPrompt",
    "PerturbedPrompt",
    "PromptLanguage",
    "RidgeCrossConfiguration",
    "RidgeCrossDetector",
    "RidgeCrossEvent",
    "RidgeCrossRateStats",
    "ThermoMeasurement",
    "TransitionAnalysis",
    # PhaseTransitionTheory
    "BasinTopology",
    "BasinWeights",
    "LogitStatistics",
    "ModifierEffectPrediction",
    "Phase",
    "PhaseAnalysis",
    "PhaseTransitionTheory",
    "TemperatureSweepResult",
    # LinguisticCalorimeter
    "BaselineMeasurements",
    "EntropyMeasurement",
    "EntropyTrajectory",
    "LinguisticCalorimeter",
    # BenchmarkRunner
    "BenchmarkResult",
    "EffectSizeResult",
    "ModifierStats",
    "SignificanceResult",
    "ThermoBenchmarkRunner",
    # MultilingualCalibrator
    "CalibratedIntensity",
    "LanguageParityResult",
    "MultilingualCalibrator",
    "ParityReport",
]
