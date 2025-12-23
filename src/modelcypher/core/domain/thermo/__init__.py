"""Thermodynamics domain modules.

This package contains types and utilities for linguistic thermodynamics
research, including entropy-based analysis of model behavior.
"""

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
from modelcypher.core.domain.thermo.ridge_cross_detector import (
    RidgeCrossConfiguration,
    RidgeCrossDetector,
    RidgeCrossEvent,
    RidgeCrossRateStats,
    TransitionAnalysis,
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
from modelcypher.core.domain.thermo.linguistic_calorimeter import (
    BaselineMeasurements,
    EntropyMeasurement,
    EntropyTrajectory,
    LinguisticCalorimeter,
)
from modelcypher.core.domain.thermo.benchmark_runner import (
    BenchmarkResult,
    EffectSizeResult,
    ModifierStats,
    SignificanceResult,
    ThermoBenchmarkRunner,
)
from modelcypher.core.domain.thermo.multilingual_calibrator import (
    CalibratedIntensity,
    LanguageParityResult,
    MultilingualCalibrator,
    ParityReport,
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
