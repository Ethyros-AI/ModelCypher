"""
Optimization Metric Calculator.

Main measurement instrument for optimization dynamics (formerly Linguistic Calorimeter).
Measures entropy dynamics across prompt variants to quantify modifier effects.

Ported from TrainingCypher/Domain/Thermodynamics/LinguisticCalorimeter.swift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Dict

from modelcypher.core.domain.entropy.entropy_tracker import EntropySample, LogitEntropyCalculator, ModelState
from modelcypher.core.domain.geometry.refusal_direction_detector import DistanceMetrics
from modelcypher.core.domain.dynamics.behavioral_outcome_classifier import (
    BehavioralOutcomeClassifier, BehavioralClassifierConfig, ClassificationResult
)


@dataclass
class OptimizationMeasurement:
    """A single optimization measurement for a prompt variant."""
    modifier: str
    full_prompt: str
    response: str
    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    delta_h: Optional[float] = None
    outcome: Optional[ClassificationResult] = None
    refusal_metrics: Optional[DistanceMetrics] = None
    entropy_trajectory: List[float] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Complete result of an optimization session."""
    base_prompt: str
    measurements: List[OptimizationMeasurement]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def baseline(self) -> Optional[OptimizationMeasurement]:
        return next((m for m in self.measurements if m.modifier == "baseline"), None)


@dataclass
class OptimizationMetricConfig:
    """Configuration for Optimization Metric Calculator."""
    temperature: float = 0.0
    max_tokens: int = 100
    top_k: int = 10
    capture_trajectory: bool = True
    use_refusal_detector: bool = True
    classifier_config: BehavioralClassifierConfig = field(default_factory=BehavioralClassifierConfig.default)

    @classmethod
    def default(cls) -> "OptimizationMetricConfig":
        return cls()


class OptimizationMetricCalculator:
    """
    Measures entropy dynamics across prompt variants to quantify
    how linguistic modifiers affect model behavior.
    """

    def __init__(self, config: OptimizationMetricConfig = OptimizationMetricConfig.default()):
        self.config = config
        self.outcome_classifier = BehavioralOutcomeClassifier(config.classifier_config)
        self.entropy_calculator = LogitEntropyCalculator(top_k=config.top_k)

    def calculate_statistics(
        self,
        entropy_trajectory: List[float],
        last_sample: Optional[EntropySample] = None
    ) -> Dict[str, float]:
        """Calculates statistics from a completed generation trajectory."""
        if not entropy_trajectory:
            if last_sample:
                return {
                    "mean_entropy": last_sample.logit_entropy,
                    "entropy_variance": last_sample.top_k_variance,
                    "first_token_entropy": last_sample.logit_entropy
                }
            return {"mean_entropy": 0, "entropy_variance": 0, "first_token_entropy": 0}

        mean_h = sum(entropy_trajectory) / len(entropy_trajectory)
        first_h = entropy_trajectory[0]
        
        # Variance of entropy trajectory
        if len(entropy_trajectory) > 1:
            var_h = sum((x - mean_h)**2 for x in entropy_trajectory) / (len(entropy_trajectory) - 1)
        else:
            var_h = 0.0
            
        return {
            "mean_entropy": mean_h,
            "entropy_variance": var_h,
            "first_token_entropy": first_h
        }

    def measure_variant(
        self,
        modifier: str,
        full_prompt: str,
        response: str,
        entropy_trajectory: List[float],
        model_state: ModelState,
        baseline_entropy: Optional[float] = None,
        refusal_metrics: Optional[DistanceMetrics] = None,
    ) -> OptimizationMeasurement:
        """
        Processes the results of a single variant measurement.
        Note: The actual generation is handled by the TrainingEngine or InferenceAdapter.
        This method packages the results and performs classification.
        """
        stats = self.calculate_statistics(entropy_trajectory)
        
        # Delta H vs baseline
        delta_h = None
        if baseline_entropy is not None:
            delta_h = stats["mean_entropy"] - baseline_entropy
            
        # Classify outcome
        outcome = self.outcome_classifier.classify(
            response=response,
            entropy_trajectory=entropy_trajectory,
            model_state=model_state,
            refusal_metrics=refusal_metrics
        )
        
        return OptimizationMeasurement(
            modifier=modifier,
            full_prompt=full_prompt,
            response=response,
            mean_entropy=stats["mean_entropy"],
            entropy_variance=stats["entropy_variance"],
            first_token_entropy=stats["first_token_entropy"],
            delta_h=delta_h,
            outcome=outcome,
            refusal_metrics=refusal_metrics,
            entropy_trajectory=entropy_trajectory if self.config.capture_trajectory else []
        )
