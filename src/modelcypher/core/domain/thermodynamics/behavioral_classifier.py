"""Compatibility re-export for behavioral classifier."""
from modelcypher.core.domain.dynamics.behavioral_outcome_classifier import (
    BehavioralOutcomeClassifier,
    BehavioralOutcome,
    BehavioralClassifierConfig,
    ClassificationResult,
    DetectionSignal,
)

__all__ = [
    "BehavioralOutcomeClassifier",
    "BehavioralOutcome",
    "BehavioralClassifierConfig",
    "ClassificationResult",
    "DetectionSignal",
]
