# Training Dynamics Package
from .metrics import OptimizationMetricCalculator, OptimizationState
from .regimes import RegimeStateDetector, OptimizationRegime, RegimeTransitionEvent
from .monitoring import DivergenceInterventionMonitor
from .phase_transition import (
    Phase,
    BasinTopology,
    PhaseAnalysis,
    TemperatureSweepResult,
    estimate_critical_temperature,
    entropy_derivative,
    compute_logit_variance,
    effective_vocabulary_size,
    compute_entropy,
    classify_phase,
    temperature_sweep,
    analyze,
    predict_modifier_effect,
    theoretical_tc,
)

