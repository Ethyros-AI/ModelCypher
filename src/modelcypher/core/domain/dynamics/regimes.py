"""
Optimization Regime State Detection.

Integrates with phase_transition.py for physics-based regime classification.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import mlx.core as mx

from .phase_transition import (
    Phase,
    classify_phase,
    estimate_critical_temperature,
    compute_logit_statistics,
    effective_vocabulary_size,
    predict_modifier_effect,
)


class OptimizationRegime(str, Enum):
    """Training regime classifications."""
    OVERFITTING = "overfitting"  # Low entropy/variance
    STABLE = "stable"            # Healthy learning
    VOLATILE = "volatile"        # High variance, exploring
    DIVERGENT = "divergent"      # Exploding gradients/entropy


@dataclass
class RegimeTransitionEvent:
    """Records a regime transition."""
    from_regime: OptimizationRegime
    to_regime: OptimizationRegime
    step: int
    phase: Optional[Phase] = None
    estimated_tc: Optional[float] = None


class RegimeStateDetector:
    """
    Detects the current optimization regime based on metrics.

    Integrates physics-based phase transition theory for enhanced detection.
    """

    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.state_history: List[OptimizationRegime] = []
        self.current_regime: OptimizationRegime = OptimizationRegime.STABLE
        self.current_phase: Optional[Phase] = None
        self.estimated_tc: Optional[float] = None

    def assess_regime(
        self,
        perplexity: float,
        entropy_delta: float,
        logits: Optional[mx.array] = None,
        temperature: float = 1.0,
    ) -> OptimizationRegime:
        """
        Determines the current optimization regime.

        Args:
            perplexity: Current perplexity value
            entropy_delta: Change in entropy
            logits: Optional raw logits for phase-based classification
            temperature: Current temperature setting

        Returns:
            OptimizationRegime classification
        """
        # Phase-based classification (if logits available)
        if logits is not None:
            _, _, std_dev = compute_logit_statistics(logits)
            v_eff = effective_vocabulary_size(logits, 1.0)
            tc = estimate_critical_temperature(std_dev, v_eff)
            phase = classify_phase(temperature, tc)

            self.current_phase = phase
            self.estimated_tc = tc

            # Map Phase to OptimizationRegime
            if phase == Phase.ORDERED:
                if perplexity < 0.8:
                    return OptimizationRegime.OVERFITTING
                else:
                    return OptimizationRegime.STABLE
            elif phase == Phase.CRITICAL:
                return OptimizationRegime.VOLATILE
            else:  # DISORDERED
                if perplexity > 100.0:
                    return OptimizationRegime.DIVERGENT
                else:
                    return OptimizationRegime.VOLATILE

        # Fallback: threshold-based classification
        if perplexity > 100.0 or entropy_delta > 5.0:
            return OptimizationRegime.DIVERGENT
        elif perplexity > 10.0:
            return OptimizationRegime.VOLATILE
        elif perplexity < 0.8:
            return OptimizationRegime.OVERFITTING
        else:
            return OptimizationRegime.STABLE

    def update(
        self,
        perplexity: float,
        entropy_delta: Optional[float] = None,
        step: int = 0,
        logits: Optional[mx.array] = None,
        temperature: float = 1.0,
    ) -> Optional[RegimeTransitionEvent]:
        """
        Update regime state and detect transitions.

        Args:
            perplexity: Current perplexity
            entropy_delta: Entropy change (optional)
            step: Current training step
            logits: Optional logits for phase analysis
            temperature: Current temperature

        Returns:
            RegimeTransitionEvent if a transition occurred
        """
        delta = entropy_delta if entropy_delta is not None else 0.0

        new_regime = self.assess_regime(
            perplexity=perplexity,
            entropy_delta=delta,
            logits=logits,
            temperature=temperature,
        )

        event = None
        if new_regime != self.current_regime:
            event = RegimeTransitionEvent(
                from_regime=self.current_regime,
                to_regime=new_regime,
                step=step,
                phase=self.current_phase,
                estimated_tc=self.estimated_tc,
            )
            self.current_regime = new_regime

        self.state_history.append(new_regime)
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)

        return event

    def get_modifier_prediction(
        self,
        intensity_score: float,
        base_entropy: float,
    ) -> tuple[float, float]:
        """
        Predict the effect of a modifier based on current phase.

        Args:
            intensity_score: Modifier intensity [0, 1]
            base_entropy: Current baseline entropy

        Returns:
            Tuple of (predicted_delta_h, confidence)
        """
        if self.current_phase is None:
            return (0.0, 0.0)

        return predict_modifier_effect(
            self.current_phase,
            intensity_score,
            base_entropy,
        )

    def get_regime_recommendation(self) -> str:
        """Get actionable recommendation for current regime."""
        if self.current_regime == OptimizationRegime.DIVERGENT:
            return "Training is diverging. Reduce learning rate or gradient clipping."
        elif self.current_regime == OptimizationRegime.OVERFITTING:
            return "Model may be overfitting. Consider early stopping or regularization."
        elif self.current_regime == OptimizationRegime.VOLATILE:
            return "Training is volatile. Monitor closely and consider warmup."
        else:
            return "Training is stable. No intervention needed."

