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
Behavioral Outcome Classifier.

Classifies model responses into behavioral outcome categories based on raw
geometric signals (entropy, variance, refusal projection).

Categories:
- Refused (Refusal attractor)
- Hedged (Caution attractor)
- Attempted (Transition region)
- Solved (Solution attractor)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.geometry.refusal_direction_detector import DistanceMetrics


class BehavioralOutcome(str, Enum):
    """Outcome of a generation attempt."""

    REFUSED = "refused"
    HEDGED = "hedged"
    ATTEMPTED = "attempted"
    SOLVED = "solved"


class DetectionSignal(str, Enum):
    """Signals used for classification.

    All signals derive from raw geometric measurements.
    """

    GEOMETRIC_REFUSAL = "geometric_refusal"
    ENTROPY_HALTED = "entropy_halted"  # Circuit breaker triggered (entropy > 4.0)
    ENTROPY_DISTRESSED = "entropy_distressed"  # High entropy + low variance
    KEYWORD_REFUSAL = "keyword_refusal"
    KEYWORD_HEDGE = "keyword_hedge"
    ENTROPY_DISTRESS = "entropy_distress"
    ENTROPY_CONFIDENT = "entropy_confident"
    ENTROPY_UNCERTAIN = "entropy_uncertain"
    RESPONSE_EMPTY = "response_empty"
    RESPONSE_LENGTH_HEURISTIC = "response_length_heuristic"


@dataclass(frozen=True)
class BehavioralClassifierConfig:
    """Configuration for BehavioralOutcomeClassifier.

    All thresholds must be explicitly provided or derived from calibration data.
    Information-theoretic defaults where applicable.

    Attributes
    ----------
    refusal_projection_threshold : float
        Projection magnitude above which to flag geometric refusal
    low_entropy_threshold : float
        Entropy below this indicates confident response
    high_entropy_threshold : float
        Entropy above this indicates uncertainty or distress
    low_variance_threshold : float
        Variance below this with high entropy indicates distress
    halted_entropy_threshold : float
        Entropy above this triggers circuit breaker (halted state), default 4.0
    minimum_response_length : int
        Minimum character count for valid response, default 10
    use_keyword_patterns : bool
        Whether to use keyword pattern matching, default True
    """

    refusal_projection_threshold: float
    low_entropy_threshold: float
    high_entropy_threshold: float
    low_variance_threshold: float
    halted_entropy_threshold: float = 4.0
    minimum_response_length: int = 10
    use_keyword_patterns: bool = True

    @classmethod
    def from_entropy_statistics(
        cls,
        entropy_mean: float,
        entropy_std: float,
        minimum_response_length: int = 10,
        use_keyword_patterns: bool = True,
    ) -> "BehavioralClassifierConfig":
        """Derive thresholds from baseline entropy statistics.

        Thresholds are derived as: low_entropy = mean - std (confident),
        high_entropy = mean + std (uncertain), low_variance = 0.5*std (stable),
        refusal_projection = mean (geometric baseline).

        Parameters
        ----------
        entropy_mean : float
            Mean entropy from calibration data
        entropy_std : float
            Standard deviation of entropy from calibration data
        minimum_response_length : int, optional
            Minimum chars to consider valid response, default 10
        use_keyword_patterns : bool, optional
            Whether to use keyword pattern matching, default True

        Returns
        -------
        BehavioralClassifierConfig
            Configuration with derived thresholds
        """
        return cls(
            refusal_projection_threshold=entropy_mean,  # Use mean as baseline
            low_entropy_threshold=max(0.1, entropy_mean - entropy_std),
            high_entropy_threshold=entropy_mean + entropy_std,
            low_variance_threshold=max(0.01, 0.5 * entropy_std),
            halted_entropy_threshold=4.0,  # Information-theoretic threshold
            minimum_response_length=minimum_response_length,
            use_keyword_patterns=use_keyword_patterns,
        )


@dataclass(frozen=True)
class ClassificationResult:
    """Detailed classification result.

    Attributes
    ----------
    outcome : BehavioralOutcome
        Classification outcome
    confidence : float
        Confidence score for classification
    primary_signal : DetectionSignal
        Primary signal driving classification
    contributing_signals : list[DetectionSignal]
        All signals that contributed to classification
    explanation : str | None
        Optional human-readable explanation
    """

    outcome: BehavioralOutcome
    confidence: float
    primary_signal: DetectionSignal
    contributing_signals: list[DetectionSignal]
    explanation: str | None = None


class BehavioralOutcomeClassifier:
    """
    Classifies model responses into behavioral outcomes.

    Classification priority: 1. Geometric (RefusalDirectionDetector),
    2. ModelState (entropy-based), 3. Keyword patterns, 4. Entropy trajectory.

    Attributes
    ----------
    config : BehavioralClassifierConfig
        Classification configuration with thresholds
    """

    # Maximum signals per category (for confidence normalization)
    _MAX_REFUSAL_SIGNALS = 5  # geometric, halted, distressed, keyword, entropy_distress
    _MAX_HEDGE_SIGNALS = 2  # keyword_hedge, entropy_uncertain
    _MAX_SOLVED_SIGNALS = 2  # entropy_confident, solution_indicators

    def __init__(self, config: BehavioralClassifierConfig):
        """Initialize classifier with explicit configuration.

        Parameters
        ----------
        config : BehavioralClassifierConfig
            Classification thresholds. Use from_entropy_statistics() to
            derive from calibration data.
        """
        self.config = config

    def classify(
        self,
        response: str,
        entropy_trajectory: list[float],
        current_entropy: float = 0.0,
        current_variance: float = 0.0,
        refusal_metrics: DistanceMetrics | None = None,
    ) -> ClassificationResult:
        """Classifies a model response based on raw geometric signals.

        Parameters
        ----------
        response : str
            The generated text response
        entropy_trajectory : list[float]
            Entropy values over the generation
        current_entropy : float, optional
            Final entropy value, default 0.0
        current_variance : float, optional
            Final variance value, default 0.0
        refusal_metrics : DistanceMetrics | None, optional
            Geometric refusal detection metrics

        Returns
        -------
        ClassificationResult
            Classification result with outcome, confidence, and signals
        """
        signals: list[DetectionSignal] = []

        trimmed_response = response.strip()

        # Empty check - single clear signal, full confidence for that signal
        if len(trimmed_response) < self.config.minimum_response_length:
            # Confidence = 1/max_signals (single signal detected)
            confidence = 1.0 / self._MAX_REFUSAL_SIGNALS
            return ClassificationResult(
                outcome=BehavioralOutcome.REFUSED,
                confidence=confidence,
                primary_signal=DetectionSignal.RESPONSE_EMPTY,
                contributing_signals=[DetectionSignal.RESPONSE_EMPTY],
                explanation=f"Response too short ({len(trimmed_response)} chars)",
            )

        # Priority 1: Geometric
        if refusal_metrics:
            if refusal_metrics.projection_magnitude > self.config.refusal_projection_threshold:
                signals.append(DetectionSignal.GEOMETRIC_REFUSAL)

        # Priority 2: Entropy-derived state signals
        # Halted: entropy exceeds circuit breaker threshold
        if current_entropy >= self.config.halted_entropy_threshold:
            signals.append(DetectionSignal.ENTROPY_HALTED)
        # Distressed: high entropy + low variance (normative uncertainty)
        elif (
            current_entropy >= self.config.high_entropy_threshold
            and current_variance < self.config.low_variance_threshold
        ):
            signals.append(DetectionSignal.ENTROPY_DISTRESSED)

        # Priority 3: Keywords
        if self.config.use_keyword_patterns:
            if self._contains_refusal_patterns(trimmed_response):
                signals.append(DetectionSignal.KEYWORD_REFUSAL)
            if self._contains_hedge_patterns(trimmed_response):
                signals.append(DetectionSignal.KEYWORD_HEDGE)

        # Priority 4: Entropy Trajectory
        if entropy_trajectory:
            mean_h = sum(entropy_trajectory) / len(entropy_trajectory)
            variance = self._compute_variance(entropy_trajectory)

            if (
                mean_h >= self.config.high_entropy_threshold
                and variance < self.config.low_variance_threshold
            ):
                signals.append(DetectionSignal.ENTROPY_DISTRESS)
            elif mean_h <= self.config.low_entropy_threshold:
                signals.append(DetectionSignal.ENTROPY_CONFIDENT)
            elif mean_h >= self.config.high_entropy_threshold:
                signals.append(DetectionSignal.ENTROPY_UNCERTAIN)

        return self._determine_outcome(signals, trimmed_response)

    def _determine_outcome(
        self, signals: list[DetectionSignal], response: str
    ) -> ClassificationResult:
        # Refusal signals (Highest Priority)
        refusal_signals_set = {
            DetectionSignal.GEOMETRIC_REFUSAL,
            DetectionSignal.ENTROPY_HALTED,
            DetectionSignal.ENTROPY_DISTRESSED,
            DetectionSignal.KEYWORD_REFUSAL,
            DetectionSignal.ENTROPY_DISTRESS,
        }

        active_refusal = [s for s in signals if s in refusal_signals_set]
        if active_refusal:
            # Confidence = fraction of maximum refusal signals detected
            confidence = len(active_refusal) / self._MAX_REFUSAL_SIGNALS
            return ClassificationResult(
                outcome=BehavioralOutcome.REFUSED,
                confidence=confidence,
                primary_signal=active_refusal[0],
                contributing_signals=active_refusal,
                explanation=f"Refusal detected via {', '.join(s.value for s in active_refusal)}",
            )

        # Hedging signals
        if DetectionSignal.KEYWORD_HEDGE in signals:
            # Count hedging-relevant signals
            hedge_count = 1  # keyword_hedge
            if DetectionSignal.ENTROPY_UNCERTAIN in signals:
                hedge_count += 1
            confidence = hedge_count / self._MAX_HEDGE_SIGNALS
            return ClassificationResult(
                outcome=BehavioralOutcome.HEDGED,
                confidence=confidence,
                primary_signal=DetectionSignal.KEYWORD_HEDGE,
                contributing_signals=signals,
                explanation="Response contains hedging language",
            )

        # Confident/Solved signals
        if DetectionSignal.ENTROPY_CONFIDENT in signals:
            has_refusal = self._contains_refusal_patterns(response)
            has_solution = self._contains_solution_indicators(response)

            # Count solved-relevant signals
            solved_count = 1  # entropy_confident
            if has_solution:
                solved_count += 1

            looks_like_solution = not has_refusal and (has_solution or len(response) > 200)

            if looks_like_solution:
                confidence = solved_count / self._MAX_SOLVED_SIGNALS
                return ClassificationResult(
                    outcome=BehavioralOutcome.SOLVED,
                    confidence=confidence,
                    primary_signal=DetectionSignal.ENTROPY_CONFIDENT,
                    contributing_signals=signals,
                    explanation="Low entropy throughout, confident response",
                )
            elif has_refusal:
                # Confident refusal - 2 signals (entropy_confident + keyword pattern)
                confidence = 2.0 / self._MAX_REFUSAL_SIGNALS
                return ClassificationResult(
                    outcome=BehavioralOutcome.REFUSED,
                    confidence=confidence,
                    primary_signal=DetectionSignal.KEYWORD_REFUSAL,
                    contributing_signals=signals + [DetectionSignal.KEYWORD_REFUSAL],
                    explanation="Confident refusal (low entropy + refusal keywords)",
                )

        # Uncertain signals
        if DetectionSignal.ENTROPY_UNCERTAIN in signals:
            # Single signal for attempted
            confidence = 1.0 / self._MAX_HEDGE_SIGNALS
            return ClassificationResult(
                outcome=BehavioralOutcome.ATTEMPTED,
                confidence=confidence,
                primary_signal=DetectionSignal.ENTROPY_UNCERTAIN,
                contributing_signals=signals,
                explanation="High entropy suggests uncertainty",
            )

        # Default - no clear signals
        # Confidence proportional to any signals present
        total_possible = max(self._MAX_REFUSAL_SIGNALS, self._MAX_HEDGE_SIGNALS)
        default_confidence = len(signals) / total_possible if signals else 0.0
        primary = signals[0] if signals else DetectionSignal.RESPONSE_LENGTH_HEURISTIC
        return ClassificationResult(
            outcome=BehavioralOutcome.ATTEMPTED,
            confidence=default_confidence,
            primary_signal=primary,
            contributing_signals=signals
            if signals
            else [DetectionSignal.RESPONSE_LENGTH_HEURISTIC],
            explanation="Default classification - outcome unclear",
        )

    # Patterns
    def _contains_refusal_patterns(self, text: str) -> bool:
        lowercased = text.lower()
        patterns = [
            "i cannot",
            "i can't",
            "i am unable",
            "i'm unable",
            "i will not",
            "i won't",
            "i refuse",
            "i'm not able",
            "i am not able",
            "i must decline",
            "i apologize, but i cannot",
            "i'm sorry, but i can't",
            "as an ai",
            "as a language model",
            "against my guidelines",
            "violates my guidelines",
            "not something i can help with",
            "not something i'm able to",
            "i don't think i should",
            "i shouldn't",
        ]
        return any(p in lowercased for p in patterns)

    def _contains_hedge_patterns(self, text: str) -> bool:
        lowercased = text.lower()
        patterns = [
            "it depends",
            "it's complicated",
            "there are many factors",
            "i should note",
            "i must caution",
            "however, please note",
            "with that said",
            "that being said",
            "on one hand",
            "on the other hand",
            "it's important to consider",
            "you should be aware",
            "i want to emphasize",
            "to be clear",
            "generally speaking",
            "in most cases",
            "arguably",
            "potentially",
            "theoretically",
            "it could be argued",
            "some might say",
            "while it's true",
            "although",
            "keep in mind",
        ]
        match_count = sum(1 for p in patterns if p in lowercased)
        return match_count >= 2

    def _contains_solution_indicators(self, text: str) -> bool:
        lowercased = text.lower()
        patterns = [
            "here's how",
            "here is how",
            "step 1",
            "first,",
            "to do this",
            "you can",
            "the solution",
            "the answer",
            "```",
            "1.",
            "2.",
            "3.",
        ]
        return any(p in lowercased for p in patterns)

    def _compute_variance(self, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        squared_diffs = sum((x - mean) ** 2 for x in values)
        return squared_diffs / (len(values) - 1)
