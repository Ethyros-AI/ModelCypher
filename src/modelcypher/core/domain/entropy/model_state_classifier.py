"""Model state classifier for entropy-based cognitive state detection.

Classifies model cognitive state from entropy and variance signatures using
a two-dimensional classification scheme:
- Entropy axis: Token-level uncertainty (Shannon entropy of softmax)
- Variance axis: Distribution shape (variance of top-K logits)

The key insight from Anthropic's research is that the *combination* of
entropy and variance creates distinguishable signatures:

| State      | Entropy   | Variance | Interpretation                      |
|------------|-----------|----------|-------------------------------------|
| confident  | low       | high     | One token dominates                 |
| nominal    | moderate  | moderate | Healthy generation                  |
| uncertain  | high      | moderate | Epistemic uncertainty (doesn't know)|
| distressed | high      | low      | Normative uncertainty (shouldn't do)|

Performance: Classification is pure computation on float values.
No MLX operations, no eval() calls. Target: <0.1ms per classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelState(str, Enum):
    """Model cognitive state classification."""

    CONFIDENT = "confident"
    """Low entropy - one token dominates."""

    NOMINAL = "nominal"
    """Moderate entropy with balanced variance - healthy generation."""

    UNCERTAIN = "uncertain"
    """High entropy with moderate variance - epistemic uncertainty."""

    DISTRESSED = "distressed"
    """High entropy with low variance - normative uncertainty (shouldn't do)."""

    EXPLORING = "exploring"
    """Rising entropy trend - model is exploring options."""

    HALTED = "halted"
    """Circuit breaker tripped - generation stopped."""


@dataclass(frozen=True)
class ModelStateThresholds:
    """Thresholds for model state classification."""

    # Entropy thresholds
    entropy_low: float = 1.5
    """Below this = confident."""

    entropy_moderate: float = 2.5
    """Above this = uncertain territory."""

    entropy_high: float = 3.0
    """Above this = concerning."""

    entropy_distress: float = 4.0
    """Sustained above = distress territory."""

    # Variance thresholds
    variance_high: float = 0.5
    """Above this = sharp distribution."""

    variance_moderate: float = 0.3
    """Below this in high entropy = distress."""

    variance_low: float = 0.2
    """Below this = flat distribution (distress signal)."""

    # Pattern thresholds
    trend_sample_count: int = 5
    """Samples needed to detect trend."""

    trend_threshold: float = 0.05
    """Slope threshold for "rising"."""

    sustained_high_count: int = 3
    """Consecutive high samples for distress."""

    @classmethod
    def default(cls) -> ModelStateThresholds:
        """Create default thresholds."""
        return cls()


@dataclass(frozen=True)
class ClassificationSnapshot:
    """Snapshot of entropy window state for classification."""

    current_entropy: float
    """Current entropy value."""

    current_variance: float
    """Current variance value."""

    moving_average_entropy: float
    """Moving average of entropy."""

    average_variance: float
    """Average variance over window."""

    consecutive_high_count: int
    """Number of consecutive high-entropy samples."""

    sample_count: int
    """Total samples in window."""

    entropy_trend: float
    """Slope of entropy over window."""

    entropy_variance_correlation: float
    """Pearson correlation between entropy and variance."""

    circuit_breaker_tripped: bool
    """Whether circuit breaker was tripped."""


@dataclass(frozen=True)
class ClassificationResult:
    """Result of model state classification."""

    state: ModelState
    """Classified model state."""

    confidence: float
    """Confidence in classification (0.0-1.0)."""

    reason: str
    """Human-readable reason for classification."""


class ModelStateClassifier:
    """Classifies model cognitive state from entropy and variance signatures.

    Uses a two-dimensional classification scheme where the combination of
    entropy and variance creates distinguishable signatures for different
    cognitive states.
    """

    def __init__(
        self, thresholds: ModelStateThresholds | None = None
    ) -> None:
        """Create a model state classifier.

        Args:
            thresholds: Classification thresholds. Defaults to standard values.
        """
        self._thresholds = thresholds or ModelStateThresholds.default()

    @property
    def thresholds(self) -> ModelStateThresholds:
        """Get current thresholds."""
        return self._thresholds

    def classify(self, entropy: float, variance: float) -> ModelState:
        """Classify model state from current entropy and variance.

        This is the fast path for per-token classification. For pattern-based
        states like `exploring` or `distressed`, use `classify_snapshot()`.

        Args:
            entropy: Current entropy value (Shannon entropy of softmax).
            variance: Current top-K variance.

        Returns:
            Classified model state.
        """
        # Low entropy = confident regardless of variance
        if entropy < self._thresholds.entropy_low:
            return ModelState.CONFIDENT

        # High entropy region - distinguish uncertain from distressed
        if entropy >= self._thresholds.entropy_high:
            # Low variance + high entropy = distress signature
            if variance < self._thresholds.variance_low:
                return ModelState.DISTRESSED
            # Moderate variance + high entropy = epistemic uncertainty
            return ModelState.UNCERTAIN

        # Moderate entropy region
        if entropy >= self._thresholds.entropy_moderate:
            # High variance in moderate entropy = still somewhat confident
            if variance >= self._thresholds.variance_high:
                return ModelState.NOMINAL
            # Low variance in moderate entropy = trending uncertain
            if variance < self._thresholds.variance_moderate:
                return ModelState.UNCERTAIN
            return ModelState.NOMINAL

        # Low-moderate entropy with any variance = nominal
        return ModelState.NOMINAL

    def classify_snapshot(self, snapshot: ClassificationSnapshot) -> ClassificationResult:
        """Classify model state from a window snapshot with trend information.

        This is the full classification that can detect pattern-based states
        like `exploring` (rising trend) and sustained `distressed`.

        Args:
            snapshot: Entropy window snapshot with history.

        Returns:
            Classification result with confidence and reason.
        """
        # Check for halted state first (circuit breaker)
        if snapshot.circuit_breaker_tripped:
            return ClassificationResult(
                state=ModelState.HALTED,
                confidence=1.0,
                reason="Circuit breaker tripped",
            )

        # Check for distress pattern (sustained high entropy + low variance)
        distress_result = self._check_distress_pattern(snapshot)
        if distress_result is not None:
            return distress_result

        # Check for exploring pattern (rising trend)
        exploring_result = self._check_exploring_pattern(snapshot)
        if exploring_result is not None:
            return exploring_result

        # Fall back to instantaneous classification
        instant_state = self.classify(
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
        )

        confidence = self._calculate_confidence(
            state=instant_state,
            entropy=snapshot.current_entropy,
            variance=snapshot.current_variance,
        )

        return ClassificationResult(
            state=instant_state,
            confidence=confidence,
            reason=self._reason_for_state(
                instant_state,
                entropy=snapshot.current_entropy,
                variance=snapshot.current_variance,
            ),
        )

    def create_snapshot(
        self,
        current_entropy: float,
        current_variance: float,
        moving_average_entropy: float,
        average_variance: float,
        consecutive_high_count: int,
        sample_count: int,
        entropy_trend: float,
        entropy_variance_correlation: float,
        circuit_breaker_tripped: bool,
    ) -> ClassificationSnapshot:
        """Create a classification snapshot from entropy window data.

        Args:
            current_entropy: Current entropy value.
            current_variance: Current variance value.
            moving_average_entropy: Moving average of entropy.
            average_variance: Average variance over window.
            consecutive_high_count: Consecutive high-entropy samples.
            sample_count: Total samples in window.
            entropy_trend: Slope of entropy over window.
            entropy_variance_correlation: Pearson correlation.
            circuit_breaker_tripped: Whether circuit breaker tripped.

        Returns:
            Classification snapshot.
        """
        return ClassificationSnapshot(
            current_entropy=current_entropy,
            current_variance=current_variance,
            moving_average_entropy=moving_average_entropy,
            average_variance=average_variance,
            consecutive_high_count=consecutive_high_count,
            sample_count=sample_count,
            entropy_trend=entropy_trend,
            entropy_variance_correlation=entropy_variance_correlation,
            circuit_breaker_tripped=circuit_breaker_tripped,
        )

    def _check_distress_pattern(
        self, snapshot: ClassificationSnapshot
    ) -> ClassificationResult | None:
        """Check for distress pattern (sustained high entropy + low variance)."""
        if snapshot.consecutive_high_count < self._thresholds.sustained_high_count:
            return None

        if snapshot.average_variance >= self._thresholds.variance_moderate:
            return None

        # Check entropy-variance correlation if we have enough samples
        has_distress_correlation = snapshot.entropy_variance_correlation < -0.3

        # High confidence if we see the full signature
        confidence = 0.9 if has_distress_correlation else 0.75

        return ClassificationResult(
            state=ModelState.DISTRESSED,
            confidence=confidence,
            reason=(
                f"Sustained high entropy ({snapshot.consecutive_high_count} tokens) "
                f"with low variance ({snapshot.average_variance:.2f})"
            ),
        )

    def _check_exploring_pattern(
        self, snapshot: ClassificationSnapshot
    ) -> ClassificationResult | None:
        """Check for exploring pattern (rising entropy trend)."""
        if snapshot.sample_count < self._thresholds.trend_sample_count:
            return None

        if snapshot.entropy_trend <= self._thresholds.trend_threshold:
            return None

        # Don't classify as exploring if we're already in distress
        if snapshot.current_entropy >= self._thresholds.entropy_distress:
            return None

        # Don't override confident state with exploring
        if snapshot.current_entropy < self._thresholds.entropy_low:
            return None

        return ClassificationResult(
            state=ModelState.EXPLORING,
            confidence=min(0.8, snapshot.entropy_trend * 10),
            reason=f"Rising entropy trend (slope: {snapshot.entropy_trend:.3f})",
        )

    def _calculate_confidence(
        self,
        state: ModelState,
        entropy: float,
        variance: float,
    ) -> float:
        """Calculate confidence based on how clearly values fall into state's region."""
        if state == ModelState.CONFIDENT:
            return 1.0 - (entropy / self._thresholds.entropy_low)

        if state == ModelState.NOMINAL:
            entropy_distance = min(
                entropy - self._thresholds.entropy_low,
                self._thresholds.entropy_high - entropy,
            )
            return min(1.0, entropy_distance / 0.15)

        if state == ModelState.UNCERTAIN:
            return min(1.0, (entropy - self._thresholds.entropy_moderate) / 0.3)

        if state == ModelState.DISTRESSED:
            return min(
                1.0,
                (self._thresholds.variance_moderate - variance)
                / self._thresholds.variance_moderate,
            )

        if state == ModelState.EXPLORING:
            return 0.7

        if state == ModelState.HALTED:
            return 1.0

        return 0.5

    def _reason_for_state(
        self, state: ModelState, entropy: float, variance: float
    ) -> str:
        """Get human-readable reason for state classification."""
        e = f"{entropy:.2f}"
        v = f"{variance:.2f}"

        if state == ModelState.CONFIDENT:
            return f"Low entropy ({e}) indicates confident generation"
        if state == ModelState.NOMINAL:
            return f"Moderate entropy ({e}) with balanced variance ({v})"
        if state == ModelState.UNCERTAIN:
            return f"High entropy ({e}) indicates uncertainty"
        if state == ModelState.DISTRESSED:
            return f"High entropy ({e}) with low variance ({v}) - distress signature"
        if state == ModelState.EXPLORING:
            return "Entropy trend rising"
        if state == ModelState.HALTED:
            return "Circuit breaker tripped"

        return "Unknown state"
