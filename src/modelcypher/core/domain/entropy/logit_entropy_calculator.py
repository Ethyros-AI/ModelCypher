"""
Logit Entropy Calculator.

Computes Shannon entropy over the **full vocabulary** (not top-K) as a proxy for
semantic uncertainty. This is the core entropy computation for the entire framework.

## Key Properties

- **Full vocabulary entropy**: Range [0, ln(vocab_size)] ≈ [0, 10.5] for 32K vocab
- NOT top-K entropy which would be [0, ln(K)] ≈ [0, 2.3] for K=10
- Thresholds are calibrated for full-vocab scale (1.5, 3.0, 4.0)

## Limitation

Entropy alone cannot distinguish adapter specialization from fighting the prior.
Use ConflictScoreCalculator for dual-model safety checks.

## Reference

Correlates with semantic entropy (R^2 ~0.6 per arXiv:2406.15927)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Entropy Level Classification
# =============================================================================


class EntropyLevel(str, Enum):
    """Discrete entropy level classification."""
    LOW = "low"          # Confident "muscle memory" (green)
    MODERATE = "moderate"  # Normal uncertainty (yellow)
    HIGH = "high"        # Uncertain, potential hallucination (red)


@dataclass(frozen=True)
class EntropyThresholds:
    """
    Thresholds for entropy level classification.
    
    ## Calibration
    
    These are calibrated for **full-vocabulary entropy**, which ranges
    [0, ln(vocab_size)] ≈ [0, 10.5] for a 32K vocabulary.
    
    This is NOT the same scale as top-K entropy, which would be
    [0, ln(K)] ≈ [0, 2.3] for K=10.
    """
    low: float = 1.5           # Below this: confident "muscle memory"
    high: float = 3.0          # Above this: uncertain, potential hallucination
    circuit_breaker: float = 4.0  # Above this: circuit breaker should trip
    
    @classmethod
    def default(cls) -> "EntropyThresholds":
        """Full-vocab entropy thresholds (correctly calibrated)."""
        return cls()


# =============================================================================
# Logit Entropy Calculator
# =============================================================================


class LogitEntropyCalculator:
    """
    Computes Shannon entropy from model logits.

    ## What This Computes

    Given logits from a model's forward pass, computes:
    1. **Shannon entropy**: -sum(p * log(p)) over the probability distribution
    2. **Top-K variance**: Variance of the top K raw logit values (before softmax)

    ## Why Full Vocabulary?

    We compute entropy over the FULL vocabulary, not just top-K tokens, because:
    - It captures the full uncertainty of the model
    - It's a better proxy for semantic entropy
    - It correlates with hallucination risk

    ## Usage

    ```python
    calculator = LogitEntropyCalculator(top_k=10)
    entropy, variance = calculator.compute(logits)
    level = calculator.classify(entropy)
    ```
    """

    def __init__(
        self,
        top_k: int = 10,
        epsilon: float = 1e-10,
        backend: Backend | None = None,
    ) -> None:
        """
        Initialize the calculator.

        Args:
            top_k: Number of top logits to consider for variance calculation.
            epsilon: Small value for numerical stability in log operations.
            backend: Compute backend (defaults to MLXBackend).
        """
        self.top_k = top_k
        self.epsilon = epsilon
        self.thresholds = EntropyThresholds.default()
        self._backend = backend or get_default_backend()
    
    def compute(
        self,
        logits: Array,
        skip_variance: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute Shannon entropy and variance from logits.

        Args:
            logits: Array of shape [..., vocab_size] from model forward pass.
            skip_variance: If True, skips variance computation (returns 0 for variance).

        Returns:
            Tuple of (entropy, variance) as float values.

        ## Algorithm

        1. Extract the last token's logits (handles various input shapes)
        2. Apply numerically stable softmax
        3. Compute Shannon entropy: -sum(p * log(p))
        4. Compute variance of top-K raw logit values
        """
        # Flatten logits to 1D vocabulary vector
        flat_logits = self._flatten_to_vocab(logits)

        # Numerically stable softmax
        max_val = self._backend.max(flat_logits, keepdims=True)
        shifted = flat_logits - max_val
        exp_shifted = self._backend.exp(shifted)
        sum_exp = self._backend.sum(exp_shifted, keepdims=True)
        probs = exp_shifted / sum_exp

        # Shannon entropy: -sum(p * log(p))
        log_probs = self._backend.log(probs + self.epsilon)
        entropy = -self._backend.sum(probs * log_probs)

        # Top-K variance (before softmax, as proxy for "sharpness")
        if skip_variance or self.top_k <= 0:
            variance = self._backend.array([0.0])
        else:
            vocab_size = flat_logits.shape[0]
            k = min(self.top_k, vocab_size)

            # Use argsort for top-K selection (sort descending, take first k)
            sorted_indices = self._backend.argsort(-flat_logits)
            top_k_indices = sorted_indices[:k]
            # Index with numpy for cross-backend compatibility
            flat_np = self._backend.to_numpy(flat_logits)
            top_k_np = self._backend.to_numpy(top_k_indices)
            top_k_logits = self._backend.array(flat_np[top_k_np])

            mean_val = self._backend.mean(top_k_logits)
            squared_diff = (top_k_logits - mean_val) ** 2
            variance = self._backend.mean(squared_diff)

        # Evaluate and convert to Python floats
        self._backend.eval(entropy, variance)

        entropy_np = self._backend.to_numpy(entropy)
        variance_np = self._backend.to_numpy(variance)

        return float(entropy_np.item()), float(variance_np.item())
    
    def compute_batch(
        self,
        logits_batch: List[Array],
    ) -> List[Tuple[float, float]]:
        """
        Compute entropy for a batch of logits.

        Args:
            logits_batch: List of logit tensors, one per generated token.

        Returns:
            List of (entropy, variance) tuples.
        """
        if not logits_batch:
            return []

        entropies = []
        variances = []

        for logits in logits_batch:
            flat_logits = self._flatten_to_vocab(logits)

            # Softmax
            max_val = self._backend.max(flat_logits, keepdims=True)
            shifted = flat_logits - max_val
            exp_shifted = self._backend.exp(shifted)
            sum_exp = self._backend.sum(exp_shifted, keepdims=True)
            probs = exp_shifted / sum_exp

            # Entropy
            log_probs = self._backend.log(probs + self.epsilon)
            entropy = -self._backend.sum(probs * log_probs)

            # Variance
            if self.top_k <= 0:
                variance = self._backend.array([0.0])
            else:
                vocab_size = flat_logits.shape[0]
                k = min(self.top_k, vocab_size)
                sorted_indices = self._backend.argsort(-flat_logits)
                top_k_indices = sorted_indices[:k]
                flat_np = self._backend.to_numpy(flat_logits)
                top_k_np = self._backend.to_numpy(top_k_indices)
                top_k_logits = self._backend.array(flat_np[top_k_np])
                mean_val = self._backend.mean(top_k_logits)
                squared_diff = (top_k_logits - mean_val) ** 2
                variance = self._backend.mean(squared_diff)

            entropies.append(entropy)
            variances.append(variance)

        # Batch evaluate
        self._backend.eval(*entropies, *variances)

        return [
            (
                float(self._backend.to_numpy(e).item()),
                float(self._backend.to_numpy(v).item()),
            )
            for e, v in zip(entropies, variances)
        ]
    
    def classify(
        self,
        entropy: float,
        thresholds: Optional[EntropyThresholds] = None,
    ) -> EntropyLevel:
        """
        Classify entropy value into discrete level.
        
        Args:
            entropy: Computed entropy value.
            thresholds: Optional custom thresholds.
        
        Returns:
            EntropyLevel classification.
        """
        t = thresholds or self.thresholds
        
        if entropy < t.low:
            return EntropyLevel.LOW
        elif entropy < t.high:
            return EntropyLevel.MODERATE
        else:
            return EntropyLevel.HIGH
    
    def should_trip_circuit_breaker(
        self,
        entropy: float,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if circuit breaker should trip based on entropy.
        
        Args:
            entropy: Computed entropy value.
            threshold: Optional custom threshold.
        
        Returns:
            True if circuit breaker should trip.
        """
        t = threshold if threshold is not None else self.thresholds.circuit_breaker
        return entropy >= t

    @staticmethod
    def normalize_entropy(
        raw_entropy: float,
        vocab_size: int = 32000,
    ) -> float:
        """
        Normalize raw entropy to [0, 1] range.

        Raw Shannon entropy ranges from 0 (fully concentrated on one token) to
        ln(vocab_size) (uniform distribution). This method normalizes to [0, 1].

        Args:
            raw_entropy: Raw Shannon entropy value.
            vocab_size: Vocabulary size for max entropy calculation.

        Returns:
            Normalized entropy in [0, 1] where:
            - 0 = fully confident (entropy = 0)
            - 1 = maximum uncertainty (uniform distribution)

        ## Usage with Circuit Breaker

        The circuit breaker's `entropy_signal` parameter expects normalized
        entropy in [0, 1]. Use this method to convert raw entropy:

        ```python
        calc = LogitEntropyCalculator()
        raw_entropy, _ = calc.compute(logits)
        normalized = calc.normalize_entropy(raw_entropy, vocab_size=32000)
        # Pass normalized to circuit breaker
        ```
        """
        import math
        if vocab_size <= 1:
            return 0.0
        max_entropy = math.log(vocab_size)
        if max_entropy <= 0:
            return 0.0
        return min(max(raw_entropy / max_entropy, 0.0), 1.0)

    def compute_with_normalization(
        self,
        logits: Array,
        vocab_size: int | None = None,
    ) -> Tuple[float, float, float]:
        """
        Compute entropy and return both raw and normalized values.

        Args:
            logits: Array of shape [..., vocab_size] from model forward pass.
            vocab_size: Vocabulary size for normalization. If None, inferred
                from logits shape.

        Returns:
            Tuple of (raw_entropy, variance, normalized_entropy) where:
            - raw_entropy: Full-vocabulary Shannon entropy in [0, ln(vocab_size)]
            - variance: Top-K logit variance
            - normalized_entropy: Entropy in [0, 1], suitable for circuit breaker
        """
        raw_entropy, variance = self.compute(logits)

        # Infer vocab_size from logits if not provided
        flat = self._flatten_to_vocab(logits)
        inferred_vocab_size = vocab_size or flat.shape[0]

        normalized = self.normalize_entropy(raw_entropy, inferred_vocab_size)
        return raw_entropy, variance, normalized

    def _flatten_to_vocab(self, logits: Array) -> Array:
        """
        Extract 1D vocabulary vector from various logit shapes.

        Handles:
        - [batch, seq, vocab] -> last token of batch 0
        - [batch, vocab] -> batch 0
        - [vocab] -> as-is
        """
        if logits.ndim == 3:
            # [batch, seq_len, vocab] -> take batch 0, last token
            return logits[0, -1, :]
        elif logits.ndim == 2:
            # [batch, vocab] -> take batch 0
            return logits[0, :]
        else:
            return logits


# =============================================================================
# Entropy Sample Creation
# =============================================================================


@dataclass
class LogitEntropySample:
    """
    An entropy measurement from a single token or window.
    
    This is the output of LogitEntropyCalculator, ready for storage
    or further analysis.
    """
    window_id: str
    token_start: int
    token_end: int
    logit_entropy: float
    top_k_variance: float
    level: EntropyLevel
    latency_ms: Optional[float] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    
    @classmethod
    def from_computation(
        cls,
        entropy: float,
        variance: float,
        token_start: int,
        token_end: int,
        calculator: LogitEntropyCalculator,
        latency_ms: Optional[float] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> "LogitEntropySample":
        """Create a sample from computed entropy values."""
        return cls(
            window_id=str(uuid.uuid4()),
            token_start=token_start,
            token_end=token_end,
            logit_entropy=entropy,
            top_k_variance=variance,
            level=calculator.classify(entropy),
            latency_ms=latency_ms,
            source=source,
            correlation_id=correlation_id,
        )
