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
Logit Entropy Calculator.

Computes Shannon entropy over the **full vocabulary** (not top-K) as a measure of
logit dispersion. This is the core entropy computation for the entire framework.

Notes
-----
Key Properties:
- Full vocabulary entropy: Range [0, ln(vocab_size)] ≈ [0, 10.5] for 32K vocab
- NOT top-K entropy which would be [0, ln(K)] ≈ [0, 2.3] for K=10
- Threshold policies are external; this module only reports raw measurements.

Limitation:
Entropy alone cannot distinguish adapter specialization from fighting the prior.
Use ConflictScoreCalculator for dual-model safety checks.

References
----------
Correlates with semantic entropy (R^2 ~0.6 per arXiv:2406.15927)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array, Backend

# =============================================================================
# Logit Entropy Calculator
# =============================================================================


class LogitEntropyCalculator:
    """Computes Shannon entropy from model logits.

    Given logits from a model's forward pass, computes:
    1. Shannon entropy: -sum(p * log(p)) over the simplex-normalized logits
    2. Logit variance: Variance of raw logit values (before softmax)

    Notes
    -----
    We compute entropy over the FULL vocabulary, not just top-K tokens, because:
    - It captures the full dispersion of the logit landscape
    - It correlates with semantic entropy (arXiv:2406.15927)
    - High dispersion correlates with geometric trajectory instability

    Examples
    --------
    Basic usage:

        calculator = LogitEntropyCalculator()
        entropy, variance = calculator.compute(logits)
    """

    def __init__(
        self,
        *,
        epsilon: float | None = None,
        backend: Backend | None = None,
    ) -> None:
        """
        Initialize the calculator.

        Args:
            epsilon: Optional log-stability floor. If None, derived from dtype.
            backend: Compute backend.
        """
        self.epsilon = epsilon
        self._backend = backend or get_default_backend()

    def compute(
        self,
        logits: Array,
        skip_variance: bool = False,
    ) -> tuple[float, float]:
        """Compute Shannon entropy and variance from logits.

        Parameters
        ----------
        logits : Array
            Array of shape [..., vocab_size] from model forward pass.
        skip_variance : bool, optional
            If True, skips variance computation (returns 0 for variance).

        Returns
        -------
        tuple of float
            Tuple of (entropy, variance) as float values.

        Notes
        -----
        Algorithm:
        1. Extract the last token's logits (handles various input shapes)
        2. Apply numerically stable softmax
        3. Compute Shannon entropy: -sum(w * log(w)) over simplex weights
        4. Compute variance of raw logit values
        """
        # Flatten logits to 1D vocabulary vector
        flat_logits = self._flatten_to_vocab(logits)
        # Numerically stable softmax
        max_val = self._backend.max(flat_logits, keepdims=True)
        shifted = flat_logits - max_val
        exp_shifted = self._backend.exp(shifted)
        sum_exp = self._backend.sum(exp_shifted, keepdims=True)
        weights = exp_shifted / sum_exp

        # Shannon entropy: -sum(w * log(w))
        from modelcypher.core.domain.geometry.numerical_stability import safe_log_epsilon

        eps = self.epsilon if self.epsilon is not None else safe_log_epsilon(self._backend, weights)
        log_weights = self._backend.log(weights + eps)
        entropy = -self._backend.sum(weights * log_weights)

        # Logit variance (before softmax, as proxy for "sharpness")
        if skip_variance:
            variance = self._backend.array([0.0])
        else:
            mean_val = self._backend.mean(flat_logits)
            squared_diff = (flat_logits - mean_val) ** 2
            variance = self._backend.mean(squared_diff)

        # Evaluate and convert to Python floats
        self._backend.eval(entropy, variance)

        return float(self._backend.to_scalar(entropy)), float(self._backend.to_scalar(variance))

    def compute_batch(
        self,
        logits_batch: list[Array],
    ) -> list[tuple[float, float]]:
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
            weights = exp_shifted / sum_exp

            # Entropy
            from modelcypher.core.domain.geometry.numerical_stability import safe_log_epsilon

            eps = self.epsilon if self.epsilon is not None else safe_log_epsilon(self._backend, weights)
            log_weights = self._backend.log(weights + eps)
            entropy = -self._backend.sum(weights * log_weights)

            # Variance across full logits
            mean_val = self._backend.mean(flat_logits)
            squared_diff = (flat_logits - mean_val) ** 2
            variance = self._backend.mean(squared_diff)

            entropies.append(entropy)
            variances.append(variance)

        # Batch evaluate
        self._backend.eval(*entropies, *variances)

        return [
            (
                float(self._backend.to_scalar(e)),
                float(self._backend.to_scalar(v)),
            )
            for e, v in zip(entropies, variances)
        ]

    @staticmethod
    def normalize_entropy(
        raw_entropy: float,
        vocab_size: int,
    ) -> float:
        """Normalize raw entropy to [0, 1] range.

        Raw Shannon entropy ranges from 0 (fully concentrated on one token) to
        ln(vocab_size) (uniform dispersion). This method normalizes to [0, 1].

        Parameters
        ----------
        raw_entropy : float
            Raw Shannon entropy value.
        vocab_size : int, optional
            Vocabulary size for max entropy calculation.

        Returns
        -------
        float
            Normalized entropy in [0, 1] where:
            - 0 = fully confident (entropy = 0)
            - 1 = maximum dispersion (uniform across simplex)

        Examples
        --------
        Usage with circuit breaker:

            calc = LogitEntropyCalculator()
            raw_entropy, _ = calc.compute(logits)
            normalized = calc.normalize_entropy(raw_entropy, vocab_size=32000)
            # Pass normalized to circuit breaker
        """
        from modelcypher.core.domain.geometry.numerical_stability import log_scalar

        if vocab_size <= 1:
            return 0.0
        _b = get_default_backend()
        max_entropy = log_scalar(float(vocab_size), _b)
        if max_entropy <= 0:
            return 0.0
        return min(max(raw_entropy / max_entropy, 0.0), 1.0)

    def compute_with_normalization(
        self,
        logits: Array,
        vocab_size: int | None = None,
    ) -> tuple[float, float, float]:
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
    """An entropy measurement from a single token or window.

    Raw measurements only. Caller applies thresholds for classification.

    Attributes
    ----------
    window_id : str
        Unique window identifier.
    token_start : int
        Starting token index.
    token_end : int
        Ending token index.
    logit_entropy : float
        Shannon entropy over full vocabulary.
    logit_variance : float
        Variance of raw logits (full vocabulary).
    latency_ms : float, optional
        Computation latency in milliseconds.
    source : str, optional
        Source identifier.
    correlation_id : str, optional
        Correlation ID for tracking.
    """

    window_id: str
    token_start: int
    token_end: int
    logit_entropy: float
    logit_variance: float
    latency_ms: float | None = None
    source: str | None = None
    correlation_id: str | None = None

    @classmethod
    def from_computation(
        cls,
        entropy: float,
        variance: float,
        token_start: int,
        token_end: int,
        latency_ms: float | None = None,
        source: str | None = None,
        correlation_id: str | None = None,
    ) -> "LogitEntropySample":
        """Create a sample from computed entropy values."""
        return cls(
            window_id=str(uuid.uuid4()),
            token_start=token_start,
            token_end=token_end,
            logit_entropy=entropy,
            logit_variance=variance,
            latency_ms=latency_ms,
            source=source,
            correlation_id=correlation_id,
        )
