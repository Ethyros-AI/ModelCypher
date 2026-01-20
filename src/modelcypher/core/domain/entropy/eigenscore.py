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

"""EigenScore: Geometric Uncertainty via Eigenvalue Spread.

Measures uncertainty through the eigenvalue distribution of hidden state covariance,
providing a geometric signal distinct from Shannon entropy. This implements concepts
from INSIDE (ICLR 2024) adapted for real-time monitoring.

Theory
------
Shannon entropy measures distribution spread over vocabulary (probabilistic).
EigenScore measures geometric spread of activations in representation space.

High EigenScore = activations spread across many dimensions = sparse manifold = UNCERTAIN
Low EigenScore = activations concentrated in few dimensions = dense manifold = CONFIDENT

The key insight: these metrics measure different aspects of uncertainty.
- Shannon entropy: "How spread is the probability mass over tokens?"
- EigenScore: "How spread is the activation geometry in representation space?"

A model can have low Shannon entropy (confident about next token) but high EigenScore
(in a sparse manifold region). This combination signals potential hallucination -
confidently wrong because the sparse region has no data to contradict.

Implementation
--------------
We provide three computation modes:

1. **Sequence mode**: Given hidden states from a sequence [seq, hidden], compute
   eigenvalue spread of the token-token covariance matrix. Measures how diverse
   the geometric positions are across the sequence.

2. **Layer mode**: Given hidden states from multiple layers [n_layers, hidden],
   compute eigenvalue spread of layer-layer covariance. Measures how differently
   the model encodes the same input across depth.

3. **Streaming mode**: Maintain a running covariance estimate across generations.
   Update incrementally with each new hidden state. Measures trajectory diversity
   over a generation session.

References
----------
Chen et al. (2024) "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"
    ICLR 2024 - EigenScore concept (eigenvalues of response covariance)

Fang et al. (2024) "Uncertainty Quantification in Language Models: A Geometric Perspective"
    Connects eigenvalue spread to manifold density
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy.eigenscore")


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class EigenScoreResult:
    """EigenScore uncertainty measurement.

    Attributes
    ----------
    eigenscore : float
        Primary uncertainty metric. Higher = more uncertain.
        Range: [0, 1] after normalization.
    effective_rank : float
        Number of dimensions carrying significant variance.
        Computed as exp(entropy of normalized eigenvalues).
    eigenvalue_entropy : float
        Shannon entropy over normalized eigenvalue distribution.
        Measures how evenly spread the variance is across dimensions.
    condition_number : float
        Ratio of largest to smallest eigenvalue.
        High condition number = ill-conditioned = potentially unstable.
    top_eigenvalue_ratio : float
        Fraction of variance explained by top eigenvalue.
        Low ratio = spread across many dimensions = uncertain.
    n_samples : int
        Number of samples used in covariance estimate.
    n_dimensions : int
        Dimensionality of hidden space.
    """

    eigenscore: float
    effective_rank: float
    eigenvalue_entropy: float
    condition_number: float
    top_eigenvalue_ratio: float
    n_samples: int
    n_dimensions: int


@dataclass
class StreamingEigenScore:
    """Streaming EigenScore estimator with running covariance.

    Maintains a running estimate of the covariance matrix using Welford's
    online algorithm, allowing incremental updates without storing all samples.

    Attributes
    ----------
    n_samples : int
        Number of samples seen so far.
    hidden_dim : int
        Dimensionality of hidden states.
    mean : Array
        Running mean of hidden states.
    m2 : Array
        Running sum of squared deviations (for covariance).
    """

    n_samples: int = 0
    hidden_dim: int = 0
    mean: "Array | None" = None
    m2: "Array | None" = None  # Running sum of outer products of deviations
    _backend: "Backend | None" = field(default=None, repr=False)

    def update(self, hidden_state: "Array") -> None:
        """Update running statistics with a new hidden state.

        Uses Welford's online algorithm for numerical stability.

        Parameters
        ----------
        hidden_state : Array
            New hidden state vector. Shape: [hidden_dim] or will be flattened.
        """
        b = self._backend or get_default_backend()

        # Flatten to 1D
        h = self._flatten_hidden(hidden_state, b)

        if self.n_samples == 0:
            # First sample
            self.hidden_dim = h.shape[0]
            self.mean = h
            self.m2 = b.zeros((self.hidden_dim, self.hidden_dim))
            self.n_samples = 1
            self._backend = b
            return

        # Welford's algorithm
        self.n_samples += 1
        delta = h - self.mean
        self.mean = self.mean + delta / self.n_samples

        # Update M2 for covariance: M2 += outer(delta, delta2) where delta2 = h - new_mean
        delta2 = h - self.mean
        # outer product: delta[:, None] @ delta2[None, :]
        outer = b.matmul(
            b.reshape(delta, (self.hidden_dim, 1)),
            b.reshape(delta2, (1, self.hidden_dim)),
        )
        self.m2 = self.m2 + outer

    def compute(self) -> EigenScoreResult:
        """Compute EigenScore from current running statistics.

        Returns
        -------
        EigenScoreResult
            Current uncertainty measurement based on accumulated samples.

        Raises
        ------
        ValueError
            If fewer than 2 samples have been accumulated.
        """
        if self.n_samples < 2:
            raise ValueError(
                f"Need at least 2 samples for covariance, have {self.n_samples}"
            )

        b = self._backend or get_default_backend()

        # Covariance = M2 / (n - 1)
        cov = self.m2 / (self.n_samples - 1)

        return _compute_eigenscore_from_covariance(cov, self.n_samples, b)

    def reset(self) -> None:
        """Reset streaming estimator to initial state."""
        self.n_samples = 0
        self.hidden_dim = 0
        self.mean = None
        self.m2 = None

    def _flatten_hidden(self, hidden_state: "Array", b: "Backend") -> "Array":
        """Flatten hidden state to 1D vector."""
        if hidden_state.ndim == 3:
            # [batch, seq, hidden] -> last token of batch 0
            return hidden_state[0, -1, :]
        elif hidden_state.ndim == 2:
            # [seq, hidden] -> last token
            return hidden_state[-1, :]
        return hidden_state


# =============================================================================
# EigenScore Calculator
# =============================================================================


class EigenScoreCalculator:
    """Computes EigenScore uncertainty from hidden state geometry.

    Provides multiple computation modes:
    - Sequence: Eigenvalue spread across tokens in a sequence
    - Layer: Eigenvalue spread across layers for same input
    - Covariance: Direct computation from pre-computed covariance matrix

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to MLXBackend.

    Examples
    --------
    Sequence mode (most common):

        calc = EigenScoreCalculator()
        # hidden_states shape: [seq_len, hidden_dim]
        result = calc.compute_from_sequence(hidden_states)
        print(f"EigenScore: {result.eigenscore:.3f}")
        print(f"Effective rank: {result.effective_rank:.1f}")

    Streaming mode for generation:

        calc = EigenScoreCalculator()
        streamer = calc.create_streamer()

        for token_hidden in generation_loop:
            streamer.update(token_hidden)
            if streamer.n_samples >= 5:
                result = streamer.compute()
                if result.eigenscore > 0.8:
                    print("Warning: high uncertainty region")
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_from_sequence(
        self,
        hidden_states: "Array",
        min_tokens: int = 3,
    ) -> EigenScoreResult:
        """Compute EigenScore from sequence of hidden states.

        Given hidden states from multiple token positions, computes the
        covariance matrix and extracts eigenvalue-based uncertainty metrics.

        Parameters
        ----------
        hidden_states : Array
            Hidden states from sequence. Shape: [seq_len, hidden_dim] or
            [batch, seq_len, hidden_dim] (takes batch 0).
        min_tokens : int, optional
            Minimum tokens required. Default 3.

        Returns
        -------
        EigenScoreResult
            Geometric uncertainty measurement.

        Raises
        ------
        ValueError
            If sequence too short.
        """
        b = self._backend

        # Handle batch dimension
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[0]  # Take batch 0

        seq_len, hidden_dim = hidden_states.shape

        if seq_len < min_tokens:
            raise ValueError(
                f"Sequence length {seq_len} < minimum {min_tokens}"
            )

        # Center the data
        mean = b.mean(hidden_states, axis=0, keepdims=True)
        centered = hidden_states - mean

        # Covariance matrix: (X^T X) / (n-1)
        # centered is [seq, hidden], so centered.T @ centered is [hidden, hidden]
        cov = b.matmul(b.transpose(centered), centered) / (seq_len - 1)

        return _compute_eigenscore_from_covariance(cov, seq_len, b)

    def compute_from_layers(
        self,
        layer_hidden_states: list["Array"],
    ) -> EigenScoreResult:
        """Compute EigenScore from hidden states across layers.

        Given the same input's hidden state at each layer, measures how
        differently the model encodes across depth.

        Parameters
        ----------
        layer_hidden_states : list[Array]
            Hidden state at each layer. Each array shape: [hidden_dim].

        Returns
        -------
        EigenScoreResult
            Cross-layer uncertainty measurement.

        Raises
        ------
        ValueError
            If fewer than 3 layers provided.
        """
        if len(layer_hidden_states) < 3:
            raise ValueError(
                f"Need at least 3 layers, got {len(layer_hidden_states)}"
            )

        b = self._backend

        # Stack into [n_layers, hidden_dim]
        # First flatten each to 1D
        flattened = []
        for h in layer_hidden_states:
            if h.ndim == 3:
                h = h[0, -1, :]
            elif h.ndim == 2:
                h = h[-1, :]
            flattened.append(h)

        stacked = b.stack(flattened, axis=0)
        return self.compute_from_sequence(stacked, min_tokens=3)

    def compute_from_covariance(
        self,
        covariance: "Array",
        n_samples: int,
    ) -> EigenScoreResult:
        """Compute EigenScore from pre-computed covariance matrix.

        Parameters
        ----------
        covariance : Array
            Covariance matrix. Shape: [hidden_dim, hidden_dim].
        n_samples : int
            Number of samples used to compute covariance.

        Returns
        -------
        EigenScoreResult
            Geometric uncertainty measurement.
        """
        return _compute_eigenscore_from_covariance(
            covariance, n_samples, self._backend
        )

    def create_streamer(self) -> StreamingEigenScore:
        """Create a streaming EigenScore estimator.

        Returns a stateful object that can be updated incrementally with
        new hidden states during generation.

        Returns
        -------
        StreamingEigenScore
            Streaming estimator with update() and compute() methods.
        """
        return StreamingEigenScore(_backend=self._backend)


# =============================================================================
# Internal Implementation
# =============================================================================


def _compute_eigenscore_from_covariance(
    covariance: "Array",
    n_samples: int,
    backend: "Backend",
) -> EigenScoreResult:
    """Compute EigenScore metrics from covariance matrix.

    Core computation shared by all EigenScore modes.

    Parameters
    ----------
    covariance : Array
        Covariance matrix [d, d].
    n_samples : int
        Number of samples used.
    backend : Backend
        Compute backend.

    Returns
    -------
    EigenScoreResult
        Full uncertainty measurement.
    """
    b = backend
    d = covariance.shape[0]

    # Compute eigenvalues (covariance is symmetric, use eigvalsh for speed)
    eigenvalues = b.eigvalsh(covariance)

    # Sort descending (eigvalsh returns ascending)
    # Use argsort and reverse
    n = eigenvalues.shape[0]
    reversed_idx = b.arange(n - 1, -1, -1)
    eigenvalues = b.take(eigenvalues, reversed_idx, axis=0)

    # Clamp to non-negative (numerical noise can produce small negatives)
    eigenvalues = b.where(
        eigenvalues > 0,
        eigenvalues,
        b.zeros_like(eigenvalues),
    )

    # Normalize eigenvalues to probability distribution
    total = b.sum(eigenvalues)
    eps = division_epsilon(b, eigenvalues)

    if float(b.to_scalar(total)) < float(b.to_scalar(eps)):
        # Degenerate case: all eigenvalues zero
        return EigenScoreResult(
            eigenscore=0.0,
            effective_rank=1.0,
            eigenvalue_entropy=0.0,
            condition_number=1.0,
            top_eigenvalue_ratio=1.0,
            n_samples=n_samples,
            n_dimensions=d,
        )

    p = eigenvalues / total

    # Eigenvalue entropy: -sum(p * log(p))
    log_eps = safe_log_epsilon(b, p)
    log_p = b.log(p + log_eps)
    eigenvalue_entropy = -b.sum(p * log_p)

    # Effective rank: exp(entropy)
    # Maximum entropy = ln(d) when all eigenvalues equal
    # Effective rank in [1, d]
    effective_rank = b.exp(eigenvalue_entropy)

    # Normalize eigenscore to [0, 1]
    # eigenscore = effective_rank / d gives fraction of dimensions used
    # Higher = more spread = more uncertain
    eigenscore = effective_rank / d

    # Condition number: max / min (for numerical stability insights)
    max_eig = eigenvalues[0]  # Already sorted descending
    # Find smallest non-zero eigenvalue, clamp to eps for stability
    min_eig_raw = eigenvalues[-1]
    eps_scalar = b.array([eps])
    min_eig = b.where(
        min_eig_raw > eps_scalar,
        b.reshape(min_eig_raw, (1,)),
        eps_scalar,
    )[0]
    condition_number = max_eig / min_eig

    # Top eigenvalue ratio: fraction of variance in top direction
    # Low ratio = spread across many directions = uncertain
    top_ratio = eigenvalues[0] / total

    # Evaluate all computations
    b.eval(eigenscore, effective_rank, eigenvalue_entropy, condition_number, top_ratio)

    return EigenScoreResult(
        eigenscore=float(b.to_scalar(eigenscore)),
        effective_rank=float(b.to_scalar(effective_rank)),
        eigenvalue_entropy=float(b.to_scalar(eigenvalue_entropy)),
        condition_number=float(b.to_scalar(condition_number)),
        top_eigenvalue_ratio=float(b.to_scalar(top_ratio)),
        n_samples=n_samples,
        n_dimensions=d,
    )
