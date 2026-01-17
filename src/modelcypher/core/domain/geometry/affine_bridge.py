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

"""Affine bridge for coordinate alignment.

Fits an affine mapping between representation spaces using ridge regression.

References:
    - Belrose et al. (2023) "Eliciting Latent Predictions from Transformers
      with the Tuned Lens." arXiv:2303.08112
    - Fisher et al. (2025) "Activation-Informed Merging (AIM)." arXiv:2502.02421
    - Hastie et al. (2009) "Elements of Statistical Learning" - Ridge regression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    geodesic_svd,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AffineBridgeResult:
    """Result of affine bridge training."""

    # Transformation parameters
    W: list[list[float]]  # [d_in, d_out] transformation matrix
    b: list[float]  # [d_out] bias vector

    # Training metrics
    train_mse: float
    train_cosine: float
    test_cosine: float | None  # None if no test set provided
    generalization_gap: float | None  # train_cosine - test_cosine

    # Dimensions
    source_dim: int
    target_dim: int
    n_train_samples: int
    n_test_samples: int | None

    # Regularization used
    regularization: float

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        test_str = f"{self.test_cosine:.4f}" if self.test_cosine is not None else "N/A"
        gap_str = f"{self.generalization_gap:.4f}" if self.generalization_gap is not None else "N/A"
        return (
            "Affine Bridge Training Result\n"
            "==============================\n"
            f"Dimensions: {self.source_dim} -> {self.target_dim}\n"
            f"Training samples: {self.n_train_samples}\n"
            f"Test samples: {self.n_test_samples or 0}\n"
            f"Regularization: {self.regularization}\n\n"
            "Metrics:\n"
            f"- Train MSE: {self.train_mse:.6f}\n"
            f"- Train cosine: {self.train_cosine:.4f}\n"
            f"- Test cosine: {test_str}\n"
            f"- Generalization gap: {gap_str}"
        )


@dataclass(frozen=True)
class VocabConstrainedResult:
    """Result of vocabulary-constrained projection."""

    # Aligned embeddings as soft mixture of vocabulary
    aligned: list[list[float]]  # [n_samples, vocab_dim]

    # Attention weights over vocabulary (interpretable)
    attention_weights: list[list[float]]  # [n_samples, vocab_size]

    # Nearest tokens for each sample
    nearest_token_ids: list[int]

    # Auto-derived temperature (exposed for diagnostics)
    temperature_used: float

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            "Vocabulary-Constrained Projection Result\n"
            "========================================\n"
            f"Samples: {len(self.aligned)}\n"
            f"Temperature (auto-derived): {self.temperature_used:.4f}\n"
            f"First 5 nearest tokens: {self.nearest_token_ids[:5]}"
        )


class AffineBridge:
    """
    Affine transformation for cross-space alignment.

    Learns Y = X @ W + b via ridge regression, enabling alignment between
    different embedding spaces (e.g., CLIP -> LLM vocabulary space).

    Unlike Procrustes (orthogonal rotation only), affine learns:
    - Rotation, scaling, shearing (via W)
    - Translation (via b)

    This provides sufficient degrees of freedom for cross-modal alignment
    while regularization prevents overfitting.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._W: "Array | None" = None
        self._b: "Array | None" = None
        self._source_dim: int = 0
        self._target_dim: int = 0

    def _array_to_list(self, array: "Array") -> list[float]:
        """Convert 1D array to Python list."""
        return self._backend.tolist(self._backend.reshape(array, (-1,)))

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list."""
        return self._backend.tolist(array)

    def train(
        self,
        X_train: "Array",
        Y_train: "Array",
        X_test: "Array | None" = None,
        Y_test: "Array | None" = None,
    ) -> AffineBridgeResult:
        """
        Train affine transformation via ridge regression.

        All parameters derived from data:
        - Regularization: derived from eigenvalue spectrum of X^T X

        Args:
            X_train: Source embeddings [n_samples, source_dim]
            Y_train: Target embeddings [n_samples, target_dim]
            X_test: Optional test source embeddings
            Y_test: Optional test target embeddings

        Returns:
            AffineBridgeResult with learned transformation and metrics
        """
        backend = self._backend

        # Get dimensions
        n_samples = int(X_train.shape[0])
        source_dim = int(X_train.shape[1])
        target_dim = int(Y_train.shape[1])

        self._source_dim = source_dim
        self._target_dim = target_dim

        # Promote to highest available precision for stability
        compute_dtype = precision_dtype(backend, reference=X_train)
        if hasattr(Y_train, "dtype"):
            try:
                if backend.finfo(Y_train.dtype).eps < backend.finfo(compute_dtype).eps:
                    compute_dtype = Y_train.dtype
            except Exception:
                pass
        X = backend.astype(X_train, compute_dtype)
        Y = backend.astype(Y_train, compute_dtype)

        # Compute X^T X
        XtX = backend.matmul(backend.transpose(X), X)

        # Derive regularization from eigenvalue spectrum
        # Use sqrt(machine_epsilon) * trace(XtX) / d as adaptive regularization
        eps = machine_epsilon(backend, X)
        trace_XtX = backend.sum(backend.diag(XtX))
        backend.eval(trace_XtX)
        trace_val = float(backend.to_scalar(trace_XtX))
        regularization = float(eps ** 0.5) * max(trace_val / source_dim, 1.0)

        # Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        reg_term = regularization * backend.eye(source_dim, dtype=X.dtype)
        A = XtX + reg_term
        XtY = backend.matmul(backend.transpose(X), Y)

        # Solve via SVD for numerical stability
        U, S, Vt = geodesic_svd(backend, A)

        # Compute pseudo-inverse: V @ S^(-1) @ U^T
        S_inv = 1.0 / (S + division_epsilon(backend, S))
        A_inv = backend.matmul(
            backend.transpose(Vt),
            backend.matmul(backend.diag(S_inv), backend.transpose(U))
        )
        W = backend.matmul(A_inv, XtY)
        backend.eval(W)

        # Compute bias: b = mean(Y - X @ W)
        residual = Y - backend.matmul(X, W)
        b = backend.mean(residual, axis=0)
        backend.eval(b)

        self._W = W
        self._b = b

        # Evaluate on training set
        train_mse, train_cosine = self._evaluate(X, Y)

        # Evaluate on test set if provided
        test_cosine = None
        n_test = None
        generalization_gap = None
        if X_test is not None and Y_test is not None:
            X_test_f = backend.astype(X_test, compute_dtype)
            Y_test_f = backend.astype(Y_test, compute_dtype)
            _, test_cosine = self._evaluate(X_test_f, Y_test_f)
            n_test = int(X_test.shape[0])
            generalization_gap = train_cosine - test_cosine

        return AffineBridgeResult(
            W=self._array_to_2d_list(W),
            b=self._array_to_list(b),
            train_mse=train_mse,
            train_cosine=train_cosine,
            test_cosine=test_cosine,
            generalization_gap=generalization_gap,
            source_dim=source_dim,
            target_dim=target_dim,
            n_train_samples=n_samples,
            n_test_samples=n_test,
            regularization=regularization,
        )

    def _evaluate(
        self,
        X: "Array",
        Y: "Array",
    ) -> tuple[float, float]:
        """Evaluate affine bridge: compute MSE and mean cosine similarity."""
        backend = self._backend

        if self._W is None or self._b is None:
            msg = "Must call train() before evaluate()"
            raise ValueError(msg)

        # Transform
        pred = backend.matmul(X, self._W) + self._b
        backend.eval(pred)

        # MSE
        diff = pred - Y
        mse_arr = backend.mean(diff * diff)
        backend.eval(mse_arr)
        mse = float(backend.to_scalar(mse_arr))

        # Cosine similarity (per sample, then mean)
        n = int(X.shape[0])
        if n >= 3:
            # Geodesic cosine for manifold-correct similarity
            cosines, _ = geodesic_pairwise_metrics(pred, Y, backend)
        else:
            # Chord cosine for small samples (geodesic requires n >= 3)
            cosines = self._chord_cosine_paired(pred, Y)
        mean_cosine_arr = backend.mean(cosines)
        backend.eval(mean_cosine_arr)
        mean_cosine = float(backend.to_scalar(mean_cosine_arr))

        return mse, mean_cosine

    def _chord_cosine_paired(self, a: "Array", b: "Array") -> "Array":
        """Compute chord (Euclidean) cosine for paired vectors."""
        backend = self._backend
        eps = division_epsilon(backend, a)
        norm_a = backend.sqrt(backend.sum(a * a, axis=1))
        norm_b = backend.sqrt(backend.sum(b * b, axis=1))
        dot = backend.sum(a * b, axis=1)
        denom = norm_a * norm_b
        safe_denom = backend.maximum(denom, backend.full(backend.shape(denom), eps))
        cosines = dot / safe_denom
        return backend.clip(cosines, -1.0, 1.0)

    def transform(self, X: "Array") -> "Array":
        """
        Apply learned affine transformation.

        Args:
            X: Source embeddings [n_samples, source_dim]

        Returns:
            Transformed embeddings [n_samples, target_dim]
        """
        if self._W is None or self._b is None:
            msg = "Must call train() before transform()"
            raise ValueError(msg)

        backend = self._backend
        X_f = backend.astype(X, precision_dtype(backend, reference=self._W))
        aligned = backend.matmul(X_f, self._W) + self._b
        backend.eval(aligned)
        return aligned

    def load_weights(self, W: "Array", b: "Array") -> None:
        """
        Load pre-trained transformation weights.

        Args:
            W: Transformation matrix [source_dim, target_dim]
            b: Bias vector [target_dim]
        """
        backend = self._backend
        self._W = backend.astype(W, precision_dtype(backend, reference=W))
        self._b = backend.astype(b, precision_dtype(backend, reference=b))
        backend.eval(self._W, self._b)

        self._source_dim = int(self._W.shape[0])
        self._target_dim = int(self._W.shape[1])


class VocabConstrainedProjection:
    """
    Vocabulary-constrained projection for token-space alignment.

    Vocabulary-constrained projection forces output onto the vocabulary manifold:
        attention = softmax(X @ vocab.T / temperature)
        aligned = attention @ vocab

    Temperature is derived from effective dimensionality:
        d_eff = (Σλ)² / Σλ² (Rényi entropy-based effective rank)
        temperature = 1 / √(d_eff)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._vocab_embeddings: "Array | None" = None
        self._vocab_norms: "Array | None" = None
        self._effective_dim: float = 1.0  # Derived from vocabulary

    def set_vocabulary(self, vocab_embeddings: "Array") -> None:
        """
        Set the vocabulary embeddings to project onto.

        Args:
            vocab_embeddings: [vocab_size, embed_dim] token embeddings
        """
        backend = self._backend
        self._vocab_embeddings = backend.astype(
            vocab_embeddings, precision_dtype(backend, reference=vocab_embeddings)
        )

        # Pre-compute normalized vocabulary for cosine attention
        norms = geodesic_norms(self._vocab_embeddings, backend)
        eps = division_epsilon(backend, self._vocab_embeddings)
        self._vocab_norms = self._vocab_embeddings / (
            backend.reshape(norms, (-1, 1)) + eps
        )
        backend.eval(self._vocab_embeddings, self._vocab_norms)

        # Compute effective dimensionality from vocabulary Gram matrix
        # d_eff = (Σλ)² / Σλ² (Rényi entropy-based effective rank)
        # This determines optimal temperature: T = 1/√(d_eff)
        K_vocab = backend.matmul(self._vocab_norms, backend.transpose(self._vocab_norms))
        eigenvalues = backend.eigvalsh(K_vocab)
        eigenvalues = backend.maximum(eigenvalues, backend.zeros_like(eigenvalues))
        sum_eigenvals = backend.sum(eigenvalues)
        sum_sq_eigenvals = backend.sum(eigenvalues * eigenvalues)
        d_eff = (sum_eigenvals * sum_eigenvals) / (sum_sq_eigenvals + eps)
        backend.eval(d_eff)
        self._effective_dim = max(1.0, float(backend.to_scalar(d_eff)))

    def project(self, X: "Array") -> VocabConstrainedResult:
        """
        Project embeddings onto vocabulary manifold.

        Temperature is auto-derived from similarity distribution.
        No user-configurable parameters.

        Args:
            X: Source embeddings [n_samples, embed_dim]

        Returns:
            VocabConstrainedResult with aligned embeddings and attention weights
        """
        if self._vocab_embeddings is None or self._vocab_norms is None:
            msg = "Must call set_vocabulary() before project()"
            raise ValueError(msg)

        backend = self._backend
        X_f = backend.astype(X, precision_dtype(backend, reference=self._vocab_embeddings))

        # Normalize input for cosine similarity
        x_norms = geodesic_norms(X_f, backend)
        eps = division_epsilon(backend, X_f)
        X_normed = X_f / (backend.reshape(x_norms, (-1, 1)) + eps)
        backend.eval(X_normed)

        # Temperature derived from effective dimensionality
        # Formula: T = 1/√(d_eff) where d_eff = (Σλ)²/Σλ² (Rényi entropy)
        # This is mathematically derived, not a heuristic
        import math
        temperature = 1.0 / math.sqrt(self._effective_dim)

        # Compute cosine similarities: X @ vocab^T
        similarities = backend.matmul(X_normed, backend.transpose(self._vocab_norms))

        # Softmax attention over vocabulary
        # attention[i, j] = exp(sim[i,j] / temp) / sum_k exp(sim[i,k] / temp)
        scaled = similarities / temperature

        # Numerical stability: subtract max before exp
        max_vals = backend.max(scaled, axis=1, keepdims=True)
        exp_vals = backend.exp(scaled - max_vals)
        attention = exp_vals / backend.sum(exp_vals, axis=1, keepdims=True)
        backend.eval(attention)

        # Aligned = weighted sum of vocabulary embeddings
        aligned = backend.matmul(attention, self._vocab_embeddings)
        backend.eval(aligned)

        # Get nearest tokens (argmax of attention)
        nearest_ids_arr = backend.argmax(attention, axis=1)
        backend.eval(nearest_ids_arr)
        nearest_ids = [int(x) for x in backend.tolist(nearest_ids_arr)]

        return VocabConstrainedResult(
            aligned=self._array_to_2d_list(aligned),
            attention_weights=self._array_to_2d_list(attention),
            nearest_token_ids=nearest_ids,
            temperature_used=temperature,
        )

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list."""
        return self._backend.tolist(array)


class HybridBridge:
    """
    Combined affine + vocabulary-constrained bridge.

    All parameters are auto-derived from the data. No user-configurable knobs.

    Applies affine transformation first (for direction alignment),
    then vocabulary-constrained projection (for token neighborhood).

    Pipeline:
        1. X_affine = X @ W + b  (affine alignment)
        2. temperature = auto-derive from similarity distribution
        3. attention = softmax(X_affine @ vocab^T / temp)  (soft token lookup)
        4. aligned = attention @ vocab  (vocabulary-constrained output)

    This gets the best of both:
    - Affine: Learns global rotation/scaling between spaces
    - Vocab-constrained: Ensures output is in valid token neighborhood
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._affine = AffineBridge(backend)
        self._vocab_proj = VocabConstrainedProjection(backend)

    def train(
        self,
        X_train: "Array",
        Y_train: "Array",
        vocab_embeddings: "Array",
        X_test: "Array | None" = None,
        Y_test: "Array | None" = None,
    ) -> AffineBridgeResult:
        """
        Train the hybrid bridge.

        Args:
            X_train: Source embeddings [n_samples, source_dim]
            Y_train: Target embeddings [n_samples, target_dim]
            vocab_embeddings: Vocabulary embeddings [vocab_size, embed_dim]
            X_test: Optional test source embeddings
            Y_test: Optional test target embeddings

        Returns:
            AffineBridgeResult from affine training
        """
        # Train affine bridge
        result = self._affine.train(X_train, Y_train, X_test, Y_test)

        # Set vocabulary for constrained projection
        self._vocab_proj.set_vocabulary(vocab_embeddings)

        return result

    def transform(self, X: "Array") -> VocabConstrainedResult:
        """
        Apply hybrid transformation.

        Temperature is auto-derived from similarity distribution.
        No user-configurable parameters.

        Args:
            X: Source embeddings [n_samples, source_dim]

        Returns:
            VocabConstrainedResult with vocabulary-constrained output
        """
        # Apply affine transformation
        X_affine = self._affine.transform(X)

        # Project onto vocabulary manifold (temperature auto-derived)
        return self._vocab_proj.project(X_affine)

    def load_affine_weights(self, W: "Array", b: "Array") -> None:
        """Load pre-trained affine weights."""
        self._affine.load_weights(W, b)

    def set_vocabulary(self, vocab_embeddings: "Array") -> None:
        """Set vocabulary embeddings."""
        self._vocab_proj.set_vocabulary(vocab_embeddings)
