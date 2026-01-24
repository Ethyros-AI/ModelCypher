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

"""RMT-aware MLP compressor.

Uses Marchenko-Pastur distribution to separate signal from noise in the
input activation space for compression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressionResult:
    """Result of MLP compression.

    Attributes:
        T: Linear transform [d_out, d_in] that approximates MLP.
        signal_rank: RMT-detected signal rank used in compression.
        total_rank: Total available rank (min(n_samples, d_in)).
        noise_filtered: Number of dimensions filtered as noise.
        reconstruction_error: Relative Frobenius error on calibration.
        mp_upper_edge: Marchenko-Pastur upper edge threshold.
        signal_variance_fraction: Fraction of variance in signal components.
    """

    T: "Array"
    signal_rank: int
    total_rank: int
    noise_filtered: int
    reconstruction_error: float
    mp_upper_edge: float
    signal_variance_fraction: float


@dataclass(frozen=True)
class EvaluationResult:
    """Result of compression evaluation on held-out data.

    Attributes:
        reconstruction_error: Relative error on held-out MLP outputs.
        ranking_preserved: Fraction of output rankings preserved.
        margin_preserved: Fraction of samples where margin sign preserved.
    """

    reconstruction_error: float
    ranking_preserved: float
    margin_preserved: float


class RMTAwareCompressor:
    """Compresses MLP using RMT signal/noise separation.

    The algorithm:
    1. Compute SVD of input activations X
    2. Use Marchenko-Pastur to identify signal vs noise singular values
    3. Compute rank-truncated pinv using only signal components
    4. Solve T = Y @ pinv_k(X) where k = signal_rank

    This filters out noise that would otherwise corrupt the compression.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compress_layer(
        self,
        X: "Array",
        Y: "Array",
    ) -> CompressionResult:
        """Compress MLP by finding linear approximation T such that T @ X ≈ Y.

        Args:
            X: MLP input activations [n_samples, d_in].
            Y: MLP output activations [n_samples, d_out].

        Returns:
            CompressionResult with transform T and diagnostics.
        """
        from modelcypher.core.domain.geometry.rmt_signal_separation import (
            compute_signal_rank_from_singular_values,
        )

        b = self._backend

        X = b.array(X)
        Y = b.array(Y)
        b.eval(X, Y)

        n_samples = int(X.shape[0])
        d_in = int(X.shape[1])
        d_out = int(Y.shape[1])

        logger.info(
            "RMT COMPRESSOR: Compressing MLP [%d, %d] -> [%d, %d]",
            n_samples, d_in, n_samples, d_out
        )

        # Step 1: SVD of input activations
        U, S, Vt = b.svd(X)
        b.eval(U, S, Vt)

        total_rank = int(S.shape[0])

        # Step 2: RMT signal/noise separation
        mp_result = compute_signal_rank_from_singular_values(
            S, n_samples=n_samples, n_features=d_in, backend=b
        )
        signal_rank = max(1, min(int(mp_result.signal_rank), total_rank))
        noise_filtered = total_rank - signal_rank

        logger.info(
            "RMT COMPRESSOR: signal_rank=%d/%d, MP_edge=%.4f, signal_var=%.1f%%",
            signal_rank, total_rank, mp_result.mp_upper_edge,
            100.0 * mp_result.signal_variance_fraction
        )

        # Step 3: Compute rank-truncated pseudoinverse
        # pinv_k(X) = V_k @ diag(1/S_k) @ U_k.T
        eps = float(division_epsilon(b, S))

        U_k = U[:, :signal_rank]  # [n, k]
        S_k = S[:signal_rank]      # [k]
        Vt_k = Vt[:signal_rank, :] # [k, d_in]

        S_inv = 1.0 / (S_k + eps)  # [k]
        V_k = b.transpose(Vt_k)    # [d_in, k]
        VS = V_k * S_inv           # [d_in, k]
        pinv_k = b.matmul(VS, b.transpose(U_k))  # [d_in, n]
        b.eval(pinv_k)

        # Step 4: Solve T.T = pinv_k(X) @ Y => T = (pinv_k @ Y).T
        T_T = b.matmul(pinv_k, Y)  # [d_in, d_out]
        T = b.transpose(T_T)       # [d_out, d_in]
        b.eval(T)

        # Step 5: Compute reconstruction error
        Y_reconstructed = b.matmul(X, T_T)  # [n, d_out]
        error = Y - Y_reconstructed
        error_norm = b.sqrt(b.sum(error * error))
        Y_norm = b.sqrt(b.sum(Y * Y))
        b.eval(error_norm, Y_norm)

        reconstruction_error = float(b.to_scalar(error_norm)) / (
            float(b.to_scalar(Y_norm)) + eps
        )

        logger.info(
            "RMT COMPRESSOR: T shape=[%d, %d], reconstruction_error=%.4f",
            d_out, d_in, reconstruction_error
        )

        return CompressionResult(
            T=T,
            signal_rank=signal_rank,
            total_rank=total_rank,
            noise_filtered=noise_filtered,
            reconstruction_error=reconstruction_error,
            mp_upper_edge=mp_result.mp_upper_edge,
            signal_variance_fraction=mp_result.signal_variance_fraction,
        )

    def evaluate(
        self,
        T: "Array",
        held_out_X: "Array",
        held_out_Y: "Array",
    ) -> EvaluationResult:
        """Evaluate compression on held-out data.

        Args:
            T: Linear transform [d_out, d_in].
            held_out_X: Held-out MLP inputs [n, d_in].
            held_out_Y: Held-out MLP outputs [n, d_out].

        Returns:
            EvaluationResult with error and ranking metrics.
        """
        b = self._backend

        T = b.array(T)
        held_out_X = b.array(held_out_X)
        held_out_Y = b.array(held_out_Y)
        b.eval(T, held_out_X, held_out_Y)

        n = int(held_out_X.shape[0])

        # Reconstruction error
        Y_pred = b.matmul(held_out_X, b.transpose(T))  # [n, d_out]
        b.eval(Y_pred)

        error = held_out_Y - Y_pred
        error_norm = b.sqrt(b.sum(error * error))
        Y_norm = b.sqrt(b.sum(held_out_Y * held_out_Y))
        b.eval(error_norm, Y_norm)

        eps = float(division_epsilon(b, held_out_Y))
        reconstruction_error = float(b.to_scalar(error_norm)) / (
            float(b.to_scalar(Y_norm)) + eps
        )

        # Ranking preservation: for each sample, check if argmax is preserved
        # This is a proxy for token accuracy
        rankings_preserved = 0
        margins_preserved = 0

        for i in range(n):
            y_true = held_out_Y[i, :]
            y_pred = Y_pred[i, :]

            # Argmax preservation
            true_argmax = b.argmax(y_true)
            pred_argmax = b.argmax(y_pred)
            b.eval(true_argmax, pred_argmax)

            if int(b.to_scalar(true_argmax)) == int(b.to_scalar(pred_argmax)):
                rankings_preserved += 1

            # Margin preservation (top-1 minus top-2)
            true_sorted = b.sort(y_true)[::-1]
            pred_sorted = b.sort(y_pred)[::-1]
            b.eval(true_sorted, pred_sorted)

            true_margin = float(b.to_scalar(true_sorted[0] - true_sorted[1]))
            pred_margin = float(b.to_scalar(pred_sorted[0] - pred_sorted[1]))

            # Sign of margin preserved?
            if (true_margin > 0 and pred_margin > 0) or (
                true_margin <= 0 and pred_margin <= 0
            ):
                margins_preserved += 1

        ranking_preserved = rankings_preserved / n if n > 0 else 0.0
        margin_preserved = margins_preserved / n if n > 0 else 0.0

        logger.info(
            "RMT COMPRESSOR EVAL: recon_error=%.4f, ranking=%.1f%%, margin=%.1f%%",
            reconstruction_error, 100 * ranking_preserved, 100 * margin_preserved
        )

        return EvaluationResult(
            reconstruction_error=reconstruction_error,
            ranking_preserved=ranking_preserved,
            margin_preserved=margin_preserved,
        )

    def compress_with_naive_comparison(
        self,
        X: "Array",
        Y: "Array",
    ) -> tuple[CompressionResult, CompressionResult]:
        """Compress using both RMT and naive methods for comparison.

        Returns:
            (rmt_result, naive_result)
        """
        b = self._backend

        X = b.array(X)
        Y = b.array(Y)
        b.eval(X, Y)

        # RMT compression
        rmt_result = self.compress_layer(X, Y)

        # Naive compression (full rank pinv)
        n_samples = int(X.shape[0])
        d_in = int(X.shape[1])
        d_out = int(Y.shape[1])

        U, S, Vt = b.svd(X)
        b.eval(U, S, Vt)

        total_rank = int(S.shape[0])
        eps = float(division_epsilon(b, S))

        # Full rank pinv - need to slice Vt to match S dimensions
        # SVD: X = U @ diag(S) @ Vt where U=[n,k], S=[k], Vt=[d,d] (MLX returns full)
        # We need Vt[:k, :] to match S
        Vt_k = Vt[:total_rank, :]  # [k, d_in]
        S_inv = 1.0 / (S + eps)
        V_k = b.transpose(Vt_k)    # [d_in, k]
        VS = V_k * S_inv           # [d_in, k]
        pinv_full = b.matmul(VS, b.transpose(U))  # [d_in, n]
        b.eval(pinv_full)

        T_T_naive = b.matmul(pinv_full, Y)
        T_naive = b.transpose(T_T_naive)
        b.eval(T_naive)

        # Naive reconstruction error
        Y_reconstructed = b.matmul(X, T_T_naive)
        error = Y - Y_reconstructed
        error_norm = b.sqrt(b.sum(error * error))
        Y_norm = b.sqrt(b.sum(Y * Y))
        b.eval(error_norm, Y_norm)

        naive_recon_error = float(b.to_scalar(error_norm)) / (
            float(b.to_scalar(Y_norm)) + eps
        )

        naive_result = CompressionResult(
            T=T_naive,
            signal_rank=total_rank,
            total_rank=total_rank,
            noise_filtered=0,
            reconstruction_error=naive_recon_error,
            mp_upper_edge=0.0,  # Not applicable
            signal_variance_fraction=1.0,  # All variance used
        )

        return rmt_result, naive_result
