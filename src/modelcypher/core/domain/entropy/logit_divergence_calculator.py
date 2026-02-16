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

"""KL Divergence Calculator for logit geometry.

Provides KL divergence computation between model logit vectors.
Used for measuring divergence between base and adapted model outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class LogitDivergenceCalculator:
    """KL divergence calculator for logit vectors.

    Computes D_KL(primary || probe) between two logit vectors,
    useful for measuring how much an adapted model diverges from base.

    Examples
    --------
    Basic usage:

        calc = LogitDivergenceCalculator()
        kl = calc.kl_divergence(adapter_logits, base_logits)
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            backend: Compute backend. If None, uses default.
        """
        self._backend = backend or get_default_backend()
        self._epsilon = machine_epsilon(self._backend, self._backend.array([0.0]))

    def stable_softmax(self, flat_logits: "Array") -> "Array":
        """Compute numerically stable softmax over logits.

        Args:
            flat_logits: 1D array of logits [vocab_size].

        Returns:
            Simplex-normalized scores over vocabulary.
        """
        b = self._backend
        max_val = b.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_val
        exp_shifted = b.exp(shifted)
        sum_exp = b.sum(exp_shifted, axis=-1, keepdims=True)
        return exp_shifted / sum_exp

    def kl_divergence(self, primary_logits: "Array", probe_logits: "Array") -> float:
        """Compute KL divergence D_KL(primary || probe).

        Args:
            primary_logits: Logits from primary model (e.g., adapter).
                Shape: [vocab], [batch, vocab], or [batch, seq, vocab].
            probe_logits: Logits from probe model (e.g., base model).
                Same shape requirements as primary_logits.

        Returns:
            KL divergence as a non-negative float. Higher values indicate
            more divergence between the logit vectors.
        """
        b = self._backend

        # Flatten to 1D
        p_flat = primary_logits
        q_flat = probe_logits
        if primary_logits.ndim > 1:
            p_flat = primary_logits[0, -1] if primary_logits.ndim == 3 else primary_logits[0]
        if probe_logits.ndim > 1:
            q_flat = probe_logits[0, -1] if probe_logits.ndim == 3 else probe_logits[0]

        p = self.stable_softmax(p_flat)
        q = self.stable_softmax(q_flat)

        log_p = b.log(p + self._epsilon)
        log_q = b.log(q + self._epsilon)

        kl = b.sum(p * (log_p - log_q), axis=-1)
        b.eval(kl)
        return max(0.0, float(b.to_scalar(kl)))


__all__ = ["LogitDivergenceCalculator"]
