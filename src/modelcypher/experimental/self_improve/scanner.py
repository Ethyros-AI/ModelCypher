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

from __future__ import annotations

from typing import Any, Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.self_improve.types import (
    DEFAULT_PRIMES,
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
)


class CapabilityScanner:
    """Scan model capabilities using backend-driven inference."""

    def __init__(self, model: Any, tokenizer: Any, backend: Any | None = None) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._backend = backend or get_default_backend()

    def scan(
        self,
        capability: Capability,
        accuracy_threshold: float,
        primes: Iterable[str] | None = None,
    ) -> CapabilityAnalysis:
        raw_accuracy = self._evaluate(capability.problems)
        best_prime = ""
        primed_accuracy = raw_accuracy

        primes_to_try = list(primes) if primes is not None else list(DEFAULT_PRIMES)
        for prime in primes_to_try:
            accuracy = self._evaluate(capability.problems, prime=prime)
            if accuracy > primed_accuracy:
                primed_accuracy = accuracy
                best_prime = prime

        if raw_accuracy >= accuracy_threshold:
            status = CapabilityStatus.WORKING
        elif primed_accuracy >= accuracy_threshold:
            status = CapabilityStatus.DISCONNECTED
        else:
            status = CapabilityStatus.TRUE_GAP

        return CapabilityAnalysis(
            capability=capability,
            status=status,
            accuracy_raw=raw_accuracy,
            accuracy_primed=primed_accuracy,
            kappa_raw=float("nan"),
            kappa_primed=float("nan"),
            best_prime=best_prime,
        )

    def collect_contrastive_activations(
        self,
        capability: Capability,
        best_prime: str,
        target_layer: int,
    ) -> tuple[Any, Any]:
        """Collect activations with and without priming for contrastive steering.

        For each prompt in the capability, collects hidden states at
        ``target_layer`` in two conditions:

        1. **Positive** — prompt prepended with ``best_prime`` (model succeeds)
        2. **Negative** — prompt alone (model fails)

        Parameters
        ----------
        capability : Capability
            The capability whose prompts to use for activation collection.
        best_prime : str
            The priming text that makes the capability work.
        target_layer : int
            Layer index to collect activations from.

        Returns
        -------
        tuple[Array, Array]
            ``(positive_activations, negative_activations)`` each with shape
            ``[n_prompts, hidden_dim]``.
        """
        b = self._backend
        layer_indices = [target_layer]
        positive_list: list[Any] = []
        negative_list: list[Any] = []

        for prompt in capability.prompts:
            # Negative: raw prompt (no priming)
            neg_states = b.collect_hidden_activations(
                self._model, self._tokenizer, [prompt],
                layer_indices=layer_indices,
            )
            # Positive: primed prompt
            primed_prompt = f"{best_prime} {prompt}"
            pos_states = b.collect_hidden_activations(
                self._model, self._tokenizer, [primed_prompt],
                layer_indices=layer_indices,
            )

            if target_layer in neg_states and target_layer in pos_states:
                neg_act = neg_states[target_layer]
                pos_act = pos_states[target_layer]
                # Take last token's hidden state: [batch, seq, hidden] → [hidden]
                if hasattr(neg_act, "ndim") and neg_act.ndim >= 3:
                    neg_act = neg_act[0, -1, :]
                elif hasattr(neg_act, "ndim") and neg_act.ndim == 2:
                    neg_act = neg_act[-1, :]
                if hasattr(pos_act, "ndim") and pos_act.ndim >= 3:
                    pos_act = pos_act[0, -1, :]
                elif hasattr(pos_act, "ndim") and pos_act.ndim == 2:
                    pos_act = pos_act[-1, :]
                b.eval(neg_act, pos_act)
                negative_list.append(b.reshape(neg_act, (1, -1)))
                positive_list.append(b.reshape(pos_act, (1, -1)))

        # Stack into [n_prompts, hidden_dim]
        positive = b.concatenate(positive_list, axis=0)
        negative = b.concatenate(negative_list, axis=0)
        b.eval(positive, negative)
        return positive, negative

    def _evaluate(self, problems: Iterable[tuple[str, str]], prime: str | None = None) -> float:
        total = 0
        correct = 0
        for prompt, expected in problems:
            total += 1
            full_prompt = f"{prime} {prompt}" if prime else prompt
            response = self._backend.generate(self._model, self._tokenizer, full_prompt)
            if expected.strip().lower() in response.lower():
                correct += 1
        return correct / total if total else 0.0


__all__ = ["CapabilityScanner"]
