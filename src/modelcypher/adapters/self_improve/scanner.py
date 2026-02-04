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
from modelcypher.core.use_cases.self_improve.types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    DEFAULT_PRIMES,
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
