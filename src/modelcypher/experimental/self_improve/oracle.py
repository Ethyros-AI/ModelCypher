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

import re
from typing import Any, Iterable


_EQ_RE = re.compile(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)")


class VerificationOracle:
    """Deterministic verifier for simple arithmetic prompts."""

    def __init__(self, model: Any | None = None, tokenizer: Any | None = None) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def verify(self, equation: str, expected: str) -> tuple[bool, str]:
        computed = self._compute_equation(equation)
        if computed is None:
            return False, ""
        return str(computed) == expected, str(computed)

    def calibrate(self, tests: Iterable[tuple[str, str]]) -> tuple[float, list[dict[str, str]]]:
        failures: list[dict[str, str]] = []
        total = 0
        correct = 0
        for equation, expected in tests:
            total += 1
            ok, computed = self.verify(equation, expected)
            if ok:
                correct += 1
            else:
                failures.append({"equation": equation, "expected": expected, "computed": computed})
        accuracy = correct / total if total else 0.0
        return accuracy, failures

    @staticmethod
    def default_calibration_tests() -> list[tuple[str, str]]:
        return [
            ("1+1=", "2"),
            ("2+3=", "5"),
            ("9-4=", "5"),
            ("7-2=", "5"),
            ("6*3=", "18"),
            ("8/4=", "2"),
        ]

    @staticmethod
    def _compute_equation(equation: str) -> int | float | None:
        match = _EQ_RE.search(equation)
        if not match:
            return None
        lhs = int(match.group(1))
        rhs = int(match.group(3))
        op = match.group(2)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if rhs == 0:
                return None
            result = lhs / rhs
            if result.is_integer():
                return int(result)
            return result
        return None


__all__ = ["VerificationOracle"]
