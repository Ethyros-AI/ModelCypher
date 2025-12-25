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

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Path:
    anchor_ids: list[str]

    @property
    def transition_count(self) -> int:
        return max(0, len(self.anchor_ids) - 1)


@dataclass(frozen=True)
class Result:
    transition_gram_correlation: float
    transition_count: int
    path_count: int


class TraversalCoherence:
    @staticmethod
    def transition_inner_product(
        gram: list[float], n: int, a: int, b: int, c: int, d: int
    ) -> float:
        if not _valid_index(a, b, c, d, n):
            return float("nan")
        g_bd = float(gram[b * n + d])
        g_bc = float(gram[b * n + c])
        g_ad = float(gram[a * n + d])
        g_ac = float(gram[a * n + c])
        return g_bd - g_bc - g_ad + g_ac

    @staticmethod
    def transition_norm_squared(gram: list[float], n: int, a: int, b: int) -> float:
        if a < 0 or b < 0 or a >= n or b >= n:
            return float("nan")
        g_bb = float(gram[b * n + b])
        g_ab = float(gram[a * n + b])
        g_aa = float(gram[a * n + a])
        return g_bb - 2.0 * g_ab + g_aa

    @staticmethod
    def normalized_transition_inner_product(
        gram: list[float],
        n: int,
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> float:
        raw = TraversalCoherence.transition_inner_product(gram, n, a, b, c, d)
        norm_ab = TraversalCoherence.transition_norm_squared(gram, n, a, b)
        norm_cd = TraversalCoherence.transition_norm_squared(gram, n, c, d)
        if norm_ab <= 1e-12 or norm_cd <= 1e-12:
            return float("nan")
        return raw / math.sqrt(norm_ab * norm_cd)

    @staticmethod
    def transition_gram(
        paths: Iterable[Path],
        anchor_gram: list[float],
        anchor_ids: list[str],
    ) -> tuple[list[float], int]:
        n = len(anchor_ids)
        if n == 0 or len(anchor_gram) != n * n:
            return [], 0

        id_to_index = {anchor_id: idx for idx, anchor_id in enumerate(anchor_ids)}

        transitions: list[tuple[int, int]] = []
        for path in paths:
            for i in range(path.transition_count):
                start = id_to_index.get(path.anchor_ids[i])
                end = id_to_index.get(path.anchor_ids[i + 1])
                if start is None or end is None:
                    continue
                transitions.append((start, end))

        m = len(transitions)
        if m == 0:
            return [], 0

        transition_gram: list[float] = [0.0] * (m * m)
        for i in range(m):
            a, b = transitions[i]
            for j in range(m):
                c, d = transitions[j]
                value = TraversalCoherence.normalized_transition_inner_product(
                    gram=anchor_gram,
                    n=n,
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                )
                transition_gram[i * m + j] = float(value)

        return transition_gram, m

    @staticmethod
    def compare(
        paths: Iterable[Path],
        gram_a: list[float],
        gram_b: list[float],
        anchor_ids: list[str],
    ) -> Result | None:
        path_list = list(paths)
        trans_a, count_a = TraversalCoherence.transition_gram(path_list, gram_a, anchor_ids)
        trans_b, count_b = TraversalCoherence.transition_gram(path_list, gram_b, anchor_ids)
        if count_a != count_b or count_a <= 1:
            return None
        m = count_a

        vec_a: list[float] = []
        vec_b: list[float] = []
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                a_val = trans_a[i * m + j]
                b_val = trans_b[i * m + j]
                if not math.isfinite(a_val) or not math.isfinite(b_val):
                    continue
                vec_a.append(a_val)
                vec_b.append(b_val)

        if len(vec_a) < 2:
            return None
        correlation = _pearson_correlation(vec_a, vec_b)
        if not math.isfinite(correlation):
            return None
        return Result(
            transition_gram_correlation=correlation,
            transition_count=m,
            path_count=len(path_list),
        )


standard_computational_paths = [
    Path(anchor_ids=["1", "4", "41"]),
    Path(anchor_ids=["2", "6", "19"]),
    Path(anchor_ids=["6", "3", "52"]),
    Path(anchor_ids=["5", "3", "19"]),
    Path(anchor_ids=["48", "49", "50"]),
    Path(anchor_ids=["13", "5", "41"]),
    Path(anchor_ids=["41", "45", "19"]),
]


def _valid_index(a: int, b: int, c: int, d: int, n: int) -> bool:
    return all(0 <= idx < n for idx in (a, b, c, d))


def _pearson_correlation(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or len(lhs) != len(rhs):
        return float("nan")
    n = len(lhs)
    mean_l = sum(lhs) / n
    mean_r = sum(rhs) / n
    num = 0.0
    den_l = 0.0
    den_r = 0.0
    for i in range(n):
        diff_l = lhs[i] - mean_l
        diff_r = rhs[i] - mean_r
        num += diff_l * diff_r
        den_l += diff_l * diff_l
        den_r += diff_r * diff_r
    denom = math.sqrt(den_l * den_r)
    if denom <= 1e-12:
        max_delta = max(abs(a - b) for a, b in zip(lhs, rhs))
        if max_delta <= 1e-9:
            return 1.0
        return float("nan")
    return num / denom
