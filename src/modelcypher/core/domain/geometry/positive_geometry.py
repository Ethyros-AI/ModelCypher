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

"""Positive geometry diagnostics (positive Grassmannian signatures)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    is_finite,
    log_scalar,
    machine_epsilon,
    sqrt_scalar,
    svd_auto_rank,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class PositiveGrassmannSignature:
    """Raw measurements for positive Grassmannian signatures."""

    probe_count: int
    ambient_dim: int
    subspace_rank: int
    total_minors: int
    evaluated_minors: int
    finite_minors: int
    non_finite_minors: int
    selection: str
    zero_threshold: float
    positive_count: int
    negative_count: int
    zero_count: int
    positive_fraction: float
    negative_fraction: float
    zero_fraction: float
    sign_entropy: float
    min_minor: float
    max_minor: float
    mean_minor: float
    mean_abs_minor: float
    mean_positive_minor: float | None
    mean_negative_minor: float | None
    plucker_norm: float
    max_abs_minor: float

    def to_dict(self) -> dict:
        """Return signature as a JSON-serializable dict."""
        return {
            "probeCount": self.probe_count,
            "ambientDim": self.ambient_dim,
            "subspaceRank": self.subspace_rank,
            "totalMinors": self.total_minors,
            "evaluatedMinors": self.evaluated_minors,
            "finiteMinors": self.finite_minors,
            "nonFiniteMinors": self.non_finite_minors,
            "selection": self.selection,
            "zeroThreshold": self.zero_threshold,
            "positiveCount": self.positive_count,
            "negativeCount": self.negative_count,
            "zeroCount": self.zero_count,
            "positiveFraction": self.positive_fraction,
            "negativeFraction": self.negative_fraction,
            "zeroFraction": self.zero_fraction,
            "signEntropy": self.sign_entropy,
            "minMinor": self.min_minor,
            "maxMinor": self.max_minor,
            "meanMinor": self.mean_minor,
            "meanAbsMinor": self.mean_abs_minor,
            "meanPositiveMinor": self.mean_positive_minor,
            "meanNegativeMinor": self.mean_negative_minor,
            "pluckerNorm": self.plucker_norm,
            "maxAbsMinor": self.max_abs_minor,
        }


def _take_element(matrix: "Array", row: int, col: int, backend: "Backend") -> "Array":
    idx_r = backend.array([row])
    idx_c = backend.array([col])
    elem = backend.take(matrix, idx_r, axis=0)
    elem = backend.take(elem, idx_c, axis=1)
    return backend.squeeze(elem)


def _determinant_small(matrix: "Array", backend: "Backend") -> "Array":
    n = int(matrix.shape[0])
    if n == 1:
        return _take_element(matrix, 0, 0, backend)
    if n == 2:
        a = _take_element(matrix, 0, 0, backend)
        b = _take_element(matrix, 0, 1, backend)
        c = _take_element(matrix, 1, 0, backend)
        d = _take_element(matrix, 1, 1, backend)
        return a * d - b * c
    if n == 3:
        a = _take_element(matrix, 0, 0, backend)
        b = _take_element(matrix, 0, 1, backend)
        c = _take_element(matrix, 0, 2, backend)
        d = _take_element(matrix, 1, 0, backend)
        e = _take_element(matrix, 1, 1, backend)
        f = _take_element(matrix, 1, 2, backend)
        g = _take_element(matrix, 2, 0, backend)
        h = _take_element(matrix, 2, 1, backend)
        i = _take_element(matrix, 2, 2, backend)
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return backend.det(matrix)


def compute_positive_grassmann_signature(
    activations: Array,
    backend: Backend | None = None,
    max_minors: int | None = None,
    selection: str = "lexicographic",
) -> PositiveGrassmannSignature:
    """Compute positive Grassmannian signatures from activation subspace.

    Treats the column space of the activation matrix (n_probes x hidden_dim)
    as a point on Gr(k, n_probes). The ordered columns are determined by the
    probe order used to construct the activation matrix.
    """
    b = backend or get_default_backend()
    acts = activations
    b.eval(acts)

    n = int(acts.shape[0])
    d = int(acts.shape[1]) if len(acts.shape) > 1 else 0

    if n == 0 or d == 0:
        return PositiveGrassmannSignature(
            probe_count=n,
            ambient_dim=d,
            subspace_rank=0,
            total_minors=0,
            evaluated_minors=0,
            finite_minors=0,
            non_finite_minors=0,
            selection=selection,
            zero_threshold=0.0,
            positive_count=0,
            negative_count=0,
            zero_count=0,
            positive_fraction=0.0,
            negative_fraction=0.0,
            zero_fraction=0.0,
            sign_entropy=0.0,
            min_minor=0.0,
            max_minor=0.0,
            mean_minor=0.0,
            mean_abs_minor=0.0,
            mean_positive_minor=None,
            mean_negative_minor=None,
            plucker_norm=0.0,
            max_abs_minor=0.0,
        )

    # Compute orthonormal basis of column space
    U, S, _ = b.svd(acts, compute_uv=True)
    b.eval(U, S)
    rank = svd_auto_rank(S, b, max_dim=max(n, d))

    if rank <= 0:
        return PositiveGrassmannSignature(
            probe_count=n,
            ambient_dim=d,
            subspace_rank=0,
            total_minors=0,
            evaluated_minors=0,
            finite_minors=0,
            non_finite_minors=0,
            selection=selection,
            zero_threshold=0.0,
            positive_count=0,
            negative_count=0,
            zero_count=0,
            positive_fraction=0.0,
            negative_fraction=0.0,
            zero_fraction=0.0,
            sign_entropy=0.0,
            min_minor=0.0,
            max_minor=0.0,
            mean_minor=0.0,
            mean_abs_minor=0.0,
            mean_positive_minor=None,
            mean_negative_minor=None,
            plucker_norm=0.0,
            max_abs_minor=0.0,
        )

    U_k = U[:, :rank]
    G = b.transpose(U_k)  # [rank, n]
    b.eval(G)

    total_minors = comb(n, rank) if n >= rank else 0
    evaluated_minors = 0
    dets: list[float] = []

    if total_minors == 0:
        return PositiveGrassmannSignature(
            probe_count=n,
            ambient_dim=d,
            subspace_rank=rank,
            total_minors=0,
            evaluated_minors=0,
            finite_minors=0,
            non_finite_minors=0,
            selection=selection,
            zero_threshold=0.0,
            positive_count=0,
            negative_count=0,
            zero_count=0,
            positive_fraction=0.0,
            negative_fraction=0.0,
            zero_fraction=0.0,
            sign_entropy=0.0,
            min_minor=0.0,
            max_minor=0.0,
            mean_minor=0.0,
            mean_abs_minor=0.0,
            mean_positive_minor=None,
            mean_negative_minor=None,
            plucker_norm=0.0,
            max_abs_minor=0.0,
        )

    for idxs in combinations(range(n), rank):
        if max_minors is not None and evaluated_minors >= max_minors:
            break
        idx_arr = b.array(list(idxs))
        sub = b.take(G, idx_arr, axis=1)
        det = _determinant_small(sub, b)
        b.eval(det)
        det_val = float(b.to_scalar(det))
        dets.append(det_val)
        evaluated_minors += 1

    finite_dets = [dval for dval in dets if is_finite(dval, b)]
    non_finite = len(dets) - len(finite_dets)

    if not finite_dets:
        return PositiveGrassmannSignature(
            probe_count=n,
            ambient_dim=d,
            subspace_rank=rank,
            total_minors=total_minors,
            evaluated_minors=evaluated_minors,
            finite_minors=0,
            non_finite_minors=non_finite,
            selection=selection,
            zero_threshold=0.0,
            positive_count=0,
            negative_count=0,
            zero_count=0,
            positive_fraction=0.0,
            negative_fraction=0.0,
            zero_fraction=0.0,
            sign_entropy=0.0,
            min_minor=0.0,
            max_minor=0.0,
            mean_minor=0.0,
            mean_abs_minor=0.0,
            mean_positive_minor=None,
            mean_negative_minor=None,
            plucker_norm=0.0,
            max_abs_minor=0.0,
        )

    max_abs = max(abs(dval) for dval in finite_dets)
    eps = machine_epsilon(b, S)
    zero_threshold = sqrt_scalar(eps, b) * max_abs

    positives = [dval for dval in finite_dets if dval > zero_threshold]
    negatives = [dval for dval in finite_dets if dval < -zero_threshold]
    zeros = [dval for dval in finite_dets if abs(dval) <= zero_threshold]

    finite_count = len(finite_dets)
    pos_count = len(positives)
    neg_count = len(negatives)
    zero_count = len(zeros)

    pos_frac = pos_count / finite_count if finite_count > 0 else 0.0
    neg_frac = neg_count / finite_count if finite_count > 0 else 0.0
    zero_frac = zero_count / finite_count if finite_count > 0 else 0.0

    entropy = 0.0
    for frac in (pos_frac, neg_frac, zero_frac):
        if frac > 0.0:
            entropy -= frac * log_scalar(frac, b)

    sum_det = sum(finite_dets)
    sum_abs = sum(abs(dval) for dval in finite_dets)
    sum_sq = sum(dval * dval for dval in finite_dets)

    mean_minor = sum_det / finite_count if finite_count > 0 else 0.0
    mean_abs = sum_abs / finite_count if finite_count > 0 else 0.0

    mean_pos = sum(positives) / pos_count if pos_count > 0 else None
    mean_neg = sum(negatives) / neg_count if neg_count > 0 else None

    plucker_norm = sqrt_scalar(sum_sq, b)

    return PositiveGrassmannSignature(
        probe_count=n,
        ambient_dim=d,
        subspace_rank=rank,
        total_minors=total_minors,
        evaluated_minors=evaluated_minors,
        finite_minors=finite_count,
        non_finite_minors=non_finite,
        selection=selection,
        zero_threshold=zero_threshold,
        positive_count=pos_count,
        negative_count=neg_count,
        zero_count=zero_count,
        positive_fraction=pos_frac,
        negative_fraction=neg_frac,
        zero_fraction=zero_frac,
        sign_entropy=entropy,
        min_minor=min(finite_dets),
        max_minor=max(finite_dets),
        mean_minor=mean_minor,
        mean_abs_minor=mean_abs,
        mean_positive_minor=mean_pos,
        mean_negative_minor=mean_neg,
        plucker_norm=plucker_norm,
        max_abs_minor=max_abs,
    )


__all__ = ["PositiveGrassmannSignature", "compute_positive_grassmann_signature"]
