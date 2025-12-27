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
WUDI interference metrics for data-free model merging.

Implements the WUDI loss (ICML 2025) as a deterministic interference signal:
    L_l = Σ_i 1/||τ_i||_F^2 * ||(τ_m - τ_i) τ_i^T||_F^2

Where τ_i are task vectors for a linear layer and τ_m is their sum.
This module provides a shape-aware grouping utility and overlap metrics
to avoid SVD on GPU backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class WUDIGroupResult:
    """Per-shape WUDI interference metrics."""

    shape: tuple[int, int]
    task_count: int
    loss: float
    mean_overlap: float
    max_overlap: float


@dataclass(frozen=True)
class WUDIInterferenceResult:
    """Aggregated WUDI interference metrics across shape groups."""

    group_results: list[WUDIGroupResult]
    mean_loss: float
    mean_overlap: float
    max_overlap: float

    @property
    def normalized_loss(self) -> float:
        """Monotonic normalization to map loss into [0, 1)."""
        return self.mean_loss / (1.0 + self.mean_loss)


def group_task_vectors_by_shape(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    backend: "Backend | None" = None,
) -> dict[tuple[int, int], list["Array"]]:
    """Group task vectors by shape for WUDI loss evaluation."""
    b = backend or get_default_backend()
    groups: dict[tuple[int, int], list["Array"]] = {}
    for key, src_weight in source_weights.items():
        if key not in target_weights:
            continue
        tgt_weight = target_weights[key]
        if src_weight.shape != tgt_weight.shape:
            continue
        if src_weight.ndim != 2:
            continue
        delta = src_weight - tgt_weight
        shape = (int(delta.shape[0]), int(delta.shape[1]))
        groups.setdefault(shape, []).append(delta)
        b.eval(delta)
    return groups


def subspace_overlap(
    matrix_a: "Array",
    matrix_b: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute normalized overlap between two task vectors.

    Uses squared cosine similarity in Frobenius space so identical
    matrices yield 1.0 and orthogonal matrices yield 0.0.
    """
    b = backend or get_default_backend()
    eps = division_epsilon(b, matrix_a)

    dot = b.sum(matrix_a * matrix_b)
    norm_a = b.sum(matrix_a * matrix_a)
    norm_b = b.sum(matrix_b * matrix_b)
    denom = b.sqrt(norm_a * norm_b) + eps
    cosine = dot / denom
    overlap = cosine * cosine
    b.eval(overlap)
    return float(b.to_numpy(overlap).item())


def compute_wudi_interference(
    groups: dict[tuple[int, int], list["Array"]],
    backend: "Backend | None" = None,
) -> WUDIInterferenceResult:
    """Compute WUDI interference metrics from grouped task vectors."""
    b = backend or get_default_backend()
    group_results: list[WUDIGroupResult] = []
    all_losses: list[float] = []
    all_mean_overlaps: list[float] = []
    max_overlap = 0.0

    for shape, vectors in groups.items():
        if not vectors:
            continue
        eps = division_epsilon(b, vectors[0])
        tau_m = vectors[0]
        for v in vectors[1:]:
            tau_m = tau_m + v
        b.eval(tau_m)

        loss_arr = b.sum(tau_m * 0.0)
        overlaps: list[float] = []

        for i, tau_i in enumerate(vectors):
            diff = tau_m - tau_i
            inner = b.matmul(diff, b.transpose(tau_i))
            num = b.sum(inner * inner)
            denom = b.sum(tau_i * tau_i) + eps
            loss_arr = loss_arr + num / denom
            b.eval(loss_arr)
            for j in range(i + 1, len(vectors)):
                overlaps.append(subspace_overlap(tau_i, vectors[j], b))

        if overlaps:
            mean_overlap = sum(overlaps) / len(overlaps)
            max_group_overlap = max(overlaps)
        else:
            mean_overlap = 0.0
            max_group_overlap = 0.0

        loss_val = float(b.to_numpy(loss_arr).item())
        group_results.append(
            WUDIGroupResult(
                shape=shape,
                task_count=len(vectors),
                loss=loss_val,
                mean_overlap=mean_overlap,
                max_overlap=max_group_overlap,
            )
        )
        all_losses.append(loss_val)
        all_mean_overlaps.append(mean_overlap)
        max_overlap = max(max_overlap, max_group_overlap)

    mean_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    mean_overlap = (
        sum(all_mean_overlaps) / len(all_mean_overlaps) if all_mean_overlaps else 0.0
    )

    return WUDIInterferenceResult(
        group_results=group_results,
        mean_loss=mean_loss,
        mean_overlap=mean_overlap,
        max_overlap=max_overlap,
    )
