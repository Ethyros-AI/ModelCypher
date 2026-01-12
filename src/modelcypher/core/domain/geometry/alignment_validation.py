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

"""Alignment generalization validation utilities.

Provides train/holdout CKA measurements and probe coverage metrics so
alignment generalization is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_linear_cka
from modelcypher.core.domain.geometry.atlas_protocols import enum_key
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.numerical_stability import invariant_alignment

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol


@dataclass(frozen=True)
class AlignmentGeneralizationReport:
    """Alignment generalization metrics on train/holdout splits."""

    train_samples: int
    holdout_samples: int
    train_cka: float
    holdout_cka: float
    raw_holdout_cka: float
    alignment_gain: float
    coverage_ratio: float


@dataclass(frozen=True)
class DomainAlignmentReport:
    """Alignment generalization reports grouped by probe domain."""

    domain_reports: dict[str, AlignmentGeneralizationReport]
    domain_counts: dict[str, int]
    skipped_domains: list[str]


def _coverage_ratio(source: "Array", backend: "Backend") -> float:
    """Compute probe coverage as effective rank divided by ambient dimension."""
    n = int(source.shape[0])
    d = int(source.shape[1])
    if n <= 0 or d <= 0:
        return 0.0

    gram = backend.matmul(source, backend.transpose(source))
    eigvals, _ = backend.eigh(gram)
    zeros = backend.zeros_like(eigvals)
    eigvals = backend.maximum(eigvals, zeros)

    sum_vals = backend.sum(eigvals)
    sum_sq = backend.sum(eigvals * eigvals)
    backend.eval(sum_vals, sum_sq)

    sum_val = float(backend.to_scalar(sum_vals))
    sum_sq_val = float(backend.to_scalar(sum_sq))
    eps = division_epsilon(backend, gram)
    if sum_sq_val <= eps:
        return 0.0

    eff_rank = (sum_val * sum_val) / sum_sq_val
    return min(1.0, eff_rank / float(d))


def alignment_generalization_report(
    source: "Array",
    target: "Array",
    train_indices: list[int] | tuple[int, ...],
    holdout_indices: list[int] | tuple[int, ...],
    backend: "Backend | None" = None,
) -> AlignmentGeneralizationReport:
    """Compute alignment generalization metrics for explicit splits."""
    backend = backend or get_default_backend()

    source_arr = backend.array(source)
    target_arr = backend.array(target)
    backend.eval(source_arr, target_arr)

    if len(train_indices) < 2 or len(holdout_indices) < 2:
        raise ValueError("Need at least 2 samples in both train and holdout splits.")

    train_idx = backend.array(list(train_indices), dtype="int32")
    holdout_idx = backend.array(list(holdout_indices), dtype="int32")

    source_train = backend.take(source_arr, train_idx, axis=0)
    target_train = backend.take(target_arr, train_idx, axis=0)
    source_holdout = backend.take(source_arr, holdout_idx, axis=0)
    target_holdout = backend.take(target_arr, holdout_idx, axis=0)
    backend.eval(source_train, target_train, source_holdout, target_holdout)

    F = invariant_alignment(backend, source_train, target_train)
    aligned_train = backend.matmul(source_train, F)
    aligned_holdout = backend.matmul(source_holdout, F)
    backend.eval(aligned_train, aligned_holdout)

    train_cka = compute_linear_cka(aligned_train, target_train, backend)
    holdout_cka = compute_linear_cka(aligned_holdout, target_holdout, backend)
    raw_holdout_cka = compute_linear_cka(source_holdout, target_holdout, backend)

    alignment_gain = holdout_cka - raw_holdout_cka
    coverage_ratio = _coverage_ratio(source_train, backend)

    return AlignmentGeneralizationReport(
        train_samples=int(source_train.shape[0]),
        holdout_samples=int(source_holdout.shape[0]),
        train_cka=float(train_cka),
        holdout_cka=float(holdout_cka),
        raw_holdout_cka=float(raw_holdout_cka),
        alignment_gain=float(alignment_gain),
        coverage_ratio=float(coverage_ratio),
    )


def _even_odd_split(indices: list[int]) -> tuple[list[int], list[int]]:
    """Deterministic split for domain-scoped alignment."""
    train = indices[::2]
    holdout = indices[1::2]
    if len(train) < 2 or len(holdout) < 2:
        raise ValueError("Need at least 2 samples in both train and holdout splits.")
    return train, holdout


def alignment_generalization_by_domain(
    source: "Array",
    target: "Array",
    probes: list["AtlasProbeProtocol"],
    backend: "Backend | None" = None,
) -> DomainAlignmentReport:
    """Compute alignment generalization per probe domain."""
    backend = backend or get_default_backend()
    if len(probes) == 0:
        return DomainAlignmentReport(domain_reports={}, domain_counts={}, skipped_domains=[])

    domain_indices: dict[str, list[int]] = {}
    for idx, probe in enumerate(probes):
        key = enum_key(getattr(probe, "domain", "unknown"))
        domain_indices.setdefault(key, []).append(idx)

    domain_reports: dict[str, AlignmentGeneralizationReport] = {}
    domain_counts: dict[str, int] = {}
    skipped: list[str] = []

    for domain, indices in domain_indices.items():
        domain_counts[domain] = len(indices)
        try:
            train_idx, holdout_idx = _even_odd_split(indices)
        except ValueError:
            skipped.append(domain)
            continue

        domain_reports[domain] = alignment_generalization_report(
            source=source,
            target=target,
            train_indices=train_idx,
            holdout_indices=holdout_idx,
            backend=backend,
        )

    return DomainAlignmentReport(
        domain_reports=domain_reports,
        domain_counts=domain_counts,
        skipped_domains=skipped,
    )


__all__ = [
    "AlignmentGeneralizationReport",
    "DomainAlignmentReport",
    "alignment_generalization_by_domain",
    "alignment_generalization_report",
]
