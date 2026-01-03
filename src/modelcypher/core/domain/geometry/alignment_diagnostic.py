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
Alignment diagnostics.

CKA < 1.0 is a diagnostic signal, not an explanation.
It can mean boundary conditions are misaligned or that CKA is biased by
finite sampling (feature or input). This module turns the residual gap into
actionable geometric signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)


@dataclass(frozen=True)
class AlignmentSignal:
    """Signal from an alignment attempt (not a failure)."""

    dimension: int  # 1 = binary, 2 = vocabulary, 3+ = conceptual
    cka_achieved: float
    cka_target: float = 1.0
    gap: float = 0.0

    misaligned_anchors: list[str] = field(default_factory=list)
    anchor_labels: list[str] = field(default_factory=list)
    anchor_divergence: list[float] = field(default_factory=list)
    iteration: int = 0
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gap == 0.0:
            object.__setattr__(
                self,
                "gap",
                max(0.0, float(self.cka_target) - float(self.cka_achieved)),
            )

    @property
    def is_phase_locked(self) -> bool:
        """Check if alignment gap is within precision tolerance.

        Requires phase_tol in metadata - no arbitrary fallbacks.
        """
        phase_tol = self.metadata.get("phase_tol")
        if phase_tol is None:
            raise ValueError(
                "AlignmentSignal.is_phase_locked requires 'phase_tol' in metadata. "
                "Ensure phase_tol is set when creating the signal."
            )
        return self.gap <= float(phase_tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "cka_achieved": self.cka_achieved,
            "cka_target": self.cka_target,
            "gap": self.gap,
            "misaligned_anchors": list(self.misaligned_anchors),
            "anchor_labels": list(self.anchor_labels),
            "anchor_divergence": list(self.anchor_divergence),
            "iteration": self.iteration,
            "metadata": dict(self.metadata),
        }


def alignment_signal_from_matrices(
    source_matrix: "object",
    target_matrix: "object",
    labels: Sequence[str] | None = None,
    backend: "object | None" = None,
    dimension: int = 3,
    cka_achieved: float = 0.0,
    iteration: int = 0,
) -> AlignmentSignal:
    """Build an AlignmentSignal from paired anchor matrices."""
    b = backend or get_default_backend()
    phase_tol = machine_epsilon(b, source_matrix)
    if cka_achieved >= 1.0 - phase_tol:
        return AlignmentSignal(
            dimension=dimension,
            cka_achieved=float(cka_achieved),
            cka_target=1.0,
            iteration=iteration,
            metadata={"phase_tol": float(phase_tol)},
        )
    n_samples = b.shape(source_matrix)[0]
    labels = list(labels) if labels is not None else [f"sample:{i}" for i in range(n_samples)]

    # Per-anchor divergence (fallback to Gram-space when dimensions differ)
    if b.shape(source_matrix) != b.shape(target_matrix):
        source_gram = b.matmul(source_matrix, b.transpose(source_matrix))
        target_gram = b.matmul(target_matrix, b.transpose(target_matrix))
        b.eval(source_gram, target_gram)
        diff = source_gram - target_gram
        distances = b.norm(diff, axis=1)
    else:
        diff = source_matrix - target_matrix
        distances = b.norm(diff, axis=1)
    b.eval(distances)
    dist_list = b.tolist(distances)

    ranked = b.argsort(-distances)
    b.eval(ranked)
    ranked_idx = [int(x) for x in b.tolist(ranked)]
    misaligned = [labels[i] for i in ranked_idx]

    shape_mismatch = b.shape(source_matrix) != b.shape(target_matrix)

    # Rank diagnostics
    rank_source = _matrix_rank(source_matrix, b)
    rank_target = _matrix_rank(target_matrix, b)
    min_rank = min(b.shape(source_matrix)[0], b.shape(source_matrix)[1])

    # Scale diagnostics
    div_eps = division_epsilon(b, source_matrix)
    src_norm = b.mean(b.norm(source_matrix, axis=1))
    tgt_norm = b.mean(b.norm(target_matrix, axis=1))
    b.eval(src_norm, tgt_norm)
    src_norm_val = float(b.to_scalar(src_norm))
    tgt_norm_val = float(b.to_scalar(tgt_norm))
    scale_ratio = src_norm_val / (tgt_norm_val + div_eps)

    mean_divergence = sum(dist_list) / len(dist_list) if dist_list else 0.0
    max_divergence = max(dist_list) if dist_list else 0.0
    balance_ratio = max_divergence / (mean_divergence + div_eps)

    metadata = {
        "rank_source": float(rank_source),
        "rank_target": float(rank_target),
        "rank_gap": float(abs(rank_source - rank_target)),
        "scale_ratio": float(scale_ratio),
        "scale_gap": float(abs(scale_ratio - 1.0)),
        "max_divergence": max_divergence,
        "mean_divergence": mean_divergence,
        "balance_ratio": balance_ratio,
        "shape_mismatch": 1.0 if shape_mismatch else 0.0,
        "phase_tol": float(phase_tol),
    }

    return AlignmentSignal(
        dimension=dimension,
        cka_achieved=float(cka_achieved),
        cka_target=1.0,
        misaligned_anchors=misaligned,
        anchor_labels=labels,
        anchor_divergence=dist_list,
        iteration=iteration,
        metadata=metadata,
    )


def _matrix_rank(matrix: "object", backend: "object") -> int:
    """Compute effective rank using dtype-derived threshold."""
    gram = backend.matmul(matrix, backend.transpose(matrix))
    eigvals, _ = backend.eigh(gram)
    backend.eval(eigvals)
    max_val = backend.max(eigvals)
    eps = machine_epsilon(backend, gram)
    threshold = max_val * eps
    mask = eigvals > threshold
    rank = backend.sum(backend.astype(mask, "int32"))
    backend.eval(rank)
    return int(backend.to_scalar(rank))
