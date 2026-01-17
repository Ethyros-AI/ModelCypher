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
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    power_iteration_eigh,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_paired_distances,
)

_cache = ComputationCache.shared()


@dataclass(frozen=True)
class AlignmentSignal:
    """Signal from an alignment attempt (not a failure)."""

    dimension: int  # 1 = binary, 2 = vocabulary, 3+ = conceptual
    cka_achieved: float
    cka_target: float = 1.0
    gap: float = 0.0

    misaligned_anchors: tuple[str, ...] = field(default_factory=tuple)
    anchor_labels: tuple[str, ...] = field(default_factory=tuple)
    anchor_divergence: tuple[float, ...] = field(default_factory=tuple)
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
    n_samples = int(b.shape(source_matrix)[0])
    labels = list(labels) if labels is not None else [f"sample:{i}" for i in range(n_samples)]

    # Edge case: fewer than 3 samples - can't infer manifold structure
    # Return degenerate signal with chord-based distance approximation
    if n_samples < 3:
        sample_labels = list(labels) if labels is not None else [f"sample:{i}" for i in range(n_samples)]
        if n_samples == 0:
            return AlignmentSignal(
                dimension=dimension,
                cka_achieved=float(cka_achieved),
                cka_target=1.0,
                iteration=iteration,
                anchor_labels=tuple(sample_labels),
                anchor_divergence=(),
                metadata={"degenerate": True, "n_samples": 0, "phase_tol": float(phase_tol)},
            )
        # For 1-2 samples, use chord distance as approximation
        diff = source_matrix - target_matrix
        chord_dist = b.sqrt(b.sum(diff * diff, axis=1))
        b.eval(chord_dist)
        dist_list = b.tolist(chord_dist)
        mean_dist = float(b.to_scalar(b.mean(chord_dist)))
        max_dist = float(b.to_scalar(b.max(chord_dist)))
        return AlignmentSignal(
            dimension=dimension,
            cka_achieved=float(cka_achieved),
            cka_target=1.0,
            iteration=iteration,
            anchor_labels=tuple(sample_labels),
            anchor_divergence=tuple(dist_list),
            misaligned_anchors=tuple(sample_labels),  # All samples are "misaligned" for degenerate case
            metadata={
                "degenerate": True,
                "n_samples": n_samples,
                "mean_distance": mean_dist,
                "max_distance": max_dist,
                "phase_tol": float(phase_tol),
            },
        )

    # Per-anchor divergence: geodesic distance respects manifold curvature.
    # Chord distance ignores curvature in high dimensions.
    if b.shape(source_matrix) != b.shape(target_matrix):
        # Gram-space comparison when dimensions differ
        source_gram = _cache.get_or_compute_gram(source_matrix, b)
        target_gram = _cache.get_or_compute_gram(target_matrix, b)
        distances = geodesic_paired_distances(source_gram, target_gram, b)
    else:
        distances = geodesic_paired_distances(source_matrix, target_matrix, b)
    mean_dist = b.mean(distances)
    max_dist = b.max(distances)
    b.eval(distances, mean_dist, max_dist)
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
    src_norm = b.mean(geodesic_norms(source_matrix, b))
    tgt_norm = b.mean(geodesic_norms(target_matrix, b))
    b.eval(src_norm, tgt_norm)
    src_norm_val = float(b.to_scalar(src_norm))
    tgt_norm_val = float(b.to_scalar(tgt_norm))
    scale_ratio = src_norm_val / (tgt_norm_val + div_eps)

    mean_divergence = float(b.to_scalar(mean_dist)) if dist_list else 0.0
    max_divergence = float(b.to_scalar(max_dist)) if dist_list else 0.0
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
        misaligned_anchors=tuple(misaligned),
        anchor_labels=tuple(labels),
        anchor_divergence=tuple(dist_list),
        iteration=iteration,
        metadata=metadata,
    )


def _matrix_rank(matrix: "object", backend: "object") -> int:
    """Compute effective rank using dtype-derived threshold."""
    gram = _cache.get_or_compute_gram(matrix, backend)
    n_gram = int(gram.shape[0])
    eigvals, _ = power_iteration_eigh(backend, gram, k=n_gram)
    backend.eval(eigvals)
    max_val = backend.max(eigvals)
    eps = machine_epsilon(backend, gram)
    threshold = max_val * eps
    mask = eigvals > threshold
    rank = backend.sum(backend.astype(mask, "int32"))
    backend.eval(rank)
    return int(backend.to_scalar(rank))
