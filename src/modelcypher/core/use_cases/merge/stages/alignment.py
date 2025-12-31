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

import logging
from typing import TYPE_CHECKING

from ..data_models import LayerGeometry
from ..infrastructure import select_shared_full_rank_indices

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_compute_alignment(
    layer_geom: LayerGeometry,
    src_acts: list["Array"] | None,
    tgt_acts: list["Array"] | None,
    src_weights: dict[str, "Array"],
    tgt_weights: dict[str, "Array"],
    backend: "Backend",
    *,
    is_cross_architecture: bool = False,
) -> None:
    """STAGE 4: Compute alignment transformations."""
    if not src_acts or not tgt_acts:
        return

    n = min(len(src_acts), len(tgt_acts))
    if n < 2:
        logger.warning(
            "Layer %d: Exact kernel alignment needs >= 2 activation samples, got %d",
            layer_geom.layer_idx,
            n,
        )
        layer_geom.transform_requirements.append("PHASE_LOCK_INSUFFICIENT_SAMPLES")
        return

    # Exact kernel alignment from activations (CKA = 1.0)
    try:
        from modelcypher.core.domain.geometry.gram_aligner import GramAligner
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        src_stacked = backend.stack(src_acts[:n], axis=0)
        tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
        backend.eval(src_stacked, tgt_stacked)

        max_samples = min(
            int(src_stacked.shape[0]),
            max(2, int(src_stacked.shape[1]) - 1),
            max(2, int(tgt_stacked.shape[1]) - 1),
        )
        rank_indices = select_shared_full_rank_indices(
            src_stacked,
            tgt_stacked,
            max_samples,
            backend,
            center=True,
        )
        if len(rank_indices) < 2:
            raise RuntimeError(
                "Layer %d exact kernel alignment failed: rank-deficient activations (%d)."
                % (layer_geom.layer_idx, len(rank_indices))
            )
        if len(rank_indices) != int(src_stacked.shape[0]):
            idx_arr = backend.array(rank_indices)
            src_stacked = backend.take(src_stacked, idx_arr, axis=0)
            tgt_stacked = backend.take(tgt_stacked, idx_arr, axis=0)
            backend.eval(src_stacked, tgt_stacked)

        # Tolerance depends on architecture compatibility:
        # - Same architecture, same fine-tuning: CKA=1.0 achievable, use machine epsilon
        # - Same architecture, different fine-tuning: CKA~0.99999, use 1e-5
        # - Cross-architecture: CKA~0.99-0.999, use 1e-2 (relational structures differ)
        if is_cross_architecture:
            # Cross-architecture: relational structures fundamentally differ
            # Accept CKA > 0.99 as "aligned" - null-space grafting handles the rest
            precision_tol = 1e-2
        else:
            # Same architecture: can achieve very high alignment
            precision_tol = max(machine_epsilon(backend, src_stacked), 1e-5)
        aligner = GramAligner(
            backend=backend,
            max_iterations=5000,
            max_rounds=3,
            tolerance=precision_tol,
            regularization=0.0,
        )
        result = aligner.find_perfect_alignment(src_stacked, tgt_stacked)
        transform = backend.array(result.feature_transform)
        backend.eval(transform)

        layer_geom.procrustes_rotation = transform
        layer_geom.alignment_quality = result.achieved_cka
        if result.diagnostic is not None:
            layer_geom.transform_requirements.append(
                f"PHASE_LOCK_SIGNAL:{result.diagnostic.divergence_pattern}"
            )

        logger.debug(
            "Layer %d: exact_kernel_alignment_cka=%.8f (iters=%d, error=%.6f)",
            layer_geom.layer_idx,
            result.achieved_cka,
            result.iterations,
            result.alignment_error,
        )
        if result.achieved_cka < 1.0 - precision_tol:
            raise RuntimeError(
                "Layer %d exact kernel alignment failed (CKA=%.8f)"
                % (layer_geom.layer_idx, result.achieved_cka)
            )
    except Exception as e:
        logger.error(
            "Exact kernel alignment failed for layer %d: %s",
            layer_geom.layer_idx,
            e,
        )
        layer_geom.transform_requirements.append("PHASE_LOCK_FAILED")
        raise

    # tangent_space_alignment - local alignment
    try:
        pass
        # Would compute tangent space alignment here
    except Exception:
        pass
