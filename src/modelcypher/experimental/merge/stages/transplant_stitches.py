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

"""Stitch construction helpers for transplant stage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .stages.transplant_helpers import (
    _geodesic_pinv,
    _promote_precision,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def compute_composite_stitches(
    *,
    transforms_map: dict[int, Any] | None,
    desc: str,
    backend: "Backend",
    layer_mapping: dict[int, int] | None,
    layer_scale_ratios: dict[int, float] | None = None,
) -> dict[int, dict[int, tuple[Any, Any]]]:
    """Compute stitch matrices (P, Q) for each layer, supporting composite sources."""
    result_stitches: dict[int, dict[int, tuple[Any, Any]]] = {}
    if not transforms_map:
        return result_stitches

    logger.info(
        "%s: Processing stitches for %d target layers...",
        desc,
        len(transforms_map),
    )

    for tgt_layer, data in transforms_map.items():
        try:
            src_map = data if isinstance(data, dict) else {layer_mapping.get(tgt_layer, 0): data}
            sorted_srcs = sorted(src_map.keys())

            parts = []
            dims = []
            for s in sorted_srcs:
                arr = _promote_precision(src_map[s], backend)
                parts.append(arr)
                dims.append(arr.shape[0])

            F = backend.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            backend.eval(F)

            stitch_output_full = backend.transpose(F)
            F_pinv = _geodesic_pinv(backend, F)
            stitch_input_full = backend.transpose(F_pinv)
            backend.eval(stitch_output_full, stitch_input_full)

            # NOTE: scale_ratio is applied in transplant_weight_processor.py after stitching.
            # For cross-dimensional merges (d_s != d_t), F = pinv(A_s) @ A_t is NOT
            # norm-preserving. scale_ratio = ||target|| / ||source @ F|| compensates.
            # The correction is applied to the stitched weights, not the stitch matrices.
            if layer_scale_ratios and tgt_layer in layer_scale_ratios:
                sr = layer_scale_ratios[tgt_layer]
                eps = float(machine_epsilon(backend, F))
                if abs(sr - 1.0) > eps:
                    logger.debug(
                        "%s layer %d: scale_ratio=%.4f (not applied to weight stitch)",
                        desc,
                        tgt_layer,
                        sr,
                    )

            stitches: dict[int, tuple[Any, Any]] = {}
            idx_out = 0
            idx_in = 0

            for s, d in zip(sorted_srcs, dims):
                p_slice = stitch_output_full[:, idx_out : idx_out + d]
                q_slice = stitch_input_full[idx_in : idx_in + d, :]

                stitches[s] = (p_slice, q_slice)
                idx_out += d
                idx_in += d

            result_stitches[tgt_layer] = stitches

        except Exception as e:
            logger.warning(
                "Failed to process stitches for %s layer %d: %s",
                desc,
                tgt_layer,
                e,
            )

    return result_stitches
