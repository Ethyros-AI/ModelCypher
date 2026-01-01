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
from typing import TYPE_CHECKING, Any

from ..models import MergeGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_probe_fingerprint(
    geometry: MergeGeometry,
    source_activations: dict[int, list["Array"]],
    target_activations: dict[int, list["Array"]],
    tokenizer: Any | None,
    backend: "Backend",
) -> None:
    """STAGE 1: Probe and fingerprint models."""
    # cka - compute overall CKA
    from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka

    # Get all activations stacked
    src_all = []
    tgt_all = []
    for layer_idx in sorted(source_activations.keys()):
        if layer_idx in target_activations:
            src_acts = source_activations[layer_idx]
            tgt_acts = target_activations[layer_idx]
            if src_acts and tgt_acts:
                n = min(len(src_acts), len(tgt_acts))
                for i in range(n):
                    src_all.append(src_acts[i])
                    tgt_all.append(tgt_acts[i])

    if src_all and tgt_all:
        try:
            src_stacked = backend.stack(src_all, axis=0)
            tgt_stacked = backend.stack(tgt_all, axis=0)
            backend.eval(src_stacked, tgt_stacked)
            cka_result = compute_cka(
                src_stacked,
                tgt_stacked,
                estimator=HSICEstimator.AUTO,
                feature_bias_correction=True,
            )
            if cka_result.is_valid:
                geometry.overall_cka = (
                    cka_result.cka_corrected
                    if cka_result.cka_corrected is not None
                    else cka_result.cka
                )
            else:
                geometry.overall_cka = 0.0
            logger.info("STAGE 1: Overall CKA = %.4f", geometry.overall_cka)
        except Exception as e:
            logger.warning("STAGE 1: CKA computation failed: %s", e)

    # topological_fingerprint - compute topological signature
    try:
        from modelcypher.core.domain.geometry.topological_fingerprint import (
            TopologicalFingerprint,  # noqa: F401 - for feature detection
        )
        # Would compute persistent homology here
    except ImportError:
        pass
