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

"""Alignment metric helpers for transplant stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_paired_distances,
)
from modelcypher.core.domain.merging.exceptions import DimensionMismatchError

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _compute_alignment_metrics(
    core_acts: "Array",
    weight_before: "Array",
    weight_after: "Array",
    weight_source: "Array",
    backend: "Backend",
) -> dict[str, float]:
    """Measure core distance shift toward the source for a single weight."""
    from modelcypher.core.domain.geometry.cka import compute_cka

    b = backend

    if int(core_acts.shape[1]) != int(weight_before.shape[1]):
        raise DimensionMismatchError(
            f"Alignment metrics require matching input dims; "
            f"acts={int(core_acts.shape[1])}, weight_in={int(weight_before.shape[1])}"
        )

    output_before = b.matmul(core_acts, b.transpose(weight_before))
    output_after = b.matmul(core_acts, b.transpose(weight_after))
    output_source = b.matmul(core_acts, b.transpose(weight_source))
    b.eval(output_before, output_after, output_source)

    geo_distances_before = geodesic_paired_distances(
        output_before, output_source, b, use_cache=False
    )
    geo_distances_after = geodesic_paired_distances(
        output_after, output_source, b, use_cache=False
    )
    dist_before_arr = geodesic_norms(
        b.reshape(geo_distances_before, (1, -1)), b, use_cache=False
    )
    dist_after_arr = geodesic_norms(
        b.reshape(geo_distances_after, (1, -1)), b, use_cache=False
    )
    b.eval(dist_before_arr, dist_after_arr)

    dist_before = float(b.to_scalar(dist_before_arr))
    dist_after = float(b.to_scalar(dist_after_arr))

    eps = float(machine_epsilon(b, weight_before))
    if dist_before > eps:
        core_distance_reduction = (dist_before - dist_after) / dist_before
    else:
        core_distance_reduction = 0.0

    cka_before = compute_cka(output_before, output_source, backend=b)
    cka_after = compute_cka(output_after, output_source, backend=b)

    return {
        "core_dist_to_source_before": dist_before,
        "core_dist_to_source_after": dist_after,
        "core_distance_reduction": core_distance_reduction,
        "cka_before": cka_before.best,
        "cka_after": cka_after.best,
    }
