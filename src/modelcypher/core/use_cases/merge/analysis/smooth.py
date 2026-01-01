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

from ..data_models import MergeGeometry


def stage_smooth_alphas(geometry: MergeGeometry) -> None:
    """STAGE 7: Smooth alphas across layers."""
    from modelcypher.core.domain.geometry.alpha_smoothing import (
        AlphaSmoothingConfig,
        gaussian_smooth_alpha_profile,
    )

    layer_alphas = {
        idx: lg.base_alpha
        for idx, lg in geometry.layer_geometries.items()
    }

    if len(layer_alphas) > 2:
        import math

        window = max(1, int(round(math.sqrt(len(layer_alphas)) / 2)))
        sigma = max(1.0, window / 2.0)
        config = AlphaSmoothingConfig.with_parameters(
            smoothing_window=window,
            sigma=sigma,
        )
        smoothed = gaussian_smooth_alpha_profile(layer_alphas, config)

        for idx, alpha in smoothed.items():
            if idx in geometry.layer_geometries:
                geometry.layer_geometries[idx].smoothed_alpha = alpha
