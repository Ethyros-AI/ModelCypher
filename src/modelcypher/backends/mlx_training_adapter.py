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

"""MLX adapter for geometric LoRA training via NB-LoRA.

This module is intentionally thin so it stays one-shot readable.
Implementation details are split across backend-only modules.
"""

from __future__ import annotations

from modelcypher.backends._mlx_training_adapter_core_mixin import _MLXTrainingAdapterCoreMixin
from modelcypher.backends._mlx_training_adapter_diagnostics_mixin import (
    _MLXTrainingAdapterDiagnosticsMixin,
)
from modelcypher.backends._mlx_training_adapter_train_mixin import _MLXTrainingAdapterTrainMixin
from modelcypher.backends.mlx_training_adapter_core import (
    EpochMetrics,
    NBLoRALinear,
    calibrate_geometric_weights,
    iterate_structured_batches,
    make_constrained_loss,
    make_geometric_reshaping_loss,
    make_outcome_loss,
    prepare_outcome_batches,
)


class MLXTrainingAdapter(
    _MLXTrainingAdapterCoreMixin,
    _MLXTrainingAdapterTrainMixin,
    _MLXTrainingAdapterDiagnosticsMixin,
):
    """Composed MLX training adapter implementation."""


__all__ = [
    "EpochMetrics",
    "MLXTrainingAdapter",
    "iterate_structured_batches",
    "make_geometric_reshaping_loss",
    "make_constrained_loss",
    "calibrate_geometric_weights",
    "NBLoRALinear",
    "make_outcome_loss",
    "prepare_outcome_batches",
]
