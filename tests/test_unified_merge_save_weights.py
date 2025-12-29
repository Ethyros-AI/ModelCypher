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

import pytest


class _MockLoader:
    def load_model_for_training(self, model_path, lora_config=None):  # pragma: no cover
        raise NotImplementedError

    def load_weights_as_numpy(self, model_path):  # pragma: no cover
        raise NotImplementedError

    def load_weights(self, model_path):  # pragma: no cover
        raise NotImplementedError


def test_save_weights_mixed_mlx_and_numpy_does_not_use_mx_save(tmp_path):
    mx = pytest.importorskip("mlx.core")

    import numpy as np

    from modelcypher.core.use_cases.unified_geometric_merge import UnifiedGeometricMerger

    merger = UnifiedGeometricMerger(model_loader=_MockLoader())

    # Mixed dicts previously triggered MLX std::bad_cast when saved via mx.save_safetensors.
    weights = {
        "a": mx.array([1.0], dtype=mx.float16),
        "b": np.array([2**32 - 1], dtype=np.uint32),
    }

    merger._save_weights(str(tmp_path), weights, "safetensors")

    assert (tmp_path / "model.safetensors").exists()

