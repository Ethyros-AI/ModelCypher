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

import json
import struct
from pathlib import Path

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.model_merge_service import ModelMergeService


def _write_bf16_safetensors(path: Path, tensor_name: str, values) -> None:
    import numpy as np
    backend = get_default_backend()
    # Convert backend array to numpy if needed
    if hasattr(values, 'shape'):
        values_np = backend.to_numpy(values) if not isinstance(values, np.ndarray) else values
    else:
        values_np = values

    values_f32 = np.array(values_np, dtype=np.float32)
    bf16 = (values_f32.view(np.uint32) >> 16).astype(np.uint16)
    data = bf16.tobytes()

    header = {
        tensor_name: {
            "dtype": "BF16",
            "shape": list(values_f32.shape),
            "data_offsets": [0, len(data)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(data)


def test_load_safetensors_bf16_without_torch(tmp_path: Path) -> None:
    import numpy as np
    backend = get_default_backend()
    weight_file = tmp_path / "bf16.safetensors"
    values_data = [[1.0, 2.0], [3.0, 4.0]]
    values = backend.array(values_data, dtype=backend.float32)
    values_np = backend.to_numpy(values)
    _write_bf16_safetensors(weight_file, "layer.weight", values_np)

    weights = ModelMergeService._load_safetensors(weight_file)
    assert "layer.weight" in weights
    loaded = weights["layer.weight"]

    assert loaded.dtype == np.float32
    assert loaded.shape == values_np.shape
    assert np.allclose(loaded, values_np)
