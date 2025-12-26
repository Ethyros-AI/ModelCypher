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

"""Integration tests for Training adapters (requires MLX)."""

from unittest.mock import MagicMock, patch
import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    nn = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.adapters.model_loader import load_model_for_training
from modelcypher.core.domain.training.lora_mlx import LoRAConfig


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10)]
        self.embed_tokens = nn.Linear(10, 10)

    def __call__(self, x):
        return x


@patch("modelcypher.adapters.model_loader.mlx_lm_load")
def test_model_loader_lora_injection(mock_load):
    """Test that model loader correctly injects LoRA and freezes base weights."""
    mock_model = MockModel()
    # Add some dummy submodules to simulate a real model structure
    mock_model.q_proj = nn.Linear(10, 10)
    mock_model.v_proj = nn.Linear(10, 10)

    mock_tokenizer = MagicMock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])

    with patch("modelcypher.core.domain.training.lora_mlx.logger"):
        model, tokenizer = load_model_for_training("dummy-path", config)

    assert tokenizer == mock_tokenizer
    # Check that q_proj and v_proj are now LoRALinear (contains lora_a/b)
    assert hasattr(model.q_proj, "lora_a")
    assert hasattr(model.v_proj, "lora_a")

    # Check freezing using MLX idiomatic way
    from mlx.utils import tree_flatten

    # Get all names of trainable parameters
    trainable_params = tree_flatten(model.trainable_parameters())
    trainable_names = {name for name, _ in trainable_params}

    # Get all parameters
    all_params = tree_flatten(model.parameters())

    for name, _ in all_params:
        if "lora" in name.lower():
            assert name in trainable_names
        else:
            assert name not in trainable_names
