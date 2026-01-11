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

"""Tests for model config field resolution helpers."""

from __future__ import annotations

from modelcypher.utils.model_config import (
    resolve_hidden_size,
    resolve_num_attention_heads,
    resolve_num_hidden_layers,
    resolve_vocab_size,
)


def test_resolve_fields_prefers_top_level() -> None:
    config = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "text_config": {
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 6,
            "vocab_size": 16000,
        },
    }

    assert resolve_hidden_size(config) == 256
    assert resolve_num_attention_heads(config) == 8
    assert resolve_num_hidden_layers(config) == 12
    assert resolve_vocab_size(config) == 32000


def test_resolve_fields_from_nested_aliases() -> None:
    config = {
        "language_model": {
            "n_embd": "512",
            "n_head": "16",
            "n_layer": "24",
            "vocab_size": 50000,
        }
    }

    assert resolve_hidden_size(config) == 512
    assert resolve_num_attention_heads(config) == 16
    assert resolve_num_hidden_layers(config) == 24
    assert resolve_vocab_size(config) == 50000


def test_resolve_fields_returns_zero_when_missing() -> None:
    config = {"model_type": "unknown"}

    assert resolve_hidden_size(config) == 0
    assert resolve_num_attention_heads(config) == 0
    assert resolve_num_hidden_layers(config) == 0
    assert resolve_vocab_size(config) == 0
