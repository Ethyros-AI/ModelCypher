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

"""Tests for model architecture heuristics (VRAM estimation, batch size suggestion)."""

import pytest

from modelcypher.core.domain.training.checkpoint_models import ModelArchitectureConfig
from modelcypher.core.domain.training.model_architecture_heuristics import (
    ModelArchitectureHeuristics,
)


class TestConfigForParameterCount:
    """Tests for config_for_parameter_count() method."""

    def test_none_returns_7b_default(self):
        config = ModelArchitectureHeuristics.config_for_parameter_count(None)
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.vocabulary_size == 32000

    def test_7b_plus_returns_large_config(self):
        # 7B parameters
        config = ModelArchitectureHeuristics.config_for_parameter_count(7_000_000_000)
        assert config.hidden_size == 4096
        assert config.num_layers == 32

    def test_6b_boundary_returns_large_config(self):
        # Exactly 6B - should use large config
        config = ModelArchitectureHeuristics.config_for_parameter_count(6_000_000_000)
        assert config.hidden_size == 4096
        assert config.num_layers == 32

    def test_3b_to_6b_returns_medium_config(self):
        # 3B parameters
        config = ModelArchitectureHeuristics.config_for_parameter_count(3_000_000_000)
        assert config.hidden_size == 3072
        assert config.num_layers == 28
        assert config.num_heads == 24

    def test_2b_boundary_returns_medium_config(self):
        # Exactly 2B - should use medium config
        config = ModelArchitectureHeuristics.config_for_parameter_count(2_000_000_000)
        assert config.hidden_size == 3072
        assert config.num_layers == 28

    def test_under_2b_returns_small_config(self):
        # 1B parameters
        config = ModelArchitectureHeuristics.config_for_parameter_count(1_000_000_000)
        assert config.hidden_size == 2048
        assert config.num_layers == 16
        assert config.num_heads == 16

    def test_small_model_returns_small_config(self):
        # 100M parameters
        config = ModelArchitectureHeuristics.config_for_parameter_count(100_000_000)
        assert config.hidden_size == 2048
        assert config.num_layers == 16

    def test_all_configs_have_model_type(self):
        for param_count in [None, 100_000_000, 3_000_000_000, 7_000_000_000]:
            config = ModelArchitectureHeuristics.config_for_parameter_count(param_count)
            assert config.model_type == "simple_transformer"


class TestEstimateVramBytes:
    """Tests for estimate_vram_bytes() method."""

    @pytest.fixture
    def small_config(self):
        return ModelArchitectureConfig(
            model_type="simple_transformer",
            vocabulary_size=32000,
            hidden_size=2048,
            num_layers=16,
            num_heads=16,
        )

    @pytest.fixture
    def large_config(self):
        return ModelArchitectureConfig(
            model_type="simple_transformer",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

    def test_returns_positive_value(self, small_config):
        vram = ModelArchitectureHeuristics.estimate_vram_bytes(small_config)
        assert vram > 0

    def test_scales_with_batch_size(self, small_config):
        vram_bs1 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, batch_size=1
        )
        vram_bs4 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, batch_size=4
        )
        assert vram_bs4 > vram_bs1

    def test_scales_with_sequence_length(self, small_config):
        vram_seq512 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, sequence_length=512
        )
        vram_seq2048 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, sequence_length=2048
        )
        assert vram_seq2048 > vram_seq512

    def test_large_model_uses_more_vram(self, small_config, large_config):
        vram_small = ModelArchitectureHeuristics.estimate_vram_bytes(small_config)
        vram_large = ModelArchitectureHeuristics.estimate_vram_bytes(large_config)
        assert vram_large > vram_small

    def test_precision_affects_vram(self, small_config):
        vram_fp16 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, precision_bytes=2
        )
        vram_fp32 = ModelArchitectureHeuristics.estimate_vram_bytes(
            small_config, precision_bytes=4
        )
        assert vram_fp32 > vram_fp16

    def test_includes_optimizer_memory(self, small_config):
        # Optimizer uses 8 bytes/param for AdamW (first + second moments)
        # Should be significant portion of total
        vram = ModelArchitectureHeuristics.estimate_vram_bytes(small_config)
        # VRAM should be in reasonable range (not just model weights)
        # For a 2048 hidden size model, should be > 1GB
        assert vram > 1_000_000_000

    def test_reasonable_7b_estimate(self, large_config):
        # A 7B model with batch size 1 should need tens of GB
        vram = ModelArchitectureHeuristics.estimate_vram_bytes(
            large_config, batch_size=1, sequence_length=1024
        )
        # Should be > 20GB for 7B model with optimizer states
        assert vram > 20_000_000_000


class TestSuggestBatchSize:
    """Tests for suggest_batch_size() method."""

    @pytest.fixture
    def small_config(self):
        return ModelArchitectureConfig(
            model_type="simple_transformer",
            vocabulary_size=32000,
            hidden_size=2048,
            num_layers=16,
            num_heads=16,
        )

    def test_returns_minimum_1(self, small_config):
        # Even with tiny VRAM, should return at least 1
        batch_size = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=1
        )
        assert batch_size >= 1

    def test_increases_with_available_vram(self, small_config):
        bs_10gb = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=10_000_000_000
        )
        bs_50gb = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=50_000_000_000
        )
        assert bs_50gb >= bs_10gb

    def test_decreases_with_sequence_length(self, small_config):
        bs_short = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=20_000_000_000, sequence_length=512
        )
        bs_long = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=20_000_000_000, sequence_length=2048
        )
        assert bs_short >= bs_long

    def test_with_large_vram_allows_larger_batch(self, small_config):
        # 100GB should allow reasonable batch size
        batch_size = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=100_000_000_000
        )
        assert batch_size > 1

    def test_respects_max_batch_size_limit(self, small_config):
        # Even with huge VRAM, shouldn't exceed 128 (binary search upper bound)
        batch_size = ModelArchitectureHeuristics.suggest_batch_size(
            small_config, available_vram_bytes=1_000_000_000_000  # 1TB
        )
        assert batch_size <= 128
