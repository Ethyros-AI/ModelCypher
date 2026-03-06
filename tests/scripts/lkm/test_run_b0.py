# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the B0 training harness (LKM validation protocol)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.lkm.run_b0 import make_b0_config, make_run_id


class TestLoraScaleIsOne:
    """scale = alpha/rank = r/r = 1.0 is the paper-matched invariant."""

    def test_lora_scale_is_one(self):
        config = make_b0_config(r_cap=16)
        assert config["lora_parameters"]["scale"] == 1.0


class TestLoraScaleInvariantToRank:
    """scale=1.0 must hold for all ranks, not just the default."""

    @pytest.mark.parametrize("rank", [2, 4, 8, 16, 64, 256, 1024])
    def test_lora_scale_invariant_to_rank(self, rank: int):
        config = make_b0_config(r_cap=rank)
        assert config["lora_parameters"]["scale"] == 1.0


class TestDefaultTrainingParams:
    """Paper-matched defaults: batch=8, iters=1500, lr=5e-4."""

    def test_default_training_params(self):
        config = make_b0_config(r_cap=16)
        assert config["batch_size"] == 8
        assert config["iters"] == 1500
        assert config["learning_rate"] == 5e-4


class TestRunIdFormat:
    """Run ID encodes arm, rank, and token count."""

    def test_run_id_format(self):
        run_id = make_run_id("B0", 16, 4000)
        assert "B0" in run_id
        assert "r16" in run_id
        assert "4000tok" in run_id

    def test_run_id_exact(self):
        run_id = make_run_id("B0", 16, 4000)
        assert run_id == "B0_r16_4000tok"

    def test_run_id_different_params(self):
        run_id = make_run_id("B1", 64, 8000)
        assert run_id == "B1_r64_8000tok"


class TestMakeB0ConfigLoraParams:
    """LoRA parameters sub-dict has exactly the required keys."""

    def test_lora_parameters_keys(self):
        config = make_b0_config(r_cap=8)
        lora = config["lora_parameters"]
        assert lora["rank"] == 8
        assert lora["scale"] == 1.0
        assert lora["dropout"] == 0.0

    def test_config_top_level_keys(self):
        config = make_b0_config(r_cap=16)
        assert "batch_size" in config
        assert "iters" in config
        assert "learning_rate" in config
        assert "lora_parameters" in config


class TestMakeB0ConfigOverrides:
    """Custom batch_size, iters, learning_rate are respected."""

    def test_custom_params(self):
        config = make_b0_config(r_cap=32, batch_size=4, iters=500, learning_rate=1e-3)
        assert config["batch_size"] == 4
        assert config["iters"] == 500
        assert config["learning_rate"] == 1e-3
        # scale invariant still holds
        assert config["lora_parameters"]["scale"] == 1.0
        assert config["lora_parameters"]["rank"] == 32
