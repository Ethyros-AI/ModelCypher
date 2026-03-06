# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the geometry table emitter (LKM validation protocol)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.lkm.emit_geometry_table import build_geometry_table


class MockGeometry:
    """Mock geometry object matching LayerGeometry attribute interface."""

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        full_rank: int,
        effective_rank: int,
        tail_dims: int,
        sigma_max: float,
        sigma_k: float,
        spectral_gap: float,
        shannon_effective_rank: float,
    ):
        self.shape = shape
        self.full_rank = full_rank
        self.effective_rank = effective_rank
        self.tail_dims = tail_dims
        self.sigma_max = sigma_max
        self.sigma_k = sigma_k
        self.spectral_gap = spectral_gap
        self.shannon_effective_rank = shannon_effective_rank


class TestSchemaHasRequiredFields:
    """build_geometry_table with mock geometry objects returns all required layer fields."""

    def test_schema_has_required_fields(self):
        geometries = {
            "layers.0.self_attn.q_proj.weight": MockGeometry(
                shape=(512, 512),
                full_rank=512,
                effective_rank=480,
                tail_dims=32,
                sigma_max=1.5,
                sigma_k=0.01,
                spectral_gap=0.05,
                shannon_effective_rank=480.0,
            ),
        }

        result = build_geometry_table(
            model_id="test-model",
            model_family="test",
            dtype="bf16",
            geometries=geometries,
        )

        # Top-level required fields
        assert "model_id" in result
        assert "model_family" in result
        assert "dtype" in result
        assert "timestamp" in result
        assert "layers" in result
        assert "summary" in result

        # Layer-level required fields
        layer = result["layers"][0]
        required_layer_fields = [
            "layer_key",
            "shape",
            "full_rank",
            "effective_rank",
            "shannon_effective_rank",
            "tail_dims",
            "sigma_max",
            "sigma_k",
            "spectral_gap",
            "condition_number",
        ]
        for field in required_layer_fields:
            assert field in layer, f"Missing layer field: {field}"

        # Summary required fields
        summary = result["summary"]
        required_summary_fields = [
            "total_layers",
            "total_tail_dims",
            "mean_tail_dims",
            "layers_with_capacity",
            "layers_without_capacity",
        ]
        for field in required_summary_fields:
            assert field in summary, f"Missing summary field: {field}"

    def test_layer_values_match_input(self):
        geometries = {
            "layers.0.q_proj.weight": MockGeometry(
                shape=(256, 128),
                full_rank=128,
                effective_rank=100,
                tail_dims=20,
                sigma_max=2.0,
                sigma_k=0.05,
                spectral_gap=0.1,
                shannon_effective_rank=108.0,
            ),
        }

        result = build_geometry_table(
            model_id="test-model",
            model_family="test",
            dtype="float32",
            geometries=geometries,
        )

        layer = result["layers"][0]
        assert layer["layer_key"] == "layers.0.q_proj.weight"
        assert layer["shape"] == [256, 128]
        assert layer["full_rank"] == 128
        assert layer["effective_rank"] == 100
        assert layer["tail_dims"] == 20
        assert layer["sigma_max"] == 2.0
        assert layer["sigma_k"] == 0.05
        assert layer["spectral_gap"] == 0.1
        assert layer["shannon_effective_rank"] == 108.0
        assert layer["condition_number"] == pytest.approx(2.0 / 0.05)


class TestSummaryCounts:
    """layers_with_capacity and layers_without_capacity computed correctly."""

    def test_summary_counts(self):
        geometries = {
            "layer.0.weight": MockGeometry(
                shape=(512, 512),
                full_rank=512,
                effective_rank=480,
                tail_dims=32,  # has capacity
                sigma_max=1.5,
                sigma_k=0.01,
                spectral_gap=0.05,
                shannon_effective_rank=480.0,
            ),
            "layer.1.weight": MockGeometry(
                shape=(256, 256),
                full_rank=256,
                effective_rank=256,
                tail_dims=0,  # no capacity
                sigma_max=1.0,
                sigma_k=0.5,
                spectral_gap=0.0,
                shannon_effective_rank=256.0,
            ),
            "layer.2.weight": MockGeometry(
                shape=(1024, 512),
                full_rank=512,
                effective_rank=400,
                tail_dims=112,  # has capacity
                sigma_max=2.0,
                sigma_k=0.02,
                spectral_gap=0.1,
                shannon_effective_rank=400.0,
            ),
        }

        result = build_geometry_table(
            model_id="test-model",
            model_family="test",
            dtype="bf16",
            geometries=geometries,
        )

        summary = result["summary"]
        assert summary["total_layers"] == 3
        assert summary["total_tail_dims"] == 32 + 0 + 112
        assert summary["mean_tail_dims"] == pytest.approx((32 + 0 + 112) / 3)
        assert summary["layers_with_capacity"] == 2
        assert summary["layers_without_capacity"] == 1


class TestConditionNumberInfinite:
    """sigma_k=0 yields condition_number=inf."""

    def test_condition_number_infinite(self):
        geometries = {
            "layer.0.weight": MockGeometry(
                shape=(512, 512),
                full_rank=512,
                effective_rank=100,
                tail_dims=412,
                sigma_max=1.5,
                sigma_k=0.0,  # zero => infinite condition number
                spectral_gap=0.05,
                shannon_effective_rank=100.0,
            ),
        }

        result = build_geometry_table(
            model_id="test-model",
            model_family="test",
            dtype="bf16",
            geometries=geometries,
        )

        layer = result["layers"][0]
        assert math.isinf(layer["condition_number"])
        assert layer["condition_number"] > 0  # positive infinity
