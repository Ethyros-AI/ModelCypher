# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.geometric_lora import (
    LayerGeometry,
    apply_data_rank_ceiling,
    compute_adaptation_budget_ranks,
    compute_coupled_ranks,
    compute_geometric_rank,
    compute_layer_geometry,
    compute_layer_geometry_randomized,
    compute_per_layer_ranks,
    derive_lora_configs,
    estimate_nb_lora_parameter_count,
    select_target_modules,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geometry(
    key: str,
    *,
    tail_dims: int = 1,
    full_rank: int = 4,
    shannon_eff_rank: float = 2.0,
    sigma_max: float = 1.0,
    sigma_k: float = 0.5,
    spectral_gap: float = 0.1,
    shape: tuple[int, int] = (4, 4),
    recommended_rank: int = 1,
) -> LayerGeometry:
    """Build a LayerGeometry with sensible defaults for unit tests."""
    return LayerGeometry(
        layer_key=key,
        shape=shape,
        sigma_max=sigma_max,
        sigma_k=sigma_k,
        effective_rank=full_rank,
        full_rank=full_rank,
        decay_ratio=sigma_max / max(sigma_k, 1e-9),
        tail_dims=tail_dims,
        shannon_effective_rank=shannon_eff_rank,
        spectral_gap=spectral_gap,
        recommended_rank=recommended_rank,
    )


# ===========================================================================
# 1. compute_layer_geometry — Core SVD Analysis
# ===========================================================================


class TestComputeLayerGeometry:
    """Tests for compute_layer_geometry() with actual matrix SVD."""

    def test_identity_matrix(self, any_backend):
        b = any_backend
        W = b.eye(4)
        geom = compute_layer_geometry(W, "test.identity", b)

        assert geom.sigma_max == pytest.approx(1.0, abs=1e-5)
        assert geom.sigma_k > 0
        assert geom.full_rank == 4
        assert geom.effective_rank == 4
        # Identity: all SVs = 1 → Shannon eff rank = N
        assert geom.shannon_effective_rank == pytest.approx(4.0, abs=0.1)

    def test_diagonal_known_svs(self, any_backend):
        b = any_backend
        W = b.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.001],
        ])
        geom = compute_layer_geometry(W, "test.diag", b)

        assert geom.sigma_max == pytest.approx(10.0, abs=1e-4)
        assert geom.full_rank == 4
        # Energy concentrated in top SVs → Shannon eff rank < N
        assert geom.shannon_effective_rank < 3.0
        assert geom.tail_dims >= 1

    def test_rank_deficient_outer_product(self, any_backend):
        b = any_backend
        # Rank-1 matrix
        W = b.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        geom = compute_layer_geometry(W, "test.rank1", b)

        assert geom.full_rank == 2  # min(2, 3)
        assert geom.tail_dims >= 1
        assert geom.is_targetable is True
        assert geom.sigma_max > 0
        # One dominant SV → Shannon eff rank near 1
        assert geom.shannon_effective_rank < 1.5

    def test_near_zero_matrix(self, any_backend):
        b = any_backend
        W = b.array([[1e-10, 2e-10], [3e-10, 4e-10]])
        geom = compute_layer_geometry(W, "test.nearzero", b)

        assert geom.full_rank == 2
        assert geom.sigma_max >= 0
        assert isinstance(geom.tail_dims, int)

    def test_rectangular_tall(self, any_backend):
        b = any_backend
        W = b.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        geom = compute_layer_geometry(W, "test.tall", b)

        assert geom.shape == (6, 3)
        assert geom.full_rank == 3  # min(6, 3)

    def test_rectangular_wide(self, any_backend):
        b = any_backend
        W = b.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        ])
        geom = compute_layer_geometry(W, "test.wide", b)

        assert geom.shape == (3, 6)
        assert geom.full_rank == 3  # min(3, 6)

    def test_spectral_gap_nonnegative(self, any_backend):
        """Spectral gap should be non-negative at structural rank boundary."""
        b = any_backend
        W = b.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ])
        geom = compute_layer_geometry(W, "test.gap", b)

        assert geom.spectral_gap >= 0

    def test_shannon_eff_rank_flat_spectrum(self, any_backend):
        """Flat spectrum (identity) → Shannon eff rank ≈ N."""
        b = any_backend
        W = b.eye(4)
        geom = compute_layer_geometry(W, "test.flat", b)

        assert geom.shannon_effective_rank == pytest.approx(4.0, abs=0.1)

    def test_shannon_eff_rank_peaked_spectrum(self, any_backend):
        """Peaked spectrum → Shannon eff rank ≈ 1."""
        b = any_backend
        W = b.array([
            [100.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ])
        geom = compute_layer_geometry(W, "test.peaked", b)

        assert geom.shannon_effective_rank < 1.5


# ===========================================================================
# 2. select_target_modules — Module Selection
# ===========================================================================


class TestSelectTargetModules:

    def test_mix_targetable_and_non(self):
        geoms = {
            "layer.0": _geometry("layer.0", tail_dims=3),
            "layer.1": _geometry("layer.1", tail_dims=0),
            "layer.2": _geometry("layer.2", tail_dims=5),
        }
        targets = select_target_modules(geoms)
        assert sorted(targets) == ["layer.0", "layer.2"]

    def test_all_targetable(self):
        geoms = {
            "a": _geometry("a", tail_dims=2),
            "b": _geometry("b", tail_dims=1),
        }
        assert sorted(select_target_modules(geoms)) == ["a", "b"]

    def test_none_targetable(self):
        geoms = {
            "a": _geometry("a", tail_dims=0),
            "b": _geometry("b", tail_dims=0),
        }
        assert select_target_modules(geoms) == []


# ===========================================================================
# 3. compute_geometric_rank and compute_per_layer_ranks
# ===========================================================================


class TestGeometricRanks:

    def test_global_rank_is_min_tail_dims(self):
        geoms = {
            "a": _geometry("a", tail_dims=3),
            "b": _geometry("b", tail_dims=5),
            "c": _geometry("c", tail_dims=2),
        }
        rank = compute_geometric_rank(geoms, ["a", "b", "c"])
        assert rank == 2

    def test_global_rank_empty_raises(self):
        with pytest.raises(ValueError, match="No target modules found"):
            compute_geometric_rank({}, [])

    def test_global_rank_missing_keys_raises(self):
        geoms = {"a": _geometry("a", tail_dims=3)}
        with pytest.raises(ValueError, match="No target modules found"):
            compute_geometric_rank(geoms, ["missing"])

    def test_per_layer_ranks(self):
        geoms = {
            "a": _geometry("a", tail_dims=3),
            "b": _geometry("b", tail_dims=7),
        }
        ranks = compute_per_layer_ranks(geoms, ["a", "b"])
        assert ranks == {"a": 3, "b": 7}

    def test_per_layer_ranks_empty_raises(self):
        with pytest.raises(ValueError, match="No target modules provided"):
            compute_per_layer_ranks({}, [])

    def test_per_layer_ranks_missing_keys_raises(self):
        with pytest.raises(ValueError, match="No geometries found"):
            compute_per_layer_ranks({"a": _geometry("a")}, ["missing"])


# ===========================================================================
# 3b. compute_coupled_ranks — Cross-Projection Rank Coupling
# ===========================================================================


class TestCoupledRanks:
    """Tests for compute_coupled_ranks() — q_proj capped by k_proj capacity."""

    def test_q_capped_by_k(self):
        """q_proj rank should be min(tail_dims_q, tail_dims_k)."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight",
                tail_dims=1375, shape=(2048, 2048),
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight",
                tail_dims=200, shape=(512, 2048),
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 200
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 200

    def test_qwen_language_model_prefix_is_coupled(self):
        """Qwen MLX keys may include model.language_model.layers.N."""
        geoms = {
            "model.language_model.layers.19.self_attn.q_proj.weight": _geometry(
                "model.language_model.layers.19.self_attn.q_proj.weight",
                tail_dims=500,
                shape=(2048, 2048),
            ),
            "model.language_model.layers.19.self_attn.k_proj.weight": _geometry(
                "model.language_model.layers.19.self_attn.k_proj.weight",
                tail_dims=64,
                shape=(512, 2048),
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.language_model.layers.19.self_attn.q_proj.weight"] == 64
        assert ranks["model.language_model.layers.19.self_attn.k_proj.weight"] == 64

    def test_q_already_smaller_than_k_no_change(self):
        """When q tail_dims <= k tail_dims, no capping needed."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight",
                tail_dims=100, shape=(2048, 2048),
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight",
                tail_dims=200, shape=(512, 2048),
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 100
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 200

    def test_v_proj_and_o_proj_unchanged(self):
        """v_proj and o_proj are not coupled — keep their own tail_dims."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight", tail_dims=500,
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight", tail_dims=100,
            ),
            "model.layers.0.self_attn.v_proj.weight": _geometry(
                "model.layers.0.self_attn.v_proj.weight", tail_dims=300,
            ),
            "model.layers.0.self_attn.o_proj.weight": _geometry(
                "model.layers.0.self_attn.o_proj.weight", tail_dims=400,
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 100  # capped
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 100
        assert ranks["model.layers.0.self_attn.v_proj.weight"] == 300  # unchanged
        assert ranks["model.layers.0.self_attn.o_proj.weight"] == 400  # unchanged

    def test_mlp_projections_unchanged(self):
        """MLP projections are never coupled."""
        geoms = {
            "model.layers.0.mlp.up_proj.weight": _geometry(
                "model.layers.0.mlp.up_proj.weight", tail_dims=800,
            ),
            "model.layers.0.mlp.down_proj.weight": _geometry(
                "model.layers.0.mlp.down_proj.weight", tail_dims=600,
            ),
            "model.layers.0.mlp.gate_proj.weight": _geometry(
                "model.layers.0.mlp.gate_proj.weight", tail_dims=700,
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.mlp.up_proj.weight"] == 800
        assert ranks["model.layers.0.mlp.down_proj.weight"] == 600
        assert ranks["model.layers.0.mlp.gate_proj.weight"] == 700

    def test_multiple_layers_coupled_independently(self):
        """Each attention layer is coupled independently."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight", tail_dims=500,
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight", tail_dims=100,
            ),
            "model.layers.1.self_attn.q_proj.weight": _geometry(
                "model.layers.1.self_attn.q_proj.weight", tail_dims=300,
            ),
            "model.layers.1.self_attn.k_proj.weight": _geometry(
                "model.layers.1.self_attn.k_proj.weight", tail_dims=400,
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        # Layer 0: q capped at 100
        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 100
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 100
        # Layer 1: q already <= k, no capping
        assert ranks["model.layers.1.self_attn.q_proj.weight"] == 300
        assert ranks["model.layers.1.self_attn.k_proj.weight"] == 400

    def test_missing_k_proj_no_coupling(self):
        """If k_proj is not in target_modules, q_proj keeps its own rank."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight", tail_dims=500,
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 500

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No target modules provided"):
            compute_coupled_ranks({}, [])

    def test_equal_tail_dims_no_change(self):
        """When q and k have identical tail_dims, no capping needed."""
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight", tail_dims=200,
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight", tail_dims=200,
            ),
        }
        targets = list(geoms.keys())
        ranks = compute_coupled_ranks(geoms, targets)

        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 200
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 200


# ===========================================================================
# 4. compute_adaptation_budget_ranks — Canonical PiSSA Budget
# ===========================================================================


class TestAdaptationBudgetRanks:
    def test_budget_scales_knee_by_structural_slack(self):
        geoms = {
            "model.layers.0.self_attn.q_proj.weight": _geometry(
                "model.layers.0.self_attn.q_proj.weight",
                tail_dims=256,
                full_rank=1024,
                recommended_rank=128,
                shape=(1024, 1024),
            ),
            "model.layers.0.self_attn.k_proj.weight": _geometry(
                "model.layers.0.self_attn.k_proj.weight",
                tail_dims=128,
                full_rank=512,
                recommended_rank=64,
                shape=(512, 1024),
            ),
        }

        ranks = compute_adaptation_budget_ranks(geoms, list(geoms.keys()))

        # q budget: floor(128 * 256 / 1024) = 32, then coupled to k budget
        # k budget: floor(64 * 128 / 512) = 16
        assert ranks["model.layers.0.self_attn.q_proj.weight"] == 16
        assert ranks["model.layers.0.self_attn.k_proj.weight"] == 16

    def test_budget_floors_targetable_layer_at_one(self):
        geoms = {
            "model.layers.0.self_attn.v_proj.weight": _geometry(
                "model.layers.0.self_attn.v_proj.weight",
                tail_dims=1,
                full_rank=1024,
                recommended_rank=8,
                shape=(512, 1024),
            ),
        }

        ranks = compute_adaptation_budget_ranks(geoms, list(geoms.keys()))
        assert ranks["model.layers.0.self_attn.v_proj.weight"] == 1

    def test_budget_empty_raises(self):
        with pytest.raises(ValueError, match="No target modules provided"):
            compute_adaptation_budget_ranks({}, [])


# ===========================================================================
# 5. Data Ceiling + Parameter Count Helpers
# ===========================================================================


class TestDataRankCeiling:

    def test_caps_rank_by_n_samples(self):
        ranks = {"a": 900, "b": 120, "c": 0}
        capped = apply_data_rank_ceiling(ranks, n_samples=352)
        assert capped == {"a": 352, "b": 120, "c": 0}

    def test_non_positive_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            apply_data_rank_ceiling({"a": 3}, n_samples=0)


class TestEstimateNbLoraParameterCount:

    def test_counts_a_b_s_parameters(self):
        geoms = {
            "a": _geometry("a", shape=(10, 20)),  # out=10, in=20
            "b": _geometry("b", shape=(7, 5)),    # out=7, in=5
        }
        ranks = {"a": 3, "b": 2}
        # a: 3 * (20 + 10 + 1) = 93
        # b: 2 * (5 + 7 + 1) = 26
        assert estimate_nb_lora_parameter_count(geoms, ranks) == 119

    def test_ignores_missing_or_non_positive_ranks(self):
        geoms = {"a": _geometry("a", shape=(4, 4))}
        ranks = {"a": 0, "missing": 10}
        assert estimate_nb_lora_parameter_count(geoms, ranks) == 0


# ===========================================================================
# 5. derive_lora_configs — End-to-End Config Derivation
# ===========================================================================


class TestDeriveLoraConfigs:

    def test_adaptive_rank_per_layer(self):
        geoms = {
            "a": _geometry("a", tail_dims=3, full_rank=8,
                           shannon_eff_rank=5.0, shape=(8, 8), sigma_k=0.5),
            "b": _geometry("b", tail_dims=5, full_rank=10,
                           shannon_eff_rank=5.0, shape=(10, 10), sigma_k=0.3),
        }
        configs = derive_lora_configs(geoms, ["a", "b"], adaptive_rank=True)
        by_key = {c.layer_key: c for c in configs}

        assert by_key["a"].rank == 3
        assert by_key["b"].rank == 5

    def test_global_rank(self):
        geoms = {
            "a": _geometry("a", tail_dims=3, full_rank=8, shannon_eff_rank=5.0),
            "b": _geometry("b", tail_dims=5, full_rank=10, shannon_eff_rank=5.0),
        }
        configs = derive_lora_configs(geoms, ["a", "b"], adaptive_rank=False)
        by_key = {c.layer_key: c for c in configs}

        # Global rank = min(3, 5) = 3
        assert by_key["a"].rank == 3
        assert by_key["b"].rank == 3

    def test_config_fields_populated(self):
        geom = _geometry("layer.0", tail_dims=4, full_rank=8,
                         shannon_eff_rank=4.0, sigma_k=0.5, shape=(8, 16))
        configs = derive_lora_configs({"layer.0": geom}, ["layer.0"])

        assert len(configs) == 1
        c = configs[0]
        assert c.layer_key == "layer.0"
        assert c.rank == 4
        assert c.sigma_k == pytest.approx(0.5)
        assert c.in_features == 16
        assert c.out_features == 8
        assert c.dropout == 0.0

    def test_skips_missing_geometries(self):
        geom = _geometry("present", tail_dims=2)
        configs = derive_lora_configs(
            {"present": geom},
            ["present", "missing"],
        )
        assert len(configs) == 1
        assert configs[0].layer_key == "present"


# ===========================================================================
# 6. compute_layer_geometry_randomized — Randomized SVD Geometry
# ===========================================================================


class TestComputeLayerGeometryRandomized:
    """Tests for randomized SVD geometry analysis.

    Validates that randomized SVD produces geometry matching full SVD
    within sqrt(eps) relative error on all key quantities.
    """

    def test_matches_full_svd_identity(self, any_backend):
        """Identity matrix: randomized must match full SVD exactly."""
        b = any_backend
        W = b.eye(4)
        full = compute_layer_geometry(W, "test.id", b)
        rand = compute_layer_geometry_randomized(W, "test.id", b, seed=42)

        # Small matrix — randomized falls back to full SVD
        assert rand.sigma_max == pytest.approx(full.sigma_max, abs=1e-5)
        assert rand.tail_dims == full.tail_dims
        assert rand.shannon_effective_rank == pytest.approx(
            full.shannon_effective_rank, abs=0.1,
        )

    def test_matches_full_svd_diagonal(self, any_backend):
        """Diagonal matrix with known spectrum."""
        b = any_backend
        W = b.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.001],
        ])
        full = compute_layer_geometry(W, "test.diag", b)
        rand = compute_layer_geometry_randomized(W, "test.diag", b, seed=42)

        assert rand.sigma_max == pytest.approx(full.sigma_max, abs=1e-4)
        assert rand.tail_dims == full.tail_dims
        assert rand.sigma_k == pytest.approx(full.sigma_k, rel=1e-3)

    def test_matches_full_svd_rank_deficient(self, any_backend):
        """Rank-1 matrix."""
        b = any_backend
        W = b.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        full = compute_layer_geometry(W, "test.rank1", b)
        rand = compute_layer_geometry_randomized(W, "test.rank1", b, seed=42)

        assert rand.tail_dims == full.tail_dims
        assert rand.sigma_max == pytest.approx(full.sigma_max, abs=1e-4)
        assert rand.is_targetable == full.is_targetable

    def test_matches_full_svd_larger_matrix(self, any_backend):
        """Larger random matrix where randomized SVD is actually used."""
        b = any_backend
        # Create a matrix with known spectral decay: top SVs dominant, long tail
        import math
        n = 64
        # Diagonal with exponential decay
        diag_vals = [math.exp(-0.1 * i) for i in range(n)]
        rows = []
        for i in range(n):
            row = [0.0] * n
            row[i] = diag_vals[i]
            rows.append(row)
        W = b.array(rows)
        b.eval(W)

        full = compute_layer_geometry(W, "test.large", b)
        rand = compute_layer_geometry_randomized(
            W, "test.large", b, seed=42,
        )

        # sigma_max must match closely
        assert rand.sigma_max == pytest.approx(full.sigma_max, rel=1e-3)
        # sigma_k must match within sqrt(eps) relative error
        if full.sigma_k > 0:
            assert rand.sigma_k == pytest.approx(full.sigma_k, rel=2e-3)
        # tail_dims must match exactly
        assert rand.tail_dims == full.tail_dims
        # Shannon effective rank must be close
        assert rand.shannon_effective_rank == pytest.approx(
            full.shannon_effective_rank, rel=0.05,
        )

    def test_spectral_gap_nonnegative(self, any_backend):
        """Spectral gap from randomized SVD is non-negative."""
        b = any_backend
        W = b.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ])
        geom = compute_layer_geometry_randomized(W, "test.gap", b, seed=42)
        assert geom.spectral_gap >= 0

    def test_seed_reproducibility(self, any_backend):
        """Same seed produces identical results."""
        b = any_backend
        import math
        n = 64
        diag_vals = [math.exp(-0.1 * i) for i in range(n)]
        rows = []
        for i in range(n):
            row = [0.0] * n
            row[i] = diag_vals[i]
            rows.append(row)
        W = b.array(rows)
        b.eval(W)

        g1 = compute_layer_geometry_randomized(W, "test.repro", b, seed=123)
        g2 = compute_layer_geometry_randomized(W, "test.repro", b, seed=123)

        assert g1.sigma_max == pytest.approx(g2.sigma_max)
        assert g1.sigma_k == pytest.approx(g2.sigma_k)
        assert g1.tail_dims == g2.tail_dims
        assert g1.shannon_effective_rank == pytest.approx(
            g2.shannon_effective_rank,
        )

    def test_zero_matrix(self, any_backend):
        """Zero matrix returns valid geometry."""
        b = any_backend
        W = b.array([[0.0, 0.0], [0.0, 0.0]])
        geom = compute_layer_geometry_randomized(W, "test.zero", b, seed=42)
        assert geom.sigma_max == 0.0
        assert geom.full_rank == 2
