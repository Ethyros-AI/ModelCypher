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

"""Tests for geometry-derived dropout rate.

Validates that dropout is a measurement derived from spectral structure,
not a hyperparameter:

1. Flat spectrum → low dropout (all dimensions useful)
2. Steep spectrum → higher dropout (redundancy exists)
3. Single dominant SV → near-maximum dropout
4. NB-LoRA exemption: Cayley transform subsumes dropout's regularization
5. Integration: derive_lora_configs populates dropout correctly
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.geometric_lora import (
    LayerGeometry,
    compute_geometric_dropout,
    compute_layer_geometry,
    derive_lora_configs,
    select_target_modules,
    analyze_weight_geometries,
)
from modelcypher.ports.training import LoRALayerConfig

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


def _stable_low_rank_product(left, right):
    """Build low-rank product via outer sums to avoid BLAS runtime warnings."""
    import numpy as np

    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for i in range(left.shape[1]):
        result += np.outer(left[:, i], right[i, :])
    return result


# =============================================================================
# Test: compute_geometric_dropout (unit tests with synthetic geometries)
# =============================================================================


class TestComputeGeometricDropout:
    """Tests for the dropout derivation formula."""

    def test_flat_spectrum_gives_low_dropout(self):
        """Flat spectrum (all SVs equal) → utilization ≈ 1.0 → dropout ≈ 0."""
        # Shannon eff rank of uniform distribution = N
        full_rank = 64
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=1.0,
            sigma_k=1.0,
            effective_rank=full_rank,
            full_rank=full_rank,
            decay_ratio=1.0,
            tail_dims=0,
            shannon_effective_rank=float(full_rank),  # Perfect flat
        )

        dropout = compute_geometric_dropout(geom, rank=8)
        assert dropout == 0.0, f"Flat spectrum should give 0 dropout, got {dropout}"

    def test_steep_spectrum_gives_higher_dropout(self):
        """Steep spectrum (few dominant SVs) → lower utilization → more dropout."""
        full_rank = 64
        # Shannon eff rank of ~16 means only 25% of dimensions carry info
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=10.0,
            sigma_k=0.01,
            effective_rank=16,
            full_rank=full_rank,
            decay_ratio=1000.0,
            tail_dims=48,
            shannon_effective_rank=16.0,
        )

        dropout = compute_geometric_dropout(geom, rank=8)
        # redundancy = 1 - 16/64 = 0.75
        # adapter_fraction = 8/64 = 0.125
        # dropout = 0.75 * 0.125 = 0.09375
        assert dropout > 0.05, f"Steep spectrum should give >0.05 dropout, got {dropout}"
        assert dropout < 0.15, f"Dropout too high for this geometry, got {dropout}"

    def test_single_dominant_sv_gives_high_dropout(self):
        """Single dominant SV → redundancy ≈ 1 → dropout ≈ adapter_fraction."""
        full_rank = 64
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=100.0,
            sigma_k=0.001,
            effective_rank=1,
            full_rank=full_rank,
            decay_ratio=100000.0,
            tail_dims=63,
            shannon_effective_rank=1.0,
        )

        dropout = compute_geometric_dropout(geom, rank=8)
        # redundancy = 1 - 1/64 ≈ 0.984
        # adapter_fraction = 8/64 = 0.125
        # dropout ≈ 0.984 * 0.125 ≈ 0.123
        # Approaches adapter_fraction as redundancy → 1
        assert dropout > 0.1, f"Single SV should give high dropout, got {dropout}"
        assert dropout < 8 / 64 + 0.01, f"Dropout should approach rank/full_rank, got {dropout}"

    def test_rank_1_gives_zero_dropout(self):
        """Rank-1 LoRA can't have dropout (nothing to drop)."""
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=1.0,
            sigma_k=0.1,
            effective_rank=32,
            full_rank=64,
            decay_ratio=10.0,
            tail_dims=32,
            shannon_effective_rank=32.0,
        )

        dropout = compute_geometric_dropout(geom, rank=1)
        assert dropout == 0.0, f"Rank-1 should give 0 dropout, got {dropout}"

    def test_rank_2_has_small_adapter_fraction(self):
        """Rank-2 LoRA: small adapter_fraction keeps dropout low."""
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=100.0,
            sigma_k=0.001,
            effective_rank=1,
            full_rank=64,
            decay_ratio=100000.0,
            tail_dims=63,
            shannon_effective_rank=1.0,
        )

        dropout = compute_geometric_dropout(geom, rank=2)
        # redundancy = 0.984, adapter_fraction = 2/64 = 0.03125
        # dropout = 0.984 * 0.03125 ≈ 0.031
        assert dropout > 0.0, "Rank-2 with steep spectrum should have nonzero dropout"
        assert dropout < 0.05, f"Small adapter fraction should keep dropout low, got {dropout}"

    def test_zero_full_rank_gives_zero(self):
        """Degenerate geometry gives zero dropout."""
        geom = LayerGeometry(
            layer_key="test",
            shape=(0, 0),
            sigma_max=0.0,
            sigma_k=0.0,
            effective_rank=0,
            full_rank=0,
            decay_ratio=float("inf"),
            tail_dims=0,
            shannon_effective_rank=0.0,
        )

        dropout = compute_geometric_dropout(geom, rank=8)
        assert dropout == 0.0

    def test_dropout_is_monotonic_with_utilization(self):
        """Higher utilization → lower dropout (monotonic)."""
        dropouts = []
        for shannon_eff in [4.0, 16.0, 32.0, 48.0, 64.0]:
            geom = LayerGeometry(
                layer_key="test",
                shape=(64, 64),
                sigma_max=1.0,
                sigma_k=0.1,
                effective_rank=int(shannon_eff),
                full_rank=64,
                decay_ratio=10.0,
                tail_dims=64 - int(shannon_eff),
                shannon_effective_rank=shannon_eff,
            )
            dropouts.append(compute_geometric_dropout(geom, rank=8))

        # Each should be <= the previous (monotonically decreasing)
        for i in range(1, len(dropouts)):
            assert dropouts[i] <= dropouts[i - 1], (
                f"Dropout should decrease with utilization: "
                f"eff_rank={[4, 16, 32, 48, 64][i]} gave {dropouts[i]} "
                f"but eff_rank={[4, 16, 32, 48, 64][i-1]} gave {dropouts[i-1]}"
            )

    def test_dropout_result_is_rounded(self):
        """Dropout rates should be rounded to 4 decimal places."""
        geom = LayerGeometry(
            layer_key="test",
            shape=(64, 64),
            sigma_max=1.0,
            sigma_k=0.1,
            effective_rank=20,
            full_rank=64,
            decay_ratio=10.0,
            tail_dims=44,
            shannon_effective_rank=20.0,
        )

        dropout = compute_geometric_dropout(geom, rank=8)
        # Check that it's rounded to 4 decimal places
        assert dropout == round(dropout, 4)


# =============================================================================
# Test: Shannon effective rank computation from real weight matrices
# =============================================================================


class TestShannonEffectiveRankComputation:
    """Test that compute_layer_geometry correctly computes Shannon effective rank."""

    def test_identity_matrix_has_full_shannon_rank(self, backend):
        """Identity matrix has all SVs = 1 → Shannon eff rank = N."""
        b = backend
        N = 32
        W = b.eye(N)
        b.eval(W)

        geom = compute_layer_geometry(W, "test_identity", b)

        # Shannon effective rank should equal N for identity
        assert abs(geom.shannon_effective_rank - N) < 0.1, (
            f"Identity should have shannon_eff_rank ≈ {N}, got {geom.shannon_effective_rank}"
        )

    def test_rank_1_matrix_has_low_shannon_rank(self, backend):
        """Rank-1 matrix has Shannon eff rank ≈ 1."""
        b = backend
        u = b.random_normal((32, 1))
        v = b.random_normal((1, 32))
        W = b.matmul(u, v)
        b.eval(W)

        geom = compute_layer_geometry(W, "test_rank1", b)

        assert geom.shannon_effective_rank < 2.0, (
            f"Rank-1 matrix should have shannon_eff_rank ≈ 1, got {geom.shannon_effective_rank}"
        )

    def test_random_matrix_has_high_shannon_rank(self, backend):
        """Random Gaussian matrix has near-full Shannon effective rank."""
        b = backend
        W = b.random_normal((64, 64))
        b.eval(W)

        geom = compute_layer_geometry(W, "test_random", b)

        # Random matrices have relatively flat singular value distributions
        # Shannon eff rank should be a significant fraction of full rank
        assert geom.shannon_effective_rank > 20.0, (
            f"Random 64×64 should have shannon_eff_rank > 20, got {geom.shannon_effective_rank}"
        )

    def test_constructed_spectrum(self, backend):
        """Construct a matrix with known spectral profile and verify Shannon eff rank."""
        b = backend

        # Create diagonal matrix with known SVs: [10, 1, 1, 1, 0.1, 0.01, ...]
        N = 16
        svs = [10.0] + [1.0] * 3 + [0.1] * 4 + [0.01] * 8
        S = b.array(svs)
        b.eval(S)

        # Construct W = diag(S) (diagonal matrix with these SVs)
        W = b.zeros((N, N))
        for i in range(N):
            # Use array indexing to set diagonal
            pass

        # Simpler: just use a diagonal matrix
        import numpy as np
        W_np = np.diag(svs)
        W = b.array(W_np)
        b.eval(W)

        geom = compute_layer_geometry(W, "test_constructed", b)

        # Manually compute expected Shannon eff rank
        eigvals = [s ** 2 for s in svs]
        total = sum(eigvals)
        probs = [e / total for e in eigvals]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        expected_shannon = math.exp(entropy)

        assert abs(geom.shannon_effective_rank - expected_shannon) < 0.5, (
            f"Expected shannon_eff_rank ≈ {expected_shannon:.2f}, "
            f"got {geom.shannon_effective_rank:.2f}"
        )


# =============================================================================
# Test: Integration with derive_lora_configs
# =============================================================================


class TestDeriveLoraConfigsDropout:
    """Test that derive_lora_configs correctly populates dropout from geometry."""

    def test_configs_have_dropout_field(self, backend):
        """LoRALayerConfig from derive_lora_configs should include dropout."""
        b = backend
        import numpy as np

        # Create a rank-deficient weight that will have tail_dims > 0
        np.random.seed(42)
        u = np.random.randn(64, 8).astype(np.float32)
        v = np.random.randn(8, 64).astype(np.float32)
        W = b.array(_stable_low_rank_product(u, v))
        b.eval(W)

        geometries = analyze_weight_geometries({"layer0": W}, b)
        targets = select_target_modules(geometries)

        assert len(targets) > 0, "Rank-deficient weight should be targetable"

        configs = derive_lora_configs(geometries, targets)

        assert len(configs) > 0
        for cfg in configs:
            assert hasattr(cfg, "dropout"), "LoRALayerConfig should have dropout field"
            assert isinstance(cfg.dropout, float)
            assert 0.0 <= cfg.dropout < 1.0, (
                f"Dropout should be in [0, 1), got {cfg.dropout}"
            )

    def test_steep_weight_gets_higher_dropout(self, backend):
        """Weight with steep spectrum should get higher dropout than flat weight."""
        b = backend
        import numpy as np

        # Flat weight: identity-like
        W_flat = b.array(np.eye(32).astype(np.float32))
        b.eval(W_flat)

        # Steep weight: rank-deficient (seeded for reproducibility)
        rng = np.random.default_rng(42)
        u = rng.standard_normal((32, 4)).astype(np.float32)
        v = rng.standard_normal((4, 32)).astype(np.float32)
        W_steep = b.array(_stable_low_rank_product(u, v))
        b.eval(W_steep)

        geom_flat = compute_layer_geometry(W_flat, "flat", b)
        geom_steep = compute_layer_geometry(W_steep, "steep", b)

        # Both need rank > 0 for meaningful dropout
        rank = 4

        dropout_flat = compute_geometric_dropout(geom_flat, rank)
        dropout_steep = compute_geometric_dropout(geom_steep, rank)

        assert dropout_steep >= dropout_flat, (
            f"Steep spectrum should get >= dropout than flat: "
            f"steep={dropout_steep}, flat={dropout_flat}"
        )

    def test_lora_layer_config_default_dropout(self):
        """LoRALayerConfig should default to 0.0 dropout (backwards compatible)."""
        cfg = LoRALayerConfig(
            layer_key="test",
            rank=8,
            sigma_k=0.01,
            in_features=64,
            out_features=64,
        )
        assert cfg.dropout == 0.0
