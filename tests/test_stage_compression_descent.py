# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for stage_compression_descent contracts.

Covers:
  CD1 — Empty transmission_layers returns empty result with no errors.
  CD2 — Weights in layers with no activations are counted as skipped.
  CD3 — Non-2D weights (bias tensors) are skipped.
  CD4 — Dimension mismatch (weight in_dim ≠ act_dim) is skipped.
  CD5 — CKA ≥ 1 − √eps after compression (lossless on manifold) for full-rank input.
  CD6 — Low-rank activations produce actual compression (ratio < 1.0) with CKA invariant held.
  CD7 — apply_compression_descent_to_weights substitutes compressed weights, leaves others.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.experimental.merge.stages.compression_descent import (
    CompressionDescentResult,
    apply_compression_descent_to_weights,
    stage_compression_descent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_layer_fn(key: str) -> int | None:
    """Extract layer index from keys of the form 'model.<idx>.weight'."""
    parts = key.split(".")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _low_rank_acts(b, n_samples: int, dim: int, rank: int):
    """Activations spanning only a `rank`-dimensional subspace of `dim`."""
    Z = b.random_normal((n_samples, rank))
    B = b.random_normal((rank, dim))
    acts = b.matmul(Z, B)
    b.eval(acts)
    return acts


# ---------------------------------------------------------------------------
# CD1: Empty transmission_layers
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_transmission_layers_returns_empty_result(self):
        b = get_default_backend()
        W = b.random_normal((32, 32))
        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[],
            layer_activations={0: b.random_normal((8, 32))},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )
        assert result.weights_compressed == 0
        assert result.weights_skipped == 0
        assert result.compressed_weights == {}


# ---------------------------------------------------------------------------
# CD2: Missing activations → skipped
# ---------------------------------------------------------------------------

class TestSkipConditions:
    def test_missing_activations_skip_all_layer_weights(self):
        b = get_default_backend()
        W = b.random_normal((32, 32))
        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[0],
            layer_activations={},  # no activations for layer 0
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )
        assert result.weights_skipped == 1
        assert result.weights_compressed == 0

    # CD3: Non-2D weight skipped
    def test_non_2d_weight_is_skipped(self):
        """1-D bias tensor: not a weight matrix, must be skipped."""
        b = get_default_backend()
        bias = b.random_normal((32,))  # 1-D
        acts = b.random_normal((8, 32))
        result = stage_compression_descent(
            merged_weights={"model.0.weight": bias},
            transmission_layers=[0],
            layer_activations={0: acts},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )
        assert result.weights_skipped == 1
        assert result.weights_compressed == 0
        assert "model.0.weight" not in result.compressed_weights

    # CD4: Dimension mismatch skipped
    def test_dimension_mismatch_is_skipped(self):
        """Weight in_dim=64 but activations act_dim=32 → skip."""
        b = get_default_backend()
        W = b.random_normal((64, 64))   # in_dim = 64
        acts = b.random_normal((8, 32))  # act_dim = 32 ≠ 64
        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[0],
            layer_activations={0: acts},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )
        assert result.weights_skipped == 1
        assert result.weights_compressed == 0

    def test_skip_weights_parameter_excludes_weight(self):
        """Weights in skip_weights set are counted as skipped."""
        b = get_default_backend()
        W = b.random_normal((32, 32))
        acts = b.random_normal((8, 32))
        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[0],
            layer_activations={0: acts},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
            skip_weights={"model.0.weight"},
        )
        assert result.weights_skipped == 1
        assert result.weights_compressed == 0


# ---------------------------------------------------------------------------
# CD5: CKA invariant — full-rank input
# ---------------------------------------------------------------------------

class TestCompressionContract:
    def test_cka_invariant_full_rank_input(self):
        """Full-rank activations: compression is lossless (CKA ≥ 1 − √eps).

        n_samples=128 > in_dim=64 ensures activations have full column-rank 64.
        """
        b = get_default_backend()
        in_dim = 64
        W = b.random_normal((in_dim, in_dim))
        acts = b.random_normal((128, in_dim))  # 128 samples → rank = min(128,64) = 64

        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[0],
            layer_activations={0: acts},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )

        assert result.weights_compressed == 1
        assert "model.0.weight" in result.cka_validations

        cka = result.cka_validations["model.0.weight"]
        eps = machine_epsilon(b, b.array([1.0]))
        threshold = 1.0 - float(sqrt_scalar(eps, b))
        assert cka >= threshold, (
            f"CKA={cka:.8f} below lossless threshold {threshold:.8f} "
            f"(1 - sqrt(float32_eps))"
        )

    # CD6: CKA invariant — low-rank input with actual compression
    def test_cka_invariant_low_rank_input(self):
        """Rank-4 activations: compression_ratio < 1.0 AND CKA ≥ 1 − √eps.

        Activations live in a 4-dimensional subspace of a 64-dimensional space.
        Compression should discard the 60 null-space directions, giving
        compression_ratio = factorized_params / original_params < 1.0.
        CKA must still hold (lossless on the manifold by construction).
        """
        b = get_default_backend()
        in_dim = 64
        intrinsic_rank = 4
        W = b.random_normal((in_dim, in_dim))
        acts = _low_rank_acts(b, n_samples=32, dim=in_dim, rank=intrinsic_rank)

        result = stage_compression_descent(
            merged_weights={"model.0.weight": W},
            transmission_layers=[0],
            layer_activations={0: acts},
            extract_layer_index_fn=_extract_layer_fn,
            backend=b,
        )

        assert result.weights_compressed == 1, "Expected compression to succeed"

        cka = result.cka_validations["model.0.weight"]
        eps = machine_epsilon(b, b.array([1.0]))
        threshold = 1.0 - float(sqrt_scalar(eps, b))
        assert cka >= threshold, (
            f"CKA={cka:.8f} below lossless threshold {threshold:.8f}"
        )

        ratio = result.compression_ratios["model.0.weight"]
        assert ratio < 1.0, (
            f"Expected compression_ratio < 1.0 for rank-{intrinsic_rank} activations "
            f"in {in_dim}-dim space, got {ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# CD7: apply_compression_descent_to_weights
# ---------------------------------------------------------------------------

class TestApplyCompression:
    def test_replaces_compressed_and_leaves_others_unchanged(self):
        b = get_default_backend()
        W_orig = b.random_normal((8, 8))
        W_bias = b.random_normal((8,))   # stays unchanged
        W_comp = b.random_normal((4, 8)) # replacement

        compression_result = CompressionDescentResult(
            compressed_weights={"model.0.weight": W_comp},
        )
        merged = {
            "model.0.weight": W_orig,
            "model.0.bias": W_bias,
        }

        updated = apply_compression_descent_to_weights(merged, compression_result)

        # Compressed weight replaced
        assert updated["model.0.weight"] is W_comp
        # Untouched weight preserved
        assert updated["model.0.bias"] is W_bias
        # Original dict not mutated
        assert merged["model.0.weight"] is W_orig

    def test_empty_compression_result_leaves_all_weights(self):
        b = get_default_backend()
        W = b.random_normal((8, 8))
        merged = {"model.0.weight": W}

        updated = apply_compression_descent_to_weights(merged, CompressionDescentResult())
        assert updated["model.0.weight"] is W
