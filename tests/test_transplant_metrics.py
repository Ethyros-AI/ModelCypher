# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for transplant_metrics.py — _compute_alignment_metrics contracts.

Covers:
  TM1 — DimensionMismatchError raised when acts and weight in_dim differ.
  TM2 — All five expected keys present in the returned dict.
  TM3 — core_distance_reduction: ≈1.0 when after==source; 0.0 when no change;
         0.0 when before==source (zero distance fallback).
  TM4 — CKA contract: cka_after ≥ 1−√eps when after==source; both scores in [0,1].
  TM5 — core_dist_to_source_after == 0.0 when after==source.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.experimental.merge.exceptions import DimensionMismatchError
from modelcypher.experimental.merge.stages.transplant_metrics import (
    _compute_alignment_metrics,
)

_EXPECTED_KEYS = {
    "core_dist_to_source_before",
    "core_dist_to_source_after",
    "core_distance_reduction",
    "cka_before",
    "cka_after",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mats(b, n: int = 16, in_dim: int = 8, out_dim: int = 12, seed: int = 0):
    """Return (acts, w_before, w_after, w_source) — all distinct random matrices."""
    rng = np.random.default_rng(seed)
    acts = b.array(rng.standard_normal((n, in_dim)).astype(np.float32))
    w_before = b.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    w_after = b.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    w_source = b.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    return acts, w_before, w_after, w_source


# ---------------------------------------------------------------------------
# TM1: Dimension validation
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_raises_on_acts_weight_dim_mismatch(self):
        """acts.shape[1]=8, weight.shape[1]=99 → DimensionMismatchError."""
        b = get_default_backend()
        rng = np.random.default_rng(1)
        acts = b.array(rng.standard_normal((16, 8)).astype(np.float32))
        bad_weight = b.array(rng.standard_normal((12, 99)).astype(np.float32))
        with pytest.raises(DimensionMismatchError):
            _compute_alignment_metrics(acts, bad_weight, bad_weight, bad_weight, b)


# ---------------------------------------------------------------------------
# TM2: Returned keys
# ---------------------------------------------------------------------------

class TestReturnedKeys:
    def test_all_five_keys_present(self):
        b = get_default_backend()
        acts, w_before, w_after, w_source = _mats(b)
        result = _compute_alignment_metrics(acts, w_before, w_after, w_source, b)
        assert _EXPECTED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# TM3: core_distance_reduction invariants
# ---------------------------------------------------------------------------

class TestCoreDistanceReduction:
    def test_reduction_is_one_when_after_equals_source(self):
        """If weight_after == weight_source, output_after == output_source → dist_after = 0
        → core_distance_reduction = (dist_before - 0) / dist_before = 1.0."""
        b = get_default_backend()
        acts, w_before, _, w_source = _mats(b, seed=42)
        result = _compute_alignment_metrics(acts, w_before, w_source, w_source, b)
        assert abs(result["core_distance_reduction"] - 1.0) < 1e-4

    def test_reduction_is_zero_when_no_change(self):
        """weight_before == weight_after → dist_before == dist_after → reduction = 0."""
        b = get_default_backend()
        acts, w_before, _, w_source = _mats(b, seed=7)
        result = _compute_alignment_metrics(acts, w_before, w_before, w_source, b)
        assert result["core_distance_reduction"] == pytest.approx(0.0, abs=1e-5)

    def test_zero_dist_before_falls_back_to_zero(self):
        """weight_before == weight_source → dist_before ≈ 0 → fallback: reduction = 0."""
        b = get_default_backend()
        acts, _, w_after, w_source = _mats(b, seed=3)
        result = _compute_alignment_metrics(acts, w_source, w_after, w_source, b)
        assert result["core_distance_reduction"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TM4: CKA contract
# ---------------------------------------------------------------------------

class TestCKAContract:
    def test_cka_after_is_one_when_after_equals_source(self):
        """weight_after == weight_source → output_after == output_source → CKA = 1.0."""
        b = get_default_backend()
        acts, w_before, _, w_source = _mats(b, seed=99)
        result = _compute_alignment_metrics(acts, w_before, w_source, w_source, b)
        eps = float(machine_epsilon(b, b.array([1.0])))
        threshold = 1.0 - float(sqrt_scalar(eps, b))
        assert result["cka_after"] >= threshold, (
            f"cka_after={result['cka_after']:.8f} below threshold {threshold:.8f}"
        )

    def test_cka_values_are_valid_scores(self):
        """With distinct matrices, both CKA values are in [0, 1]."""
        b = get_default_backend()
        acts, w_before, w_after, w_source = _mats(b, seed=5)
        result = _compute_alignment_metrics(acts, w_before, w_after, w_source, b)
        assert 0.0 <= result["cka_before"] <= 1.0
        assert 0.0 <= result["cka_after"] <= 1.0


# ---------------------------------------------------------------------------
# TM5: Distance monotonicity
# ---------------------------------------------------------------------------

class TestDistanceMonotonicity:
    def test_dist_after_zero_when_after_equals_source(self):
        """weight_after == weight_source → output_after == output_source → dist_after = 0."""
        b = get_default_backend()
        acts, w_before, _, w_source = _mats(b, seed=11)
        result = _compute_alignment_metrics(acts, w_before, w_source, w_source, b)
        assert result["core_dist_to_source_after"] == pytest.approx(0.0, abs=1e-5)
