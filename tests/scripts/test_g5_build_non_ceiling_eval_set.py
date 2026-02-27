# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for non-ceiling eval-set count selection derivation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import g5_build_non_ceiling_eval_set as g5_non_ceiling


def test_auto_selection_prefers_max_variance_non_ceiling_count() -> None:
    target_k, meta = g5_non_ceiling._choose_target_correct_count(
        n_total=20,
        band_low=None,
        band_high=None,
        n_correct_available=20,
        n_incorrect_available=20,
    )

    # Var[X/N] = p(1-p)/N is maximal at p=0.5 -> k=10 for N=20.
    assert target_k == 10
    assert meta["mode"] == "auto_non_ceiling_max_variance"
    assert meta["target_accuracy"] == pytest.approx(0.5)


def test_user_band_selection_uses_midpoint_with_non_ceiling_constraint() -> None:
    target_k, meta = g5_non_ceiling._choose_target_correct_count(
        n_total=20,
        band_low=0.60,
        band_high=0.70,
        n_correct_available=20,
        n_incorrect_available=20,
    )

    # Band [0.60, 0.70] -> counts [12, 14], midpoint is 13.
    assert target_k == 13
    assert meta["mode"] == "user_band"


def test_one_sided_band_specification_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        g5_non_ceiling._choose_target_correct_count(
            n_total=20,
            band_low=0.60,
            band_high=None,
            n_correct_available=20,
            n_incorrect_available=20,
        )


def test_rejects_when_only_ceiling_counts_are_feasible() -> None:
    with pytest.raises(ValueError, match="No feasible non-ceiling target"):
        g5_non_ceiling._choose_target_correct_count(
            n_total=20,
            band_low=None,
            band_high=None,
            n_correct_available=20,
            n_incorrect_available=0,
        )
