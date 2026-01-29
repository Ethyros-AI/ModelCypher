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

"""Tests for positive_geometry.py - positive Grassmannian signatures."""

from __future__ import annotations

import math

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.positive_geometry import (
    PositiveGrassmannSignature,
    compute_positive_grassmann_signature,
)


def test_positive_grassmann_signature_dataclass_to_dict():
    """Signature dataclass exposes JSON-friendly dict keys."""
    signature = PositiveGrassmannSignature(
        probe_count=3,
        ambient_dim=2,
        subspace_rank=2,
        total_minors=3,
        evaluated_minors=3,
        finite_minors=3,
        non_finite_minors=0,
        selection="lexicographic",
        zero_threshold=0.0,
        positive_count=1,
        negative_count=1,
        zero_count=1,
        positive_fraction=1 / 3,
        negative_fraction=1 / 3,
        zero_fraction=1 / 3,
        sign_entropy=1.0,
        min_minor=-1.0,
        max_minor=1.0,
        mean_minor=0.0,
        mean_abs_minor=1.0,
        mean_positive_minor=1.0,
        mean_negative_minor=-1.0,
        plucker_norm=1.0,
        max_abs_minor=1.0,
    )
    data = signature.to_dict()
    assert data["probeCount"] == 3
    assert data["subspaceRank"] == 2
    assert data["signEntropy"] == 1.0


def test_positive_grassmann_signature_basic_matrix():
    """Compute signature for a simple rank-2 activation matrix."""
    backend = get_default_backend()
    activations = backend.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    backend.eval(activations)

    signature = compute_positive_grassmann_signature(activations, backend=backend)

    assert signature.probe_count == 3
    assert signature.subspace_rank == 2
    assert signature.total_minors == 3
    assert signature.evaluated_minors == 3
    assert signature.finite_minors == 3

    frac_sum = (
        signature.positive_fraction
        + signature.negative_fraction
        + signature.zero_fraction
    )
    assert math.isclose(frac_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert signature.max_abs_minor >= 0.0
    assert signature.plucker_norm >= 0.0


def test_positive_grassmann_signature_spectral_gap_rank():
    """Spectral-gap rank selection returns a valid signature."""
    backend = get_default_backend()
    activations = backend.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    backend.eval(activations)

    signature = compute_positive_grassmann_signature(
        activations,
        backend=backend,
        rank_source="spectral-gap",
    )

    assert signature.subspace_rank > 0
    assert signature.total_minors >= 1


def test_positive_grassmann_signature_max_minors_limit():
    """Signature respects max_minors cap."""
    backend = get_default_backend()
    activations = backend.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    backend.eval(activations)

    signature = compute_positive_grassmann_signature(
        activations,
        backend=backend,
        max_minors=3,
    )

    assert signature.total_minors == 10
    assert signature.evaluated_minors == 3


def test_positive_grassmann_signature_zero_rank():
    """Zero activations return empty signature metrics."""
    backend = get_default_backend()
    activations = backend.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    backend.eval(activations)

    signature = compute_positive_grassmann_signature(activations, backend=backend)

    assert signature.subspace_rank == 0
    assert signature.total_minors == 0
    assert signature.evaluated_minors == 0
    assert signature.positive_count == 0
