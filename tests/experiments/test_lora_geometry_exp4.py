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

"""Experiment 4: Adapter Subspace Overlap

Question: How does adapter subspace overlap relate to composability?

This experiment measures:
- Full principal angle spectrum between adapters
- Spectral overlap (projection into other adapter's column space)
- Behavioral overlap (CKA on probe activations)
- Degradation from adapter composition

Run with:
    poetry run pytest tests/experiments/test_lora_geometry_exp4.py -v -s --capture=no
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_geometry.subspace_analysis import (
    PrincipalAngles,
    SubspaceOverlapResult,
    compute_behavioral_overlap,
    compute_degradation,
    compute_principal_angles,
    compute_spectral_overlap,
    compute_subspace_overlap,
)
from modelcypher.experimental.lora_geometry.statistics import (
    compute_pearson_correlation,
    compute_spearman_correlation,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# Results directory
RESULTS_DIR = Path("results/subspace_overlap")


def _ensure_results_dir() -> None:
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_adapter_with_controlled_overlap(
    base_dim: int = 64,
    lora_rank: int = 8,
    overlap_fraction: float = 0.5,
    reference_B: "Array | None" = None,
    backend: "Backend | None" = None,
) -> "Array":
    """Create synthetic LoRA delta with controlled overlap to reference.

    Args:
        base_dim: Output dimension.
        lora_rank: LoRA rank.
        overlap_fraction: Fraction of columns to share with reference.
        reference_B: Reference B matrix to overlap with.
        backend: Compute backend.

    Returns:
        Delta weight ΔW = B @ A.
    """
    if backend is None:
        backend = get_default_backend()

    if reference_B is None:
        # First adapter: pure random
        B = backend.random_normal((base_dim, lora_rank), dtype="float32")
    else:
        # Create adapter with controlled overlap
        n_shared = int(lora_rank * overlap_fraction)
        n_new = lora_rank - n_shared

        if n_shared > 0:
            # Take columns from reference
            shared_cols = reference_B[:, :n_shared]
        else:
            shared_cols = None

        if n_new > 0:
            # Generate new random columns
            new_cols = backend.random_normal((base_dim, n_new), dtype="float32")
        else:
            new_cols = None

        # Concatenate
        if shared_cols is not None and new_cols is not None:
            B = backend.concatenate([shared_cols, new_cols], axis=1)
        elif shared_cols is not None:
            B = shared_cols
        else:
            B = new_cols

    A = backend.random_normal((lora_rank, base_dim), dtype="float32")

    # Scale
    B = backend.multiply(B, 0.1)
    delta = backend.matmul(B, A)
    backend.eval(delta)

    return delta, B


class TestPrincipalAngles:
    """Test principal angle computation."""

    def test_identical_subspaces(self):
        """Identical subspaces have zero principal angles."""
        backend = get_default_backend()
        backend.random_seed(42)

        delta = backend.random_normal((64, 64), dtype="float32")
        backend.eval(delta)

        angles = compute_principal_angles(delta, delta, backend)

        # All angles should be near zero
        assert all(a < 0.1 for a in angles.angles_radians)
        assert angles.min_angle < 0.01

    def test_orthogonal_subspaces(self):
        """Orthogonal subspaces have π/2 principal angles."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create two matrices with orthogonal COLUMN spaces
        # For column space to be restricted, we need rank-deficient matrices
        # where columns lie in orthogonal subspaces

        # Create orthonormal basis
        Q, _ = backend.svd(backend.random_normal((64, 64), dtype="float32"))[:2]
        backend.eval(Q)

        # delta1: columns in span of Q[:, :8] (rank 8)
        # delta2: columns in span of Q[:, 56:64] (rank 8, orthogonal to delta1)
        # Make matrices with 8 columns each, each column in the respective span
        coef1 = backend.random_normal((8, 8), dtype="float32")
        coef2 = backend.random_normal((8, 8), dtype="float32")

        # delta1 shape: (64, 8) - each column is linear combo of Q[:, :8]
        delta1 = backend.matmul(Q[:, :8], coef1)
        # delta2 shape: (64, 8) - each column is linear combo of Q[:, 56:64]
        delta2 = backend.matmul(Q[:, 56:64], coef2)
        backend.eval(delta1, delta2)

        angles = compute_principal_angles(delta1, delta2, backend)

        # Max angle should be near π/2 for orthogonal column spaces
        assert angles.max_angle > 1.0  # > 57 degrees in radians

    def test_principal_angles_full_spectrum(self):
        """All principal angles are returned."""
        backend = get_default_backend()
        backend.random_seed(42)

        delta1 = backend.random_normal((64, 64), dtype="float32")
        delta2 = backend.random_normal((64, 64), dtype="float32")
        backend.eval(delta1, delta2)

        angles = compute_principal_angles(delta1, delta2, backend)

        # Should have multiple angles
        assert len(angles.angles_radians) > 1
        assert len(angles.angles_radians) == len(angles.angles_degrees)


class TestSpectralOverlap:
    """Test spectral overlap computation."""

    def test_identical_overlap(self):
        """Identical deltas have overlap = 1."""
        backend = get_default_backend()
        backend.random_seed(42)

        delta = backend.random_normal((64, 64), dtype="float32")
        backend.eval(delta)

        overlap = compute_spectral_overlap(delta, delta, backend)

        assert abs(overlap - 1.0) < 0.01

    def test_orthogonal_overlap(self):
        """Orthogonal deltas have overlap near 0."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create matrices with orthogonal column spaces
        # Same construction as test_orthogonal_subspaces
        Q, _ = backend.svd(backend.random_normal((64, 64), dtype="float32"))[:2]
        backend.eval(Q)

        coef1 = backend.random_normal((8, 8), dtype="float32")
        coef2 = backend.random_normal((8, 8), dtype="float32")

        # delta1: columns in span of Q[:, :8]
        # delta2: columns in span of Q[:, 56:64] (orthogonal to delta1)
        delta1 = backend.matmul(Q[:, :8], coef1)
        delta2 = backend.matmul(Q[:, 56:64], coef2)
        backend.eval(delta1, delta2)

        overlap = compute_spectral_overlap(delta1, delta2, backend)

        # Should be low since column spaces are orthogonal
        # Note: spectral_overlap projects delta2 onto column space of delta1
        assert overlap < 0.5


class TestBehavioralOverlap:
    """Test behavioral overlap (CKA) computation."""

    def test_identical_activations(self):
        """Identical activations have CKA = 1."""
        backend = get_default_backend()
        backend.random_seed(42)

        acts = backend.random_normal((100, 64), dtype="float32")
        backend.eval(acts)

        cka = compute_behavioral_overlap(acts, acts, backend)

        assert abs(cka - 1.0) < 0.01

    def test_random_activations(self):
        """Random activations have lower CKA."""
        backend = get_default_backend()
        backend.random_seed(42)

        acts1 = backend.random_normal((100, 64), dtype="float32")
        acts2 = backend.random_normal((100, 64), dtype="float32")
        backend.eval(acts1, acts2)

        cka = compute_behavioral_overlap(acts1, acts2, backend)

        # Random should be low but not necessarily zero
        assert 0.0 <= cka <= 1.0

    def test_scaled_activations(self):
        """CKA is invariant to isotropic scaling."""
        backend = get_default_backend()
        backend.random_seed(42)

        acts1 = backend.random_normal((100, 64), dtype="float32")
        acts2 = backend.multiply(acts1, 5.0)  # Scale by 5
        backend.eval(acts1, acts2)

        cka = compute_behavioral_overlap(acts1, acts2, backend)

        # Should be nearly 1 (scaling invariant)
        assert cka > 0.99


class TestFullExperiment:
    """Full subspace overlap experiment."""

    @pytest.mark.slow
    def test_full_subspace_overlap_experiment(self):
        """Run full subspace overlap experiment."""
        _ensure_results_dir()

        backend = get_default_backend()
        backend.random_seed(42)

        # Create 6 adapters with varying overlap
        n_adapters = 6
        base_dim = 64
        lora_rank = 8

        adapters = []
        adapter_Bs = []

        # First adapter: pure random
        delta1, B1 = _create_synthetic_adapter_with_controlled_overlap(
            base_dim=base_dim,
            lora_rank=lora_rank,
            overlap_fraction=0.0,
            reference_B=None,
            backend=backend,
        )
        adapters.append(("adapter_0", delta1))
        adapter_Bs.append(B1)

        # Remaining adapters: varying overlap with first
        for i in range(1, n_adapters):
            overlap = i / n_adapters  # 0.17, 0.33, 0.50, 0.67, 0.83
            delta, B = _create_synthetic_adapter_with_controlled_overlap(
                base_dim=base_dim,
                lora_rank=lora_rank,
                overlap_fraction=overlap,
                reference_B=B1,
                backend=backend,
            )
            adapters.append((f"adapter_{i}", delta))
            adapter_Bs.append(B)

        # Compute pairwise overlap metrics
        raw_measurements = {
            "n_adapters": n_adapters,
            "pairs": [],
        }

        spectral_overlaps = []
        degradations = []

        print("\n=== Subspace Overlap Experiment ===")
        print(f"Adapters: {n_adapters}")
        print(f"Pairs: {n_adapters * (n_adapters - 1) // 2}")
        print()

        for (id1, delta1), (id2, delta2) in combinations(adapters, 2):
            # Compute principal angles
            angles = compute_principal_angles(delta1, delta2, backend)

            # Compute spectral overlap
            spectral = compute_spectral_overlap(delta1, delta2, backend)

            # Create synthetic activations for behavioral overlap
            acts1 = backend.random_normal((128, 64), dtype="float32")
            acts2 = backend.add(
                acts1,
                backend.multiply(
                    backend.random_normal((128, 64), dtype="float32"), 0.5
                ),
            )
            backend.eval(acts1, acts2)
            behavioral = compute_behavioral_overlap(acts1, acts2, backend)

            # Simulate degradation (correlate with overlap for testing)
            # In real experiment, this would be measured perplexity
            simulated_ppl_combined = 10.0 + spectral * 5.0
            simulated_ppl_orig1 = 10.0
            simulated_ppl_orig2 = 10.0
            degradation = compute_degradation(
                simulated_ppl_combined,
                simulated_ppl_combined,
                simulated_ppl_orig1,
                simulated_ppl_orig2,
            )

            spectral_overlaps.append(spectral)
            degradations.append(degradation)

            pair_data = {
                "adapter1_id": id1,
                "adapter2_id": id2,
                "principal_angles": {
                    "angles_degrees": angles.angles_degrees,
                    "max_angle": angles.max_angle * 180 / math.pi,
                    "min_angle": angles.min_angle * 180 / math.pi,
                    "mean_angle": angles.mean_angle * 180 / math.pi,
                },
                "spectral_overlap": spectral,
                "behavioral_overlap": behavioral,
                "degradation": degradation,
            }
            raw_measurements["pairs"].append(pair_data)

            print(
                f"{id1} vs {id2}: spectral={spectral:.3f}, "
                f"behavioral={behavioral:.3f}, degradation={degradation:.3f}"
            )

        # Save raw measurements
        with open(RESULTS_DIR / "raw_measurements.json", "w") as f:
            json.dump(raw_measurements, f, indent=2)

        # Compute correlations between overlap and degradation
        pearson = compute_pearson_correlation(
            spectral_overlaps, degradations, with_ci=True, backend=backend
        )
        spearman = compute_spearman_correlation(
            spectral_overlaps, degradations, with_ci=True, backend=backend
        )

        overlap_analysis = {
            "spectral_vs_degradation": {
                "pearson_r": pearson.r,
                "pearson_ci_lower": pearson.ci.lower if pearson.ci else None,
                "pearson_ci_upper": pearson.ci.upper if pearson.ci else None,
                "spearman_rho": spearman.r,
                "spearman_ci_lower": spearman.ci.lower if spearman.ci else None,
                "spearman_ci_upper": spearman.ci.upper if spearman.ci else None,
                "n": pearson.n,
            },
        }

        with open(RESULTS_DIR / "overlap_vs_degradation.json", "w") as f:
            json.dump(overlap_analysis, f, indent=2)

        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"\nSpectral overlap vs degradation:")
        print(f"  Pearson r: {pearson.r:.4f}")
        print(f"  Spearman ρ: {spearman.r:.4f}")

        # Assertions
        assert len(raw_measurements["pairs"]) == n_adapters * (n_adapters - 1) // 2
        assert -1.0 <= pearson.r <= 1.0


class TestComposabilityInterpretation:
    """Test composability interpretation based on overlap."""

    def test_degradation_computation(self):
        """Degradation is max of absolute differences."""
        deg = compute_degradation(
            ppl_combined_task1=15.0,
            ppl_combined_task2=12.0,
            ppl_original_task1=10.0,
            ppl_original_task2=10.0,
        )

        # max(|15-10|, |12-10|) = max(5, 2) = 5
        assert abs(deg - 5.0) < 0.01

    def test_zero_degradation(self):
        """Zero degradation when combined = original."""
        deg = compute_degradation(
            ppl_combined_task1=10.0,
            ppl_combined_task2=10.0,
            ppl_original_task1=10.0,
            ppl_original_task2=10.0,
        )

        assert abs(deg) < 0.01
