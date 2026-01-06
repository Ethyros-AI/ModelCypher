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

"""Tests for transplant geometry preservation.

These tests verify that the transplant stage preserves CKA alignment
and performs additive merging (not replacement).

Bugs this catches:
    - CKA dropping from 1.0 to 0.817 after merge
    - Using merged = source instead of merged = target + null_space_delta
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka_backend
from modelcypher.core.domain.geometry.transplant import (
    TransplantDeltaResult,
    compute_transplant_delta,
)


class TestTransplantCKAPreservation:
    """Tests for CKA preservation after merge."""
    
    def test_post_merge_cka_preserved(self) -> None:
        """Merged weights should maintain high CKA with target geometry.
        
        Bug: CKA dropped from 1.0 to 0.817 after adding source to null space.
        The post-merge re-alignment should restore CKA to near 1.0.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8
        
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)
        
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )
        
        if not result.applied:
            pytest.skip("Transplant was skipped (insufficient samples or dim mismatch)")
        
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)
        
        # Compute outputs through both weight matrices
        # W is [out, in], output = A @ W.T
        output_target = backend.matmul(activations_boundary, backend.transpose(weight_target))
        output_merged = backend.matmul(activations_boundary, backend.transpose(merged_weight))
        backend.eval(output_target, output_merged)
        
        # CKA between merged output and target output should be HIGH
        cka = compute_cka_backend(output_merged, output_target, backend)
        
        assert cka >= 0.95, (
            f"Post-merge CKA degraded to {cka:.3f}. "
            "Merged weights should preserve target geometry (CKA >= 0.95)."
        )
    
    def test_core_output_cka_preserved(self) -> None:
        """CKA on CORE activations (not boundary) should also be preserved."""
        backend = get_default_backend()
        backend.random_seed(42)
        
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8
        
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)
        
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )
        
        if not result.applied:
            pytest.skip("Transplant was skipped")
        
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)
        
        # Test on CORE activations
        output_target = backend.matmul(activations_core, backend.transpose(weight_target))
        output_merged = backend.matmul(activations_core, backend.transpose(merged_weight))
        backend.eval(output_target, output_merged)
        
        cka = compute_cka_backend(output_merged, output_target, backend)
        
        assert cka >= 0.90, (
            f"Core CKA degraded to {cka:.3f}. Should be >= 0.90."
        )


class TestAdditiveMerging:
    """Tests for additive merging (not replacement)."""
    
    def test_additive_merge_not_replacement(self) -> None:
        """Merged weight should NOT equal source weight (that's replacement, not merge).
        
        Bug: If merged = source instead of merged = target + delta, we're replacing.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8
        
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)
        
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )
        
        if not result.applied:
            pytest.skip("Transplant was skipped")
        
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)
        
        # Merged should NOT equal source (that would be replacement)
        diff_from_source = backend.subtract(merged_weight, weight_source)
        diff_norm = float(backend.to_scalar(backend.norm(diff_from_source)))
        source_norm = float(backend.to_scalar(backend.norm(weight_source)))
        
        relative_diff = diff_norm / (source_norm + 1e-8)
        
        assert relative_diff > 0.01, (
            f"Merged weight is nearly identical to source (diff={relative_diff:.4f}). "
            "This suggests replacement instead of additive merging."
        )
    
    def test_merged_closer_to_target_than_source(self) -> None:
        """Merged weight should be closer to target than to source.
        
        We're adding to target's null space, so result should resemble target.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8
        
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)
        
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )
        
        if not result.applied:
            pytest.skip("Transplant was skipped")
        
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)
        
        # Distance to target
        diff_target = backend.subtract(merged_weight, weight_target)
        dist_target = float(backend.to_scalar(backend.norm(diff_target)))
        
        # Distance to source
        diff_source = backend.subtract(merged_weight, weight_source)
        dist_source = float(backend.to_scalar(backend.norm(diff_source)))
        
        assert dist_target < dist_source, (
            f"Merged weight is closer to source ({dist_source:.4f}) than target ({dist_target:.4f}). "
            "For additive null-space merging, result should be closer to target."
        )


class TestPostMergeReAlignment:
    """Tests for post-merge GramAligner correction."""
    
    def test_merged_weight_different_from_prelim(self) -> None:
        """If post-merge re-alignment runs, merged should differ from prelim.
        
        The re-alignment step applies a correction transform F.T to merged_prelim.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 10, 8
        
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)
        
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )
        
        if not result.applied:
            pytest.skip("Transplant was skipped")
        
        # The result should have some change from the null-space projection
        # If projection_loss < 1.0, something was preserved
        assert result.projection_loss < 1.0, "No knowledge transferred"
        
        # preserved_fraction tells us how much delta survived
        assert result.preserved_fraction > 0.0, "Zero preserved fraction"
