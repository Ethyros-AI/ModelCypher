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

"""Tests for GramAligner API stability and integration.

These tests verify that GramAligner's public API is correct and integrates
properly with the transplant stage.

Bug this catches:
    - Calling `optimize_alignment` when the method is actually `find_perfect_alignment`
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import (
    GramAligner,
    AlignmentResult,
    find_alignment,
)


class TestGramAlignerPublicAPI:
    """Tests for GramAligner's public interface."""
    
    def test_find_perfect_alignment_exists(self) -> None:
        """Verify the correct public API method exists.
        
        Bug: Code called `optimize_alignment` but method is `find_perfect_alignment`.
        """
        backend = get_default_backend()
        aligner = GramAligner(backend)
        
        # This is the ONLY public alignment method
        assert hasattr(aligner, "find_perfect_alignment"), (
            "GramAligner missing find_perfect_alignment method"
        )
        
        # Verify old/wrong name doesn't exist
        assert not hasattr(aligner, "optimize_alignment"), (
            "GramAligner has deprecated optimize_alignment - use find_perfect_alignment"
        )
        assert not hasattr(aligner, "align"), (
            "GramAligner has deprecated align - use find_perfect_alignment"
        )
    
    def test_find_alignment_function_exists(self) -> None:
        """Module-level find_alignment function should exist."""
        assert callable(find_alignment)
    
    def test_alignment_result_structure(self) -> None:
        """AlignmentResult should have required fields."""
        backend = get_default_backend()
        backend.random_seed(42)
        
        # Create simple test data
        n_samples, d_source, d_target = 10, 32, 32
        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)
        
        result = find_alignment(source, target, backend)
        
        # Check required fields exist
        assert hasattr(result, "feature_transform")
        assert hasattr(result, "achieved_cka")
        assert hasattr(result, "iterations")
        assert hasattr(result, "numerical_deviation")  # Renamed from alignment_error
        assert hasattr(result, "precision_threshold")
        assert hasattr(result, "is_perfect")
        assert hasattr(result, "is_converged")
        assert hasattr(result, "is_numerically_exact")  # New property

        # Check types - feature_transform stays on GPU as Array
        assert hasattr(result.feature_transform, "shape")  # Array-like
        assert isinstance(result.achieved_cka, float)
        assert isinstance(result.numerical_deviation, float)

        # CKA = 1.0 is invariant
        assert result.achieved_cka == 1.0
        assert result.is_perfect is True  # Always True (invariant)
    
    def test_compositional_stitch_exists(self) -> None:
        """compositional_stitch method should exist for attention transforms."""
        backend = get_default_backend()
        aligner = GramAligner(backend)
        
        assert hasattr(aligner, "compositional_stitch"), (
            "GramAligner missing compositional_stitch method for attention transforms"
        )


class TestGramAlignerFunctionality:
    """Tests for GramAligner's alignment functionality."""
    
    def test_identity_alignment_achieves_cka_1(self) -> None:
        """When source == target, CKA should be exactly 1.0."""
        backend = get_default_backend()
        backend.random_seed(42)
        
        n_samples, dim = 20, 64
        data = backend.random_normal((n_samples, dim))
        backend.eval(data)
        
        result = find_alignment(data, data, backend)
        
        assert result.achieved_cka >= 0.9999, (
            f"Self-alignment should achieve CKA=1.0, got {result.achieved_cka}"
        )
        assert result.is_perfect
    
    def test_same_dim_alignment_completes(self) -> None:
        """Same-dimension alignment should complete without error.
        
        Note: Randomly initialized activations are INDEPENDENT, so high CKA
        is NOT expected. This test verifies the algorithm runs to completion.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        
        n_samples, dim = 30, 64
        source = backend.random_normal((n_samples, dim))
        target = backend.random_normal((n_samples, dim))
        backend.eval(source, target)
        
        result = find_alignment(source, target, backend)
        
        # Should complete without error
        assert result is not None
        assert isinstance(result.achieved_cka, float)
        # Transform should have correct shape
        assert len(result.feature_transform) == dim
        assert len(result.feature_transform[0]) == dim
    
    def test_cross_dim_alignment_produces_correct_shape_transform(self) -> None:
        """Cross-dimensional alignment should produce [d_source, d_target] transform."""
        backend = get_default_backend()
        backend.random_seed(42)
        
        n_samples, d_source, d_target = 20, 64, 32
        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)
        
        result = find_alignment(source, target, backend)
        
        # Transform should be [d_source, d_target]
        transform = result.feature_transform
        assert len(transform) == d_source, f"Expected {d_source} rows, got {len(transform)}"
        assert len(transform[0]) == d_target, f"Expected {d_target} cols, got {len(transform[0])}"


class TestGramAlignerIntegrationWithTransplant:
    """Tests verifying GramAligner works correctly in transplant context."""
    
    def test_transform_can_be_applied_to_weights(self) -> None:
        """Feature transform should be applicable to weight matrices."""
        backend = get_default_backend()
        backend.random_seed(42)
        
        n_samples, d_source, d_target = 20, 64, 32
        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)
        
        result = find_alignment(source_acts, target_acts, backend)
        
        # Convert transform to array and apply
        F = backend.array(result.feature_transform)
        backend.eval(F)
        
        # Apply to source activations
        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)
        
        # Aligned should have target's dimension
        assert backend.shape(aligned) == (n_samples, d_target)
