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

"""MLX-specific tests for PermutationAligner.

These tests verify Metal GPU acceleration on Apple Silicon.
They are automatically skipped on non-Apple machines.
"""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner


def test_permutation_alignment_identity():
    # Identity matrix should result in identity permutation
    N = 10
    weights = mx.eye(N).astype(mx.float32)

    result = PermutationAligner.align(weights, weights)

    # Check permutation is identity
    expected_perm = mx.eye(N).astype(mx.float32)
    assert mx.array_equal(result.permutation, expected_perm)

    # Check signs are all 1s
    expected_signs = mx.eye(N).astype(mx.float32)
    assert mx.array_equal(result.signs, expected_signs)

    # Check quality
    assert result.match_quality == 1.0
    assert result.sign_flip_count == 0


def test_permutation_alignment_permuted():
    # Create a random permutation
    N = 10
    perm_indices = mx.array([1, 0, 2, 4, 3, 5, 6, 8, 7, 9])

    # Construct permutation matrix
    perm_matrix = mx.zeros((N, N), dtype=mx.float32)
    for i, idx in enumerate(perm_indices.tolist()):
        perm_matrix[idx, i] = 1.0  # P[tgt, src] = 1 so W_target = P @ W_source

    source = mx.eye(N).astype(mx.float32)
    target = perm_matrix @ source

    result = PermutationAligner.align(source, target)

    # The result.permutation should map source to target such that P_res @ source ~ target
    # In apply(align_output=True), we do P @ W.

    aligned = PermutationAligner.apply(source, result, align_output=True, align_input=False)

    assert mx.allclose(aligned, target).item()
    assert result.match_quality == 1.0


def test_permutation_alignment_sign_flip():
    N = 4
    source = mx.eye(N).astype(mx.float32)
    target = mx.eye(N).astype(mx.float32)

    # Flip sign of first row in target
    target[0] = -target[0]

    result = PermutationAligner.align(source, target)

    # Permutation should be identity
    expected_perm = mx.eye(N).astype(mx.float32)
    assert mx.array_equal(result.permutation, expected_perm)

    # Signs should have -1 at [0,0]
    signs = result.signs
    assert signs[0, 0].item() == -1.0
    assert signs[1, 1].item() == 1.0
    assert result.sign_flip_count == 1

    aligned = PermutationAligner.apply(source, result, align_output=True)
    assert mx.allclose(aligned, target).item()
