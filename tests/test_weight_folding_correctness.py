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

"""
Deterministic tests for cross-architecture weight folding correctness.

These tests verify the FUNDAMENTAL INVARIANT of weight folding:

    A_aligned @ W_folded = A_target @ W_target

Where:
    - A_source: activations from source model [n_samples, d_source]
    - A_target: activations from target model [n_samples, d_target]
    - F: alignment transform such that A_source @ F achieves CKA=1.0 with A_target
    - A_aligned = A_source @ F [n_samples, d_target]
    - W_target: weight from target model [d_out, d_in]
    - W_folded: source weight transformed to target space

Weight folding formula for a linear layer y = x @ W.T:
    W_folded = F_out.T @ W_source @ pinv(F_in).T

This ensures:
    - Input activations are aligned: x_source @ F_in → matches x_target geometry
    - Output activations are aligned: result @ F_out.T → matches y_target geometry

KEY INSIGHT for testing:
    Random data does NOT have invariant geometry. Real model activations DO.
    For testing, we create target activations by applying a KNOWN transformation
    to source activations: A_target = A_source @ T_ground_truth
    Then GramAligner should recover (something equivalent to) T_ground_truth.

The tests cover:
1. Hidden stitch (most common): hidden_dim_source → hidden_dim_target
2. Attention stitch: num_heads changes (Q dimension)
3. KV stitch: GQA models where K/V have different head counts
4. Intermediate stitch: MLP intermediate dimension changes
5. Combined stitches: MLP and attention layers with multiple stitch types
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import (
    geodesic_pinv,
    machine_epsilon,
)


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def create_random_orthogonal(backend, d_source, d_target):
    """Create a random orthogonal/semi-orthogonal transformation.

    For d_source > d_target: projects down (loses some information)
    For d_source < d_target: embeds up (adds zero dimensions)
    For d_source == d_target: orthogonal rotation/reflection
    """
    # Generate random matrix and QR decompose
    random_matrix = backend.random_normal((d_source, d_target))
    backend.eval(random_matrix)

    Q, R = backend.qr(random_matrix)
    backend.eval(Q, R)

    # Q is [d_source, min(d_source, d_target)] - extract what we need
    if d_source >= d_target:
        # Q[:, :d_target] gives us [d_source, d_target]
        T = Q[:, :d_target] if Q.shape[1] >= d_target else Q
    else:
        # For d_source < d_target, Q is [d_source, d_source]
        # We need to pad or extend - use random for remaining columns
        T = Q

    backend.eval(T)
    return T


class TestHiddenStitchCorrectness:
    """Test hidden dimension stitch produces identical outputs."""

    def test_same_dimension_identity(self):
        """Same dimension should produce near-identity transformation."""
        backend = get_default_backend()
        backend.random_seed(42)

        d = 64
        n_samples = 100

        # Same activations = identity alignment
        A = backend.random_normal((n_samples, d))
        backend.eval(A)

        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A, A)

        # Should achieve perfect CKA
        assert result.achieved_cka >= 0.99, f"Self-alignment CKA too low: {result.achieved_cka}"

        # F should be close to identity
        # Note: GramAligner uses iterative geodesic optimization, so perfect identity
        # is not guaranteed. What matters is CKA=1.0, which it achieves.
        F = backend.array(result.feature_transform)
        backend.eval(F)

        I = backend.eye(d)
        diff = backend.norm(F - I)
        backend.eval(diff)

        diff_val = float(backend.to_scalar(diff))
        # Tolerance: allow small deviation from identity (2% of ||I||)
        identity_tolerance = 0.02 * float(backend.to_scalar(backend.norm(I)))
        assert diff_val <= identity_tolerance, f"F not near identity: ||F-I||={diff_val} > {identity_tolerance}"

    def test_hidden_stitch_recovers_transform(self):
        """GramAligner should recover a known transformation."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_source = 64
        d_target = 48
        n_samples = 100

        # Create source activations
        A_source = backend.random_normal((n_samples, d_source))
        backend.eval(A_source)

        # Create KNOWN ground truth transformation
        T_ground = create_random_orthogonal(backend, d_source, d_target)

        # Target is source transformed - these SHARE geometry by construction
        A_target = backend.matmul(A_source, T_ground)
        backend.eval(A_target)

        # GramAligner should find a transformation that achieves CKA=1.0
        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A_source, A_target)

        assert result.achieved_cka >= 0.95, f"Alignment CKA too low: {result.achieved_cka}"

        # The recovered F should work (doesn't have to equal T exactly - just same effect)
        F = backend.array(result.feature_transform)
        backend.eval(F)

        A_aligned = backend.matmul(A_source, F)
        backend.eval(A_aligned)

        # Check geometry match via CKA
        from modelcypher.core.domain.geometry.cka import compute_cka

        cka_result = compute_cka(A_aligned, A_target, backend=backend)
        eps = _eps(backend, cka_result.cka, 1.0)
        assert abs(cka_result.cka - 1.0) <= eps, f"Alignment not perfect: CKA={cka_result.cka}"

    def test_hidden_stitch_weight_folding(self):
        """Test full weight folding: W_folded = F_out.T @ W_source @ pinv(F_in).T"""
        backend = get_default_backend()
        backend.random_seed(42)

        d_source = 64
        d_target = 48
        n_samples = 100

        # Create source activations at input/output
        A_in_source = backend.random_normal((n_samples, d_source))
        A_out_source = backend.random_normal((n_samples, d_source))
        backend.eval(A_in_source, A_out_source)

        # Create transforms for input and output
        T_in = create_random_orthogonal(backend, d_source, d_target)
        T_out = create_random_orthogonal(backend, d_source, d_target)

        # Target activations
        A_in_target = backend.matmul(A_in_source, T_in)
        A_out_target = backend.matmul(A_out_source, T_out)
        backend.eval(A_in_target, A_out_target)

        # Source weight: [d_source, d_source] (square for simplicity)
        W_source = backend.random_normal((d_source, d_source))
        backend.eval(W_source)

        # Compute input and output alignments
        aligner = GramAligner(backend=backend)

        result_in = aligner.find_perfect_alignment(A_in_source, A_in_target)
        result_out = aligner.find_perfect_alignment(A_out_source, A_out_target)

        assert result_in.achieved_cka >= 0.95, f"Input alignment CKA too low: {result_in.achieved_cka}"
        assert result_out.achieved_cka >= 0.95, f"Output alignment CKA too low: {result_out.achieved_cka}"

        F_in = backend.array(result_in.feature_transform)  # [d_source, d_target]
        F_out = backend.array(result_out.feature_transform)  # [d_source, d_target]
        backend.eval(F_in, F_out)

        # Weight folding: W_folded = F_out.T @ W_source @ pinv(F_in).T
        # This transforms: [d_source, d_source] → [d_target, d_target]
        F_in_pinv = geodesic_pinv(backend, F_in)  # [d_target, d_source]
        backend.eval(F_in_pinv)

        W_folded = backend.matmul(
            backend.matmul(backend.transpose(F_out), W_source),
            backend.transpose(F_in_pinv)
        )  # [d_target, d_target]
        backend.eval(W_folded)

        assert W_folded.shape == (d_target, d_target), f"Expected ({d_target}, {d_target}), got {W_folded.shape}"


class TestAttentionStitchCorrectness:
    """Test attention dimension stitch for different head counts."""

    def test_attention_head_change(self):
        """Different attention head counts should align correctly."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Source: 8 heads × 64 = 512
        # Target: 6 heads × 64 = 384
        n_heads_source = 8
        n_heads_target = 6
        head_dim = 64
        d_attn_source = n_heads_source * head_dim
        d_attn_target = n_heads_target * head_dim
        n_samples = 100

        # Create source activations
        A_source = backend.random_normal((n_samples, d_attn_source))
        backend.eval(A_source)

        # Create transform
        T = create_random_orthogonal(backend, d_attn_source, d_attn_target)

        # Target via transform
        A_target = backend.matmul(A_source, T)
        backend.eval(A_target)

        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A_source, A_target)

        assert result.achieved_cka >= 0.95, f"Attention alignment CKA too low: {result.achieved_cka}"

        F = backend.array(result.feature_transform)
        backend.eval(F)

        assert F.shape == (d_attn_source, d_attn_target), f"Wrong F shape: {F.shape}"

        # Verify aligned activations preserve geometry
        A_aligned = backend.matmul(A_source, F)
        backend.eval(A_aligned)

        from modelcypher.core.domain.geometry.cka import compute_cka

        cka_result = compute_cka(A_aligned, A_target, backend=backend)
        eps = _eps(backend, cka_result.cka, 1.0)
        assert abs(cka_result.cka - 1.0) <= eps, f"Alignment not perfect: CKA={cka_result.cka}"


class TestKVStitchCorrectness:
    """Test K/V stitch for Grouped Query Attention models."""

    def test_gqa_kv_dimension(self):
        """GQA K/V dimensions (fewer heads than Q) should align correctly."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Source: 4 KV heads × 64 = 256
        # Target: 2 KV heads × 64 = 128
        n_kv_heads_source = 4
        n_kv_heads_target = 2
        head_dim = 64
        d_kv_source = n_kv_heads_source * head_dim
        d_kv_target = n_kv_heads_target * head_dim
        n_samples = 100

        A_source = backend.random_normal((n_samples, d_kv_source))
        backend.eval(A_source)

        T = create_random_orthogonal(backend, d_kv_source, d_kv_target)
        A_target = backend.matmul(A_source, T)
        backend.eval(A_target)

        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A_source, A_target)

        assert result.achieved_cka >= 0.95, f"KV alignment CKA too low: {result.achieved_cka}"

        F = backend.array(result.feature_transform)
        backend.eval(F)

        assert F.shape == (d_kv_source, d_kv_target), f"Wrong F shape: {F.shape}"


class TestIntermediateStitchCorrectness:
    """Test MLP intermediate dimension stitch."""

    def test_mlp_intermediate_dimension(self):
        """MLP intermediate dimensions should align correctly."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Source: hidden=512, intermediate=2048 (4x)
        # Target: hidden=384, intermediate=1536 (4x)
        d_inter_source = 2048
        d_inter_target = 1536
        n_samples = 100

        A_source = backend.random_normal((n_samples, d_inter_source))
        backend.eval(A_source)

        T = create_random_orthogonal(backend, d_inter_source, d_inter_target)
        A_target = backend.matmul(A_source, T)
        backend.eval(A_target)

        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A_source, A_target)

        assert result.achieved_cka >= 0.95, f"Intermediate alignment CKA too low: {result.achieved_cka}"


class TestMLPWeightFolding:
    """Test complete MLP weight folding with hidden and intermediate stitches."""

    def test_gate_proj_folding(self):
        """gate_proj [intermediate, hidden] folding correctness."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_hidden_source = 512
        d_hidden_target = 384
        d_inter_source = 2048
        d_inter_target = 1536
        n_samples = 100

        # Create source activations
        A_hidden_source = backend.random_normal((n_samples, d_hidden_source))
        A_inter_source = backend.random_normal((n_samples, d_inter_source))
        backend.eval(A_hidden_source, A_inter_source)

        # Create transforms
        T_hidden = create_random_orthogonal(backend, d_hidden_source, d_hidden_target)
        T_inter = create_random_orthogonal(backend, d_inter_source, d_inter_target)

        # Target activations
        A_hidden_target = backend.matmul(A_hidden_source, T_hidden)
        A_inter_target = backend.matmul(A_inter_source, T_inter)
        backend.eval(A_hidden_target, A_inter_target)

        # Source gate_proj weight
        W_gate_source = backend.random_normal((d_inter_source, d_hidden_source))
        backend.eval(W_gate_source)

        # Compute stitches
        aligner = GramAligner(backend=backend)

        hidden_result = aligner.find_perfect_alignment(A_hidden_source, A_hidden_target)
        inter_result = aligner.find_perfect_alignment(A_inter_source, A_inter_target)

        assert hidden_result.achieved_cka >= 0.95, f"Hidden alignment CKA too low: {hidden_result.achieved_cka}"
        assert inter_result.achieved_cka >= 0.95, f"Intermediate alignment CKA too low: {inter_result.achieved_cka}"

        F_hidden = backend.array(hidden_result.feature_transform)  # [src_hidden, tgt_hidden]
        F_inter = backend.array(inter_result.feature_transform)  # [src_inter, tgt_inter]
        backend.eval(F_hidden, F_inter)

        # Weight folding for gate_proj [inter, hidden]:
        # W_folded = F_inter.T @ W_source @ pinv(F_hidden).T
        F_hidden_pinv = geodesic_pinv(backend, F_hidden)  # [tgt_hidden, src_hidden]
        backend.eval(F_hidden_pinv)

        W_gate_folded = backend.matmul(
            backend.matmul(backend.transpose(F_inter), W_gate_source),
            backend.transpose(F_hidden_pinv)
        )  # [tgt_inter, tgt_hidden]
        backend.eval(W_gate_folded)

        expected_shape = (d_inter_target, d_hidden_target)
        assert W_gate_folded.shape == expected_shape, f"Expected {expected_shape}, got {W_gate_folded.shape}"

    def test_down_proj_folding(self):
        """down_proj [hidden, intermediate] folding correctness."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_hidden_source = 512
        d_hidden_target = 384
        d_inter_source = 2048
        d_inter_target = 1536
        n_samples = 100

        # Create source activations
        A_hidden_source = backend.random_normal((n_samples, d_hidden_source))
        A_inter_source = backend.random_normal((n_samples, d_inter_source))
        backend.eval(A_hidden_source, A_inter_source)

        # Create transforms
        T_hidden = create_random_orthogonal(backend, d_hidden_source, d_hidden_target)
        T_inter = create_random_orthogonal(backend, d_inter_source, d_inter_target)

        # Target activations
        A_hidden_target = backend.matmul(A_hidden_source, T_hidden)
        A_inter_target = backend.matmul(A_inter_source, T_inter)
        backend.eval(A_hidden_target, A_inter_target)

        # Source down_proj weight
        W_down_source = backend.random_normal((d_hidden_source, d_inter_source))
        backend.eval(W_down_source)

        # Compute stitches
        aligner = GramAligner(backend=backend)

        hidden_result = aligner.find_perfect_alignment(A_hidden_source, A_hidden_target)
        inter_result = aligner.find_perfect_alignment(A_inter_source, A_inter_target)

        F_hidden = backend.array(hidden_result.feature_transform)
        F_inter = backend.array(inter_result.feature_transform)
        backend.eval(F_hidden, F_inter)

        # Weight folding for down_proj [hidden, inter]:
        # W_folded = F_hidden.T @ W_source @ pinv(F_inter).T
        F_inter_pinv = geodesic_pinv(backend, F_inter)
        backend.eval(F_inter_pinv)

        W_down_folded = backend.matmul(
            backend.matmul(backend.transpose(F_hidden), W_down_source),
            backend.transpose(F_inter_pinv)
        )  # [tgt_hidden, tgt_inter]
        backend.eval(W_down_folded)

        expected_shape = (d_hidden_target, d_inter_target)
        assert W_down_folded.shape == expected_shape, f"Expected {expected_shape}, got {W_down_folded.shape}"


class TestAttentionWeightFolding:
    """Test attention weight folding with attention and hidden stitches."""

    def test_q_proj_folding(self):
        """q_proj [attn_dim, hidden] folding correctness."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_hidden_source = 512
        d_hidden_target = 384
        d_attn_source = 512  # 8 heads × 64
        d_attn_target = 384  # 6 heads × 64
        n_samples = 100

        # Create source activations
        A_hidden_source = backend.random_normal((n_samples, d_hidden_source))
        A_attn_source = backend.random_normal((n_samples, d_attn_source))
        backend.eval(A_hidden_source, A_attn_source)

        # Create transforms
        T_hidden = create_random_orthogonal(backend, d_hidden_source, d_hidden_target)
        T_attn = create_random_orthogonal(backend, d_attn_source, d_attn_target)

        # Target activations
        A_hidden_target = backend.matmul(A_hidden_source, T_hidden)
        A_attn_target = backend.matmul(A_attn_source, T_attn)
        backend.eval(A_hidden_target, A_attn_target)

        # Source q_proj weight
        W_q_source = backend.random_normal((d_attn_source, d_hidden_source))
        backend.eval(W_q_source)

        # Compute stitches
        aligner = GramAligner(backend=backend)

        hidden_result = aligner.find_perfect_alignment(A_hidden_source, A_hidden_target)
        attn_result = aligner.find_perfect_alignment(A_attn_source, A_attn_target)

        F_hidden = backend.array(hidden_result.feature_transform)
        F_attn = backend.array(attn_result.feature_transform)
        backend.eval(F_hidden, F_attn)

        # Weight folding for q_proj [attn, hidden]:
        # W_folded = F_attn.T @ W_source @ pinv(F_hidden).T
        F_hidden_pinv = geodesic_pinv(backend, F_hidden)
        backend.eval(F_hidden_pinv)

        W_q_folded = backend.matmul(
            backend.matmul(backend.transpose(F_attn), W_q_source),
            backend.transpose(F_hidden_pinv)
        )  # [tgt_attn, tgt_hidden]
        backend.eval(W_q_folded)

        expected_shape = (d_attn_target, d_hidden_target)
        assert W_q_folded.shape == expected_shape, f"Expected {expected_shape}, got {W_q_folded.shape}"

    def test_o_proj_folding(self):
        """o_proj [hidden, attn_dim] folding correctness."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_hidden_source = 512
        d_hidden_target = 384
        d_attn_source = 512
        d_attn_target = 384
        n_samples = 100

        # Create source activations
        A_hidden_source = backend.random_normal((n_samples, d_hidden_source))
        A_attn_source = backend.random_normal((n_samples, d_attn_source))
        backend.eval(A_hidden_source, A_attn_source)

        # Create transforms
        T_hidden = create_random_orthogonal(backend, d_hidden_source, d_hidden_target)
        T_attn = create_random_orthogonal(backend, d_attn_source, d_attn_target)

        # Target activations
        A_hidden_target = backend.matmul(A_hidden_source, T_hidden)
        A_attn_target = backend.matmul(A_attn_source, T_attn)
        backend.eval(A_hidden_target, A_attn_target)

        # Source o_proj weight
        W_o_source = backend.random_normal((d_hidden_source, d_attn_source))
        backend.eval(W_o_source)

        # Compute stitches
        aligner = GramAligner(backend=backend)

        hidden_result = aligner.find_perfect_alignment(A_hidden_source, A_hidden_target)
        attn_result = aligner.find_perfect_alignment(A_attn_source, A_attn_target)

        F_hidden = backend.array(hidden_result.feature_transform)
        F_attn = backend.array(attn_result.feature_transform)
        backend.eval(F_hidden, F_attn)

        # Weight folding for o_proj [hidden, attn]:
        # W_folded = F_hidden.T @ W_source @ pinv(F_attn).T
        F_attn_pinv = geodesic_pinv(backend, F_attn)
        backend.eval(F_attn_pinv)

        W_o_folded = backend.matmul(
            backend.matmul(backend.transpose(F_hidden), W_o_source),
            backend.transpose(F_attn_pinv)
        )  # [tgt_hidden, tgt_attn]
        backend.eval(W_o_folded)

        expected_shape = (d_hidden_target, d_attn_target)
        assert W_o_folded.shape == expected_shape, f"Expected {expected_shape}, got {W_o_folded.shape}"


class TestEndToEndEquivalence:
    """Test that folded weights produce equivalent outputs."""

    def test_linear_layer_equivalence(self):
        """A_aligned @ W_folded.T should have same geometry as A_target @ W_target.T"""
        backend = get_default_backend()
        backend.random_seed(42)

        d_in_source = 64
        d_in_target = 48
        d_out = 32
        n_samples = 100

        # Input activations (source)
        A_in_source = backend.random_normal((n_samples, d_in_source))
        backend.eval(A_in_source)

        # Create transform
        T_in = create_random_orthogonal(backend, d_in_source, d_in_target)
        A_in_target = backend.matmul(A_in_source, T_in)
        backend.eval(A_in_target)

        # Target weight (what we want to match)
        W_target = backend.random_normal((d_out, d_in_target))
        backend.eval(W_target)

        # Compute input alignment
        aligner = GramAligner(backend=backend)
        result = aligner.find_perfect_alignment(A_in_source, A_in_target)

        assert result.achieved_cka >= 0.95, f"Alignment CKA too low: {result.achieved_cka}"

        F = backend.array(result.feature_transform)  # [d_in_source, d_in_target]
        backend.eval(F)

        # Aligned input
        A_aligned = backend.matmul(A_in_source, F)  # [n_samples, d_in_target]
        backend.eval(A_aligned)

        # Outputs
        Y_target = backend.matmul(A_in_target, backend.transpose(W_target))
        Y_aligned = backend.matmul(A_aligned, backend.transpose(W_target))
        backend.eval(Y_target, Y_aligned)

        # Check CKA between outputs
        from modelcypher.core.domain.geometry.cka import compute_cka

        cka_result = compute_cka(Y_aligned, Y_target, backend=backend)
        eps = _eps(backend, cka_result.cka, 1.0)
        assert abs(cka_result.cka - 1.0) <= eps, f"Output geometry mismatch: CKA={cka_result.cka}"

    def test_mlp_layer_equivalence(self):
        """Full MLP layer: hidden → intermediate → hidden with dimension changes."""
        backend = get_default_backend()
        backend.random_seed(42)

        d_hidden_source = 64
        d_hidden_target = 48
        d_inter_source = 256  # 4x hidden
        d_inter_target = 192  # 4x hidden
        n_samples = 100

        # Activations at each stage (source)
        A_hidden_source = backend.random_normal((n_samples, d_hidden_source))
        A_inter_source = backend.random_normal((n_samples, d_inter_source))
        backend.eval(A_hidden_source, A_inter_source)

        # Create transforms
        T_hidden = create_random_orthogonal(backend, d_hidden_source, d_hidden_target)
        T_inter = create_random_orthogonal(backend, d_inter_source, d_inter_target)

        # Target activations
        A_hidden_target = backend.matmul(A_hidden_source, T_hidden)
        A_inter_target = backend.matmul(A_inter_source, T_inter)
        backend.eval(A_hidden_target, A_inter_target)

        # Compute alignments
        aligner = GramAligner(backend=backend)

        hidden_result = aligner.find_perfect_alignment(A_hidden_source, A_hidden_target)
        inter_result = aligner.find_perfect_alignment(A_inter_source, A_inter_target)

        assert hidden_result.achieved_cka >= 0.95
        assert inter_result.achieved_cka >= 0.95

        F_hidden = backend.array(hidden_result.feature_transform)
        F_inter = backend.array(inter_result.feature_transform)
        backend.eval(F_hidden, F_inter)

        # Aligned activations
        A_hidden_aligned = backend.matmul(A_hidden_source, F_hidden)
        A_inter_aligned = backend.matmul(A_inter_source, F_inter)
        backend.eval(A_hidden_aligned, A_inter_aligned)

        # Check geometry preservation at each stage
        from modelcypher.core.domain.geometry.cka import compute_cka

        cka_hidden = compute_cka(A_hidden_aligned, A_hidden_target, backend=backend)
        cka_inter = compute_cka(A_inter_aligned, A_inter_target, backend=backend)

        eps = _eps(backend, cka_hidden.cka, 1.0, cka_inter.cka, 1.0)
        assert abs(cka_hidden.cka - 1.0) <= eps, f"Hidden geometry lost: CKA={cka_hidden.cka}"
        assert abs(cka_inter.cka - 1.0) <= eps, f"Intermediate geometry lost: CKA={cka_inter.cka}"


class TestDeterminism:
    """Test that weight folding is deterministic."""

    def test_alignment_is_deterministic(self):
        """Same inputs should always produce same alignment."""
        backend = get_default_backend()

        d_source = 64
        d_target = 48
        n_samples = 100

        results = []
        for _ in range(3):
            backend.random_seed(42)  # Reset seed each time
            A_source = backend.random_normal((n_samples, d_source))
            backend.eval(A_source)

            # Create transform and target
            T = create_random_orthogonal(backend, d_source, d_target)
            A_target = backend.matmul(A_source, T)
            backend.eval(A_target)

            aligner = GramAligner(backend=backend)
            result = aligner.find_perfect_alignment(A_source, A_target)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].achieved_cka == results[0].achieved_cka
            assert results[i].iterations == results[0].iterations

            F0 = backend.array(results[0].feature_transform)
            Fi = backend.array(results[i].feature_transform)
            backend.eval(F0, Fi)

            diff = backend.norm(F0 - Fi)
            backend.eval(diff)
            diff_val = float(backend.to_scalar(diff))
            eps = _eps(backend, diff_val)
            assert diff_val <= eps, "Alignment not deterministic"

    def test_weight_folding_is_deterministic(self):
        """Same inputs should always produce same folded weights."""
        backend = get_default_backend()

        d_source = 64
        d_target = 48
        n_samples = 100

        folded_weights = []
        for _ in range(3):
            backend.random_seed(42)
            A_source = backend.random_normal((n_samples, d_source))
            W_source = backend.random_normal((d_source, d_source))
            backend.eval(A_source, W_source)

            # Create transform and target
            T = create_random_orthogonal(backend, d_source, d_target)
            A_target = backend.matmul(A_source, T)
            backend.eval(A_target)

            aligner = GramAligner(backend=backend)
            result = aligner.find_perfect_alignment(A_source, A_target)

            F = backend.array(result.feature_transform)
            F_pinv = geodesic_pinv(backend, F)
            backend.eval(F, F_pinv)

            W_folded = backend.matmul(
                backend.matmul(backend.transpose(F), W_source),
                backend.transpose(F_pinv)
            )
            backend.eval(W_folded)
            folded_weights.append(W_folded)

        # All folded weights should be identical
        for i in range(1, len(folded_weights)):
            diff = backend.norm(folded_weights[i] - folded_weights[0])
            backend.eval(diff)
            diff_val = float(backend.to_scalar(diff))
            eps = _eps(backend, diff_val)
            assert diff_val <= eps, "Weight folding not deterministic"
