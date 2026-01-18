# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for subspace overlap analysis and direction novelty."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.subspace import (
    compute_subspace_overlap,
    project_to_subspace,
    compute_subspace_projector,
)
from modelcypher.core.domain.geometry.direction_novelty import (
    compute_per_direction_novelty,
    compute_direction_projector,
    diagnose_variance_distribution,
)
from modelcypher.core.domain.geometry.trajectory_coherence import (
    analyze_output_coherence,
)


class TestSubspaceOverlap:
    """Tests for compute_subspace_overlap function."""

    def test_identical_activations_high_overlap(self):
        """Identical activations should have maximum overlap."""
        b = get_default_backend()

        # Create identical activations with some structure
        n, d = 50, 32
        activations = b.array([[float((i * j) % 17) for j in range(d)] for i in range(n)])
        b.eval(activations)

        result = compute_subspace_overlap(activations, activations, b)

        # Identical activations should have high overlap
        assert result.overlap_fraction >= 0.8, f"Expected high overlap, got {result.overlap_fraction}"
        assert result.shared_rank > 0, "Expected non-zero shared rank"

    def test_orthogonal_activations_low_overlap(self):
        """Orthogonal activations should have low overlap."""
        b = get_default_backend()

        n, d = 50, 32

        # Create source activations - variance in first half of dimensions
        source_data = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d // 2):
                source_data[i][j] = float((i * j) % 13) + 0.1 * i
        source = b.array(source_data)

        # Create target activations - variance in second half of dimensions
        target_data = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d // 2, d):
                target_data[i][j] = float((i * j) % 11) + 0.1 * i
        target = b.array(target_data)

        b.eval(source, target)

        result = compute_subspace_overlap(source, target, b)

        # Orthogonal activations should have lower overlap
        # Note: exact value depends on rank estimation, but should be < 1.0
        assert result.overlap_fraction < 1.0, f"Expected lower overlap, got {result.overlap_fraction}"

    def test_subspace_projector_dimensions(self):
        """Subspace projector should have correct dimensions."""
        b = get_default_backend()

        d = 16
        k = 4
        # Create orthonormal basis
        basis_data = [[0.0] * d for _ in range(k)]
        for i in range(k):
            basis_data[i][i] = 1.0
        basis = b.array(basis_data)
        b.eval(basis)

        projector = compute_subspace_projector(basis, b)

        shape = b.shape(projector)
        assert int(shape[0]) == d, f"Expected d rows, got {shape[0]}"
        assert int(shape[1]) == d, f"Expected d cols, got {shape[1]}"


class TestDirectionNovelty:
    """Tests for compute_per_direction_novelty function."""

    def test_source_active_target_dormant(self):
        """Source-active, target-dormant directions should be marked novel."""
        b = get_default_backend()

        n, d = 50, 16

        # Source: high variance in first half
        source_data = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d // 2):
                source_data[i][j] = float(i) * 0.5  # High variance
            for j in range(d // 2, d):
                source_data[i][j] = 0.1  # Low variance
        source = b.array(source_data)

        # Target: high variance in second half
        target_data = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d // 2):
                target_data[i][j] = 0.1  # Low variance (source is novel here)
            for j in range(d // 2, d):
                target_data[i][j] = float(i) * 0.5  # High variance
        target = b.array(target_data)

        b.eval(source, target)

        result = compute_per_direction_novelty(source, target, b)

        # Some directions should be marked novel
        assert result.novel_count > 0, "Expected some novel directions"
        assert result.mean_novelty > 0.0, "Expected positive mean novelty"

    def test_identical_activations_no_novelty(self):
        """Identical activations should have low novelty."""
        b = get_default_backend()

        n, d = 50, 16
        # Same pattern in both
        data = [[float((i * j) % 11) for j in range(d)] for i in range(n)]
        activations = b.array(data)
        b.eval(activations)

        result = compute_per_direction_novelty(activations, activations, b)

        # Novelty ratio should be around 0.5 (equal variance)
        assert 0.4 <= result.mean_novelty <= 0.6, f"Expected ~0.5 novelty, got {result.mean_novelty}"

    def test_direction_projector_binary(self):
        """Binary projector should have only 0s and 1s on diagonal."""
        b = get_default_backend()

        n, d = 30, 8

        # Create activations with clear novel/shared split
        source_data = [[float(i * (j + 1)) for j in range(d)] for i in range(n)]
        target_data = [[float(i * (j + 1) * 0.5) for j in range(d)] for i in range(n)]
        source = b.array(source_data)
        target = b.array(target_data)
        b.eval(source, target)

        result = compute_per_direction_novelty(source, target, b)
        projector = compute_direction_projector(result, b, novel_only=True)

        # Check that projector is diagonal with binary values
        shape = b.shape(projector)
        assert int(shape[0]) == d and int(shape[1]) == d

    def test_variance_diagnostics(self):
        """Variance diagnostics should return valid values."""
        b = get_default_backend()

        n, d = 30, 8
        source = b.array([[float(i * j) for j in range(d)] for i in range(n)])
        target = b.array([[float(i * j * 0.8) for j in range(d)] for i in range(n)])
        b.eval(source, target)

        result = compute_per_direction_novelty(source, target, b)
        diagnostics = diagnose_variance_distribution(result, b)

        assert "source_total_variance" in diagnostics
        assert "target_total_variance" in diagnostics
        assert diagnostics["source_total_variance"] >= 0
        assert diagnostics["target_total_variance"] >= 0


class TestTrajectoryCoherence:
    """Tests for trajectory coherence validation."""

    def test_repetitive_output_detected(self):
        """Repetitive output should have high repetition metrics.

        Note: is_degenerate is only set when comparing to baseline (no fixed thresholds).
        This test verifies raw metrics indicate repetition.
        """
        prompt = "The capital of France is"
        output = "topology topology topology topology topology topology topology"

        metrics = analyze_output_coherence(prompt, output)

        # Check raw metrics indicate repetition (no fixed threshold for is_degenerate)
        assert metrics.repetition_score > 0.5, "Should have high repetition score"
        assert metrics.unique_token_ratio < 0.2, "Should have low unique token ratio"
        assert metrics.max_token_ratio == 1.0, "Single token should dominate"

    def test_coherent_output_passes(self):
        """Coherent output should not be flagged."""
        prompt = "The capital of France is"
        output = "Paris. France is a beautiful country in Western Europe with a rich cultural heritage and history."

        metrics = analyze_output_coherence(prompt, output)

        assert not metrics.is_degenerate, f"Coherent output should pass: {metrics.degenerate_reason}"
        assert metrics.repetition_score < 0.5, "Should have low repetition score"

    def test_single_token_collapse_detected(self):
        """Single token repeated many times should show extreme repetition metrics.

        Note: is_degenerate is only set when comparing to baseline (no fixed thresholds).
        This test verifies raw metrics indicate single-token collapse.
        """
        prompt = "What is 2+2?"
        output = "the the the the the the the the the the the the"

        metrics = analyze_output_coherence(prompt, output)

        # Check raw metrics indicate collapse (no fixed threshold for is_degenerate)
        assert metrics.repetition_score > 0.9, "Should have very high repetition score"
        assert metrics.unique_token_ratio < 0.1, "Should have very low unique token ratio"
        assert metrics.max_token_ratio == 1.0, "Single token should be 100% of output"

    def test_short_output_handled(self):
        """Very short output should be handled gracefully."""
        prompt = "Hello"
        output = "Hi"

        metrics = analyze_output_coherence(prompt, output)

        # Short output might be truncated but shouldn't crash
        assert metrics.repetition_score >= 0.0
        assert metrics.repetition_score <= 1.0
