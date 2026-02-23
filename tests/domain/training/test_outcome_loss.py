# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for REINFORCE outcome loss backend functions.

Exercises prepare_outcome_batches (batch construction + response_start clamping)
and make_outcome_loss (response-only masking, length normalization) end-to-end.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from modelcypher.backends.mlx_training_adapter_core import (
    make_outcome_loss,
    prepare_outcome_batches,
)


# =============================================================================
# prepare_outcome_batches tests
# =============================================================================


class TestPrepareOutcomeBatches:

    def test_returns_4_tuple(self):
        """Each batch element should be (batch, lengths, advantages, response_starts)."""
        completions = [
            ([1, 2, 3, 4, 5], 0.5, 2),
        ]
        batches = prepare_outcome_batches(completions, batch_size=4, seq_length=8)
        assert len(batches) == 1
        batch, lengths, advantages, response_starts = batches[0]
        assert batch.shape == (1, 8)
        assert lengths.shape == (1,)
        assert advantages.shape == (1,)
        assert response_starts.shape == (1,)

    def test_empty_completions(self):
        assert prepare_outcome_batches([], batch_size=4, seq_length=8) == []

    def test_truncation_clamps_response_start_to_truncated_length(self):
        """When response_start > len(truncated_tokens), clamp to len(t), not seq_length - 1.

        This is the P1 bug: clamping to seq_length - 1 leaks one prompt token
        into the response mask when the prompt fills the entire window.
        """
        # 10 tokens, seq_length=6 → truncated to 6. response_start=8 (beyond window).
        tokens = list(range(10))
        completions = [(tokens, 0.5, 8)]
        batches = prepare_outcome_batches(completions, batch_size=4, seq_length=6)
        _, lengths, _, response_starts = batches[0]

        mx.eval(lengths, response_starts)
        assert int(lengths[0].item()) == 6
        # Should clamp to len(t)=6, NOT seq_length-1=5
        assert int(response_starts[0].item()) == 6

    def test_response_start_within_bounds_unchanged(self):
        """When response_start < len(truncated_tokens), it passes through unchanged."""
        tokens = [1, 2, 3, 4, 5]
        completions = [(tokens, 0.3, 3)]
        batches = prepare_outcome_batches(completions, batch_size=4, seq_length=8)
        _, _, _, response_starts = batches[0]
        mx.eval(response_starts)
        assert int(response_starts[0].item()) == 3

    def test_multiple_batches(self):
        """Completions are split into multiple batches when exceeding batch_size."""
        completions = [
            ([1, 2, 3], 0.5, 1),
            ([4, 5, 6], -0.5, 2),
            ([7, 8, 9], 0.3, 1),
        ]
        batches = prepare_outcome_batches(completions, batch_size=2, seq_length=4)
        assert len(batches) == 2
        assert batches[0][0].shape[0] == 2  # first batch: 2 samples
        assert batches[1][0].shape[0] == 1  # second batch: 1 sample


# =============================================================================
# make_outcome_loss tests (require minimal model)
# =============================================================================


class _TinyModel(nn.Module):
    """Minimal model for testing loss function mechanics."""

    def __init__(self, vocab_size: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        return self.proj(self.embed(x))


class TestMakeOutcomeLoss:

    @pytest.fixture
    def model(self):
        m = _TinyModel(vocab_size=16, dim=8)
        mx.eval(m.parameters())
        return m

    def test_response_mask_zeros_prompt_ones_response(self, model):
        """Verify the combined mask is 0 for prompt positions, 1 for response positions."""
        loss_fn = make_outcome_loss()

        # Sequence of 8 tokens, response starts at index 5
        batch = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # [1, 8]
        lengths = mx.array([8])
        advantages = mx.array([1.0])
        response_starts = mx.array([5])

        # We need to inspect the mask. Run loss to verify it works, then
        # verify the mask arithmetic directly.
        loss, ntoks = loss_fn(model, batch, lengths, advantages, response_starts)
        mx.eval(loss, ntoks)

        # Verify ntoks = number of response tokens.
        # After teacher-forcing shift, targets have 7 positions.
        # Response starts at index 5, shifted to 4 in targets.
        # Positions 4, 5, 6 are response → 3 response tokens.
        assert float(ntoks.item()) == 3.0

    def test_all_prompt_no_response_zero_ntoks(self, model):
        """When response_start >= length, mask is all zeros → ntoks = 0."""
        loss_fn = make_outcome_loss()

        batch = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])  # [1, 8], 5 real tokens
        lengths = mx.array([5])
        advantages = mx.array([1.0])
        response_starts = mx.array([5])  # response starts at end of sequence

        loss, ntoks = loss_fn(model, batch, lengths, advantages, response_starts)
        mx.eval(loss, ntoks)

        # No response tokens → ntoks should be 0
        assert float(ntoks.item()) == 0.0

    def test_all_prompt_zero_gradient_contribution(self, model):
        """When response_start >= length, the sample contributes zero to loss."""
        loss_fn = make_outcome_loss()

        # Sample where all tokens are prompt (response_start = length)
        batch = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])
        lengths = mx.array([5])
        advantages = mx.array([1.0])
        response_starts = mx.array([5])

        loss, _ = loss_fn(model, batch, lengths, advantages, response_starts)
        mx.eval(loss)

        # seq_log_probs = 0/1 = 0, weighted = 1.0 * 0 = 0, loss = -0/1 = 0
        assert abs(float(loss.item())) < 1e-7

    def test_response_start_0_uses_all_tokens(self, model):
        """When response_start=0, all valid tokens are treated as response."""
        loss_fn = make_outcome_loss()

        batch = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])
        lengths = mx.array([5])
        advantages = mx.array([1.0])
        response_starts = mx.array([0])

        loss, ntoks = loss_fn(model, batch, lengths, advantages, response_starts)
        mx.eval(loss, ntoks)

        # All 4 valid target positions are response (length=5, shift=1, so 4 targets)
        assert float(ntoks.item()) == 4.0

    def test_loss_is_finite(self, model):
        """Basic sanity: loss should be finite for valid inputs."""
        loss_fn = make_outcome_loss()

        batch = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        lengths = mx.array([8])
        advantages = mx.array([0.5])
        response_starts = mx.array([4])

        loss, ntoks = loss_fn(model, batch, lengths, advantages, response_starts)
        mx.eval(loss, ntoks)

        assert mx.isfinite(loss).item()
        assert float(ntoks.item()) > 0
