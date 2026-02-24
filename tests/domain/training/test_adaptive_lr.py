# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.backends.mlx_training_adapter import EpochMetrics


class TestEpochMetrics:
    """Tests for EpochMetrics dataclass."""

    def test_create_with_all_fields(self):
        m = EpochMetrics(
            epoch=3,
            train_loss=1.5,
            val_loss=1.8,
            eta=0.015,
            update_norm=0.42,
            max_spectral_ratio=0.49,
            mean_token_entropy=4.2,
            repetition_rate=0.12,
            elapsed_seconds=23.1,
            displacement=0.001,
            eta_sps=0.02,
            eta_weyl=0.03,
            d_norm=0.5,
        )
        assert m.epoch == 3
        assert m.eta == 0.015
        assert m.max_spectral_ratio == 0.49
        assert m.displacement == pytest.approx(0.001)
        assert m.eta_sps == pytest.approx(0.02)
        assert m.eta_weyl == pytest.approx(0.03)
        assert m.d_norm == pytest.approx(0.5)

    def test_create_with_none_optionals(self):
        m = EpochMetrics(
            epoch=1,
            train_loss=2.0,
            val_loss=None,
            eta=0.01,
            update_norm=None,
            max_spectral_ratio=None,
            mean_token_entropy=None,
            repetition_rate=None,
            elapsed_seconds=5.0,
        )
        assert m.val_loss is None

    def test_to_dict_roundtrip(self):
        m = EpochMetrics(
            epoch=2,
            train_loss=1.2,
            val_loss=1.4,
            eta=0.02,
            update_norm=0.3,
            max_spectral_ratio=0.48,
            mean_token_entropy=3.9,
            repetition_rate=0.05,
            elapsed_seconds=15.0,
            displacement=0.002,
            eta_sps=0.01,
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["epoch"] == 2
        assert d["eta"] == 0.02
        assert d["val_loss"] == 1.4
        assert d["displacement"] == pytest.approx(0.002)
        assert d["eta_sps"] == pytest.approx(0.01)

    def test_to_dict_preserves_none(self):
        m = EpochMetrics(
            epoch=1, train_loss=2.0, val_loss=None,
            eta=0.01, update_norm=None, max_spectral_ratio=None,
            mean_token_entropy=None, repetition_rate=None, elapsed_seconds=1.0,
        )
        d = m.to_dict()
        assert d["val_loss"] is None


class TestAdaptiveLRLogic:
    """Tests for the adaptive LR update rule (pure logic, no MLX).

    The rule:
    1. If val_loss increased: halve eta
    2. Monotonic bound: eta_t = min(eta_spectral, eta_{t-1})
    3. eta never increases
    """

    @staticmethod
    def adaptive_lr_step(
        current_eta: float,
        eta_spectral: float,
        val_losses: list[float],
    ) -> float:
        """Pure-function version of the adaptive LR update in train_loop."""
        new_eta = current_eta

        # Validation-guided backoff
        if len(val_losses) >= 2 and val_losses[-1] > val_losses[-2]:
            new_eta *= 0.5

        # Monotonic bound
        new_eta = min(eta_spectral, new_eta)

        return new_eta

    def test_halves_on_val_increase(self):
        eta = self.adaptive_lr_step(
            current_eta=0.01,
            eta_spectral=0.02,
            val_losses=[1.5, 1.6],  # increased
        )
        assert eta == pytest.approx(0.005)

    def test_no_change_on_val_decrease(self):
        eta = self.adaptive_lr_step(
            current_eta=0.01,
            eta_spectral=0.02,
            val_losses=[1.5, 1.4],  # decreased
        )
        assert eta == pytest.approx(0.01)

    def test_monotonic_bound_caps_eta(self):
        eta = self.adaptive_lr_step(
            current_eta=0.02,
            eta_spectral=0.01,  # spectral says lower
            val_losses=[1.5, 1.4],  # no backoff needed
        )
        assert eta == pytest.approx(0.01)

    def test_backoff_plus_spectral_bound(self):
        """Backoff halves to 0.01, but spectral says 0.005 → use 0.005."""
        eta = self.adaptive_lr_step(
            current_eta=0.02,
            eta_spectral=0.005,
            val_losses=[1.5, 1.6],  # backoff halves to 0.01
        )
        assert eta == pytest.approx(0.005)

    def test_never_increases(self):
        """Even if spectral bound is higher, eta should not increase after backoff."""
        eta = 0.01
        # First: backoff
        eta = self.adaptive_lr_step(eta, eta_spectral=0.02, val_losses=[1.5, 1.6])
        assert eta == pytest.approx(0.005)
        # Next: spectral says 0.02 but eta stays at 0.005 (monotonic bound applies
        # because min(0.02, 0.005) = 0.005)
        eta = self.adaptive_lr_step(eta, eta_spectral=0.02, val_losses=[1.5, 1.4])
        assert eta == pytest.approx(0.005)

    def test_repeated_backoff_converges(self):
        """Multiple val increases halve eta repeatedly."""
        eta = 0.01
        for _ in range(5):
            eta = self.adaptive_lr_step(eta, eta_spectral=1.0, val_losses=[1.5, 1.6])
        assert eta == pytest.approx(0.01 / 32)  # halved 5 times

    def test_single_val_loss_no_backoff(self):
        """With only 1 val_loss entry, no backoff possible."""
        eta = self.adaptive_lr_step(
            current_eta=0.01,
            eta_spectral=0.02,
            val_losses=[1.5],
        )
        assert eta == pytest.approx(0.01)

    def test_empty_val_losses_no_backoff(self):
        eta = self.adaptive_lr_step(
            current_eta=0.01,
            eta_spectral=0.02,
            val_losses=[],
        )
        assert eta == pytest.approx(0.01)


class TestNgramRepetition:
    """Tests for 4-gram repetition rate computation."""

    @staticmethod
    def repetition_rate(tokens: list[int], n: int = 4) -> float:
        """Same logic as in _probe_entropy_and_repetition."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    def test_no_repetition(self):
        tokens = list(range(20))  # all unique 4-grams
        assert self.repetition_rate(tokens) == pytest.approx(0.0)

    def test_full_repetition(self):
        tokens = [1, 2, 3, 4] * 10  # pattern repeats → high repetition
        rate = self.repetition_rate(tokens)
        assert rate > 0.85  # most 4-grams are repeated

    def test_short_sequence(self):
        tokens = [1, 2, 3]  # shorter than n=4
        assert self.repetition_rate(tokens) == 0.0

    def test_exactly_n_tokens(self):
        tokens = [1, 2, 3, 4]  # exactly one 4-gram
        assert self.repetition_rate(tokens) == 0.0  # 1 unique / 1 total = 0% repeated

    def test_partial_repetition(self):
        # 8 tokens: [1,2,3,4, 1,2,3,4] → 5 4-grams, 4 unique → 1-4/5 = 0.2
        tokens = [1, 2, 3, 4, 1, 2, 3, 4]
        rate = self.repetition_rate(tokens)
        assert rate == pytest.approx(0.2)


class TestDatasetTrainResultEpochMetrics:
    """Test that DatasetTrainResult properly handles epoch_metrics."""

    def test_none_epoch_metrics(self):
        from modelcypher.core.use_cases.dataset_training_service import DatasetTrainResult

        result = DatasetTrainResult(
            train_iters=100,
            initial_loss=3.0,
            final_loss=1.5,
            stop_reason="val_stable",
            baseline_loss=3.5,
            baseline_perplexity=33.0,
            post_loss=1.5,
            post_perplexity=4.5,
            n_lora_layers=10,
            n_trainable_params=1000000,
            adapter_path=None,
            spectral_bounds_ok=True,
            max_spectral_ratio=0.49,
            training_time_seconds=120.0,
        )
        d = result.to_dict()
        assert "epoch_metrics" not in d

    def test_with_epoch_metrics(self):
        from modelcypher.core.use_cases.dataset_training_service import DatasetTrainResult

        result = DatasetTrainResult(
            train_iters=100,
            initial_loss=3.0,
            final_loss=1.5,
            stop_reason="val_stable",
            baseline_loss=3.5,
            baseline_perplexity=33.0,
            post_loss=1.5,
            post_perplexity=4.5,
            n_lora_layers=10,
            n_trainable_params=1000000,
            adapter_path=None,
            spectral_bounds_ok=True,
            max_spectral_ratio=0.49,
            training_time_seconds=120.0,
            epoch_metrics=[{"epoch": 1, "eta": 0.01}],
            per_layer_cka={0: 0.99, 1: 0.97},
            min_cka_layer=1,
            adapter_saturation_median_ratio=0.42,
            seq_length_used=320,
            dim_final_used_fraction=0.12,
            dim_final_null_fraction=0.88,
            dim_null_recruitment_from_baseline=0.03,
        )
        d = result.to_dict()
        assert "epoch_metrics" in d
        assert d["epoch_metrics"][0]["epoch"] == 1
        assert d["adapter_saturation_median_ratio"] == pytest.approx(0.42)
        assert d["seq_length_used"] == 320
        assert d["per_layer_cka"][0] == pytest.approx(0.99)
        assert d["per_layer_cka"][1] == pytest.approx(0.97)
        assert d["min_cka_layer"] == 1
        assert d["dim_final_used_fraction"] == pytest.approx(0.12)
        assert d["dim_final_null_fraction"] == pytest.approx(0.88)
        assert d["dim_null_recruitment_from_baseline"] == pytest.approx(0.03)
