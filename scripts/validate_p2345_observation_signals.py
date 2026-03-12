#!/usr/bin/env python3
"""Validate P2-P5 observation signal wiring in the training loop.

Tests that the new training observation hierarchy (P1-P5) works end-to-end:
  - P2: Answer margin tracked as time series, margin-collapse stopping
  - P3: Adapter stable rank tracked, concentration stopping
  - P4: Token-weighted val loss (LongPPL-style) computation
  - P5: Effective rank trend detection, declining-rank stopping

Each test uses a minimal harness that patches the training loop's
external dependencies (iterate_batches, value_and_grad, Fisher, MASS)
while preserving the real stopping logic and metric collection.

This is a validation script for new observability features (P2-P5).
It stays in scripts/ per Research vs Production policy.
If these signals prove permanent, promote tests to tests/.

Usage:
    poetry run python scripts/validate_p2345_observation_signals.py

Roadmap link: R1 (baseline comparison), R2 (behavioral preservation operator).
"""

# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

import modelcypher.backends._mlx_training_adapter_train_mixin as train_mixin_module
from modelcypher.backends._mlx_training_adapter_train_mixin import (
    _MLXTrainingAdapterTrainMixin,
)
from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.backends.mlx_training_adapter_core import (
    collect_base_token_losses,
    compute_token_weighted_val_loss,
    iterate_vl_batches,
)
from modelcypher.core.domain.training.online_eval import OnlineEvalResult


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class _Tokenizer:
    def __init__(self, vocab: dict[str, int], *, eos_token_id: int, vocab_size: int) -> None:
        self._vocab = dict(vocab)
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [self._vocab[token] for token in text.split()]


class _TableModel:
    def __init__(self, logits_table: list[list[float]]) -> None:
        self._logits_table = logits_table

    def __call__(self, inputs, input_embeddings=None):
        del input_embeddings
        rows = []
        for sample in inputs.tolist():
            rows.append([self._logits_table[int(token_id)] for token_id in sample])
        return mx.array(rows, dtype=mx.float32)


class _EmbedTokens:
    def __init__(self, hidden_dim: int) -> None:
        self._hidden_dim = hidden_dim

    def __call__(self, token_ids):
        embeddings = []
        for row in token_ids.tolist():
            embeddings.append(
                [
                    [float(token_id), float(token_id) + 0.5, 1.0][: self._hidden_dim]
                    for token_id in row
                ],
            )
        return mx.array(embeddings, dtype=mx.float32)


class _VLModel(_TableModel):
    def __init__(self, logits_table: list[list[float]], hidden_dim: int = 3) -> None:
        super().__init__(logits_table)
        embed_tokens = _EmbedTokens(hidden_dim)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=embed_tokens),
        )
        self._hidden_dim = hidden_dim

    def visual(self, pixel_values, position_ids):
        del pixel_values, position_ids
        return mx.ones((1, self._hidden_dim), dtype=mx.float32)


class _NoopBackend:
    pass


class _TrainBackend:
    def finfo(self):
        return SimpleNamespace(eps=2.0 ** -23, max=3.4e38)

    def collect_logits(self, *_args, **_kwargs):
        return None


class _TrainModel:
    def __init__(self) -> None:
        self.param = mx.array([0.0], dtype=mx.float32)

    def trainable_parameters(self):
        return {"param": self.param}

    def load_weights(self, updated_params, strict=False):
        del strict
        for name, value in updated_params:
            if name == "param":
                self.param = value


class _TrainHarness(_MLXTrainingAdapterTrainMixin):
    def __init__(
        self,
        *,
        use_pissa: bool,
        stable_rank_sequence: list[float] | None = None,
        effective_rank_sequence: list[float] | None = None,
    ) -> None:
        self._backend = _TrainBackend()
        self._use_pissa = use_pissa
        self._stable_rank_sequence = list(stable_rank_sequence or [])
        self._effective_rank_sequence = list(effective_rank_sequence or [])
        self._dim_index = 0

    def _derive_spectral_ceiling(self, *, sigma_k_min: float, sigma_max_global: float) -> float:
        del sigma_k_min, sigma_max_global
        return 0.1

    def _has_pissa_lora(self, model) -> bool:
        del model
        return self._use_pissa

    def _iter_pissa_lora_modules(self, model):
        del model
        if not self._use_pissa:
            return iter(())
        lora = SimpleNamespace(
            scale=1.0,
            lora_a=mx.ones((2, 2), dtype=mx.float32),
            lora_b=mx.ones((2, 2), dtype=mx.float32),
        )
        return iter([("model.layers.0.self_attn.q_proj.weight", lora)])

    def _iter_nb_lora_modules(self, model):
        del model
        return iter(())

    def _probe_entropy_and_repetition(self, model, tokenizer, *, readout_erank=None):
        del model, tokenizer, readout_erank
        return 1.0, 0.0

    def _compute_dimensional_snapshot(self, model, tokenizer, probe_texts, epoch):
        del model, tokenizer, probe_texts, epoch
        if not self._effective_rank_sequence:
            return None
        index = min(self._dim_index, len(self._effective_rank_sequence) - 1)
        self._dim_index += 1
        final_dim = self._effective_rank_sequence[index]
        return SimpleNamespace(
            expansion_ratio=1.0,
            peak_dim=final_dim,
            final_dim=final_dim,
            final_used_fraction=0.5,
            final_null_fraction=0.5,
        )

    def _compute_certificate_quantities(self, **_kwargs):
        return SimpleNamespace(
            grad_norm=1.0,
            stationarity_floor=0.0,
            alignment=0.0,
            curvature=1.0,
            delta_max_val=1.0,
            val_ci_half_width=1.0,
            delta_max_worst=1.0,
            task_improvement_met=False,
            all_conditions_met=False,
            no_drift=True,
        )

    def _clamp_all_scales(self, model):
        del model
        return None

    def _expert_key_from_layer_key(self, layer_key: str):
        del layer_key
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_logits_table(
    *,
    vocab_size: int,
    target_by_input: dict[int, int],
    correct_logit_by_input: dict[int, float],
) -> list[list[float]]:
    table: list[list[float]] = []
    for input_id in range(vocab_size):
        row = [-1.5] * vocab_size
        target = target_by_input.get(input_id, 0)
        row[target] = correct_logit_by_input.get(input_id, 0.0)
        table.append(row)
    return table


def _assert_arrays_close(actual, expected, *, atol: float = 1e-6) -> None:
    diff = mx.max(mx.abs(actual - expected))
    mx.eval(diff)
    assert float(diff.item()) <= atol, f"arrays differ by {float(diff.item())} > {atol}"


def _weighted_average(per_token_batches, base_batches, *, use_sqrt: bool) -> float:
    total_num = mx.array(0.0, dtype=mx.float32)
    total_den = mx.array(0.0, dtype=mx.float32)
    for per_token_ce, base_ce in zip(per_token_batches, base_batches):
        weights = mx.maximum(base_ce.astype(mx.float32), mx.array(0.0, dtype=mx.float32))
        if use_sqrt:
            weights = mx.sqrt(weights)
        total_num = total_num + mx.sum(per_token_ce * weights)
        total_den = total_den + mx.sum(weights)
    mx.eval(total_num, total_den)
    return float(total_num.item()) / float(total_den.item())


def _manual_text_batch_losses(model, dataset, *, batch_size: int, seq_length: int):
    from mlx_lm.tuner.trainer import iterate_batches

    losses = []
    for batch, lengths in iterate_batches(dataset, batch_size, seq_length, loop=False):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(
            steps >= lengths[:, 0:1],
            steps <= lengths[:, 1:],
        ).astype(mx.float32)
        losses.append((nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask))
    return losses


def _manual_vl_batch_losses(
    model,
    dataset,
    *,
    batch_size: int,
    seq_length: int,
    image_token_id: int,
):
    losses = []
    for batch, lengths, _pixel_values_batch, _position_ids_batch in iterate_vl_batches(
        dataset,
        batch_size,
        seq_length,
        loop=False,
    ):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(
            steps >= lengths[:, 0:1],
            steps <= lengths[:, 1:],
        ).astype(mx.float32)
        mask = mask * (targets != image_token_id).astype(mx.float32)
        losses.append((nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask))
    return losses


def _patch_train_loop_runtime(originals: dict) -> None:
    """Monkey-patch external training dependencies with minimal fakes.

    Stores originals so they can be restored in _unpatch_train_loop_runtime.
    """
    class _FakeFisherState:
        beta1 = 0.0
        beta2 = 0.0
        v = {}
        m = {}

    def _fake_iterate_batches(dataset, batch_size, seq_length, *, loop=False, seed=None):
        del dataset, batch_size, seq_length, seed
        batch = mx.array([[1, 2]], dtype=mx.int32)
        lengths = mx.array([[0, 2]], dtype=mx.int32)
        while True:
            yield batch, lengths
            if not loop:
                break

    def _fake_value_and_grad(_model, _loss_fn):
        def _inner(model, batch, lengths):
            del model, batch, lengths
            return (
                mx.array(1.0, dtype=mx.float32),
                mx.array(1.0, dtype=mx.float32),
            ), {"param": mx.array([0.0], dtype=mx.float32)}

        return _inner

    import modelcypher.core.domain.training.diagonal_fisher_preconditioner as fisher_mod
    import modelcypher.core.domain.training.mass_step_size as mass_mod

    originals["iterate_batches"] = getattr(train_mixin_module, "iterate_batches", None)
    originals["nn_value_and_grad"] = getattr(train_mixin_module.nn, "value_and_grad", None)
    originals["init_fisher_state"] = fisher_mod.init_fisher_state
    originals["update_fisher_state"] = fisher_mod.update_fisher_state
    originals["precondition_gradient"] = fisher_mod.precondition_gradient
    originals["apply_sqrt_n"] = mass_mod.apply_sqrt_n_epoch_correction
    originals["compute_per_step_rates"] = mass_mod.compute_per_step_rates
    originals["apply_validation_backoff"] = mass_mod.apply_validation_backoff

    train_mixin_module.iterate_batches = _fake_iterate_batches
    train_mixin_module.nn.value_and_grad = _fake_value_and_grad
    fisher_mod.init_fisher_state = lambda *_a, **_kw: _FakeFisherState()
    fisher_mod.update_fisher_state = lambda state, *_a, **_kw: state
    fisher_mod.precondition_gradient = lambda grad_flat, *_a, **_kw: grad_flat
    mass_mod.apply_sqrt_n_epoch_correction = lambda eta, _n: eta
    mass_mod.compute_per_step_rates = lambda *_a, **_kw: (0.01, 0.01, 0.01, 0.0, None)
    mass_mod.apply_validation_backoff = lambda eta, _vl, *, adaptive_lr: eta


def _unpatch_train_loop_runtime(originals: dict) -> None:
    """Restore patched modules to their originals."""
    import modelcypher.core.domain.training.diagonal_fisher_preconditioner as fisher_mod
    import modelcypher.core.domain.training.mass_step_size as mass_mod

    if originals.get("iterate_batches") is not None:
        train_mixin_module.iterate_batches = originals["iterate_batches"]
    if originals.get("nn_value_and_grad") is not None:
        train_mixin_module.nn.value_and_grad = originals["nn_value_and_grad"]
    fisher_mod.init_fisher_state = originals["init_fisher_state"]
    fisher_mod.update_fisher_state = originals["update_fisher_state"]
    fisher_mod.precondition_gradient = originals["precondition_gradient"]
    mass_mod.apply_sqrt_n_epoch_correction = originals["apply_sqrt_n"]
    mass_mod.compute_per_step_rates = originals["compute_per_step_rates"]
    mass_mod.apply_validation_backoff = originals["apply_validation_backoff"]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def validate_token_weighted_val_loss_text() -> None:
    """P4: Token-weighted val loss on text batches via prepare_dataset."""
    print("  P4 text batches ... ", end="", flush=True)
    adapter = MLXTrainingAdapter(_NoopBackend())
    tokenizer = _Tokenizer(
        {"red": 1, "blue": 2, "green": 3, "yellow": 4, "orange": 5},
        eos_token_id=9,
        vocab_size=12,
    )
    prepared = adapter.prepare_dataset(
        [{"text": "red blue"}, {"text": "green yellow orange"}],
        tokenizer,
    )
    mapping = {1: 2, 2: 9, 3: 4, 4: 5, 5: 9}
    base_model = _TableModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={1: 0.2, 2: 2.0, 3: 1.0, 4: 2.8, 5: -0.3},
        ),
    )
    adapted_model = _TableModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={1: 2.8, 2: 2.6, 3: 2.2, 4: 3.3, 5: 1.1},
        ),
    )
    base_losses = collect_base_token_losses(base_model, prepared, eval_batch_size=2, seq_length=8)
    manual_base = _manual_text_batch_losses(base_model, prepared, batch_size=2, seq_length=8)

    assert len(base_losses) == len(manual_base) == 1
    assert base_losses[0] is not None
    _assert_arrays_close(base_losses[0], manual_base[0])

    actual = compute_token_weighted_val_loss(
        adapted_model, prepared, eval_batch_size=2, seq_length=8, base_token_losses=base_losses,
    )
    manual_adapted = _manual_text_batch_losses(adapted_model, prepared, batch_size=2, seq_length=8)
    expected_raw = _weighted_average(manual_adapted, manual_base, use_sqrt=False)
    expected_sqrt = _weighted_average(manual_adapted, manual_base, use_sqrt=True)

    assert abs(actual - expected_raw) < 1e-5, f"got {actual}, expected {expected_raw}"
    assert abs(actual - expected_sqrt) > 1e-4, "should NOT match sqrt variant"
    print("PASS")


def validate_token_weighted_val_loss_vl() -> None:
    """P4: Token-weighted val loss on VL batches with image-token masking."""
    print("  P4 VL batches ... ", end="", flush=True)
    image_token_id = 7
    eval_dataset = [
        {
            "tokens": mx.array([1, image_token_id, 2, 9], dtype=mx.int32),
            "pixel_values": mx.array([[1.0]], dtype=mx.float32),
            "position_ids": mx.array([[0]], dtype=mx.int32),
            "image_token_id": image_token_id,
        },
        {
            "tokens": mx.array([3, 4, image_token_id, 2, 9], dtype=mx.int32),
            "pixel_values": mx.array([[2.0]], dtype=mx.float32),
            "position_ids": mx.array([[1]], dtype=mx.int32),
            "image_token_id": image_token_id,
        },
    ]
    mapping = {1: image_token_id, 2: 9, 3: 4, 4: image_token_id, image_token_id: 2}
    base_model = _VLModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={1: 0.4, 2: 2.0, 3: 1.5, 4: 0.3, image_token_id: 0.8},
        ),
    )
    adapted_model = _VLModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={1: 1.8, 2: 2.8, 3: 2.6, 4: 1.4, image_token_id: 2.2},
        ),
    )
    base_losses = collect_base_token_losses(
        base_model, eval_dataset, eval_batch_size=2, seq_length=8,
    )
    manual_base = _manual_vl_batch_losses(
        base_model, eval_dataset, batch_size=2, seq_length=8, image_token_id=image_token_id,
    )

    assert len(base_losses) == len(manual_base) == 1
    assert base_losses[0] is not None
    _assert_arrays_close(base_losses[0], manual_base[0])
    # Image tokens should be zeroed
    assert base_losses[0].tolist()[0][0] == 0.0
    assert base_losses[0].tolist()[1][1] == 0.0

    actual = compute_token_weighted_val_loss(
        adapted_model, eval_dataset, eval_batch_size=2, seq_length=8,
        base_token_losses=base_losses,
    )
    manual_adapted = _manual_vl_batch_losses(
        adapted_model, eval_dataset, batch_size=2, seq_length=8,
        image_token_id=image_token_id,
    )
    expected_raw = _weighted_average(manual_adapted, manual_base, use_sqrt=False)
    expected_sqrt = _weighted_average(manual_adapted, manual_base, use_sqrt=True)

    assert abs(actual - expected_raw) < 1e-5, f"got {actual}, expected {expected_raw}"
    assert abs(actual - expected_sqrt) > 1e-4, "should NOT match sqrt variant"
    print("PASS")


def validate_margin_declining_stop() -> None:
    """P2: Train loop stops when margin trend declines."""
    print("  P2 margin_declining stop ... ", end="", flush=True)
    originals: dict = {}
    _patch_train_loop_runtime(originals)
    try:
        adapter = _TrainHarness(use_pissa=True, stable_rank_sequence=[5.0, 5.0, 5.0, 5.0])
        model = _TrainModel()
        tokenizer = SimpleNamespace(vocab_size=32000)
        problems = [SimpleNamespace(problem_id="p1")]
        margin_values = iter([{"p1": 5.0}, {"p1": 5.0}, {"p1": 0.0}, {"p1": 0.0}])
        stable_rank_values = iter(adapter._stable_rank_sequence)

        # Patch signal providers
        _orig_compute_stable_rank = getattr(train_mixin_module, "compute_stable_rank", None)
        train_mixin_module.compute_stable_rank = lambda *_a, **_kw: next(stable_rank_values)

        import modelcypher.core.domain.training.online_eval as online_eval_mod
        _orig_margin = online_eval_mod.compute_answer_margin
        _orig_eval = online_eval_mod.evaluate_correctness
        online_eval_mod.compute_answer_margin = lambda *_a, **_kw: next(margin_values)
        online_eval_mod.evaluate_correctness = lambda **kwargs: OnlineEvalResult(
            epoch=kwargs["epoch"],
            accuracy=1.0, n_correct=1, n_total=1,
            correct_ids=frozenset({"p1"}),
            baseline_n_correct=1, baseline_accuracy=1.0,
            n_lost=0, n_gained=0, degraded=False,
            per_type_accuracy={}, per_type_correct={}, per_type_total={},
            degraded_raw=False, degraded_significant=False,
        )

        losses, stop_reason, metrics = adapter.train_loop(
            model=model,
            train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
            batch_size=1, seq_length=4, max_iters=4, seed=0,
            sigma_max=1.0, sigma_k_min=1.0,
            tokenizer=tokenizer,
            online_eval_problems=problems,
            online_eval_baseline_ids=frozenset({"p1"}),
            baseline_margins={"p1": 5.0},
        )

        assert len(losses) == 4
        assert "margin_declining" in stop_reason, f"expected margin_declining, got: {stop_reason}"
        assert abs(metrics[-1].margin_median) < 1e-6

        # Restore
        online_eval_mod.compute_answer_margin = _orig_margin
        online_eval_mod.evaluate_correctness = _orig_eval
        if _orig_compute_stable_rank is not None:
            train_mixin_module.compute_stable_rank = _orig_compute_stable_rank
    finally:
        _unpatch_train_loop_runtime(originals)
    print("PASS")


def validate_stable_rank_concentration_stop() -> None:
    """P3: Train loop stops when PiSSA adapter stable rank concentrates."""
    print("  P3 stable_rank_concentration stop ... ", end="", flush=True)
    originals: dict = {}
    _patch_train_loop_runtime(originals)
    try:
        adapter = _TrainHarness(use_pissa=True, stable_rank_sequence=[1.0])
        model = _TrainModel()
        stable_rank_values = iter(adapter._stable_rank_sequence)

        _orig_compute_stable_rank = getattr(train_mixin_module, "compute_stable_rank", None)
        train_mixin_module.compute_stable_rank = lambda *_a, **_kw: next(stable_rank_values)

        losses, stop_reason, metrics = adapter.train_loop(
            model=model,
            train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
            batch_size=1, seq_length=4, max_iters=1, seed=0,
            sigma_max=1.0, sigma_k_min=1.0,
            tokenizer=None,
        )

        assert len(losses) == 1
        assert "stable_rank_concentration" in stop_reason, f"expected stable_rank_concentration, got: {stop_reason}"
        assert abs(metrics[0].stable_rank_median - 1.0) < 1e-6
        assert metrics[0].per_layer_stable_rank is not None

        if _orig_compute_stable_rank is not None:
            train_mixin_module.compute_stable_rank = _orig_compute_stable_rank
    finally:
        _unpatch_train_loop_runtime(originals)
    print("PASS")


def validate_effective_rank_declining_stop() -> None:
    """P5: Train loop stops after 3 consecutive effective rank declines."""
    print("  P5 effective_rank_declining stop ... ", end="", flush=True)
    originals: dict = {}
    _patch_train_loop_runtime(originals)
    try:
        adapter = _TrainHarness(
            use_pissa=True,
            stable_rank_sequence=[5.0, 5.0, 5.0, 5.0],
            effective_rank_sequence=[4.0, 3.0, 2.0, 1.0],
        )
        model = _TrainModel()
        tokenizer = SimpleNamespace(vocab_size=32000)
        stable_rank_values = iter(adapter._stable_rank_sequence)

        _orig_compute_stable_rank = getattr(train_mixin_module, "compute_stable_rank", None)
        train_mixin_module.compute_stable_rank = lambda *_a, **_kw: next(stable_rank_values)

        losses, stop_reason, metrics = adapter.train_loop(
            model=model,
            train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
            batch_size=1, seq_length=4, max_iters=4, seed=0,
            sigma_max=1.0, sigma_k_min=1.0,
            tokenizer=tokenizer,
            dim_monitor=False,
        )

        assert len(losses) == 4
        assert "effective_rank_declining" in stop_reason, f"expected effective_rank_declining, got: {stop_reason}"
        assert abs(metrics[-1].effective_rank - 1.0) < 1e-6
        assert metrics[-1].effective_rank_declining_streak == 3
        assert metrics[-1].dim_final_dim is None

        if _orig_compute_stable_rank is not None:
            train_mixin_module.compute_stable_rank = _orig_compute_stable_rank
    finally:
        _unpatch_train_loop_runtime(originals)
    print("PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failures = 0
    tests = [
        validate_token_weighted_val_loss_text,
        validate_token_weighted_val_loss_vl,
        validate_margin_declining_stop,
        validate_stable_rank_concentration_stop,
        validate_effective_rank_declining_stop,
    ]

    print(f"Running {len(tests)} P2-P5 validation tests:\n")
    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            print(f"FAIL: {exc}")
            failures += 1

    print()
    if failures:
        print(f"{failures}/{len(tests)} tests FAILED")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
