# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

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
    assert float(diff.item()) <= atol


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


def _patch_train_loop_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(train_mixin_module, "iterate_batches", _fake_iterate_batches)
    monkeypatch.setattr(train_mixin_module.nn, "value_and_grad", _fake_value_and_grad)
    monkeypatch.setattr(
        "modelcypher.core.domain.training.diagonal_fisher_preconditioner.init_fisher_state",
        lambda *_args, **_kwargs: _FakeFisherState(),
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.diagonal_fisher_preconditioner.update_fisher_state",
        lambda state, *_args, **_kwargs: state,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.diagonal_fisher_preconditioner.precondition_gradient",
        lambda grad_flat, *_args, **_kwargs: grad_flat,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.mass_step_size.apply_sqrt_n_epoch_correction",
        lambda eta_ceiling, _n_batches_per_epoch: eta_ceiling,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.mass_step_size.compute_per_step_rates",
        lambda *_args, **_kwargs: (0.01, 0.01, 0.01, 0.0, None),
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.mass_step_size.apply_validation_backoff",
        lambda eta_ceiling, _val_losses, *, adaptive_lr: eta_ceiling,
    )


def test_token_weighted_val_loss_uses_prepare_dataset_batches() -> None:
    adapter = MLXTrainingAdapter(_NoopBackend())
    tokenizer = _Tokenizer(
        {
            "red": 1,
            "blue": 2,
            "green": 3,
            "yellow": 4,
            "orange": 5,
        },
        eos_token_id=9,
        vocab_size=12,
    )
    prepared = adapter.prepare_dataset(
        [
            {"text": "red blue"},
            {"text": "green yellow orange"},
        ],
        tokenizer,
    )
    mapping = {
        1: 2,
        2: 9,
        3: 4,
        4: 5,
        5: 9,
    }
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

    base_losses = collect_base_token_losses(
        base_model,
        prepared,
        eval_batch_size=2,
        seq_length=8,
    )
    manual_base_losses = _manual_text_batch_losses(
        base_model,
        prepared,
        batch_size=2,
        seq_length=8,
    )

    assert len(base_losses) == len(manual_base_losses) == 1
    assert base_losses[0] is not None
    _assert_arrays_close(base_losses[0], manual_base_losses[0])

    actual = compute_token_weighted_val_loss(
        adapted_model,
        prepared,
        eval_batch_size=2,
        seq_length=8,
        base_token_losses=base_losses,
    )
    manual_adapted_losses = _manual_text_batch_losses(
        adapted_model,
        prepared,
        batch_size=2,
        seq_length=8,
    )
    expected_raw = _weighted_average(
        manual_adapted_losses,
        manual_base_losses,
        use_sqrt=False,
    )
    expected_sqrt = _weighted_average(
        manual_adapted_losses,
        manual_base_losses,
        use_sqrt=True,
    )

    assert actual == pytest.approx(expected_raw, rel=1e-6, abs=1e-6)
    assert abs(actual - expected_sqrt) > 1e-4


def test_token_weighted_val_loss_handles_vl_batches_and_image_masks() -> None:
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
    mapping = {
        1: image_token_id,
        2: 9,
        3: 4,
        4: image_token_id,
        image_token_id: 2,
    }
    base_model = _VLModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={
                1: 0.4,
                2: 2.0,
                3: 1.5,
                4: 0.3,
                image_token_id: 0.8,
            },
        ),
    )
    adapted_model = _VLModel(
        _build_logits_table(
            vocab_size=12,
            target_by_input=mapping,
            correct_logit_by_input={
                1: 1.8,
                2: 2.8,
                3: 2.6,
                4: 1.4,
                image_token_id: 2.2,
            },
        ),
    )

    base_losses = collect_base_token_losses(
        base_model,
        eval_dataset,
        eval_batch_size=2,
        seq_length=8,
    )
    manual_base_losses = _manual_vl_batch_losses(
        base_model,
        eval_dataset,
        batch_size=2,
        seq_length=8,
        image_token_id=image_token_id,
    )

    assert len(base_losses) == len(manual_base_losses) == 1
    assert base_losses[0] is not None
    _assert_arrays_close(base_losses[0], manual_base_losses[0])
    assert base_losses[0].tolist()[0][0] == pytest.approx(0.0, abs=1e-7)
    assert base_losses[0].tolist()[1][1] == pytest.approx(0.0, abs=1e-7)

    actual = compute_token_weighted_val_loss(
        adapted_model,
        eval_dataset,
        eval_batch_size=2,
        seq_length=8,
        base_token_losses=base_losses,
    )
    manual_adapted_losses = _manual_vl_batch_losses(
        adapted_model,
        eval_dataset,
        batch_size=2,
        seq_length=8,
        image_token_id=image_token_id,
    )
    expected_raw = _weighted_average(
        manual_adapted_losses,
        manual_base_losses,
        use_sqrt=False,
    )
    expected_sqrt = _weighted_average(
        manual_adapted_losses,
        manual_base_losses,
        use_sqrt=True,
    )

    assert actual == pytest.approx(expected_raw, rel=1e-6, abs=1e-6)
    assert abs(actual - expected_sqrt) > 1e-4


def test_train_loop_stops_on_margin_declining(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_loop_runtime(monkeypatch)
    adapter = _TrainHarness(use_pissa=True, stable_rank_sequence=[5.0, 5.0, 5.0, 5.0])
    model = _TrainModel()
    tokenizer = SimpleNamespace(vocab_size=32000)
    problems = [SimpleNamespace(problem_id="p1")]
    margin_iter = iter(
        [
            {"p1": 5.0},
            {"p1": 5.0},
            {"p1": 0.0},
            {"p1": 0.0},
        ],
    )
    stable_rank_iter = iter(adapter._stable_rank_sequence)

    monkeypatch.setattr(
        train_mixin_module,
        "compute_stable_rank",
        lambda *_args, **_kwargs: next(stable_rank_iter),
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.compute_answer_margin",
        lambda *_args, **_kwargs: next(margin_iter),
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        lambda **kwargs: OnlineEvalResult(
            epoch=kwargs["epoch"],
            accuracy=1.0,
            n_correct=1,
            n_total=1,
            correct_ids=frozenset({"p1"}),
            baseline_n_correct=1,
            baseline_accuracy=1.0,
            n_lost=0,
            n_gained=0,
            degraded=False,
            per_type_accuracy={},
            per_type_correct={},
            per_type_total={},
            degraded_raw=False,
            degraded_significant=False,
        ),
    )

    losses, stop_reason, metrics = adapter.train_loop(
        model=model,
        train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
        batch_size=1,
        seq_length=4,
        max_iters=4,
        seed=0,
        sigma_max=1.0,
        sigma_k_min=1.0,
        tokenizer=tokenizer,
        online_eval_problems=problems,
        online_eval_baseline_ids=frozenset({"p1"}),
        baseline_margins={"p1": 5.0},
    )

    assert len(losses) == 4
    assert "margin_declining" in stop_reason
    assert metrics[-1].margin_median == pytest.approx(0.0)


def test_train_loop_stops_on_pissa_stable_rank_concentration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_train_loop_runtime(monkeypatch)
    adapter = _TrainHarness(use_pissa=True, stable_rank_sequence=[1.0])
    model = _TrainModel()
    stable_rank_iter = iter(adapter._stable_rank_sequence)

    monkeypatch.setattr(
        train_mixin_module,
        "compute_stable_rank",
        lambda *_args, **_kwargs: next(stable_rank_iter),
    )

    losses, stop_reason, metrics = adapter.train_loop(
        model=model,
        train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
        batch_size=1,
        seq_length=4,
        max_iters=1,
        seed=0,
        sigma_max=1.0,
        sigma_k_min=1.0,
        tokenizer=None,
    )

    assert len(losses) == 1
    assert "stable_rank_concentration" in stop_reason
    assert metrics[0].stable_rank_median == pytest.approx(1.0)
    assert metrics[0].per_layer_stable_rank is not None


def test_train_loop_stops_on_effective_rank_declining_without_dim_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_train_loop_runtime(monkeypatch)
    adapter = _TrainHarness(
        use_pissa=True,
        stable_rank_sequence=[5.0, 5.0, 5.0, 5.0],
        effective_rank_sequence=[4.0, 3.0, 2.0, 1.0],
    )
    model = _TrainModel()
    tokenizer = SimpleNamespace(vocab_size=32000)
    stable_rank_iter = iter(adapter._stable_rank_sequence)

    monkeypatch.setattr(
        train_mixin_module,
        "compute_stable_rank",
        lambda *_args, **_kwargs: next(stable_rank_iter),
    )

    losses, stop_reason, metrics = adapter.train_loop(
        model=model,
        train_dataset=[(mx.array([1, 2], dtype=mx.int32), 0)],
        batch_size=1,
        seq_length=4,
        max_iters=4,
        seed=0,
        sigma_max=1.0,
        sigma_k_min=1.0,
        tokenizer=tokenizer,
        dim_monitor=False,
    )

    assert len(losses) == 4
    assert "effective_rank_declining" in stop_reason
    assert metrics[-1].effective_rank == pytest.approx(1.0)
    assert metrics[-1].effective_rank_declining_streak == 3
    assert metrics[-1].dim_final_dim is None
