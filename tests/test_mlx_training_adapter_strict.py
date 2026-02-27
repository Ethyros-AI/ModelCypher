# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import math

import pytest

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.training.exceptions import TrainingDerivationError


class _DummyBackend:
    pass


class _DummyModel:
    def trainable_parameters(self):
        return {}


def test_entropy_baseline_unavailable_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=None,
            dataset_samples=3,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_unavailable"
    assert diagnostics["dataset_samples"] == 3


def test_entropy_baseline_non_positive_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=0.0,
            dataset_samples=7,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_non_positive"
    assert diagnostics["baseline_entropy_value"] == 0.0


def test_entropy_baseline_negative_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_entropy_floor_or_fail(
            baseline_entropy=-1.0,
            dataset_samples=9,
            scope="full_sequence_training",
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_entropy_baseline"
    diagnostics = err.diagnostics or {}
    assert diagnostics["baseline_entropy_status"] == "baseline_non_positive"
    assert diagnostics["baseline_entropy_value"] == -1.0


def test_spectral_ceiling_nonpositive_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_spectral_ceiling(
            sigma_k_min=0.0,
            sigma_max_global=1.0,
            lr_override=None,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_adapter_geometry"
    diagnostics = err.diagnostics or {}
    assert "sigma_k_min" in diagnostics
    assert "sigma_max_global" in diagnostics


def test_spectral_ceiling_nonfinite_fails_fast():
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        adapter._derive_spectral_ceiling(
            sigma_k_min=float("inf"),
            sigma_max_global=1.0,
            lr_override=None,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_adapter_geometry"


def test_spectral_ceiling_computes_ratio():
    adapter = MLXTrainingAdapter(_DummyBackend())

    ceiling = adapter._derive_spectral_ceiling(
        sigma_k_min=0.05,
        sigma_max_global=10.0,
        lr_override=None,
    )

    assert ceiling == pytest.approx(0.005, rel=1e-6)


def test_spectral_ceiling_override_bypasses():
    adapter = MLXTrainingAdapter(_DummyBackend())

    ceiling = adapter._derive_spectral_ceiling(
        sigma_k_min=0.05,
        sigma_max_global=10.0,
        lr_override=0.123,
    )

    assert ceiling == pytest.approx(0.123, rel=1e-6)


def test_entropy_baseline_derives_floor():
    adapter = MLXTrainingAdapter(_DummyBackend())

    floor = adapter._derive_entropy_floor_or_fail(
        baseline_entropy=2.0,
        dataset_samples=8,
        scope="answer_masked_training",
    )

    expected_floor = 2.0 * (1.0 - math.sqrt(math.ldexp(1.0, -23)))
    assert floor == pytest.approx(expected_floor, rel=1e-6)


def test_rollback_fails_fast_without_online_eval():
    """Rollback requires online_eval_problems to detect degradation.

    If outcome_rollback_on_degradation=True but no online_eval_problems
    are provided, train_loop should raise immediately rather than
    silently becoming inactive.
    """
    adapter = MLXTrainingAdapter(_DummyBackend())

    with pytest.raises(ValueError, match="outcome_rollback_on_degradation.*requires.*online_eval"):
        adapter.train_loop(
            model=_DummyModel(),
            train_dataset=[{"input_ids": [1, 2, 3], "length": 3}],
            batch_size=1,
            seq_length=128,
            max_iters=1,
            seed=42,
            sigma_max=1.0,
            outcome_training=True,
            outcome_problems=[{"problem": "test"}],
            online_eval_problems=None,
            outcome_rollback_on_degradation=True,
        )


# ---------------------------------------------------------------------------
# Integration test: REINFORCE rollback event with full metric assertions
# ---------------------------------------------------------------------------

@pytest.mark.mlx
def test_rollback_event_emits_correct_epoch_metrics(monkeypatch):
    """Run train_loop through a real rollback and assert every metric.

    Scenario:
    - Pre-REINFORCE online eval: 5/5 correct
    - REINFORCE gradient step actually runs (nonzero-advantage completions)
    - Post-REINFORCE online eval: 3/5 correct (Δcorrect = -2)
    - Rollback triggers: parameters restored, metrics zeroed
    """
    import mlx.core as mx
    import mlx.nn as nn

    from modelcypher.backends import get_backend
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult
    from modelcypher.core.domain.training.outcome_objective import (
        OutcomeCollectionResult,
        OutcomeCompletion,
    )

    # ── Minimal language model ──────────────────────────────────────
    class _TinyLM(nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def __call__(self, x):
            h = self.embed(x)
            return self.head(h)

    class _TinyTokenizer:
        """Minimal tokenizer stub — only needs to not be None."""
        eos_token_id = 0

    model = _TinyLM(vocab_size=32, hidden_dim=8)
    mx.eval(model.parameters())
    tokenizer = _TinyTokenizer()

    backend = get_backend("mlx")
    adapter = MLXTrainingAdapter(backend)

    # ── Dataset: 2 samples of 20 tokens, enough for 1 batch at seq_length=16 ──
    train_dataset = [
        (mx.array(list(range(1, 21)), dtype=mx.int32), 0),
        (mx.array(list(range(5, 25)), dtype=mx.int32), 0),
    ]
    eval_dataset = [
        (mx.array(list(range(2, 22)), dtype=mx.int32), 0),
    ]

    # ── Monkeypatch: evaluate_correctness ───────────────────────────
    # Call 1 (pre-RE): 5/5 correct. Call 2 (post-RE): 3/5 correct.
    eval_call_count = [0]

    def _mock_evaluate_correctness(**kwargs):
        eval_call_count[0] += 1
        if eval_call_count[0] == 1:
            # Pre-REINFORCE: 5/5
            return OnlineEvalResult(
                epoch=kwargs.get("epoch", 1),
                accuracy=1.0,
                n_correct=5,
                n_total=5,
                correct_ids=frozenset({"p0", "p1", "p2", "p3", "p4"}),
                baseline_n_correct=5,
                baseline_accuracy=1.0,
                n_lost=0,
                n_gained=0,
                degraded=False,
                degraded_raw=False,
                degraded_significant=False,
            )
        else:
            # Post-REINFORCE: 3/5 (degraded)
            return OnlineEvalResult(
                epoch=kwargs.get("epoch", 1),
                accuracy=0.6,
                n_correct=3,
                n_total=5,
                correct_ids=frozenset({"p0", "p1", "p2"}),
                baseline_n_correct=5,
                baseline_accuracy=1.0,
                n_lost=2,
                n_gained=0,
                degraded=True,
                degraded_raw=True,
                degraded_significant=True,
            )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        _mock_evaluate_correctness,
    )

    # ── Monkeypatch: collect_outcomes ───────────────────────────────
    # Return 2 completions with nonzero advantage so REINFORCE runs.
    def _mock_collect_outcomes(**kwargs):
        return OutcomeCollectionResult(
            completions=[
                OutcomeCompletion(
                    problem_id="p0",
                    prompt_variant=0,
                    tokens=list(range(1, 18)),
                    correct=True,
                    advantage=1.0,
                    response_start=3,
                ),
                OutcomeCompletion(
                    problem_id="p1",
                    prompt_variant=0,
                    tokens=list(range(2, 19)),
                    correct=False,
                    advantage=-1.0,
                    response_start=3,
                ),
            ],
            n_problems=2,
            n_correct=1,
            n_incorrect=1,
            n_mixed_problems=1,
            mean_advantage=0.0,
        )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.outcome_objective.collect_outcomes",
        _mock_collect_outcomes,
    )

    # ── Monkeypatch: default_few_shot_examples ──────────────────────
    monkeypatch.setattr(
        "modelcypher.core.domain.star.prompting.default_few_shot_examples",
        lambda: ["a", "b", "c"],
    )

    # ── Run train_loop ──────────────────────────────────────────────
    # lr_override=0.001 → tiny CE step, update_norm << sigma_k_min
    # so REINFORCE budget is not exhausted.
    # max_iters=2 because n_batches_per_epoch=2 (2 samples), so
    # epoch boundary fires at it=1: (1+1) % 2 == 0.
    losses, stop_reason, epoch_metrics = adapter.train_loop(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=1,
        seq_length=16,
        max_iters=2,
        seed=42,
        sigma_max=1.0,
        sigma_k_min=1.0,
        lr_override=0.001,
        tokenizer=tokenizer,
        outcome_training=True,
        outcome_problems=[{"problem": "test"}],
        online_eval_problems=[{"problem": "test"}],
        outcome_post_eval=True,
        outcome_rollback_on_degradation=True,
    )

    # ── Assertions on epoch metrics ─────────────────────────────────
    assert len(epoch_metrics) >= 1, "Expected at least one epoch of metrics"
    em = epoch_metrics[-1]

    # Rollback flag
    assert em.outcome_rollback is True, (
        f"Expected outcome_rollback=True, got {em.outcome_rollback}"
    )

    # Effective REINFORCE steps zeroed by rollback
    assert em.outcome_n_steps == 0, (
        f"Expected outcome_n_steps=0 after rollback, got {em.outcome_n_steps}"
    )

    # Post-eval metrics reflect degradation
    assert em.outcome_post_eval_degraded is True, (
        f"Expected post_eval_degraded=True, got {em.outcome_post_eval_degraded}"
    )
    assert em.outcome_post_eval_delta_correct is not None
    assert em.outcome_post_eval_delta_correct < 0, (
        f"Expected negative Δcorrect, got {em.outcome_post_eval_delta_correct}"
    )
    assert em.outcome_post_eval_delta_correct == -2

    # Pre-RE online eval was healthy
    assert em.online_eval_n_correct == 5
    assert em.online_eval_degraded is False

    # Post-RE eval details
    assert em.outcome_post_eval_n_correct == 3
    assert em.outcome_post_eval_n_total == 5
    assert em.outcome_post_eval_accuracy == pytest.approx(0.6, rel=1e-6)

    # REINFORCE attempted (problems & active completions were present)
    assert em.outcome_n_problems == 2
    assert em.outcome_n_active == 2

    # evaluate_correctness was called exactly twice (pre-RE + post-RE)
    assert eval_call_count[0] == 2


@pytest.mark.mlx
def test_post_outcome_stop_basis_uses_post_eval(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn

    from modelcypher.backends import get_backend
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult
    from modelcypher.core.domain.training.outcome_objective import (
        OutcomeCollectionResult,
        OutcomeCompletion,
    )

    class _TinyLM(nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def __call__(self, x):
            return self.head(self.embed(x))

    class _TinyTokenizer:
        eos_token_id = 0

    class _Problem:
        def __init__(self, pid: str):
            self.problem_id = pid

    model = _TinyLM(vocab_size=32, hidden_dim=8)
    mx.eval(model.parameters())
    tokenizer = _TinyTokenizer()
    adapter = MLXTrainingAdapter(get_backend("mlx"))

    train_dataset = [
        (mx.array(list(range(1, 21)), dtype=mx.int32), 0),
        (mx.array(list(range(5, 25)), dtype=mx.int32), 0),
    ]
    eval_dataset = [(mx.array(list(range(2, 22)), dtype=mx.int32), 0)]
    online_eval_problems = [_Problem(f"p{i}") for i in range(5)]
    baseline_ids = frozenset({p.problem_id for p in online_eval_problems})

    eval_call_count = [0]

    def _mock_evaluate_correctness(**kwargs):
        eval_call_count[0] += 1
        if eval_call_count[0] == 1:
            return OnlineEvalResult(
                epoch=kwargs.get("epoch", 0),
                accuracy=0.8,
                n_correct=4,
                n_total=5,
                correct_ids=frozenset({"p0", "p1", "p2", "p3"}),
                baseline_n_correct=5,
                baseline_accuracy=1.0,
                n_lost=1,
                n_gained=0,
                degraded=True,
                degraded_raw=True,
                degraded_significant=True,
            )
        return OnlineEvalResult(
            epoch=kwargs.get("epoch", 0),
            accuracy=1.0,
            n_correct=5,
            n_total=5,
            correct_ids=frozenset({"p0", "p1", "p2", "p3", "p4"}),
            baseline_n_correct=5,
            baseline_accuracy=1.0,
            n_lost=0,
            n_gained=0,
            degraded=False,
            degraded_raw=False,
            degraded_significant=False,
        )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        _mock_evaluate_correctness,
    )

    def _mock_collect_outcomes(**_kwargs):
        return OutcomeCollectionResult(
            completions=[
                OutcomeCompletion(
                    problem_id="p0",
                    prompt_variant=0,
                    tokens=list(range(1, 18)),
                    correct=True,
                    advantage=1.0,
                    response_start=3,
                ),
                OutcomeCompletion(
                    problem_id="p1",
                    prompt_variant=0,
                    tokens=list(range(2, 19)),
                    correct=False,
                    advantage=-1.0,
                    response_start=3,
                ),
            ],
            n_problems=2,
            n_correct=1,
            n_incorrect=1,
            n_mixed_problems=1,
            mean_advantage=0.0,
        )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.outcome_objective.collect_outcomes",
        _mock_collect_outcomes,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.star.prompting.default_few_shot_examples",
        lambda: ["a", "b", "c"],
    )

    _, stop_reason, epoch_metrics = adapter.train_loop(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=1,
        seq_length=16,
        max_iters=2,
        seed=42,
        sigma_max=1.0,
        sigma_k_min=1.0,
        lr_override=0.001,
        tokenizer=tokenizer,
        outcome_training=True,
        outcome_problems=online_eval_problems,
        online_eval_problems=online_eval_problems,
        online_eval_baseline_ids=baseline_ids,
        outcome_post_eval=True,
        outcome_rollback_on_degradation=False,
        research_online_eval_stop_stage="post_outcome",
    )

    assert "online_eval_degraded" not in stop_reason
    assert epoch_metrics
    em = epoch_metrics[-1]
    assert em.online_eval_pre_degraded is True
    assert em.online_eval_pre_degraded_raw is True
    assert em.online_eval_pre_degraded_significant is True
    assert em.online_eval_post_degraded is False
    assert em.online_eval_post_degraded_raw is False
    assert em.online_eval_post_degraded_significant is False
    assert em.online_eval_stop_basis_stage == "post_outcome"
    assert em.online_eval_stop_basis_degraded is False
    assert em.online_eval_stop_basis_degraded_raw is False
    assert em.online_eval_stop_basis_degraded_significant is False
    assert em.gate_confound_event is True
    assert eval_call_count[0] == 2


@pytest.mark.mlx
def test_lost_only_selector_uses_pre_eval_lost_problem_ids(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn

    from modelcypher.backends import get_backend
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult
    from modelcypher.core.domain.training.outcome_objective import (
        OutcomeCollectionResult,
        OutcomeCompletion,
    )

    class _TinyLM(nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def __call__(self, x):
            return self.head(self.embed(x))

    class _TinyTokenizer:
        eos_token_id = 0

    class _Problem:
        def __init__(self, pid: str):
            self.problem_id = pid

    model = _TinyLM(vocab_size=32, hidden_dim=8)
    mx.eval(model.parameters())
    tokenizer = _TinyTokenizer()
    adapter = MLXTrainingAdapter(get_backend("mlx"))

    train_dataset = [
        (mx.array(list(range(1, 21)), dtype=mx.int32), 0),
        (mx.array(list(range(5, 25)), dtype=mx.int32), 0),
    ]
    eval_dataset = [(mx.array(list(range(2, 22)), dtype=mx.int32), 0)]

    online_eval_problems = [_Problem(f"p{i}") for i in range(5)]
    baseline_ids = frozenset({p.problem_id for p in online_eval_problems})

    def _mock_evaluate_correctness(**kwargs):
        return OnlineEvalResult(
            epoch=kwargs.get("epoch", 0),
            accuracy=0.6,
            n_correct=3,
            n_total=5,
            correct_ids=frozenset({"p0", "p1", "p2"}),
            baseline_n_correct=5,
            baseline_accuracy=1.0,
            n_lost=2,
            n_gained=0,
            degraded=True,
            degraded_raw=True,
            degraded_significant=True,
        )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        _mock_evaluate_correctness,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.star.prompting.default_few_shot_examples",
        lambda: ["a", "b", "c"],
    )

    captured_problem_ids: list[str] = []

    def _mock_collect_outcomes(**kwargs):
        problems = kwargs.get("problems", [])
        captured_problem_ids[:] = [
            getattr(problem, "problem_id", "<missing>")
            for problem in problems
        ]
        return OutcomeCollectionResult(
            completions=[
                OutcomeCompletion(
                    problem_id="p3",
                    prompt_variant=0,
                    tokens=list(range(1, 18)),
                    correct=False,
                    advantage=-1.0,
                    response_start=3,
                ),
                OutcomeCompletion(
                    problem_id="p4",
                    prompt_variant=1,
                    tokens=list(range(2, 19)),
                    correct=True,
                    advantage=1.0,
                    response_start=3,
                ),
            ],
            n_problems=len(problems),
            n_correct=1,
            n_incorrect=1,
            n_mixed_problems=1,
            mean_advantage=0.0,
        )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.outcome_objective.collect_outcomes",
        _mock_collect_outcomes,
    )

    _, stop_reason, epoch_metrics = adapter.train_loop(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=1,
        seq_length=16,
        max_iters=2,
        seed=42,
        sigma_max=1.0,
        sigma_k_min=1.0,
        lr_override=0.001,
        tokenizer=tokenizer,
        outcome_training=True,
        outcome_problems=[_Problem("q0"), _Problem("q1"), _Problem("q2")],
        online_eval_problems=online_eval_problems,
        online_eval_baseline_ids=baseline_ids,
        research_outcome_selector="lost_only",
    )

    assert "online_eval_degraded" in stop_reason
    assert sorted(captured_problem_ids) == ["p3", "p4"]
    assert epoch_metrics
    em = epoch_metrics[-1]
    assert em.outcome_n_problems == 2
    assert em.outcome_n_active == 2
