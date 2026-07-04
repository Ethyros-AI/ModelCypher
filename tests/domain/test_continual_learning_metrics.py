# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import pytest

from modelcypher.core.domain.continual_learning_metrics import (
    BackendContinualLearningMetrics,
    ContinualLearningMetrics,
)


class MockFInfo:
    eps = 1e-7

class MockBackend:
    def __init__(self):
        self._norm_returns = [2.0, 3.0]
        self._norm_idx = 0

    def finfo(self):
        return MockFInfo()

    def svd(self, array, compute_uv=False):
        val = self._norm_returns[self._norm_idx % len(self._norm_returns)]
        self._norm_idx += 1
        return [val]

    def max(self, array):
        return array[0]

    def eval(self, array):
        pass

    def to_scalar(self, val):
        return val

def test_null_space_depletion_rate():
    ranks = [1024.0, 800.0, 600.0, 424.0]
    rate = ContinualLearningMetrics.null_space_depletion_rate(ranks)
    assert rate == -200.0  # (424 - 1024) / 3

def test_cka_stability():
    # cka_history[i][j] = CKA of task j evaluated after training task i
    cka_matrix = [
        [1.0],               # after task 0
        [0.9, 1.0],          # after task 1
        [0.7, 0.8, 1.0],     # after task 2
    ]
    res = ContinualLearningMetrics.cka_stability(cka_matrix)
    # the last row is [0.7, 0.8, 1.0]. Exclude self (1.0).
    # prior evals: 0.7, 0.8
    assert res["min"] == 0.7
    assert res["mean"] == 0.75

def test_standard_cl_metrics():
    acc_matrix = [
        [0.9, 0.5, 0.5], # task 0
        [0.8, 0.9, 0.5], # task 1
        [0.6, 0.7, 0.9]  # task 2
    ]
    summary = ContinualLearningMetrics.standard_cl_metrics(acc_matrix)

    # avg_acc
    assert summary.average_accuracy == pytest.approx((0.6 + 0.7 + 0.9) / 3.0)

    # BWT
    # 0 = N-1 (2) - 0 (0) -> acc_matrix[2][0] - acc_matrix[0][0] = 0.6 - 0.9 = -0.3
    # 1 -> acc_matrix[2][1] - acc_matrix[1][1] = 0.7 - 0.9 = -0.2
    # BWT = (-0.3 - 0.2) / 2 = -0.25
    assert summary.backward_transfer == pytest.approx(-0.25)

    # Forgetting
    # max task 0 = max(0.9, 0.8) = 0.9. Diff = 0.9 - 0.6 = 0.3
    # max task 1 = max(0.9) = 0.9. Diff = 0.9 - 0.7 = 0.2
    # Forgetting = (0.3 + 0.2) / 2 = 0.25
    assert summary.forgetting_measure == pytest.approx(0.25)

    # FWT
    baselines = [0.1, 0.2, 0.3]
    summary_fwt = ContinualLearningMetrics.standard_cl_metrics(acc_matrix, random_baselines=baselines)
    # task 1: acc_matrix[0][1] (0.5) - baselines[1] (0.2) = 0.3
    # task 2: acc_matrix[1][2] (0.5) - baselines[2] (0.3) = 0.2
    # FWT = (0.3 + 0.2) / 2 = 0.25
    assert summary_fwt.forward_transfer == pytest.approx(0.25)

def test_edge_cases():
    # Empty inputs
    assert ContinualLearningMetrics.null_space_depletion_rate([]) is None
    assert ContinualLearningMetrics.cka_stability([])["min"] is None

    summary = ContinualLearningMetrics.standard_cl_metrics([])
    assert summary.average_accuracy is None

    # Zero sigma_k validation
    backend = MockBackend()
    metrics = BackendContinualLearningMetrics(backend)
    traj = metrics.spectral_budget_trajectory(["dummy"], 0.0)
    assert traj[0] == pytest.approx(2.0 / 1e-7)

def test_to_scalar_robustness():
    backend = MockBackend()
    metrics = BackendContinualLearningMetrics(backend)

    assert metrics._to_scalar([5.0]) == 5.0

    with pytest.raises(ValueError):
        metrics._to_scalar("not_a_number")

def test_backend_spectral_budget_trajectory():
    backend = MockBackend()
    metrics = BackendContinualLearningMetrics(backend)

    # Two tasks. Mock backend returns 2.0 and 3.0 for norm.
    sigma_k = 0.5
    deltas = ["dummy1", "dummy2"]

    traj = metrics.spectral_budget_trajectory(deltas, sigma_k)
    assert traj == [4.0, 6.0]  # 2.0 / 0.5, 3.0 / 0.5

def test_backend_weyl_accumulation():
    backend = MockBackend()
    metrics = BackendContinualLearningMetrics(backend)

    deltas = ["dummy1", "dummy2"]
    accum = metrics.weyl_accumulation(deltas)
    assert accum == 5.0  # 2.0 + 3.0
