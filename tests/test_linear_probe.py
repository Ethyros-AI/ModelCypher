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

"""Tests for linear probes for correctness prediction.

Tests mathematical properties:
- Difference-in-means on linearly separable data -> high AUROC
- Random data -> AUROC ~0.5
- AUROC computation (Mann-Whitney U)
- Fisher separation score
- Edge cases
"""

from __future__ import annotations

import math
import random

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.linear_probe import (
    CorrectnessProbe,
    LinearProbeResult,
    compute_difference_in_means,
    train_correctness_probe,
)


@pytest.fixture
def backend():
    return get_default_backend()


def _make_separable_data(backend, n_per_class: int = 20, dim: int = 10, seed: int = 42):
    """Create linearly separable data for testing."""
    rng = random.Random(seed)
    correct = []
    incorrect = []
    for _ in range(n_per_class):
        # Correct: centered at [1, 0, 0, ...] + noise
        c = [1.0 + rng.gauss(0, 0.1)] + [rng.gauss(0, 0.1) for _ in range(dim - 1)]
        correct.append(backend.array(c))
        # Incorrect: centered at [-1, 0, 0, ...] + noise
        i = [-1.0 + rng.gauss(0, 0.1)] + [rng.gauss(0, 0.1) for _ in range(dim - 1)]
        incorrect.append(backend.array(i))
    return correct, incorrect


def _make_random_data(backend, n_per_class: int = 20, dim: int = 10, seed: int = 42):
    """Create non-separable (random) data for testing."""
    rng = random.Random(seed)
    correct = []
    incorrect = []
    for _ in range(n_per_class):
        c = [rng.gauss(0, 1) for _ in range(dim)]
        correct.append(backend.array(c))
        i = [rng.gauss(0, 1) for _ in range(dim)]
        incorrect.append(backend.array(i))
    return correct, incorrect


class TestCorrectnessProbe:
    """Tests for the CorrectnessProbe class."""

    def test_separable_data_high_auroc(self, backend) -> None:
        """Linearly separable data should yield AUROC > 0.9."""
        correct, incorrect = _make_separable_data(backend)
        probe = CorrectnessProbe(backend=backend, method="difference_in_means")
        # Use first 15 for training, last 5 for test
        probe.fit(correct[:15], incorrect[:15])
        auroc, accuracy = probe.evaluate(correct[15:], incorrect[15:])
        assert auroc > 0.9

    def test_random_data_auroc_near_chance(self, backend) -> None:
        """Non-separable data should yield AUROC near 0.5."""
        correct, incorrect = _make_random_data(backend, n_per_class=50)
        probe = CorrectnessProbe(backend=backend, method="difference_in_means")
        probe.fit(correct[:40], incorrect[:40])
        auroc, _ = probe.evaluate(correct[40:], incorrect[40:])
        # Should be roughly 0.5 (chance), allow wide margin for randomness
        assert 0.2 < auroc < 0.8

    def test_predict_score_direction(self, backend) -> None:
        """Correct-like states should score higher than incorrect-like."""
        correct, incorrect = _make_separable_data(backend)
        probe = CorrectnessProbe(backend=backend)
        probe.fit(correct, incorrect)

        score_correct = probe.predict_score(correct[0])
        score_incorrect = probe.predict_score(incorrect[0])
        assert score_correct > score_incorrect

    def test_predict_scores_batch(self, backend) -> None:
        """Batch prediction should match individual predictions."""
        correct, incorrect = _make_separable_data(backend, n_per_class=5)
        probe = CorrectnessProbe(backend=backend)
        probe.fit(correct, incorrect)

        batch_scores = probe.predict_scores_batch(correct)
        individual_scores = [probe.predict_score(c) for c in correct]

        for bs, is_ in zip(batch_scores, individual_scores):
            assert abs(bs - is_) < 1e-4

    def test_not_fitted_raises(self, backend) -> None:
        """Predict before fit should raise."""
        probe = CorrectnessProbe(backend=backend)
        with pytest.raises(ValueError, match="not fitted"):
            probe.predict_score(backend.array([1.0, 2.0]))

    def test_empty_class_raises(self, backend) -> None:
        """Empty correct or incorrect set should raise."""
        probe = CorrectnessProbe(backend=backend)
        with pytest.raises(ValueError):
            probe.fit([], [backend.array([1.0])])

    def test_direction_is_normalized(self, backend) -> None:
        """Learned direction should be approximately unit norm."""
        correct, incorrect = _make_separable_data(backend)
        probe = CorrectnessProbe(backend=backend)
        probe.fit(correct, incorrect)

        direction = probe.get_direction()
        assert direction is not None
        backend.eval(direction)
        norm = float(backend.to_scalar(backend.sqrt(backend.sum(direction * direction))))
        assert abs(norm - 1.0) < 1e-4


class TestLogisticProbe:
    """Tests for logistic regression probe variant."""

    def test_logistic_separable_data(self, backend) -> None:
        """Logistic probe should also work on separable data."""
        correct, incorrect = _make_separable_data(backend)
        probe = CorrectnessProbe(backend=backend, method="logistic")
        probe.fit(correct[:15], incorrect[:15])
        auroc, _ = probe.evaluate(correct[15:], incorrect[15:])
        assert auroc > 0.8


class TestAUROCComputation:
    """Tests for AUROC via Mann-Whitney U."""

    def test_perfect_separation(self, backend) -> None:
        """Perfect separation should give AUROC = 1.0."""
        probe = CorrectnessProbe(backend=backend)
        # All correct > all incorrect
        auroc = probe._compute_auroc([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
        assert auroc == 1.0

    def test_perfect_reversal(self, backend) -> None:
        """Perfect reversal should give AUROC = 0.0."""
        probe = CorrectnessProbe(backend=backend)
        auroc = probe._compute_auroc([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        assert auroc == 0.0

    def test_identical_scores(self, backend) -> None:
        """All identical scores should give AUROC = 0.5."""
        probe = CorrectnessProbe(backend=backend)
        auroc = probe._compute_auroc([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert auroc == 0.5

    def test_empty_list(self, backend) -> None:
        """Empty list should give AUROC = 0.5 (undefined)."""
        probe = CorrectnessProbe(backend=backend)
        assert probe._compute_auroc([], [1.0]) == 0.5
        assert probe._compute_auroc([1.0], []) == 0.5


class TestFisherSeparation:
    """Tests for Fisher separation score via compute_difference_in_means."""

    def test_separable_high_fisher(self, backend) -> None:
        """Well-separated data should have high Fisher score."""
        correct, incorrect = _make_separable_data(backend, n_per_class=30)
        _, separation = compute_difference_in_means(correct, incorrect, backend=backend)
        # Mean distance is ~2.0 in dim 0, noise ~0.1, so Fisher should be high
        assert separation > 10.0

    def test_random_low_fisher(self, backend) -> None:
        """Random data should have low Fisher score."""
        correct, incorrect = _make_random_data(backend, n_per_class=30)
        _, separation = compute_difference_in_means(correct, incorrect, backend=backend)
        # Random data: Fisher should be small
        assert separation < 5.0

    def test_direction_normalized(self, backend) -> None:
        """Returned direction should be unit norm."""
        correct, incorrect = _make_separable_data(backend)
        direction, _ = compute_difference_in_means(correct, incorrect, backend=backend)
        backend.eval(direction)
        norm = float(backend.to_scalar(backend.sqrt(backend.sum(direction * direction))))
        assert abs(norm - 1.0) < 1e-4


class TestTrainCorrectnessProbe:
    """Tests for the convenience function train_correctness_probe."""

    def test_returns_linear_probe_result(self, backend) -> None:
        """Should return a LinearProbeResult."""
        correct, incorrect = _make_separable_data(backend, n_per_class=10)
        result = train_correctness_probe(
            correct[:8], incorrect[:8],
            test_correct=correct[8:], test_incorrect=incorrect[8:],
            backend=backend,
        )
        assert isinstance(result, LinearProbeResult)
        assert result.auroc > 0.5
        assert result.n_train_correct == 8
        assert result.n_train_incorrect == 8
        assert result.n_test_correct == 2
        assert result.n_test_incorrect == 2

    def test_no_test_data_evaluates_on_train(self, backend) -> None:
        """Without test data, should evaluate on training data."""
        correct, incorrect = _make_separable_data(backend, n_per_class=10)
        result = train_correctness_probe(
            correct, incorrect,
            backend=backend,
        )
        assert result.n_test_correct == 10
        assert result.n_test_incorrect == 10
