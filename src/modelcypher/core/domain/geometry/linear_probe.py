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

"""Linear probes for correctness prediction from hidden states.

Implements linear probing methodology from Zhang et al. 2025 "Hidden States
Reveal Correctness Before Answers in Large Language Models"
(arXiv:2504.05419) and Marks & Tegmark 2024 "The Geometry of Truth"
(arXiv:2310.06824).

Key insight: Correctness is LINEARLY encoded in hidden states. The model
"knows" whether its answer will be correct before generating it. This
information is extractable with simple linear probes:

1. Difference-in-means: direction = mean(h_correct) - mean(h_incorrect)
2. Logistic regression: trained linear classifier
3. Mass-mean probe: variant that focuses on high-confidence regions

The probe is trained on hidden states from reasoning chunks (not answer tokens)
and evaluated on held-out samples. AUROC > 0.65 indicates the model encodes
correctness information that can predict errors before they occur.

This is fundamentally different from aggregate trajectory metrics (velocity,
curvature at answer token). Those don't work. Linear probes do.

Reference:
- Zhang et al. (2025) arXiv:2504.05419
- Marks & Tegmark (2024) arXiv:2310.06824
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class LinearProbeResult:
    """Result of training and evaluating a linear probe.

    Attributes:
        direction: The learned probe direction [hidden_dim].
        bias: The learned bias term (for logistic regression).
        auroc: Area under ROC curve on held-out data.
        accuracy: Classification accuracy on held-out data.
        n_train_correct: Number of correct training samples.
        n_train_incorrect: Number of incorrect training samples.
        n_test_correct: Number of correct test samples.
        n_test_incorrect: Number of incorrect test samples.
    """

    direction: tuple[float, ...]
    bias: float
    auroc: float
    accuracy: float
    n_train_correct: int
    n_train_incorrect: int
    n_test_correct: int
    n_test_incorrect: int


@dataclass(frozen=True)
class ProbeActivation:
    """A single sample for probe training/evaluation.

    Attributes:
        hidden_state: Hidden state vector [hidden_dim].
        is_correct: Whether the associated answer was correct.
        sample_id: Optional identifier for the sample.
    """

    hidden_state: tuple[float, ...]
    is_correct: bool
    sample_id: str | None = None


class CorrectnessProbe:
    """Linear probe for predicting correctness from hidden states.

    Following Zhang et al. 2025, this trains a simple linear classifier
    to predict whether a model's answer will be correct based on hidden
    states collected BEFORE the answer is generated.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        method: str = "difference_in_means",
    ) -> None:
        """Initialize the probe.

        Args:
            backend: Backend for tensor operations. If None, uses default.
            method: Probe training method. Options:
                - "difference_in_means": Simple mean difference direction
                - "logistic": Logistic regression (requires more samples)
        """
        self._backend = backend or get_default_backend()
        self._method = method
        self._direction: Array | None = None
        self._bias: float = 0.0
        self._is_fitted = False

    def fit(
        self,
        correct_states: list[Array],
        incorrect_states: list[Array],
    ) -> None:
        """Train the probe on labeled hidden states.

        Args:
            correct_states: Hidden states from correct answers [n_correct, hidden_dim].
            incorrect_states: Hidden states from incorrect answers [n_incorrect, hidden_dim].
        """
        b = self._backend

        if not correct_states or not incorrect_states:
            raise ValueError("Need at least one sample of each class")

        # Stack into arrays
        correct_arr = b.stack([b.array(s) for s in correct_states], axis=0)
        incorrect_arr = b.stack([b.array(s) for s in incorrect_states], axis=0)
        b.eval(correct_arr, incorrect_arr)

        if self._method == "difference_in_means":
            self._fit_difference_in_means(correct_arr, incorrect_arr)
        elif self._method == "logistic":
            self._fit_logistic(correct_arr, incorrect_arr)
        else:
            raise ValueError(f"Unknown method: {self._method}")

        self._is_fitted = True

    def _fit_difference_in_means(
        self,
        correct_arr: Array,
        incorrect_arr: Array,
    ) -> None:
        """Fit using simple difference in means.

        This is the baseline from Marks & Tegmark (2024): the "truth direction"
        is simply the vector from incorrect mean to correct mean.
        """
        b = self._backend

        # Compute means
        mean_correct = b.mean(correct_arr, axis=0)
        mean_incorrect = b.mean(incorrect_arr, axis=0)
        b.eval(mean_correct, mean_incorrect)

        # Direction: correct - incorrect (positive = more correct-like)
        direction = mean_correct - mean_incorrect
        b.eval(direction)

        # Normalize
        norm = b.sqrt(b.sum(direction * direction))
        b.eval(norm)
        norm_val = float(b.to_scalar(norm))
        if norm_val > 1e-10:
            direction = direction / norm_val
            b.eval(direction)

        self._direction = direction

        # Bias: threshold at midpoint of projections
        proj_correct = b.matmul(correct_arr, b.reshape(direction, (-1, 1)))
        proj_incorrect = b.matmul(incorrect_arr, b.reshape(direction, (-1, 1)))
        b.eval(proj_correct, proj_incorrect)

        mean_proj_correct = float(b.to_scalar(b.mean(proj_correct)))
        mean_proj_incorrect = float(b.to_scalar(b.mean(proj_incorrect)))
        self._bias = -(mean_proj_correct + mean_proj_incorrect) / 2

    def _fit_logistic(
        self,
        correct_arr: Array,
        incorrect_arr: Array,
    ) -> None:
        """Fit using logistic regression.

        Uses gradient descent on binary cross-entropy loss.
        For small samples, difference_in_means is often better.
        """
        b = self._backend

        # Concatenate data
        X = b.concatenate([correct_arr, incorrect_arr], axis=0)
        n_correct = int(b.shape(correct_arr)[0])
        n_incorrect = int(b.shape(incorrect_arr)[0])
        n_total = n_correct + n_incorrect
        hidden_dim = int(b.shape(X)[1])

        # Labels: 1 for correct, 0 for incorrect
        y = b.array([1.0] * n_correct + [0.0] * n_incorrect)
        y = b.reshape(y, (-1, 1))
        b.eval(X, y)

        # Initialize weights
        w = b.zeros((hidden_dim, 1))
        bias = b.zeros((1,))
        learning_rate = 0.01
        n_iterations = 1000

        for _ in range(n_iterations):
            # Forward pass
            logits = b.matmul(X, w) + bias
            probs = 1.0 / (1.0 + b.exp(-logits))
            b.eval(probs)

            # Gradient (BCE loss)
            error = probs - y
            grad_w = b.matmul(b.transpose(X, (1, 0)), error) / n_total
            grad_b = b.mean(error)
            b.eval(grad_w, grad_b)

            # Update
            w = w - learning_rate * grad_w
            bias = bias - learning_rate * grad_b
            b.eval(w, bias)

        self._direction = b.squeeze(w)
        b.eval(self._direction)

        # Normalize direction
        norm = b.sqrt(b.sum(self._direction * self._direction))
        b.eval(norm)
        norm_val = float(b.to_scalar(norm))
        if norm_val > 1e-10:
            self._direction = self._direction / norm_val
            b.eval(self._direction)

        self._bias = float(b.to_scalar(bias))

    def predict_score(self, hidden_state: Array) -> float:
        """Predict correctness score for a single hidden state.

        Args:
            hidden_state: Hidden state vector [hidden_dim].

        Returns:
            Score where higher = more likely correct. Not a probability.
        """
        if not self._is_fitted or self._direction is None:
            raise ValueError("Probe not fitted. Call fit() first.")

        b = self._backend

        h = b.array(hidden_state) if not hasattr(hidden_state, "shape") else hidden_state
        b.eval(h)

        # Project onto direction
        score = b.sum(h * self._direction) + self._bias
        b.eval(score)
        return float(b.to_scalar(score))

    def predict_scores_batch(self, hidden_states: list[Array]) -> list[float]:
        """Predict correctness scores for multiple hidden states.

        Args:
            hidden_states: List of hidden state vectors.

        Returns:
            List of scores (higher = more correct-like).
        """
        if not self._is_fitted or self._direction is None:
            raise ValueError("Probe not fitted. Call fit() first.")

        b = self._backend

        X = b.stack([b.array(s) for s in hidden_states], axis=0)
        b.eval(X)

        scores = b.matmul(X, b.reshape(self._direction, (-1, 1))) + self._bias
        b.eval(scores)
        return list(b.tolist(b.squeeze(scores)))

    def evaluate(
        self,
        correct_states: list[Array],
        incorrect_states: list[Array],
    ) -> tuple[float, float]:
        """Evaluate probe on held-out data.

        Args:
            correct_states: Hidden states from correct answers.
            incorrect_states: Hidden states from incorrect answers.

        Returns:
            Tuple of (AUROC, accuracy).
        """
        if not correct_states and not incorrect_states:
            return 0.5, 0.5

        scores_correct = self.predict_scores_batch(correct_states) if correct_states else []
        scores_incorrect = self.predict_scores_batch(incorrect_states) if incorrect_states else []

        # Compute AUROC (probability that correct > incorrect)
        auroc = self._compute_auroc(scores_correct, scores_incorrect)

        # Compute accuracy at threshold = 0
        n_correct_positive = sum(1 for s in scores_correct if s > 0)
        n_incorrect_negative = sum(1 for s in scores_incorrect if s <= 0)
        total = len(scores_correct) + len(scores_incorrect)
        accuracy = (n_correct_positive + n_incorrect_negative) / total if total > 0 else 0.5

        return auroc, accuracy

    def _compute_auroc(
        self,
        scores_correct: list[float],
        scores_incorrect: list[float],
    ) -> float:
        """Compute AUROC using Mann-Whitney U statistic.

        AUROC = P(score_correct > score_incorrect) for random pair.
        """
        if not scores_correct or not scores_incorrect:
            return 0.5

        n_correct = len(scores_correct)
        n_incorrect = len(scores_incorrect)

        # Count pairs where correct > incorrect
        n_greater = 0
        n_equal = 0
        for sc in scores_correct:
            for si in scores_incorrect:
                if sc > si:
                    n_greater += 1
                elif sc == si:
                    n_equal += 1

        # AUROC with tie handling
        total_pairs = n_correct * n_incorrect
        auroc = (n_greater + 0.5 * n_equal) / total_pairs if total_pairs > 0 else 0.5
        return auroc

    def get_direction(self) -> Array | None:
        """Get the learned probe direction."""
        return self._direction


def train_correctness_probe(
    correct_hidden_states: list[Array],
    incorrect_hidden_states: list[Array],
    test_correct: list[Array] | None = None,
    test_incorrect: list[Array] | None = None,
    backend: Backend | None = None,
    method: str = "difference_in_means",
) -> LinearProbeResult:
    """Train and evaluate a correctness probe.

    Args:
        correct_hidden_states: Training hidden states from correct answers.
        incorrect_hidden_states: Training hidden states from incorrect answers.
        test_correct: Test hidden states from correct answers.
        test_incorrect: Test hidden states from incorrect answers.
        backend: Backend for tensor operations.
        method: Probe method ("difference_in_means" or "logistic").

    Returns:
        LinearProbeResult with probe direction and evaluation metrics.
    """
    b = backend or get_default_backend()
    probe = CorrectnessProbe(backend=b, method=method)

    # Train
    probe.fit(correct_hidden_states, incorrect_hidden_states)

    # Evaluate
    if test_correct is not None and test_incorrect is not None:
        auroc, accuracy = probe.evaluate(test_correct, test_incorrect)
        n_test_correct = len(test_correct)
        n_test_incorrect = len(test_incorrect)
    else:
        # Evaluate on training data if no test data
        auroc, accuracy = probe.evaluate(correct_hidden_states, incorrect_hidden_states)
        n_test_correct = len(correct_hidden_states)
        n_test_incorrect = len(incorrect_hidden_states)

    # Extract direction
    direction = probe.get_direction()
    if direction is not None:
        b.eval(direction)
        direction_tuple = tuple(b.tolist(direction))
    else:
        direction_tuple = ()

    return LinearProbeResult(
        direction=direction_tuple,
        bias=probe._bias,
        auroc=auroc,
        accuracy=accuracy,
        n_train_correct=len(correct_hidden_states),
        n_train_incorrect=len(incorrect_hidden_states),
        n_test_correct=n_test_correct,
        n_test_incorrect=n_test_incorrect,
    )


def compute_difference_in_means(
    correct_hidden_states: list[Array],
    incorrect_hidden_states: list[Array],
    backend: Backend | None = None,
) -> tuple[Array, float]:
    """Compute the simple difference-in-means direction.

    This is the baseline "truth direction" from Marks & Tegmark (2024).

    Args:
        correct_hidden_states: Hidden states from correct answers.
        incorrect_hidden_states: Hidden states from incorrect answers.
        backend: Backend for tensor operations.

    Returns:
        Tuple of (normalized direction vector, separation score).
        Separation score = d^2 / (var_correct + var_incorrect), a measure
        of how well the classes are separated along this direction.
    """
    b = backend or get_default_backend()

    if not correct_hidden_states or not incorrect_hidden_states:
        raise ValueError("Need at least one sample of each class")

    # Stack and compute means
    correct_arr = b.stack([b.array(s) for s in correct_hidden_states], axis=0)
    incorrect_arr = b.stack([b.array(s) for s in incorrect_hidden_states], axis=0)
    b.eval(correct_arr, incorrect_arr)

    mean_correct = b.mean(correct_arr, axis=0)
    mean_incorrect = b.mean(incorrect_arr, axis=0)
    b.eval(mean_correct, mean_incorrect)

    # Direction
    direction = mean_correct - mean_incorrect
    b.eval(direction)

    # Normalize
    norm = b.sqrt(b.sum(direction * direction))
    b.eval(norm)
    norm_val = float(b.to_scalar(norm))
    if norm_val > 1e-10:
        direction_normalized = direction / norm_val
    else:
        direction_normalized = direction
    b.eval(direction_normalized)

    # Compute separation score (Fisher's criterion)
    # Project onto direction
    proj_correct = b.matmul(correct_arr, b.reshape(direction_normalized, (-1, 1)))
    proj_incorrect = b.matmul(incorrect_arr, b.reshape(direction_normalized, (-1, 1)))
    b.eval(proj_correct, proj_incorrect)

    var_correct = float(b.to_scalar(b.var(proj_correct)))
    var_incorrect = float(b.to_scalar(b.var(proj_incorrect)))

    mean_diff_sq = norm_val ** 2
    pooled_var = var_correct + var_incorrect
    separation = mean_diff_sq / pooled_var if pooled_var > 1e-10 else 0.0

    return direction_normalized, separation


__all__ = [
    "CorrectnessProbe",
    "LinearProbeResult",
    "ProbeActivation",
    "compute_difference_in_means",
    "train_correctness_probe",
]
