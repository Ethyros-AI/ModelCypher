# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Continual learning geometric telemetry metrics.

Provides the mathematical definitions for tracking sub-space collisions,
null space capacity, Weyl bounds, and capability preservation via CKA.
All geometric metrics run exclusively through the unified Backend protocol.

References:
    - ModelCypher Geometric Continual Learning Thesis
    - Chaudhry et al. 2018 (Forgetting measure)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.null_space_tracker import NullSpaceTracker
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class GeometricTelemetrySummary:
    """Summary of the geometric health of a continual learning integration."""

    null_space_depletion_rate: float | None
    weyl_accumulation: float
    min_cka_stability: float | None
    mean_cka_stability: float | None


@dataclass(frozen=True)
class StandardCLSummary:
    """Standard continual learning performance metrics."""

    average_accuracy: float | None
    backward_transfer: float | None
    forward_transfer: float | None
    forgetting_measure: float | None


class ContinualLearningMetrics:
    """CPU-bound operations for lists of scalars (e.g. CKA matrices, accuracy arrays)."""

    @staticmethod
    def rank_from_tracker(tracker: "NullSpaceTracker", layer_id: str) -> float | None:
        """Extract the current null-space rank for a layer from the NullSpaceTracker.

        Args:
            tracker: The active NullSpaceTracker instance.
            layer_id: The target layer identifier.

        Returns:
            Current null-space rank float, or None if layer not tracked.
        """
        state = tracker.get_layer_state(layer_id)
        if state is None:
            return None
        return float(state.null_rank)

    @staticmethod
    def null_space_depletion_rate(task_ranks: list[float]) -> float | None:
        """Calculate the rate of null-space rank depletion.

        Args:
            task_ranks: List of null-space ranks measured after each task sequence.

        Returns:
            Depletion rate per task, or None if insufficient history.
        """
        if len(task_ranks) < 2:
            return None

        delta = task_ranks[-1] - task_ranks[0]
        return float(delta) / float(len(task_ranks) - 1)

    @staticmethod
    def cka_stability(cka_history: list[list[float]]) -> dict[str, float | None]:
        """Calculate CKA stability metrics across sequential tasks.

        Args:
            cka_history: A matrix where `cka_history[i][j]` is the CKA of task `j`
                         evaluated after training sequentially on task `i` (i >= j).

        Returns:
            Dictionary containing min and mean CKA stability.
        """
        if not cka_history or len(cka_history) < 2:
            return {"min": None, "mean": None}

        # The latest state is the last row. We check retention on all prior tasks.
        latest_evals = cka_history[-1]

        # Exclude evaluating the latest task against itself
        prior_evals = latest_evals[:-1]
        if not prior_evals:
            return {"min": None, "mean": None}

        return {"min": min(prior_evals), "mean": sum(prior_evals) / float(len(prior_evals))}

    @staticmethod
    def standard_cl_metrics(
        accuracy_matrix: list[list[float]], random_baselines: list[float] | None = None
    ) -> StandardCLSummary:
        """Calculate standard Continual Learning metrics.

        Evaluates Forward Transfer (FWT), Backward Transfer (BWT), Forgetting,
        and Average Accuracy according to standard literature (e.g., Chaudhry 2018).

        Args:
            accuracy_matrix: R where R[i][j] is test accuracy on task j after observing task i.
                             If j > i, R[i][j] is the zero-shot accuracy before training j.
            random_baselines: Optional list b where b[i] is random-init accuracy on task i.
                              Required for correct FWT calculation.

        Returns:
            StandardCLSummary containing the classical metrics.
        """
        if not accuracy_matrix:
            return StandardCLSummary(None, None, None, None)

        N = len(accuracy_matrix)
        if N == 0:
            return StandardCLSummary(None, None, None, None)

        # Average accuracy at the end of training all tasks
        final_accuracies = accuracy_matrix[N - 1][:N]
        avg_acc = sum(final_accuracies) / float(len(final_accuracies))

        if N < 2:
            return StandardCLSummary(avg_acc, None, None, None)

        # Backward Transfer (BWT) and Forgetting
        bwt_sum = 0.0
        forget_sum = 0.0

        for j in range(N - 1):
            bwt_sum += accuracy_matrix[N - 1][j] - accuracy_matrix[j][j]
            max_acc = max(accuracy_matrix[i][j] for i in range(j, N - 1))
            forget_sum += max_acc - accuracy_matrix[N - 1][j]

        bwt = bwt_sum / float(N - 1)
        forgetting = forget_sum / float(N - 1)

        # Forward Transfer (FWT)
        fwt_sum = 0.0
        valid_fwt_count = 0
        for i in range(1, N):
            if len(accuracy_matrix[i - 1]) > i:
                b_i = random_baselines[i] if random_baselines and i < len(random_baselines) else 0.0
                fwt_sum += accuracy_matrix[i - 1][i] - b_i
                valid_fwt_count += 1

        fwt = fwt_sum / float(valid_fwt_count) if valid_fwt_count > 0 else None

        return StandardCLSummary(
            average_accuracy=avg_acc,
            backward_transfer=bwt,
            forward_transfer=fwt,
            forgetting_measure=forgetting,
        )


class BackendContinualLearningMetrics:
    """GPU-accelerated tensor metrics using the Backend protocol."""

    def __init__(self, backend: "Backend"):
        self.backend = backend
        self._finfo = backend.finfo()

    def spectral_budget_trajectory(self, delta_history: list[Array], sigma_k: float) -> list[float]:
        """Track the capacity consumed via symmetric perturbations.

        Args:
            delta_history: List of ΔW tensors added for each task.
            sigma_k: The targeted spectral bound (e.g., k-th singular value).

        Returns:
            List of spectral norm ratios ||ΔW||_2 / sigma_k for each task.
        """
        trajectory = []
        # sigma_k is a physical spectral constant; do not enforce eps division
        # if sigma_k is genuinely 0, the rank is zero, division by eps is a safe fallback
        safe_sigma = max(sigma_k, self._finfo.eps)

        for delta in delta_history:
            # ||ΔW||_2 (Spectral norm via max singular value)
            S = self.backend.svd(delta, compute_uv=False)
            norm_tensor = self.backend.max(S)
            self.backend.eval(norm_tensor)
            norm_val = self._to_scalar(norm_tensor)
            trajectory.append(norm_val / float(safe_sigma))

        return trajectory

    def weyl_accumulation(self, delta_history: list[Array]) -> float:
        """Calculate cumulative spectral perturbation across tasks.

        Tracks Σ ||δ_i||_2 to verify Weyl bounding bounds over time.

        Args:
            delta_history: List of parameter deltas.

        Returns:
            Total cumulative spectral perturbation.
        """
        accumulation = 0.0
        for delta in delta_history:
            # ||ΔW||_2 (Spectral norm)
            S = self.backend.svd(delta, compute_uv=False)
            norm_tensor = self.backend.max(S)
            self.backend.eval(norm_tensor)
            accumulation += self._to_scalar(norm_tensor)

        return float(accumulation)

    def _to_scalar(self, val: Any) -> float:
        """Convert backend scalar to Python float."""
        if hasattr(val, "shape") or hasattr(val, "item") or hasattr(val, "tolist"):
            self.backend.eval(val)
            return float(self.backend.to_scalar(val))
        try:
            return float(val)
        except (TypeError, ValueError) as e:
            if isinstance(val, (list, tuple)) and val:
                return float(val[0])
            raise ValueError(f"Cannot convert to scalar: {val}") from e


def get_continual_learning_metrics(
    backend: "Backend | None" = None,
) -> ContinualLearningMetrics | BackendContinualLearningMetrics:
    """Get the appropriate continual learning metrics implementation."""
    if backend is not None:
        return BackendContinualLearningMetrics(backend)
    return ContinualLearningMetrics()
