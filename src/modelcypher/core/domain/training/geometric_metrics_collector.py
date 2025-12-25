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

from __future__ import annotations

from typing import Callable

import numpy as np

from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
    GeometricMetricsHistory,
    GeometricTrainingMetrics,
    GeometryMetricKey,
)
from modelcypher.core.domain.training.hessian_estimator import (
    condition_proxy,
    config_for_level,
    effective_step_ratio,
    gradient_quality,
    hutchinson_trace_estimate,
    per_layer_analysis,
    top_eigenvalue,
    trajectory,
)


class GeometricMetricsCollector:
    def __init__(
        self, level: GeometricInstrumentationLevel = GeometricInstrumentationLevel.moderate
    ) -> None:
        self.level = level
        self.initial_parameters: dict[str, np.ndarray] | None = None
        self.previous_parameters: dict[str, np.ndarray] | None = None
        self.history = GeometricMetricsHistory()
        self.last_metrics: GeometricTrainingMetrics | None = None

    def set_level(self, new_level: GeometricInstrumentationLevel) -> None:
        self.level = new_level

    def capture_initial_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.initial_parameters = self._clone_params(params)
        self.previous_parameters = self._clone_params(params)

    def reset(self) -> None:
        self.initial_parameters = None
        self.previous_parameters = None
        self.history = GeometricMetricsHistory()
        self.last_metrics = None

    def should_compute_metrics(self, step: int) -> bool:
        if self.level is GeometricInstrumentationLevel.minimal:
            return False
        return step % self.level.hessian_computation_interval == 0

    def compute_metrics(
        self,
        trainable_params: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
        loss_and_grad_function: Callable[
            [dict[str, np.ndarray]], tuple[np.ndarray, dict[str, np.ndarray]]
        ]
        | None = None,
    ) -> GeometricTrainingMetrics:
        per_layer_stats = (
            per_layer_analysis(gradients) if self.level.compute_per_layer_metrics else None
        )

        param_divergence = None
        param_cosine_similarity = None
        if self.initial_parameters:
            traj = trajectory(trainable_params, self.initial_parameters)
            if traj:
                param_divergence = traj.divergence
                param_cosine_similarity = traj.cosine_similarity

        step_ratio = None
        if self.previous_parameters:
            actual_step = {
                key: trainable_params[key] - prev
                for key, prev in self.previous_parameters.items()
                if key in trainable_params
            }
            step_ratio = effective_step_ratio(actual_step, gradients, learning_rate)

        hessian_trace = None
        top_eigen = None
        condition = None
        if self.level.compute_top_eigenvalue and loss_and_grad_function:
            config = config_for_level(self.level)
            top_eigen = top_eigenvalue(loss_and_grad_function, trainable_params, config)
            if self.level is GeometricInstrumentationLevel.research:
                hessian_trace = hutchinson_trace_estimate(
                    loss_and_grad_function, trainable_params, config
                )
                if hessian_trace is not None and top_eigen is not None:
                    param_count = int(sum(value.size for value in trainable_params.values()))
                    condition = condition_proxy(top_eigen, hessian_trace, param_count)

        self.previous_parameters = self._clone_params(trainable_params)

        metrics = GeometricTrainingMetrics(
            hessian_trace_estimate=hessian_trace,
            top_hessian_eigenvalue=top_eigen,
            hessian_condition_proxy=condition,
            gradient_variance=None,
            gradient_snr=None,
            effective_step_ratio=step_ratio,
            per_layer_gradient_norms=per_layer_stats.norms if per_layer_stats else {},
            per_layer_gradient_fractions=per_layer_stats.fractions if per_layer_stats else {},
            active_layers=per_layer_stats.active_layers if per_layer_stats else [],
            parameter_divergence=param_divergence,
            parameter_cosine_similarity=param_cosine_similarity,
        )

        self.last_metrics = metrics
        return metrics

    def compute_gradient_quality(
        self,
        per_sample_gradients: list[dict[str, np.ndarray]],
    ) -> tuple[float, float] | None:
        quality = gradient_quality(per_sample_gradients)
        if not quality:
            return None
        return quality.variance, quality.snr

    def record_in_history(self, step: int, metrics: GeometricTrainingMetrics) -> None:
        self.history.append(step=step, metrics=metrics)

    def get_history(self) -> GeometricMetricsHistory:
        return self.history

    def get_last_metrics(self) -> GeometricTrainingMetrics | None:
        return self.last_metrics

    def compute_lightweight_metrics(
        self,
        trainable_params: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}

        per_layer_stats = per_layer_analysis(gradients)
        top_layers = sorted(
            per_layer_stats.fractions.items(), key=lambda item: item[1], reverse=True
        )[:5]
        for layer, fraction in top_layers:
            short_name = self.shorten_layer_name(layer)
            result[GeometryMetricKey.layer_grad_fraction(short_name)] = float(fraction)

        if self.initial_parameters:
            traj = trajectory(trainable_params, self.initial_parameters)
            if traj:
                result[GeometryMetricKey.param_divergence] = float(traj.divergence)
                result[GeometryMetricKey.param_cosine_similarity] = float(traj.cosine_similarity)

        if self.previous_parameters:
            actual_step = {
                key: trainable_params[key] - prev
                for key, prev in self.previous_parameters.items()
                if key in trainable_params
            }
            ratio = effective_step_ratio(actual_step, gradients, learning_rate)
            if ratio is not None:
                result[GeometryMetricKey.effective_step_ratio] = float(ratio)

        self.previous_parameters = self._clone_params(trainable_params)
        return result

    @staticmethod
    def shorten_layer_name(full_name: str) -> str:
        short = (
            full_name.replace("layers.", "L")
            .replace("attention", "attn")
            .replace("mlp", "mlp")
            .replace(".lora_a", ".a")
            .replace(".lora_b", ".b")
            .replace("self_attn", "attn")
            .replace("feed_forward", "ff")
        )
        if len(short) > 30:
            short = short[:30]
        return short

    @staticmethod
    def _clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        cloned: dict[str, np.ndarray] = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                cloned[key] = value.copy()
            else:
                cloned[key] = np.array(value, copy=True)
        return cloned
