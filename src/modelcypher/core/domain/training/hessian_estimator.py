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

import math
from dataclasses import dataclass
from typing import Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
)


@dataclass(frozen=True)
class Config:
    hutchinson_vectors: int = 5
    power_iterations: int = 20
    finite_difference_epsilon: float = 1e-4
    power_iteration_tolerance: float = 1e-6

    @staticmethod
    def moderate() -> "Config":
        return Config(hutchinson_vectors=3, power_iterations=10, finite_difference_epsilon=1e-3)

    @staticmethod
    def full() -> "Config":
        return Config(hutchinson_vectors=10, power_iterations=30, finite_difference_epsilon=1e-5)


@dataclass(frozen=True)
class GradientQualityMetrics:
    variance: float
    snr: float
    mean_norm: float


@dataclass(frozen=True)
class PerLayerStats:
    norms: dict[str, float]
    fractions: dict[str, float]
    active_layers: list[str]


@dataclass(frozen=True)
class TrajectoryMetrics:
    divergence: float
    cosine_similarity: float


def config_for_level(level: GeometricInstrumentationLevel) -> Config:
    if level is GeometricInstrumentationLevel.minimal:
        return Config(hutchinson_vectors=0, power_iterations=0)
    if level is GeometricInstrumentationLevel.moderate:
        return Config.moderate()
    return Config.full()


def gradient_quality(
    per_sample_gradients: list[dict[str, "Array"]],
) -> GradientQualityMetrics | None:
    if len(per_sample_gradients) <= 1:
        return None

    backend = get_default_backend()
    flat_gradients = [_flatten_parameters(sample, backend) for sample in per_sample_gradients]
    if not flat_gradients:
        return None

    stacked = backend.stack(flat_gradients, axis=0)
    mean_grad = backend.mean(stacked, axis=0)
    centered = stacked - mean_grad
    squared_diffs = backend.sum(centered * centered, axis=1)
    variance = float(backend.mean(squared_diffs))
    mean_norm_sq = float(backend.sum(mean_grad * mean_grad))
    mean_norm = float(math.sqrt(mean_norm_sq))
    snr = mean_norm_sq / variance if variance > 0 else float("inf")

    return GradientQualityMetrics(variance=variance, snr=snr, mean_norm=mean_norm)


def per_layer_analysis(
    gradients: dict[str, "Array"], active_threshold: float = 0.05
) -> PerLayerStats:
    backend = get_default_backend()
    norms: dict[str, float] = {}
    total_squared = 0.0
    for key, grad in gradients.items():
        grad_arr = backend.array(grad)
        backend.eval(grad_arr)
        norm = float(backend.norm(grad_arr))
        norms[key] = norm
        total_squared += norm * norm

    total_norm = float(math.sqrt(total_squared))
    fractions: dict[str, float] = {}
    active_layers: list[str] = []
    for key, norm_value in norms.items():
        fraction = norm_value / total_norm if total_norm > 0 else 0.0
        fractions[key] = float(fraction)
        if fraction > active_threshold:
            active_layers.append(key)

    return PerLayerStats(norms=norms, fractions=fractions, active_layers=sorted(active_layers))


def trajectory(
    current_params: dict[str, "Array"], initial_params: dict[str, "Array"]
) -> TrajectoryMetrics | None:
    if not current_params or not initial_params:
        return None

    backend = get_default_backend()
    divergence_sq = 0.0
    dot_product = 0.0
    current_norm_sq = 0.0
    initial_norm_sq = 0.0

    for key, current in current_params.items():
        initial = initial_params.get(key)
        if initial is None:
            continue
        current_arr = backend.array(current)
        initial_arr = backend.array(initial)
        backend.eval(current_arr, initial_arr)
        delta = current_arr - initial_arr
        divergence_sq += float(backend.sum(delta * delta))
        dot_product += float(backend.sum(current_arr * initial_arr))
        current_norm_sq += float(backend.sum(current_arr * current_arr))
        initial_norm_sq += float(backend.sum(initial_arr * initial_arr))

    divergence = float(math.sqrt(divergence_sq))
    denom = max(math.sqrt(current_norm_sq) * math.sqrt(initial_norm_sq), 1e-10)
    cosine = float(dot_product / denom) if denom > 0 else 0.0

    return TrajectoryMetrics(divergence=divergence, cosine_similarity=cosine)


def effective_step_ratio(
    actual_step: dict[str, "Array"],
    gradient: dict[str, "Array"],
    learning_rate: float,
) -> float | None:
    if not actual_step or not gradient or learning_rate <= 0:
        return None

    backend = get_default_backend()
    actual_sq = 0.0
    theoretical_sq = 0.0
    for key, actual in actual_step.items():
        grad = gradient.get(key)
        if grad is None:
            continue
        actual_arr = backend.array(actual)
        grad_arr = backend.array(grad)
        backend.eval(actual_arr, grad_arr)
        theo = grad_arr * learning_rate
        actual_sq += float(backend.sum(actual_arr * actual_arr))
        theoretical_sq += float(backend.sum(theo * theo))

    actual_norm = float(math.sqrt(actual_sq))
    theoretical_norm = float(math.sqrt(theoretical_sq))
    denom = max(theoretical_norm, 1e-10)
    return float(actual_norm / denom)


def hutchinson_trace_estimate(
    loss_and_grad_function: Callable[
        [dict[str, "Array"]], tuple["Array", dict[str, "Array"]]
    ],
    trainable_params: dict[str, "Array"],
    config: Config,
) -> float | None:
    if not trainable_params or config.hutchinson_vectors <= 0:
        return None

    backend = get_default_backend()
    trace_sum = 0.0
    successful = 0
    for seed in range(config.hutchinson_vectors):
        direction = _generate_rademacher_direction(trainable_params, backend, seed=seed)
        hvp = _hessian_vector_product(loss_and_grad_function, trainable_params, direction, config, backend)
        if hvp is None:
            continue
        zhz = 0.0
        for key, z_val in direction.items():
            hv_val = hvp.get(key)
            if hv_val is None:
                continue
            zhz += float(backend.sum(z_val * hv_val))
        trace_sum += zhz
        successful += 1

    if successful == 0:
        return None
    return trace_sum / float(successful)


def top_eigenvalue(
    loss_and_grad_function: Callable[
        [dict[str, "Array"]], tuple["Array", dict[str, "Array"]]
    ],
    trainable_params: dict[str, "Array"],
    config: Config,
) -> float | None:
    if not trainable_params or config.power_iterations <= 0:
        return None

    backend = get_default_backend()
    v = _generate_normal_direction(trainable_params, backend, seed=12345)
    v = _normalize_direction(v, backend)
    eigenvalue = 0.0
    prev_eigenvalue = float("inf")

    for _ in range(config.power_iterations):
        hv = _hessian_vector_product(loss_and_grad_function, trainable_params, v, config, backend)
        if hv is None:
            return None
        rayleigh = 0.0
        for key, v_val in v.items():
            hv_val = hv.get(key)
            if hv_val is None:
                continue
            rayleigh += float(backend.sum(v_val * hv_val))
        eigenvalue = rayleigh
        if abs(eigenvalue - prev_eigenvalue) < config.power_iteration_tolerance:
            break
        prev_eigenvalue = eigenvalue
        v = _normalize_direction(hv, backend)

    return abs(float(eigenvalue))


def condition_proxy(
    top_eigenvalue: float, trace_estimate: float, parameter_count: int
) -> float | None:
    if parameter_count <= 0 or trace_estimate == 0:
        return None
    avg_eigenvalue = trace_estimate / float(parameter_count)
    if avg_eigenvalue <= 0:
        return None
    return float(top_eigenvalue / avg_eigenvalue)


def _flatten_parameters(params: dict[str, "Array"], backend) -> "Array":
    flattened = [backend.reshape(params[key], (-1,)) for key in sorted(params.keys())]
    if not flattened:
        return backend.zeros((0,))
    return backend.concatenate(flattened, axis=0)


def _generate_rademacher_direction(
    params: dict[str, "Array"],
    backend,
    seed: int,
) -> dict[str, "Array"]:
    backend.random_seed(seed)
    direction: dict[str, "Array"] = {}
    for key, value in params.items():
        shape = backend.shape(value)
        samples = backend.random_uniform(shape)
        # Rademacher: -1 or +1 with equal probability
        direction[key] = backend.where(samples < 0.5, backend.full(shape, -1.0), backend.full(shape, 1.0))
    return direction


def _generate_normal_direction(
    params: dict[str, "Array"],
    backend,
    seed: int,
) -> dict[str, "Array"]:
    backend.random_seed(seed)
    direction: dict[str, "Array"] = {}
    for key, value in params.items():
        shape = backend.shape(value)
        direction[key] = backend.random_normal(shape)
    return direction


def _normalize_direction(direction: dict[str, "Array"], backend) -> dict[str, "Array"]:
    norm_sq = 0.0
    for value in direction.values():
        norm_sq += float(backend.sum(value ** 2))
    norm = float(math.sqrt(norm_sq))
    if norm <= 0:
        return direction
    normalized: dict[str, "Array"] = {}
    for key, value in direction.items():
        normalized[key] = backend.divide_scalar(value, norm)
    return normalized


def _hessian_vector_product(
    loss_and_grad_function: Callable[
        [dict[str, "Array"]], tuple["Array", dict[str, "Array"]]
    ],
    current_params: dict[str, "Array"],
    direction: dict[str, "Array"],
    config: Config,
    backend,
) -> dict[str, "Array"] | None:
    if not current_params or not direction:
        return None

    epsilon = float(config.finite_difference_epsilon)
    plus_params: dict[str, "Array"] = {}
    minus_params: dict[str, "Array"] = {}
    for key, param in current_params.items():
        dir_vec = direction.get(key)
        if dir_vec is None:
            plus_params[key] = param
            minus_params[key] = param
            continue
        plus_params[key] = param + backend.multiply_scalar(dir_vec, epsilon)
        minus_params[key] = param - backend.multiply_scalar(dir_vec, epsilon)

    _, grad_plus = loss_and_grad_function(plus_params)
    _, grad_minus = loss_and_grad_function(minus_params)

    hvp: dict[str, "Array"] = {}
    denom = 2.0 * epsilon
    for key, g_plus in grad_plus.items():
        g_minus = grad_minus.get(key)
        if g_minus is None:
            continue
        hvp[key] = backend.divide_scalar(g_plus - g_minus, denom)

    return hvp or None
