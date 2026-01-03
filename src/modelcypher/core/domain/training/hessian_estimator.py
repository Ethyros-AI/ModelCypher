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

"""
Hessian property estimation for training diagnostics.

Estimates key Hessian properties without forming the full Hessian matrix (which
is O(d^2) memory for d parameters - infeasible for modern LLMs). These estimates
guide learning rate selection, detect training instability, and diagnose
optimization landscape curvature.

Algorithms:

    Hutchinson Trace Estimation:
        tr(H) = (1/m) * sum_{i=1}^{m} z_i^T H z_i

        where z_i are Rademacher random vectors (+1/-1 with equal probability).
        The estimator is unbiased: E[z^T H z] = tr(H).
        Variance decreases as O(1/m).

    Power Iteration for lambda_max(H):
        v_{k+1} = H v_k / ||H v_k||
        lambda_max = v^T H v  (Rayleigh quotient at convergence)

        Converges at rate |lambda_2 / lambda_max|^k.

    Hessian-Vector Products via Finite Differences:
        H @ v = (grad L(w + eps*v) - grad L(w - eps*v)) / (2*eps)

        Central difference achieves O(eps^2) truncation error. Epsilon balances
        truncation error (O(eps^2)) against numerical rounding (O(machine_eps / eps)).

    Condition Number Proxy:
        cond(H) ~ lambda_max / (tr(H) / d)

        Uses trace / dimension as proxy for minimum eigenvalue. Not exact,
        but tracks relative conditioning across training steps.

Configuration:

    finite_difference_epsilon and power_iteration_tolerance default to None.
    When None, they are derived from dtype machine epsilon at runtime.

Usage:
    config = Config()  # or Config.moderate() for faster, less accurate
    trace = hutchinson_trace_estimate(loss_grad_fn, params, config)
    top_eig = top_eigenvalue(loss_grad_fn, params, config)
    cond = condition_proxy(top_eig, trace, param_count)

References:
    - Hutchinson (1990) "A Stochastic Estimator of the Trace of the Influence Matrix"
    - Pearlmutter (1994) "Fast Exact Multiplication by the Hessian"
    - Yao et al. (2020) "PyHessian: Neural Networks Through the Lens of the Hessian"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array
from modelcypher.core.domain.training.geometric_training_metrics import (
    GeometricInstrumentationLevel,
)


@dataclass(frozen=True)
class Config:
    hutchinson_vectors: int = 5
    power_iterations: int = 20
    finite_difference_epsilon: float | None = None
    power_iteration_tolerance: float | None = None

    @staticmethod
    def moderate() -> "Config":
        return Config(hutchinson_vectors=3, power_iterations=10)

    @staticmethod
    def full() -> "Config":
        return Config(hutchinson_vectors=10, power_iterations=30)


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
    variance_arr = backend.mean(squared_diffs)
    mean_norm_sq_arr = backend.sum(mean_grad * mean_grad)
    backend.eval(variance_arr, mean_norm_sq_arr)
    variance = float(backend.to_scalar(variance_arr))
    mean_norm_sq = float(backend.to_scalar(mean_norm_sq_arr))
    mean_norm_arr = backend.sqrt(backend.array([mean_norm_sq]))
    backend.eval(mean_norm_arr)
    mean_norm = float(backend.to_scalar(mean_norm_arr))
    snr = mean_norm_sq / variance if variance > 0 else float("inf")

    return GradientQualityMetrics(variance=variance, snr=snr, mean_norm=mean_norm)


def per_layer_analysis(gradients: dict[str, "Array"]) -> PerLayerStats:
    backend = get_default_backend()
    norms: dict[str, float] = {}
    total_squared = 0.0
    eps_ref = _reference_array(gradients, backend)
    for key, grad in gradients.items():
        grad_arr = backend.array(grad)
        backend.eval(grad_arr)
        # Reshape to 2D row vector for geodesic norm computation
        flat_grad = backend.reshape(grad_arr, (1, -1))
        norm_arr = geodesic_norms(flat_grad, backend)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        norms[key] = norm
        total_squared += norm * norm

    total_norm = float(sqrt_scalar(total_squared, backend))
    fractions: dict[str, float] = {}
    active_layers: list[str] = []
    layer_count = len(norms)
    mean_fraction = 1.0 / float(layer_count) if layer_count > 0 else 0.0
    eps = division_epsilon(backend, eps_ref)
    for key, norm_value in norms.items():
        fraction = norm_value / total_norm if total_norm > 0 else 0.0
        fractions[key] = float(fraction)
        if fraction + eps >= mean_fraction:
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
        divergence_sq += _sum_scalar(delta * delta, backend)
        dot_product += _sum_scalar(current_arr * initial_arr, backend)
        current_norm_sq += _sum_scalar(current_arr * current_arr, backend)
        initial_norm_sq += _sum_scalar(initial_arr * initial_arr, backend)

    eps_ref = _reference_array(current_params, backend)
    divergence = float(sqrt_scalar(divergence_sq, backend))
    current_norm = sqrt_scalar(current_norm_sq, backend)
    initial_norm = sqrt_scalar(initial_norm_sq, backend)
    norm_eps = division_epsilon(backend, eps_ref)
    denom = max(current_norm * initial_norm, norm_eps)
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
        actual_sq += _sum_scalar(actual_arr * actual_arr, backend)
        theoretical_sq += _sum_scalar(theo * theo, backend)

    actual_norm = float(sqrt_scalar(actual_sq, backend))
    theoretical_norm = float(sqrt_scalar(theoretical_sq, backend))
    eps_ref = _reference_array(actual_step, backend)
    denom = max(theoretical_norm, division_epsilon(backend, eps_ref))
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
            zhz += _sum_scalar(z_val * hv_val, backend)
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
    eps_ref = _reference_array(trainable_params, backend)
    tolerance = _power_iteration_tolerance(config, backend, eps_ref)

    for _ in range(config.power_iterations):
        hv = _hessian_vector_product(loss_and_grad_function, trainable_params, v, config, backend)
        if hv is None:
            return None
        rayleigh = 0.0
        for key, v_val in v.items():
            hv_val = hv.get(key)
            if hv_val is None:
                continue
            rayleigh += _sum_scalar(v_val * hv_val, backend)
        eigenvalue = rayleigh
        if abs(eigenvalue - prev_eigenvalue) < tolerance:
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
        samples = backend.random_uniform(low=0.0, high=1.0, shape=shape)
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
        norm_sq += _sum_scalar(value ** 2, backend)
    norm = float(sqrt_scalar(norm_sq, backend))
    eps_ref = _reference_array(direction, backend)
    if norm <= division_epsilon(backend, eps_ref):
        return direction
    normalized: dict[str, "Array"] = {}
    for key, value in direction.items():
        normalized[key] = value / norm
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

    eps_ref = _reference_array(current_params, backend)
    epsilon = _finite_difference_epsilon(config, backend, eps_ref)
    plus_params: dict[str, "Array"] = {}
    minus_params: dict[str, "Array"] = {}
    for key, param in current_params.items():
        dir_vec = direction.get(key)
        if dir_vec is None:
            plus_params[key] = param
            minus_params[key] = param
            continue
        plus_params[key] = param + dir_vec * epsilon
        minus_params[key] = param - dir_vec * epsilon

    _, grad_plus = loss_and_grad_function(plus_params)
    _, grad_minus = loss_and_grad_function(minus_params)

    hvp: dict[str, "Array"] = {}
    denom = 2.0 * epsilon
    for key, g_plus in grad_plus.items():
        g_minus = grad_minus.get(key)
        if g_minus is None:
            continue
        hvp[key] = (g_plus - g_minus) / denom

    return hvp or None


def _sum_scalar(array: "Array", backend) -> float:
    """Return backend sum as a Python float with eval."""
    sum_arr = backend.sum(array)
    backend.eval(sum_arr)
    return float(backend.to_scalar(sum_arr))


def _reference_array(params: dict[str, "Array"], backend) -> "Array":
    for value in params.values():
        return backend.array(value)
    return backend.array([0.0])


def _finite_difference_epsilon(config: Config, backend, array: "Array") -> float:
    if config.finite_difference_epsilon is not None:
        return float(config.finite_difference_epsilon)
    return division_epsilon(backend, array)


def _power_iteration_tolerance(config: Config, backend, array: "Array") -> float:
    if config.power_iteration_tolerance is not None:
        return float(config.power_iteration_tolerance)
    return machine_epsilon(backend, array)
