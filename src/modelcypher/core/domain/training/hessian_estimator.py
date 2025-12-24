from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

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


def gradient_quality(per_sample_gradients: list[dict[str, np.ndarray]]) -> GradientQualityMetrics | None:
    if len(per_sample_gradients) <= 1:
        return None

    flat_gradients = [_flatten_parameters(sample) for sample in per_sample_gradients]
    if not flat_gradients:
        return None

    stacked = np.stack(flat_gradients, axis=0)
    mean_grad = stacked.mean(axis=0)
    centered = stacked - mean_grad
    squared_diffs = np.sum(centered * centered, axis=1)
    variance = float(np.mean(squared_diffs))
    mean_norm_sq = float(np.sum(mean_grad * mean_grad))
    mean_norm = float(np.sqrt(mean_norm_sq))
    snr = mean_norm_sq / variance if variance > 0 else float("inf")

    return GradientQualityMetrics(variance=variance, snr=snr, mean_norm=mean_norm)


def per_layer_analysis(gradients: dict[str, np.ndarray], active_threshold: float = 0.05) -> PerLayerStats:
    norms: dict[str, float] = {}
    total_squared = 0.0
    for key, grad in gradients.items():
        norm = float(np.linalg.norm(grad.astype(np.float32)))
        norms[key] = norm
        total_squared += norm * norm

    total_norm = float(np.sqrt(total_squared))
    fractions: dict[str, float] = {}
    active_layers: list[str] = []
    for key, norm_value in norms.items():
        fraction = norm_value / total_norm if total_norm > 0 else 0.0
        fractions[key] = float(fraction)
        if fraction > active_threshold:
            active_layers.append(key)

    return PerLayerStats(norms=norms, fractions=fractions, active_layers=sorted(active_layers))


def trajectory(current_params: dict[str, np.ndarray], initial_params: dict[str, np.ndarray]) -> TrajectoryMetrics | None:
    if not current_params or not initial_params:
        return None

    divergence_sq = 0.0
    dot_product = 0.0
    current_norm_sq = 0.0
    initial_norm_sq = 0.0

    for key, current in current_params.items():
        initial = initial_params.get(key)
        if initial is None:
            continue
        curr = current.astype(np.float32)
        init = initial.astype(np.float32)
        delta = curr - init
        divergence_sq += float(np.sum(delta * delta))
        dot_product += float(np.sum(curr * init))
        current_norm_sq += float(np.sum(curr * curr))
        initial_norm_sq += float(np.sum(init * init))

    divergence = float(np.sqrt(divergence_sq))
    denom = max(np.sqrt(current_norm_sq) * np.sqrt(initial_norm_sq), 1e-10)
    cosine = float(dot_product / denom) if denom > 0 else 0.0

    return TrajectoryMetrics(divergence=divergence, cosine_similarity=cosine)


def effective_step_ratio(
    actual_step: dict[str, np.ndarray],
    gradient: dict[str, np.ndarray],
    learning_rate: float,
) -> float | None:
    if not actual_step or not gradient or learning_rate <= 0:
        return None

    actual_sq = 0.0
    theoretical_sq = 0.0
    for key, actual in actual_step.items():
        grad = gradient.get(key)
        if grad is None:
            continue
        act = actual.astype(np.float32)
        theo = learning_rate * grad.astype(np.float32)
        actual_sq += float(np.sum(act * act))
        theoretical_sq += float(np.sum(theo * theo))

    actual_norm = float(np.sqrt(actual_sq))
    theoretical_norm = float(np.sqrt(theoretical_sq))
    denom = max(theoretical_norm, 1e-10)
    return float(actual_norm / denom)


def hutchinson_trace_estimate(
    loss_and_grad_function: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, dict[str, np.ndarray]]],
    trainable_params: dict[str, np.ndarray],
    config: Config,
) -> float | None:
    if not trainable_params or config.hutchinson_vectors <= 0:
        return None

    trace_sum = 0.0
    successful = 0
    for seed in range(config.hutchinson_vectors):
        direction = _generate_rademacher_direction(trainable_params, seed=seed)
        hvp = _hessian_vector_product(loss_and_grad_function, trainable_params, direction, config)
        if hvp is None:
            continue
        zhz = 0.0
        for key, z_val in direction.items():
            hv_val = hvp.get(key)
            if hv_val is None:
                continue
            zhz += float(np.dot(z_val.astype(np.float32).ravel(), hv_val.astype(np.float32).ravel()))
        trace_sum += zhz
        successful += 1

    if successful == 0:
        return None
    return trace_sum / float(successful)


def top_eigenvalue(
    loss_and_grad_function: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, dict[str, np.ndarray]]],
    trainable_params: dict[str, np.ndarray],
    config: Config,
) -> float | None:
    if not trainable_params or config.power_iterations <= 0:
        return None

    v = _generate_normal_direction(trainable_params, seed=12345)
    v = _normalize_direction(v)
    eigenvalue = 0.0
    prev_eigenvalue = float("inf")

    for _ in range(config.power_iterations):
        hv = _hessian_vector_product(loss_and_grad_function, trainable_params, v, config)
        if hv is None:
            return None
        rayleigh = 0.0
        for key, v_val in v.items():
            hv_val = hv.get(key)
            if hv_val is None:
                continue
            rayleigh += float(np.dot(v_val.astype(np.float32).ravel(), hv_val.astype(np.float32).ravel()))
        eigenvalue = rayleigh
        if abs(eigenvalue - prev_eigenvalue) < config.power_iteration_tolerance:
            break
        prev_eigenvalue = eigenvalue
        v = _normalize_direction(hv)

    return abs(float(eigenvalue))


def condition_proxy(top_eigenvalue: float, trace_estimate: float, parameter_count: int) -> float | None:
    if parameter_count <= 0 or trace_estimate == 0:
        return None
    avg_eigenvalue = trace_estimate / float(parameter_count)
    if avg_eigenvalue <= 0:
        return None
    return float(top_eigenvalue / avg_eigenvalue)


def _flatten_parameters(params: dict[str, np.ndarray]) -> np.ndarray:
    flattened = [params[key].astype(np.float32).ravel() for key in sorted(params.keys())]
    if not flattened:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(flattened, axis=0)


def _generate_rademacher_direction(
    params: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction: dict[str, np.ndarray] = {}
    for key, value in params.items():
        shape = value.shape
        samples = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        direction[key] = np.where(samples < 0.5, -1.0, 1.0).astype(np.float32)
    return direction


def _generate_normal_direction(
    params: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction: dict[str, np.ndarray] = {}
    for key, value in params.items():
        direction[key] = rng.standard_normal(size=value.shape).astype(np.float32)
    return direction


def _normalize_direction(direction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    norm_sq = 0.0
    for value in direction.values():
        norm_sq += float(np.sum(value.astype(np.float32) ** 2))
    norm = float(np.sqrt(norm_sq))
    if norm <= 0:
        return direction
    normalized: dict[str, np.ndarray] = {}
    for key, value in direction.items():
        normalized[key] = (value.astype(np.float32) / norm).astype(np.float32)
    return normalized


def _hessian_vector_product(
    loss_and_grad_function: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, dict[str, np.ndarray]]],
    current_params: dict[str, np.ndarray],
    direction: dict[str, np.ndarray],
    config: Config,
) -> dict[str, np.ndarray] | None:
    if not current_params or not direction:
        return None

    epsilon = float(config.finite_difference_epsilon)
    plus_params: dict[str, np.ndarray] = {}
    minus_params: dict[str, np.ndarray] = {}
    for key, param in current_params.items():
        dir_vec = direction.get(key)
        if dir_vec is None:
            plus_params[key] = param
            minus_params[key] = param
            continue
        plus_params[key] = param.astype(np.float32) + epsilon * dir_vec.astype(np.float32)
        minus_params[key] = param.astype(np.float32) - epsilon * dir_vec.astype(np.float32)

    _, grad_plus = loss_and_grad_function(plus_params)
    _, grad_minus = loss_and_grad_function(minus_params)

    hvp: dict[str, np.ndarray] = {}
    denom = 2.0 * epsilon
    for key, g_plus in grad_plus.items():
        g_minus = grad_minus.get(key)
        if g_minus is None:
            continue
        hvp[key] = (g_plus.astype(np.float32) - g_minus.astype(np.float32)) / denom

    return hvp or None
