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
Loss Landscape Analysis for Training Diagnostics (MLX Backend).

This is the MLX/macOS implementation. For other backends:
- CUDA/Linux: see loss_landscape_cuda.py
- JAX/TPU: see loss_landscape_jax.py

Use _platform.get_loss_landscape_computer() for automatic platform selection.

Ported from the reference Swift implementation.

Features:
- Loss surface visualization data
- Curvature estimation (Hessian eigenvalues)
- Sharpness metrics for generalization prediction
- Filter-wise normalization

Research Basis:
- arxiv:1712.09913 - Visualizing Loss Landscapes
- arxiv:2002.09572 - Sharpness-Aware Minimization

MLX-Specific:
- Uses mx.grad for automatic differentiation
- Uses mx.random.normal for direction generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)


@dataclass
class SurfacePoint:
    """A point on the loss surface."""

    x: float  # First principal direction
    y: float  # Second principal direction
    loss: float


@dataclass
class LossSurfaceData:
    """2D loss surface visualization data."""

    points: list[SurfacePoint]
    min_loss: float
    max_loss: float
    center_loss: float
    resolution: int
    scale: float


@dataclass
class CurvatureMetrics:
    """Curvature information from Hessian analysis."""

    max_eigenvalue: float
    min_eigenvalue: float
    condition_number: float
    trace: float
    sharpness: float  # max_eigenvalue / (1 + max_eigenvalue)


class LossLandscapeComputer:
    """
    Computes loss landscape visualization and curvature metrics.

    Uses filter-normalized directions for consistent scale across layers.
    """

    def __init__(self, resolution: int = 21, scale: float = 1.0):
        """
        Args:
            resolution: Number of points per dimension (default 21 = 441 total)
            scale: Range of perturbations in each direction
        """
        self.resolution = resolution
        self.scale = scale

    def compute_surface(
        self,
        model_params: dict[str, mx.array],
        loss_fn: Callable[[dict[str, mx.array]], float],
        direction1: dict[str, mx.array] | None = None,
        direction2: dict[str, mx.array] | None = None,
    ) -> LossSurfaceData:
        """
        Compute 2D loss surface around current parameters.

        Args:
            model_params: Current model parameters
            loss_fn: Function that computes loss given parameters
            direction1: First perturbation direction (random if None)
            direction2: Second perturbation direction (random if None)

        Returns:
            LossSurfaceData with grid of loss values
        """
        # Generate random directions if not provided
        if direction1 is None:
            direction1 = self._random_direction(model_params)
        if direction2 is None:
            direction2 = self._random_direction(model_params)

        # Normalize directions (filter normalization)
        direction1 = self._normalize_direction(direction1, model_params)
        direction2 = self._normalize_direction(direction2, model_params)

        # Compute center loss
        center_loss = loss_fn(model_params)

        # Create grid
        half = self.resolution // 2
        points: list[SurfacePoint] = []
        min_loss = float("inf")
        max_loss = float("-inf")

        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (i - half) / half * self.scale
                y = (j - half) / half * self.scale

                # Perturbed parameters: θ + x*d1 + y*d2
                perturbed = self._perturb(model_params, direction1, direction2, x, y)
                loss = loss_fn(perturbed)

                points.append(SurfacePoint(x=x, y=y, loss=loss))
                min_loss = min(min_loss, loss)
                max_loss = max(max_loss, loss)

        return LossSurfaceData(
            points=points,
            min_loss=min_loss,
            max_loss=max_loss,
            center_loss=center_loss,
            resolution=self.resolution,
            scale=self.scale,
        )

    def estimate_curvature(
        self,
        model_params: dict[str, mx.array],
        loss_fn: Callable[[dict[str, mx.array]], float],
        num_samples: int = 20,
        epsilon: float | None = None,
    ) -> CurvatureMetrics:
        """
        Estimate curvature metrics using Hessian-vector products.

        Uses power iteration to estimate max eigenvalue.

        Args:
            model_params: Current model parameters
            loss_fn: Loss function
            num_samples: Number of power iterations
            epsilon: Finite difference step size (dtype/scale-derived if None)

        Returns:
            CurvatureMetrics with eigenvalue estimates
        """
        epsilon = self._finite_diff_epsilon(model_params, epsilon)

        # Initialize random vector
        v = self._random_direction(model_params)
        v = self._normalize_direction(v, model_params, filter_norm=False)

        max_eigenvalue = 0.0

        # Power iteration to find max eigenvalue
        for _ in range(num_samples):
            # Hessian-vector product via finite differences
            hv = self._hessian_vector_product(model_params, loss_fn, v, epsilon)

            # Rayleigh quotient: v^T H v
            eigenvalue = self._dot_product(v, hv)
            max_eigenvalue = max(max_eigenvalue, abs(eigenvalue))

            # Normalize for next iteration
            v = self._normalize_direction(hv, model_params, filter_norm=False)

        # Estimate min eigenvalue (use negative direction)
        v_neg = {k: -arr for k, arr in v.items()}
        for _ in range(num_samples // 2):
            hv = self._hessian_vector_product(model_params, loss_fn, v_neg, epsilon)
            v_neg = self._normalize_direction(hv, model_params, filter_norm=False)

        min_eigenvalue = abs(
            self._dot_product(
                v_neg, self._hessian_vector_product(model_params, loss_fn, v_neg, epsilon)
            )
        )

        # Estimate trace using random vectors
        trace = 0.0
        for _ in range(5):
            r = self._random_direction(model_params)
            r = self._normalize_direction(r, model_params, filter_norm=False)
            hr = self._hessian_vector_product(model_params, loss_fn, r, epsilon)
            trace += self._dot_product(r, hr)
        trace /= 5

        precision_eps = self._precision_epsilon(model_params)
        condition_number = max_eigenvalue / max(min_eigenvalue, precision_eps)
        sharpness = max_eigenvalue / (1.0 + max_eigenvalue)

        return CurvatureMetrics(
            max_eigenvalue=max_eigenvalue,
            min_eigenvalue=min_eigenvalue,
            condition_number=condition_number,
            trace=trace,
            sharpness=sharpness,
        )

    def _random_direction(self, params: dict[str, mx.array]) -> dict[str, mx.array]:
        """Generate random direction with same structure as params."""
        return {k: mx.random.normal(v.shape) for k, v in params.items()}

    def _normalize_direction(
        self,
        direction: dict[str, mx.array],
        params: dict[str, mx.array],
        filter_norm: bool = True,
    ) -> dict[str, mx.array]:
        """
        Normalize direction, optionally using filter normalization.

        Filter normalization scales each tensor to match the norm of
        corresponding parameters, providing architecture-independent
        visualization.
        """
        eps = self._precision_epsilon(params)
        if filter_norm:
            # Filter-wise normalization using geodesic norms
            _b = get_default_backend()
            result = {}
            for k in direction:
                d = direction[k]
                p = params[k]
                # Compute geodesic norm for each tensor
                d_flat = _b.reshape(_b.array(d.flatten()), (1, -1))
                p_flat = _b.reshape(_b.array(p.flatten()), (1, -1))
                d_norm_arr = geodesic_norms(d_flat, _b)
                p_norm_arr = geodesic_norms(p_flat, _b)
                _b.eval(d_norm_arr, p_norm_arr)
                d_norm = float(_b.to_scalar(d_norm_arr))
                p_norm = float(_b.to_scalar(p_norm_arr))
                if d_norm > eps:
                    result[k] = d * (p_norm / d_norm)
                else:
                    result[k] = d
            return result
        else:
            # Global normalization using geodesic norms
            _b = get_default_backend()
            total_norm_sq = 0.0
            for d in direction.values():
                d_flat = _b.reshape(_b.array(d.flatten()), (1, -1))
                d_norm_arr = geodesic_norms(d_flat, _b)
                _b.eval(d_norm_arr)
                d_norm = float(_b.to_scalar(d_norm_arr))
                total_norm_sq += d_norm * d_norm
            total_norm = sqrt_scalar(total_norm_sq, _b)

            if total_norm > eps:
                return {k: d / total_norm for k, d in direction.items()}
            return direction

    def _perturb(
        self,
        params: dict[str, mx.array],
        d1: dict[str, mx.array],
        d2: dict[str, mx.array],
        x: float,
        y: float,
    ) -> dict[str, mx.array]:
        """Perturb parameters: θ + x*d1 + y*d2."""
        return {k: params[k] + x * d1[k] + y * d2[k] for k in params}

    def _hessian_vector_product(
        self,
        params: dict[str, mx.array],
        loss_fn: Callable[[dict[str, mx.array]], float],
        v: dict[str, mx.array],
        epsilon: float,
    ) -> dict[str, mx.array]:
        """
        Compute Hessian-vector product via finite differences.

        H*v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)
        """
        # Forward perturbation
        params_plus = {k: params[k] + epsilon * v[k] for k in params}
        grad_plus = self._compute_gradient(params_plus, loss_fn, epsilon)

        # Backward perturbation
        params_minus = {k: params[k] - epsilon * v[k] for k in params}
        grad_minus = self._compute_gradient(params_minus, loss_fn, epsilon)

        # Hessian-vector product
        return {k: (grad_plus[k] - grad_minus[k]) / (2 * epsilon) for k in params}

    def _compute_gradient(
        self,
        params: dict[str, mx.array],
        loss_fn: Callable[[dict[str, mx.array]], float],
        epsilon: float | None = None,
    ) -> dict[str, mx.array]:
        """Compute gradient using MLX autodiff."""

        def loss_wrapper(*flat_params):
            # Reconstruct dict from flat params
            param_dict = dict(zip(params.keys(), flat_params))
            result = loss_fn(param_dict)
            return result if isinstance(result, mx.array) else mx.array(result)

        # Probe loss function output to decide autodiff vs finite differences.
        sample = loss_fn(params)
        if isinstance(sample, mx.array):
            flat_params = list(params.values())
            grads = mx.grad(loss_wrapper)(*flat_params)
            if isinstance(grads, mx.array):
                grads = [grads]
            return dict(zip(params.keys(), grads))

        # Fallback: numeric gradients for scalar Python loss functions using MLX operations
        step = self._finite_diff_epsilon(params, epsilon)
        gradients: dict[str, mx.array] = {}
        for name, param in params.items():
            grad = mx.zeros_like(param)

            # Flatten for indexing
            param_flat = mx.reshape(param, (-1,))
            grad_flat = mx.reshape(grad, (-1,))

            # Build gradient element by element
            grad_values = []
            for i in range(param_flat.size):
                # Create perturbation
                perturb_flat = mx.zeros_like(param_flat)
                perturb_flat[i] = step
                perturb = mx.reshape(perturb_flat, param.shape)

                params_plus = dict(params)
                params_minus = dict(params)
                params_plus[name] = param + perturb
                params_minus[name] = param - perturb

                loss_plus = loss_fn(params_plus)
                loss_minus = loss_fn(params_minus)

                grad_val = (float(loss_plus) - float(loss_minus)) / (2.0 * step)
                grad_values.append(grad_val)

            grad_flat = mx.array(grad_values)
            gradients[name] = mx.reshape(grad_flat, param.shape)
        return gradients

    def _precision_epsilon(self, params: dict[str, mx.array]) -> float:
        """Derive epsilon from dtype precision of the parameters."""
        _b = get_default_backend()
        for value in params.values():
            if hasattr(value, "shape"):
                ref = _b.array(value)
                return float(division_epsilon(_b, ref))
        return float(division_epsilon(_b, _b.array([1.0])))

    def _finite_diff_epsilon(
        self,
        params: dict[str, mx.array],
        epsilon: float | None,
    ) -> float:
        """Finite difference step derived from dtype precision and parameter scale."""
        if epsilon is not None:
            return float(epsilon)
        _b = get_default_backend()
        max_norm = 0.0
        for value in params.values():
            if not hasattr(value, "shape"):
                continue
            p_arr = _b.array(value)
            p_flat = _b.reshape(p_arr, (-1,))
            norm_sq = _b.sum(p_flat * p_flat)
            norm = _b.sqrt(norm_sq)
            _b.eval(norm)
            norm_val = float(_b.to_scalar(norm))
            if norm_val > max_norm:
                max_norm = norm_val
        scale = max(max_norm, 1.0)
        return self._precision_epsilon(params) * scale

    def _dot_product(
        self,
        a: dict[str, mx.array],
        b: dict[str, mx.array],
    ) -> float:
        """Compute geodesic dot between two parameter dicts."""
        _b = get_default_backend()
        flat_a = []
        flat_b = []
        for k in a:
            if k not in b:
                continue
            a_val = a[k] if hasattr(a[k], "shape") else _b.array(a[k])
            b_val = b[k] if hasattr(b[k], "shape") else _b.array(b[k])
            flat_a.append(_b.reshape(a_val, (-1,)))
            flat_b.append(_b.reshape(b_val, (-1,)))
        if not flat_a or not flat_b:
            return 0.0
        vec_a = _b.concatenate(flat_a, axis=0)
        vec_b = _b.concatenate(flat_b, axis=0)
        a_mat = _b.reshape(vec_a, (1, -1))
        b_mat = _b.reshape(vec_b, (1, -1))
        cos_arr, _ = geodesic_pairwise_metrics(a_mat, b_mat, _b)
        norm_a = geodesic_norms(a_mat, _b)
        norm_b = geodesic_norms(b_mat, _b)
        _b.eval(cos_arr, norm_a, norm_b)
        cos_val = float(_b.to_scalar(cos_arr[0]))
        norm_a_val = float(_b.to_scalar(norm_a[0]))
        norm_b_val = float(_b.to_scalar(norm_b[0]))
        return cos_val * norm_a_val * norm_b_val
