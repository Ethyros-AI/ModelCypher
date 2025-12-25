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

import math
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


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
        epsilon: float = 1e-3,
    ) -> CurvatureMetrics:
        """
        Estimate curvature metrics using Hessian-vector products.

        Uses power iteration to estimate max eigenvalue.

        Args:
            model_params: Current model parameters
            loss_fn: Loss function
            num_samples: Number of power iterations
            epsilon: Finite difference step size

        Returns:
            CurvatureMetrics with eigenvalue estimates
        """
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

        condition_number = max_eigenvalue / max(min_eigenvalue, 1e-10)
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
        if filter_norm:
            # Filter-wise normalization
            result = {}
            for k in direction:
                d = direction[k]
                p = params[k]
                d_norm = float(mx.sqrt(mx.sum(d**2)).item())
                p_norm = float(mx.sqrt(mx.sum(p**2)).item())
                if d_norm > 1e-10:
                    result[k] = d * (p_norm / d_norm)
                else:
                    result[k] = d
            return result
        else:
            # Global normalization
            total_norm = 0.0
            for d in direction.values():
                total_norm += float(mx.sum(d**2).item())
            total_norm = math.sqrt(total_norm)

            if total_norm > 1e-10:
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
        step = float(epsilon) if epsilon is not None else 1e-4
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
                perturb_flat = mx.scatter(perturb_flat, mx.array([i]), mx.array([step]))
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

    def _dot_product(
        self,
        a: dict[str, mx.array],
        b: dict[str, mx.array],
    ) -> float:
        """Compute dot product between two parameter dicts."""
        total = 0.0
        for k in a:
            if k in b:
                total += float(mx.sum(a[k] * b[k]).item())
        return total
