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
JAX Loss Landscape Analysis for Training Diagnostics.

This is the JAX implementation. For other backends:
- MLX/macOS: see loss_landscape_mlx.py
- CUDA/PyTorch: see loss_landscape_cuda.py

Use _platform.get_loss_landscape_computer() for automatic platform selection.

Implementation based on JAX best practices (2025):
- jax.random for direction generation
- jax.grad for gradient computation
- jax.jit for optimized evaluation
- JAX pytree operations for parameter handling

Research Basis:
- arxiv:1712.09913 - Visualizing Loss Landscapes
- arxiv:2002.09572 - Sharpness-Aware Minimization

References:
- https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
- https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SurfacePointJAX:
    """A point on the loss surface."""
    x: float  # First principal direction
    y: float  # Second principal direction
    loss: float


@dataclass
class LossSurfaceDataJAX:
    """2D loss surface visualization data."""
    points: list[SurfacePointJAX]
    min_loss: float
    max_loss: float
    center_loss: float
    resolution: int
    scale: float


@dataclass
class CurvatureMetricsJAX:
    """Curvature information from Hessian analysis."""
    max_eigenvalue: float
    min_eigenvalue: float
    condition_number: float
    trace: float
    sharpness: float  # max_eigenvalue / (1 + max_eigenvalue)


class LossLandscapeComputerJAX:
    """
    JAX Loss Landscape Computer.

    Computes loss landscape visualization and curvature metrics
    using JAX's autodiff capabilities.

    Features (matching MLX parity):
    - Loss surface visualization data
    - Curvature estimation (Hessian eigenvalues)
    - Sharpness metrics for generalization prediction
    - Filter-wise normalization

    Uses filter-normalized directions for consistent scale across layers.
    """

    def __init__(
        self,
        resolution: int = 21,
        scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            resolution: Number of points per dimension (default 21 = 441 total)
            scale: Range of perturbations in each direction
            seed: Random seed for direction generation
        """
        self.resolution = resolution
        self.scale = scale
        self.key = jax.random.PRNGKey(seed)

    def compute_surface(
        self,
        model_params: dict[str, Any],
        loss_fn: Callable[[dict[str, Any]], float],
        direction1: dict[str, Any] | None = None,
        direction2: dict[str, Any] | None = None,
    ) -> LossSurfaceDataJAX:
        """
        Compute 2D loss surface around current parameters.

        Args:
            model_params: Current model parameters (JAX pytree)
            loss_fn: Function that computes loss given parameters
            direction1: First perturbation direction (random if None)
            direction2: Second perturbation direction (random if None)

        Returns:
            LossSurfaceDataJAX with grid of loss values
        """
        # Generate random directions if not provided
        if direction1 is None:
            self.key, subkey = jax.random.split(self.key)
            direction1 = self._random_direction(model_params, subkey)
        if direction2 is None:
            self.key, subkey = jax.random.split(self.key)
            direction2 = self._random_direction(model_params, subkey)

        # Normalize directions (filter normalization)
        direction1 = self._normalize_direction(direction1, model_params)
        direction2 = self._normalize_direction(direction2, model_params)

        # Compute center loss
        center_loss = float(loss_fn(model_params))

        # Create grid
        half = self.resolution // 2
        points: list[SurfacePointJAX] = []
        min_loss = float('inf')
        max_loss = float('-inf')

        logger.info(
            "Computing loss surface: %dx%d grid, scale=%.2f",
            self.resolution,
            self.resolution,
            self.scale,
        )

        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (i - half) / half * self.scale
                y = (j - half) / half * self.scale

                # Perturbed parameters: θ + x*d1 + y*d2
                perturbed = self._perturb(model_params, direction1, direction2, x, y)
                loss = float(loss_fn(perturbed))

                points.append(SurfacePointJAX(x=x, y=y, loss=loss))
                min_loss = min(min_loss, loss)
                max_loss = max(max_loss, loss)

        logger.info(
            "Loss surface computed: min=%.4f, max=%.4f, center=%.4f",
            min_loss,
            max_loss,
            center_loss,
        )

        return LossSurfaceDataJAX(
            points=points,
            min_loss=min_loss,
            max_loss=max_loss,
            center_loss=center_loss,
            resolution=self.resolution,
            scale=self.scale,
        )

    def estimate_curvature(
        self,
        model_params: dict[str, Any],
        loss_fn: Callable[[dict[str, Any]], float],
        num_samples: int = 20,
        epsilon: float = 1e-3,
    ) -> CurvatureMetricsJAX:
        """
        Estimate curvature metrics using Hessian-vector products.

        Uses power iteration to estimate max eigenvalue.

        Args:
            model_params: Current model parameters
            loss_fn: Loss function
            num_samples: Number of power iterations
            epsilon: Finite difference step size

        Returns:
            CurvatureMetricsJAX with eigenvalue estimates
        """
        # Initialize random vector
        self.key, subkey = jax.random.split(self.key)
        v = self._random_direction(model_params, subkey)
        v = self._normalize_direction(v, model_params, filter_norm=False)

        max_eigenvalue = 0.0

        logger.info("Estimating curvature with %d power iterations", num_samples)

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
        v_neg = jax.tree.map(lambda x: -x, v)
        for _ in range(num_samples // 2):
            hv = self._hessian_vector_product(model_params, loss_fn, v_neg, epsilon)
            v_neg = self._normalize_direction(hv, model_params, filter_norm=False)

        min_eigenvalue = abs(self._dot_product(
            v_neg,
            self._hessian_vector_product(model_params, loss_fn, v_neg, epsilon)
        ))

        # Estimate trace using random vectors
        trace = 0.0
        for _ in range(5):
            self.key, subkey = jax.random.split(self.key)
            r = self._random_direction(model_params, subkey)
            r = self._normalize_direction(r, model_params, filter_norm=False)
            hr = self._hessian_vector_product(model_params, loss_fn, r, epsilon)
            trace += self._dot_product(r, hr)
        trace /= 5

        condition_number = max_eigenvalue / max(min_eigenvalue, 1e-10)
        sharpness = max_eigenvalue / (1.0 + max_eigenvalue)

        logger.info(
            "Curvature estimated: max_eig=%.4f, min_eig=%.4f, sharpness=%.4f",
            max_eigenvalue,
            min_eigenvalue,
            sharpness,
        )

        return CurvatureMetricsJAX(
            max_eigenvalue=max_eigenvalue,
            min_eigenvalue=min_eigenvalue,
            condition_number=condition_number,
            trace=trace,
            sharpness=sharpness,
        )

    def _random_direction(
        self,
        params: dict[str, Any],
        key: jax.random.PRNGKey,
    ) -> dict[str, Any]:
        """Generate random direction with same structure as params."""
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves))
        random_leaves = [
            jax.random.normal(k, leaf.shape) if hasattr(leaf, 'shape') else leaf
            for k, leaf in zip(keys, leaves)
        ]
        return jax.tree_util.tree_unflatten(treedef, random_leaves)

    def _normalize_direction(
        self,
        direction: dict[str, Any],
        params: dict[str, Any],
        filter_norm: bool = True,
    ) -> dict[str, Any]:
        """
        Normalize direction, optionally using filter normalization.

        Filter normalization scales each tensor to match the norm of
        corresponding parameters, providing architecture-independent
        visualization.
        """
        if filter_norm:
            # Filter-wise normalization
            def normalize_leaf(d, p):
                if not hasattr(d, 'shape'):
                    return d
                d_norm = float(jnp.linalg.norm(d))
                p_norm = float(jnp.linalg.norm(p))
                if d_norm > 1e-10:
                    return d * (p_norm / d_norm)
                return d

            return jax.tree.map(normalize_leaf, direction, params)
        else:
            # Global normalization
            leaves = jax.tree_util.tree_leaves(direction)
            total_norm_sq = sum(
                float(jnp.sum(d ** 2)) for d in leaves if hasattr(d, 'shape')
            )
            total_norm = math.sqrt(total_norm_sq)

            if total_norm > 1e-10:
                return jax.tree.map(
                    lambda d: d / total_norm if hasattr(d, 'shape') else d,
                    direction,
                )
            return direction

    def _perturb(
        self,
        params: dict[str, Any],
        d1: dict[str, Any],
        d2: dict[str, Any],
        x: float,
        y: float,
    ) -> dict[str, Any]:
        """Perturb parameters: θ + x*d1 + y*d2."""
        return jax.tree.map(
            lambda p, dir1, dir2: p + x * dir1 + y * dir2,
            params,
            d1,
            d2,
        )

    def _hessian_vector_product(
        self,
        params: dict[str, Any],
        loss_fn: Callable[[dict[str, Any]], float],
        v: dict[str, Any],
        epsilon: float,
    ) -> dict[str, Any]:
        """
        Compute Hessian-vector product via finite differences.

        H*v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)
        """
        # Forward perturbation
        params_plus = jax.tree.map(lambda p, d: p + epsilon * d, params, v)
        grad_plus = self._compute_gradient(params_plus, loss_fn)

        # Backward perturbation
        params_minus = jax.tree.map(lambda p, d: p - epsilon * d, params, v)
        grad_minus = self._compute_gradient(params_minus, loss_fn)

        # Hessian-vector product
        return jax.tree.map(
            lambda gp, gm: (gp - gm) / (2 * epsilon),
            grad_plus,
            grad_minus,
        )

    def _compute_gradient(
        self,
        params: dict[str, Any],
        loss_fn: Callable[[dict[str, Any]], float],
    ) -> dict[str, Any]:
        """Compute gradient using JAX autodiff."""
        return jax.grad(loss_fn)(params)

    def _dot_product(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> float:
        """Compute dot product between two parameter pytrees."""
        leaves_a = jax.tree_util.tree_leaves(a)
        leaves_b = jax.tree_util.tree_leaves(b)
        total = 0.0
        for la, lb in zip(leaves_a, leaves_b):
            if hasattr(la, 'shape') and hasattr(lb, 'shape'):
                total += float(jnp.sum(la * lb))
        return total


# =============================================================================
# Model-based convenience functions
# =============================================================================

def compute_loss_surface_for_model(
    apply_fn: Callable,
    params: dict[str, Any],
    data_batch: tuple[jnp.ndarray, jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    resolution: int = 21,
    scale: float = 1.0,
) -> LossSurfaceDataJAX:
    """
    Compute loss surface for a JAX model.

    Args:
        apply_fn: Model forward function: apply_fn(params, inputs) -> outputs
        params: Model parameters
        data_batch: Tuple of (inputs, targets)
        loss_fn: Loss function: loss_fn(outputs, targets) -> scalar
        resolution: Grid resolution
        scale: Perturbation scale

    Returns:
        LossSurfaceDataJAX with visualization data
    """
    inputs, targets = data_batch

    def model_loss_fn(p: dict[str, Any]) -> float:
        outputs = apply_fn(p, inputs)
        return float(loss_fn(outputs, targets))

    computer = LossLandscapeComputerJAX(resolution=resolution, scale=scale)
    return computer.compute_surface(params, model_loss_fn)


def estimate_curvature_for_model(
    apply_fn: Callable,
    params: dict[str, Any],
    data_batch: tuple[jnp.ndarray, jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    num_samples: int = 20,
    epsilon: float = 1e-3,
) -> CurvatureMetricsJAX:
    """
    Estimate curvature metrics for a JAX model.

    Args:
        apply_fn: Model forward function
        params: Model parameters
        data_batch: Tuple of (inputs, targets)
        loss_fn: Loss function
        num_samples: Number of power iterations
        epsilon: Finite difference step

    Returns:
        CurvatureMetricsJAX with sharpness and eigenvalue estimates
    """
    inputs, targets = data_batch

    def model_loss_fn(p: dict[str, Any]) -> float:
        outputs = apply_fn(p, inputs)
        return float(loss_fn(outputs, targets))

    computer = LossLandscapeComputerJAX()
    return computer.estimate_curvature(
        params,
        model_loss_fn,
        num_samples=num_samples,
        epsilon=epsilon,
    )


__all__ = [
    "SurfacePointJAX",
    "LossSurfaceDataJAX",
    "CurvatureMetricsJAX",
    "LossLandscapeComputerJAX",
    "compute_loss_surface_for_model",
    "estimate_curvature_for_model",
]
