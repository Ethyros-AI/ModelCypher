"""
CUDA Loss Landscape Analysis for Training Diagnostics.

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see loss_landscape_mlx.py
- JAX/TPU: see loss_landscape_jax.py

Use _platform.get_loss_landscape_computer() for automatic platform selection.

Implementation based on PyTorch 2.x best practices (2025):
- torch.randn_like for random direction generation
- torch.autograd.grad for gradient computation
- torch.no_grad() for parameter perturbation
- Finite differences for Hessian-vector products

Research Basis:
- arxiv:1712.09913 - Visualizing Loss Landscapes
- arxiv:2002.09572 - Sharpness-Aware Minimization

References:
- https://pytorch.org/docs/stable/autograd.html
- https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SurfacePointCUDA:
    """A point on the loss surface."""
    x: float  # First principal direction
    y: float  # Second principal direction
    loss: float


@dataclass
class LossSurfaceDataCUDA:
    """2D loss surface visualization data."""
    points: list[SurfacePointCUDA]
    min_loss: float
    max_loss: float
    center_loss: float
    resolution: int
    scale: float


@dataclass
class CurvatureMetricsCUDA:
    """Curvature information from Hessian analysis."""
    max_eigenvalue: float
    min_eigenvalue: float
    condition_number: float
    trace: float
    sharpness: float  # max_eigenvalue / (1 + max_eigenvalue)


class LossLandscapeComputerCUDA:
    """
    CUDA Loss Landscape Computer (PyTorch backend).

    Computes loss landscape visualization and curvature metrics
    using GPU-accelerated operations.

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
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            resolution: Number of points per dimension (default 21 = 441 total)
            scale: Range of perturbations in each direction
            device: CUDA device to use
        """
        self.resolution = resolution
        self.scale = scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute_surface(
        self,
        model_params: dict[str, torch.Tensor],
        loss_fn: Callable[[dict[str, torch.Tensor]], float],
        direction1: dict[str, torch.Tensor] | None = None,
        direction2: dict[str, torch.Tensor] | None = None,
    ) -> LossSurfaceDataCUDA:
        """
        Compute 2D loss surface around current parameters.

        Args:
            model_params: Current model parameters
            loss_fn: Function that computes loss given parameters
            direction1: First perturbation direction (random if None)
            direction2: Second perturbation direction (random if None)

        Returns:
            LossSurfaceDataCUDA with grid of loss values
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
        center_loss = float(loss_fn(model_params))

        # Create grid
        half = self.resolution // 2
        points: list[SurfacePointCUDA] = []
        min_loss = float('inf')
        max_loss = float('-inf')

        logger.info(
            "Computing loss surface: %dx%d grid, scale=%.2f",
            self.resolution,
            self.resolution,
            self.scale,
        )

        with torch.no_grad():
            for i in range(self.resolution):
                for j in range(self.resolution):
                    x = (i - half) / half * self.scale
                    y = (j - half) / half * self.scale

                    # Perturbed parameters: θ + x*d1 + y*d2
                    perturbed = self._perturb(model_params, direction1, direction2, x, y)
                    loss = float(loss_fn(perturbed))

                    points.append(SurfacePointCUDA(x=x, y=y, loss=loss))
                    min_loss = min(min_loss, loss)
                    max_loss = max(max_loss, loss)

        logger.info(
            "Loss surface computed: min=%.4f, max=%.4f, center=%.4f",
            min_loss,
            max_loss,
            center_loss,
        )

        return LossSurfaceDataCUDA(
            points=points,
            min_loss=min_loss,
            max_loss=max_loss,
            center_loss=center_loss,
            resolution=self.resolution,
            scale=self.scale,
        )

    def estimate_curvature(
        self,
        model_params: dict[str, torch.Tensor],
        loss_fn: Callable[[dict[str, torch.Tensor]], float],
        num_samples: int = 20,
        epsilon: float = 1e-3,
    ) -> CurvatureMetricsCUDA:
        """
        Estimate curvature metrics using Hessian-vector products.

        Uses power iteration to estimate max eigenvalue.

        Args:
            model_params: Current model parameters
            loss_fn: Loss function
            num_samples: Number of power iterations
            epsilon: Finite difference step size

        Returns:
            CurvatureMetricsCUDA with eigenvalue estimates
        """
        # Initialize random vector
        v = self._random_direction(model_params)
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
        v_neg = {k: -arr for k, arr in v.items()}
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
            r = self._random_direction(model_params)
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

        return CurvatureMetricsCUDA(
            max_eigenvalue=max_eigenvalue,
            min_eigenvalue=min_eigenvalue,
            condition_number=condition_number,
            trace=trace,
            sharpness=sharpness,
        )

    def _random_direction(
        self,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Generate random direction with same structure as params."""
        return {
            k: torch.randn_like(v)
            for k, v in params.items()
        }

    def _normalize_direction(
        self,
        direction: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        filter_norm: bool = True,
    ) -> dict[str, torch.Tensor]:
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
                d_norm = float(torch.norm(d).item())
                p_norm = float(torch.norm(p).item())
                if d_norm > 1e-10:
                    result[k] = d * (p_norm / d_norm)
                else:
                    result[k] = d.clone()
            return result
        else:
            # Global normalization
            total_norm_sq = 0.0
            for d in direction.values():
                total_norm_sq += float(torch.sum(d ** 2).item())
            total_norm = math.sqrt(total_norm_sq)

            if total_norm > 1e-10:
                return {k: d / total_norm for k, d in direction.items()}
            return {k: d.clone() for k, d in direction.items()}

    def _perturb(
        self,
        params: dict[str, torch.Tensor],
        d1: dict[str, torch.Tensor],
        d2: dict[str, torch.Tensor],
        x: float,
        y: float,
    ) -> dict[str, torch.Tensor]:
        """Perturb parameters: θ + x*d1 + y*d2."""
        return {
            k: params[k] + x * d1[k] + y * d2[k]
            for k in params
        }

    def _hessian_vector_product(
        self,
        params: dict[str, torch.Tensor],
        loss_fn: Callable[[dict[str, torch.Tensor]], float],
        v: dict[str, torch.Tensor],
        epsilon: float,
    ) -> dict[str, torch.Tensor]:
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
        return {
            k: (grad_plus[k] - grad_minus[k]) / (2 * epsilon)
            for k in params
        }

    def _compute_gradient(
        self,
        params: dict[str, torch.Tensor],
        loss_fn: Callable[[dict[str, torch.Tensor]], float],
        epsilon: float,
    ) -> dict[str, torch.Tensor]:
        """
        Compute gradient using PyTorch autograd or finite differences.

        Attempts to use autograd if loss_fn returns a differentiable tensor,
        otherwise falls back to finite differences.
        """
        # Try to compute using autograd
        try:
            # Create parameters that require grad
            params_req_grad = {
                k: v.clone().requires_grad_(True)
                for k, v in params.items()
            }

            loss = loss_fn(params_req_grad)

            # Check if loss is a tensor we can differentiate
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                grads = torch.autograd.grad(
                    loss,
                    list(params_req_grad.values()),
                    create_graph=False,
                )
                return dict(zip(params_req_grad.keys(), grads))
        except Exception:
            pass  # Fall back to finite differences

        # Fallback: numeric gradients
        gradients: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            param_np = param.detach().cpu().numpy()
            grad_np = np.zeros_like(param_np, dtype=np.float32)

            for idx in np.ndindex(param_np.shape):
                perturb = np.zeros_like(param_np)
                perturb[idx] = epsilon

                params_plus = dict(params)
                params_minus = dict(params)
                params_plus[name] = torch.tensor(
                    param_np + perturb,
                    device=param.device,
                    dtype=param.dtype,
                )
                params_minus[name] = torch.tensor(
                    param_np - perturb,
                    device=param.device,
                    dtype=param.dtype,
                )

                loss_plus = loss_fn(params_plus)
                loss_minus = loss_fn(params_minus)

                grad_np[idx] = (float(loss_plus) - float(loss_minus)) / (2.0 * epsilon)

            gradients[name] = torch.tensor(
                grad_np,
                device=param.device,
                dtype=param.dtype,
            )

        return gradients

    def _dot_product(
        self,
        a: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor],
    ) -> float:
        """Compute dot product between two parameter dicts."""
        total = 0.0
        for k in a:
            if k in b:
                total += float(torch.sum(a[k] * b[k]).item())
        return total


# =============================================================================
# Model-based convenience functions
# =============================================================================

def compute_loss_surface_for_model(
    model: nn.Module,
    data_batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    resolution: int = 21,
    scale: float = 1.0,
    device: str = "cuda:0",
) -> LossSurfaceDataCUDA:
    """
    Compute loss surface for a PyTorch model.

    Convenience function that extracts model parameters and creates
    a compatible loss function.

    Args:
        model: PyTorch model
        data_batch: Tuple of (inputs, targets)
        loss_fn: Loss function taking (outputs, targets)
        resolution: Grid resolution
        scale: Perturbation scale
        device: CUDA device

    Returns:
        LossSurfaceDataCUDA with visualization data
    """
    inputs, targets = data_batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model = model.to(device)

    # Extract current parameters
    original_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    def model_loss_fn(params: dict[str, torch.Tensor]) -> float:
        """Compute loss with given parameters."""
        # Load params into model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        return float(loss.item())

    # Compute surface
    computer = LossLandscapeComputerCUDA(resolution=resolution, scale=scale, device=device)
    result = computer.compute_surface(original_params, model_loss_fn)

    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    return result


def estimate_curvature_for_model(
    model: nn.Module,
    data_batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_samples: int = 20,
    epsilon: float = 1e-3,
    device: str = "cuda:0",
) -> CurvatureMetricsCUDA:
    """
    Estimate curvature metrics for a PyTorch model.

    Args:
        model: PyTorch model
        data_batch: Tuple of (inputs, targets)
        loss_fn: Loss function
        num_samples: Number of power iterations
        epsilon: Finite difference step
        device: CUDA device

    Returns:
        CurvatureMetricsCUDA with sharpness and eigenvalue estimates
    """
    inputs, targets = data_batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model = model.to(device)

    original_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    def model_loss_fn(params: dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        return float(loss.item())

    computer = LossLandscapeComputerCUDA(device=device)
    result = computer.estimate_curvature(
        original_params,
        model_loss_fn,
        num_samples=num_samples,
        epsilon=epsilon,
    )

    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    return result


__all__ = [
    "SurfacePointCUDA",
    "LossSurfaceDataCUDA",
    "CurvatureMetricsCUDA",
    "LossLandscapeComputerCUDA",
    "compute_loss_surface_for_model",
    "estimate_curvature_for_model",
]
