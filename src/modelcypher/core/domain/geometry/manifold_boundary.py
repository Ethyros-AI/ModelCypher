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

"""Manifold boundary detection via flood fill.

Probes model response along perturbation directions to estimate boundary
radius and coherence measures.

References:
    - Lipschitz continuity as a measure of neural network stability
    - Adversarial robustness literature (boundary detection)
    - Manifold hypothesis in deep learning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    safe_log_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class CoherenceResult:
    """Result of measuring coherence at a point.

    Attributes:
        coherence: Coherence score in [0, 1]. High = inside manifold, low = boundary/outside.
        curvature: Response curvature (second derivative magnitude).
        output_scale: Scale of the output at the center point.
        is_coherent: True if coherence exceeds threshold.
    """
    coherence: float
    curvature: float
    output_scale: float
    is_coherent: bool


@dataclass
class BoundaryResult:
    """Result of boundary detection in one direction.

    Attributes:
        direction: Unit vector direction probed [hidden_dim].
        boundary_radius: Radius at which coherence drops below threshold.
        coherence_at_boundary: Coherence value at the boundary.
        max_radius_tested: Maximum radius tested.
        is_bounded: True if boundary was found, False if extends to max_radius.
    """
    direction: "Array"
    boundary_radius: float
    coherence_at_boundary: float
    max_radius_tested: float
    is_bounded: bool


@dataclass
class ManifoldBoundaryResult:
    """Result of full manifold boundary detection.

    Attributes:
        centroid: Activation centroid used as origin [hidden_dim].
        directions: Orthonormal directions probed [n_directions, hidden_dim].
        boundary_radii: Boundary radius in each direction [n_directions].
        mean_radius: Mean boundary radius across all directions.
        min_radius: Minimum boundary radius (tightest constraint).
        max_radius: Maximum boundary radius (loosest constraint).
        utilized_volume_fraction: Estimated fraction of space utilized.
        n_bounded: Number of directions where boundary was found.
    """
    centroid: "Array"
    directions: "Array"
    boundary_radii: list[float]
    mean_radius: float
    min_radius: float
    max_radius: float
    utilized_volume_fraction: float
    n_bounded: int


def measure_coherence(
    activation: "Array",
    direction: "Array",
    radius: float,
    forward_fn: Callable[["Array"], "Array"],
    backend: "Backend",
    metric: str = "magnitude_stability",
) -> CoherenceResult:
    """Measure coherence at a point by probing response quality.

    Metrics:
    1. "magnitude_stability": Measures if output magnitude stays proportional to
       input perturbation. Inside manifold = proportional. At boundary = explodes.
    2. "entropy": Uses entropy of output distribution (requires prob output).
    3. "curvature": Uses second-order finite difference. Classic Lipschitz measure.

    Args:
        activation: Center activation vector [hidden_dim].
        direction: Unit direction to perturb [hidden_dim].
        radius: Perturbation radius (δ = radius * direction).
        forward_fn: Function that maps activation -> output.
        backend: Backend for tensor operations.
        metric: "magnitude_stability", "entropy", or "curvature".

    Returns:
        CoherenceResult with coherence score and diagnostics.
    """
    b = backend

    # Ensure inputs are proper arrays
    activation = _promote_precision(b.array(activation), b)
    direction = _promote_precision(b.array(direction), b)
    b.eval(activation, direction)

    # Compute perturbed point
    delta = direction * radius
    a_perturbed = activation + delta
    b.eval(a_perturbed)

    # Forward through model
    y_center = forward_fn(activation)
    y_perturbed = forward_fn(a_perturbed)
    b.eval(y_center, y_perturbed)

    eps = machine_epsilon(b, activation)

    if metric == "magnitude_stability":
        # Magnitude stability: output change should be proportional to input change
        # If Lipschitz constant L is stable, ||y_pert - y_center|| ≈ L * radius
        # If it explodes (>>L*radius), we're at a boundary

        # Compute output difference magnitude
        y_diff = y_perturbed - y_center
        b.eval(y_diff)
        y_diff_norm_sq = b.sum(y_diff * y_diff)
        b.eval(y_diff_norm_sq)
        y_diff_norm = float(b.to_scalar(b.sqrt(y_diff_norm_sq)))

        # Compute center output magnitude for normalization
        y_center_norm_sq = b.sum(y_center * y_center)
        b.eval(y_center_norm_sq)
        y_center_norm = float(b.to_scalar(b.sqrt(y_center_norm_sq)))

        # Sensitivity = output change / (radius * output_scale)
        # For stable function, this should be O(1)
        # At boundary, this explodes
        if y_center_norm > eps and radius > eps:
            sensitivity = y_diff_norm / (radius * y_center_norm)
        else:
            sensitivity = 0.0

        # Coherence = 1 / (1 + sensitivity)
        # High (≈1) when output change is small relative to perturbation
        # Low (≈0) when output explodes
        coherence = 1.0 / (1.0 + sensitivity)

        output_scale = y_center_norm
        curvature = sensitivity

    elif metric == "entropy":
        # Entropy-based coherence: low entropy = confident = inside manifold
        # Entropy = -sum(p * log(p))
        # For softmax output, max entropy = log(vocab_size)

        # Compute entropy of perturbed output
        # Clamp to avoid log(0) using dtype-derived safe_log_epsilon
        log_eps = safe_log_epsilon(b, y_perturbed)
        y_clamped = b.maximum(y_perturbed, b.full(b.shape(y_perturbed), log_eps))
        b.eval(y_clamped)

        # Check for numerical issues
        y_sum = float(b.to_scalar(b.sum(y_perturbed)))
        y_min = float(b.to_scalar(b.min(y_perturbed)))
        y_max = float(b.to_scalar(b.max(y_perturbed)))

        # Compute entropy: -sum(p * log(p))
        log_probs = b.log(y_clamped)
        b.eval(log_probs)
        p_log_p = y_perturbed * log_probs
        b.eval(p_log_p)
        entropy_raw = b.sum(p_log_p)
        b.eval(entropy_raw)
        entropy = -float(b.to_scalar(entropy_raw))

        # Compute max entropy for normalization
        vocab_size = int(b.shape(y_perturbed)[-1])
        import math
        max_entropy = math.log(float(vocab_size))

        # Normalized entropy: 0 = perfectly confident, 1 = uniform random
        normalized_entropy = entropy / max_entropy

        # Coherence = 1 - normalized_entropy
        # High (≈1) when confident, low (≈0) when confused
        # Clamp to [0, 1] in case of numerical issues
        coherence = max(0.0, min(1.0, 1.0 - normalized_entropy))

        # Debug: log if we have numerical issues
        if y_sum < 0.9 or y_sum > 1.1 or y_min < -0.01 or y_max > 1.01:
            logger.warning(
                "Numerical issue: y_sum=%.4f, y_min=%.4f, y_max=%.4f, entropy=%.4f",
                y_sum, y_min, y_max, entropy
            )

        # Output scale = negative entropy (lower is better)
        output_scale = entropy
        curvature = normalized_entropy  # Use as proxy

    else:
        # Curvature-based coherence (original approach)
        a_minus = activation - delta
        b.eval(a_minus)
        y_minus = forward_fn(a_minus)
        b.eval(y_minus)

        # Second-order finite difference
        second_diff = y_perturbed + y_minus - 2.0 * y_center
        b.eval(second_diff)

        curvature_sq = b.sum(second_diff * second_diff)
        b.eval(curvature_sq)
        curvature = float(b.to_scalar(b.sqrt(curvature_sq)))

        if radius > 0:
            curvature = curvature / (radius * radius)

        output_scale_sq = b.sum(y_center * y_center)
        b.eval(output_scale_sq)
        output_scale = float(b.to_scalar(b.sqrt(output_scale_sq)))

        relative_curvature = curvature / (output_scale + eps)
        coherence = 1.0 / (1.0 + relative_curvature)

    # Threshold based on sqrt(eps)
    threshold = 1.0 - sqrt_scalar(eps, b)
    is_coherent = coherence > threshold

    return CoherenceResult(
        coherence=coherence,
        curvature=curvature,
        output_scale=output_scale,
        is_coherent=is_coherent,
    )


def find_boundary_radius(
    activation: "Array",
    direction: "Array",
    forward_fn: Callable[["Array"], "Array"],
    backend: "Backend",
    max_radius: float = 10.0,
    tolerance: float = 0.01,
    coherence_threshold: float | None = None,
    coherence_drop_fraction: float = 0.5,
    metric: str = "magnitude_stability",
) -> BoundaryResult:
    """Find the boundary radius in a given direction via binary search.

    Starting from a known inhabited point (activation), we probe outward
    in the given direction until coherence drops below threshold.

    The threshold is ADAPTIVE: we measure baseline coherence at small radii,
    then look for where coherence drops by coherence_drop_fraction.

    Args:
        activation: Center activation vector [hidden_dim].
        direction: Unit direction to probe [hidden_dim].
        forward_fn: Function that maps activation -> output.
        backend: Backend for tensor operations.
        max_radius: Maximum radius to test.
        tolerance: Binary search tolerance.
        coherence_threshold: Absolute threshold for "inside" vs "outside".
            If None, uses adaptive threshold based on baseline coherence.
        coherence_drop_fraction: Fraction of baseline coherence that indicates
            boundary. Default 0.5 means boundary is where coherence drops to
            50% of baseline.

    Returns:
        BoundaryResult with boundary radius and diagnostics.
    """
    b = backend

    # Normalize direction
    direction = _promote_precision(b.array(direction), b)
    dir_norm_sq = b.sum(direction * direction)
    b.eval(dir_norm_sq)
    dir_norm = float(b.to_scalar(b.sqrt(dir_norm_sq)))
    if dir_norm > 0:
        direction = direction / dir_norm
    b.eval(direction)

    # Measure baseline coherence at small radius to calibrate threshold
    baseline_result = measure_coherence(activation, direction, 0.001, forward_fn, b, metric=metric)
    baseline_coherence = baseline_result.coherence

    # Adaptive threshold: boundary is where coherence drops to fraction of baseline
    if coherence_threshold is None:
        coherence_threshold = baseline_coherence * coherence_drop_fraction

    # Check if baseline is extremely low (something wrong with forward_fn)
    if baseline_coherence < 0.1:
        logger.warning(
            "Baseline coherence very low (%.4f) - forward_fn may be unstable",
            baseline_coherence
        )
        return BoundaryResult(
            direction=direction,
            boundary_radius=0.0,
            coherence_at_boundary=baseline_coherence,
            max_radius_tested=0.001,
            is_bounded=True,
        )

    # Check if bounded at max_radius
    result_max = measure_coherence(activation, direction, max_radius, forward_fn, b, metric=metric)

    # Log coherence for debugging (at DEBUG level)
    logger.debug(
        "  Coherence: baseline=%.4f, at_max_radius=%.4f, threshold=%.4f, sensitivity=%.4f",
        baseline_coherence, result_max.coherence, coherence_threshold, result_max.curvature
    )

    if result_max.coherence > coherence_threshold:
        # Still coherent at max_radius - not bounded in this direction
        return BoundaryResult(
            direction=direction,
            boundary_radius=max_radius,
            coherence_at_boundary=result_max.coherence,
            max_radius_tested=max_radius,
            is_bounded=False,
        )

    # Binary search for boundary
    low = 0.0
    high = max_radius
    coherence_at_boundary = result_max.coherence

    while high - low > tolerance:
        mid = (low + high) / 2.0
        result = measure_coherence(activation, direction, mid, forward_fn, b, metric=metric)

        if result.coherence > coherence_threshold:
            # Still inside - move outward
            low = mid
        else:
            # At or past boundary - move inward
            high = mid
            coherence_at_boundary = result.coherence

    return BoundaryResult(
        direction=direction,
        boundary_radius=low,  # Last known "inside" radius
        coherence_at_boundary=coherence_at_boundary,
        max_radius_tested=max_radius,
        is_bounded=True,
    )


def detect_manifold_boundary(
    activations: "Array",
    forward_fn: Callable[["Array"], "Array"],
    backend: "Backend",
    n_directions: int = 100,
    max_radius: float = 10.0,
    seed: int = 42,
) -> ManifoldBoundaryResult:
    """Detect manifold boundary by probing in random orthogonal directions.

    This is the main entry point for flood fill boundary detection.

    Algorithm:
    1. Compute activation centroid as starting point
    2. Generate random orthonormal directions
    3. For each direction, find boundary radius via binary search
    4. Aggregate results into a manifold boundary map

    Args:
        activations: Sample activations [n_samples, hidden_dim].
        forward_fn: Function that maps activation -> output.
        backend: Backend for tensor operations.
        n_directions: Number of random directions to probe.
        max_radius: Maximum radius to probe in each direction.
        seed: Random seed for reproducibility.

    Returns:
        ManifoldBoundaryResult with boundary radii and utilization estimate.
    """
    b = backend
    b.random_seed(seed)

    activations = _promote_precision(b.array(activations), b)
    b.eval(activations)

    shape = b.shape(activations)
    n_samples = int(shape[0])
    hidden_dim = int(shape[1])

    logger.info(
        "MANIFOLD BOUNDARY: Detecting from [%d, %d] activations, %d directions",
        n_samples, hidden_dim, n_directions
    )

    # Compute centroid as starting point
    centroid = b.mean(activations, axis=0)
    b.eval(centroid)

    # Generate random orthonormal directions via QR decomposition
    # Start with random Gaussian matrix
    random_matrix = b.random_normal((hidden_dim, n_directions))
    b.eval(random_matrix)

    # QR decomposition gives orthonormal columns
    Q, R = b.qr(random_matrix)
    b.eval(Q)

    # Take first n_directions columns as directions
    directions = Q[:, :n_directions]  # [hidden_dim, n_directions]
    b.eval(directions)

    # Probe each direction
    boundary_radii = []
    n_bounded = 0

    for i in range(n_directions):
        direction = directions[:, i]
        b.eval(direction)

        result = find_boundary_radius(
            centroid, direction, forward_fn, b,
            max_radius=max_radius,
        )

        boundary_radii.append(result.boundary_radius)
        if result.is_bounded:
            n_bounded += 1

        if (i + 1) % 20 == 0:
            logger.info(
                "  Probed %d/%d directions, %d bounded",
                i + 1, n_directions, n_bounded
            )

    # Statistics
    mean_radius = sum(boundary_radii) / len(boundary_radii)
    min_radius = min(boundary_radii)
    max_radius_found = max(boundary_radii)

    # Estimate utilized volume fraction
    # If boundary radius is r in each direction, volume is proportional to r^d
    # For high-d, use log-average to avoid overflow
    log_eps = safe_log_epsilon(b, activations)
    log_radii = [
        float(b.to_scalar(b.log(b.array(r + log_eps))))
        for r in boundary_radii
    ]
    mean_log_radius = sum(log_radii) / len(log_radii)

    # Compare to max_radius volume
    log_max_volume = hidden_dim * float(b.to_scalar(b.log(b.array(max_radius))))
    log_utilized_volume = hidden_dim * mean_log_radius

    # Fraction = exp(log_utilized - log_max)
    volume_fraction = float(b.to_scalar(b.exp(b.array(log_utilized_volume - log_max_volume))))
    volume_fraction = min(1.0, max(0.0, volume_fraction))  # Clamp to [0, 1]

    logger.info(
        "MANIFOLD BOUNDARY: mean_radius=%.3f, min=%.3f, max=%.3f, "
        "bounded=%d/%d, utilized_volume=%.1f%%",
        mean_radius, min_radius, max_radius_found,
        n_bounded, n_directions,
        volume_fraction * 100
    )

    return ManifoldBoundaryResult(
        centroid=centroid,
        directions=b.transpose(directions),  # [n_directions, hidden_dim]
        boundary_radii=boundary_radii,
        mean_radius=mean_radius,
        min_radius=min_radius,
        max_radius=max_radius_found,
        utilized_volume_fraction=volume_fraction,
        n_bounded=n_bounded,
    )


def create_layer_forward_fn(
    model: Any,
    layer_idx: int,
    config: dict,
    backend: "Backend",
    mode: str = "full_model",
) -> Callable[["Array"], "Array"]:
    """Create a forward function for a specific layer.

    The returned function takes an activation [hidden_dim] and returns
    the output after forwarding through the model.

    NOTE: This function requires MLX models. For other backends, use the
    appropriate adapter in modelcypher.adapters.geometry.

    Args:
        model: The model (must be an MLX model).
        layer_idx: Layer index to inject activation at.
        config: Model config.
        backend: Backend for tensor operations.
        mode: "mlp" for just the MLP, "full_model" for remaining layers + lm_head.

    Returns:
        Callable that maps activation -> output.
    """
    from modelcypher.ports.model_architecture_factory import get_model_architecture

    b = backend
    arch = get_model_architecture(model, config=config)
    num_layers = config.get("num_hidden_layers", len(arch.layers))

    if mode == "mlp":
        # Just forward through MLP (original behavior)
        accessor = arch.layer_accessor(layer_idx)
        mlp = accessor.mlp

        def forward_fn(activation: "Array") -> "Array":
            if len(b.shape(activation)) == 1:
                activation = b.expand_dims(activation, axis=0)
                activation = b.expand_dims(activation, axis=0)
            output = mlp(activation)
            b.eval(output)
            return b.reshape(output, (-1,))

        return forward_fn

    else:
        # Forward through remaining layers + lm_head for semantic coherence
        layers = arch.layers

        def forward_fn(activation: "Array") -> "Array":
            # Shape to [1, 1, hidden_dim]
            if len(b.shape(activation)) == 1:
                h = b.expand_dims(activation, axis=0)
                h = b.expand_dims(h, axis=0)  # [1, 1, hidden_dim]
            else:
                h = activation

            # Get sequence length and create causal mask
            # Use Backend's causal mask if available, otherwise create manually
            seq_len = int(b.shape(h)[1])
            # Create additive causal mask: 0 for allowed, -inf for masked
            mask = b.triu_indices(seq_len, k=1)
            mask_array = b.zeros((seq_len, seq_len))
            # For now, skip mask for simplicity - layers handle it internally
            mask = None

            # Forward through remaining layers
            for i in range(layer_idx + 1, num_layers):
                layer = layers[i]

                # Try different layer call signatures
                # Most transformer layers take (hidden_states, mask, cache)
                try:
                    result = layer(h, mask=mask, cache=None)
                except TypeError:
                    try:
                        result = layer(h, mask, None)
                    except TypeError:
                        try:
                            result = layer(h, mask)
                        except TypeError:
                            # Fallback: no mask at all
                            result = layer(h)

                # Handle tuple returns: (hidden, cache) or (hidden, cache, ...)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

                b.eval(h)

            # Apply final norm if present
            if arch.norm is not None:
                h = arch.norm(h)
                b.eval(h)

            # Apply lm_head if present (get logit distribution)
            if arch.output_projection is not None:
                logits = arch.output_projection(h)
                b.eval(logits)
                # Return softmax probabilities (measures distribution quality)
                logits = b.reshape(logits, (-1,))
                # Softmax for distribution
                probs = b.softmax(logits)
                b.eval(probs)
                return probs
            else:
                return b.reshape(h, (-1,))

        return forward_fn


def detect_layer_boundary(
    model: Any,
    layer_idx: int,
    activations: "Array",
    config: dict,
    backend: "Backend",
    n_directions: int = 50,
    max_radius: float = 5.0,
    forward_mode: str = "full_model",
) -> ManifoldBoundaryResult:
    """Detect manifold boundary for a specific layer.

    Convenience wrapper that creates the forward function and runs detection.

    Args:
        model: The model.
        layer_idx: Layer to analyze.
        activations: Sample activations at this layer [n_samples, hidden_dim].
        config: Model config.
        backend: Backend for tensor operations.
        n_directions: Number of directions to probe.
        max_radius: Maximum radius to probe.
        forward_mode: "mlp" for just MLP, "full_model" for remaining layers + lm_head.

    Returns:
        ManifoldBoundaryResult with boundary radii and utilization.
    """
    forward_fn = create_layer_forward_fn(model, layer_idx, config, backend, mode=forward_mode)

    return detect_manifold_boundary(
        activations=activations,
        forward_fn=forward_fn,
        backend=backend,
        n_directions=n_directions,
        max_radius=max_radius,
    )


def compute_boundary_radii_from_weights(
    target_activations: dict[int, "Array"],
    target_weights: dict[str, "Array"],
    backend: "Backend",
    n_directions: int = 20,
    seed: int = 42,
) -> dict[int, float]:
    """Compute boundary radii per layer using linear weight approximation.

    This is a lightweight version that doesn't require the full model.
    Instead of forwarding through remaining layers + lm_head, we use
    a linear approximation with the MLP up_proj weight.

    The coherence metric measures: how much does the output change when
    we perturb the input? Layers with more "headroom" (larger null space)
    can tolerate larger perturbations before output changes significantly.

    IMPORTANT: The max_radius is derived from activation scale, not fixed.
    This makes the boundary detection scale-invariant.

    Args:
        target_activations: Per-layer activations [n_samples, hidden_dim].
        target_weights: Full weight dict (we extract MLP weights).
        backend: Backend for tensor operations.
        n_directions: Directions to probe per layer.
        seed: Random seed.

    Returns:
        Dict mapping layer_idx -> boundary_radius (min across directions).
    """
    b = backend
    b.random_seed(seed)

    # Derive tolerance from model dtype (can't invent precision)
    # Use sqrt(eps) as meaningful threshold - same as numerical stability code
    sample_weight = next(iter(target_weights.values()), None)
    if sample_weight is not None:
        tolerance = sqrt_scalar(machine_epsilon(b, sample_weight), b)
        # But also need a reasonable minimum for binary search to converge
        tolerance = max(tolerance, 0.001)
    else:
        tolerance = 0.01  # Fallback

    boundary_radii: dict[int, float] = {}

    # Find MLP UP projection weights per layer (architecture-agnostic)
    # We use up_proj because it takes hidden_dim input (what we're probing).
    # Down_proj takes intermediate_dim input which we don't have.
    #
    # Different architectures use different naming:
    #   - Llama/Qwen: mlp.up_proj or mlp.gate_proj
    #   - LFM2/Mamba: feed_forward.w1 or feed_forward.w3
    #   - GPT-2: mlp.c_fc
    #
    # Weight shape for up_proj: [intermediate_dim, hidden_dim]
    # Forward: x [batch, hidden] @ W.T [hidden, intermediate] → [batch, intermediate]
    mlp_weights: dict[int, "Array"] = {}
    for key, weight in target_weights.items():
        key_lower = key.lower()
        is_mlp_up = (
            ("mlp" in key_lower and "up_proj" in key_lower) or
            ("mlp" in key_lower and "gate_proj" in key_lower) or
            ("feed_forward" in key_lower and ".w1." in key_lower) or
            ("feed_forward" in key_lower and ".w3." in key_lower) or
            ("mlp" in key_lower and "c_fc" in key_lower)
        )
        if is_mlp_up and "weight" in key_lower:
            # Extract layer index from key
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        mlp_weights[layer_idx] = weight
                    except ValueError:
                        continue
                    break

    if not mlp_weights:
        logger.warning("BOUNDARY: No MLP weights found - returning empty radii")
        return boundary_radii

    # =========================================================================
    # CASCADE SPECTRAL CORRECTION
    # =========================================================================
    # Local boundary measures MLP stability in isolation. But perturbations
    # cascade through subsequent layers. The worst-case amplification is bounded
    # by the product of spectral norms (submultiplicative property):
    #
    #   ||J_{l→L}|| ≤ σ_max(W_{l+1}) × σ_max(W_{l+2}) × ... × σ_max(W_{L-1})
    #
    # Global boundary = local boundary / cascade_amplification
    #
    # Early layers have longer cascades → higher amplification → smaller radius
    # Late layers have shorter cascades → lower amplification → larger radius
    # =========================================================================
    layer_indices_sorted = sorted(mlp_weights.keys())
    spectral_norms: dict[int, float] = {}

    logger.info(
        "BOUNDARY: Computing spectral norms for %d layers (cascade correction)",
        len(mlp_weights)
    )

    for layer_idx in layer_indices_sorted:
        W = mlp_weights[layer_idx]
        W = _promote_precision(b.array(W), b)
        b.eval(W)

        # Spectral norm = max singular value
        # For efficiency, use power iteration approximation for large matrices
        shape = b.shape(W)
        if int(shape[0]) > 1000 or int(shape[1]) > 1000:
            # Power iteration for large matrices (10 iterations suffices)
            div_eps = division_epsilon(b, W)
            v = b.random_normal((int(shape[1]),))
            b.eval(v)
            v = v / b.sqrt(b.sum(v * v))
            b.eval(v)

            for _ in range(10):
                u = b.matmul(W, b.reshape(v, (-1, 1)))
                u = b.reshape(u, (-1,))
                b.eval(u)
                u_norm = b.sqrt(b.sum(u * u))
                b.eval(u_norm)
                u = u / b.maximum(u_norm, b.array([div_eps]))
                b.eval(u)

                v = b.matmul(b.transpose(W), b.reshape(u, (-1, 1)))
                v = b.reshape(v, (-1,))
                b.eval(v)
                v_norm = b.sqrt(b.sum(v * v))
                b.eval(v_norm)
                v = v / b.maximum(v_norm, b.array([div_eps]))
                b.eval(v)

            # σ_max ≈ ||W @ v||
            Wv = b.matmul(W, b.reshape(v, (-1, 1)))
            Wv = b.reshape(Wv, (-1,))
            b.eval(Wv)
            sigma_max = float(b.to_scalar(b.sqrt(b.sum(Wv * Wv))))
        else:
            # Full SVD for small matrices
            U, S, Vt = b.svd(W, full_matrices=False)
            b.eval(S)
            sigma_max = float(b.to_scalar(b.max(S)))

        spectral_norms[layer_idx] = sigma_max

    # Compute cascade amplification: product of spectral norms from l+1 to L-1
    cascade_amplification: dict[int, float] = {}
    for layer_idx in layer_indices_sorted:
        subsequent = [l for l in layer_indices_sorted if l > layer_idx]
        if subsequent:
            amplification = 1.0
            for l in subsequent:
                amplification *= spectral_norms[l]
            cascade_amplification[layer_idx] = amplification
        else:
            cascade_amplification[layer_idx] = 1.0  # Last layer, no cascade

    # Log cascade structure
    if cascade_amplification:
        min_amp = min(cascade_amplification.values())
        max_amp = max(cascade_amplification.values())
        # Find layers with min/max amplification
        layer_with_min = min(cascade_amplification, key=cascade_amplification.get)
        layer_with_max = max(cascade_amplification, key=cascade_amplification.get)
        logger.info(
            "BOUNDARY: Cascade amplification range: %.3f (layer %d) to %.3f (layer %d)",
            max_amp,
            layer_with_max,
            min_amp,
            layer_with_min,
        )

    logger.info(
        "BOUNDARY: Computing local radii for %d layers",
        len(mlp_weights)
    )

    for layer_idx, acts in target_activations.items():
        if layer_idx not in mlp_weights:
            continue

        W = mlp_weights[layer_idx]
        W = _promote_precision(b.array(W), b)
        b.eval(W)

        # Stack activations if needed
        if isinstance(acts, list):
            stacked = b.stack(acts, axis=0)
        else:
            stacked = b.array(acts)
        stacked = _promote_precision(stacked, b)
        b.eval(stacked)

        shape = b.shape(stacked)
        n_samples = int(shape[0])
        hidden_dim = int(shape[1])

        # Compute centroid
        centroid = b.mean(stacked, axis=0)
        b.eval(centroid)

        # Derive max_radius from centroid norm (scale-invariant)
        # This ensures radius is meaningful relative to activation magnitude
        # Use sqrt(sum(x^2)) directly - MLX norm() doesn't support ord parameter
        centroid_sq = b.sum(centroid * centroid)
        b.eval(centroid_sq)
        centroid_norm = float(b.to_scalar(centroid_sq)) ** 0.5
        # Max radius = full activation magnitude (100% perturbation)
        # Binary search will find where coherence breaks within this range
        max_radius = max(centroid_norm, 1.0)  # Floor of 1.0 for numerical stability

        # Create linear forward function (capture W by value via default arg)
        def make_linear_forward(W_captured: "Array") -> Callable[["Array"], "Array"]:
            def linear_forward(x: "Array") -> "Array":
                # Reshape for matmul if needed
                if len(b.shape(x)) == 1:
                    x = b.reshape(x, (1, -1))
                # W is [intermediate_dim, hidden_dim], x is [1, hidden_dim]
                # Need to transpose W for matmul: x @ W.T
                W_T = b.transpose(W_captured)
                out = b.matmul(x, W_T)
                b.eval(out)
                return b.reshape(out, (-1,))
            return linear_forward

        linear_forward = make_linear_forward(W)

        # Generate random orthonormal directions
        random_matrix = b.random_normal((hidden_dim, n_directions))
        b.eval(random_matrix)
        Q, R = b.qr(random_matrix)
        b.eval(Q)
        directions = Q[:, :n_directions]
        b.eval(directions)

        # Find minimum boundary radius across directions
        min_radius_found = max_radius
        for i in range(n_directions):
            direction = directions[:, i]
            b.eval(direction)

            result = find_boundary_radius(
                activation=centroid,
                direction=direction,
                forward_fn=linear_forward,
                backend=b,
                max_radius=max_radius,
                tolerance=tolerance,  # Derived from model dtype
            )

            if result.boundary_radius < min_radius_found:
                min_radius_found = result.boundary_radius

        # Apply cascade correction: global_radius = local_radius / amplification
        amplification = cascade_amplification.get(layer_idx, 1.0)
        global_radius = min_radius_found / amplification if amplification > 0 else min_radius_found

        boundary_radii[layer_idx] = global_radius
        logger.debug(
            "BOUNDARY: Layer %d -> local=%.3f, amplification=%.3f, global=%.3f",
            layer_idx, min_radius_found, amplification, global_radius
        )

    if boundary_radii:
        min_r = min(boundary_radii.values())
        max_r = max(boundary_radii.values())
        mean_r = sum(boundary_radii.values()) / len(boundary_radii)
        logger.info(
            "BOUNDARY: Global radii (cascade-corrected): %d layers, min=%.4f, max=%.4f, mean=%.4f",
            len(boundary_radii), min_r, max_r, mean_r
        )

    return boundary_radii


__all__ = [
    "CoherenceResult",
    "BoundaryResult",
    "ManifoldBoundaryResult",
    "measure_coherence",
    "find_boundary_radius",
    "detect_manifold_boundary",
    "create_layer_forward_fn",
    "detect_layer_boundary",
    "compute_boundary_radii_from_weights",
]
