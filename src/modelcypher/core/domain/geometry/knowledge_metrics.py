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

"""Core metrics for knowledge discovery.

This module provides backend-agnostic implementations of key metrics for
distinguishing factual knowledge from opinions in neural network representations.

Key Findings (from geometric_knowledge_discovery.py research):
- Counterfactual sensitivity effect size: 0.94 (strong) for distinguishing facts vs opinions
- Facts show high sensitivity (~0.2+) when counterfactual statements are compared
- Opinions show low sensitivity (~0.06) - representation similar regardless of content

Design Principles:
- Return NaN for degenerate/undefined cases (not semantic defaults)
- All thresholds are dtype-derived from machine epsilon
- No heuristic composite scores - raw metrics only
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def counterfactual_sensitivity(
    rep_original: "Array",
    rep_counterfactual: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute cosine distance between original and counterfactual representations.

    This is the PRIMARY metric for distinguishing factual knowledge from opinions.
    Effect size: 0.94 (strong) in research experiments.

    Args:
        rep_original: Activation vector for original statement.
        rep_counterfactual: Activation vector for counterfactual statement.
        backend: Compute backend. If None, uses default.

    Returns:
        Cosine distance in [0, 2]. Higher values indicate greater sensitivity.
        Returns NaN if either vector has near-zero norm (degenerate input).

    Example:
        >>> # "2+2=4" vs "2+2=5" should show HIGH sensitivity (model knows math)
        >>> # "Pizza is best" vs "Sushi is best" should show LOW sensitivity (opinions)
    """
    b = backend or get_default_backend()

    # Ensure 1D vectors
    arr_orig = b.array(rep_original) if not hasattr(rep_original, "shape") else rep_original
    arr_cf = b.array(rep_counterfactual) if not hasattr(rep_counterfactual, "shape") else rep_counterfactual

    if len(b.shape(arr_orig)) > 1:
        arr_orig = b.reshape(arr_orig, (-1,))
    if len(b.shape(arr_cf)) > 1:
        arr_cf = b.reshape(arr_cf, (-1,))

    # Compute norms
    norm_orig = b.norm(arr_orig)
    norm_cf = b.norm(arr_cf)
    b.eval(norm_orig, norm_cf)

    eps = division_epsilon(b, arr_orig)
    norm_orig_val = float(b.to_scalar(norm_orig))
    norm_cf_val = float(b.to_scalar(norm_cf))

    # Degenerate case: return NaN (undefined, not "maximally different")
    if norm_orig_val <= eps or norm_cf_val <= eps:
        return float("nan")

    # Cosine similarity
    dot_product = b.sum(arr_orig * arr_cf)
    b.eval(dot_product)
    cosine_sim = float(b.to_scalar(dot_product)) / (norm_orig_val * norm_cf_val)

    # Return cosine distance (1 - similarity)
    return 1.0 - cosine_sim


def compute_kurtosis(
    activations: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute excess kurtosis of activation distribution.

    Kurtosis measures the "peakedness" of a distribution:
    - High kurtosis = peaked/concentrated distribution = confident representation
    - Low kurtosis = flat/spread distribution = uncertain representation

    This is computed without scipy dependency using the fourth standardized moment:
    kurtosis = E[(X - μ)^4] / σ^4 - 3

    Args:
        activations: Activation array (any shape, will be flattened).
        backend: Compute backend. If None, uses default.

    Returns:
        Excess kurtosis (Fisher's definition). Normal distribution has kurtosis = 0.
        Returns NaN if:
        - Fewer than 4 samples (statistically undefined)
        - Zero variance (mathematically undefined)
    """
    b = backend or get_default_backend()

    arr = b.array(activations) if not hasattr(activations, "shape") else activations

    # Flatten to 1D
    flat = b.reshape(arr, (-1,))
    n = int(b.shape(flat)[0])

    # Kurtosis requires at least 4 samples - return NaN if insufficient
    if n < 4:
        return float("nan")

    # Compute mean and center
    mean = b.mean(flat)
    centered = flat - mean
    b.eval(centered)

    # Compute variance (second moment)
    var = b.mean(centered * centered)
    b.eval(var)
    var_val = float(b.to_scalar(var))

    eps = division_epsilon(b, flat)
    # Zero variance makes kurtosis undefined - return NaN
    if var_val <= eps:
        return float("nan")

    # Compute fourth moment
    fourth_moment = b.mean(centered * centered * centered * centered)
    b.eval(fourth_moment)
    fourth_val = float(b.to_scalar(fourth_moment))

    # Excess kurtosis = m4/m2^2 - 3
    kurtosis = (fourth_val / (var_val * var_val)) - 3.0

    return kurtosis


def repetition_consistency(
    representations: list["Array"],
    backend: "Backend | None" = None,
) -> float:
    """Compute pairwise cosine similarity across representations.

    This measures encoding stability - how consistent the representation is
    across identical (or near-identical) inference runs.

    Args:
        representations: List of activation arrays from repeated runs.
        backend: Compute backend. If None, uses default.

    Returns:
        Mean pairwise cosine similarity in [-1, 1].
        Returns NaN if fewer than 2 representations (no pairs to compare).
    """
    b = backend or get_default_backend()

    # Need at least 2 representations to compute pairwise similarity
    if len(representations) < 2:
        return float("nan")

    # Convert all to backend arrays and flatten
    arrays = []
    for rep in representations:
        arr = b.array(rep) if not hasattr(rep, "shape") else rep
        if len(b.shape(arr)) > 1:
            arr = b.reshape(arr, (-1,))
        arrays.append(arr)

    # Compute pairwise cosine similarities
    similarities = []
    eps = division_epsilon(b, arrays[0])

    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            norm_i = b.norm(arrays[i])
            norm_j = b.norm(arrays[j])
            b.eval(norm_i, norm_j)

            norm_i_val = float(b.to_scalar(norm_i))
            norm_j_val = float(b.to_scalar(norm_j))

            # Skip degenerate pairs (contributes NaN to mean)
            if norm_i_val <= eps or norm_j_val <= eps:
                similarities.append(float("nan"))
                continue

            dot_product = b.sum(arrays[i] * arrays[j])
            b.eval(dot_product)
            sim = float(b.to_scalar(dot_product)) / (norm_i_val * norm_j_val)
            similarities.append(sim)

    # Filter out NaN values for mean computation
    valid = [s for s in similarities if not math.isnan(s)]
    if not valid:
        return float("nan")

    return sum(valid) / len(valid)


def layer_consistency(
    layer_activations: dict[int, "Array"],
    backend: "Backend | None" = None,
) -> float:
    """Compute CKA-like consistency across consecutive layers.

    Measures how stable the encoding is through model depth:
    - High consistency = stable encoding through depth = locked in
    - Low consistency = changing representation = still processing

    Uses linear CKA (Centered Kernel Alignment) between consecutive layers.

    Args:
        layer_activations: Dict mapping layer_idx to activation array.
        backend: Compute backend. If None, uses default.

    Returns:
        Mean CKA between consecutive layers in [0, 1].
        Returns NaN if fewer than 2 layers (no consecutive pairs).
    """
    b = backend or get_default_backend()

    # Need at least 2 layers for consecutive comparison
    if len(layer_activations) < 2:
        return float("nan")

    # Sort by layer index
    sorted_indices = sorted(layer_activations.keys())

    consistencies = []
    for i in range(len(sorted_indices) - 1):
        idx_curr = sorted_indices[i]
        idx_next = sorted_indices[i + 1]

        arr_curr = layer_activations[idx_curr]
        arr_next = layer_activations[idx_next]

        if not hasattr(arr_curr, "shape"):
            arr_curr = b.array(arr_curr)
        if not hasattr(arr_next, "shape"):
            arr_next = b.array(arr_next)

        cka = _linear_cka(arr_curr, arr_next, b)
        consistencies.append(cka)

    # Filter out NaN values
    valid = [c for c in consistencies if not math.isnan(c)]
    if not valid:
        return float("nan")

    return sum(valid) / len(valid)


def _linear_cka(
    x: "Array",
    y: "Array",
    backend: "Backend",
) -> float:
    """Compute linear CKA between two representations.

    CKA (Centered Kernel Alignment) measures representation similarity
    in a way that is invariant to orthogonal transformations and isotropic scaling.

    Args:
        x: First representation array.
        y: Second representation array.
        backend: Compute backend.

    Returns:
        CKA value in [0, 1]. 1.0 means identical representations (up to linear transform).
        Returns NaN if degenerate (zero HSIC in denominator).
    """
    b = backend

    # Ensure 2D (samples × features)
    if len(b.shape(x)) == 1:
        x = b.reshape(x, (1, -1))
    if len(b.shape(y)) == 1:
        y = b.reshape(y, (1, -1))

    # Center the representations
    x_centered = x - b.mean(x, axis=0)
    y_centered = y - b.mean(y, axis=0)
    b.eval(x_centered, y_centered)

    # Gram matrices
    xxt = b.matmul(x_centered, b.transpose(x_centered))
    yyt = b.matmul(y_centered, b.transpose(y_centered))
    b.eval(xxt, yyt)

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = b.sum(xxt * yyt)
    hsic_xx = b.sum(xxt * xxt)
    hsic_yy = b.sum(yyt * yyt)
    b.eval(hsic_xy, hsic_xx, hsic_yy)

    hsic_xy_val = float(b.to_scalar(hsic_xy))
    hsic_xx_val = float(b.to_scalar(hsic_xx))
    hsic_yy_val = float(b.to_scalar(hsic_yy))

    denom = (hsic_xx_val * hsic_yy_val) ** 0.5
    eps = division_epsilon(b, xxt)

    # Degenerate case: return NaN
    if denom <= eps:
        return float("nan")

    return hsic_xy_val / denom
