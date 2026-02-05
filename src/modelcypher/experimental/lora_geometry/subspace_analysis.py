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

"""Subspace overlap analysis for LoRA adapter composability.

Implements:
- Full principal angle spectrum between adapter subspaces
- Spectral overlap metrics
- Behavioral overlap via CKA
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class PrincipalAngles:
    """Full spectrum of principal angles between two subspaces.

    Attributes:
        angles_radians: All principal angles θᵢ in radians.
        angles_degrees: All principal angles θᵢ in degrees.
        max_angle: Maximum principal angle (worst alignment).
        min_angle: Minimum principal angle (best alignment).
        mean_angle: Mean of all principal angles.
    """

    angles_radians: list[float]
    angles_degrees: list[float]
    max_angle: float
    min_angle: float
    mean_angle: float


@dataclass(frozen=True)
class SubspaceOverlapResult:
    """Complete subspace overlap analysis between two LoRA adapters.

    Attributes:
        principal_angles: Full principal angle spectrum.
        spectral_overlap: ||U₁ @ U₁ᵀ @ ΔW₂||_F / ||ΔW₂||_F
        behavioral_overlap: CKA between probe activations (if computed).
        adapter1_id: Identifier of first adapter.
        adapter2_id: Identifier of second adapter.
    """

    principal_angles: PrincipalAngles
    spectral_overlap: float
    behavioral_overlap: float | None
    adapter1_id: str
    adapter2_id: str


def compute_principal_angles(
    delta_w1: "Array",
    delta_w2: "Array",
    backend: "Backend | None" = None,
) -> PrincipalAngles:
    """Compute full principal angle spectrum between two adapter subspaces.

    Principal angles are computed from the SVD of U₁ᵀ @ U₂, where U₁ and U₂
    are the column space bases of ΔW₁ and ΔW₂.

    Args:
        delta_w1: First adapter's perturbation ΔW₁.
        delta_w2: Second adapter's perturbation ΔW₂.
        backend: Compute backend.

    Returns:
        PrincipalAngles with full spectrum.
    """
    if backend is None:
        backend = get_default_backend()

    # Get column space bases via SVD
    # Truncate U to effective rank (columns with non-negligible singular values)
    U1_full, S1, _ = backend.svd(delta_w1)
    U2_full, S2, _ = backend.svd(delta_w2)
    backend.eval(U1_full, S1, U2_full, S2)

    # Determine effective ranks based on singular value threshold
    eps = machine_epsilon(backend, S1)
    s1_list = backend.tolist(S1)
    s2_list = backend.tolist(S2)

    # Threshold: singular values > eps * max_singular_value
    thresh1 = float(s1_list[0]) * eps if s1_list else eps
    thresh2 = float(s2_list[0]) * eps if s2_list else eps

    rank1 = sum(1 for s in s1_list if float(s) > thresh1)
    rank2 = sum(1 for s in s2_list if float(s) > thresh2)

    # Truncate U to effective rank
    U1 = U1_full[:, :rank1] if rank1 > 0 else U1_full[:, :1]
    U2 = U2_full[:, :rank2] if rank2 > 0 else U2_full[:, :1]

    # Overlap matrix
    overlap = backend.matmul(backend.transpose(U1), U2)
    backend.eval(overlap)

    # SVD of overlap gives cos(θᵢ) as singular values
    _, sigmas, _ = backend.svd(overlap)
    backend.eval(sigmas)

    sigma_list = backend.tolist(sigmas)
    eps = machine_epsilon(backend, sigmas)

    # Convert to angles, clamping to [-1, 1] for arccos
    angles_rad: list[float] = []
    for s in sigma_list:
        s_val = float(s)
        s_clamped = max(-1.0, min(1.0, s_val))
        angle = math.acos(s_clamped)
        angles_rad.append(angle)

    angles_deg = [a * 180.0 / math.pi for a in angles_rad]

    if not angles_rad:
        return PrincipalAngles(
            angles_radians=[],
            angles_degrees=[],
            max_angle=math.pi / 2,
            min_angle=0.0,
            mean_angle=math.pi / 4,
        )

    return PrincipalAngles(
        angles_radians=angles_rad,
        angles_degrees=angles_deg,
        max_angle=max(angles_rad),
        min_angle=min(angles_rad),
        mean_angle=sum(angles_rad) / len(angles_rad),
    )


def compute_spectral_overlap(
    delta_w1: "Array",
    delta_w2: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute spectral overlap: ||U₁ @ U₁ᵀ @ ΔW₂||_F / ||ΔW₂||_F.

    This measures how much of ΔW₂ lies in the column space of ΔW₁.
    Range [0, 1]: 0 = orthogonal, 1 = same column space.

    Args:
        delta_w1: First adapter's perturbation.
        delta_w2: Second adapter's perturbation.
        backend: Compute backend.

    Returns:
        Spectral overlap ratio.
    """
    if backend is None:
        backend = get_default_backend()

    # Get column space basis of ΔW₁
    # Truncate to effective rank
    U1_full, S1, _ = backend.svd(delta_w1)
    backend.eval(U1_full, S1)

    eps = machine_epsilon(backend, delta_w1)
    s1_list = backend.tolist(S1)
    thresh1 = float(s1_list[0]) * eps if s1_list else eps
    rank1 = sum(1 for s in s1_list if float(s) > thresh1)
    U1 = U1_full[:, :rank1] if rank1 > 0 else U1_full[:, :1]

    # Frobenius norm of ΔW₂
    norm_delta2 = float(backend.to_scalar(backend.norm(delta_w2)))

    if norm_delta2 < eps:
        return 0.0

    # Project ΔW₂ onto U₁'s column space: U₁ @ U₁ᵀ @ ΔW₂
    projection = backend.matmul(U1, backend.matmul(backend.transpose(U1), delta_w2))
    backend.eval(projection)

    norm_proj = float(backend.to_scalar(backend.norm(projection)))

    return norm_proj / norm_delta2


def compute_behavioral_overlap(
    activations1: "Array",
    activations2: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute behavioral overlap via CKA (Centered Kernel Alignment).

    CKA measures similarity between activation patterns independent of
    orthogonal transformations and isotropic scaling.

    Args:
        activations1: Activations from model + ΔW₁ on probes [n_probes, hidden_dim].
        activations2: Activations from model + ΔW₂ on probes [n_probes, hidden_dim].
        backend: Compute backend.

    Returns:
        CKA similarity in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()

    # Center activations (subtract mean)
    acts1 = backend.astype(activations1, "float32")
    acts2 = backend.astype(activations2, "float32")

    n = acts1.shape[0]
    if n < 2:
        return 0.0

    mean1 = backend.sum(acts1, axis=0) / n
    mean2 = backend.sum(acts2, axis=0) / n

    centered1 = acts1 - mean1
    centered2 = acts2 - mean2
    backend.eval(centered1, centered2)

    # Compute Gram matrices K = X @ X.T
    K1 = backend.matmul(centered1, backend.transpose(centered1))
    K2 = backend.matmul(centered2, backend.transpose(centered2))

    # HSIC (Hilbert-Schmidt Independence Criterion) with linear kernel
    # HSIC(K1, K2) = trace(K1 @ H @ K2 @ H) / (n-1)^2
    # where H = I - 1/n * ones is the centering matrix
    # Simplified: trace(K1_c @ K2_c) where K_c = H @ K @ H

    # Center Gram matrices
    ones = backend.ones((n, n), dtype="float32") / n
    H = backend.eye(n) - ones

    K1_centered = backend.matmul(H, backend.matmul(K1, H))
    K2_centered = backend.matmul(H, backend.matmul(K2, H))
    backend.eval(K1_centered, K2_centered)

    # HSIC values
    hsic_12 = backend.sum(K1_centered * K2_centered)
    hsic_11 = backend.sum(K1_centered * K1_centered)
    hsic_22 = backend.sum(K2_centered * K2_centered)

    backend.eval(hsic_12, hsic_11, hsic_22)

    hsic_12_val = float(backend.to_scalar(hsic_12))
    hsic_11_val = float(backend.to_scalar(hsic_11))
    hsic_22_val = float(backend.to_scalar(hsic_22))

    # CKA = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
    eps = machine_epsilon(backend, K1)
    denom = (hsic_11_val * hsic_22_val) ** 0.5

    if denom < eps:
        return 0.0

    cka = hsic_12_val / denom

    # Clamp to [0, 1] for numerical stability
    return max(0.0, min(1.0, cka))


def compute_subspace_overlap(
    delta_w1: "Array",
    delta_w2: "Array",
    adapter1_id: str,
    adapter2_id: str,
    activations1: "Array | None" = None,
    activations2: "Array | None" = None,
    backend: "Backend | None" = None,
) -> SubspaceOverlapResult:
    """Compute complete subspace overlap analysis between two adapters.

    Args:
        delta_w1: First adapter's perturbation.
        delta_w2: Second adapter's perturbation.
        adapter1_id: Identifier of first adapter.
        adapter2_id: Identifier of second adapter.
        activations1: Optional activations for behavioral overlap.
        activations2: Optional activations for behavioral overlap.
        backend: Compute backend.

    Returns:
        SubspaceOverlapResult with all overlap metrics.
    """
    if backend is None:
        backend = get_default_backend()

    principal_angles = compute_principal_angles(delta_w1, delta_w2, backend)
    spectral_overlap = compute_spectral_overlap(delta_w1, delta_w2, backend)

    behavioral_overlap = None
    if activations1 is not None and activations2 is not None:
        behavioral_overlap = compute_behavioral_overlap(
            activations1, activations2, backend
        )

    return SubspaceOverlapResult(
        principal_angles=principal_angles,
        spectral_overlap=spectral_overlap,
        behavioral_overlap=behavioral_overlap,
        adapter1_id=adapter1_id,
        adapter2_id=adapter2_id,
    )


@dataclass
class ComposabilityResult:
    """Result of adapter composability analysis.

    Attributes:
        overlap: Subspace overlap metrics between the two adapters.
        degradation: max(|ppl_combined - ppl_task1|, |ppl_combined - ppl_task2|)
        ppl_task1: Perplexity on task 1 with combined adapter.
        ppl_task2: Perplexity on task 2 with combined adapter.
        ppl_task1_original: Perplexity on task 1 with adapter 1 alone.
        ppl_task2_original: Perplexity on task 2 with adapter 2 alone.
    """

    overlap: SubspaceOverlapResult
    degradation: float
    ppl_task1: float
    ppl_task2: float
    ppl_task1_original: float
    ppl_task2_original: float


def compute_degradation(
    ppl_combined_task1: float,
    ppl_combined_task2: float,
    ppl_original_task1: float,
    ppl_original_task2: float,
) -> float:
    """Compute degradation metric for adapter composition.

    degradation = max(|ppl_combined - ppl_task1|, |ppl_combined - ppl_task2|)

    Args:
        ppl_combined_task1: Perplexity on task 1 with combined adapter.
        ppl_combined_task2: Perplexity on task 2 with combined adapter.
        ppl_original_task1: Perplexity on task 1 with adapter 1 alone.
        ppl_original_task2: Perplexity on task 2 with adapter 2 alone.

    Returns:
        Degradation metric.
    """
    diff1 = abs(ppl_combined_task1 - ppl_original_task1)
    diff2 = abs(ppl_combined_task2 - ppl_original_task2)
    return max(diff1, diff2)
