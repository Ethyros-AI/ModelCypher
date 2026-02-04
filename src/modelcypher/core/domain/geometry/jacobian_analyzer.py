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

"""Jacobian spectrum analysis for layer transformations.

Computes the singular value spectrum of the Jacobian ∂h_l/∂h_{l-1} at each layer
using randomized methods (avoids materializing the full hidden_dim × hidden_dim matrix).

Key insight from research plan:
- Full Jacobian is [hidden_dim × hidden_dim] = 1M+ entries for typical models
- Randomized SVD using power iteration approximates top-k singular values efficiently
- Singular value spectrum reveals: amplified directions (σ > 1) vs compressed (σ < 1)

Mathematical foundation:
- J = ∂h_l/∂h_{l-1} is the layer transformation Jacobian
- SVD: J = U @ Σ @ V^T
- σ_i tells us how much the i-th direction is amplified/compressed
- Effective rank = exp(H(p)) where p_i = σ_i² / Σσ², H = Shannon entropy
- Condition number κ = σ_max / σ_min indicates numerical stability

Implementation uses randomized range finder:
1. Generate k random probe vectors
2. Compute Y = J @ Ω via JVP (using finite differences)
3. Orthogonalize Y via QR decomposition
4. Compute B = Y^T @ J @ Y (small k×k matrix)
5. SVD of B gives approximate top-k singular values

Integration notes:
- Uses geodesic_svd from numerical_stability for stable SVD
- Uses division_epsilon, safe_log_epsilon for numerical stability
- Computes effective rank directly from singular values (same formulas as EffectiveRank)

References:
- Halko et al. (2011) "Finding structure with randomness"
- Research plan: docs/RESEARCH-PLAN-SMALL-LLM-ANATOMY.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class JacobianProfile:
    """Jacobian spectrum profile for a single layer.

    Attributes:
        layer_idx: Layer index in the model.
        top_k_singular_values: Top-k singular values (descending order).
        effective_rank_renyi: Renyi (order-2) effective rank = (Σσ²)² / Σσ⁴.
        effective_rank_shannon: Shannon effective rank = exp(entropy).
        spectral_entropy: Shannon entropy of normalized singular values.
        condition_number: σ_max / σ_min (numerical stability indicator).
        spectral_gap: σ_1 / σ_2 (gap between top two singular values).
        spectral_decay_rate: Slope of log(σ) vs rank (exponential decay rate).
        norm_amplification: σ_max (worst-case norm amplification).
        hidden_dim: Hidden dimension of the layer.
        num_probes: Number of random probes used for estimation.
    """

    layer_idx: int
    top_k_singular_values: list[float]
    effective_rank_renyi: float
    effective_rank_shannon: float
    spectral_entropy: float
    condition_number: float
    spectral_gap: float
    spectral_decay_rate: float
    norm_amplification: float
    hidden_dim: int
    num_probes: int

    def as_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "layer_idx": self.layer_idx,
            "top_k_singular_values": self.top_k_singular_values,
            "effective_rank_renyi": self.effective_rank_renyi,
            "effective_rank_shannon": self.effective_rank_shannon,
            "spectral_entropy": self.spectral_entropy,
            "condition_number": self.condition_number,
            "spectral_gap": self.spectral_gap,
            "spectral_decay_rate": self.spectral_decay_rate,
            "norm_amplification": self.norm_amplification,
            "hidden_dim": self.hidden_dim,
            "num_probes": self.num_probes,
        }


@dataclass(frozen=True)
class JacobianTraceResult:
    """Full Jacobian trace across all layers.

    Attributes:
        profiles: List of JacobianProfile for each layer.
        prompt: The input prompt used for analysis.
        model_path: Path to the model analyzed.
        total_layers: Total number of layers.
        mean_effective_rank: Mean effective rank across layers.
        mean_condition_number: Mean condition number across layers.
        max_norm_amplification: Maximum norm amplification (worst-case across layers).
        cumulative_amplification: Product of all σ_max values (total amplification).
        bottleneck_layer: Layer with lowest effective rank (information bottleneck).
        expansion_layer: Layer with highest norm amplification.
    """

    profiles: list[JacobianProfile]
    prompt: str
    model_path: str
    total_layers: int
    mean_effective_rank: float
    mean_condition_number: float
    max_norm_amplification: float
    cumulative_amplification: float
    bottleneck_layer: int
    expansion_layer: int

    def as_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "profiles": [p.as_dict() for p in self.profiles],
            "prompt": self.prompt,
            "model_path": self.model_path,
            "total_layers": self.total_layers,
            "mean_effective_rank": self.mean_effective_rank,
            "mean_condition_number": self.mean_condition_number,
            "max_norm_amplification": self.max_norm_amplification,
            "cumulative_amplification": self.cumulative_amplification,
            "bottleneck_layer": self.bottleneck_layer,
            "expansion_layer": self.expansion_layer,
        }


class JacobianAnalyzer:
    """Analyze Jacobian spectrum of layer transformations.

    Uses randomized methods to estimate singular value spectrum without
    materializing the full hidden_dim × hidden_dim Jacobian matrix.

    The Jacobian ∂h_l/∂h_{l-1} tells us exactly how the layer transforms
    representations: which directions are amplified (σ > 1), compressed (σ < 1),
    or preserved (σ ≈ 1).

    Note: Effective rank is computed directly from singular values using the
    same formulas as EffectiveRank (Renyi: (Σσ²)²/Σσ⁴, Shannon: exp(entropy)),
    but without the intermediate SVD that EffectiveRank applies to activation
    matrices.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        num_probes: int = 64,
        num_power_iterations: int = 2,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize Jacobian analyzer.

        Args:
            backend: Compute backend (MLX, JAX, etc.). Uses default if None.
            num_probes: Number of random probes for randomized SVD (default 64).
                Higher = more accurate but slower. 64 is good for top-20 SVs.
            num_power_iterations: Power iterations for improving accuracy (default 2).
            epsilon: Finite difference epsilon for JVP computation.
        """
        self._backend = backend or get_default_backend()
        self._num_probes = num_probes
        self._num_power_iterations = num_power_iterations
        self._epsilon = epsilon

    def _compute_jvp_finite_diff(
        self,
        layer_fn: Callable[["Array"], "Array"],
        x: "Array",
        v: "Array",
    ) -> "Array":
        """Compute Jacobian-vector product J @ v using finite differences.

        JVP: J @ v ≈ (f(x + ε*v) - f(x)) / ε

        This is the fundamental operation for randomized SVD.
        """
        b = self._backend
        eps = self._epsilon

        # Forward evaluation at x
        fx = layer_fn(x)

        # Forward evaluation at x + ε*v
        x_perturbed = x + eps * v
        fx_perturbed = layer_fn(x_perturbed)

        # Finite difference approximation
        jvp = (fx_perturbed - fx) / eps

        return jvp

    def _randomized_svd(
        self,
        layer_fn: Callable[["Array"], "Array"],
        x: "Array",
        k: int,
    ) -> tuple["Array", "Array", "Array"]:
        """Compute top-k SVD of Jacobian using randomized range finder.

        Algorithm (Halko et al. 2011):
        1. Generate random probe matrix Ω ∈ R^{n×k}
        2. Form Y = J @ Ω via JVP
        3. QR decomposition: Q, _ = qr(Y)
        4. Form B = Q^T @ J @ Q (requires k more JVPs)
        5. SVD of B gives approximate singular values

        Uses geodesic_svd from numerical_stability for stable computation.

        Returns:
            U: Left singular vectors [hidden_dim, k]
            S: Singular values [k]
            V: Right singular vectors [hidden_dim, k]
        """
        b = self._backend
        hidden_dim = int(b.shape(x)[-1])

        # Step 1: Generate random probe matrix
        omega = b.random_normal((hidden_dim, k))
        b.eval(omega)

        # Step 2: Form Y = J @ Ω
        # Compute JVP for each column of Ω
        y_cols = []
        for i in range(k):
            v = omega[:, i]
            jv = self._compute_jvp_finite_diff(layer_fn, x, v)
            y_cols.append(jv)

        Y = b.stack(y_cols, axis=1)  # [hidden_dim, k]
        b.eval(Y)

        # Step 3: Power iteration for improved accuracy
        for _ in range(self._num_power_iterations):
            omega = Y
            y_cols = []
            for i in range(k):
                v = omega[:, i]
                jv = self._compute_jvp_finite_diff(layer_fn, x, v)
                y_cols.append(jv)
            Y = b.stack(y_cols, axis=1)
            b.eval(Y)

        # Step 4: QR decomposition for orthonormal basis
        Q, R = b.qr(Y)
        b.eval(Q, R)

        # Step 5: Form small matrix B = Q^T @ J @ Q
        # Compute J @ Q column by column
        jq_cols = []
        for i in range(k):
            v = Q[:, i]
            jv = self._compute_jvp_finite_diff(layer_fn, x, v)
            jq_cols.append(jv)

        JQ = b.stack(jq_cols, axis=1)  # [hidden_dim, k]
        b.eval(JQ)

        B = b.matmul(b.transpose(Q), JQ)  # [k, k]
        b.eval(B)

        # Step 6: SVD of small matrix B using geodesic_svd for stability
        U_b, S, Vt_b = geodesic_svd(b, B, k=k)
        b.eval(U_b, S, Vt_b)

        # Map back to full space
        U = b.matmul(Q, U_b)
        V = b.matmul(Q, b.transpose(Vt_b))

        return U, S, V

    def compute_layer_jacobian_profile(
        self,
        layer_fn: Callable[["Array"], "Array"],
        input_activation: "Array",
        layer_idx: int,
    ) -> JacobianProfile:
        """Compute Jacobian spectrum profile for a single layer.

        Args:
            layer_fn: Function that computes layer(x) -> h.
            input_activation: Input activation h_{l-1} [hidden_dim].
            layer_idx: Layer index for labeling.

        Returns:
            JacobianProfile with singular value spectrum and derived metrics.
        """
        b = self._backend

        # Ensure input is 1D
        x = input_activation
        if len(b.shape(x)) > 1:
            x = b.reshape(x, (-1,))

        hidden_dim = int(b.shape(x)[0])
        k = min(self._num_probes, hidden_dim)

        # Compute randomized SVD
        _, singular_values, _ = self._randomized_svd(layer_fn, x, k)
        b.eval(singular_values)

        # Convert to Python floats
        sv_list = b.tolist(singular_values)
        if not isinstance(sv_list, list):
            sv_list = [sv_list]

        # Filter out non-positive values and sort descending
        eps = division_epsilon(b, singular_values)
        sv_list = sorted([max(0.0, float(s)) for s in sv_list], reverse=True)

        # Compute effective rank directly from singular values
        # (EffectiveRank expects activation matrices, not pre-computed SVs)
        sv_squared = [s * s for s in sv_list if s > eps]
        sum_sv_sq = sum(sv_squared)
        sum_sv_fourth = sum(s * s for s in sv_squared)

        if sum_sv_fourth > eps and sum_sv_sq > eps:
            renyi_rank = (sum_sv_sq * sum_sv_sq) / sum_sv_fourth
        else:
            renyi_rank = 0.0

        if sum_sv_sq > eps:
            import math

            log_eps = safe_log_epsilon(b, singular_values)
            p = [s / sum_sv_sq for s in sv_squared]
            spectral_entropy = -sum(pi * math.log(pi + log_eps) for pi in p if pi > log_eps)
            shannon_rank = math.exp(spectral_entropy)
        else:
            spectral_entropy = 0.0
            shannon_rank = 0.0

        # Compute condition number and spectral gap
        sv_nonzero = [s for s in sv_list if s > eps]

        if len(sv_nonzero) >= 2:
            condition_number = sv_nonzero[0] / sv_nonzero[-1]
            spectral_gap = sv_nonzero[0] / sv_nonzero[1]
        elif len(sv_nonzero) == 1:
            condition_number = 1.0
            spectral_gap = float("inf")
        else:
            condition_number = float("inf")
            spectral_gap = float("inf")

        # Spectral decay rate (slope of log(σ) vs rank)
        log_eps = safe_log_epsilon(b, singular_values)
        if len(sv_nonzero) >= 2:
            import math

            log_sv = [math.log(s + log_eps) for s in sv_nonzero[: min(10, len(sv_nonzero))]]
            ranks = list(range(len(log_sv)))
            n = len(ranks)
            mean_r = sum(ranks) / n
            mean_log = sum(log_sv) / n
            cov = sum((r - mean_r) * (ls - mean_log) for r, ls in zip(ranks, log_sv))
            var_r = sum((r - mean_r) ** 2 for r in ranks)
            spectral_decay_rate = cov / var_r if var_r > eps else 0.0
        else:
            spectral_decay_rate = 0.0

        # Norm amplification (worst-case)
        norm_amplification = sv_list[0] if sv_list else 0.0

        return JacobianProfile(
            layer_idx=layer_idx,
            top_k_singular_values=sv_list[: min(20, len(sv_list))],  # Keep top-20
            effective_rank_renyi=renyi_rank,
            effective_rank_shannon=shannon_rank,
            spectral_entropy=spectral_entropy,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            spectral_decay_rate=spectral_decay_rate,
            norm_amplification=norm_amplification,
            hidden_dim=hidden_dim,
            num_probes=k,
        )


def trace_jacobian_spectrum(
    model: Any,
    tokenizer: Any,
    prompt: str,
    model_path: str = "",
    num_probes: int = 64,
    backend: "Backend | None" = None,
) -> JacobianTraceResult:
    """Trace Jacobian spectrum across all layers for a prompt.

    This is the main entry point for Jacobian analysis. It:
    1. Tokenizes the prompt
    2. Runs forward pass capturing layer inputs/outputs
    3. Computes Jacobian profile at each layer
    4. Aggregates into JacobianTraceResult

    Args:
        model: Loaded model (e.g., from ModelLoader.load_model).
        tokenizer: Tokenizer for the model.
        prompt: Text prompt to analyze.
        model_path: Path to model (for metadata).
        num_probes: Number of random probes for randomized SVD.
        backend: Compute backend (uses default if None).

    Returns:
        JacobianTraceResult with per-layer profiles and aggregate metrics.
    """
    b = backend or get_default_backend()
    analyzer = JacobianAnalyzer(backend=b, num_probes=num_probes)

    # Tokenize
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    if isinstance(tokens, list):
        token_ids = tokens
    else:
        token_ids = list(tokens.ids)
    input_ids = b.array([token_ids])

    # Get model structure
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    embed_module = getattr(base_model, "embed_tokens", None)

    if layers is None or embed_module is None:
        raise ValueError("Could not find model layers or embedding module")

    num_layers = len(layers)
    profiles: list[JacobianProfile] = []

    # Get initial embedding
    h = embed_module(input_ids)
    b.eval(h)

    # Mean pool to single vector for Jacobian computation
    h_pooled = b.mean(h, axis=(0, 1))
    b.eval(h_pooled)

    for layer_idx, layer in enumerate(layers):
        # Create layer function for Jacobian computation
        # Capture backend by reference for closure
        def make_layer_fn(layer_ref, backend_ref):
            def layer_fn(x: "Array") -> "Array":
                # Reshape back to [1, 1, hidden_dim] for layer
                x_reshaped = backend_ref.reshape(x, (1, 1, -1))
                result = layer_ref(x_reshaped)
                if isinstance(result, tuple):
                    result = result[0]
                # Mean pool output
                return backend_ref.mean(result, axis=(0, 1))
            return layer_fn

        layer_fn = make_layer_fn(layer, b)

        # Compute Jacobian profile
        profile = analyzer.compute_layer_jacobian_profile(
            layer_fn=layer_fn,
            input_activation=h_pooled,
            layer_idx=layer_idx,
        )
        profiles.append(profile)

        # Update h for next layer
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result
        b.eval(h)
        h_pooled = b.mean(h, axis=(0, 1))
        b.eval(h_pooled)

    # Compute aggregate metrics
    valid_profiles = [p for p in profiles if p.effective_rank_shannon > 0]

    if valid_profiles:
        mean_effective_rank = sum(p.effective_rank_shannon for p in valid_profiles) / len(
            valid_profiles
        )
        finite_conditions = [
            p.condition_number for p in valid_profiles if p.condition_number < float("inf")
        ]
        mean_condition_number = (
            sum(finite_conditions) / len(finite_conditions) if finite_conditions else 0.0
        )
        max_norm_amplification = max(p.norm_amplification for p in profiles)

        # Cumulative amplification = product of all σ_max
        cumulative = 1.0
        for p in profiles:
            if p.norm_amplification > 0:
                cumulative *= p.norm_amplification

        # Find bottleneck (lowest effective rank)
        bottleneck_layer = min(
            range(len(profiles)), key=lambda i: profiles[i].effective_rank_shannon
        )

        # Find expansion layer (highest norm amplification)
        expansion_layer = max(range(len(profiles)), key=lambda i: profiles[i].norm_amplification)
    else:
        mean_effective_rank = 0.0
        mean_condition_number = 0.0
        max_norm_amplification = 0.0
        cumulative = 0.0
        bottleneck_layer = 0
        expansion_layer = 0

    return JacobianTraceResult(
        profiles=profiles,
        prompt=prompt,
        model_path=model_path,
        total_layers=num_layers,
        mean_effective_rank=mean_effective_rank,
        mean_condition_number=mean_condition_number,
        max_norm_amplification=max_norm_amplification,
        cumulative_amplification=cumulative,
        bottleneck_layer=bottleneck_layer,
        expansion_layer=expansion_layer,
    )


__all__ = [
    "JacobianProfile",
    "JacobianTraceResult",
    "JacobianAnalyzer",
    "trace_jacobian_spectrum",
]
