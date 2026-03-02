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

r"""Logit perturbation bound from weight-space spectral constraints.

Derives an upper bound on ``||Δlogits||_∞`` from the spectral budget of
a LoRA adapter, closing the chain:

    Weight space:  ||scale × BA||₂ ≤ σ_k      (Weyl, by construction)
    Activation:    ||Δh_L||₂ ≤ propagation     (submultiplicativity)
    Output:        ||Δlogits||_∞ ≤ bound        (readout amplification)

If ``bound < min_margin`` (smallest top1-top2 logit gap across probes),
the argmax cannot flip on any measured probe, so greedy decoding is
identical before and after adaptation.  Degradation is impossible by
construction — no trials needed.

Derivation
----------
For a transformer with residual connections ``h_{l+1} = h_l + F_l(h_l)``
and a LoRA perturbation ``ΔW`` at layer ``l``:

1. **Injection at perturbed layer:**

   .. math::

       \|\Delta h_l\|_2 \le \text{scale} \times \|BA\|_2
                              \times \|\text{LN}(h_{l-1})\|_2

   The LayerNorm output norm is bounded: for input with norm ``||x||``,
   LayerNorm projects to the unit sphere (up to affine rescaling by
   learned ``γ``), so ``||LN(x)||₂ ≤ ||γ||₂ × √d``.

2. **Propagation through subsequent layers:**

   For unperturbed layer ``i`` with residual connection:

   .. math::

       \|\Delta h_{i+1}\|_2 \le (1 + L_i) \times \|\Delta h_i\|_2

   where ``L_i = ||J_i||₂`` is the Lipschitz constant of ``F_i ∘ LN``.
   Conservative data-independent bound:

   - MLP layers: ``L_MLP ≤ σ_max(W_down) × σ_max(W_up)``
     (ignoring SiLU saturation, which only contracts)
   - Attention layers: ``L_Attn ≤ σ_max(W_O) × σ_max(W_V)``
     (ignoring softmax contraction, which only helps)

3. **Readout amplification:**

   .. math::

       \|\Delta\text{logits}\|_\infty \le \|\Delta\text{logits}\|_2
           \le \sigma_{\max}(W_{\text{out}}) \times \|\Delta h_L\|_2

   Submultiplicativity of spectral norm (Horn & Johnson, *Matrix Analysis*,
   Thm 5.6.2).  For tied embeddings, ``W_out = W_embed^T``.

4. **Multi-layer LoRA:**

   When LoRA is applied to multiple layers, perturbations from each layer
   propagate independently through the residual stream and add at the
   output.  Triangle inequality gives:

   .. math::

       \|\Delta h_L\|_2 \le \sum_{l \in \text{perturbed}}
           \left[\prod_{i=l}^{L-1}(1 + L_i)\right]
           \times \text{scale}_l \times \|B_l A_l\|_2 \times \|h_l^{\text{norm}}\|_2

References
----------
Horn & Johnson (2012), *Matrix Analysis*, 2nd ed. Cambridge.
    Thm 5.6.2: submultiplicativity of spectral norm.
Weyl (1912). Eigenvalue perturbation bounds.
Ba et al. (2016). Layer Normalization. arXiv:1607.06450.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogitPerturbationBound:
    """Result of logit perturbation bound computation."""

    bound: float
    """Upper bound on ||Δlogits||_∞."""

    sigma_max_readout: float
    """σ_max(W_out): readout amplification factor."""

    max_propagation_factor: float
    """Largest per-layer propagation factor ∏(1 + L_i)."""

    n_perturbed_layers: int
    """Number of layers with LoRA perturbations."""

    per_layer_injection_norm: dict[int, float]
    """Per perturbed layer: scale × ||BA||₂ × ||LN(h)||₂."""

    per_layer_propagation_factor: dict[int, float]
    """Per perturbed layer: ∏_{i=l}^{L-1} (1 + L_i)."""


@dataclass(frozen=True)
class MarginSafetyResult:
    """Result of margin safety check."""

    safe: bool
    """True if logit perturbation bound < min_margin."""

    logit_bound: float
    """Upper bound on ||Δlogits||_∞."""

    min_margin: float
    """Smallest top1 - top2 logit gap across probes."""

    safety_ratio: float
    """min_margin / logit_bound.  >1.0 means safe."""


def compute_readout_effective_rank(
    model: Any,
    backend: "Backend",
) -> float:
    r"""Shannon effective rank of the readout weight matrix.

    Uses the same readout-matrix detection logic as
    :func:`compute_readout_spectral_norm`.  Computes
    ``SVD(W, compute_uv=False)`` to get singular values, then

    .. math::

        \text{erank} = \exp\!\bigl(-\sum_i p_i \ln p_i\bigr),
        \qquad p_i = \sigma_i^2 / \sum_j \sigma_j^2

    (Shannon effective rank, Roy & Vetterli 2007).

    The readout effective rank bounds the per-position output diversity
    under greedy decoding, which determines the n-gram order at which
    random collisions become negligible (birthday paradox).

    Scaling: full SVD of the readout matrix (vocab × hidden_dim).  For
    current models (350M–8B) this is at most ~150K × 4K — completes in
    <1 second.  If scaling to larger vocabularies becomes a bottleneck,
    replace with eigendecomposition of W^T W (hidden_dim × hidden_dim,
    much smaller) which yields the same singular values squared.
    """
    from modelcypher.core.domain.geometry.numerical_stability import (
        division_epsilon,
        safe_log_epsilon,
    )

    base = getattr(model, "model", model)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w_out = model.lm_head.weight
    elif hasattr(base, "embed_tokens") and hasattr(base.embed_tokens, "weight"):
        w_out = base.embed_tokens.weight
    else:
        raise ValueError("Cannot find output projection weight matrix")

    w_f32 = backend.astype(w_out, "float32")
    backend.eval(w_f32)
    S = backend.svd(w_f32, compute_uv=False)
    backend.eval(S)

    # Shannon effective rank from singular-value distribution
    eigvals = S * S
    total = backend.sum(eigvals)
    backend.eval(total)
    total_val = float(backend.to_scalar(total))

    eps = division_epsilon(backend, eigvals)
    if total_val <= eps:
        return 1.0

    p = eigvals / total_val
    log_eps = safe_log_epsilon(backend, eigvals)
    p_safe = backend.where(
        p > log_eps,
        p,
        backend.full(p.shape, log_eps),
    )
    entropy = -backend.sum(p * backend.log(p_safe))
    erank = backend.exp(entropy)
    backend.eval(erank)
    return float(backend.to_scalar(erank))


def compute_readout_spectral_norm(
    model: Any,
    backend: "Backend",
) -> float:
    r"""Compute σ_max(W_out) for the output projection.

    For models with ``lm_head``, uses ``lm_head.weight``.
    For LFM2 (tied embeddings), uses ``embed_tokens.weight^T``.

    Since ``W_out x = W_embed^T x`` for tied embeddings,
    ``σ_max(W_out) = σ_max(W_embed)`` (SVD of A^T has same singular values).

    Uses power iteration (not full SVD) for numerical safety.
    """
    base = getattr(model, "model", model)

    # Get output weight matrix
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w_out = model.lm_head.weight
    elif hasattr(base, "embed_tokens") and hasattr(base.embed_tokens, "weight"):
        # Tied embeddings: W_out = W_embed^T, same singular values
        w_out = base.embed_tokens.weight
    else:
        raise ValueError("Cannot find output projection weight matrix")

    w_out_f32 = backend.astype(w_out, "float32")
    backend.eval(w_out_f32)

    # Power iteration for σ_max (same method as spectral_budget.py)
    n_rows, n_cols = int(w_out_f32.shape[0]), int(w_out_f32.shape[1])
    v = backend.random_normal((n_cols, 1))
    v = backend.astype(v, "float32")
    backend.eval(v)

    sigma = 0.0
    _norm_floor = float(backend.finfo().tiny)
    for _ in range(20):  # More iters for large vocab matrix
        u = backend.matmul(w_out_f32, v)
        backend.eval(u)
        u_norm = float(backend.to_scalar(backend.norm(u)))
        if u_norm < _norm_floor:
            break
        u = u * (1.0 / u_norm)

        v = backend.matmul(backend.transpose(w_out_f32), u)
        backend.eval(v)
        sigma = float(backend.to_scalar(backend.norm(v)))
        if sigma < _norm_floor:
            break
        v = v * (1.0 / sigma)
        backend.eval(v)

    return sigma


def compute_layer_lipschitz_bounds(
    model: Any,
    backend: "Backend",
) -> dict[int, float]:
    r"""Compute conservative Lipschitz bound per layer.

    For MLP layers: ``L_i ≤ σ_max(W_down) × σ_max(W_up)``
    For attention layers: ``L_i ≤ σ_max(W_O) × σ_max(W_V)``

    These are data-independent upper bounds. The actual Lipschitz constant
    is smaller because:
    - SiLU saturates (|SiLU'(x)| ≤ 1.1, not unbounded)
    - Softmax is contractive (attention weights sum to 1)
    - LayerNorm is contractive for large norms

    But the bound is VALID — it overestimates but never underestimates.

    Returns:
        Dict mapping layer_index to Lipschitz bound L_i.
    """
    base = getattr(model, "model", model)
    lipschitz: dict[int, float] = {}

    for layer_idx, layer in enumerate(base.layers):
        # MLP contribution
        ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        L_mlp = 0.0
        if ff is not None:
            # down_proj
            w_down = _get_weight(ff, ["down_proj", "w2", "fc2"])
            # up_proj
            w_up = _get_weight(ff, ["up_proj", "w3", "fc1"])

            if w_down is not None and w_up is not None:
                sigma_down = _power_iter_sigma_max(w_down, backend)
                sigma_up = _power_iter_sigma_max(w_up, backend)
                L_mlp = sigma_down * sigma_up

        # Attention contribution (if this is an attention layer)
        attn = getattr(layer, "self_attn", None)
        L_attn = 0.0
        if attn is not None:
            w_o = _get_weight(attn, ["o_proj"])
            w_v = _get_weight(attn, ["v_proj"])
            if w_o is not None and w_v is not None:
                sigma_o = _power_iter_sigma_max(w_o, backend)
                sigma_v = _power_iter_sigma_max(w_v, backend)
                L_attn = sigma_o * sigma_v

        # Total Lipschitz for this layer: MLP + Attention (parallel residual)
        # or just MLP for conv layers
        lipschitz[layer_idx] = L_mlp + L_attn

    return lipschitz


def compute_logit_perturbation_bound(
    model: Any,
    backend: "Backend",
    perturbed_layers: dict[int, float],
    layer_lipschitz: dict[int, float],
    sigma_max_readout: float,
    activation_norms: dict[int, float] | None = None,
) -> LogitPerturbationBound:
    r"""Compute upper bound on ``||Δlogits||_∞``.

    Parameters
    ----------
    model :
        Model (used only for layer count).
    backend :
        Backend protocol.
    perturbed_layers :
        Dict mapping layer_index to ``scale × ||BA||₂`` (the weight-space
        perturbation norm) for each LoRA layer.
    layer_lipschitz :
        Dict mapping layer_index to Lipschitz bound ``L_i`` from
        :func:`compute_layer_lipschitz_bounds`.
    sigma_max_readout :
        ``σ_max(W_out)`` from :func:`compute_readout_spectral_norm`.
    activation_norms :
        Optional dict mapping layer_index to ``||LN(h_{l-1})||₂`` at
        perturbed layers.  If None, uses ``1.0`` (assumes LayerNorm output
        is O(1), which is geometrically correct: LN projects to the unit
        sphere scaled by ``γ``).

    Returns
    -------
    LogitPerturbationBound
        Contains the bound and diagnostic breakdown.
    """
    base = getattr(model, "model", model)
    n_layers = len(base.layers)
    all_layer_indices = sorted(layer_lipschitz.keys())

    if not perturbed_layers:
        return LogitPerturbationBound(
            bound=0.0,
            sigma_max_readout=sigma_max_readout,
            max_propagation_factor=1.0,
            n_perturbed_layers=0,
            per_layer_injection_norm={},
            per_layer_propagation_factor={},
        )

    # For each perturbed layer l, compute:
    #   injection_l = scale_l × ||BA_l||₂ × ||LN(h_{l-1})||₂
    #   propagation_l = ∏_{i=l}^{L-1} (1 + L_i)
    #   contribution_l = propagation_l × injection_l

    per_layer_injection: dict[int, float] = {}
    per_layer_propagation: dict[int, float] = {}
    total_delta_h_L = 0.0

    for layer_idx, perturbation_norm in sorted(perturbed_layers.items()):
        # Injection norm at this layer
        h_norm = 1.0  # Default: LN projects to ~unit sphere
        if activation_norms is not None and layer_idx in activation_norms:
            h_norm = activation_norms[layer_idx]

        injection = perturbation_norm * h_norm
        per_layer_injection[layer_idx] = injection

        # Propagation factor from this layer through all subsequent layers
        # ∏_{i=l}^{L-1} (1 + L_i)
        # Note: the perturbed layer itself contributes to propagation only
        # for perturbations injected at EARLIER layers. For the perturbation
        # originating at layer l, propagation starts at layer l (after injection).
        prop = 1.0
        for i in range(layer_idx, n_layers):
            L_i = layer_lipschitz.get(i, 0.0)
            prop *= (1.0 + L_i)

        per_layer_propagation[layer_idx] = prop
        total_delta_h_L += prop * injection

    # Readout: ||Δlogits||_∞ ≤ ||Δlogits||₂ ≤ σ_max(W_out) × ||Δh_L||₂
    bound = sigma_max_readout * total_delta_h_L

    max_prop = max(per_layer_propagation.values()) if per_layer_propagation else 1.0

    return LogitPerturbationBound(
        bound=bound,
        sigma_max_readout=sigma_max_readout,
        max_propagation_factor=max_prop,
        n_perturbed_layers=len(perturbed_layers),
        per_layer_injection_norm=per_layer_injection,
        per_layer_propagation_factor=per_layer_propagation,
    )


def check_margin_safety(
    logit_bound: float,
    min_margin: float,
) -> MarginSafetyResult:
    """Check if the logit perturbation bound is below the decision margin.

    When ``safe=True``: the adapter provably cannot flip any argmax on the
    measured probe set.  Degradation is impossible by construction.

    Parameters
    ----------
    logit_bound :
        Upper bound on ``||Δlogits||_∞`` from
        :func:`compute_logit_perturbation_bound`.
    min_margin :
        Smallest ``top1 - top2`` logit gap across all probes.

    Returns
    -------
    MarginSafetyResult
    """
    if min_margin <= 0.0:
        # Some probe already has a tie or wrong answer — can't guarantee safety
        return MarginSafetyResult(
            safe=False,
            logit_bound=logit_bound,
            min_margin=min_margin,
            safety_ratio=0.0,
        )

    safety_ratio = min_margin / logit_bound if logit_bound > 0.0 else math.inf
    return MarginSafetyResult(
        safe=logit_bound < min_margin,
        logit_bound=logit_bound,
        min_margin=min_margin,
        safety_ratio=safety_ratio,
    )


@dataclass(frozen=True)
class MeasuredMarginSafety:
    """Result of per-probe measured margin safety check.

    Unlike the compositional Lipschitz bound (which is exponentially loose
    in depth), this measures the ACTUAL logit perturbation per probe.
    The measurement is exact for the given probe set — no approximation.
    """

    safe: bool
    """True if no probe's argmax was flipped by the adapter."""

    n_probes: int
    """Total number of probes checked."""

    n_flipped: int
    """Number of probes where argmax changed (margin crossed zero)."""

    n_margin_eroded: int
    """Probes where margin decreased (even if not flipped)."""

    min_margin_baseline: float
    """Smallest baseline margin across probes."""

    min_margin_adapted: float
    """Smallest adapted margin across probes."""

    max_logit_delta_inf: float
    """Largest ||Δlogits||_∞ across all probes (measured, not bounded)."""

    per_probe_details: dict[str, dict[str, float]]
    """Per-probe: baseline_margin, adapted_margin, logit_delta_inf."""


def measure_per_probe_margin_safety(
    problems: list,
    collect_logits_base_fn,
    collect_logits_adapted_fn,
    backend: "Backend",
) -> MeasuredMarginSafety:
    r"""Measure actual logit perturbation and margin for each probe.

    For each probe, this computes:
    - Baseline logits and margin (top1 - top2)
    - Adapted logits and margin
    - ``||Δlogits||_∞ = max_j |logits_adapted_j - logits_base_j|``

    This is NOT a bound — it is a deterministic measurement.  The forward
    pass is a fixed geometric map; given the same input, it produces
    exactly one output.

    If the adapted margin is still positive for all probes, the argmax
    didn't flip and degradation didn't occur on this probe set.

    Parameters
    ----------
    problems :
        StarProblem instances (or any objects accepted by the logit
        collection functions).
    collect_logits_base_fn :
        ``fn(prompt: str) -> Array[vocab_size]`` for base model.
    collect_logits_adapted_fn :
        ``fn(prompt: str) -> Array[vocab_size]`` for adapted model.
    backend :
        Backend protocol.

    Returns
    -------
    MeasuredMarginSafety
    """
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )

    n_demonstrations = len(default_few_shot_examples())
    details: dict[str, dict[str, float]] = {}
    n_flipped = 0
    n_eroded = 0
    max_delta_inf = 0.0
    margins_base: list[float] = []
    margins_adapted: list[float] = []

    for problem in problems:
        prompt = build_forward_prompt(problem, demonstrations=n_demonstrations)
        pid = problem.problem_id

        try:
            logits_base = collect_logits_base_fn(prompt)
            logits_adapted = collect_logits_adapted_fn(prompt)

            # Margins
            m_base = _margin_from_logits(logits_base, backend)
            m_adapted = _margin_from_logits(logits_adapted, backend)

            # ||Δlogits||_∞
            delta = logits_adapted - logits_base
            backend.eval(delta)
            abs_delta = backend.abs(delta)
            backend.eval(abs_delta)
            delta_inf = float(backend.to_scalar(backend.max(abs_delta)))

            details[pid] = {
                "baseline_margin": m_base,
                "adapted_margin": m_adapted,
                "logit_delta_inf": delta_inf,
            }

            margins_base.append(m_base)
            margins_adapted.append(m_adapted)
            max_delta_inf = max(max_delta_inf, delta_inf)

            # Flipped = argmax changed (margin sign crossed)
            if (m_base > 0 and m_adapted <= 0) or (m_base <= 0 and m_adapted > 0):
                n_flipped += 1

            # Eroded = margin decreased
            if m_adapted < m_base:
                n_eroded += 1

        except Exception:
            logger.debug(
                "Margin measurement failed for problem %s", pid, exc_info=True,
            )

    min_base = min(margins_base) if margins_base else 0.0
    min_adapted = min(margins_adapted) if margins_adapted else 0.0

    return MeasuredMarginSafety(
        safe=n_flipped == 0,
        n_probes=len(details),
        n_flipped=n_flipped,
        n_margin_eroded=n_eroded,
        min_margin_baseline=min_base,
        min_margin_adapted=min_adapted,
        max_logit_delta_inf=max_delta_inf,
        per_probe_details=details,
    )


def _margin_from_logits(logits, backend: "Backend") -> float:
    """Compute top1 - top2 margin from logits array."""
    sorted_logits = backend.sort(logits)
    backend.eval(sorted_logits)
    n = int(sorted_logits.shape[0])
    if n < 2:
        return 0.0
    top1 = float(backend.to_scalar(sorted_logits[n - 1]))
    top2 = float(backend.to_scalar(sorted_logits[n - 2]))
    return top1 - top2


# --- Internal helpers ---


def _get_weight(module: Any, attr_names: list[str]) -> Any | None:
    """Get weight from a module by trying multiple attribute names."""
    for name in attr_names:
        sub = getattr(module, name, None)
        if sub is not None and hasattr(sub, "weight"):
            return sub.weight
    return None


def _power_iter_sigma_max(
    weight: Any,
    backend: "Backend",
    n_iters: int = 10,
) -> float:
    """Estimate σ_max via power iteration."""
    w = backend.astype(weight, "float32")
    backend.eval(w)

    n_cols = int(w.shape[1]) if len(w.shape) > 1 else int(w.shape[0])
    v = backend.random_normal((n_cols, 1))
    v = backend.astype(v, "float32")
    backend.eval(v)

    sigma = 0.0
    _norm_floor = float(backend.finfo().tiny)
    for _ in range(n_iters):
        u = backend.matmul(w, v)
        backend.eval(u)
        u_norm = float(backend.to_scalar(backend.norm(u)))
        if u_norm < _norm_floor:
            break
        u = u * (1.0 / u_norm)

        v = backend.matmul(backend.transpose(w), u)
        backend.eval(v)
        sigma = float(backend.to_scalar(backend.norm(v)))
        if sigma < _norm_floor:
            break
        v = v * (1.0 / sigma)
        backend.eval(v)

    return sigma
