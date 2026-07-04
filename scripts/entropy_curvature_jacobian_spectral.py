#!/usr/bin/env python3
"""B7: Jacobian Spectral Probe — Norm-Entropy Coupling Mechanism.

Measures per-layer Jacobian spectral structure (leading singular value σ₁)
via power iteration on finite-difference Jacobian-vector products, at the
sublayer level, to determine whether sublayer amplification (σ₁ > 1) vs
contraction (σ₁ < 1) varies with entropy differently across architectures.

B6.1 confirmed norm-entropy coupling direction varies by architecture:
    - LFM2: r(H, log(||h||²)) strongly negative (-0.80 to -0.92)
    - Llama/Mistral/Qwen3: strongly positive (+0.60 to +0.77)
    - Qwen3.5: near zero (-0.07 to +0.11)

This step tests whether the Jacobian spectral structure explains that
difference: does σ₁(J_core) track entropy differently in LFM2 vs Qwen3.5?

Method: Power iteration on finite-difference Jacobian-vector products:
    Jv ≈ (sublayer(h + ε·v) - sublayer(h)) / ε
    v_{k+1} = Jv / ||Jv||,  σ₁ = ||Jv||

Constants (IEEE 754 derived):
    ε = √(eps_bf16) = √(2⁻⁷) ≈ 0.0884
    Convergence tolerance = √(eps_bf16)
    Max power iterations = 10 (with convergence check)
    Query position = -2 (same as estimate_bl_jacobian.py)

Falsifiers:
    F1: Jacobian-norm consistency — sign(r(H, σ₁_core | depth)) matches
        B6.1 norm coupling sign per model
    F2: Norm change matches Jacobian — sign(r(H, Δ_core | depth)) ==
        sign(r(H, σ₁_core | depth)) per model
    F3: Cross-family divergence — LFM2 and Qwen3.5 show different
        σ₁-entropy coupling patterns
    F4: Convergence quality — ≥80% of power iterations converge

Usage:
    poetry run python scripts/entropy_curvature_jacobian_spectral.py --smoke
    poetry run python scripts/entropy_curvature_jacobian_spectral.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")

# IEEE 754 bf16 machine epsilon: 2^-7 = 0.0078125
_EPS_BF16 = math.ldexp(1.0, -7)
_SQRT_EPS_BF16 = math.sqrt(_EPS_BF16)

# Finite-difference perturbation step (same as estimate_bl_jacobian.py)
_EPSILON = _SQRT_EPS_BF16  # ≈ 0.0884

# Power iteration convergence tolerance (same scale as perturbation)
_CONVERGENCE_TOL = _SQRT_EPS_BF16

# Max power iterations (with early exit on convergence)
_MAX_POWER_ITERS = 10

# Query position (second-to-last token, same as estimate_bl_jacobian.py)
_QUERY_POS = -2

# Permutation count (matches convention)
N_PERMUTATIONS = 500

# B6.1 results base for cross-reference
B6_1_BASE = Path("results/entropy_curvature_norm_corrected_sublayer")

OUTPUT_BASE = Path("results/entropy_curvature_jacobian_spectral")


# ---------------------------------------------------------------------------
# Model registry (two models: smallest viable, opposite norm-entropy signs)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024,
        "architecture": "lfm2",
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
    },
}

# Same 30 probes as operator_split for cross-validation with B6.1 data
PROBES = [
    # Retrieval
    "The capital of France is",
    "Who wrote Romeo and Juliet?",
    "The chemical symbol for water is",
    "The largest planet in our solar system is",
    "The first president of the United States was",
    "The boiling point of water at sea level is",
    # Arithmetic
    "What is 347 + 528?",
    "What is 15 * 23?",
    "What is 1024 / 16?",
    "What is 99 - 37?",
    "What is 8 * 7 + 13?",
    "What is 256 + 384 - 100?",
    # Reasoning
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
    "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    # Creative
    "Write a haiku about the ocean.",
    "Describe a sunset over the mountains in one vivid sentence.",
    "Write a short poem about the passage of time.",
    "Describe the taste of your favorite food using only three words.",
    "Write a one-sentence story with a twist ending.",
    "Describe the sound of rain on a tin roof.",
    # Narrative
    "Once upon a time in a faraway kingdom, there lived a",
    "The old lighthouse keeper watched the storm approach from",
    "In the year 2150, humanity had finally achieved",
    "She opened the letter and read the first line:",
    "The forest was silent except for the sound of",
    "He had been walking for three days when he finally saw",
]


# ---------------------------------------------------------------------------
# Statistical utilities (same as B6.1 norm_corrected_sublayer)
# ---------------------------------------------------------------------------


def _residualize_ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing linear effect of x."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _effective_df(residuals: np.ndarray) -> tuple[float, float]:
    """Effective degrees of freedom under AR(1) autocorrelation.

    Bretherton et al. (1999) Eq. 31: n_eff = n * (1 - rho_1) / (1 + rho_1).
    """
    n = len(residuals)
    if n < 4:
        return float(n), 0.0
    r = np.asarray(residuals, dtype=float)
    r_centered = r - np.mean(r)
    denom = np.sum(r_centered**2)
    if abs(denom) < 1e-15:
        return float(n), 0.0
    rho_1 = float(np.sum(r_centered[:-1] * r_centered[1:]) / denom)
    rho_1_clamped = max(-0.99, min(0.99, rho_1))
    n_eff = n * (1 - rho_1_clamped) / (1 + rho_1_clamped)
    n_eff = max(4.0, min(float(n), n_eff))
    return n_eff, rho_1


def _minimum_detectable_effect(n_eff: float, n_controls: int = 1) -> float:
    """Fisher-SE MDE: smallest |r| with SNR >= 1 at given effective df."""
    df = n_eff - n_controls - 2
    if df <= 0:
        return 1.0
    return float(np.tanh(1.0 / np.sqrt(df)))


def safe_spearman(x, y):
    """Spearman correlation with NaN handling."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    try:
        r, p = sp_stats.spearmanr(x_arr[mask], y_arr[mask])
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def _permutation_test(
    x_resid: np.ndarray,
    y_resid: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
) -> dict:
    """Empirical p-value via permutation of x_resid."""
    n = len(x_resid)
    if n < 4:
        return {
            "observed_abs_r": float("nan"),
            "exceedance_fraction": float("nan"),
            "null_mean": float("nan"),
            "null_max": float("nan"),
            "n_permutations": n_permutations,
        }

    obs_r, _ = sp_stats.spearmanr(x_resid, y_resid)
    obs_abs_r = abs(float(obs_r))

    rng = np.random.default_rng(seed=seed)
    null_abs_r = np.empty(n_permutations)

    for i in range(n_permutations):
        x_perm = rng.permutation(x_resid)
        r_perm, _ = sp_stats.spearmanr(x_perm, y_resid)
        null_abs_r[i] = abs(float(r_perm))

    exceedance = float(np.mean(null_abs_r >= obs_abs_r))

    return {
        "observed_abs_r": obs_abs_r,
        "exceedance_fraction": exceedance,
        "null_mean": float(np.mean(null_abs_r)),
        "null_max": float(np.max(null_abs_r)),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Model infrastructure (from operator_split / estimate_bl_jacobian)
# ---------------------------------------------------------------------------


def _resolve_backbone(model):
    """Resolve model backbone to the level with embed_tokens and layers."""
    base = getattr(model, "model", None)
    if base is not None:
        if (
            getattr(base, "layers", None) is not None
            and getattr(base, "embed_tokens", None) is not None
        ):
            return base
        lm = getattr(base, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if (
                inner is not None
                and getattr(inner, "layers", None) is not None
            ):
                return inner
            if getattr(lm, "layers", None) is not None:
                return lm
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if (
            inner is not None
            and getattr(inner, "layers", None) is not None
        ):
            return inner
        if getattr(lm, "layers", None) is not None:
            return lm
    return model


def _resolve_final_norm(base):
    """Final readout norm, aligned with backend readout order."""
    return getattr(base, "embedding_norm", None) or getattr(base, "norm", None)


def _resolve_output_head(model, base):
    """Resolve callable output projection module."""
    candidates = []
    if hasattr(model, "lm_head"):
        candidates.append(getattr(model, "lm_head"))
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        candidates.append(getattr(model.model, "lm_head"))
    if hasattr(base, "lm_head"):
        candidates.append(getattr(base, "lm_head"))
    if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
        candidates.append(getattr(base.language_model, "lm_head"))
    for head in candidates:
        if callable(head):
            return head
    return None


def _get_pre_norm(layer):
    """Find pre-attention norm."""
    for name in ("input_layernorm", "operator_norm", "attention_norm"):
        norm = getattr(layer, name, None)
        if norm is not None:
            return norm
    return None


def _call_with_fallback(module, x, mask=None):
    """Try multiple calling conventions for sublayer modules."""
    try:
        if mask is not None:
            return module(x, mask=mask)
        return module(x)
    except Exception:
        pass
    try:
        if mask is not None:
            return module(x, mask=mask, cache=None)
        return module(x, cache=None)
    except Exception:
        pass
    try:
        if mask is not None:
            return module(x, mask)
        return module(x)
    except Exception:
        pass
    return module(x)


def _replace_query_row(hidden, q_idx: int, new_row, mx):
    """Return hidden with query row replaced by new_row [1,1,d]."""
    seq_len = int(hidden.shape[1])
    if q_idx == 0:
        return mx.concatenate([new_row, hidden[:, 1:, :]], axis=1)
    if q_idx == seq_len - 1:
        return mx.concatenate([hidden[:, :-1, :], new_row], axis=1)
    return mx.concatenate(
        [hidden[:, :q_idx, :], new_row, hidden[:, q_idx + 1:, :]],
        axis=1,
    )


def _entropy_from_logits(logits, mx):
    """Stable Shannon entropy from logits tensor."""
    logits = logits.astype(mx.float32)
    logits = logits - mx.max(logits, axis=-1, keepdims=True)
    probs = mx.softmax(logits, axis=-1)
    probs = probs + 1e-12
    entropy = -mx.sum(probs * mx.log(probs), axis=-1)
    mx.eval(entropy)
    return float(np.array(entropy.tolist(), dtype=np.float32).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Core sublayer forward helpers
# ---------------------------------------------------------------------------


def _resolve_sublayer_components(layer):
    """Extract sublayer components from a layer.

    Returns (input_norm, core_module, core_type, post_attn_norm, mlp, layer_mask_fn).
    core_type is one of: "attention", "conv", "linear_attn", "identity".
    """
    input_norm = _get_pre_norm(layer)
    self_attn = getattr(layer, "self_attn", None)
    conv = getattr(layer, "conv", None)
    linear_attn = getattr(layer, "linear_attn", None)

    post_attn_norm = getattr(layer, "post_attention_layernorm", None)
    if post_attn_norm is None:
        post_attn_norm = getattr(layer, "ffn_norm", None)

    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        mlp = getattr(layer, "feed_forward", None)

    if input_norm is not None and self_attn is not None and mlp is not None:
        return input_norm, self_attn, "attention", post_attn_norm, mlp
    elif input_norm is not None and conv is not None and mlp is not None:
        return input_norm, conv, "conv", post_attn_norm, mlp
    elif input_norm is not None and linear_attn is not None and mlp is not None:
        return input_norm, linear_attn, "linear_attn", post_attn_norm, mlp
    elif input_norm is not None and mlp is not None:
        return input_norm, None, "identity", post_attn_norm, mlp
    else:
        return None, None, None, None, None


def _sublayer_core_forward(h_in, input_norm, core_module, core_type, layer_mask, mx):
    """h_in → h_post_core via core operator (norm + core + residual).

    For identity core: h_post_core = h_in (no-op).
    For all others: h_post_core = h_in + core(input_norm(h_in)).
    """
    if core_type == "identity":
        return h_in

    normed = input_norm(h_in)
    core_out = _call_with_fallback(core_module, normed, mask=layer_mask)
    if isinstance(core_out, tuple):
        core_out = core_out[0]
    return h_in + core_out


def _sublayer_mlp_forward(h_post_core, post_attn_norm, mlp, mx):
    """h_post_core → h_out via MLP (norm + mlp + residual)."""
    normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
    mlp_out = mlp(normed2)
    if isinstance(mlp_out, tuple):
        mlp_out = mlp_out[0]
    return h_post_core + mlp_out


# ---------------------------------------------------------------------------
# Power iteration for σ₁(J_sublayer)
# ---------------------------------------------------------------------------


def _power_iterate_sigma1(
    sublayer_fn,
    h_base,
    q_idx: int,
    hidden_dim: int,
    epsilon: float,
    mx,
    rng: np.random.Generator,
) -> dict:
    """Estimate σ₁ of the Jacobian of sublayer_fn at h_base via power iteration.

    sublayer_fn: callable h_full → h_out_full (operates on full [1, seq, d] tensor)
    h_base: baseline hidden state [1, seq, d]
    q_idx: resolved query position index (non-negative)
    hidden_dim: d
    epsilon: finite-difference step
    rng: numpy random generator for initial vector

    Returns dict with sigma1, converged (bool), n_iters, convergence_history.
    """
    # Baseline output at query position
    out_base = sublayer_fn(h_base)
    out_base_q = out_base[0, q_idx, :].astype(mx.float32)
    mx.eval(out_base_q)
    out_base_np = np.array(out_base_q.tolist(), dtype=np.float32)

    # Initial random unit vector
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-12:
        v = np.ones(hidden_dim, dtype=np.float32) / np.sqrt(hidden_dim)
    else:
        v = v / v_norm

    sigma1 = 0.0
    sigma1_prev = 0.0
    converged = False
    history = []

    # Pre-extract base query row (avoids redundant work each iteration)
    h_base_q = h_base[0, q_idx, :].astype(mx.float32)
    mx.eval(h_base_q)
    h_q_np = np.array(h_base_q.tolist(), dtype=np.float32)

    for iteration in range(_MAX_POWER_ITERS):
        # Perturb h_base at query position: h_q + ε·v
        h_q_pert = h_q_np + epsilon * v
        new_row = mx.array(h_q_pert.reshape(1, 1, -1))
        h_pert = _replace_query_row(h_base, q_idx, new_row, mx)

        # Forward through sublayer
        out_pert = sublayer_fn(h_pert)
        out_pert_q = out_pert[0, q_idx, :].astype(mx.float32)
        mx.eval(out_pert_q)
        out_pert_np = np.array(out_pert_q.tolist(), dtype=np.float32)

        # Jv ≈ (out_pert - out_base) / ε
        jv = (out_pert_np - out_base_np) / epsilon

        sigma1_prev = sigma1
        sigma1 = float(np.linalg.norm(jv))
        history.append(sigma1)

        if sigma1 < 1e-12:
            # Jacobian is effectively zero at this point
            converged = True
            break

        v_new = jv / sigma1

        # Convergence: eigenvector direction OR eigenvalue stability
        delta_v = float(np.linalg.norm(v_new - v))
        if delta_v < _CONVERGENCE_TOL:
            converged = True
            v = v_new
            break

        # Eigenvalue convergence: |σ₁_new - σ₁_old| / σ₁_new < tol
        if iteration > 0 and sigma1 > 1e-12:
            rel_change = abs(sigma1 - sigma1_prev) / sigma1
            if rel_change < _CONVERGENCE_TOL:
                converged = True
                v = v_new
                break

        v = v_new

    return {
        "sigma1": sigma1,
        "converged": converged,
        "n_iters": len(history),
        "convergence_history": [float(s) for s in history],
    }


# ---------------------------------------------------------------------------
# H_logit_norm computation (from operator_split)
# ---------------------------------------------------------------------------


def _compute_h_logit_norm(h_out, final_norm, output_head, embed_tokens, mx):
    """Compute H_logit_norm: apply final_norm then unembedding → Shannon H."""
    h_mx = mx.array(h_out.reshape(1, 1, -1))
    if final_norm is not None:
        h_mx = final_norm(h_mx)

    if output_head is not None:
        logits = output_head(h_mx)
    else:
        logits = embed_tokens.as_linear(h_mx)
    if isinstance(logits, tuple):
        logits = logits[0]

    return _entropy_from_logits(logits[0, 0, :], mx)


# ---------------------------------------------------------------------------
# Per-model data collection
# ---------------------------------------------------------------------------


def collect_jacobian_data(
    model, tokenizer, model_name: str, model_info: dict,
    probes: list[str], backend,
) -> dict:
    """Collect per-layer Jacobian spectral data for all probes.

    Returns dict with per_layer measurements (averaged over probes) and
    per_probe raw data.
    """
    import mlx.core as mx

    num_layers = model_info["L"]
    hidden_dim = model_info["d"]

    base = _resolve_backbone(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    final_norm = _resolve_final_norm(base)
    output_head = _resolve_output_head(model, base)

    rng = np.random.default_rng(seed=42)

    # Storage: per_layer[layer_idx] = list of probe dicts
    per_layer = [[] for _ in range(num_layers)]

    for pi, prompt in enumerate(probes):
        if pi % 5 == 0:
            logger.info("  [%s] Probe %d/%d", model_name, pi + 1, len(probes))

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        # Resolve query position
        q_idx = seq_len + _QUERY_POS  # _QUERY_POS = -2
        if q_idx < 0:
            q_idx = 0

        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden

            # Resolve mask
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            # Resolve sublayer components
            components = _resolve_sublayer_components(layer)
            input_norm, core_module, core_type, post_attn_norm, mlp = components

            if core_type is None:
                # Fallback: run full layer, no decomposition
                try:
                    h_out = layer(h_in, mask=layer_mask)
                except (TypeError, ValueError):
                    try:
                        h_out = layer(h_in, layer_mask)
                    except (TypeError, ValueError):
                        h_out = layer(h_in)
                if isinstance(h_out, tuple):
                    h_out = h_out[0]

                # Record without Jacobian data
                h_in_q = h_in[0, q_idx, :].astype(mx.float32)
                h_out_q = h_out[0, q_idx, :].astype(mx.float32)
                mx.eval(h_in_q, h_out_q)
                h_in_np = np.array(h_in_q.tolist(), dtype=np.float32)
                h_out_np = np.array(h_out_q.tolist(), dtype=np.float32)

                per_layer[i].append({
                    "probe_idx": pi,
                    "core_type": None,
                    "sigma1_core": None,
                    "sigma1_mlp": None,
                    "core_converged": None,
                    "mlp_converged": None,
                    "h_in_norm_sq": float(np.dot(h_in_np, h_in_np)),
                    "h_post_core_norm_sq": None,
                    "h_out_norm_sq": float(np.dot(h_out_np, h_out_np)),
                    "delta_core_norm_sq": None,
                    "delta_mlp_norm_sq": None,
                    "H_logit_norm": _compute_h_logit_norm(
                        h_out_np, final_norm, output_head, embed, mx
                    ),
                })
                hidden = h_out
                mx.eval(hidden)
                continue

            # --- Core sublayer Jacobian ---
            def core_fn(h):
                return _sublayer_core_forward(
                    h, input_norm, core_module, core_type, layer_mask, mx
                )

            core_result = _power_iterate_sigma1(
                core_fn, h_in, q_idx, hidden_dim, _EPSILON, mx, rng,
            )

            # Baseline core forward
            h_post_core = core_fn(h_in)
            mx.eval(h_post_core)

            # --- MLP sublayer Jacobian ---
            def mlp_fn(h):
                return _sublayer_mlp_forward(h, post_attn_norm, mlp, mx)

            mlp_result = _power_iterate_sigma1(
                mlp_fn, h_post_core, q_idx, hidden_dim, _EPSILON, mx, rng,
            )

            # Baseline MLP forward
            h_out = mlp_fn(h_post_core)
            mx.eval(h_out)

            # Extract query-position vectors
            h_in_q = h_in[0, q_idx, :].astype(mx.float32)
            h_pc_q = h_post_core[0, q_idx, :].astype(mx.float32)
            h_out_q = h_out[0, q_idx, :].astype(mx.float32)
            mx.eval(h_in_q, h_pc_q, h_out_q)

            h_in_np = np.array(h_in_q.tolist(), dtype=np.float32)
            h_pc_np = np.array(h_pc_q.tolist(), dtype=np.float32)
            h_out_np = np.array(h_out_q.tolist(), dtype=np.float32)

            # Norms and norm changes
            h_in_norm_sq = float(np.dot(h_in_np, h_in_np))
            h_pc_norm_sq = float(np.dot(h_pc_np, h_pc_np))
            h_out_norm_sq = float(np.dot(h_out_np, h_out_np))

            delta_core = h_pc_np - h_in_np
            delta_mlp = h_out_np - h_pc_np
            delta_core_norm_sq = float(np.dot(delta_core, delta_core))
            delta_mlp_norm_sq = float(np.dot(delta_mlp, delta_mlp))

            # H_logit_norm
            h_logit_norm = _compute_h_logit_norm(
                h_out_np, final_norm, output_head, embed, mx
            )

            per_layer[i].append({
                "probe_idx": pi,
                "core_type": core_type,
                "sigma1_core": core_result["sigma1"],
                "sigma1_mlp": mlp_result["sigma1"],
                "core_converged": core_result["converged"],
                "mlp_converged": mlp_result["converged"],
                "core_n_iters": core_result["n_iters"],
                "mlp_n_iters": mlp_result["n_iters"],
                "h_in_norm_sq": h_in_norm_sq,
                "h_post_core_norm_sq": h_pc_norm_sq,
                "h_out_norm_sq": h_out_norm_sq,
                "delta_core_norm_sq": delta_core_norm_sq,
                "delta_mlp_norm_sq": delta_mlp_norm_sq,
                "H_logit_norm": h_logit_norm,
            })

            hidden = h_out
            mx.eval(hidden)

    # --- Aggregate per-layer measurements ---
    measurements = []
    total_power_iters = 0
    converged_power_iters = 0

    for i in range(num_layers):
        probe_data = per_layer[i]
        if not probe_data:
            continue

        # Filter to probes with Jacobian data
        jac_probes = [p for p in probe_data if p["sigma1_core"] is not None]
        all_probes = probe_data

        depth_fraction = i / max(num_layers - 1, 1)

        # Core type (should be consistent across probes)
        core_types = [p["core_type"] for p in jac_probes if p["core_type"]]
        core_type = core_types[0] if core_types else None

        # Convergence tracking
        for p in jac_probes:
            total_power_iters += 2  # core + mlp
            if p["core_converged"]:
                converged_power_iters += 1
            if p["mlp_converged"]:
                converged_power_iters += 1

        if jac_probes:
            sigma1_core_vals = [p["sigma1_core"] for p in jac_probes]
            sigma1_mlp_vals = [p["sigma1_mlp"] for p in jac_probes]
            delta_core_vals = [p["delta_core_norm_sq"] for p in jac_probes]
            delta_mlp_vals = [p["delta_mlp_norm_sq"] for p in jac_probes]
            h_in_norms = [p["h_in_norm_sq"] for p in jac_probes]
            h_pc_norms = [p["h_post_core_norm_sq"] for p in jac_probes]
            h_out_norms = [p["h_out_norm_sq"] for p in jac_probes]
        else:
            sigma1_core_vals = []
            sigma1_mlp_vals = []
            delta_core_vals = []
            delta_mlp_vals = []
            h_in_norms = []
            h_pc_norms = []
            h_out_norms = []

        H_vals = [p["H_logit_norm"] for p in all_probes if p["H_logit_norm"] is not None]

        measurements.append({
            "layer_idx": i,
            "depth_fraction": depth_fraction,
            "core_type": core_type,
            "n_probes": len(all_probes),
            "n_jac_probes": len(jac_probes),
            # Per-probe arrays (for layer-level correlation, not probe-level)
            "sigma1_core_mean": float(np.mean(sigma1_core_vals)) if sigma1_core_vals else None,
            "sigma1_mlp_mean": float(np.mean(sigma1_mlp_vals)) if sigma1_mlp_vals else None,
            "sigma1_core_std": float(np.std(sigma1_core_vals)) if sigma1_core_vals else None,
            "sigma1_mlp_std": float(np.std(sigma1_mlp_vals)) if sigma1_mlp_vals else None,
            "delta_core_norm_sq_mean": float(np.mean(delta_core_vals)) if delta_core_vals else None,
            "delta_mlp_norm_sq_mean": float(np.mean(delta_mlp_vals)) if delta_mlp_vals else None,
            "h_in_norm_sq_mean": float(np.mean(h_in_norms)) if h_in_norms else None,
            "h_post_core_norm_sq_mean": float(np.mean(h_pc_norms)) if h_pc_norms else None,
            "h_out_norm_sq_mean": float(np.mean(h_out_norms)) if h_out_norms else None,
            "H_logit_norm_mean": float(np.mean(H_vals)) if H_vals else None,
            # Convergence stats
            "core_converge_frac": (
                float(np.mean([p["core_converged"] for p in jac_probes]))
                if jac_probes else None
            ),
            "mlp_converge_frac": (
                float(np.mean([p["mlp_converged"] for p in jac_probes]))
                if jac_probes else None
            ),
        })

    convergence_rate = (
        converged_power_iters / total_power_iters
        if total_power_iters > 0 else 0.0
    )

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "n_probes": len(probes),
        "measurements": measurements,
        "convergence_summary": {
            "total_power_iters": total_power_iters,
            "converged_power_iters": converged_power_iters,
            "convergence_rate": convergence_rate,
        },
        "constants": {
            "epsilon": _EPSILON,
            "convergence_tol": _CONVERGENCE_TOL,
            "max_power_iters": _MAX_POWER_ITERS,
            "query_pos": _QUERY_POS,
        },
    }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def analyze_model(data: dict, b6_1_norm_sign: str | None) -> dict:
    """Compute depth-controlled correlations and falsifier outcomes.

    b6_1_norm_sign: "positive", "negative", "zero", or None (from B6.1 results).
    """
    model_name = data["model_name"]
    architecture = data["architecture"]
    measurements = data["measurements"]

    # Filter to layers with full Jacobian data
    valid = [m for m in measurements if m["sigma1_core_mean"] is not None
             and m["H_logit_norm_mean"] is not None]

    if len(valid) < 4:
        logger.warning("%s: only %d valid layers, need >= 4", model_name, len(valid))
        return {
            "model_name": model_name,
            "architecture": architecture,
            "error": f"insufficient valid layers: {len(valid)}",
        }

    n_layers = len(valid)

    # Extract arrays
    depth = np.array([m["depth_fraction"] for m in valid])
    sigma1_core = np.array([m["sigma1_core_mean"] for m in valid])
    sigma1_mlp = np.array([m["sigma1_mlp_mean"] for m in valid])
    delta_core = np.array([m["delta_core_norm_sq_mean"] for m in valid])
    delta_mlp = np.array([m["delta_mlp_norm_sq_mean"] for m in valid])
    H = np.array([m["H_logit_norm_mean"] for m in valid])

    # Depth-residualize everything
    H_resid = _residualize_ols(H, depth)
    sigma1_core_resid = _residualize_ols(sigma1_core, depth)
    sigma1_mlp_resid = _residualize_ols(sigma1_mlp, depth)
    delta_core_resid = _residualize_ols(delta_core, depth)
    delta_mlp_resid = _residualize_ols(delta_mlp, depth)

    # Effective df and MDE
    n_eff_H, rho1_H = _effective_df(H_resid)
    mde = _minimum_detectable_effect(n_eff_H, n_controls=1)

    # Spearman correlations + permutation tests
    correlations = {}
    components = {
        "sigma1_core": {
            "label": "σ₁(J_core)",
            "prediction": "F1: sign matches B6.1 norm coupling",
            "resid": sigma1_core_resid,
        },
        "sigma1_mlp": {
            "label": "σ₁(J_mlp)",
            "prediction": "architecture-dependent",
            "resid": sigma1_mlp_resid,
        },
        "delta_core": {
            "label": "Δ||h||²_core",
            "prediction": "F2: sign matches σ₁_core coupling",
            "resid": delta_core_resid,
        },
        "delta_mlp": {
            "label": "Δ||h||²_mlp",
            "prediction": "architecture-dependent",
            "resid": delta_mlp_resid,
        },
    }

    for comp_name, comp_data in components.items():
        r, p = safe_spearman(H_resid, comp_data["resid"])
        perm = _permutation_test(H_resid, comp_data["resid"])
        correlations[comp_name] = {
            "label": comp_data["label"],
            "prediction": comp_data["prediction"],
            "spearman_r": r,
            "spearman_p": p,
            "abs_r": abs(r) if not math.isnan(r) else float("nan"),
            "resolvable": abs(r) > mde if not math.isnan(r) else False,
            "sign": (
                "positive" if r > 0 else ("negative" if r < 0 else "zero")
                if not math.isnan(r) else "nan"
            ),
            "permutation": perm,
        }

    # --- Falsifiers ---
    r_sigma1_core = correlations["sigma1_core"]["spearman_r"]
    r_delta_core = correlations["delta_core"]["spearman_r"]
    sigma1_core_sign = correlations["sigma1_core"]["sign"]

    # F1: Jacobian-norm consistency
    # sign(r(H, σ₁_core | depth)) matches B6.1 norm coupling sign
    if b6_1_norm_sign is not None and not math.isnan(r_sigma1_core):
        f1_pass = sigma1_core_sign == b6_1_norm_sign
    else:
        f1_pass = None

    # F2: Norm change matches Jacobian
    # sign(r(H, Δ_core | depth)) == sign(r(H, σ₁_core | depth))
    delta_core_sign = correlations["delta_core"]["sign"]
    if not (math.isnan(r_sigma1_core) or math.isnan(r_delta_core)):
        f2_pass = sigma1_core_sign == delta_core_sign
    else:
        f2_pass = None

    # F3: Cross-family divergence (evaluated at cross-model level, placeholder here)
    f3_pass = None  # Set in cross-model summary

    # F4: Convergence quality
    conv_rate = data["convergence_summary"]["convergence_rate"]
    f4_pass = conv_rate >= 0.80

    falsifiers = {
        "F1_jacobian_norm_consistency": {
            "prediction": "sign(r(H, σ₁_core | depth)) matches B6.1 norm coupling sign",
            "sigma1_core_sign": sigma1_core_sign,
            "b6_1_norm_sign": b6_1_norm_sign,
            "sigma1_core_r": r_sigma1_core,
            "pass": f1_pass,
        },
        "F2_norm_change_matches_jacobian": {
            "prediction": "sign(r(H, Δ_core | depth)) == sign(r(H, σ₁_core | depth))",
            "sigma1_core_sign": sigma1_core_sign,
            "delta_core_sign": delta_core_sign,
            "pass": f2_pass,
        },
        "F3_cross_family_divergence": {
            "prediction": "LFM2 and Qwen3.5 show different σ₁-entropy coupling",
            "pass": f3_pass,  # evaluated cross-model
        },
        "F4_convergence_quality": {
            "prediction": "≥80% of power iterations converge",
            "convergence_rate": conv_rate,
            "pass": f4_pass,
        },
    }

    # Raw statistics
    raw_stats = {
        "sigma1_core_range": [float(np.min(sigma1_core)), float(np.max(sigma1_core))],
        "sigma1_mlp_range": [float(np.min(sigma1_mlp)), float(np.max(sigma1_mlp))],
        "delta_core_range": [float(np.min(delta_core)), float(np.max(delta_core))],
        "delta_mlp_range": [float(np.min(delta_mlp)), float(np.max(delta_mlp))],
        "H_logit_norm_range": [float(np.min(H)), float(np.max(H))],
    }

    return {
        "model_name": model_name,
        "architecture": architecture,
        "num_layers": n_layers,
        "detection_floor": {
            "n_eff": n_eff_H,
            "rho1_H": rho1_H,
            "mde": mde,
            "n_layers": n_layers,
        },
        "correlations": correlations,
        "falsifiers": falsifiers,
        "convergence_summary": data["convergence_summary"],
        "raw_stats": raw_stats,
        "measurements": data["measurements"],
    }


# ---------------------------------------------------------------------------
# B6.1 cross-reference
# ---------------------------------------------------------------------------


def load_b6_1_norm_sign(model_name: str) -> str | None:
    """Load B6.1 norm coupling sign for cross-reference."""
    json_path = B6_1_BASE / model_name / "norm_corrected_sublayer.json"
    if not json_path.exists():
        logger.warning("No B6.1 results for %s at %s", model_name, json_path)
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data["correlations"]["norm"]["sign"]
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("Failed to read B6.1 norm sign for %s: %s", model_name, e)
        return None


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------


def build_cross_model_summary(results: list[dict]) -> dict:
    """Build cross-model summary with falsifier evaluation."""
    summary_rows = []

    for r in results:
        if "error" in r:
            continue
        row = {
            "model": r["model_name"],
            "architecture": r["architecture"],
            "n_layers": r["num_layers"],
            "mde": r["detection_floor"]["mde"],
            "n_eff": r["detection_floor"]["n_eff"],
            "convergence_rate": r["convergence_summary"]["convergence_rate"],
        }
        for comp in ["sigma1_core", "sigma1_mlp", "delta_core", "delta_mlp"]:
            c = r["correlations"][comp]
            row[f"r_{comp}"] = c["spearman_r"]
            row[f"p_{comp}"] = c["spearman_p"]
            row[f"resolvable_{comp}"] = c["resolvable"]

        for f_name, f_data in r["falsifiers"].items():
            row[f"{f_name}_pass"] = f_data["pass"]

        summary_rows.append(row)

    valid_results = [r for r in results if "error" not in r]

    # F1: Jacobian-norm consistency (per-model)
    f1_results = [
        r["falsifiers"]["F1_jacobian_norm_consistency"]["pass"]
        for r in valid_results
    ]
    f1_testable = [p for p in f1_results if p is not None]
    f1_all_pass = all(f1_testable) if f1_testable else False
    f1_failing = [
        r["model_name"] for r in valid_results
        if r["falsifiers"]["F1_jacobian_norm_consistency"]["pass"] is False
    ]

    # F2: Norm change matches Jacobian (per-model)
    f2_results = [
        r["falsifiers"]["F2_norm_change_matches_jacobian"]["pass"]
        for r in valid_results
    ]
    f2_testable = [p for p in f2_results if p is not None]
    f2_all_pass = all(f2_testable) if f2_testable else False

    # F3: Cross-family divergence
    # LFM2 and Qwen3.5 should show different σ₁-entropy coupling patterns
    lfm2_results = [
        r for r in valid_results if r["architecture"] == "lfm2"
    ]
    qwen35_results = [
        r for r in valid_results if r["architecture"] == "qwen3.5"
    ]

    f3_pass = None
    if lfm2_results and qwen35_results:
        lfm2_signs = [
            r["correlations"]["sigma1_core"]["sign"] for r in lfm2_results
        ]
        qwen35_signs = [
            r["correlations"]["sigma1_core"]["sign"] for r in qwen35_results
        ]
        # Different if the sign sets don't fully overlap
        f3_pass = set(lfm2_signs) != set(qwen35_signs)

    # Update F3 in each result
    for r in valid_results:
        r["falsifiers"]["F3_cross_family_divergence"]["pass"] = f3_pass

    # F4: Convergence quality (per-model)
    f4_results = [
        r["falsifiers"]["F4_convergence_quality"]["pass"]
        for r in valid_results
    ]
    f4_testable = [p for p in f4_results if p is not None]
    f4_all_pass = all(f4_testable) if f4_testable else False

    falsifier_summary = {
        "F1_jacobian_norm_consistency": {
            "pass": f1_all_pass,
            "pass_count": sum(1 for p in f1_testable if p),
            "total": len(f1_testable),
            "failing_models": f1_failing,
            "verdict": (
                "PASS — Jacobian spectral structure matches B6.1 norm coupling"
                if f1_all_pass
                else f"FAIL — {len(f1_failing)} model(s) mismatch"
            ),
        },
        "F2_norm_change_matches_jacobian": {
            "pass": f2_all_pass,
            "pass_count": sum(1 for p in f2_testable if p),
            "total": len(f2_testable),
            "verdict": (
                "PASS — σ₁ translates to actual norm change"
                if f2_all_pass
                else "FAIL — σ₁ doesn't translate to actual norm change"
            ),
        },
        "F3_cross_family_divergence": {
            "pass": f3_pass,
            "lfm2_signs": [
                r["correlations"]["sigma1_core"]["sign"] for r in lfm2_results
            ] if lfm2_results else [],
            "qwen35_signs": [
                r["correlations"]["sigma1_core"]["sign"] for r in qwen35_results
            ] if qwen35_results else [],
            "verdict": (
                "PASS — architectures show different σ₁-entropy coupling"
                if f3_pass
                else (
                    "FAIL — architectures show same σ₁-entropy coupling"
                    if f3_pass is False
                    else "UNTESTABLE — need both architectures"
                )
            ),
        },
        "F4_convergence_quality": {
            "pass": f4_all_pass,
            "pass_count": sum(1 for p in f4_testable if p),
            "total": len(f4_testable),
            "rates": {
                r["model_name"]: r["convergence_summary"]["convergence_rate"]
                for r in valid_results
            },
            "verdict": (
                "PASS — ≥80% convergence for all models"
                if f4_all_pass
                else "FAIL — some models below 80% convergence"
            ),
        },
    }

    return {
        "n_models": len(valid_results),
        "models": [r["model_name"] for r in valid_results],
        "summary_rows": summary_rows,
        "falsifier_summary": falsifier_summary,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], cross_summary: dict) -> None:
    """Print human-readable summary table."""
    logger.info("\n%s", "=" * 95)
    logger.info("B7: JACOBIAN SPECTRAL PROBE — NORM-ENTROPY COUPLING MECHANISM")
    logger.info("%s", "=" * 95)

    valid_results = [r for r in results if "error" not in r]

    header = (
        f"{'Model':<18} {'L':>3} {'MDE':>5} {'Conv%':>5} "
        f"{'r(σ₁_core)':>10} {'r(σ₁_mlp)':>10} "
        f"{'r(Δ_core)':>10} {'r(Δ_mlp)':>10} "
        f"{'F1':>4} {'F2':>4}"
    )
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for r in valid_results:
        c = r["correlations"]
        conv_rate = r["convergence_summary"]["convergence_rate"]

        def _fmt_r(comp_name):
            val = c[comp_name]["spearman_r"]
            resolvable = c[comp_name]["resolvable"]
            if math.isnan(val):
                return "      nan"
            marker = "*" if resolvable else " "
            return f"{val:+.4f}{marker}"

        f1 = r["falsifiers"]["F1_jacobian_norm_consistency"]["pass"]
        f2 = r["falsifiers"]["F2_norm_change_matches_jacobian"]["pass"]
        f1_str = "PASS" if f1 else ("FAIL" if f1 is False else "  - ")
        f2_str = "PASS" if f2 else ("FAIL" if f2 is False else "  - ")

        logger.info(
            "%-18s %3d %5.3f %4.0f%% %10s %10s %10s %10s %4s %4s",
            r["model_name"],
            r["num_layers"],
            r["detection_floor"]["mde"],
            conv_rate * 100,
            _fmt_r("sigma1_core"),
            _fmt_r("sigma1_mlp"),
            _fmt_r("delta_core"),
            _fmt_r("delta_mlp"),
            f1_str,
            f2_str,
        )

    logger.info("\n* = resolvable (|r| > MDE)")

    # σ₁ summary per model
    logger.info("\n%s", "-" * 60)
    logger.info("JACOBIAN SPECTRAL SUMMARY:")
    logger.info("%s", "-" * 60)
    for r in valid_results:
        m = r["measurements"]
        valid_m = [l for l in m if l["sigma1_core_mean"] is not None]
        if valid_m:
            core_vals = [l["sigma1_core_mean"] for l in valid_m]
            mlp_vals = [l["sigma1_mlp_mean"] for l in valid_m]
            logger.info(
                "  %s: σ₁_core=[%.3f, %.3f] (mean=%.3f), σ₁_mlp=[%.3f, %.3f] (mean=%.3f)",
                r["model_name"],
                min(core_vals), max(core_vals), np.mean(core_vals),
                min(mlp_vals), max(mlp_vals), np.mean(mlp_vals),
            )

    # Falsifier summary
    logger.info("\n%s", "-" * 60)
    logger.info("FALSIFIER OUTCOMES:")
    logger.info("%s", "-" * 60)
    for f_name, f_data in cross_summary["falsifier_summary"].items():
        if "pass_count" in f_data:
            logger.info(
                "  %s: %d/%d pass — %s",
                f_name, f_data["pass_count"], f_data["total"], f_data["verdict"],
            )
        else:
            logger.info("  %s: %s", f_name, f_data["verdict"])

    # B6.1 cross-reference
    logger.info("\n%s", "-" * 60)
    logger.info("B6.1 CROSS-REFERENCE:")
    logger.info("%s", "-" * 60)
    for r in valid_results:
        f1 = r["falsifiers"]["F1_jacobian_norm_consistency"]
        logger.info(
            "  %s: B6.1 norm sign=%s, B7 σ₁_core sign=%s, r=%.4f — %s",
            r["model_name"],
            f1["b6_1_norm_sign"],
            f1["sigma1_core_sign"],
            f1["sigma1_core_r"],
            "MATCH" if f1["pass"] else ("MISMATCH" if f1["pass"] is False else "N/A"),
        )

    # Permutation test details
    logger.info("\n%s", "-" * 60)
    logger.info("PERMUTATION TEST DETAILS (sigma1_core — F1 critical):")
    logger.info("%s", "-" * 60)
    for r in valid_results:
        perm = r["correlations"]["sigma1_core"]["permutation"]
        logger.info(
            "  %s: |r|=%.4f, exceedance=%.3f, null_max=%.4f",
            r["model_name"],
            perm["observed_abs_r"],
            perm["exceedance_fraction"],
            perm["null_max"],
        )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(args):
    """Run the full B7 Jacobian spectral probe experiment."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    if args.smoke:
        probes = PROBES[:3]
        logger.info("SMOKE TEST: using %d probes", len(probes))
    else:
        probes = PROBES

    if args.models:
        model_names = args.models
    else:
        model_names = list(MODEL_REGISTRY.keys())

    output_base = Path(args.output)

    all_results = []

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model: %s (skipping)", model_name)
            continue

        model_info = MODEL_REGISTRY[model_name]
        model_path = model_info["path"]

        logger.info("\n%s", "=" * 60)
        logger.info("Processing: %s (%s)", model_name, model_path)
        logger.info("%s", "=" * 60)

        if not Path(model_path).exists():
            logger.error("Model path not found: %s", model_path)
            continue

        t0 = time.time()

        # Load model
        model, tokenizer = backend.load_model(model_path)

        # Collect Jacobian data
        data = collect_jacobian_data(
            model, tokenizer, model_name, model_info, probes, backend,
        )

        # Load B6.1 cross-reference
        b6_1_norm_sign = load_b6_1_norm_sign(model_name)
        logger.info("B6.1 norm coupling sign for %s: %s", model_name, b6_1_norm_sign)

        # Analyze
        result = analyze_model(data, b6_1_norm_sign)
        all_results.append(result)

        elapsed = time.time() - t0
        logger.info("  %s completed in %.1fs", model_name, elapsed)

        # Free memory
        del model, tokenizer, data
        gc.collect()

    if not all_results:
        logger.error("No models processed successfully.")
        return

    # Cross-model summary
    cross_summary = build_cross_model_summary(all_results)

    # Print summary
    print_summary(all_results, cross_summary)

    # Save results
    output_base.mkdir(parents=True, exist_ok=True)
    for r in all_results:
        if "error" in r:
            continue
        model_dir = output_base / r["model_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "jacobian_spectral.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        logger.info("Saved: %s", model_dir / "jacobian_spectral.json")

    with open(output_base / "cross_model_summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2, default=str)
    logger.info("Saved: %s", output_base / "cross_model_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="B7: Jacobian Spectral Probe — Norm-Entropy Coupling Mechanism"
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_BASE),
        help="Output directory for results",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to analyze (default: all in registry)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: use only 3 probes",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
