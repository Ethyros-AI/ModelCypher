#!/usr/bin/env python3
"""B7.1: Sublayer Alignment Probe — Why Norm-Entropy Coupling Varies.

B7 (Jacobian spectral probe) FAILED on critical falsifier F1 (0/2 models).
Tracing the math:

    Actual norm change: Δ||h||² = 2⟨h_in, op(h_in)⟩ + ||op(h_in)||²

The ALIGNMENT term ⟨h_in, δ⟩ drives norm change, not the Jacobian spectral
radius σ₁. For LFM2-350M: r(H, alignment | depth) = -0.82 (p=0.0001), while
r(H, σ₁ | depth) = +0.07 (p=0.80).

But per-layer alignment doesn't predict B6.1's norm LEVEL coupling either,
because norm level is CUMULATIVE (sum of all prior changes). The correlation
between this cumulative quantity and per-layer entropy depends on how entropy
and norm-change DEPTH PROFILES interact — architecture-specific.

This script measures:
    1. Per-layer sublayer alignment: ⟨h_in, δ_core⟩, ⟨h_post_core, δ_mlp⟩
    2. Directional decomposition: magnitude × cosine
    3. Cumulative alignment profile: running sum ≈ ||h_l||² - ||h_0||²
    4. Operator-type stratification (attention vs conv vs linear_attn)

All tensor operations via backend protocol (backend.dot, backend.norm, etc).
Numpy used ONLY for scipy.stats, OLS residualization, and permutation RNG.

Constants (IEEE 754 derived):
    eps_bf16 = 2^-7
    TINY = eps_bf16² ≈ 6.1e-5 (division floor)
    Query position = -2 (same as B7 / estimate_bl_jacobian.py)

Falsifiers:
    F1: Alignment explains norm change — sign(r(H, alignment_total | depth))
        consistent with sign(r(H, Δ||h||² | depth)) per model
    F2: Cumulative alignment matches B6.1 — sign(r(H, cum_alignment | depth))
        == B6.1 norm coupling sign per model [CRITICAL TEST]
    F3: Direction or magnitude resolvable — at least one of {dir_align, δ_norm}
        has |r| > MDE per model
    F4: Operator-type differential coupling — mixed-operator models show
        different alignment patterns by type
    F5: Cross-family divergence — LFM2 and Qwen3.5 show different cumulative
        alignment coupling

Usage:
    poetry run python scripts/entropy_curvature_alignment_profile.py --smoke
    poetry run python scripts/entropy_curvature_alignment_profile.py
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

# Division floor: eps_bf16² ≈ 6.1e-5
_TINY = _EPS_BF16 ** 2

# Query position (second-to-last token, same as B7 / estimate_bl_jacobian.py)
_QUERY_POS = -2

# Permutation count (matches convention)
N_PERMUTATIONS = 500

# B6.1 results base for cross-reference
B6_1_BASE = Path("results/entropy_curvature_norm_corrected_sublayer")

OUTPUT_BASE = Path("results/entropy_curvature_alignment_profile")


# ---------------------------------------------------------------------------
# Model registry (all 10 models with B6.1 data)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024,
        "architecture": "lfm2",
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16, "d": 1280,
        "architecture": "lfm2",
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "L": 36, "d": 4096,
        "architecture": "qwen3",
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24, "d": 2048,
        "architecture": "qwen3.5",
    },
    "Qwen3.5-4B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-bf16",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
    },
    "Qwen3.5-4B-4bit": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-4bit-g64",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
    },
    "Mistral-7B": {
        "path": f"{MODELS_BASE}/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "L": 32, "d": 4096,
        "architecture": "mistral",
    },
}

# Same 30 probes as operator_split / B7 for cross-validation
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
# Statistical utilities (numpy only — no backend equivalent for scipy/OLS)
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
# Model infrastructure (from B7 / operator_split)
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


# ---------------------------------------------------------------------------
# Sublayer decomposition helpers (from B7)
# ---------------------------------------------------------------------------


def _resolve_sublayer_components(layer):
    """Extract sublayer components from a layer.

    Returns (input_norm, core_module, core_type, post_attn_norm, mlp).
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


# ---------------------------------------------------------------------------
# H_logit_norm computation (from B7 / operator_split)
# ---------------------------------------------------------------------------


def _entropy_from_logits(logits, backend):
    """Stable Shannon entropy from logits tensor via backend ops."""
    logits_f = backend.astype(logits, "float32")
    logits_shifted = logits_f - backend.max(logits_f)
    probs = backend.softmax(logits_shifted, axis=-1)
    log_probs = backend.log(probs + 1e-12)
    entropy = -backend.sum(probs * log_probs)
    backend.eval(entropy)
    return float(backend.to_scalar(entropy))


def _compute_h_logit_norm(h_out_q, final_norm, output_head, embed_tokens, backend):
    """Compute H_logit_norm from query-position hidden state (backend array).

    h_out_q: 1D backend array [d] (already float32)
    """
    import mlx.core as mx

    h_mx = mx.array(backend.tolist(h_out_q)).reshape(1, 1, -1)
    if final_norm is not None:
        h_mx = final_norm(h_mx)

    if output_head is not None:
        logits = output_head(h_mx)
    else:
        logits = embed_tokens.as_linear(h_mx)
    if isinstance(logits, tuple):
        logits = logits[0]

    return _entropy_from_logits(logits[0, 0, :], backend)


# ---------------------------------------------------------------------------
# Per-layer alignment measurement (ALL ops through backend)
# ---------------------------------------------------------------------------


def _measure_layer_alignment(h_in_q, h_pc_q, h_out_q, backend) -> dict:
    """Measure sublayer alignment terms using backend protocol.

    h_in_q:  1D backend array [d] — layer input at query position (float32)
    h_pc_q:  1D backend array [d] — post-core at query position (float32)
    h_out_q: 1D backend array [d] — layer output at query position (float32)

    Returns dict of Python floats (all scalars extracted after eval).
    """
    # Sublayer deltas
    delta_core = h_pc_q - h_in_q
    delta_mlp = h_out_q - h_pc_q

    # Alignment inner products: ⟨h_in, δ_core⟩ and ⟨h_post_core, δ_mlp⟩
    align_core = backend.dot(h_in_q, delta_core)
    align_mlp = backend.dot(h_pc_q, delta_mlp)

    # Norms for directional decomposition
    h_in_norm = backend.norm(h_in_q)
    h_pc_norm = backend.norm(h_pc_q)
    delta_core_norm = backend.norm(delta_core)
    delta_mlp_norm = backend.norm(delta_mlp)
    h_out_norm = backend.norm(h_out_q)

    # Evaluate the lazy graph
    backend.eval(
        align_core, align_mlp,
        h_in_norm, h_pc_norm, delta_core_norm, delta_mlp_norm, h_out_norm,
    )

    # Extract scalars
    align_core_f = float(backend.to_scalar(align_core))
    align_mlp_f = float(backend.to_scalar(align_mlp))
    h_in_norm_f = float(backend.to_scalar(h_in_norm))
    h_pc_norm_f = float(backend.to_scalar(h_pc_norm))
    delta_core_norm_f = float(backend.to_scalar(delta_core_norm))
    delta_mlp_norm_f = float(backend.to_scalar(delta_mlp_norm))
    h_out_norm_f = float(backend.to_scalar(h_out_norm))

    # Directional alignment (cosine): align / (||h|| × ||δ|| + TINY)
    denom_core = h_in_norm_f * delta_core_norm_f + _TINY
    denom_mlp = h_pc_norm_f * delta_mlp_norm_f + _TINY
    dir_align_core = align_core_f / denom_core
    dir_align_mlp = align_mlp_f / denom_mlp

    # Total alignment: alignment_core + alignment_mlp
    # (this is the leading term in Δ||h||² = 2⟨h_in,δ_core⟩ + ||δ_core||² +
    #  2⟨h_pc,δ_mlp⟩ + ||δ_mlp||² + 2⟨δ_core,δ_mlp⟩)
    # More precisely, Δ||h||² = ||h_out||² - ||h_in||²
    # = (2·align_core + ||δ_core||²) + (2·align_mlp + ||δ_mlp||²) + 2⟨δ_core,δ_mlp⟩
    # We track alignment_total = align_core + align_mlp as the dominant term
    align_total = align_core_f + align_mlp_f

    # Norm squared values
    h_in_norm_sq = h_in_norm_f ** 2
    h_pc_norm_sq = h_pc_norm_f ** 2
    h_out_norm_sq = h_out_norm_f ** 2

    # Actual norm change (for F1 validation)
    delta_norm_sq = h_out_norm_sq - h_in_norm_sq

    return {
        "alignment_core": align_core_f,
        "alignment_mlp": align_mlp_f,
        "alignment_total": align_total,
        "dir_align_core": dir_align_core,
        "dir_align_mlp": dir_align_mlp,
        "delta_core_norm": delta_core_norm_f,
        "delta_mlp_norm": delta_mlp_norm_f,
        "h_in_norm_sq": h_in_norm_sq,
        "h_pc_norm_sq": h_pc_norm_sq,
        "h_out_norm_sq": h_out_norm_sq,
        "delta_norm_sq": delta_norm_sq,
    }


# ---------------------------------------------------------------------------
# Per-model data collection
# ---------------------------------------------------------------------------


def collect_alignment_data(
    model, tokenizer, model_name: str, model_info: dict,
    probes: list[str], backend,
) -> dict:
    """Collect per-layer alignment data for all probes.

    All tensor ops through backend. No numpy for geometry.
    """
    import mlx.core as mx

    num_layers = model_info["L"]

    base = _resolve_backbone(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    final_norm = _resolve_final_norm(base)
    output_head = _resolve_output_head(model, base)

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

                # Extract query-position vectors via backend
                h_in_q = backend.astype(h_in[0, q_idx, :], "float32")
                h_out_q = backend.astype(h_out[0, q_idx, :], "float32")
                backend.eval(h_in_q, h_out_q)

                h_in_norm_sq = float(backend.to_scalar(backend.dot(h_in_q, h_in_q)))
                h_out_norm_sq = float(backend.to_scalar(backend.dot(h_out_q, h_out_q)))

                per_layer[i].append({
                    "probe_idx": pi,
                    "core_type": None,
                    "alignment_core": None,
                    "alignment_mlp": None,
                    "alignment_total": None,
                    "dir_align_core": None,
                    "dir_align_mlp": None,
                    "delta_core_norm": None,
                    "delta_mlp_norm": None,
                    "h_in_norm_sq": h_in_norm_sq,
                    "h_pc_norm_sq": None,
                    "h_out_norm_sq": h_out_norm_sq,
                    "delta_norm_sq": h_out_norm_sq - h_in_norm_sq,
                    "H_logit_norm": _compute_h_logit_norm(
                        h_out_q, final_norm, output_head, embed, backend,
                    ),
                })
                hidden = h_out
                mx.eval(hidden)
                continue

            # --- Sublayer decomposition ---
            # Core forward
            normed = input_norm(h_in)
            core_out = _call_with_fallback(core_module, normed, mask=layer_mask) if core_module is not None else h_in * 0
            if isinstance(core_out, tuple):
                core_out = core_out[0]
            if core_type == "identity":
                h_post_core = h_in
            else:
                h_post_core = h_in + core_out

            # MLP forward
            normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
            mlp_out = mlp(normed2)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            h_out = h_post_core + mlp_out

            # Extract query-position vectors (backend, float32)
            h_in_q = backend.astype(h_in[0, q_idx, :], "float32")
            h_pc_q = backend.astype(h_post_core[0, q_idx, :], "float32")
            h_out_q = backend.astype(h_out[0, q_idx, :], "float32")
            backend.eval(h_in_q, h_pc_q, h_out_q)

            # Measure alignment via backend
            alignment = _measure_layer_alignment(h_in_q, h_pc_q, h_out_q, backend)

            # H_logit_norm
            h_logit_norm = _compute_h_logit_norm(
                h_out_q, final_norm, output_head, embed, backend,
            )

            per_layer[i].append({
                "probe_idx": pi,
                "core_type": core_type,
                **alignment,
                "H_logit_norm": h_logit_norm,
            })

            hidden = h_out
            mx.eval(hidden)

    # --- Aggregate per-layer measurements ---
    measurements = []
    for i in range(num_layers):
        probe_data = per_layer[i]
        if not probe_data:
            continue

        decomp_probes = [p for p in probe_data if p["alignment_core"] is not None]
        all_probes = probe_data

        depth_fraction = i / max(num_layers - 1, 1)

        # Core type (should be consistent across probes for a given layer)
        core_types = [p["core_type"] for p in decomp_probes if p["core_type"]]
        core_type = core_types[0] if core_types else None

        def _safe_mean(vals):
            finite = [v for v in vals if v is not None and math.isfinite(v)]
            return float(np.mean(finite)) if finite else None

        H_vals = [p["H_logit_norm"] for p in all_probes if p["H_logit_norm"] is not None]

        m = {
            "layer_idx": i,
            "depth_fraction": depth_fraction,
            "core_type": core_type,
            "n_probes": len(all_probes),
            "n_decomp_probes": len(decomp_probes),
        }

        if decomp_probes:
            m["alignment_core_mean"] = _safe_mean([p["alignment_core"] for p in decomp_probes])
            m["alignment_mlp_mean"] = _safe_mean([p["alignment_mlp"] for p in decomp_probes])
            m["alignment_total_mean"] = _safe_mean([p["alignment_total"] for p in decomp_probes])
            m["dir_align_core_mean"] = _safe_mean([p["dir_align_core"] for p in decomp_probes])
            m["dir_align_mlp_mean"] = _safe_mean([p["dir_align_mlp"] for p in decomp_probes])
            m["delta_core_norm_mean"] = _safe_mean([p["delta_core_norm"] for p in decomp_probes])
            m["delta_mlp_norm_mean"] = _safe_mean([p["delta_mlp_norm"] for p in decomp_probes])
            m["h_in_norm_sq_mean"] = _safe_mean([p["h_in_norm_sq"] for p in decomp_probes])
            m["h_pc_norm_sq_mean"] = _safe_mean([p["h_pc_norm_sq"] for p in decomp_probes])
            m["h_out_norm_sq_mean"] = _safe_mean([p["h_out_norm_sq"] for p in decomp_probes])
            m["delta_norm_sq_mean"] = _safe_mean([p["delta_norm_sq"] for p in decomp_probes])
        else:
            for key in [
                "alignment_core_mean", "alignment_mlp_mean", "alignment_total_mean",
                "dir_align_core_mean", "dir_align_mlp_mean",
                "delta_core_norm_mean", "delta_mlp_norm_mean",
                "h_in_norm_sq_mean", "h_pc_norm_sq_mean", "h_out_norm_sq_mean",
                "delta_norm_sq_mean",
            ]:
                m[key] = None

        m["H_logit_norm_mean"] = _safe_mean(H_vals)

        measurements.append(m)

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "hidden_dim": model_info["d"],
        "n_probes": len(probes),
        "measurements": measurements,
        "constants": {
            "eps_bf16": _EPS_BF16,
            "tiny": _TINY,
            "query_pos": _QUERY_POS,
        },
    }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def analyze_model(data: dict, b6_1_norm_sign: str | None) -> dict:
    """Compute depth-controlled correlations and falsifier outcomes."""
    model_name = data["model_name"]
    architecture = data["architecture"]
    measurements = data["measurements"]

    # Filter to layers with full decomposition data
    valid = [
        m for m in measurements
        if m["alignment_total_mean"] is not None and m["H_logit_norm_mean"] is not None
    ]

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
    align_core = np.array([m["alignment_core_mean"] for m in valid])
    align_mlp = np.array([m["alignment_mlp_mean"] for m in valid])
    align_total = np.array([m["alignment_total_mean"] for m in valid])
    dir_align_core = np.array([m["dir_align_core_mean"] for m in valid])
    dir_align_mlp = np.array([m["dir_align_mlp_mean"] for m in valid])
    delta_core_norm = np.array([m["delta_core_norm_mean"] for m in valid])
    delta_mlp_norm = np.array([m["delta_mlp_norm_mean"] for m in valid])
    delta_norm_sq = np.array([m["delta_norm_sq_mean"] for m in valid])
    H = np.array([m["H_logit_norm_mean"] for m in valid])

    # Cumulative alignment profile: running sum of total alignment
    # This approximates ||h_l||² - ||h_0||² (the norm level)
    cum_alignment = np.cumsum(align_total)

    # Depth-residualize everything
    H_resid = _residualize_ols(H, depth)
    align_core_resid = _residualize_ols(align_core, depth)
    align_mlp_resid = _residualize_ols(align_mlp, depth)
    align_total_resid = _residualize_ols(align_total, depth)
    dir_align_core_resid = _residualize_ols(dir_align_core, depth)
    dir_align_mlp_resid = _residualize_ols(dir_align_mlp, depth)
    delta_core_norm_resid = _residualize_ols(delta_core_norm, depth)
    delta_mlp_norm_resid = _residualize_ols(delta_mlp_norm, depth)
    delta_norm_sq_resid = _residualize_ols(delta_norm_sq, depth)
    cum_alignment_resid = _residualize_ols(cum_alignment, depth)

    # Effective df and MDE
    n_eff_H, rho1_H = _effective_df(H_resid)
    mde = _minimum_detectable_effect(n_eff_H, n_controls=1)

    # Spearman correlations + permutation tests
    components = {
        "alignment_core": {
            "label": "⟨h_in, δ_core⟩",
            "resid": align_core_resid,
        },
        "alignment_mlp": {
            "label": "⟨h_pc, δ_mlp⟩",
            "resid": align_mlp_resid,
        },
        "alignment_total": {
            "label": "align_core + align_mlp",
            "resid": align_total_resid,
        },
        "dir_align_core": {
            "label": "cos(θ_core) = ⟨h_in, δ_core⟩ / (||h_in|| ||δ_core||)",
            "resid": dir_align_core_resid,
        },
        "dir_align_mlp": {
            "label": "cos(θ_mlp) = ⟨h_pc, δ_mlp⟩ / (||h_pc|| ||δ_mlp||)",
            "resid": dir_align_mlp_resid,
        },
        "delta_core_norm": {
            "label": "||δ_core||",
            "resid": delta_core_norm_resid,
        },
        "delta_mlp_norm": {
            "label": "||δ_mlp||",
            "resid": delta_mlp_norm_resid,
        },
        "delta_norm_sq": {
            "label": "Δ||h||² = ||h_out||² - ||h_in||²",
            "resid": delta_norm_sq_resid,
        },
        "cumulative_alignment": {
            "label": "Σ(align_total) ≈ ||h_l||² - ||h_0||²",
            "resid": cum_alignment_resid,
        },
    }

    correlations = {}
    for comp_name, comp_data in components.items():
        r, p = safe_spearman(H_resid, comp_data["resid"])
        perm = _permutation_test(H_resid, comp_data["resid"])
        correlations[comp_name] = {
            "label": comp_data["label"],
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

    # --- Operator-type stratification ---
    operator_types = set(m["core_type"] for m in valid if m["core_type"])
    operator_correlations = {}
    if len(operator_types) > 1:
        for op_type in sorted(operator_types):
            op_layers = [m for m in valid if m["core_type"] == op_type]
            if len(op_layers) < 4:
                operator_correlations[op_type] = {"n_layers": len(op_layers), "insufficient": True}
                continue
            op_depth = np.array([m["depth_fraction"] for m in op_layers])
            op_align = np.array([m["alignment_total_mean"] for m in op_layers])
            op_H = np.array([m["H_logit_norm_mean"] for m in op_layers])
            op_H_resid = _residualize_ols(op_H, op_depth)
            op_align_resid = _residualize_ols(op_align, op_depth)
            r, p = safe_spearman(op_H_resid, op_align_resid)
            operator_correlations[op_type] = {
                "n_layers": len(op_layers),
                "spearman_r": r,
                "spearman_p": p,
                "sign": (
                    "positive" if r > 0 else ("negative" if r < 0 else "zero")
                    if not math.isnan(r) else "nan"
                ),
            }

    # --- Falsifiers ---
    r_align_total = correlations["alignment_total"]["spearman_r"]
    r_delta_norm_sq = correlations["delta_norm_sq"]["spearman_r"]
    r_cum_alignment = correlations["cumulative_alignment"]["spearman_r"]
    cum_alignment_sign = correlations["cumulative_alignment"]["sign"]

    # F1: Alignment explains norm change
    # sign(r(H, alignment_total | depth)) consistent with sign(r(H, Δ||h||² | depth))
    if not (math.isnan(r_align_total) or math.isnan(r_delta_norm_sq)):
        align_sign = "positive" if r_align_total > 0 else "negative"
        delta_sign = "positive" if r_delta_norm_sq > 0 else "negative"
        f1_pass = align_sign == delta_sign
    else:
        f1_pass = None

    # F2: Cumulative alignment matches B6.1 [CRITICAL]
    if b6_1_norm_sign is not None and not math.isnan(r_cum_alignment):
        f2_pass = cum_alignment_sign == b6_1_norm_sign
    else:
        f2_pass = None

    # F3: Direction or magnitude resolvable
    # At least one of {dir_align_core, dir_align_mlp, delta_core_norm, delta_mlp_norm}
    # has |r| > MDE
    resolvable_components = [
        correlations["dir_align_core"]["resolvable"],
        correlations["dir_align_mlp"]["resolvable"],
        correlations["delta_core_norm"]["resolvable"],
        correlations["delta_mlp_norm"]["resolvable"],
    ]
    f3_pass = any(resolvable_components) if any(r is not None for r in resolvable_components) else None

    # F4: Operator-type differential coupling
    # Mixed-operator models show different alignment patterns by type
    if len(operator_types) > 1 and len(operator_correlations) > 1:
        op_signs = {
            op: data.get("sign")
            for op, data in operator_correlations.items()
            if not data.get("insufficient", False)
        }
        if len(op_signs) >= 2:
            f4_pass = len(set(op_signs.values())) > 1
        else:
            f4_pass = None
    else:
        f4_pass = None

    # F5: Cross-family divergence (evaluated at cross-model level)
    f5_pass = None

    falsifiers = {
        "F1_alignment_explains_norm_change": {
            "prediction": "sign(r(H, align_total | depth)) == sign(r(H, Δ||h||² | depth))",
            "r_alignment_total": r_align_total,
            "r_delta_norm_sq": r_delta_norm_sq,
            "pass": f1_pass,
        },
        "F2_cumulative_matches_b6_1": {
            "prediction": "sign(r(H, cum_alignment | depth)) == B6.1 norm sign",
            "cum_alignment_sign": cum_alignment_sign,
            "b6_1_norm_sign": b6_1_norm_sign,
            "r_cumulative_alignment": r_cum_alignment,
            "pass": f2_pass,
        },
        "F3_direction_or_magnitude_resolvable": {
            "prediction": "at least one of {dir_align, δ_norm} has |r| > MDE",
            "resolvable_dir_core": correlations["dir_align_core"]["resolvable"],
            "resolvable_dir_mlp": correlations["dir_align_mlp"]["resolvable"],
            "resolvable_delta_core_norm": correlations["delta_core_norm"]["resolvable"],
            "resolvable_delta_mlp_norm": correlations["delta_mlp_norm"]["resolvable"],
            "pass": f3_pass,
        },
        "F4_operator_type_differential": {
            "prediction": "mixed-operator models show different alignment by type",
            "operator_types": sorted(operator_types) if operator_types else [],
            "operator_correlations": operator_correlations,
            "pass": f4_pass,
        },
        "F5_cross_family_divergence": {
            "prediction": "LFM2 and Qwen3.5 show different cum_alignment coupling",
            "pass": f5_pass,
        },
    }

    # Raw statistics
    raw_stats = {
        "alignment_core_range": [float(np.min(align_core)), float(np.max(align_core))],
        "alignment_mlp_range": [float(np.min(align_mlp)), float(np.max(align_mlp))],
        "alignment_total_range": [float(np.min(align_total)), float(np.max(align_total))],
        "cum_alignment_range": [float(np.min(cum_alignment)), float(np.max(cum_alignment))],
        "dir_align_core_range": [float(np.min(dir_align_core)), float(np.max(dir_align_core))],
        "delta_core_norm_range": [float(np.min(delta_core_norm)), float(np.max(delta_core_norm))],
        "H_logit_norm_range": [float(np.min(H)), float(np.max(H))],
    }

    # Alignment depth profiles (for manual inspection)
    alignment_profiles = {
        "depth": [float(d) for d in depth],
        "alignment_core": [float(v) for v in align_core],
        "alignment_mlp": [float(v) for v in align_mlp],
        "alignment_total": [float(v) for v in align_total],
        "cumulative_alignment": [float(v) for v in cum_alignment],
        "H_logit_norm": [float(v) for v in H],
        "core_types": [m["core_type"] for m in valid],
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
        "operator_correlations": operator_correlations,
        "falsifiers": falsifiers,
        "raw_stats": raw_stats,
        "alignment_profiles": alignment_profiles,
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
    valid_results = [r for r in results if "error" not in r]
    summary_rows = []

    for r in valid_results:
        row = {
            "model": r["model_name"],
            "architecture": r["architecture"],
            "n_layers": r["num_layers"],
            "mde": r["detection_floor"]["mde"],
            "n_eff": r["detection_floor"]["n_eff"],
        }
        for comp in [
            "alignment_core", "alignment_mlp", "alignment_total",
            "dir_align_core", "delta_core_norm",
            "delta_norm_sq", "cumulative_alignment",
        ]:
            c = r["correlations"][comp]
            row[f"r_{comp}"] = c["spearman_r"]
            row[f"p_{comp}"] = c["spearman_p"]
            row[f"resolvable_{comp}"] = c["resolvable"]

        for f_name, f_data in r["falsifiers"].items():
            row[f"{f_name}_pass"] = f_data["pass"]

        summary_rows.append(row)

    # F1: Alignment explains norm change (per-model)
    f1_results = [
        r["falsifiers"]["F1_alignment_explains_norm_change"]["pass"]
        for r in valid_results
    ]
    f1_testable = [p for p in f1_results if p is not None]
    f1_all_pass = all(f1_testable) if f1_testable else False
    f1_failing = [
        r["model_name"] for r in valid_results
        if r["falsifiers"]["F1_alignment_explains_norm_change"]["pass"] is False
    ]

    # F2: Cumulative alignment matches B6.1 [CRITICAL]
    f2_results = [
        r["falsifiers"]["F2_cumulative_matches_b6_1"]["pass"]
        for r in valid_results
    ]
    f2_testable = [p for p in f2_results if p is not None]
    f2_all_pass = all(f2_testable) if f2_testable else False
    f2_failing = [
        r["model_name"] for r in valid_results
        if r["falsifiers"]["F2_cumulative_matches_b6_1"]["pass"] is False
    ]

    # F3: Direction or magnitude resolvable (per-model)
    f3_results = [
        r["falsifiers"]["F3_direction_or_magnitude_resolvable"]["pass"]
        for r in valid_results
    ]
    f3_testable = [p for p in f3_results if p is not None]
    f3_all_pass = all(f3_testable) if f3_testable else False

    # F4: Operator-type differential (per-model)
    f4_results = [
        r["falsifiers"]["F4_operator_type_differential"]["pass"]
        for r in valid_results
    ]
    f4_testable = [p for p in f4_results if p is not None]
    f4_all_pass = all(f4_testable) if f4_testable else False

    # F5: Cross-family divergence
    lfm2_results = [r for r in valid_results if r["architecture"] == "lfm2"]
    qwen35_results = [r for r in valid_results if r["architecture"] == "qwen3.5"]

    f5_pass = None
    if lfm2_results and qwen35_results:
        lfm2_cum_signs = [
            r["correlations"]["cumulative_alignment"]["sign"] for r in lfm2_results
        ]
        qwen35_cum_signs = [
            r["correlations"]["cumulative_alignment"]["sign"] for r in qwen35_results
        ]
        f5_pass = set(lfm2_cum_signs) != set(qwen35_cum_signs)

    # Update F5 in each result
    for r in valid_results:
        r["falsifiers"]["F5_cross_family_divergence"]["pass"] = f5_pass

    falsifier_summary = {
        "F1_alignment_explains_norm_change": {
            "pass": f1_all_pass,
            "pass_count": sum(1 for p in f1_testable if p),
            "total": len(f1_testable),
            "failing_models": f1_failing,
            "verdict": (
                "PASS — alignment explains norm change direction"
                if f1_all_pass
                else f"FAIL — {len(f1_failing)} model(s) mismatch"
            ),
        },
        "F2_cumulative_matches_b6_1": {
            "pass": f2_all_pass,
            "pass_count": sum(1 for p in f2_testable if p),
            "total": len(f2_testable),
            "failing_models": f2_failing,
            "verdict": (
                "PASS — cumulative alignment predicts B6.1 norm coupling sign"
                if f2_all_pass
                else f"FAIL — {len(f2_failing)} model(s) mismatch B6.1"
            ),
        },
        "F3_direction_or_magnitude_resolvable": {
            "pass": f3_all_pass,
            "pass_count": sum(1 for p in f3_testable if p),
            "total": len(f3_testable),
            "verdict": (
                "PASS — direction or magnitude component is resolvable"
                if f3_all_pass
                else "FAIL — neither direction nor magnitude resolvable in some models"
            ),
        },
        "F4_operator_type_differential": {
            "pass": f4_all_pass,
            "pass_count": sum(1 for p in f4_testable if p),
            "total": len(f4_testable),
            "verdict": (
                "PASS — mixed-operator models show differential alignment coupling"
                if f4_all_pass
                else (
                    "FAIL — operator types show same alignment coupling"
                    if any(p is False for p in f4_results)
                    else "UNTESTABLE — no mixed-operator models or insufficient layers"
                )
            ),
        },
        "F5_cross_family_divergence": {
            "pass": f5_pass,
            "lfm2_cum_signs": [
                r["correlations"]["cumulative_alignment"]["sign"] for r in lfm2_results
            ] if lfm2_results else [],
            "qwen35_cum_signs": [
                r["correlations"]["cumulative_alignment"]["sign"] for r in qwen35_results
            ] if qwen35_results else [],
            "verdict": (
                "PASS — architectures show different cumulative alignment coupling"
                if f5_pass
                else (
                    "FAIL — architectures show same cumulative alignment coupling"
                    if f5_pass is False
                    else "UNTESTABLE — need both architectures"
                )
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
    logger.info("\n%s", "=" * 100)
    logger.info("B7.1: SUBLAYER ALIGNMENT PROBE — WHY NORM-ENTROPY COUPLING VARIES")
    logger.info("%s", "=" * 100)

    valid_results = [r for r in results if "error" not in r]

    header = (
        f"{'Model':<18} {'L':>3} {'MDE':>5} "
        f"{'r(align_c)':>10} {'r(align_m)':>10} {'r(align_t)':>10} "
        f"{'r(cum_al)':>10} {'r(Δ||h||²)':>10} "
        f"{'F1':>4} {'F2':>4}"
    )
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for r in valid_results:
        c = r["correlations"]

        def _fmt_r(comp_name):
            val = c[comp_name]["spearman_r"]
            resolvable = c[comp_name]["resolvable"]
            if math.isnan(val):
                return "      nan"
            marker = "*" if resolvable else " "
            return f"{val:+.4f}{marker}"

        f1 = r["falsifiers"]["F1_alignment_explains_norm_change"]["pass"]
        f2 = r["falsifiers"]["F2_cumulative_matches_b6_1"]["pass"]
        f1_str = "PASS" if f1 else ("FAIL" if f1 is False else "  - ")
        f2_str = "PASS" if f2 else ("FAIL" if f2 is False else "  - ")

        logger.info(
            "%-18s %3d %5.3f %10s %10s %10s %10s %10s %4s %4s",
            r["model_name"],
            r["num_layers"],
            r["detection_floor"]["mde"],
            _fmt_r("alignment_core"),
            _fmt_r("alignment_mlp"),
            _fmt_r("alignment_total"),
            _fmt_r("cumulative_alignment"),
            _fmt_r("delta_norm_sq"),
            f1_str,
            f2_str,
        )

    logger.info("\n* = resolvable (|r| > MDE)")

    # Alignment profile summary
    logger.info("\n%s", "-" * 70)
    logger.info("ALIGNMENT DEPTH PROFILES:")
    logger.info("%s", "-" * 70)
    for r in valid_results:
        prof = r["alignment_profiles"]
        align_total = prof["alignment_total"]
        cum_align = prof["cumulative_alignment"]
        logger.info(
            "  %s: align_total=[%.2f, %.2f], cum_align=[%.2f, %.2f]",
            r["model_name"],
            min(align_total), max(align_total),
            min(cum_align), max(cum_align),
        )

    # Directional decomposition
    logger.info("\n%s", "-" * 70)
    logger.info("DIRECTIONAL DECOMPOSITION:")
    logger.info("%s", "-" * 70)
    for r in valid_results:
        c = r["correlations"]
        logger.info(
            "  %s: r(dir_core)=%+.4f, r(dir_mlp)=%+.4f, "
            "r(||δ_core||)=%+.4f, r(||δ_mlp||)=%+.4f",
            r["model_name"],
            c["dir_align_core"]["spearman_r"],
            c["dir_align_mlp"]["spearman_r"],
            c["delta_core_norm"]["spearman_r"],
            c["delta_mlp_norm"]["spearman_r"],
        )

    # Operator-type stratification
    logger.info("\n%s", "-" * 70)
    logger.info("OPERATOR-TYPE STRATIFICATION:")
    logger.info("%s", "-" * 70)
    for r in valid_results:
        if r["operator_correlations"]:
            for op_type, op_data in r["operator_correlations"].items():
                if op_data.get("insufficient"):
                    logger.info(
                        "  %s [%s]: %d layers (insufficient)",
                        r["model_name"], op_type, op_data["n_layers"],
                    )
                else:
                    logger.info(
                        "  %s [%s]: %d layers, r(H, align_total)=%+.4f (p=%.4f)",
                        r["model_name"], op_type,
                        op_data["n_layers"],
                        op_data["spearman_r"],
                        op_data["spearman_p"],
                    )
        else:
            logger.info("  %s: single operator type (no stratification)", r["model_name"])

    # Falsifier summary
    logger.info("\n%s", "-" * 70)
    logger.info("FALSIFIER OUTCOMES:")
    logger.info("%s", "-" * 70)
    for f_name, f_data in cross_summary["falsifier_summary"].items():
        if "pass_count" in f_data:
            logger.info(
                "  %s: %d/%d pass — %s",
                f_name, f_data["pass_count"], f_data["total"], f_data["verdict"],
            )
        else:
            logger.info("  %s: %s", f_name, f_data["verdict"])

    # B6.1 cross-reference
    logger.info("\n%s", "-" * 70)
    logger.info("B6.1 CROSS-REFERENCE (F2 — CRITICAL):")
    logger.info("%s", "-" * 70)
    for r in valid_results:
        f2 = r["falsifiers"]["F2_cumulative_matches_b6_1"]
        logger.info(
            "  %s: B6.1 norm sign=%s, cum_alignment sign=%s, r=%.4f — %s",
            r["model_name"],
            f2["b6_1_norm_sign"],
            f2["cum_alignment_sign"],
            f2["r_cumulative_alignment"],
            "MATCH" if f2["pass"] else ("MISMATCH" if f2["pass"] is False else "N/A"),
        )

    # Permutation test details
    logger.info("\n%s", "-" * 70)
    logger.info("PERMUTATION TEST DETAILS (cumulative_alignment — F2 critical):")
    logger.info("%s", "-" * 70)
    for r in valid_results:
        perm = r["correlations"]["cumulative_alignment"]["permutation"]
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
    """Run the full B7.1 sublayer alignment probe experiment."""
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

        # Collect alignment data
        data = collect_alignment_data(
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
        with open(model_dir / "alignment_profile.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        logger.info("Saved: %s", model_dir / "alignment_profile.json")

    with open(output_base / "cross_model_summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2, default=str)
    logger.info("Saved: %s", output_base / "cross_model_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="B7.1: Sublayer Alignment Probe — Why Norm-Entropy Coupling Varies"
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
