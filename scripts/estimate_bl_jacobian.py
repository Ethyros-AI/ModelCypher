#!/usr/bin/env python3
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

"""Estimate depth-local B_l coupling from Jacobian factors.

Derivation target (entropy-curvature-derivation.md, Bedrock-B/C):

    sin^2(theta_l) <= a_l D + b_l D^2
    a_l = 4 ||B_l||^2 / ||h + delta||^2
    f_l(h) = P_perp(h) delta(h) = B_l (p(h) - u_V) + r_l(h)

This script estimates:
1. Architecture ceiling (weights-only): a_l_ceiling
2. Measured local coupling (input-conditioned Jacobian factors): a_l_measured
3. Measured quadratic remainder proxy: b_l_measured

No claim-promotion thresholds are used here; output is raw measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.geometry.decomposition import geodesic_svd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")

MODEL_REGISTRY = {
    "LFM2-350M": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
    "Qwen3.5-0.8B": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
}

PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "In mathematics, the derivative of x squared is",
    "Once upon a time in a land far away there lived",
]

_EPS_F32 = math.ldexp(1.0, -23)
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)
_EPS_BF16 = math.ldexp(1.0, -7)
_SQRT_EPS_BF16 = math.sqrt(_EPS_BF16)
_QUERY_POS = -2


def _resolve_backbone(model: Any) -> Any:
    """Resolve model backbone to object exposing embed_tokens + layers."""
    base = getattr(model, "model", None)
    if base is not None:
        if getattr(base, "layers", None) is not None and getattr(base, "embed_tokens", None) is not None:
            return base
        lm = getattr(base, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if inner is not None and getattr(inner, "layers", None) is not None:
                return inner
            if getattr(lm, "layers", None) is not None:
                return lm
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            return inner
        if getattr(lm, "layers", None) is not None:
            return lm
    return model


def _resolve_output_head(model: Any, base: Any) -> Any | None:
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


def _resolve_final_norm(base: Any) -> Any | None:
    """Final readout norm, aligned with backend readout order."""
    return getattr(base, "norm", None) or getattr(base, "embedding_norm", None)


def _call_with_fallback(module: Any, x: Any, mask: Any = None) -> Any:
    """Call module with tolerant signature fallback."""
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


def _attention_module(layer: Any) -> Any | None:
    return getattr(layer, "self_attn", None) or getattr(layer, "attn", None)


def _is_full_attention_layer(layer: Any) -> bool:
    attn = _attention_module(layer)
    if attn is None:
        return False
    return all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj"))


def _layer_mask(layer: Any) -> str | None:
    if hasattr(layer, "is_attention_layer"):
        return "causal" if bool(layer.is_attention_layer) else None
    return "causal" if _is_full_attention_layer(layer) else None


def _query_index(seq_len: int) -> int | None:
    idx = seq_len + _QUERY_POS if _QUERY_POS < 0 else _QUERY_POS
    if idx < 0 or idx >= seq_len:
        return None
    return idx


def _derive_num_tangent_probes(hidden_dim: int) -> int:
    """Derive probe count from state-space bit-depth (dimension resolution)."""
    if hidden_dim <= 2:
        return 1
    return min(hidden_dim - 1, max(2, int(math.ceil(math.log2(hidden_dim)))))


def _weight_matrix(linear_module: Any, mx: Any) -> Any:
    """Return dequantized weight matrix when needed."""
    w = linear_module.weight
    if hasattr(linear_module, "scales") and hasattr(linear_module, "bits"):
        biases = getattr(linear_module, "biases", None)
        if biases is not None:
            return mx.dequantize(
                w,
                linear_module.scales,
                biases,
                linear_module.group_size,
                linear_module.bits,
            )
    return w


def _spectral_norm(weight: Any, backend: Any) -> float:
    """Top singular value via backend SVD."""
    w_f32 = backend.astype(weight, "float32")
    _, s, _ = geodesic_svd(backend, w_f32, k=1)
    backend.eval(s)
    if int(s.shape[0]) == 0:
        return 0.0
    return float(backend.to_scalar(s[0]))


def _head_dim(attn: Any) -> int:
    hd = getattr(attn, "head_dim", None)
    if hd is not None:
        return int(hd)
    num_heads = (
        getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
        or getattr(attn, "n_heads", None)
    )
    num_kv = (
        getattr(attn, "num_key_value_heads", None)
        or getattr(attn, "n_kv_heads", None)
        or num_heads
    )
    k_weight = attn.k_proj.weight
    if num_kv is None or int(num_kv) <= 0:
        raise ValueError("Cannot resolve head_dim: missing num_kv_heads")
    return int(k_weight.shape[0]) // int(num_kv)


def _perp_projection(delta: np.ndarray, h: np.ndarray) -> np.ndarray:
    h_sq = float(np.dot(h, h))
    if h_sq <= _SQRT_EPS_F32:
        return delta.copy()
    proj_coeff = float(np.dot(delta, h) / h_sq)
    return delta - proj_coeff * h


def _stable_softmax_probs(logits_np: np.ndarray) -> np.ndarray:
    shifted = logits_np - float(np.max(logits_np))
    exp_shifted = np.exp(shifted)
    denom = float(np.sum(exp_shifted))
    if denom <= _SQRT_EPS_F32:
        return np.full_like(logits_np, 1.0 / max(1, logits_np.size), dtype=np.float32)
    return (exp_shifted / denom).astype(np.float32)


def _posterior_from_hidden(
    h_query: np.ndarray,
    final_norm: Any | None,
    output_head: Any | None,
    embed_tokens: Any,
    mx: Any,
) -> np.ndarray:
    """Compute posterior p from layer-local hidden state."""
    h_mx = mx.array(h_query.reshape(1, 1, -1))
    if final_norm is not None:
        h_mx = final_norm(h_mx)

    if output_head is not None:
        logits = output_head(h_mx)
    else:
        logits = embed_tokens.as_linear(h_mx)
    if isinstance(logits, tuple):
        logits = logits[0]

    logits_row = logits[0, 0, :].astype(mx.float32)
    mx.eval(logits_row)
    logits_np = np.array(logits_row.tolist(), dtype=np.float32)
    return _stable_softmax_probs(logits_np)


def _replace_query_row(hidden: Any, q_idx: int, new_row: Any, mx: Any) -> Any:
    """Return hidden with query row replaced by new_row [1,1,d]."""
    seq_len = int(hidden.shape[1])
    if q_idx == 0:
        return mx.concatenate([new_row, hidden[:, 1:, :]], axis=1)
    if q_idx == seq_len - 1:
        return mx.concatenate([hidden[:, :-1, :], new_row], axis=1)
    return mx.concatenate(
        [hidden[:, :q_idx, :], new_row, hidden[:, q_idx + 1 :, :]],
        axis=1,
    )


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def _orthonormal_tangent_basis(h_hat: np.ndarray, n_dirs: int, seed: int) -> np.ndarray:
    """Build orthonormal basis vectors in tangent space at h_hat."""
    dim = h_hat.shape[0]
    rng = np.random.default_rng(seed)
    basis: list[np.ndarray] = []
    attempt_limit = max(dim, 2 * n_dirs)
    attempts = 0

    while len(basis) < n_dirs and attempts < attempt_limit:
        v = rng.standard_normal(dim).astype(np.float32)
        v -= float(np.dot(v, h_hat)) * h_hat
        for b in basis:
            v -= float(np.dot(v, b)) * b
        nrm = float(np.linalg.norm(v))
        if nrm > _SQRT_EPS_F32:
            basis.append(v / nrm)
        attempts += 1

    if len(basis) < n_dirs:
        raise RuntimeError(
            f"Failed to build tangent basis: got {len(basis)} / {n_dirs} vectors."
        )

    return np.column_stack(basis)  # [d, n_dirs]


def _baseline_terms(
    h_in: Any,
    h_out: Any,
    q_idx: int,
    final_norm: Any | None,
    output_head: Any | None,
    embed_tokens: Any,
    mx: Any,
) -> dict[str, Any]:
    """Compute baseline f(h), p(h), and norms at query position."""
    h_q = np.array(h_in[0, q_idx, :].astype(mx.float32).tolist(), dtype=np.float32)
    h_out_q = np.array(h_out[0, q_idx, :].astype(mx.float32).tolist(), dtype=np.float32)
    delta = h_out_q - h_q
    f_val = _perp_projection(delta, h_q)
    p_val = _posterior_from_hidden(h_out_q, final_norm, output_head, embed_tokens, mx)

    h_norm = float(np.linalg.norm(h_q))
    h_out_norm_sq = float(np.dot(h_out_q, h_out_q))
    delta_norm = float(np.linalg.norm(delta))
    r_ratio = delta_norm / max(h_norm, _SQRT_EPS_F32)

    return {
        "h_q": h_q,
        "h_out_q": h_out_q,
        "f_val": f_val,
        "p_val": p_val,
        "h_norm": h_norm,
        "h_out_norm_sq": h_out_norm_sq,
        "r_ratio": r_ratio,
    }


def _estimate_local_coupling(
    layer: Any,
    h_in: Any,
    q_idx: int,
    layer_mask: Any,
    baseline: dict[str, Any],
    final_norm: Any | None,
    output_head: Any | None,
    embed_tokens: Any,
    mx: Any,
    epsilon: float,
    n_dirs: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate ||B_l|| from sampled local Jacobian factors."""
    h_q = baseline["h_q"]
    f_base = baseline["f_val"]
    p_base = baseline["p_val"]
    h_norm = baseline["h_norm"]
    h_out_norm_sq = baseline["h_out_norm_sq"]

    dim = h_q.shape[0]
    if dim <= 1:
        return {
            "n_dirs": 0,
            "sigma_jf_max": float("nan"),
            "sigma_jg_min": float("nan"),
            "sigma_jg_min_effective": float("nan"),
            "sigma_jg_floor_applied": False,
            "B_norm_measured": float("nan"),
            "a_l_measured": float("nan"),
            "c_l_measured": float("nan"),
            "b_l_measured": float("nan"),
        }

    n_dirs = min(n_dirs, dim - 1)
    h_hat = h_q / max(h_norm, _SQRT_EPS_F32)
    basis = _orthonormal_tangent_basis(h_hat, n_dirs, seed)

    f_cols: list[np.ndarray] = []
    p_cols: list[np.ndarray] = []

    for j in range(n_dirs):
        direction = basis[:, j]
        h_hat_pert = h_hat + epsilon * direction
        h_hat_pert /= max(float(np.linalg.norm(h_hat_pert)), _SQRT_EPS_F32)
        h_q_pert = (h_norm * h_hat_pert).astype(np.float32)

        new_row = mx.array(h_q_pert.reshape(1, 1, -1)).astype(h_in.dtype)
        h_pert = _replace_query_row(h_in, q_idx, new_row, mx)
        h_out_pert = _call_with_fallback(layer, h_pert, mask=layer_mask)
        if isinstance(h_out_pert, tuple):
            h_out_pert = h_out_pert[0]
        mx.eval(h_out_pert)

        h_out_q_pert = np.array(
            h_out_pert[0, q_idx, :].astype(mx.float32).tolist(),
            dtype=np.float32,
        )
        delta_pert = h_out_q_pert - h_q_pert
        f_pert = _perp_projection(delta_pert, h_q_pert)
        p_pert = _posterior_from_hidden(
            h_out_q_pert,
            final_norm,
            output_head,
            embed_tokens,
            mx,
        )

        df = ((f_pert - f_base) / epsilon).astype(np.float64)
        dp = ((p_pert - p_base) / epsilon).astype(np.float64)
        if np.all(np.isfinite(df)) and np.all(np.isfinite(dp)):
            f_cols.append(df)
            p_cols.append(dp)

    n_valid = len(f_cols)
    if n_valid < 2:
        return {
            "n_dirs": n_valid,
            "sigma_jf_max": float("nan"),
            "sigma_jg_min": float("nan"),
            "sigma_jg_min_effective": float("nan"),
            "sigma_jg_floor_applied": False,
            "B_norm_measured": float("nan"),
            "a_l_measured": float("nan"),
            "c_l_measured": float("nan"),
            "b_l_measured": float("nan"),
        }

    F = np.column_stack(f_cols)  # [d, n_valid]
    P = np.column_stack(p_cols)  # [vocab, n_valid]

    s_f = np.linalg.svd(F, compute_uv=False, full_matrices=False)
    s_p = np.linalg.svd(P, compute_uv=False, full_matrices=False)
    sigma_jf_max = float(s_f[0]) if s_f.size else 0.0
    sigma_jg_min = float(s_p[-1]) if s_p.size else 0.0

    sigma_floor = _SQRT_EPS_F32
    sigma_jg_min_effective = max(sigma_jg_min, sigma_floor)
    sigma_jg_floor_applied = sigma_jg_min < sigma_floor

    b_norm_measured = sigma_jf_max / sigma_jg_min_effective
    a_l_measured = 4.0 * (b_norm_measured ** 2) / max(h_out_norm_sq, sigma_floor)

    # c_l proxy from hold-out residuals:
    # fit linear map on one subset of perturbations, evaluate on held-out subset.
    c_l_measured = float("nan")
    b_l_measured = float("nan")
    if n_valid >= 3:
        n_fit = max(2, n_valid // 2)
        P_fit = P[:, :n_fit]
        F_fit = F[:, :n_fit]
        p_scale = max(float(np.max(np.abs(P_fit))), 1.0)
        P_fit_scaled = P_fit / p_scale
        ptp = P_fit_scaled.T @ P_fit_scaled
        ptp_scale = float(np.trace(ptp)) / max(1, n_fit)
        ridge = _SQRT_EPS_F32 * max(ptp_scale, _SQRT_EPS_F32)
        eye = np.eye(n_fit, dtype=np.float64)
        c_candidates = []
        for j in range(n_fit, n_valid):
            p_val = P[:, j]
            f_val = F[:, j]
            p_val_scaled = p_val / p_scale
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    rhs = P_fit_scaled.T @ p_val_scaled
                    coeff = np.linalg.solve(ptp + ridge * eye, rhs)
                    if not np.all(np.isfinite(coeff)):
                        continue
                    f_pred = F_fit @ coeff
                    if not np.all(np.isfinite(f_pred)):
                        continue
            except FloatingPointError:
                continue
            resid_norm = float(np.linalg.norm(f_val - f_pred))
            dp_norm = float(np.linalg.norm(p_val))
            denom = max(dp_norm * dp_norm, sigma_floor)
            c_candidates.append(resid_norm / denom)

        if c_candidates:
            c_l_measured = float(max(c_candidates))
            b_l_measured = 8.0 * (c_l_measured ** 2) / max(h_out_norm_sq, sigma_floor)

    return {
        "n_dirs": n_valid,
        "sigma_jf_max": sigma_jf_max,
        "sigma_jg_min": sigma_jg_min,
        "sigma_jg_min_effective": sigma_jg_min_effective,
        "sigma_jg_floor_applied": sigma_jg_floor_applied,
        "B_norm_measured": b_norm_measured,
        "a_l_measured": a_l_measured,
        "c_l_measured": c_l_measured,
        "b_l_measured": b_l_measured,
    }


def _layer_architecture_terms(layer: Any, backend: Any, mx: Any) -> dict[str, float]:
    """Extract layer-wise architecture terms for Jacobian ceiling."""
    attn = _attention_module(layer)
    if attn is None:
        raise ValueError("Layer has no attention module")

    q_w = _weight_matrix(attn.q_proj, mx)
    k_w = _weight_matrix(attn.k_proj, mx)
    v_w = _weight_matrix(attn.v_proj, mx)

    o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
    if o_proj is None:
        raise ValueError("Attention module missing o_proj/out_proj")
    o_w = _weight_matrix(o_proj, mx)

    sigma_q = _spectral_norm(q_w, backend)
    sigma_k = _spectral_norm(k_w, backend)
    sigma_v = _spectral_norm(v_w, backend)
    sigma_o = _spectral_norm(o_w, backend)
    d_k = _head_dim(attn)

    path_value = sigma_o * sigma_v
    path_score = path_value * sigma_q * sigma_k / math.sqrt(max(1, d_k))

    return {
        "sigma_q": sigma_q,
        "sigma_k": sigma_k,
        "sigma_v": sigma_v,
        "sigma_o": sigma_o,
        "d_k": float(d_k),
        "path_value_term": path_value,
        "path_score_term": path_score,
    }


def _aggregate_layer_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric keys over probes for one layer."""
    if not records:
        return {}

    out: dict[str, Any] = {"n_probes": len(records)}
    keys = sorted(records[0].keys())
    for key in keys:
        values = [r[key] for r in records if isinstance(r.get(key), (int, float, bool))]
        if values and all(isinstance(v, bool) for v in values):
            out[key] = float(np.mean(np.array(values, dtype=np.float32)))
        elif values:
            vals = np.array(values, dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


def _run_single_model(
    model_name: str,
    model_path: str,
    probes: list[str],
    backend: Any,
    epsilon: float,
    num_tangent_probes_override: int | None,
    max_attn_layers: int | None,
) -> dict[str, Any]:
    logger.info("Loading %s from %s", model_name, model_path)
    model, tokenizer = backend.load_model(model_path)
    mx = backend.mx

    base = _resolve_backbone(model)
    layers = base.layers
    embed_tokens = base.embed_tokens
    final_norm = _resolve_final_norm(base)
    output_head = _resolve_output_head(model, base)

    selected_attn_layers = []
    layer_arch = {}
    for idx, layer in enumerate(layers):
        if not _is_full_attention_layer(layer):
            continue
        if max_attn_layers is not None and len(selected_attn_layers) >= max_attn_layers:
            break
        try:
            layer_arch[idx] = _layer_architecture_terms(layer, backend, mx)
            selected_attn_layers.append(idx)
        except Exception as exc:
            logger.warning("Skipping layer %d architecture terms: %s", idx, exc)

    logger.info(
        "%s: selected %d attention layers (%s)",
        model_name,
        len(selected_attn_layers),
        selected_attn_layers,
    )

    per_probe_rows = []
    per_layer_rows: dict[int, list[dict[str, Any]]] = {idx: [] for idx in selected_attn_layers}

    for probe_idx, text in enumerate(probes):
        logger.info("%s: probe %d/%d", model_name, probe_idx + 1, len(probes))
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 3:
            logger.warning("Probe too short after tokenization, skipping: %r", text)
            continue

        input_ids = mx.array([token_ids])
        h = embed_tokens(input_ids)
        mx.eval(h)

        seq_len = int(input_ids.shape[1])
        q_idx = _query_index(seq_len)
        if q_idx is None:
            logger.warning("Invalid query index for seq_len=%d, skipping probe.", seq_len)
            continue

        probe_layers = []
        attn_count_seen = 0

        for layer_idx, layer in enumerate(layers):
            layer_is_attn = _is_full_attention_layer(layer)
            layer_mask = _layer_mask(layer)
            h_in = h
            h_out = _call_with_fallback(layer, h_in, mask=layer_mask)
            if isinstance(h_out, tuple):
                h_out = h_out[0]
            mx.eval(h_out)

            if layer_idx in layer_arch:
                arch = layer_arch[layer_idx]
                baseline = _baseline_terms(
                    h_in,
                    h_out,
                    q_idx,
                    final_norm,
                    output_head,
                    embed_tokens,
                    mx,
                )

                hidden_dim = baseline["h_q"].shape[0]
                n_dirs = (
                    num_tangent_probes_override
                    if num_tangent_probes_override is not None
                    else _derive_num_tangent_probes(hidden_dim)
                )
                n_dirs = min(n_dirs, max(1, hidden_dim - 1))

                seed = _stable_seed(model_name, layer_idx, probe_idx, seq_len, hidden_dim)
                measured = _estimate_local_coupling(
                    layer=layer,
                    h_in=h_in,
                    q_idx=q_idx,
                    layer_mask=layer_mask,
                    baseline=baseline,
                    final_norm=final_norm,
                    output_head=output_head,
                    embed_tokens=embed_tokens,
                    mx=mx,
                    epsilon=epsilon,
                    n_dirs=n_dirs,
                    seed=seed,
                )

                j_tan_ceiling = (
                    arch["path_value_term"]
                    + arch["path_score_term"]
                    + 2.0 * baseline["r_ratio"]
                )
                a_l_ceiling = (
                    4.0 * (j_tan_ceiling ** 2) / max(baseline["h_out_norm_sq"], _SQRT_EPS_F32)
                )

                row = {
                    "layer_idx": layer_idx,
                    "probe_idx": probe_idx,
                    "probe_text": text,
                    "seq_len": seq_len,
                    "query_idx": q_idx,
                    "hidden_dim": hidden_dim,
                    "is_full_attention_layer": layer_is_attn,
                    "r_ratio": baseline["r_ratio"],
                    "h_out_norm_sq": baseline["h_out_norm_sq"],
                    "n_dirs": measured["n_dirs"],
                    "sigma_q": arch["sigma_q"],
                    "sigma_k": arch["sigma_k"],
                    "sigma_v": arch["sigma_v"],
                    "sigma_o": arch["sigma_o"],
                    "d_k": arch["d_k"],
                    "path_value_term": arch["path_value_term"],
                    "path_score_term": arch["path_score_term"],
                    "j_tan_ceiling": j_tan_ceiling,
                    "a_l_ceiling": a_l_ceiling,
                    "sigma_jf_max": measured["sigma_jf_max"],
                    "sigma_jg_min": measured["sigma_jg_min"],
                    "sigma_jg_min_effective": measured["sigma_jg_min_effective"],
                    "sigma_jg_floor_applied": measured["sigma_jg_floor_applied"],
                    "B_norm_measured": measured["B_norm_measured"],
                    "a_l_measured": measured["a_l_measured"],
                    "c_l_measured": measured["c_l_measured"],
                    "b_l_measured": measured["b_l_measured"],
                    "a_measured_over_ceiling": (
                        measured["a_l_measured"] / a_l_ceiling if a_l_ceiling > _SQRT_EPS_F32 else float("nan")
                    ),
                }

                probe_layers.append(row)
                per_layer_rows[layer_idx].append(row)
                attn_count_seen += 1

            h = h_out

        per_probe_rows.append({
            "probe_idx": probe_idx,
            "probe_text": text,
            "layers": probe_layers,
            "n_layers_measured": attn_count_seen,
        })

    layer_summary = []
    for layer_idx in selected_attn_layers:
        rows = per_layer_rows.get(layer_idx, [])
        if not rows:
            continue
        agg = _aggregate_layer_records(rows)
        agg["layer_idx"] = layer_idx
        layer_summary.append(agg)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "epsilon": epsilon,
        "n_probes": len(per_probe_rows),
        "n_attention_layers_measured": len(selected_attn_layers),
        "selected_attention_layers": selected_attn_layers,
        "layer_architecture_terms": layer_arch,
        "per_probe": per_probe_rows,
        "layer_summary": layer_summary,
    }


def _write_text_summary(out_path: Path, run_doc: dict[str, Any]) -> None:
    lines = [
        "B_l Jacobian Estimation Summary",
        f"Run ID: {run_doc['run_id']}",
        f"Timestamp: {run_doc['timestamp']}",
        f"Epsilon: {run_doc['epsilon']:.6e} (sqrt(eps_bf16), IEEE 754)",
        "",
    ]

    for model in run_doc["models"]:
        lines.append(f"Model: {model['model_name']}")
        lines.append(f"  Path: {model['model_path']}")
        lines.append(f"  Attention layers measured: {model['selected_attention_layers']}")
        lines.append(f"  Probes: {model['n_probes']}")
        lines.append("")
        lines.append("  layer | a_ceiling_mean | a_measured_mean | ratio_mean | b_measured_mean")
        lines.append("  ------+----------------+-----------------+------------+---------------")
        for row in model.get("layer_summary", []):
            a_c = row.get("a_l_ceiling_mean", float("nan"))
            a_m = row.get("a_l_measured_mean", float("nan"))
            ratio = row.get("a_measured_over_ceiling_mean", float("nan"))
            b_m = row.get("b_l_measured_mean", float("nan"))
            lines.append(
                f"  {int(row['layer_idx']):>5d} | {a_c:>14.6e} | {a_m:>15.6e} | "
                f"{ratio:>10.6f} | {b_m:>13.6e}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    backend = initialize_default_backend()

    if args.smoke:
        model_names = [args.models[0]] if args.models else ["LFM2-350M"]
        probes = PROBE_TEXTS[:1]
    else:
        model_names = args.models if args.models else list(MODEL_REGISTRY.keys())
        probes = PROBE_TEXTS

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for model_name in model_names:
        model_path = MODEL_REGISTRY.get(model_name)
        if model_path is None:
            logger.warning("Unknown model key: %s (skipping)", model_name)
            continue
        if not Path(model_path).exists():
            logger.warning("Model path not found: %s (skipping)", model_path)
            continue
        t0 = time.time()
        model_doc = _run_single_model(
            model_name=model_name,
            model_path=model_path,
            probes=probes,
            backend=backend,
            epsilon=args.epsilon,
            num_tangent_probes_override=args.num_tangent_probes,
            max_attn_layers=args.max_attn_layers,
        )
        model_doc["elapsed_sec"] = time.time() - t0
        models.append(model_doc)

    run_doc = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "epsilon": args.epsilon,
        "epsilon_source": "sqrt(eps_bf16)",
        "query_position": _QUERY_POS,
        "models": models,
    }

    json_path = out_dir / "bl_estimation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2, default=str)

    txt_path = out_dir / "bl_estimation.txt"
    _write_text_summary(txt_path, run_doc)

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", txt_path)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate B_l Jacobian coupling terms.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model keys from registry: {sorted(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--output",
        default="results/bl_estimation",
        help="Output directory root (run_id subdir is created).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id for output subdirectory.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=_SQRT_EPS_BF16,
        help="Finite-difference step epsilon (default sqrt(eps_bf16)).",
    )
    parser.add_argument(
        "--num-tangent-probes",
        type=int,
        default=None,
        help=(
            "Override tangent probe count. Default is derived per layer: "
            "ceil(log2(hidden_dim)), clipped to [2, hidden_dim-1]."
        ),
    )
    parser.add_argument(
        "--max-attn-layers",
        type=int,
        default=None,
        help="Optional cap on measured full-attention layers per model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one model and one probe for a fast pipeline check.",
    )
    args = parser.parse_args()

    out_dir = run(args)
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
