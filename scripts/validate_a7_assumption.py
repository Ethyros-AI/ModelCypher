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

"""A7 Assumption Validator: radial-dominant downstream gradient.

PRE-REGISTRATION CONTRACT (entropy-curvature-derivation.md D3.3)
================================================================

A7 states: g = -β ĥ + g_⊥,  with β > 0 and ⟨g_⊥, w_t - δ⟩ mean-zero
over t conditioned on r_t.

FALSIFIER CONDITIONS:
1. If E[Δs_t | r_t] is NOT monotone increasing in r_t - R → A7 FALSIFIED
2. If E[ΔR] < 0 → A7 FALSIFIED

METHOD: Score-level finite difference.
Key identity: log(softmax(s))_t = s_t - LSE(s). Since softmax is
shift-invariant, reconstruct effective scores from post-softmax weights:
z_t = log(α_t), perturb z_t' = z_t + ε, recompute softmax(z').

MEASUREMENT:
∂L/∂s_t ≈ (L(s + ε e_t) - L(s)) / ε  via forward finite difference.

ε = sqrt(eps_bf16) ≈ 0.088 (IEEE 754 optimal finite-difference step
for bf16 computation dtype: balances truncation error O(ε) against
rounding error O(eps_bf16/ε)). The perturbation propagates through bf16
matmuls, so the computation dtype — not accumulation format — determines
the precision floor. Reference: Numerical Recipes §5.7.

EXPERIMENTAL STATUS: scripts/ only, NOT CLI (AGENTS.md:695)
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

# IEEE 754 constants
_EPS_F32 = math.ldexp(1.0, -23)
_EPS_BF16 = math.ldexp(1.0, -7)
_SQRT_EPS_BF16 = math.sqrt(_EPS_BF16)
# IEEE 754 float32 minimum positive normal number (for log floor)
_FLOAT32_MIN_NORMAL = math.ldexp(1.0, -126)
# Optimal finite-difference step: must be derived from the COMPUTATION dtype,
# not the accumulation format. The perturbation propagates through bf16 matmuls
# (hidden state values O(1), ULP ≈ eps_bf16 ≈ 0.0078). A perturbation that
# produces Δα_t < eps_bf16 gets quantized to zero at the first bf16 op.
# Optimal step balances truncation O(ε) vs rounding O(eps_bf16/ε):
#   ε_opt = sqrt(eps_bf16) = sqrt(2^-7) ≈ 0.0884
# At this ε, Δα_t ≈ α(1-α)ε ≈ 0.014 > eps_bf16 for typical α ≈ 0.2.
_EPSILON = _SQRT_EPS_BF16
# Pre-registered significance level (Fisher convention, matches perturbation experiment)
_PRE_REGISTERED_ALPHA = 0.05
# Permutation count for Monte Carlo test (T > 8).
# Holm-Bonferroni strictest threshold is α/n_tests. For the Monte Carlo
# test to resolve this, need min_p = 1/(n_perms+1) < α/n_tests, i.e.
# n_perms > n_tests/α - 1. Worst case: 6 layers × 16 heads = 96 tests.
# n_perms > 96/0.05 - 1 = 1919. Use 10× margin for stable p-value
# estimation: 19200. (For T ≤ 8, exact test uses all T! permutations
# regardless of this constant.)
_MAX_HEADS_PER_MODEL = 192  # 6 attn layers × 32 heads (LFM2-1.2B worst case)
_N_PERMUTATIONS = int(math.ceil(10.0 * _MAX_HEADS_PER_MODEL / _PRE_REGISTERED_ALPHA))
# Query position whose logits drive the CE loss. Under causal masking,
# logits[-2] predicts token_ids[-1]. All perturbations and alpha
# extractions must target this query row for gradient path consistency.
_QUERY_POS = -2

MODELS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "Qwen3.5-0.8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16",
    "LFM2-700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "LFM2-1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Base-bf16",
    "Qwen3.5-2B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-2B-bf16",
}

# Probes must tokenize to T ≥ 7 for the exact permutation test (T ≤ 8)
# to resolve the Holm threshold α/n_tests. With n_tests ≈ 96 and α = 0.05,
# Holm threshold ≈ 0.00053. Minimum p from T! permutations: 1/T!.
# T=6: 1/720 = 0.0014 > 0.00053 — IMPOSSIBLE. T=7: 1/5040 = 0.0002 — OK.
# "The capital of France is" tokenizes to T=6 on LFM2. Replaced.
PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "In mathematics, the derivative of x squared is",
    "Once upon a time in a land far away",
]


# =====================================================================
# Statistical utilities (same framework as attention_perturbation_experiment.py)
# =====================================================================


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between x and y."""
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(data: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        for r, i in enumerate(sorted_indices):
            ranks[i] = r + 1.0
        return ranks

    rx = rank(x)
    ry = rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x < _EPS_F32 or den_y < _EPS_F32:
        return 0.0
    return num / (den_x * den_y)


def permutation_test_p_value(
    x: list[float], y: list[float], n_perms: int | None = None
) -> float:
    """One-sided permutation test for positive Spearman correlation.

    For n <= 8, enumerates ALL n! unique permutations (exact test).
    For n > 8, uses Monte Carlo with deterministic RNG.

    Returns one-sided p-value: fraction of permutations where
    ρ_perm >= ρ_observed (testing for positive correlation).
    """
    n = len(x)
    observed = spearman_rank_correlation(x, y)
    if math.isnan(observed):
        return 1.0

    if n <= 8:
        total = 0
        count_ge = 0
        for perm in itertools.permutations(y):
            perm_r = spearman_rank_correlation(x, list(perm))
            if not math.isnan(perm_r) and perm_r >= observed:
                count_ge += 1
            total += 1
        return count_ge / total
    else:
        if n_perms is None:
            n_perms = _N_PERMUTATIONS

        seed_payload = (
            tuple(round(v, 12) for v in x),
            tuple(round(v, 12) for v in y),
            n_perms,
        )
        rng = random.Random(repr(seed_payload))

        count_ge = 0
        y_perm = y.copy()
        for _ in range(n_perms):
            rng.shuffle(y_perm)
            perm_r = spearman_rank_correlation(x, y_perm)
            if not math.isnan(perm_r) and perm_r >= observed:
                count_ge += 1

        return (count_ge + 1) / (n_perms + 1)


def ols_slope(x: list[float], y: list[float]) -> float:
    """Ordinary least squares slope of y ~ x (no intercept needed for sign)."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    if den < _EPS_F32:
        return 0.0
    return num / den


# =====================================================================
# CE loss computation
# =====================================================================


def compute_ce_loss_last_token(logits, token_ids, mx) -> float:
    """Cross-entropy loss for next-token prediction at penultimate position.

    logits: [batch, seq, vocab]
    token_ids: list of token IDs (length seq)
    Loss = -log softmax(logits[0, -2, :])_{token_ids[-1]}

    Under causal masking, logits[0, q, :] is produced by query position q,
    which attends to keys 0..q. Standard next-token loss: logits[-2]
    predicts token_ids[-1]. The gradient flows through query position -2.

    CRITICAL: all attention perturbations must target query row -2 (not -1)
    to match this loss. See QUERY_POS below.
    """
    if len(token_ids) < 2:
        return 0.0
    # logits[-2] predicts token_ids[-1]; query position -2 drives this.
    last_logits = logits[0, -2, :]  # [vocab]
    target = token_ids[-1]

    # Numerically stable log-softmax: log_softmax(x)_i = x_i - LSE(x)
    max_logit = float(mx.max(last_logits))
    shifted = last_logits - max_logit
    lse = max_logit + float(mx.log(mx.sum(mx.exp(shifted))))
    log_prob = float(last_logits[target]) - lse
    return -log_prob


# =====================================================================
# Phase 1: Baseline collection
# =====================================================================


def collect_baseline(model, tokenizer, text, backend, mx):
    """Collect baseline attention weights, V/W_O parameters, and loss.

    Returns dict with:
        token_ids: list[int]
        attn_weights: dict[layer_idx -> weights array [batch, heads, seq, seq]]
        baseline_loss: float
        layer_info: dict[layer_idx -> {v_proj_weight, o_proj_weight, num_heads, head_dim}]
    """
    token_ids = tokenizer.encode(text)

    # Forward pass with no hook → baseline logits + attention weights
    logits, attn_weights = backend.collect_logits_with_attention_hook(
        model, tokenizer, text, attention_hook=None, token_ids=token_ids
    )

    baseline_loss = compute_ce_loss_last_token(logits, token_ids, mx)

    # Extract V, O projection weights per attention layer
    base = backend._resolve_model_base(model)
    layer_info = {}
    for layer_idx in sorted(attn_weights.keys()):
        layer = base.layers[layer_idx]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)

        num_heads = (
            getattr(attn, "num_heads", None)
            or getattr(attn, "num_attention_heads", None)
            or getattr(attn, "n_heads", None)
        )
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            k_weight = attn.k_proj.weight
            num_kv = (
                getattr(attn, "num_key_value_heads", None)
                or getattr(attn, "n_kv_heads", None)
                or num_heads
            )
            head_dim = k_weight.shape[0] // num_kv

        layer_info[layer_idx] = {
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    return {
        "token_ids": token_ids,
        "attn_weights": attn_weights,
        "baseline_loss": baseline_loss,
        "layer_info": layer_info,
    }


# =====================================================================
# Phase 2: Per-token score gradient via finite difference
# =====================================================================


def compute_score_gradients(
    model, tokenizer, text, baseline, backend, mx
) -> dict[int, dict[int, list[float]]]:
    """Compute ∂L/∂s_t for each (layer, head, token) via finite difference.

    Returns: dict[layer_idx -> dict[head_idx -> list[∂L/∂s_t per token]]]
    """
    token_ids = baseline["token_ids"]
    attn_weights = baseline["attn_weights"]
    baseline_loss = baseline["baseline_loss"]
    seq_len = len(token_ids)

    gradients: dict[int, dict[int, list[float]]] = {}

    for layer_idx in sorted(attn_weights.keys()):
        w = attn_weights[layer_idx]
        num_heads = w.shape[1]
        gradients[layer_idx] = {}

        for head_idx in range(num_heads):
            grad_per_token: list[float] = []

            for token_t in range(seq_len):
                # Create hook: perturb score at (layer_idx, head_idx, QUERY_POS, token_t)
                def make_hook(target_layer, target_head, target_token, eps):
                    def hook(weights, l_idx):
                        if l_idx != target_layer:
                            return weights
                        # Reconstruct log-scores from post-softmax weights
                        z = mx.log(weights + _FLOAT32_MIN_NORMAL)
                        # Perturb target score at the query position driving the loss
                        # z shape: [batch, heads, seq, seq]
                        z_list = z.tolist()
                        z_list[0][target_head][_QUERY_POS][target_token] += eps
                        z_perturbed = mx.array(z_list)
                        # Recompute softmax over key dimension (axis=-1)
                        return mx.softmax(z_perturbed, axis=-1)
                    return hook

                hook = make_hook(layer_idx, head_idx, token_t, _EPSILON)
                perturbed_logits, _ = backend.collect_logits_with_attention_hook(
                    model, tokenizer, text, attention_hook=hook, token_ids=token_ids
                )
                perturbed_loss = compute_ce_loss_last_token(perturbed_logits, token_ids, mx)

                dLds = (perturbed_loss - baseline_loss) / _EPSILON
                grad_per_token.append(dLds)

            gradients[layer_idx][head_idx] = grad_per_token
            mx.eval(mx.array([0.0]))  # sync

        print(f"    Layer {layer_idx}: {num_heads} heads × {seq_len} tokens computed")

    return gradients


def sensitivity_check(
    model, tokenizer, text, baseline, backend, mx, layer_idx: int, head_idx: int
) -> dict:
    """Verify finite difference is in linear regime for one (layer, head).

    Checks: |∂L/∂s_t(ε) - ∂L/∂s_t(ε/10)| / |∂L/∂s_t(ε)| < 0.1
    for token t=0 (first token).
    """
    token_ids = baseline["token_ids"]
    baseline_loss = baseline["baseline_loss"]

    results = {}
    for eps_label, eps_val in [("eps", _EPSILON), ("eps/10", _EPSILON / 10), ("10*eps", _EPSILON * 10)]:
        def make_hook(target_layer, target_head, eps):
            def hook(weights, l_idx):
                if l_idx != target_layer:
                    return weights
                z = mx.log(weights + _FLOAT32_MIN_NORMAL)
                z_list = z.tolist()
                z_list[0][target_head][_QUERY_POS][0] += eps
                z_perturbed = mx.array(z_list)
                return mx.softmax(z_perturbed, axis=-1)
            return hook

        hook = make_hook(layer_idx, head_idx, eps_val)
        perturbed_logits, _ = backend.collect_logits_with_attention_hook(
            model, tokenizer, text, attention_hook=hook, token_ids=token_ids
        )
        perturbed_loss = compute_ce_loss_last_token(perturbed_logits, token_ids, mx)
        results[eps_label] = (perturbed_loss - baseline_loss) / eps_val

    dLds_eps = results["eps"]
    dLds_fine = results["eps/10"]

    if abs(dLds_eps) > _EPS_F32:
        relative_diff = abs(dLds_eps - dLds_fine) / abs(dLds_eps)
    else:
        relative_diff = 0.0  # gradient effectively zero

    return {
        "dLds_eps": dLds_eps,
        "dLds_eps_over_10": dLds_fine,
        "dLds_10_eps": results["10*eps"],
        "relative_diff": relative_diff,
        # 0.1 threshold: if shrinking ε by 10× changes the estimate by >10%,
        # the O(ε) truncation term is comparable to the O(1) true derivative,
        # meaning we're outside the linear regime. Standard finite-difference
        # convergence criterion (Numerical Recipes §5.7).
        "linear_regime": relative_diff < 0.1,
    }


# =====================================================================
# Phase 3: A7 Test 1 — Monotonicity
# =====================================================================


def collect_per_head_gradient_data(
    attn_weights, gradients, baseline, mx
) -> dict[int, dict[int, dict]]:
    """Extract per-token α, ∂L/∂s_t, and ∂L/∂s_t / α_t for each (layer, head).

    This is the data-collection step. The actual A7 statistical tests
    (Spearman monotonicity against explicit r_t - R from compute_radial_projections,
    OLS β estimation) are performed in run_a7_tests().

    Excludes near-uniform heads where max(α_t) < 2/T.
    """
    token_ids = baseline["token_ids"]
    seq_len = len(token_ids)

    results: dict[int, dict[int, dict]] = {}

    for layer_idx in sorted(gradients.keys()):
        results[layer_idx] = {}
        w = attn_weights[layer_idx]  # [batch, heads, seq, seq]

        for head_idx in sorted(gradients[layer_idx].keys()):
            dLds = gradients[layer_idx][head_idx]  # list[float], length seq_len

            # Extract attention weights for the query position driving the loss
            # w[0, head_idx, _QUERY_POS, :] → attention distribution over keys
            alpha = [float(w[0, head_idx, _QUERY_POS, t]) for t in range(seq_len)]

            # Near-uniform exclusion: max(α_t) < 2/T.
            # Derivation: uniform attention gives α_t = 1/T for all t.
            # If no token exceeds 2/T, the head allocates less than 2×
            # uniform share to any single token — the attention pattern
            # has no selectivity and the radial structure (r_t - R) is
            # numerically dominated by noise.
            max_alpha = max(alpha)
            near_uniform_threshold = 2.0 / seq_len
            if max_alpha < near_uniform_threshold:
                results[layer_idx][head_idx] = {
                    "excluded": True,
                    "reason": f"near-uniform: max(α)={max_alpha:.4f} < 2/T={near_uniform_threshold:.4f}",
                    "pass_monotonicity": None,
                    "pass_delta_r": None,
                }
                continue

            # ∂L/∂s_t / α_t = ⟨g, w_t - δ⟩ (score gradient identity)
            grad_over_alpha = []
            for t in range(seq_len):
                if alpha[t] > _FLOAT32_MIN_NORMAL:
                    grad_over_alpha.append(dLds[t] / alpha[t])
                else:
                    grad_over_alpha.append(0.0)

            results[layer_idx][head_idx] = {
                "excluded": False,
                "alpha": alpha,
                "dLds": dLds,
                "grad_over_alpha": grad_over_alpha,
            }

    return results


def compute_radial_projections(
    model, tokenizer, text, baseline, attn_weights, backend, mx,
    *, include_vectors: bool = False,
) -> dict[int, dict[int, dict]]:
    """Compute r_t = ⟨w_t, ĥ⟩ and R = Σ α_t r_t for each (layer, head).

    w_t = output vector for token t = V[t] @ W_O^T (per head)
    ĥ = normalized hidden state at layer input

    When include_vectors=True, additionally stores:
        w_t_vectors: list of T lists, each [hidden_dim] — full w_t per token
        h_hat: [hidden_dim] — normalized hidden state unit vector
    This is opt-in to avoid allocation overhead for callers that only
    need the scalar projections (r_t, norms).
    """
    token_ids = baseline["token_ids"]
    input_ids = mx.array([token_ids])
    seq_len = len(token_ids)

    base = backend._resolve_model_base(model)
    h = base.embed_tokens(input_ids)

    radial: dict[int, dict[int, dict]] = {}

    for layer_idx, layer in enumerate(base.layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        is_attn = attn is not None and hasattr(attn, "q_proj")

        if is_attn and layer_idx in attn_weights:
            # Compute ĥ = h / ||h|| at this layer's input, at the query
            # position that drives the loss (consistent with perturbation target)
            h_last = h[0, _QUERY_POS, :]  # [hidden_dim]
            h_norm = float(mx.sqrt(mx.sum(h_last * h_last)))
            if h_norm > _EPS_F32:
                h_hat = h_last / h_norm
            else:
                h_hat = h_last

            # Get V projection: v = V_proj(layernorm(h)) → [batch, seq, v_dim]
            if hasattr(layer, "input_layernorm"):
                h_ln = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_ln = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_ln = layer.operator_norm(h)
            else:
                h_ln = h

            v_full = attn.v_proj(h_ln)  # [batch, seq, num_kv_heads * head_dim]

            num_heads = baseline["layer_info"][layer_idx]["num_heads"]
            head_dim = baseline["layer_info"][layer_idx]["head_dim"]
            num_kv_heads = (
                getattr(attn, "num_key_value_heads", None)
                or getattr(attn, "n_kv_heads", None)
                or num_heads
            )

            # Reshape V: [batch, seq, kv_heads, head_dim]
            v_reshaped = v_full.reshape(1, seq_len, num_kv_heads, head_dim)
            # GQA expansion
            if num_kv_heads < num_heads:
                repeats = num_heads // num_kv_heads
                v_reshaped = mx.repeat(v_reshaped, repeats, axis=2)
            # v_reshaped: [1, seq, num_heads, head_dim]

            # O projection weight: [hidden_dim, num_heads * head_dim] or transposed
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            W_O = o_proj.weight  # shape depends on framework

            radial[layer_idx] = {}
            for head_idx in range(num_heads):
                # v_t for each token: v_reshaped[0, t, head_idx, :] → [head_dim]
                # w_t = O_proj applied to head output
                # O_proj maps [num_heads * head_dim] → [hidden_dim]
                # Per head: w_t = W_O[:, head_idx*d:(head_idx+1)*d] @ v_t
                # But W_O layout varies. For MLX linear: y = x @ W.T
                # So W_O.weight is [hidden_dim, num_heads * head_dim]
                # Per-head slice: W_O_h = W_O[:, hd:hd+d]
                # w_t = W_O_h @ v_t = (v_t @ W_O_h.T) ... but we want [hidden_dim]
                # Actually: full output = concat(v_heads) @ W_O.T
                # Per head: w_t = W_O[:, h*d:(h+1)*d] @ v_t

                hd = head_idx * head_dim
                # W_O.weight shape: in MLX nn.Linear, weight is [out, in]
                # So W_O.weight is [hidden_dim, num_heads * head_dim]
                W_O_head = W_O[..., hd:hd + head_dim]  # [hidden_dim, head_dim]

                r_t_list = []
                norm_w_t_list = []
                norm_w_t_perp_list = []
                w_t_vectors = [] if include_vectors else None
                for t in range(seq_len):
                    v_t = v_reshaped[0, t, head_idx, :]  # [head_dim]
                    # w_t = W_O_head @ v_t → [hidden_dim]
                    w_t = W_O_head @ v_t
                    if include_vectors:
                        w_t_vectors.append(w_t.tolist())  # [hidden_dim] per token
                    # r_t = ⟨w_t, ĥ⟩
                    r_t = float(mx.sum(w_t * h_hat))
                    r_t_list.append(r_t)
                    # ||w_t|| and ||w_t^⊥|| = sqrt(||w_t||² - r_t²)
                    norm_w_t = float(mx.sqrt(mx.sum(w_t * w_t)))
                    norm_w_t_perp = math.sqrt(max(norm_w_t**2 - r_t**2, 0.0))
                    norm_w_t_list.append(norm_w_t)
                    norm_w_t_perp_list.append(norm_w_t_perp)

                # α from the query position driving the loss
                alpha = [float(attn_weights[layer_idx][0, head_idx, _QUERY_POS, t]) for t in range(seq_len)]
                R = sum(alpha[t] * r_t_list[t] for t in range(seq_len))

                head_data = {
                    "r_t": r_t_list,
                    "R": R,
                    "r_minus_R": [r_t_list[t] - R for t in range(seq_len)],
                    "norm_w_t": norm_w_t_list,
                    "norm_w_t_perp": norm_w_t_perp_list,
                }
                if include_vectors:
                    head_data["w_t_vectors"] = w_t_vectors  # list of T lists, each [hidden_dim]
                    head_data["h_hat"] = h_hat.tolist()       # [hidden_dim]
                radial[layer_idx][head_idx] = head_data

        # Forward through layer to advance h
        if is_attn:
            layer_mask = "causal"
        else:
            layer_mask = None
        result = layer(h, mask=layer_mask, cache=None)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

    return radial


# =====================================================================
# Phase 3 + 4: Statistical tests
# =====================================================================


def run_a7_tests(
    monotonicity_data, radial_data, gradients, attn_weights, baseline, mx
) -> dict[int, dict[int, dict]]:
    """Run A7 monotonicity test and ΔR sign test per (layer, head).

    Test 1 (Monotonicity): Spearman(∂L/∂s_t / α_t, -(r_t - R)) > 0
    Test 2 (ΔR sign): β > 0 from OLS(∂L/∂s_t / α_t ~ r_t - R), slope = -β
    """
    token_ids = baseline["token_ids"]
    seq_len = len(token_ids)

    results: dict[int, dict[int, dict]] = {}

    for layer_idx in sorted(monotonicity_data.keys()):
        results[layer_idx] = {}

        for head_idx in sorted(monotonicity_data[layer_idx].keys()):
            mono = monotonicity_data[layer_idx][head_idx]

            if mono.get("excluded", False):
                results[layer_idx][head_idx] = mono
                continue

            dLds = mono["dLds"]
            grad_over_alpha = mono["grad_over_alpha"]
            alpha = mono["alpha"]

            if layer_idx not in radial_data or head_idx not in radial_data[layer_idx]:
                results[layer_idx][head_idx] = {
                    "excluded": True,
                    "reason": "no radial data",
                    "pass_monotonicity": None,
                    "pass_delta_r": None,
                }
                continue

            rad = radial_data[layer_idx][head_idx]
            r_minus_R = rad["r_minus_R"]

            # Test 1: Spearman(∂L/∂s_t / α_t, -(r_t - R))
            # Under A7: ∂L/∂s_t / α_t = -β(r_t - R) + noise
            # So ∂L/∂s_t / α_t should be negatively correlated with (r_t - R)
            # Equivalently: positively correlated with -(r_t - R)
            neg_r_minus_R = [-x for x in r_minus_R]
            rho = spearman_rank_correlation(grad_over_alpha, neg_r_minus_R)
            p_val = permutation_test_p_value(grad_over_alpha, neg_r_minus_R)

            # Test 2: β from OLS
            # ∂L/∂s_t / α_t = -β(r_t - R) + noise
            # OLS slope of (∂L/∂s_t / α_t) on (r_t - R) = -β
            # So β = -slope
            slope = ols_slope(r_minus_R, grad_over_alpha)
            beta = -slope

            # ΔR = η β Σ α_t² (r_t - R)²
            # If β > 0, ΔR > 0 (since the sum is always positive)
            sum_alpha_sq_r_sq = sum(alpha[t] ** 2 * r_minus_R[t] ** 2 for t in range(seq_len))

            results[layer_idx][head_idx] = {
                "excluded": False,
                "rho": rho,
                "p_value": p_val,
                "beta": beta,
                "ols_slope": slope,
                "sum_alpha_sq_r_sq": sum_alpha_sq_r_sq,
                "pass_monotonicity": None,  # set after Holm correction
                "pass_delta_r": beta > 0,
                "r_minus_R": r_minus_R,
                "grad_over_alpha": grad_over_alpha,
            }

    return results


# =====================================================================
# Phase 5: Aggregate, Holm-Bonferroni, report
# =====================================================================


def aggregate_and_report(test_results: dict, model_name: str) -> dict:
    """Apply Holm-Bonferroni correction and generate summary."""

    # Collect all non-excluded (layer, head) pairs with p-values
    testable = []
    for layer_idx in sorted(test_results.keys()):
        for head_idx in sorted(test_results[layer_idx].keys()):
            r = test_results[layer_idx][head_idx]
            if not r.get("excluded", False) and r.get("p_value") is not None:
                testable.append((layer_idx, head_idx, r))

    n_tests = len(testable)
    if n_tests == 0:
        return {
            "model": model_name,
            "n_testable": 0,
            "n_pass_both": 0,
            "fraction_pass": 0.0,
            "overall_pass": False,
            "per_layer_head": {},
        }

    # Sort by p-value ascending for Holm step-down
    sorted_tests = sorted(testable, key=lambda x: x[2]["p_value"])

    # Holm-Bonferroni
    for rank_k, (l, h, r) in enumerate(sorted_tests):
        holm_threshold = _PRE_REGISTERED_ALPHA / (n_tests - rank_k)
        r["holm_threshold"] = holm_threshold
        r["holm_significant"] = r["p_value"] <= holm_threshold

    # Holm step-down: reject from smallest p until first non-rejection
    rejecting = True
    for l, h, r in sorted_tests:
        if rejecting and r["holm_significant"]:
            r["pass_monotonicity"] = True
        else:
            rejecting = False
            r["pass_monotonicity"] = False

    # Count passes
    n_pass_both = 0
    n_excluded = 0
    per_layer_head = {}

    for layer_idx in sorted(test_results.keys()):
        per_layer_head[layer_idx] = {}
        for head_idx in sorted(test_results[layer_idx].keys()):
            r = test_results[layer_idx][head_idx]
            if r.get("excluded", False):
                n_excluded += 1
                per_layer_head[layer_idx][head_idx] = {
                    "status": "EXCLUDED",
                    "reason": r.get("reason", ""),
                }
                continue

            pass_mono = r.get("pass_monotonicity", False)
            pass_dr = r.get("pass_delta_r", False)
            both = pass_mono and pass_dr

            if both:
                n_pass_both += 1

            per_layer_head[layer_idx][head_idx] = {
                "status": "PASS" if both else "FAIL",
                "rho": r.get("rho"),
                "p_value": r.get("p_value"),
                "holm_threshold": r.get("holm_threshold"),
                "beta": r.get("beta"),
                "pass_monotonicity": pass_mono,
                "pass_delta_r": pass_dr,
            }

    fraction_pass = n_pass_both / n_tests if n_tests > 0 else 0.0
    # Acceptance threshold: ≥ 80% of testable (layer, head) pairs pass both tests.
    # TODO: derive acceptance fraction from statistical power analysis or
    # architecture-dependent expected noise floor. 80% is a conventional
    # threshold chosen to allow minority heads with weak/degenerate signal
    # while requiring the mechanism to hold broadly. Not IEEE-derived.
    overall_pass = fraction_pass >= 0.80

    return {
        "model": model_name,
        "n_testable": n_tests,
        "n_excluded": n_excluded,
        "n_pass_both": n_pass_both,
        "fraction_pass": fraction_pass,
        "overall_pass": overall_pass,
        "per_layer_head": per_layer_head,
    }


# =====================================================================
# Main experiment runner
# =====================================================================


def run_experiment(model_name: str, model_path: str) -> dict:
    """Run full A7 validation on one model."""
    import mlx.core as mx
    import mlx_lm

    from modelcypher.backends.mlx_backend import MLXBackend

    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({model_path})")
    print(f"{'='*70}")

    backend = MLXBackend()
    model, tokenizer = mlx_lm.load(model_path)

    all_probe_results = []

    for probe_idx, text in enumerate(PROBE_TEXTS):
        print(f"\n--- Probe {probe_idx + 1}: {text!r} ---")

        # Phase 1: Baseline
        print("  Phase 1: Collecting baseline...")
        baseline = collect_baseline(model, tokenizer, text, backend, mx)
        print(f"    Baseline loss: {baseline['baseline_loss']:.6f}")
        print(f"    Attention layers: {sorted(baseline['attn_weights'].keys())}")
        print(f"    Sequence length: {len(baseline['token_ids'])}")

        # Sensitivity check on first attention layer, first head
        first_layer = sorted(baseline["attn_weights"].keys())[0]
        print(f"  Sensitivity check (L{first_layer}, H0)...")
        sens = sensitivity_check(
            model, tokenizer, text, baseline, backend, mx,
            layer_idx=first_layer, head_idx=0
        )
        print(f"    ε={_EPSILON:.2e}: ∂L/∂s = {sens['dLds_eps']:.6e}")
        print(f"    ε/10:           ∂L/∂s = {sens['dLds_eps_over_10']:.6e}")
        print(f"    10ε:            ∂L/∂s = {sens['dLds_10_eps']:.6e}")
        print(f"    Relative diff:  {sens['relative_diff']:.4f} {'(OK)' if sens['linear_regime'] else '(WARNING: nonlinear)'}")

        # Phase 2: Score gradients
        print("  Phase 2: Computing per-token score gradients...")
        gradients = compute_score_gradients(
            model, tokenizer, text, baseline, backend, mx
        )

        # Compute radial projections
        print("  Computing radial projections...")
        radial = compute_radial_projections(
            model, tokenizer, text, baseline, baseline["attn_weights"], backend, mx
        )

        # Phase 3: Collect per-head gradient data
        print("  Phase 3: Collecting per-head gradient data...")
        mono_data = collect_per_head_gradient_data(
            baseline["attn_weights"], gradients, baseline, mx
        )

        # Phase 3+4: Full A7 tests
        print("  Phase 4: A7 tests (β sign + ΔR)...")
        test_results = run_a7_tests(
            mono_data, radial, gradients, baseline["attn_weights"], baseline, mx
        )

        # Phase 5: Aggregate
        print("  Phase 5: Aggregation...")
        summary = aggregate_and_report(test_results, model_name)

        # Print per-(layer, head) results
        for layer_idx in sorted(summary["per_layer_head"].keys()):
            for head_idx in sorted(summary["per_layer_head"][layer_idx].keys()):
                r = summary["per_layer_head"][layer_idx][head_idx]
                if r["status"] == "EXCLUDED":
                    print(f"    L{layer_idx} H{head_idx}: EXCLUDED ({r['reason']})")
                else:
                    mono_str = "PASS" if r["pass_monotonicity"] else "FAIL"
                    dr_str = "PASS" if r["pass_delta_r"] else "FAIL"
                    print(
                        f"    L{layer_idx} H{head_idx}: {r['status']} "
                        f"(ρ={r['rho']:+.3f}, p={r['p_value']:.4f}, "
                        f"β={r['beta']:+.4f}, mono={mono_str}, ΔR={dr_str})"
                    )

        print(f"\n  Probe {probe_idx + 1} summary: {summary['n_pass_both']}/{summary['n_testable']} pass "
              f"({summary['fraction_pass']:.1%}), overall={'PASS' if summary['overall_pass'] else 'FAIL'}")

        all_probe_results.append({
            "probe_text": text,
            "baseline_loss": baseline["baseline_loss"],
            "seq_len": len(baseline["token_ids"]),
            "sensitivity": sens,
            "summary": summary,
        })

    # Cross-probe summary
    n_pass = sum(1 for r in all_probe_results if r["summary"]["overall_pass"])
    n_total = len(all_probe_results)

    return {
        "model": model_name,
        "model_path": model_path,
        "probes": all_probe_results,
        "probes_passing": n_pass,
        "probes_total": n_total,
        "overall_pass": n_pass == n_total,
    }


def main():
    print("A7 Assumption Validator")
    print("Radial-dominant downstream gradient test")
    print(f"ε = sqrt(eps_bf16) = {_EPSILON:.6e}")
    print(f"α = {_PRE_REGISTERED_ALPHA}")
    print(f"Permutations: {_N_PERMUTATIONS}")
    print(f"Probes: {len(PROBE_TEXTS)}")

    # Check volume
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"ERROR: Model not found: {path}")
            print("Is the external volume mounted?")
            sys.exit(1)

    results = []
    for name, path in MODELS.items():
        result = run_experiment(name, path)
        results.append(result)

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "PASS (A7 NOT FALSIFIED)" if r["overall_pass"] else "FAIL (A7 FALSIFIED)"
        print(f"  {r['model']}: {status} ({r['probes_passing']}/{r['probes_total']} probes)")

    # Per CLAUDE.md: "No mixed-model narrative"
    all_pass = all(r["overall_pass"] for r in results)
    all_fail = all(not r["overall_pass"] for r in results)

    n_models = len(results)
    n_pass_models = sum(1 for r in results if r["overall_pass"])
    n_fail_models = n_models - n_pass_models

    if all_pass:
        print(f"\nAll {n_models} models: A7 NOT FALSIFIED. D3.3 applicable.")
    elif all_fail:
        print(f"\nAll {n_models} models: A7 FALSIFIED. D3.3 NOT applicable.")
    else:
        print(
            f"\nMIXED RESULT: {n_pass_models}/{n_models} pass, {n_fail_models}/{n_models} fail. "
            "This indicates mechanism underspecification — "
            "the observable needs an architecture_state term. "
            "See FIRST_PRINCIPLES_REVIEW_PROTOCOL.md."
        )

    # Save results
    output_dir = Path("results/a7_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full data for reproducibility)
    json_path = output_dir / "a7_results.json"
    # Sanitize for JSON (remove non-serializable items)
    json_results = []
    for r in results:
        jr = {
            "model": r["model"],
            "model_path": r["model_path"],
            "overall_pass": r["overall_pass"],
            "probes_passing": r["probes_passing"],
            "probes_total": r["probes_total"],
            "probes": [],
        }
        for pr in r["probes"]:
            jp = {
                "probe_text": pr["probe_text"],
                "baseline_loss": pr["baseline_loss"],
                "seq_len": pr["seq_len"],
                "sensitivity": pr["sensitivity"],
                "summary": {
                    "model": pr["summary"]["model"],
                    "n_testable": pr["summary"]["n_testable"],
                    "n_excluded": pr["summary"].get("n_excluded", 0),
                    "n_pass_both": pr["summary"]["n_pass_both"],
                    "fraction_pass": pr["summary"]["fraction_pass"],
                    "overall_pass": pr["summary"]["overall_pass"],
                    "per_layer_head": {
                        str(l): {
                            str(h): v
                            for h, v in heads.items()
                        }
                        for l, heads in pr["summary"]["per_layer_head"].items()
                    },
                },
            }
            jr["probes"].append(jp)
        json_results.append(jr)

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Human-readable summary
    txt_path = output_dir / "a7_validation.txt"
    lines = [
        "A7 Assumption Validation Results",
        f"ε = {_EPSILON:.6e} (sqrt(eps_bf16), IEEE 754)",
        f"α = {_PRE_REGISTERED_ALPHA}",
        f"Acceptance: ≥80% of (layer, head) pairs pass both tests",
        "",
    ]
    for r in results:
        status = "NOT FALSIFIED" if r["overall_pass"] else "FALSIFIED"
        lines.append(f"{r['model']}: A7 {status}")
        for pr in r["probes"]:
            s = pr["summary"]
            lines.append(f"  Probe: {pr['probe_text']!r}")
            lines.append(f"    Pass: {s['n_pass_both']}/{s['n_testable']} ({s['fraction_pass']:.1%})")
            for l in sorted(s["per_layer_head"].keys()):
                for h in sorted(s["per_layer_head"][l].keys()):
                    v = s["per_layer_head"][l][h]
                    if v["status"] == "EXCLUDED":
                        lines.append(f"    L{l} H{h}: EXCLUDED")
                    else:
                        lines.append(
                            f"    L{l} H{h}: {v['status']} "
                            f"(ρ={v['rho']:+.3f}, p={v['p_value']:.4f}, β={v['beta']:+.4f})"
                        )
        lines.append("")

    all_pass_final = all(r["overall_pass"] for r in results)
    all_fail_final = all(not r["overall_pass"] for r in results)
    if all_pass_final:
        lines.append(f"CONCLUSION: A7 NOT FALSIFIED across all {len(results)} models.")
        lines.append("D3.3 status: [PROVEN under A7] → [PROVEN, A7 VALIDATED]")
    elif all_fail_final:
        lines.append(f"CONCLUSION: A7 FALSIFIED across all {len(results)} models.")
        lines.append("D3.3 status: [PROVEN under A7] → [PROVEN under A7, A7 FALSIFIED — D3.3 NOT APPLICABLE]")
    else:
        n_p = sum(1 for r in results if r["overall_pass"])
        lines.append(f"CONCLUSION: MIXED RESULT ({n_p}/{len(results)} pass) — mechanism underspecification.")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved to {txt_path}")


if __name__ == "__main__":
    main()
