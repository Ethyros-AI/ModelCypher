#!/usr/bin/env python3
"""Analysis: Curvature Accumulation Model.

Tests whether per-layer curvature change can be decomposed into attention and MLP
contributions, and whether this decomposition closes the weak link in the causal chain:

    Attention entropy → curvature (r=0.507, only 25% variance explained)
    Cumulative curvature → ID (r=0.821, 67% explained)

If delta_curvature = f(attention_entropy) + g(mlp_contribution), and this model
explains >80% of curvature variance, it completes the causal chain with no free
parameters.

Measurements per layer:
    1. h_pre_attn: hidden state before attention (after input_layernorm)
    2. h_post_attn: hidden state after attention + residual
    3. h_post_mlp: hidden state after MLP + residual (= layer output)
    4. Attention curvature: angular change from h_in to h_post_attn
    5. MLP curvature: angular change from h_post_attn to h_post_mlp
    6. Total curvature: angular change from h_in to h_out
    7. Attention entropy: from attention weights (if extractable)
    8. ID (TwoNN): from stacked last-token representations

Angular change = arccos(cosine_similarity), measuring directional shift in radians.
This is the geodesic distance on the unit sphere — a natural curvature measure.

Usage:
    poetry run python scripts/curvature_accumulation_analysis.py
    poetry run python scripts/curvature_accumulation_analysis.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")


def _resolve_existing_path(*candidates: str) -> str:
    """Return the first existing path from candidates, else the first candidate."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024,
        "architecture": "lfm2",
    },
    "LFM2-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-1.2B-bf16",
        "L": 16, "d": 2048,
        "architecture": "lfm2",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
    },
    "Qwen3-8B": {
        "path": _resolve_existing_path(
            f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
            f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        ),
        "L": 36, "d": 4096,
        "architecture": "qwen3",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
    },
    "Qwen3-1.7B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-1.7B-MLX-bf16",
        "L": 28, "d": 2048,
        "architecture": "qwen3",
    },
}

# 6 categories × 10 probes = 60 total — matches signal_propagation experiment
PROBE_CATEGORIES = {
    "retrieval": [
        "The capital of France is",
        "Who wrote Romeo and Juliet?",
        "The chemical symbol for water is",
        "The largest planet in our solar system is",
        "The speed of light in a vacuum is approximately",
        "The first president of the United States was",
        "The boiling point of water at sea level is",
        "The chemical formula for table salt is",
        "The tallest mountain on Earth is",
        "The currency of Japan is",
    ],
    "arithmetic": [
        "What is 347 + 528?",
        "What is 15 * 23?",
        "What is 1024 / 16?",
        "What is 99 - 37?",
        "What is 8 * 7 + 13?",
        "What is 256 + 384 - 100?",
        "What is 12 * 12?",
        "What is 999 - 456?",
        "What is 50 * 20 + 1?",
        "What is 128 / 4?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
        "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
        "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "If you rearrange CIFAIPC, you get the name of a country. What is it?",
        "A train leaves A at 60 mph, another leaves B at 80 mph toward A, 280 miles apart. When do they meet?",
        "What comes next: 2, 6, 12, 20, 30, ?",
        "Three friends split $90 unequally. A gets twice what B gets. B gets twice what C gets. How much does C get?",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Describe a sunset over the mountains in one vivid sentence.",
        "Write a short poem about the passage of time.",
        "Describe the taste of your favorite food using only three words.",
        "Write a one-sentence story with a twist ending.",
        "Describe the sound of rain on a tin roof.",
        "Write a metaphor for loneliness.",
        "Describe the color blue to someone who has never seen it.",
        "Write a two-line dialogue between the sun and the moon.",
        "Describe the feeling of flying in one sentence.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to compute Fibonacci up to n terms.",
        "Write a Python function to find the max element without max().",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python one-liner to flatten a nested list.",
        "Write a Python function to sort a list using bubble sort.",
        "Write a Python function to count words in a string.",
        "Write a Python function to compute factorial recursively.",
        "Write a Python function to merge two sorted lists.",
    ],
    "narrative": [
        "Once upon a time in a faraway kingdom, there lived a",
        "The old lighthouse keeper watched the storm approach from",
        "In the year 2150, humanity had finally achieved",
        "She opened the letter and read the first line:",
        "The forest was silent except for the sound of",
        "He had been walking for three days when he finally saw",
        "The library contained a secret that no one had discovered for",
        "As the last leaf fell from the ancient oak tree,",
        "The musician played a melody that made everyone in the room",
        "Deep beneath the ocean, a creature stirred for the first time in",
    ],
}


# =============================================================================
# Core Measurements
# =============================================================================


def angular_change(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angular change between two vectors in radians.

    arccos(cosine_similarity) — geodesic distance on the unit sphere.
    Returns 0 for parallel, pi/2 for orthogonal, pi for antiparallel.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_sim = np.dot(v1, v2) / (n1 * n2)
    # Clamp for numerical stability
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(np.arccos(cos_sim))


def try_compute_attn_entropy(self_attn, normed: "mx.array") -> float | None:
    """Compute mean attention entropy H(α) at the last query position.

    Recomputes Q/K projections from the already-normed input to extract softmax weights.
    Returns mean over heads in nats, or None if extraction fails (SSM layers, missing
    projections, unsupported architecture).

    Does NOT call self_attn() — uses q_proj and k_proj directly to avoid duplicate output
    computation. Safe to call before the main self_attn() call in the decomposition branch.
    """
    import mlx.core as mx

    q_proj = getattr(self_attn, "q_proj", None)
    k_proj = getattr(self_attn, "k_proj", None)
    if q_proj is None or k_proj is None:
        return None

    try:
        q = q_proj(normed)  # [1, T, n_heads * head_dim]
        k = k_proj(normed)  # [1, T, n_kv_heads * head_dim]
        mx.eval(q, k)

        T = q.shape[1]
        q_out = int(q.shape[-1])
        k_out = int(k.shape[-1])

        def _infer_head_dim_from_norm(module_names: tuple[str, ...]) -> int | None:
            for name in module_names:
                mod = getattr(self_attn, name, None)
                if mod is None:
                    continue
                w = getattr(mod, "weight", None)
                if w is None:
                    continue
                if len(w.shape) == 0:
                    continue
                # Some implementations store [head_dim], others [n_heads, head_dim].
                dim = int(w.shape[-1])
                if dim > 0:
                    return dim
            return None

        # Robust head config resolution:
        # 1) q_norm/k_norm weight shape (most reliable on Qwen variants)
        # 2) explicit module head_dim
        # 3) n_heads attributes fallback
        inferred_head_dim = _infer_head_dim_from_norm(("q_norm", "q_layernorm"))
        if inferred_head_dim is None:
            inferred_head_dim = int(getattr(self_attn, "head_dim", 0) or 0) or None

        n_heads_attr = getattr(self_attn, "n_heads", None) or getattr(self_attn, "num_heads", None)
        n_kv_heads_attr = (
            getattr(self_attn, "n_kv_heads", None)
            or getattr(self_attn, "num_key_value_heads", None)
            or n_heads_attr
        )

        if inferred_head_dim is not None and inferred_head_dim > 0:
            if q_out % inferred_head_dim != 0 or k_out % inferred_head_dim != 0:
                return None
            n_heads = q_out // inferred_head_dim
            n_kv_heads = k_out // inferred_head_dim
            head_dim = inferred_head_dim
        else:
            if n_heads_attr is None:
                return None
            n_heads = int(n_heads_attr)
            n_kv_heads = int(n_kv_heads_attr or n_heads)
            if n_heads <= 0 or n_kv_heads <= 0:
                return None
            if q_out % n_heads != 0:
                return None
            head_dim = q_out // n_heads

        if k_out % n_kv_heads != 0:
            return None
        k_head_dim = k_out // n_kv_heads
        if k_head_dim != head_dim:
            return None

        # Reshape to [1, T, n_heads, head_dim] before optional post-projection norms.
        q = q.reshape(1, T, n_heads, head_dim)
        k = k.reshape(1, T, n_kv_heads, k_head_dim)

        # Apply post-projection norm layers when present (Qwen/LFM2 variants).
        q_norm = getattr(self_attn, "q_norm", None) or getattr(self_attn, "q_layernorm", None)
        k_norm = getattr(self_attn, "k_norm", None) or getattr(self_attn, "k_layernorm", None)
        if callable(q_norm):
            q = q_norm(q)
        if callable(k_norm):
            k = k_norm(k)

        q = q.transpose(0, 2, 1, 3)  # [1, n_heads, T, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [1, n_kv_heads, T, head_dim]

        # Match model forward pass: apply RoPE before attention scoring when available.
        rope = getattr(self_attn, "rope", None)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # GQA: repeat k heads to match q heads
        if n_kv_heads < n_heads:
            if n_heads % n_kv_heads != 0:
                return None
            reps = n_heads // n_kv_heads
            k = mx.repeat(k, reps, axis=1)

        # Scaled dot-product: [1, n_heads, T, T]
        scale = float(getattr(self_attn, "scale", 1.0 / math.sqrt(float(head_dim))))
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask: upper triangle = -inf (last token attends to all previous)
        causal_mask = mx.triu(mx.ones((T, T), dtype=mx.float32), k=1) * -1e9
        scores = scores + causal_mask[None, None, :, :]

        # Softmax over key dimension: [1, n_heads, T, T]
        alpha = mx.softmax(scores, axis=-1)

        # Extract last query position: [n_heads, T]
        alpha_last = alpha[0, :, -1, :]
        mx.eval(alpha_last)
        alpha_np = np.array(alpha_last.tolist(), dtype=np.float64)

        # Entropy per head: H = -Σ_k α_k log α_k (nats)
        eps = 1e-12
        log_alpha = np.log(alpha_np + eps)
        entropy_per_head = -np.sum(alpha_np * log_alpha, axis=-1)  # [n_heads]

        return float(np.mean(entropy_per_head))
    except Exception:
        return None


def collect_sublayer_activations(
    model, tokenizer, prompts: list[str], num_layers: int, backend
) -> list[dict]:
    """Collect per-sublayer activations for curvature decomposition.

    For each layer, collects:
        h_in: input to the layer [N, d]
        h_post_attn: after attention + residual [N, d]
        h_out: after MLP + residual = layer output [N, d]

    This requires manually stepping through the layer's sub-components.
    Falls back to whole-layer delta if sub-components are not accessible.
    """
    import mlx.core as mx

    base = getattr(model, "model", model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # Per-layer collectors
    layer_h_in = [[] for _ in range(num_layers)]
    layer_h_post_attn = [[] for _ in range(num_layers)]
    layer_h_out = [[] for _ in range(num_layers)]
    layer_h_attn_entropy = [[] for _ in range(num_layers)]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden

            # Per-layer mask routing (LFM2 compatibility)
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            # Try to decompose into attention + MLP sub-steps
            h_post_attn = None
            attn_entropy = None
            input_norm = getattr(layer, "input_layernorm", None)
            self_attn = getattr(layer, "self_attn", None)
            post_attn_norm = getattr(layer, "post_attention_layernorm", None)
            mlp = getattr(layer, "mlp", None)

            if input_norm is not None and self_attn is not None and mlp is not None:
                # Standard transformer: norm → attn → residual → norm → mlp → residual
                try:
                    normed = input_norm(h_in)
                    # Entropy extracted before main attn call — reuses normed,
                    # does not duplicate the full attention output computation.
                    attn_entropy = try_compute_attn_entropy(self_attn, normed)
                    attn_out = self_attn(normed, mask=layer_mask)
                    # Handle tuple returns (output, cache)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_post_attn = h_in + attn_out  # Residual connection

                    if post_attn_norm is not None:
                        normed2 = post_attn_norm(h_post_attn)
                    else:
                        normed2 = h_post_attn
                    mlp_out = mlp(normed2)
                    h_out = h_post_attn + mlp_out  # Second residual
                except Exception:
                    # Fallback: use whole layer
                    h_post_attn = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            else:
                # Non-standard layer (SSM, conv, etc.) — whole-layer only
                try:
                    h_out = layer(h_in, mask=layer_mask)
                except (TypeError, ValueError):
                    try:
                        h_out = layer(h_in, layer_mask)
                    except (TypeError, ValueError):
                        h_out = layer(h_in)

            # Last-token representation [1, d] → [d]
            h_in_last = h_in[:, -1, :].astype(mx.float32)
            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_in_last, h_out_last)

            layer_h_in[i].append(np.array(h_in_last[0].tolist(), dtype=np.float32))
            layer_h_out[i].append(np.array(h_out_last[0].tolist(), dtype=np.float32))

            if h_post_attn is not None:
                h_pa_last = h_post_attn[:, -1, :].astype(mx.float32)
                mx.eval(h_pa_last)
                layer_h_post_attn[i].append(
                    np.array(h_pa_last[0].tolist(), dtype=np.float32)
                )
            else:
                # No decomposition available — mark as None
                layer_h_post_attn[i].append(None)

            # Entropy stored per-prompt (None for SSM/non-standard layers)
            layer_h_attn_entropy[i].append(attn_entropy)

            hidden = h_out

        mx.eval(hidden)

    # Build result
    result = []
    for i in range(num_layers):
        has_decomposition = all(x is not None for x in layer_h_post_attn[i])
        result.append({
            "h_in": np.stack(layer_h_in[i]),
            "h_out": np.stack(layer_h_out[i]),
            "h_post_attn": np.stack(layer_h_post_attn[i]) if has_decomposition else None,
            "has_decomposition": has_decomposition,
            "h_attn_entropy": layer_h_attn_entropy[i],
        })

    return result


def compute_curvature_decomposition(
    sublayer_acts: list[dict],
) -> list[dict]:
    """Compute per-layer curvature decomposition.

    For each layer, computes:
        total_curvature: mean angular change from h_in to h_out
        attn_curvature: mean angular change from h_in to h_post_attn (if available)
        mlp_curvature: mean angular change from h_post_attn to h_out (if available)
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    measurements = []

    for i, act in enumerate(sublayer_acts):
        h_in = act["h_in"]  # [N, d]
        h_out = act["h_out"]  # [N, d]
        n_probes = h_in.shape[0]

        # Total curvature: mean angular change h_in → h_out
        total_angles = []
        for j in range(n_probes):
            total_angles.append(angular_change(h_in[j], h_out[j]))
        total_curvature = float(np.mean(total_angles))

        # Delta norm ratio (for comparison with alpha from Exp 1)
        delta = h_out - h_in
        delta_norms = np.linalg.norm(delta, axis=1)
        h_in_norms = np.linalg.norm(h_in, axis=1)
        mean_alpha = float(np.mean(delta_norms / np.maximum(h_in_norms, 1e-10)))

        # ID via TwoNN
        n_samples = h_out.shape[0]
        if n_samples < IntrinsicDimension.local_dimension_min_samples():
            id_val = float("nan")
        else:
            try:
                estimate = IntrinsicDimension.compute_two_nn(h_out)
                id_val = estimate.intrinsic_dimension
            except Exception:
                id_val = float("nan")

        # Attention entropy: mean over non-None probe values
        raw_ent = act.get("h_attn_entropy", [])
        valid_ent = [e for e in raw_ent if e is not None and not np.isnan(e)]
        mean_h_attn = float(np.mean(valid_ent)) if valid_ent else None

        layer_result = {
            "layer_idx": i,
            "total_curvature": total_curvature,
            "mean_alpha": mean_alpha,
            "id_two_nn": id_val,
            "mean_h_attn": mean_h_attn,
            "attn_curvature": None,
            "mlp_curvature": None,
            "attn_fraction": None,
        }

        if act["has_decomposition"]:
            h_post_attn = act["h_post_attn"]

            attn_angles = []
            mlp_angles = []
            for j in range(n_probes):
                attn_angles.append(angular_change(h_in[j], h_post_attn[j]))
                mlp_angles.append(angular_change(h_post_attn[j], h_out[j]))

            attn_curv = float(np.mean(attn_angles))
            mlp_curv = float(np.mean(mlp_angles))

            layer_result["attn_curvature"] = attn_curv
            layer_result["mlp_curvature"] = mlp_curv
            layer_result["attn_fraction"] = (
                attn_curv / (attn_curv + mlp_curv)
                if (attn_curv + mlp_curv) > 1e-10
                else 0.5
            )

        measurements.append(layer_result)

    return measurements


def compute_correlations(measurements: list[dict]) -> dict:
    """Compute Spearman correlations between curvature components and ID."""
    from scipy import stats

    # Filter to layers with both decomposition and valid ID
    layers_with_decomp = [
        m for m in measurements
        if m["attn_curvature"] is not None and not np.isnan(m["id_two_nn"])
    ]

    if len(layers_with_decomp) < 5:
        # Also compute for all layers with valid ID (even without decomposition)
        layers_with_id = [m for m in measurements if not np.isnan(m["id_two_nn"])]
        return {
            "n_layers_with_decomposition": len(layers_with_decomp),
            "n_layers_with_valid_id": len(layers_with_id),
            "note": f"Insufficient layers for correlation ({len(layers_with_decomp)} with decomp+ID, need >=5)",
        }

    attn_curv = [m["attn_curvature"] for m in layers_with_decomp]
    mlp_curv = [m["mlp_curvature"] for m in layers_with_decomp]
    total_curv = [m["total_curvature"] for m in layers_with_decomp]
    ids = [m["id_two_nn"] for m in layers_with_decomp]

    # Cumulative curvature (integral of per-layer curvature)
    cum_total = np.cumsum(total_curv).tolist()
    cum_attn = np.cumsum(attn_curv).tolist()

    # ID gradient (change in ID per layer)
    id_gradient = np.gradient(ids).tolist()

    result = {
        "n_layers_with_decomposition": len(layers_with_decomp),
    }

    def safe_spearman(x, y, name):
        try:
            r, p = stats.spearmanr(x, y)
            result[f"spearman_{name}"] = float(r) if not np.isnan(r) else 0.0
            result[f"p_{name}"] = float(p) if not np.isnan(p) else 1.0
        except Exception:
            result[f"spearman_{name}"] = 0.0
            result[f"p_{name}"] = 1.0

    # Key correlations
    safe_spearman(attn_curv, id_gradient, "attn_curv_vs_id_gradient")
    safe_spearman(mlp_curv, id_gradient, "mlp_curv_vs_id_gradient")
    safe_spearman(total_curv, id_gradient, "total_curv_vs_id_gradient")
    safe_spearman(cum_total, ids, "cum_total_vs_id")
    safe_spearman(cum_attn, ids, "cum_attn_vs_id")

    # Attention fraction trajectory
    attn_fracs = [m["attn_fraction"] for m in layers_with_decomp]
    safe_spearman(attn_fracs, ids, "attn_fraction_vs_id")

    # Mean attention vs MLP curvature contribution
    result["mean_attn_curvature"] = float(np.mean(attn_curv))
    result["mean_mlp_curvature"] = float(np.mean(mlp_curv))
    result["mean_attn_fraction"] = float(np.mean(attn_fracs))

    return result


# =============================================================================
# Falsifier Tests (entropy-curvature-derivation.md F1, F3, F4, F5)
# =============================================================================


def run_falsifier_tests(all_results: list[dict], n_permutations: int = 500) -> dict:
    """Run falsifier tests F1, F3, F4, F5 from entropy-curvature-derivation.md.

    F2 (geometry-conditioned, requires E_mix proxy) is not yet implemented.
    Note: H in these tests is attention-weight entropy from QK softmax weights, not
    output-logit entropy. These operators should not be conflated.

    Args:
        all_results: list of per-model result dicts from run_single_model.
        n_permutations: permutation count for F4 null distribution.

    Returns:
        Dict with per-falsifier pass/fail results and supporting statistics.
    """
    from scipy import stats

    # Collect cross-model pool: one row per (model, layer) with H and curvature values.
    pool = []
    for r in all_results:
        family = r["architecture"]
        n_layers = r["num_layers"]
        for m in r["measurements"]:
            h_val = m.get("mean_h_attn")
            theta_attn = m.get("attn_curvature")
            theta_mlp = m.get("mlp_curvature")
            theta_total = m.get("total_curvature")
            depth_frac = m["layer_idx"] / max(n_layers - 1, 1)

            # Require H and both sublayer curvatures for the pool.
            if (
                h_val is not None
                and not np.isnan(h_val)
                and theta_attn is not None
                and not np.isnan(theta_attn)
                and theta_mlp is not None
                and not np.isnan(theta_mlp)
            ):
                pool.append({
                    "H": h_val,
                    "theta_attn": theta_attn,
                    "theta_mlp": theta_mlp,
                    "theta_total": theta_total or 0.0,
                    "theta_attn_sq": theta_attn ** 2,
                    "depth_frac": depth_frac,
                    "family": family,
                    "model": r["model_name"],
                    "layer": m["layer_idx"],
                })

    n_pool = len(pool)
    out: dict = {
        "n_observations": n_pool,
        "f2_status": "NOT_IMPLEMENTED: requires E_mix proxy (V, W_O matrix access)",
    }

    if n_pool < 10:
        out["status"] = f"INSUFFICIENT_DATA: need >=10 obs with entropy+decomp, have {n_pool}"
        return out

    H = np.array([p["H"] for p in pool])
    theta_attn_sq = np.array([p["theta_attn_sq"] for p in pool])
    theta_attn = np.array([p["theta_attn"] for p in pool])
    theta_mlp = np.array([p["theta_mlp"] for p in pool])
    depth_frac = np.array([p["depth_frac"] for p in pool])

    def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Remove linear effect of x from y via OLS residual."""
        X = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ b

    H_resid = residualize(H, depth_frac)
    theta_sq_resid = residualize(theta_attn_sq, depth_frac)

    # -------------------------------------------------------------------------
    # F1: Sign falsifier — regression coefficient of H on θ_attn² (depth-controlled)
    # Derivation prediction: slope >= 0 (higher entropy → higher squared curvature).
    # Fail: significant negative slope (p < 0.05) in >=2 families.
    # -------------------------------------------------------------------------
    try:
        slope, _, r_val, p_val, _ = stats.linregress(H_resid, theta_sq_resid)
        r_sp, p_sp = stats.spearmanr(H_resid, theta_sq_resid)
        passes_f1 = float(slope) >= 0 and float(p_val) < 0.05
        out["f1_sign_falsifier"] = {
            "slope_H_on_theta_attn_sq": float(slope),
            "r_squared_ols": float(r_val ** 2),
            "spearman_r": float(r_sp),
            "p_value_ols": float(p_val),
            "passes": passes_f1,
            "derivation_prediction": "slope >= 0 (higher H -> higher theta_attn^2)",
            "fail_criterion": "significant negative slope in >=2 families",
        }
    except Exception as e:
        out["f1_sign_falsifier"] = {"error": str(e), "passes": False}

    # -------------------------------------------------------------------------
    # F3: Attention-vs-MLP falsifier (architecture-qualified)
    # Qualified prediction:
    #   - LFM2 family: |corr(H, θ_attn)| > |corr(H, θ_mlp)| is expected.
    #   - Other families: no universal dominance direction is assumed.
    # -------------------------------------------------------------------------
    try:
        r_h_attn, p_h_attn = stats.spearmanr(H, theta_attn)
        r_h_mlp, p_h_mlp = stats.spearmanr(H, theta_mlp)
        family_results: dict = {}
        lfm2_checks: list[bool] = []

        families = sorted({p["family"] for p in pool})
        for fam in families:
            fam_pool = [p for p in pool if p["family"] == fam]
            if len(fam_pool) < 5:
                continue
            h_f = np.array([p["H"] for p in fam_pool])
            ta_f = np.array([p["theta_attn"] for p in fam_pool])
            tm_f = np.array([p["theta_mlp"] for p in fam_pool])
            r_attn_f, p_attn_f = stats.spearmanr(h_f, ta_f)
            r_mlp_f, p_mlp_f = stats.spearmanr(h_f, tm_f)

            dom = "attention" if abs(float(r_attn_f)) > abs(float(r_mlp_f)) else "mlp_or_tie"
            family_results[fam] = {
                "spearman_H_vs_theta_attn": float(r_attn_f),
                "p_H_vs_theta_attn": float(p_attn_f),
                "spearman_H_vs_theta_mlp": float(r_mlp_f),
                "p_H_vs_theta_mlp": float(p_mlp_f),
                "dominant_sublayer": dom,
                "n_layers": len(fam_pool),
            }
            if fam == "lfm2":
                lfm2_checks.append(dom == "attention")

        passes_f3 = bool(lfm2_checks) and all(lfm2_checks)
        out["f3_attn_vs_mlp"] = {
            "spearman_H_vs_theta_attn": float(r_h_attn),
            "p_H_vs_theta_attn": float(p_h_attn),
            "spearman_H_vs_theta_mlp": float(r_h_mlp),
            "p_H_vs_theta_mlp": float(p_h_mlp),
            "per_family": family_results,
            "lfm2_attention_dominates": passes_f3,
            "passes": passes_f3,
            "derivation_prediction": "LFM2-qualified: |corr(H, theta_attn)| > |corr(H, theta_mlp)| only for lfm2 family",
            "fail_criterion": "Any lfm2 run with |corr(H,theta_attn)| <= |corr(H,theta_mlp)|; non-lfm2 direction is unresolved",
        }
    except Exception as e:
        out["f3_attn_vs_mlp"] = {"error": str(e), "passes": False}

    # -------------------------------------------------------------------------
    # F4: Permutation falsifier — real fit must exceed permutation null
    # Permutes H within depth quintile strata to preserve depth-entropy correlation.
    # Derivation prediction: real |r| > 95th percentile of null.
    # Fail: real fit within null envelope.
    # -------------------------------------------------------------------------
    try:
        real_stat = abs(float(stats.spearmanr(H_resid, theta_sq_resid)[0]))

        depth_strata = np.digitize(
            depth_frac, np.quantile(depth_frac, [0.2, 0.4, 0.6, 0.8])
        )
        rng = np.random.default_rng(seed=0)  # Fixed for reproducibility
        perm_stats = []
        for _ in range(n_permutations):
            H_perm = H_resid.copy()
            for stratum in range(5):
                idx = np.where(depth_strata == stratum)[0]
                if len(idx) > 1:
                    H_perm[idx] = rng.permutation(H_perm[idx])
            r_perm, _ = stats.spearmanr(H_perm, theta_sq_resid)
            perm_stats.append(abs(float(r_perm)))

        perm_arr = np.array(perm_stats)
        pct_in_null = float(np.mean(perm_arr <= real_stat)) * 100
        passes_f4 = pct_in_null >= 95.0
        out["f4_permutation_falsifier"] = {
            "real_abs_spearman": real_stat,
            "null_mean": float(np.mean(perm_arr)),
            "null_95th_pct": float(np.percentile(perm_arr, 95)),
            "percentile_in_null": pct_in_null,
            "n_permutations": n_permutations,
            "passes": passes_f4,
            "derivation_prediction": "real |r| beyond 95th percentile of permutation null",
            "fail_criterion": "real fit within null envelope -> coincidence not ruled out",
        }
    except Exception as e:
        out["f4_permutation_falsifier"] = {"error": str(e), "passes": False}

    # -------------------------------------------------------------------------
    # F5: Scale/Family falsifier — same sign of H coefficient across families.
    # Derivation prediction: same-sign H coefficient (magnitude may differ).
    # Fail: sign flips in >=2 families with p < 0.05 (indicates missing arch term).
    # -------------------------------------------------------------------------
    try:
        families = list({p["family"] for p in pool})
        family_results: dict = {}
        sign_flips = 0

        for fam in families:
            fam_pool = [p for p in pool if p["family"] == fam]
            if len(fam_pool) < 5:
                continue

            H_f = np.array([p["H"] for p in fam_pool])
            theta_sq_f = np.array([p["theta_attn_sq"] for p in fam_pool])
            depth_f = np.array([p["depth_frac"] for p in fam_pool])

            H_f_res = residualize(H_f, depth_f)
            theta_sq_f_res = residualize(theta_sq_f, depth_f)

            if np.std(H_f_res) < 1e-10 or np.std(theta_sq_f_res) < 1e-10:
                continue

            slope_f, _, _, p_f, _ = stats.linregress(H_f_res, theta_sq_f_res)
            r_f, _ = stats.spearmanr(H_f_res, theta_sq_f_res)
            family_results[fam] = {
                "slope": float(slope_f),
                "spearman_r": float(r_f),
                "p_value": float(p_f),
                "n_layers": len(fam_pool),
            }
            if float(slope_f) < 0 and float(p_f) < 0.05:
                sign_flips += 1

        passes_f5 = sign_flips < 2
        out["f5_scale_family_falsifier"] = {
            "per_family": family_results,
            "n_significant_sign_flips": sign_flips,
            "passes": passes_f5,
            "derivation_prediction": "same sign of H coefficient across families",
            "fail_criterion": "sign flips in >=2 families -> claim is MECHANISM_UNDERSPECIFIED",
        }
    except Exception as e:
        out["f5_scale_family_falsifier"] = {"error": str(e), "passes": False}

    return out


# =============================================================================
# Model Runner
# =============================================================================


def run_single_model(
    model_name: str, model_info: dict, probes: list[str], backend
) -> dict:
    """Run curvature analysis for one model."""
    logger.info(f"Loading model: {model_name} from {model_info['path']}")
    model, tokenizer = backend.load_model(model_info["path"])

    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0
    d_model = model_info.get("d", 0)

    logger.info(f"Model loaded: {num_layers} layers, d={d_model}")

    t0 = time.time()
    sublayer_acts = collect_sublayer_activations(
        model, tokenizer, probes, num_layers, backend
    )
    logger.info(f"Activation collection: {time.time() - t0:.1f}s")

    n_decomposed = sum(1 for a in sublayer_acts if a["has_decomposition"])
    logger.info(
        f"  Decomposition available: {n_decomposed}/{num_layers} layers"
    )

    t0 = time.time()
    measurements = compute_curvature_decomposition(sublayer_acts)
    logger.info(f"Curvature computation: {time.time() - t0:.1f}s")

    correlations = compute_correlations(measurements)

    # Log key results
    r_cum = correlations.get("spearman_cum_total_vs_id", 0.0)
    r_attn = correlations.get("spearman_attn_curv_vs_id_gradient", 0.0)
    r_mlp = correlations.get("spearman_mlp_curv_vs_id_gradient", 0.0)
    mean_af = correlations.get("mean_attn_fraction", 0.0)

    logger.info(
        f"  Spearman(cum_curvature, ID) = {r_cum:.3f}"
    )
    logger.info(
        f"  Spearman(attn_curvature, dID/dl) = {r_attn:.3f}"
    )
    logger.info(
        f"  Spearman(mlp_curvature, dID/dl) = {r_mlp:.3f}"
    )
    logger.info(
        f"  Mean attention fraction of total curvature: {mean_af:.3f}"
    )

    # Clean up
    del model, tokenizer, sublayer_acts
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "d_model": d_model,
        "n_probes": len(probes),
        "measurements": measurements,
        "correlations": correlations,
    }


# =============================================================================
# Main
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run curvature accumulation analysis."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Build probe list from categories
    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3-8B"]
        probes = []
        for cat, prompts in PROBE_CATEGORIES.items():
            probes.extend(prompts[:2])  # 12 probes (2 per category)
    elif args.models:
        model_names = args.models
        probes = []
        for prompts in PROBE_CATEGORIES.values():
            probes.extend(prompts)
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = []
        for prompts in PROBE_CATEGORIES.values():
            probes.extend(prompts)

    logger.info(f"Analysis: {len(model_names)} models, {len(probes)} probes")

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        result = run_single_model(model_name, MODEL_REGISTRY[model_name], probes, backend)
        all_results.append(result)
        gc.collect()

    # Cross-model summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-MODEL CURVATURE DECOMPOSITION SUMMARY")
    logger.info(f"{'='*60}")

    for r in all_results:
        c = r["correlations"]
        n_ent = sum(
            1 for m in r["measurements"]
            if m.get("mean_h_attn") is not None
        )
        logger.info(
            f"  {r['model_name']:20s}: "
            f"cum_curv↔ID r={c.get('spearman_cum_total_vs_id', 0):.3f}, "
            f"attn_frac={c.get('mean_attn_fraction', 0):.3f}, "
            f"attn↔dID r={c.get('spearman_attn_curv_vs_id_gradient', 0):.3f}, "
            f"mlp↔dID r={c.get('spearman_mlp_curv_vs_id_gradient', 0):.3f}, "
            f"entropy_layers={n_ent}/{r['num_layers']}"
        )

    # Run F1/F3/F4/F5 falsifier tests
    logger.info(f"\n{'='*60}")
    logger.info("ENTROPY-CURVATURE FALSIFIER TESTS (F1, F3, F4, F5)")
    logger.info(f"{'='*60}")
    falsifier_results = run_falsifier_tests(all_results)
    n_obs = falsifier_results.get("n_observations", 0)
    logger.info(f"  Pool: {n_obs} (model, layer) observations with entropy+decomp")
    for key in ("f1_sign_falsifier", "f3_attn_vs_mlp", "f4_permutation_falsifier",
                "f5_scale_family_falsifier"):
        f = falsifier_results.get(key, {})
        if "error" in f:
            logger.warning(f"  {key}: ERROR — {f['error']}")
        elif "passes" in f:
            status = "PASS" if f["passes"] else "FAIL"
            if key == "f1_sign_falsifier":
                logger.info(
                    f"  F1 [{status}]: slope={f.get('slope_H_on_theta_attn_sq', 0):.4f}, "
                    f"spearman_r={f.get('spearman_r', 0):.3f}, p={f.get('p_value_ols', 1):.4f}"
                )
            elif key == "f3_attn_vs_mlp":
                logger.info(
                    f"  F3 [{status}]: r(H,θ_attn)={f.get('spearman_H_vs_theta_attn', 0):.3f} "
                    f"vs r(H,θ_mlp)={f.get('spearman_H_vs_theta_mlp', 0):.3f} "
                    f"(LFM2-qualified)"
                )
            elif key == "f4_permutation_falsifier":
                logger.info(
                    f"  F4 [{status}]: real |r|={f.get('real_abs_spearman', 0):.3f}, "
                    f"null 95th={f.get('null_95th_pct', 0):.3f}, "
                    f"pct={f.get('percentile_in_null', 0):.1f}%"
                )
            elif key == "f5_scale_family_falsifier":
                logger.info(
                    f"  F5 [{status}]: sign_flips={f.get('n_significant_sign_flips', 0)}, "
                    f"families={list(f.get('per_family', {}).keys())}"
                )
        else:
            logger.info(f"  {key}: {f.get('status', 'no result')}")
    logger.info(f"  F2: {falsifier_results.get('f2_status', 'unknown')}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "curvature_accumulation_results.json"

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "curvature_accumulation_decomposition",
        "n_models": len(all_results),
        "n_probes": len(probes),
        "models": all_results,
        "falsifier_tests": falsifier_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Curvature Accumulation Model Analysis"
    )
    parser.add_argument(
        "--output", default="results/curvature_accumulation/",
        help="Output directory",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to test",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test (2 models, 6 probes)",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
