#!/usr/bin/env python3
"""Experiment: RMT Spectral Gap vs Measured Attention Rank.

Tests whether Random Matrix Theory predictions from "Mind the Gap" (Noci et al.,
ICML 2025, arXiv:2410.07799v3) explain the measured spectral gap in attention
matrices across architectures.

Hypothesis:
    H1: QK-Norm models (Qwen3) have significantly different sigma_1/sigma_2
        than non-QK-Norm models (Qwen2.5, Llama), as predicted by
        normalization-dependent RMT analysis.

Measurements:
    For each attention head at each layer:
        spectral_gap = sigma_1 / sigma_2  (from SVD of attention matrix)
        effective_rank = exp(Shannon entropy of normalized SVs)

Falsification criteria:
    FAIL if Spearman(predicted, measured) < 0.3 across all models
    FAIL if Qwen3 vs Qwen2.5 gap distributions identical (KS p > 0.05)
    FAIL if within-model gap CV > 0.5

References:
    Noci et al. (ICML 2025, arXiv:2410.07799): Mind the Gap — RMT spectral
        gap analysis of softmax attention, rank collapse in depth and width
    Bhojanapalli et al. (2020): Critical head dimension d_h = Omega(log n)

Usage:
    poetry run python scripts/rmt_spectral_gap_experiment.py

    # Smoke test
    poetry run python scripts/rmt_spectral_gap_experiment.py --smoke

    # Custom output
    poetry run python scripts/rmt_spectral_gap_experiment.py \
        --output results/rmt_spectral_gap/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "architecture": "qwen3",
        "has_qk_norm": True,
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "d_head": 128,
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "architecture": "qwen2.5",
        "has_qk_norm": False,
        "d_model": 2048,
        "n_heads": 16,
        "n_kv_heads": 2,
        "d_head": 128,
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "architecture": "llama",
        "has_qk_norm": False,
        "d_model": 3072,
        "n_heads": 24,
        "n_kv_heads": 8,
        "d_head": 128,
    },
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "architecture": "lfm2",
        "has_qk_norm": False,
        "d_model": 1024,
        "n_heads": 16,
        "n_kv_heads": 8,
        "d_head": 64,
        "attn_layer_indices": [2, 5, 8, 10, 12, 14],  # Only 6/16 are attention
    },
}

# =============================================================================
# Probes (30 diverse prompts)
# =============================================================================

PROBES = [
    # Retrieval (5)
    "The capital of France is",
    "Who wrote Romeo and Juliet?",
    "The chemical symbol for water is",
    "The largest planet in our solar system is",
    "The speed of light in a vacuum is approximately",
    # Arithmetic (5)
    "What is 347 + 528?",
    "What is 15 * 23?",
    "What is 1024 / 16?",
    "What is 99 - 37?",
    "What is 8 * 7 + 13?",
    # Reasoning (5)
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 to make 100?",
    "A farmer has 17 sheep. All but 9 die. How many left?",
    "A lily pad doubles daily. It takes 48 days to cover the lake. When half?",
    "What comes next: 2, 6, 12, 20, 30, ?",
    # Creative (5)
    "Write a haiku about the ocean.",
    "Describe a sunset over the mountains in one vivid sentence.",
    "Write a short poem about the passage of time.",
    "Describe the taste of chocolate using only three words.",
    "Write a one-sentence story with a twist ending.",
    # Code (5)
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function to compute Fibonacci up to n terms.",
    "Write a Python function to find the max element without max().",
    "Write a Python function to check if a string is a palindrome.",
    # Narrative (5)
    "Once upon a time in a faraway kingdom, there lived a",
    "The old lighthouse keeper watched the storm approach from",
    "In the year 2150, humanity had finally achieved",
    "She opened the letter and read the first line:",
    "The forest was silent except for the sound of",
]


# =============================================================================
# Attention Capture — Architecture-Agnostic
# =============================================================================


def capture_attention_matrices(
    model, tokenizer, prompt: str, backend
) -> dict[int, Any]:
    """Capture post-softmax attention matrices for all attention layers.

    Returns: {layer_idx: attention_weights [n_heads, seq_len, seq_len]}

    Handles both standard transformers (Qwen, Llama) and hybrid (LFM2).
    Manually computes softmax(Q @ K^T / sqrt(d_k)) since fused kernels
    don't expose weights.
    """
    import mlx.core as mx

    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    embed = getattr(base, "embed_tokens", None)

    token_ids = tokenizer.encode(prompt)
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)

    input_ids = mx.array([token_ids])
    hidden = embed(input_ids)
    seq_len = input_ids.shape[1]

    # Create causal mask
    causal_mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)

    captured: dict[int, Any] = {}

    for i, layer in enumerate(layers):
        # Check if this is an attention layer
        attn = getattr(layer, "self_attn", None)
        is_attn = True

        # LFM2 hybrid check
        if hasattr(layer, "is_attention_layer"):
            is_attn = layer.is_attention_layer

        if attn is not None and is_attn:
            # Pre-attention norm
            if hasattr(layer, "input_layernorm"):
                x_normed = layer.input_layernorm(hidden)
            elif hasattr(layer, "operator_norm"):
                x_normed = layer.operator_norm(hidden)
            else:
                x_normed = hidden

            B, L, D = x_normed.shape

            # QKV projections
            queries = attn.q_proj(x_normed)
            keys = attn.k_proj(x_normed)

            n_heads = attn.n_heads
            n_kv_heads = getattr(attn, "n_kv_heads", n_heads)
            head_dim = D // n_heads

            # Reshape to [B, n_heads, L, head_dim]
            queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Apply QK-Norm if present
            if hasattr(attn, "q_norm"):
                queries = attn.q_norm(queries)
            if hasattr(attn, "k_norm"):
                keys = attn.k_norm(keys)
            # LFM2 uses q_layernorm / k_layernorm
            if hasattr(attn, "q_layernorm"):
                queries = queries.transpose(0, 2, 1, 3).reshape(B, L, n_heads, head_dim)
                queries = attn.q_layernorm(queries)
                queries = queries.transpose(0, 2, 1, 3)
            if hasattr(attn, "k_layernorm"):
                keys = keys.transpose(0, 2, 1, 3).reshape(B, L, n_kv_heads, head_dim)
                keys = attn.k_layernorm(keys)
                keys = keys.transpose(0, 2, 1, 3)

            # RoPE
            if hasattr(attn, "rope"):
                queries = attn.rope(queries)
                keys = attn.rope(keys)
            elif hasattr(attn, "rotary_emb"):
                queries, keys = attn.rotary_emb(queries, keys)

            # GQA expansion
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                keys = mx.repeat(keys, n_rep, axis=1)

            # Compute attention scores
            scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim))
            scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask
            scores = scores + causal_mask

            # Softmax in float32
            weights = mx.softmax(scores.astype(mx.float32), axis=-1)
            mx.eval(weights)

            captured[i] = weights[0]  # [n_heads, seq_len, seq_len]

        # Forward the actual layer for correct hidden state propagation
        # Per-layer mask routing: LFM2 hybrid layers use is_attention_layer
        if hasattr(layer, "is_attention_layer"):
            layer_mask = "causal" if layer.is_attention_layer else None
        else:
            try:
                layer_mask = backend.create_causal_mask(seq_len, hidden.dtype)
            except Exception:
                layer_mask = None
        try:
            hidden = layer(hidden, mask=layer_mask)
        except (TypeError, ValueError):
            try:
                hidden = layer(hidden)
            except Exception:
                hidden = layer(hidden, mask=None)

    return captured


# =============================================================================
# Spectral Analysis
# =============================================================================


def compute_attention_spectra(
    attn_weights: dict[int, Any],
) -> list[dict]:
    """Compute spectral gap and effective rank for each attention head at each layer.

    Returns list of per-layer dicts with per-head spectral metrics.
    """
    import mlx.core as mx
    import numpy as np

    results = []
    eps = float(np.finfo(np.float32).eps)

    for layer_idx in sorted(attn_weights.keys()):
        W = attn_weights[layer_idx]  # [n_heads, seq_len, seq_len]
        n_heads = W.shape[0]
        seq_len = W.shape[1]

        head_metrics = []
        for h in range(n_heads):
            A = W[h]  # [seq_len, seq_len]
            A_np = np.array(A.tolist(), dtype=np.float64)

            # SVD
            try:
                S = np.linalg.svd(A_np, compute_uv=False)
            except np.linalg.LinAlgError:
                head_metrics.append({
                    "head": h,
                    "spectral_gap": float("nan"),
                    "effective_rank": float("nan"),
                    "entropy": float("nan"),
                    "sigma_1": float("nan"),
                    "sigma_2": float("nan"),
                })
                continue

            sigma_1 = float(S[0])
            sigma_2 = float(S[1]) if len(S) > 1 else eps
            spectral_gap = sigma_1 / (sigma_2 + eps)

            # Shannon effective rank
            S_sq = S ** 2
            total = S_sq.sum()
            if total > 0:
                p = S_sq / total
                p = p[p > eps]
                entropy = -float(np.sum(p * np.log(p)))
                eff_rank = float(np.exp(entropy))
            else:
                entropy = 0.0
                eff_rank = 0.0

            head_metrics.append({
                "head": h,
                "spectral_gap": spectral_gap,
                "effective_rank": eff_rank,
                "entropy": entropy,
                "sigma_1": sigma_1,
                "sigma_2": sigma_2,
            })

        # Aggregate per layer
        gaps = [m["spectral_gap"] for m in head_metrics if not math.isnan(m["spectral_gap"])]
        ranks = [m["effective_rank"] for m in head_metrics if not math.isnan(m["effective_rank"])]
        entropies = [m["entropy"] for m in head_metrics if not math.isnan(m["entropy"])]

        results.append({
            "layer_idx": layer_idx,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "per_head": head_metrics,
            "mean_spectral_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "std_spectral_gap": float(np.std(gaps)) if gaps else float("nan"),
            "mean_effective_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        })

    return results


def rmt_predicted_gap(d_head: int, seq_len: int, has_qk_norm: bool) -> float:
    """RMT prediction for spectral gap from Noci et al. (ICML 2025).

    For softmax attention with isotropic Q, K of dimension d_head and sequence
    length n, the spectral gap depends on the concentration of softmax outputs.

    The key quantity is beta = 1/sqrt(d_head) (attention temperature).
    When beta*sqrt(d_head) >> 1, softmax concentrates → large gap.
    When beta*sqrt(d_head) << 1, softmax is diffuse → small gap.

    QK-Norm sets ||q|| = ||k|| = 1, making dot products O(1/sqrt(d_head)),
    which changes the effective temperature.

    Returns predicted spectral gap sigma_1/sigma_2.
    """
    import numpy as np

    # From Noci et al.: gap emerges from MP law applied to softmax output
    # The outlier eigenvalue is 1 (by row normalization of softmax)
    # The bulk is bounded by MP edges for a random stochastic matrix

    # MP bulk edge for n x n stochastic matrix with effective rank r
    # sigma_bulk_max ~ sqrt(r/n)
    # Predicted gap ~ n / r (ratio of outlier to bulk edge)

    # For standard attention: effective rank depends on beta * ||qk||
    beta = 1.0 / np.sqrt(d_head)

    if has_qk_norm:
        # QK-Norm: ||q|| = ||k|| = 1, so q·k ~ N(0, 1/d_head)
        # This concentrates scores more → larger gap
        effective_score_var = 1.0 / d_head
    else:
        # Standard: ||q||, ||k|| ~ sqrt(d_head), so q·k ~ N(0, 1)
        # Score variance is larger → softmax more concentrated
        effective_score_var = 1.0

    # Approximation from RMT: attention effective rank ~ n * exp(-score_var)
    # when scores are Gaussian with variance score_var
    eff_rank_ratio = np.exp(-effective_score_var * beta * beta * d_head)
    eff_rank = max(1.0, seq_len * eff_rank_ratio)

    # Predicted gap = outlier eigenvalue (≈1) / bulk edge (≈ sqrt(eff_rank/n))
    predicted_gap = np.sqrt(seq_len / eff_rank) if eff_rank > 0 else 1.0
    return float(predicted_gap)


# =============================================================================
# Main Experiment
# =============================================================================


def run_single_model(
    model_name: str, model_info: dict, probes: list[str], backend
) -> dict:
    """Run spectral gap analysis for a single model."""
    import numpy as np

    model_path = model_info["path"]
    logger.info(f"Loading model: {model_name} from {model_path}")

    model, tokenizer = backend.load_model(model_path)
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0

    logger.info(f"Model loaded: {num_layers} layers")

    # Collect attention matrices across probes
    all_layer_spectra: dict[int, list[dict]] = {}

    for pi, prompt in enumerate(probes):
        logger.info(f"  Probe {pi+1}/{len(probes)}: {prompt[:50]}...")
        try:
            attn_weights = capture_attention_matrices(model, tokenizer, prompt, backend)
            spectra = compute_attention_spectra(attn_weights)

            for ls in spectra:
                lid = ls["layer_idx"]
                if lid not in all_layer_spectra:
                    all_layer_spectra[lid] = []
                all_layer_spectra[lid].append(ls)

            # Free attention matrices
            del attn_weights
        except Exception as e:
            logger.warning(f"  Failed on probe {pi}: {e}")
            continue

    # Aggregate across probes per layer
    layer_results = []
    all_gaps = []
    all_ranks = []

    for lid in sorted(all_layer_spectra.keys()):
        probe_spectra = all_layer_spectra[lid]
        gaps = [s["mean_spectral_gap"] for s in probe_spectra if not math.isnan(s["mean_spectral_gap"])]
        ranks = [s["mean_effective_rank"] for s in probe_spectra if not math.isnan(s["mean_effective_rank"])]
        entropies = [s["mean_entropy"] for s in probe_spectra if not math.isnan(s["mean_entropy"])]

        mean_gap = float(np.mean(gaps)) if gaps else float("nan")
        std_gap = float(np.std(gaps)) if gaps else float("nan")
        cv_gap = std_gap / mean_gap if mean_gap > 0 else float("nan")

        # RMT prediction
        seq_lens = [s["seq_len"] for s in probe_spectra if s["seq_len"] > 0]
        median_seq = int(np.median(seq_lens)) if seq_lens else 10
        predicted = rmt_predicted_gap(
            model_info["d_head"], median_seq, model_info["has_qk_norm"]
        )

        layer_results.append({
            "layer_idx": lid,
            "n_probes": len(gaps),
            "mean_spectral_gap": mean_gap,
            "std_spectral_gap": std_gap,
            "cv_spectral_gap": cv_gap,
            "mean_effective_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "rmt_predicted_gap": predicted,
            "median_seq_len": median_seq,
        })

        all_gaps.extend(gaps)
        all_ranks.extend(ranks)

    # Model-level CV
    model_cv = float(np.std(all_gaps) / np.mean(all_gaps)) if all_gaps and np.mean(all_gaps) > 0 else float("nan")

    # Model-level statistics
    model_mean_gap = float(np.mean(all_gaps)) if all_gaps else float("nan")
    model_mean_rank = float(np.mean(all_ranks)) if all_ranks else float("nan")

    # Theoretical max effective rank: ~0.63 * n (Bhojanapalli et al.)
    median_seq_all = int(np.median([lr["median_seq_len"] for lr in layer_results])) if layer_results else 10
    theoretical_max = 0.63 * median_seq_all
    utilization = model_mean_rank / theoretical_max if theoretical_max > 0 else 0.0

    logger.info(
        f"  Mean gap={model_mean_gap:.2f}, Mean rank={model_mean_rank:.2f}, "
        f"CV={model_cv:.3f}, Utilization={utilization:.1%}"
    )

    del model, tokenizer
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "has_qk_norm": model_info["has_qk_norm"],
        "d_head": model_info["d_head"],
        "n_heads": model_info["n_heads"],
        "n_kv_heads": model_info["n_kv_heads"],
        "num_layers": num_layers,
        "layer_results": layer_results,
        "model_mean_spectral_gap": model_mean_gap,
        "model_mean_effective_rank": model_mean_rank,
        "model_cv_gap": model_cv,
        "theoretical_max_rank": theoretical_max,
        "attention_utilization": utilization,
    }


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full RMT spectral gap experiment."""
    import numpy as np
    from scipy import stats as scipy_stats

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain.statistics import spearman_correlation

    backend = initialize_default_backend()

    # Select models and probes
    if args.smoke:
        model_names = ["Qwen3-8B", "Qwen2.5-3B"]
        probes = PROBES[:6]
    elif args.models:
        model_names = args.models
        probes = PROBES[:args.n_probes]
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = PROBES[:args.n_probes]

    logger.info(f"Experiment: {len(model_names)} models, {len(probes)} probes")

    # Run per model
    model_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        result = run_single_model(model_name, MODEL_REGISTRY[model_name], probes, backend)
        model_results.append(result)
        gc.collect()

    # ==========================================================================
    # Falsification Tests
    # ==========================================================================

    # Test 1: Spearman(predicted gap, measured gap) across all layer measurements
    all_predicted = []
    all_measured = []
    for mr in model_results:
        for lr in mr["layer_results"]:
            if not math.isnan(lr["mean_spectral_gap"]) and not math.isnan(lr["rmt_predicted_gap"]):
                all_predicted.append(lr["rmt_predicted_gap"])
                all_measured.append(lr["mean_spectral_gap"])

    spearman_global = spearman_correlation(all_predicted, all_measured) if len(all_predicted) >= 3 else 0.0
    passes_spearman = spearman_global > 0.3

    # Test 2: KS test — Qwen3 (QK-Norm) vs Qwen2.5 (no QK-Norm)
    qwen3_gaps = []
    qwen25_gaps = []
    for mr in model_results:
        if mr["model_name"] == "Qwen3-8B":
            for lr in mr["layer_results"]:
                if not math.isnan(lr["mean_spectral_gap"]):
                    qwen3_gaps.append(lr["mean_spectral_gap"])
        elif mr["model_name"] == "Qwen2.5-3B":
            for lr in mr["layer_results"]:
                if not math.isnan(lr["mean_spectral_gap"]):
                    qwen25_gaps.append(lr["mean_spectral_gap"])

    if qwen3_gaps and qwen25_gaps:
        ks_stat, ks_pval = scipy_stats.ks_2samp(qwen3_gaps, qwen25_gaps)
        passes_ks = ks_pval < 0.05
    else:
        ks_stat, ks_pval = float("nan"), float("nan")
        passes_ks = False

    # Test 3: Within-model CV < 0.5
    cv_results = {}
    all_pass_cv = True
    for mr in model_results:
        cv = mr["model_cv_gap"]
        passes = cv < 0.5 if not math.isnan(cv) else False
        cv_results[mr["model_name"]] = {"cv": cv, "passes": passes}
        if not passes:
            all_pass_cv = False

    overall_pass = passes_spearman and passes_ks and all_pass_cv

    # ==========================================================================
    # Summary
    # ==========================================================================

    summary = {
        "n_models": len(model_results),
        "n_probes": len(probes),
        "test_1_spearman": {
            "value": spearman_global,
            "threshold": 0.3,
            "passes": passes_spearman,
            "n_points": len(all_predicted),
        },
        "test_2_ks_qwen3_vs_qwen25": {
            "ks_statistic": ks_stat,
            "p_value": ks_pval,
            "threshold": 0.05,
            "passes": passes_ks,
            "qwen3_n": len(qwen3_gaps),
            "qwen25_n": len(qwen25_gaps),
        },
        "test_3_within_model_cv": cv_results,
        "all_pass_cv": all_pass_cv,
        "overall_verdict": "H1 SUPPORTED" if overall_pass else "H1 REFUTED",
        "references": [
            "Noci et al. (ICML 2025, arXiv:2410.07799): Mind the Gap",
            "Bhojanapalli et al. (2020): d_h = Omega(log n) critical threshold",
        ],
    }

    verdict = summary["overall_verdict"]
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT VERDICT: {verdict}")
    logger.info(f"  Spearman(predicted, measured): {spearman_global:.3f} ({'PASS' if passes_spearman else 'FAIL'})")
    logger.info(f"  KS test Qwen3 vs Qwen2.5: p={ks_pval:.4f} ({'PASS' if passes_ks else 'FAIL'})")
    logger.info(f"  Within-model CV: {cv_results}")
    logger.info(f"{'='*60}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rmt_spectral_gap_results.json"

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "rmt_spectral_gap_vs_measured_attention_rank",
        "models": model_results,
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RMT Spectral Gap vs Measured Attention Rank Experiment"
    )
    parser.add_argument(
        "--output",
        default="results/rmt_spectral_gap/",
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=30,
        help="Number of probes (default: 30)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 models, 6 probes",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
