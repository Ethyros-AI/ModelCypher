#!/usr/bin/env python3
"""Behavior Jacobian vs Activation-Covariance projector diagnostic.

Validates the behavior Jacobian null-space projector by comparing it to
the activation-covariance projector on a known adapter delta.

Pipeline:
1. Load base model + trained adapter
2. Compute adapter delta (known ground truth)
3. Collect probe activations (for activation-covariance projector)
4. Compute per-probe gradients (for behavior Jacobian projector)
5. Build both projectors
6. Project the adapter delta through both
7. Compare survival ratios: ||P @ delta||_F / ||delta||_F

The behavior Jacobian projector should preserve MORE of the useful adapter
delta because it identifies truly behavior-neutral directions (where weight
perturbation produces zero output change), whereas the activation-covariance
projector only identifies directions where probes don't activate.

Usage:
    poetry run python scripts/behavior_jacobian_diagnostic.py
    poetry run python scripts/behavior_jacobian_diagnostic.py --adapter-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_ADAPTER_PATH = (
    "/Volumes/CodeCypher/models/experiments/reinforce-revalidation/run2/adapter"
)

# Preservation probes: diverse text to define "behavior to preserve"
PRESERVATION_PROBES = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The square root of 144 is",
    "In the year 1969, humans first",
    "The chemical formula for water is",
    "A triangle has three sides and",
    "The speed of light in a vacuum is approximately",
    "Photosynthesis converts sunlight into",
    "The largest planet in our solar system is",
    "DNA stands for deoxyribonucleic",
    "The Pacific Ocean is the largest",
    "Shakespeare wrote many famous plays including",
    "An atom consists of protons, neutrons, and",
    "The mitochondria is often called the powerhouse of",
    "Pi is approximately equal to 3.14159",
    "Gravity causes objects to fall toward",
    "The human body has 206 bones",
    "Electricity flows through conductors like copper",
    "The Earth revolves around the Sun once every",
    "Machine learning is a subset of artificial",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare behavior Jacobian vs activation-covariance projectors.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to base model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to trained adapter directory.",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=20,
        help="Number of preservation probes to use.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: all adapted layers).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Verify volume is mounted
    model_path = Path(args.model_path)
    adapter_path = Path(args.adapter_path)
    if not model_path.exists():
        logger.error("Model not found: %s (is the volume mounted?)", model_path)
        sys.exit(1)
    if not adapter_path.exists():
        logger.error("Adapter not found: %s", adapter_path)
        sys.exit(1)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.transplant import (
        compute_behavior_jacobian_projector,
        compute_null_space_projector,
    )

    backend = MLXBackend()

    # =========================================================================
    # 1. Load base model + adapter
    # =========================================================================
    logger.info("Loading base model from %s", model_path)
    base_model, tokenizer = load(str(model_path))

    logger.info("Loading model with adapter from %s", adapter_path)
    adapted_model, _ = load(str(model_path), adapter_path=str(adapter_path))

    # =========================================================================
    # 2. Find adapted layers and compute deltas
    # =========================================================================
    base_weights = dict(tree_flatten(base_model.parameters()))
    adapted_weights = dict(tree_flatten(adapted_model.parameters()))

    # Find weight matrices that differ (i.e., were adapted)
    adapted_weight_names: list[str] = []
    for name in sorted(base_weights.keys()):
        if name not in adapted_weights:
            continue
        bw = base_weights[name]
        aw = adapted_weights[name]
        if bw.shape != aw.shape:
            continue
        if bw.ndim != 2:
            continue
        diff = mx.sum(mx.abs(aw - bw)).item()
        if diff > 1e-6:
            adapted_weight_names.append(name)

    logger.info("Found %d adapted weight matrices", len(adapted_weight_names))
    for name in adapted_weight_names[:10]:
        logger.info("  %s %s", name, base_weights[name].shape)
    if len(adapted_weight_names) > 10:
        logger.info("  ... and %d more", len(adapted_weight_names) - 10)

    # Filter by requested layers
    if args.layers:
        layer_indices = {int(x) for x in args.layers.split(",")}
        adapted_weight_names = [
            n for n in adapted_weight_names
            if any(f"layers.{i}." in n for i in layer_indices)
        ]
        logger.info("Filtered to %d weights for layers %s", len(adapted_weight_names), layer_indices)

    if not adapted_weight_names:
        logger.error("No adapted weights found. Check adapter path.")
        sys.exit(1)

    # =========================================================================
    # 3. Collect probe activations (for activation-covariance projector)
    # =========================================================================
    probes = PRESERVATION_PROBES[: args.n_probes]
    logger.info("Collecting activations for %d probes", len(probes))

    # Use the backend's activation collection
    base_base = getattr(base_model, "model", base_model)
    n_layers = len(base_base.layers)

    # Simple activation collection: tokenize probes, run forward, collect hidden states
    probe_hidden_states: dict[int, list] = {i: [] for i in range(n_layers)}
    for text in probes:
        tokens = mx.array([tokenizer.encode(text)])
        h = base_base.embed_tokens(tokens)
        for layer_idx, layer in enumerate(base_base.layers):
            h = layer(h, mask="causal" if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "n_heads") else None)
            # Mean pool over sequence
            pooled = mx.mean(h, axis=1).squeeze(0)  # [hidden_dim]
            probe_hidden_states[layer_idx].append(pooled)
        mx.eval(h)

    # Stack into [n_probes, hidden_dim] per layer
    layer_activations: dict[int, mx.array] = {}
    for layer_idx in range(n_layers):
        if probe_hidden_states[layer_idx]:
            layer_activations[layer_idx] = mx.stack(probe_hidden_states[layer_idx])
            mx.eval(layer_activations[layer_idx])

    # =========================================================================
    # 4-7. For each adapted weight: compute both projectors, compare survival
    # =========================================================================
    results: list[dict] = []

    for weight_name in adapted_weight_names:
        logger.info("=" * 60)
        logger.info("Analyzing: %s", weight_name)

        bw = base_weights[weight_name]
        aw = adapted_weights[weight_name]
        delta = aw - bw  # [out_dim, in_dim]
        mx.eval(delta)

        out_dim, in_dim = delta.shape
        delta_frob = mx.sqrt(mx.sum(delta * delta)).item()

        # Determine which layer this weight belongs to
        layer_idx = None
        for i in range(n_layers):
            if f"layers.{i}." in weight_name:
                layer_idx = i
                break

        logger.info(
            "  Shape: [%d, %d], delta Frobenius: %.6f",
            out_dim, in_dim, delta_frob,
        )

        if delta_frob < 1e-10:
            logger.info("  Skipping (delta ≈ 0)")
            continue

        # ----- Activation-covariance projector -----
        act_survival = None
        if layer_idx is not None and layer_idx in layer_activations:
            acts = layer_activations[layer_idx]  # [n_probes, hidden_dim]

            # Only works if in_dim matches hidden_dim
            if acts.shape[1] == in_dim:
                t0 = time.time()
                act_projector = compute_null_space_projector(
                    input_activations=acts,
                    backend=backend,
                )
                act_t = time.time() - t0

                # Project delta through activation-covariance projector
                if act_projector.projector is not None:
                    delta_proj_act = mx.matmul(delta, act_projector.projector)
                else:
                    G_act = act_projector.weighted_activations
                    gram_inv = act_projector.gram_inv
                    delta_row = mx.matmul(delta, G_act.T)
                    correction = mx.matmul(delta_row, gram_inv)
                    correction = mx.matmul(correction, G_act)
                    delta_proj_act = delta - correction
                mx.eval(delta_proj_act)

                proj_frob_act = mx.sqrt(mx.sum(delta_proj_act * delta_proj_act)).item()
                act_survival = proj_frob_act / delta_frob

                logger.info(
                    "  ACTIVATION-COV: null_rank=%d/%d, survival=%.4f (%.1f%%), "
                    "time=%.2fs",
                    act_projector.null_rank, in_dim,
                    act_survival, act_survival * 100, act_t,
                )
            else:
                logger.info(
                    "  ACTIVATION-COV: skipped (in_dim=%d != hidden_dim=%d)",
                    in_dim, acts.shape[1],
                )

        # ----- Behavior Jacobian projector -----
        t0 = time.time()
        try:
            G = backend.compute_per_probe_gradients(
                model=base_model,
                tokenizer=tokenizer,
                probe_texts=probes,
                weight_name=weight_name,
            )
            jac_t_grad = time.time() - t0

            t1 = time.time()
            jac_projector = compute_behavior_jacobian_projector(
                gradient_matrix=G,
                backend=backend,
            )
            jac_t_proj = time.time() - t1

            # Project delta through behavior Jacobian projector (vectorized)
            delta_flat = mx.reshape(delta, (1, -1))  # [1, out*in]
            delta_row = mx.matmul(delta_flat, G.T)  # [1, N]
            correction = mx.matmul(delta_row, jac_projector.gram_inv)  # [1, N]
            correction = mx.matmul(correction, G)  # [1, D]
            delta_proj_jac = mx.reshape(delta_flat - correction, (out_dim, in_dim))
            mx.eval(delta_proj_jac)

            proj_frob_jac = mx.sqrt(mx.sum(delta_proj_jac * delta_proj_jac)).item()
            jac_survival = proj_frob_jac / delta_frob

            logger.info(
                "  BEHAVIOR-JAC:   null_rank=%d/%d, survival=%.4f (%.1f%%), "
                "grad_time=%.2fs, proj_time=%.2fs",
                jac_projector.null_rank, out_dim * in_dim,
                jac_survival, jac_survival * 100,
                jac_t_grad, jac_t_proj,
            )

            results.append({
                "weight_name": weight_name,
                "shape": [out_dim, in_dim],
                "delta_frob": delta_frob,
                "act_cov_survival": act_survival,
                "behavior_jac_survival": jac_survival,
                "act_cov_null_rank": act_projector.null_rank if act_survival is not None else None,
                "behavior_jac_null_rank": jac_projector.null_rank,
            })

        except Exception as e:
            logger.error("  BEHAVIOR-JAC: FAILED — %s", e)
            results.append({
                "weight_name": weight_name,
                "shape": [out_dim, in_dim],
                "delta_frob": delta_frob,
                "act_cov_survival": act_survival,
                "behavior_jac_survival": None,
                "error": str(e),
            })

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    act_survivals = [r["act_cov_survival"] for r in results if r.get("act_cov_survival") is not None]
    jac_survivals = [r["behavior_jac_survival"] for r in results if r.get("behavior_jac_survival") is not None]

    if act_survivals:
        logger.info(
            "Activation-covariance: mean_survival=%.4f (%.1f%%), "
            "min=%.4f, max=%.4f, n=%d",
            sum(act_survivals) / len(act_survivals),
            100 * sum(act_survivals) / len(act_survivals),
            min(act_survivals), max(act_survivals),
            len(act_survivals),
        )

    if jac_survivals:
        logger.info(
            "Behavior Jacobian:     mean_survival=%.4f (%.1f%%), "
            "min=%.4f, max=%.4f, n=%d",
            sum(jac_survivals) / len(jac_survivals),
            100 * sum(jac_survivals) / len(jac_survivals),
            min(jac_survivals), max(jac_survivals),
            len(jac_survivals),
        )

    if act_survivals and jac_survivals and len(act_survivals) == len(jac_survivals):
        jac_better = sum(
            1 for a, j in zip(act_survivals, jac_survivals) if j > a
        )
        logger.info(
            "Behavior Jacobian preserves MORE delta in %d/%d layers (%.0f%%)",
            jac_better, len(act_survivals),
            100 * jac_better / len(act_survivals),
        )

    # Hypothesis check
    logger.info("")
    if jac_survivals and act_survivals:
        mean_jac = sum(jac_survivals) / len(jac_survivals)
        mean_act = sum(act_survivals) / len(act_survivals)
        if mean_jac > mean_act:
            logger.info(
                "RESULT: Behavior Jacobian preserves MORE adapter delta (%.1f%% vs %.1f%%)",
                mean_jac * 100, mean_act * 100,
            )
            logger.info("This supports using behavior Jacobian for merge null-space projection.")
        else:
            logger.info(
                "RESULT: Activation-covariance preserves MORE adapter delta (%.1f%% vs %.1f%%)",
                mean_act * 100, mean_jac * 100,
            )
            logger.info("The adapter delta lives in behavior-sensitive directions.")
            logger.info("This is EXPECTED — the adapter delta SHOULD change behavior.")
            logger.info(
                "The key question is whether the behavior Jacobian projector "
                "preserves the delta in DIFFERENT directions that are more useful.",
            )


if __name__ == "__main__":
    main()
