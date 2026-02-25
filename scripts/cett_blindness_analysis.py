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

"""CETT blindness analysis: compare per-neuron contributions on eval vs inference probes.

Experimental research script — NOT a CLI command.

Tests whether CKA blindness is explained by the H-neuron hypothesis
(Gao et al., arXiv:2512.01797): eval probes and inference probes activate
different FFN neuron subsets, and CKA can't resolve perturbations in the
sparse inference-relevant subset.

Usage:
    poetry run python scripts/cett_blindness_analysis.py \\
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \\
        --eval-data data/training/benchmark_val.jsonl \\
        --n-inference-probes 10 \\
        --output-root results/cett_blindness_350M
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("cett_blindness_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-neuron CETT contributions on eval vs inference probes.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model directory.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("data/training/benchmark_val.jsonl"),
        help="Eval dataset (JSONL with 'text' field).",
    )
    parser.add_argument(
        "--n-eval-probes",
        type=int,
        default=20,
        help="Number of eval probe texts to use (default: 20).",
    )
    parser.add_argument(
        "--n-inference-probes",
        type=int,
        default=10,
        help="Number of StarProblem inference probes (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for StarProblem generation (default: 42).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/cett_blindness"),
        help="Output directory (default: results/cett_blindness).",
    )
    return parser.parse_args()


def _load_eval_texts(path: Path, n: int) -> list[str]:
    """Load first n eval texts from JSONL."""
    texts = []
    with open(path) as f:
        for line in f:
            if len(texts) >= n:
                break
            sample = json.loads(line)
            text = sample.get("text", "")
            if text:
                texts.append(text[:512])  # Truncate for memory
    return texts


def _build_inference_prompts(n_problems: int, seed: int) -> list[str]:
    """Create StarProblem prompts for inference probing."""
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )
    from modelcypher.core.domain.training.online_eval import create_eval_problem_set

    problems = create_eval_problem_set(n_problems=n_problems, seed=seed)
    n_demos = len(default_few_shot_examples())
    prompts = [build_forward_prompt(p, demonstrations=n_demos) for p in problems]
    return prompts


def main() -> None:
    args = _parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    if not args.eval_data.exists():
        print(f"ERROR: Eval data not found: {args.eval_data}", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from modelcypher.cli.composition import get_backend
    from modelcypher.core.domain.geometry.cett_decomposition import (
        compute_cett_per_layer,
        compute_down_proj_column_norms,
    )

    backend = get_backend()

    # --- Load model ---
    logger.info("Loading model from %s", model_path)
    model, tokenizer = backend.load_model(str(model_path))

    # --- Pre-compute down_proj column norms ---
    logger.info("Computing down_proj column norms")
    col_norms = compute_down_proj_column_norms(model, backend)
    logger.info("Got column norms for %d layers", len(col_norms))

    # --- Prepare probe texts ---
    logger.info("Loading %d eval probe texts from %s", args.n_eval_probes, args.eval_data)
    eval_texts = _load_eval_texts(args.eval_data, args.n_eval_probes)
    logger.info("Loaded %d eval texts", len(eval_texts))

    logger.info("Building %d inference probe prompts (seed=%d)", args.n_inference_probes, args.seed)
    inference_texts = _build_inference_prompts(args.n_inference_probes, args.seed)
    logger.info("Built %d inference prompts", len(inference_texts))

    # --- Collect trajectories ---
    logger.info("Collecting eval probe trajectories")
    eval_traj = backend.collect_trajectory_batch(model, tokenizer, eval_texts)
    logger.info(
        "Eval: %d tokens across %d texts, %d layers",
        eval_traj.total_tokens, eval_traj.n_texts, len(eval_traj.positions),
    )

    logger.info("Collecting inference probe trajectories")
    inf_traj = backend.collect_trajectory_batch(model, tokenizer, inference_texts)
    logger.info(
        "Inference: %d tokens across %d texts, %d layers",
        inf_traj.total_tokens, inf_traj.n_texts, len(inf_traj.positions),
    )

    # --- Compute CETT per layer for both probe types ---
    common_layers = sorted(
        set(eval_traj.intermediate_positions.keys())
        & set(inf_traj.intermediate_positions.keys())
        & set(col_norms.keys())
    )
    logger.info("Computing CETT for %d common layers", len(common_layers))

    results: dict[str, dict] = {"layers": {}}
    results["metadata"] = {
        "model": str(model_path),
        "eval_data": str(args.eval_data),
        "n_eval_texts": len(eval_texts),
        "n_inference_texts": len(inference_texts),
        "eval_total_tokens": eval_traj.total_tokens,
        "inference_total_tokens": inf_traj.total_tokens,
        "seed": args.seed,
    }

    for layer_idx in common_layers:
        eval_cett = compute_cett_per_layer(
            intermediate=eval_traj.intermediate_positions[layer_idx],
            hidden_state=eval_traj.positions[layer_idx],
            down_proj_col_norms=col_norms[layer_idx],
            backend=backend,
            layer_idx=layer_idx,
        )
        inf_cett = compute_cett_per_layer(
            intermediate=inf_traj.intermediate_positions[layer_idx],
            hidden_state=inf_traj.positions[layer_idx],
            down_proj_col_norms=col_norms[layer_idx],
            backend=backend,
            layer_idx=layer_idx,
        )

        # Convert to lists for JSON serialization
        eval_mean = [float(x) for x in eval_cett.mean_cett.tolist()]
        inf_mean = [float(x) for x in inf_cett.mean_cett.tolist()]
        n_neurons = len(eval_mean)

        # --- Compare distributions ---
        # 1. Overlap: what fraction of top-k neurons are shared?
        top_k = max(1, n_neurons // 100)  # Top 1%
        eval_top_k = set(sorted(range(n_neurons), key=lambda j: eval_mean[j], reverse=True)[:top_k])
        inf_top_k = set(sorted(range(n_neurons), key=lambda j: inf_mean[j], reverse=True)[:top_k])
        overlap_fraction = len(eval_top_k & inf_top_k) / top_k if top_k > 0 else 0.0

        # 2. Rank correlation (Spearman) between eval and inference CETT
        eval_ranks = _rank_array(eval_mean)
        inf_ranks = _rank_array(inf_mean)
        spearman_r = _spearman_from_ranks(eval_ranks, inf_ranks)

        # 3. Per-neuron CETT ratio: inference / eval (for identifying differentially active neurons)
        eps = 1e-10
        cett_ratio = [inf_mean[j] / max(eval_mean[j], eps) for j in range(n_neurons)]
        # Neurons with ratio >> 1 are differentially active on inference probes
        high_ratio_count = sum(1 for r in cett_ratio if r > 2.0)
        low_ratio_count = sum(1 for r in cett_ratio if r < 0.5)

        # 4. Summary statistics
        eval_mean_total = sum(eval_mean) / n_neurons if n_neurons > 0 else 0.0
        inf_mean_total = sum(inf_mean) / n_neurons if n_neurons > 0 else 0.0

        layer_result = {
            "n_neurons": n_neurons,
            "eval_tokens": eval_cett.n_tokens,
            "inference_tokens": inf_cett.n_tokens,
            "eval_mean_cett": eval_mean_total,
            "inference_mean_cett": inf_mean_total,
            "top_1pct_overlap": overlap_fraction,
            "spearman_rank_correlation": spearman_r,
            "high_ratio_neurons_count": high_ratio_count,
            "high_ratio_neurons_fraction": high_ratio_count / n_neurons if n_neurons > 0 else 0.0,
            "low_ratio_neurons_count": low_ratio_count,
            "low_ratio_neurons_fraction": low_ratio_count / n_neurons if n_neurons > 0 else 0.0,
        }
        results["layers"][str(layer_idx)] = layer_result

        logger.info(
            "Layer %d: overlap_1pct=%.3f spearman=%.4f high_ratio=%d (%.2f%%) low_ratio=%d (%.2f%%)",
            layer_idx,
            overlap_fraction,
            spearman_r,
            high_ratio_count,
            100 * high_ratio_count / n_neurons if n_neurons else 0,
            low_ratio_count,
            100 * low_ratio_count / n_neurons if n_neurons else 0,
        )

    # --- Summary across layers ---
    layers = results["layers"]
    if layers:
        overlaps = [v["top_1pct_overlap"] for v in layers.values()]
        spearmans = [v["spearman_rank_correlation"] for v in layers.values()]
        high_fracs = [v["high_ratio_neurons_fraction"] for v in layers.values()]

        results["summary"] = {
            "mean_top_1pct_overlap": sum(overlaps) / len(overlaps),
            "min_top_1pct_overlap": min(overlaps),
            "mean_spearman": sum(spearmans) / len(spearmans),
            "min_spearman": min(spearmans),
            "mean_high_ratio_fraction": sum(high_fracs) / len(high_fracs),
            "max_high_ratio_fraction": max(high_fracs),
        }

        logger.info("=== SUMMARY ===")
        logger.info(
            "Mean top-1%% overlap: %.3f (1.0 = identical, 0.0 = disjoint)",
            results["summary"]["mean_top_1pct_overlap"],
        )
        logger.info(
            "Mean Spearman rank correlation: %.4f",
            results["summary"]["mean_spearman"],
        )
        logger.info(
            "Mean high-ratio neuron fraction: %.4f (neurons >2x more active on inference)",
            results["summary"]["mean_high_ratio_fraction"],
        )

        # Interpretation
        if results["summary"]["mean_top_1pct_overlap"] < 0.5:
            logger.info(
                "FINDING: Eval and inference probes activate DIFFERENT top neurons. "
                "H-neuron hypothesis supported — CKA blindness is expected."
            )
        else:
            logger.info(
                "FINDING: Eval and inference probes activate SIMILAR top neurons. "
                "CKA blindness has a different cause."
            )

    # --- Write results ---
    result_path = output_dir / "cett_analysis.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", result_path)


# --- Utility functions (no numpy dependency) ---


def _rank_array(values: list[float]) -> list[float]:
    """Compute ranks (1-based) for a list of values."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank_idx, orig_idx in enumerate(indexed):
        ranks[orig_idx] = float(rank_idx + 1)
    return ranks


def _spearman_from_ranks(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Compute Spearman correlation from pre-computed ranks."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov = sum((ranks_a[i] - mean_a) * (ranks_b[i] - mean_b) for i in range(n))
    var_a = sum((ranks_a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((ranks_b[i] - mean_b) ** 2 for i in range(n))
    denom = (var_a * var_b) ** 0.5
    if denom < 1e-15:
        return 0.0
    return cov / denom


if __name__ == "__main__":
    main()
