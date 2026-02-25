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

"""Tightness assessment of the logit perturbation bound on 350M.

Measures:
1. σ_max(W_out) — readout amplification
2. Per-layer Lipschitz bounds L_i = σ_max(W_down) × σ_max(W_up)
3. Product propagation factor ∏(1 + L_i)
4. Baseline margins on StarProblem probes
5. Simulated worst-case bound assuming ||scale×BA||₂ = σ_k per layer

This script does NOT train anything. It measures the geometric constants
of the base model to assess whether the bound chain is tight enough to be
useful, or whether we need the Gram-based tightening.

Usage:
    poetry run python scripts/perturbation_bound_tightness.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
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
logger = logging.getLogger("perturbation_bound_tightness")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess tightness of logit perturbation bound on a model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model directory.",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=10,
        help="Number of StarProblem probes for margin measurement (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for probe generation (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    from modelcypher.cli.composition import get_backend
    from modelcypher.core.domain.geometry.perturbation_bound import (
        check_margin_safety,
        compute_layer_lipschitz_bounds,
        compute_logit_perturbation_bound,
        compute_readout_spectral_norm,
    )
    from modelcypher.core.domain.training.geometric_lora import (
        analyze_weight_geometries,
    )
    from modelcypher.core.domain.training.online_eval import (
        compute_answer_margin,
        create_eval_problem_set,
    )

    backend = get_backend()

    # --- Load model ---
    logger.info("Loading model from %s", model_path)
    model, tokenizer = backend.load_model(str(model_path))

    # --- Step 1: Readout σ_max ---
    logger.info("Computing readout spectral norm σ_max(W_out)")
    sigma_max_readout = compute_readout_spectral_norm(model, backend)
    logger.info("σ_max(W_out) = %.4f", sigma_max_readout)

    # --- Step 2: Per-layer Lipschitz bounds ---
    logger.info("Computing per-layer Lipschitz bounds")
    lipschitz = compute_layer_lipschitz_bounds(model, backend)
    logger.info("Lipschitz bounds for %d layers:", len(lipschitz))
    for idx in sorted(lipschitz):
        logger.info("  Layer %d: L = %.4f", idx, lipschitz[idx])

    # --- Step 3: Propagation factors ---
    base = getattr(model, "model", model)
    n_layers = len(base.layers)
    propagation_factors: dict[int, float] = {}
    for start_layer in range(n_layers):
        prop = 1.0
        for i in range(start_layer, n_layers):
            prop *= (1.0 + lipschitz.get(i, 0.0))
        propagation_factors[start_layer] = prop

    logger.info("Propagation factors (from layer l to output):")
    for idx in sorted(propagation_factors):
        logger.info("  Layer %d → output: %.2e", idx, propagation_factors[idx])

    # --- Step 4: Weight geometry (σ_k per layer) ---
    logger.info("Computing weight geometries (SVD)")
    # Get all weight matrices via MLX tree_flatten
    from mlx.utils import tree_flatten as mlx_flatten

    weights: dict[str, object] = {}
    for name, param in mlx_flatten(model.parameters()):
        if "weight" in name and param.ndim == 2:
            weights[name] = param
    geometries = analyze_weight_geometries(weights, backend)

    # Find σ_k per model layer (use the minimum across projections in each layer)
    sigma_k_per_layer: dict[int, float] = {}
    for key, geom in geometries.items():
        # Extract layer index from key like "model.layers.5.feed_forward.w2.weight"
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    if layer_idx not in sigma_k_per_layer:
                        sigma_k_per_layer[layer_idx] = geom.sigma_k
                    else:
                        sigma_k_per_layer[layer_idx] = min(
                            sigma_k_per_layer[layer_idx], geom.sigma_k
                        )
                except ValueError:
                    pass
                break

    logger.info("σ_k per layer (min across projections):")
    for idx in sorted(sigma_k_per_layer):
        logger.info("  Layer %d: σ_k = %.4f", idx, sigma_k_per_layer[idx])

    # --- Step 5: Worst-case bound (all layers perturbed at σ_k) ---
    # This simulates the maximum adapter: ||scale*BA||₂ = σ_k per layer
    perturbed_all = {
        idx: sigma_k_per_layer[idx]
        for idx in sigma_k_per_layer
    }
    bound_result = compute_logit_perturbation_bound(
        model=model,
        backend=backend,
        perturbed_layers=perturbed_all,
        layer_lipschitz=lipschitz,
        sigma_max_readout=sigma_max_readout,
    )
    logger.info("Worst-case bound (all layers at σ_k): %.2e", bound_result.bound)

    # Single-layer bounds (each layer perturbed alone)
    single_layer_bounds: dict[int, float] = {}
    for idx, sk in sigma_k_per_layer.items():
        single = compute_logit_perturbation_bound(
            model=model,
            backend=backend,
            perturbed_layers={idx: sk},
            layer_lipschitz=lipschitz,
            sigma_max_readout=sigma_max_readout,
        )
        single_layer_bounds[idx] = single.bound

    logger.info("Single-layer bounds (perturbed at σ_k):")
    for idx in sorted(single_layer_bounds):
        logger.info(
            "  Layer %d: bound = %.2e (prop_factor = %.2e × σ_k = %.4f)",
            idx, single_layer_bounds[idx],
            propagation_factors[idx], sigma_k_per_layer[idx],
        )

    # --- Step 6: Baseline margins ---
    logger.info("Computing baseline margins on %d StarProblem probes", args.n_probes)
    problems = create_eval_problem_set(n_problems=args.n_probes, seed=args.seed)

    def _collect_logits(prompt: str):
        return backend.collect_logits(model, tokenizer, prompt)

    margins = compute_answer_margin(problems, _collect_logits, backend)
    margin_values = list(margins.values())
    min_margin = min(margin_values) if margin_values else 0.0
    mean_margin = sum(margin_values) / len(margin_values) if margin_values else 0.0
    logger.info("Margins: min=%.4f, mean=%.4f, max=%.4f",
                min_margin, mean_margin, max(margin_values) if margin_values else 0.0)
    logger.info("Per-problem margins:")
    for pid, m in sorted(margins.items()):
        logger.info("  %s: %.4f", pid, m)

    # --- Step 7: Safety check ---
    safety = check_margin_safety(bound_result.bound, min_margin)
    logger.info("=== MARGIN SAFETY ===")
    logger.info("Worst-case logit bound: %.2e", safety.logit_bound)
    logger.info("Min margin: %.4f", safety.min_margin)
    logger.info("Safety ratio (margin/bound): %.2e", safety.safety_ratio)
    logger.info("Safe: %s", safety.safe)

    if safety.safe:
        logger.info(
            "RESULT: Even at maximum adapter capacity, the argmax cannot flip. "
            "Degradation impossible by construction."
        )
    else:
        logger.info(
            "RESULT: Pessimistic bound exceeds margin. "
            "Bound looseness = %.2e×. "
            "Need tighter analysis (Gram-based or measured propagation).",
            safety.logit_bound / safety.min_margin if safety.min_margin > 0 else float("inf"),
        )

    # --- Output ---
    result = {
        "model": str(model_path),
        "n_layers": n_layers,
        "sigma_max_readout": sigma_max_readout,
        "per_layer_lipschitz": {str(k): v for k, v in lipschitz.items()},
        "per_layer_propagation_factor": {str(k): v for k, v in propagation_factors.items()},
        "per_layer_sigma_k": {str(k): v for k, v in sigma_k_per_layer.items()},
        "per_layer_single_bound": {str(k): v for k, v in single_layer_bounds.items()},
        "worst_case_bound": bound_result.bound,
        "n_probes": args.n_probes,
        "min_margin": min_margin,
        "mean_margin": mean_margin,
        "per_problem_margins": margins,
        "safety_ratio": safety.safety_ratio,
        "safe": safety.safe,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.output)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
