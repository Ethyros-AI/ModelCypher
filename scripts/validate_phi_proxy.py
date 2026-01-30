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

"""Validate differentiable phi proxy against true TwoNN-based comp/phi.

This script compares:
1. True comp/phi: Computed via TwoNN on layer activations (non-differentiable)
2. Proxy comp/phi: Computed via differentiable norm trajectory

The script reports continuous correlation values and interpretation guidance.
There is no binary pass/fail - interpretation depends on your use case.

Interpretation guide (not hard thresholds):
- r > 0.8: Strong correlation - proxy tracks true comp/phi well
- r 0.5-0.8: Moderate correlation - proxy captures general trend
- r < 0.5: Weak correlation - proxy may not effectively represent comp/phi

Usage:
    python scripts/validate_phi_proxy.py --model /path/to/model

    # With custom prompts
    python scripts/validate_phi_proxy.py --model /path/to/model --prompts data/prompts.jsonl

    # Quick validation (fewer prompts)
    python scripts/validate_phi_proxy.py --model /path/to/model --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# CRT-style prompts for validation (these should show comp/phi variation)
DEFAULT_VALIDATION_PROMPTS = [
    # Cognitive Reflection Test problems (should show deliberate processing)
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
    # Simple factual (should show fast intuitive processing)
    "What is the capital of France?",
    "What is 2 + 2?",
    "How many days are in a week?",
    # Math requiring reasoning
    "What is 15 times 17?",
    "A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance?",
    "If Tom has 3 times as many apples as Jane, and Jane has 5 apples, how many does Tom have?",
    # Logic problems
    "All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
    "Some fruits are red. Apples are fruits. Are all apples red?",
    "If it rains, the ground gets wet. It rained today. Is the ground wet?",
    # Multi-step reasoning
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "How many times can you subtract 5 from 25?",
    # Simple text completion
    "The quick brown fox jumps over the",
    "Once upon a time, there was a",
    # Technical questions
    "What gas do plants produce during photosynthesis?",
    "How many chambers does a human heart have?",
    "What color do you get when you mix red and blue paint?",
    # Longer reasoning
    "If I have 3 red balls and 2 blue balls in a bag, what is the probability of drawing a red ball?",
]


@dataclass
class PhiComparison:
    """Single comparison between true and proxy comp/phi."""
    prompt: str
    true_comp_phi: float
    proxy_comp_phi: float
    true_peak_layer: int
    proxy_peak_layer: float
    n_layers: int


def compute_true_comp_phi(
    model: Any,
    tokenizer: Any,
    prompt: str,
    backend: Any,
) -> tuple[float, int, int]:
    """Compute true comp/phi using TwoNN on layer activations.

    This is the gold standard measurement but is non-differentiable.

    Returns:
        (comp_phi, peak_layer, n_layers)
    """
    import mlx.core as mx

    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    PHI = 1.618033988749895

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Collect activations at each layer
    base_model = getattr(model, "model", model)
    hidden = base_model.embed_tokens(input_ids)
    mx.eval(hidden)

    layer_activations = []
    # Flatten embedding activations: [batch, seq, hidden] -> [batch*seq, hidden]
    flat = hidden.reshape(-1, hidden.shape[-1])
    layer_activations.append(flat)

    for layer in base_model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        layer_activations.append(flat)

    n_layers = len(layer_activations) - 1  # Subtract embedding

    # Compute intrinsic dimension at each layer
    estimator = IntrinsicDimension(backend)
    layer_ids = []

    for i, acts in enumerate(layer_activations):
        try:
            # Need at least 4 samples for TwoNN
            if acts.shape[0] >= 4:
                estimate = estimator.compute(acts, with_ci=False)
                layer_ids.append(estimate.intrinsic_dimension)
            else:
                # Use norm as fallback for short sequences
                norm = float(mx.sqrt(mx.sum(acts * acts)))
                layer_ids.append(norm)
        except Exception:
            # Fallback to norm
            norm = float(mx.sqrt(mx.sum(acts * acts)))
            layer_ids.append(norm)

    if not layer_ids or len(layer_ids) < 3:
        return 0.0, 0, n_layers

    # Find peak
    peak_layer = max(range(len(layer_ids)), key=lambda i: layer_ids[i])
    peak_val = layer_ids[peak_layer]
    initial_val = layer_ids[0]
    final_val = layer_ids[-1]

    # Compute comp/phi
    eps = 1e-8
    expansion_layers = max(peak_layer, 1)
    expansion_rate = (peak_val - initial_val) / expansion_layers

    compression_layers = max(len(layer_ids) - peak_layer - 1, 1)
    compression_rate = (peak_val - final_val) / compression_layers

    denom = max(expansion_rate * PHI, eps)
    comp_phi = compression_rate / denom

    return comp_phi, peak_layer, n_layers


def compute_proxy_comp_phi(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> tuple[float, float, int]:
    """Compute proxy comp/phi using differentiable norm trajectory.

    Returns:
        (comp_phi, peak_layer, n_layers)
    """
    import mlx.core as mx

    from modelcypher.core.domain.geometry.differentiable_phi import (
        compute_trajectory_norms,
        compute_phi_metrics,
    )

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    trajectory = compute_trajectory_norms(model, input_ids)
    mx.eval(trajectory)

    metrics = compute_phi_metrics(trajectory)

    return metrics["comp_phi"], metrics["peak_layer"], metrics["n_layers"]


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    if var_x <= 0 or var_y <= 0:
        return 0.0

    return cov_xy / (var_x ** 0.5 * var_y ** 0.5)


def validate_phi_proxy(
    model_path: str,
    prompts: list[str] | None = None,
    quick: bool = False,
) -> dict:
    """Run validation comparing proxy to true comp/phi.

    Args:
        model_path: Path to the model to validate.
        prompts: Optional list of prompts. Uses defaults if not provided.
        quick: If True, use only first 5 prompts for quick validation.

    Returns:
        Dict with correlation, individual comparisons, and pass/fail status.
    """
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.core.domain._backend import get_default_backend

    logger.info("=" * 70)
    logger.info("PHI PROXY VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load(model_path)
    backend = get_default_backend()

    # Select prompts
    validation_prompts = prompts or DEFAULT_VALIDATION_PROMPTS
    if quick:
        validation_prompts = validation_prompts[:5]

    logger.info(f"Validating with {len(validation_prompts)} prompts...")

    # Run comparisons
    comparisons: list[PhiComparison] = []
    true_values: list[float] = []
    proxy_values: list[float] = []

    for i, prompt in enumerate(validation_prompts):
        try:
            # Compute true comp/phi
            true_phi, true_peak, n_layers = compute_true_comp_phi(
                model, tokenizer, prompt, backend
            )

            # Compute proxy comp/phi
            proxy_phi, proxy_peak, _ = compute_proxy_comp_phi(
                model, tokenizer, prompt
            )

            comparison = PhiComparison(
                prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                true_comp_phi=true_phi,
                proxy_comp_phi=proxy_phi,
                true_peak_layer=true_peak,
                proxy_peak_layer=proxy_peak,
                n_layers=n_layers,
            )
            comparisons.append(comparison)
            true_values.append(true_phi)
            proxy_values.append(proxy_phi)

            logger.info(
                f"[{i+1}/{len(validation_prompts)}] "
                f"true={true_phi:.3f}, proxy={proxy_phi:.3f}, "
                f"peak: {true_peak} vs {proxy_peak:.1f}"
            )

        except Exception as e:
            logger.warning(f"Failed on prompt {i+1}: {e}")
            continue

    if len(comparisons) < 3:
        logger.error("Too few successful comparisons for correlation analysis")
        return {
            "status": "FAILED",
            "error": "Insufficient successful comparisons",
            "n_comparisons": len(comparisons),
        }

    # Compute correlation
    correlation = compute_pearson_correlation(true_values, proxy_values)

    # Compute additional statistics
    true_mean = sum(true_values) / len(true_values)
    proxy_mean = sum(proxy_values) / len(proxy_values)
    abs_errors = [abs(t - p) for t, p in zip(true_values, proxy_values)]
    mean_abs_error = sum(abs_errors) / len(abs_errors)

    # Compute variance for additional context
    true_var = sum((t - true_mean) ** 2 for t in true_values) / len(true_values)
    proxy_var = sum((p - proxy_mean) ** 2 for p in proxy_values) / len(proxy_values)

    # Interpretation guidance (not hard thresholds)
    if correlation >= 0.8:
        interpretation = "strong"
        guidance = "Proxy tracks true comp/phi well"
    elif correlation >= 0.5:
        interpretation = "moderate"
        guidance = "Proxy captures general trend; may miss nuances"
    elif correlation >= 0.0:
        interpretation = "weak"
        guidance = "Proxy may not effectively represent comp/phi"
    else:
        interpretation = "inverse"
        guidance = "Proxy is negatively correlated - investigate"

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS (Continuous - No Pass/Fail)")
    logger.info("=" * 70)
    logger.info(f"Pearson correlation: {correlation:.4f}")
    logger.info(f"Interpretation: {interpretation} ({guidance})")
    logger.info(f"True comp/phi: mean={true_mean:.4f}, var={true_var:.4f}")
    logger.info(f"Proxy comp/phi: mean={proxy_mean:.4f}, var={proxy_var:.4f}")
    logger.info(f"Mean absolute error: {mean_abs_error:.4f}")

    logger.info("\nInterpretation guide (not thresholds):")
    logger.info("  r > 0.8: Strong - proxy tracks true comp/phi well")
    logger.info("  r 0.5-0.8: Moderate - proxy captures general trend")
    logger.info("  r < 0.5: Weak - proxy may not effectively represent comp/phi")

    if correlation < 0.5:
        logger.info(
            "\nSuggestions for improving proxy correlation:"
            "\n- Use more diverse prompts"
            "\n- Adjust soft_argmax power parameter"
            "\n- Check if model architecture is compatible"
        )

    return {
        "correlation": correlation,
        "interpretation": interpretation,
        "guidance": guidance,
        "n_comparisons": len(comparisons),
        "true_comp_phi_mean": true_mean,
        "true_comp_phi_var": true_var,
        "proxy_comp_phi_mean": proxy_mean,
        "proxy_comp_phi_var": proxy_var,
        "mean_absolute_error": mean_abs_error,
        "comparisons": [
            {
                "prompt": c.prompt,
                "true_comp_phi": c.true_comp_phi,
                "proxy_comp_phi": c.proxy_comp_phi,
                "true_peak_layer": c.true_peak_layer,
                "proxy_peak_layer": c.proxy_peak_layer,
                "n_layers": c.n_layers,
            }
            for c in comparisons
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate differentiable phi proxy against true TwoNN-based comp/phi"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to validate",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="Path to JSONL file with prompts (one per line, 'text' or 'prompt' field)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation with fewer prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    # Load custom prompts if provided
    prompts = None
    if args.prompts:
        prompts_path = Path(args.prompts)
        if prompts_path.suffix == ".jsonl":
            prompts = []
            for line in prompts_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("text") or record.get("prompt")
                if prompt:
                    prompts.append(prompt)
        else:
            prompts = [
                line.strip()
                for line in prompts_path.read_text().splitlines()
                if line.strip()
            ]

    # Run validation
    result = validate_phi_proxy(
        model_path=args.model,
        prompts=prompts,
        quick=args.quick,
    )

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    # Always exit successfully - no pass/fail binary
    # User should interpret the continuous correlation value
    sys.exit(0)


if __name__ == "__main__":
    main()
