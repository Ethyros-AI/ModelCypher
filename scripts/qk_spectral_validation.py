#!/usr/bin/env python3
"""QK spectral bound validation on real models.

Measures per-head sigma_max(W_Q_h) and sigma_max(W_K_h) on a loaded model,
then computes the derived softcap-equivalent bound to determine whether logit
softcapping is geometrically active or redundant.

This is the Step 0 experiment: measure before implementing any training
constraint. If most heads are naturally below typical softcap bounds,
softcap was doing almost nothing and removal should be trivial.

Usage:
    # Measure natural spectral products (no softcap comparison)
    poetry run python scripts/qk_spectral_validation.py \
        /Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16

    # Compare against a specific softcap value
    poetry run python scripts/qk_spectral_validation.py \
        /Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16 \
        --soft-cap 50

    # Scan multiple softcap values
    poetry run python scripts/qk_spectral_validation.py \
        /Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16 \
        --soft-cap-sweep 10,20,30,50,100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QK spectral bound validation on real models.",
    )
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument(
        "--soft-cap",
        type=float,
        default=None,
        help="Softcap value c to compare against (e.g., 50 for Gemma-2)",
    )
    parser.add_argument(
        "--soft-cap-sweep",
        type=str,
        default=None,
        help="Comma-separated softcap values to sweep (e.g., 10,20,30,50,100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for results",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(1)

    # Load model
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.qk_spectral_service import QKSpectralService

    backend = initialize_default_backend()
    logger.info("Loading model from %s ...", model_path)
    model, _tokenizer = ModelLoader(backend).load_model(str(model_path))

    service = QKSpectralService(backend)

    # Determine softcap values to test
    softcap_values: list[float | None] = []
    if args.soft_cap_sweep:
        softcap_values = [float(v) for v in args.soft_cap_sweep.split(",")]
    elif args.soft_cap is not None:
        softcap_values = [args.soft_cap]
    else:
        softcap_values = [None]  # Just measure, no comparison

    all_results: list[dict] = []

    for soft_cap in softcap_values:
        logger.info("")
        if soft_cap is not None:
            logger.info("=== Softcap c = %.1f ===", soft_cap)
        else:
            logger.info("=== Natural spectral products (no softcap) ===")

        report = service.analyze_model(model, soft_cap=soft_cap)

        logger.info("Model: d_model=%d, d_k=%d, heads=%d (kv=%d)",
                     report.d_model, report.d_k, report.num_heads, report.num_kv_heads)

        if report.derived_bound is not None:
            logger.info("Derived bound B = c * sqrt(d_k) / d_model = %.6f", report.derived_bound)
            logger.info("Heads total: %d", report.heads_total)
            logger.info("Heads where softcap is active: %d / %d (%.1f%%)",
                         report.heads_softcap_active, report.heads_total,
                         100 * report.heads_softcap_active / max(report.heads_total, 1))
            logger.info("Mean utilization: %.4f", report.mean_utilization)
            logger.info("Max utilization: %.4f (most constrained head)", report.max_utilization)

        logger.info("Equivalent softcap (to deactivate all heads): %.2f", report.equivalent_softcap)

        # Per-layer summary
        logger.info("")
        logger.info("Per-layer max logit magnitudes:")
        layer_max: dict[int, float] = {}
        for h in report.per_head:
            layer_max[h.layer_idx] = max(layer_max.get(h.layer_idx, 0.0), h.max_logit)
        for layer_idx in sorted(layer_max):
            logger.info("  Layer %2d: max_logit = %.2f", layer_idx, layer_max[layer_idx])

        # Top 10 most constrained heads
        if soft_cap is not None:
            sorted_heads = sorted(report.per_head, key=lambda h: h.utilization, reverse=True)
            logger.info("")
            logger.info("Top 10 most constrained heads:")
            for h in sorted_heads[:10]:
                logger.info(
                    "  L%02d H%02d: sigma_q=%.4f sigma_k=%.4f product=%.6f "
                    "util=%.4f max_logit=%.2f %s",
                    h.layer_idx, h.head_idx, h.sigma_q, h.sigma_k,
                    h.spectral_product, h.utilization, h.max_logit,
                    "ACTIVE" if h.softcap_active else "",
                )

        result_dict = report.to_dict()
        result_dict["soft_cap_tested"] = soft_cap
        all_results.append(result_dict)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\nResults written to %s", output_path)


if __name__ == "__main__":
    main()
