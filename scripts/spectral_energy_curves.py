#!/usr/bin/env python3
"""Part A: Full energy accumulation curves from weight spectra.

Computes cumulative energy curves and inflection points for all weight
matrices at layers 7,8 across LFM2-350M/700M/1.2B. Overlays domain rank
positions (126, 211, 255) to test whether inflection points coincide
with domain boundaries.

Usage:
    poetry run python scripts/spectral_energy_curves.py
    poetry run python scripts/spectral_energy_curves.py --models 350M
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DOMAIN_RANKS = {126: "linguistic/mental", 211: "computational/structural", 255: "factual"}

MODELS = {
    "350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

ATTENTION_INDICES = {2, 5, 8, 10, 12, 14}
TARGET_LAYERS = [7, 8]


def analyze_model(model_path: str, model_tag: str) -> dict | None:
    """Run capacity analysis and compute full energy curves for one model."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return None

    logger.info("Analyzing %s", model_tag)

    from modelcypher.cli.composition import get_capacity_analysis_service
    from modelcypher.core.domain.geometry.spectral_capacity import (
        compute_full_energy_curve,
        find_energy_inflection_points,
    )

    service = get_capacity_analysis_service()
    target_modules = [f"layers.{i}" for i in TARGET_LAYERS]
    report = service.analyze(model_path, target_modules=target_modules)

    logger.info("  %d layers analyzed", report.analyzed_layers)

    layers_data: dict[str, list[dict]] = {}

    for lr in report.layer_reports:
        layer_idx = _extract_layer_index(lr.layer_name)
        if layer_idx is None:
            continue

        layer_type = "attention" if layer_idx in ATTENTION_INDICES else "shortconv"
        weight_type = _extract_weight_type(lr.layer_name)

        sv = lr.singular_values
        curve = compute_full_energy_curve(sv)
        inflections = find_energy_inflection_points(curve)

        # Domain rank overlay
        domain_overlay: dict[str, dict] = {}
        for rank, domain in DOMAIN_RANKS.items():
            if rank > len(sv):
                domain_overlay[str(rank)] = {"domain": domain, "withinSpectrum": False}
                continue

            energy = curve[rank - 1] if rank <= len(curve) else None

            # Find nearest inflection point
            nearest = None
            nearest_dist = None
            for ip in inflections:
                dist = abs(ip["rank"] - rank)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ip["rank"]

            domain_overlay[str(rank)] = {
                "domain": domain,
                "withinSpectrum": True,
                "energyAtRank": energy,
                "nearestInflectionRank": nearest,
                "distanceToInflection": nearest_dist,
            }

        entry = {
            "layerName": lr.layer_name,
            "layerIndex": layer_idx,
            "layerType": layer_type,
            "weightType": weight_type,
            "weightShape": list(lr.weight_shape),
            "spectrumLength": len(sv),
            "effectiveRank": lr.effective_rank,
            "shannonEffectiveRank": lr.shannon_effective_rank,
            "recommendedRank": lr.recommended_rank,
            "decayType": lr.decay_type.value if lr.decay_type else None,
            "singularValues": sv,
            "energyCurve": curve,
            "inflectionPoints": inflections,  # All inflection points, sorted by prominence
            "domainRankOverlay": domain_overlay,
        }

        layer_key = str(layer_idx)
        layers_data.setdefault(layer_key, []).append(entry)

    return {
        "model": model_tag,
        "modelPath": model_path,
        "modelName": Path(model_path).name,
        "layers": layers_data,
    }


def _extract_layer_index(name: str) -> int | None:
    parts = name.split(".")
    try:
        idx = parts.index("layers")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


def _extract_weight_type(name: str) -> str:
    parts = name.split(".")
    try:
        idx = parts.index("layers")
        after = parts[idx + 2 :]
        if after and after[-1] == "weight":
            after = after[:-1]
        return ".".join(after)
    except (ValueError, IndexError):
        return name


def print_summary(results: dict[str, dict]) -> None:
    """Print summary of energy curves and inflection points."""
    print()
    print("=" * 90)
    print("WEIGHT ENERGY CURVES — INFLECTION POINT ANALYSIS")
    print("=" * 90)

    for model_tag, model_data in results.items():
        print(f"\n{'─' * 90}")
        print(f"Model: {model_tag}")

        for layer_idx in TARGET_LAYERS:
            layer_key = str(layer_idx)
            layers = model_data.get("layers", {}).get(layer_key, [])
            if not layers:
                continue

            ltype = layers[0]["layerType"]
            print(f"\n  Layer {layer_idx} ({ltype}):")

            for w in layers:
                wtype = w["weightType"]
                n_inflections = len(w["inflectionPoints"])
                top3 = w["inflectionPoints"][:3]
                top3_ranks = [ip["rank"] for ip in top3]

                print(f"\n    {wtype} ({w['weightShape'][0]}x{w['weightShape'][1]})")
                print(f"      Eff.rank={w['effectiveRank']:.1f}  Shannon={w['shannonEffectiveRank'] or 0:.1f}  Rec.rank={w['recommendedRank']}  Decay={w['decayType']}")
                print(f"      Inflection points ({n_inflections} total): top 3 at ranks {top3_ranks}")

                if top3:
                    for ip in top3:
                        print(f"        rank {ip['rank']:>4}: d2E={ip['d2E']:>+.6f}  |d2E|={ip['abs_d2E']:.6f}  E={ip['energy_at_rank']:.4f}")

                # Domain rank overlay
                print("      Domain rank alignment:")
                for rank_key, overlay in w["domainRankOverlay"].items():
                    if not overlay.get("withinSpectrum"):
                        print(f"        rank {rank_key} ({overlay['domain']}): outside spectrum")
                        continue
                    energy = overlay.get("energyAtRank", 0)
                    nearest = overlay.get("nearestInflectionRank")
                    dist = overlay.get("distanceToInflection")
                    if nearest is not None:
                        print(
                            f"        rank {rank_key} ({overlay['domain']}): "
                            f"E={energy:.4f}  nearest_inflection={nearest} (dist={dist})"
                        )
                    else:
                        print(
                            f"        rank {rank_key} ({overlay['domain']}): "
                            f"E={energy:.4f}  no inflection points"
                        )

    # Cross-model summary: domain rank alignment
    print(f"\n{'=' * 90}")
    print("DOMAIN RANK — INFLECTION ALIGNMENT SUMMARY")
    print(f"{'=' * 90}")

    for rank, domain in DOMAIN_RANKS.items():
        print(f"\n  Rank {rank} ({domain}):")
        for model_tag, model_data in results.items():
            for layer_idx in TARGET_LAYERS:
                layer_key = str(layer_idx)
                layers = model_data.get("layers", {}).get(layer_key, [])
                for w in layers:
                    overlay = w["domainRankOverlay"].get(str(rank), {})
                    if not overlay.get("withinSpectrum"):
                        continue
                    dist = overlay.get("distanceToInflection")
                    nearest = overlay.get("nearestInflectionRank")
                    wtype = w["weightType"]
                    if dist is not None and dist <= 5:
                        marker = " <<<" if dist == 0 else f" <-- close (d={dist})"
                    else:
                        marker = ""
                    print(
                        f"    {model_tag:>4} L{layer_idx} {wtype:<24} "
                        f"nearest={nearest} dist={dist}{marker}"
                    )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight energy curve analysis")
    parser.add_argument("--output", type=str, default="data/experiments")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()), default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tags = args.models or list(MODELS.keys())

    results: dict[str, dict] = {}
    for tag in model_tags:
        result = analyze_model(MODELS[tag], tag)
        if result is not None:
            results[tag] = result
            tag_lower = tag.lower().replace(".", "p")
            out_path = output_dir / f"spectral_energy_curves_{tag_lower}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("  Saved: %s", out_path)

    if not results:
        logger.error("No models analyzed")
        sys.exit(1)

    print_summary(results)


if __name__ == "__main__":
    main()
