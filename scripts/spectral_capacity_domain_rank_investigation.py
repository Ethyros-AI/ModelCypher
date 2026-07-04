#!/usr/bin/env python3
"""Spectral Capacity vs Domain Rank Signatures.

Investigates whether the weight matrix singular value spectrum at layers 7,8
shows structure at the activation-derived domain rank values (126, 211, 255)
found by positive geometry analysis.

Domain rank signatures (from positive_geometry_scale_comparison.md):
  - linguistic/mental: rank=126 (layers 7,8, all scales)
  - computational/structural: rank=211 (layers 7,8, all scales)
  - factual: rank=255 (layer 8, all scales; layer 7 at 700M+)

LFM2 architecture at these layers:
  - Layer 7 = ShortConv (conv.in_proj, conv.out_proj, feed_forward.w1/w2/w3)
  - Layer 8 = Attention (q_proj, k_proj, v_proj, out_proj, feed_forward.w1/w2/w3)
  - Attention indices: [2, 5, 8, 10, 12, 14]

Usage:
    poetry run python scripts/spectral_capacity_domain_rank_investigation.py
    poetry run python scripts/spectral_capacity_domain_rank_investigation.py --output data/experiments
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — from positive_geometry_scale_comparison.md
# ---------------------------------------------------------------------------

DOMAIN_RANKS = {
    126: "linguistic/mental",
    211: "computational/structural",
    255: "factual",
}

MODELS = {
    "350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

# LFM2 attention layer indices (all sizes)
ATTENTION_INDICES = {2, 5, 8, 10, 12, 14}

TARGET_LAYERS = [7, 8]


# ---------------------------------------------------------------------------
# Domain-rank analysis (pure Python on singular value lists)
# ---------------------------------------------------------------------------


def gap_at_rank(sv: list[float], k: int) -> float | None:
    """Spectral gap ratio sigma[k-1] / sigma[k] at rank boundary k.

    Returns None if k is out of bounds.
    Returns inf if sigma[k] is zero (or near-zero).
    """
    if k <= 0 or k >= len(sv):
        return None
    if sv[k] <= 1e-30:
        return float("inf")
    return sv[k - 1] / sv[k]


def energy_at_rank(sv: list[float], k: int) -> float | None:
    """Fraction of total spectral energy captured by top k singular values.

    Energy = sum(sigma_i^2).
    """
    if k <= 0 or k > len(sv):
        return None
    total = sum(s * s for s in sv)
    if total <= 0.0:
        return 0.0
    top_k = sum(s * s for s in sv[:k])
    return top_k / total


def sigma_at_rank(sv: list[float], k: int) -> float | None:
    """Singular value at position k (0-indexed: sv[k-1])."""
    if k <= 0 or k > len(sv):
        return None
    return sv[k - 1]


def analyze_domain_ranks(sv: list[float]) -> dict[str, dict]:
    """Compute gap, energy, and sigma at each domain rank position."""
    result = {}
    for rank, domain in DOMAIN_RANKS.items():
        entry: dict[str, object] = {"domain": domain}
        g = gap_at_rank(sv, rank)
        e = energy_at_rank(sv, rank)
        s = sigma_at_rank(sv, rank)
        entry["spectralGapRatio"] = g
        entry["energyFractionAtRank"] = e
        entry["sigmaAtRank"] = s
        if rank < len(sv):
            entry["sigmaAfterRank"] = sv[rank]
        else:
            entry["sigmaAfterRank"] = None
        entry["rankWithinSpectrum"] = rank <= len(sv)
        result[str(rank)] = entry
    return result


# ---------------------------------------------------------------------------
# Phase 1: Collect capacity data
# ---------------------------------------------------------------------------


@dataclass
class LayerAnalysis:
    """Capacity + domain-rank analysis for one weight matrix."""

    layer_name: str
    weight_shape: tuple[int, int]
    layer_index: int
    layer_type: str  # "attention" or "shortconv"
    weight_type: str  # e.g. "q_proj", "feed_forward.w1", "conv.in_proj"

    # From capacity analysis
    effective_rank: float
    stable_rank: float
    shannon_effective_rank: float | None
    numerical_rank_f32: int
    null_space_fraction: float
    recommended_rank: int
    spectral_gap_at_recommended: float
    capacity_utilization: float
    decay_type: str | None
    computation_method: str

    # Domain rank analysis
    domain_rank_analysis: dict[str, dict]

    # Full spectrum (for cross-reference / debugging)
    spectrum_length: int

    def to_dict(self) -> dict:
        return {
            "layerName": self.layer_name,
            "weightShape": list(self.weight_shape),
            "layerIndex": self.layer_index,
            "layerType": self.layer_type,
            "weightType": self.weight_type,
            "effectiveRank": self.effective_rank,
            "stableRank": self.stable_rank,
            "shannonEffectiveRank": self.shannon_effective_rank,
            "numericalRankF32": self.numerical_rank_f32,
            "nullSpaceFraction": self.null_space_fraction,
            "recommendedRank": self.recommended_rank,
            "spectralGapAtRecommendedRank": self.spectral_gap_at_recommended,
            "capacityUtilization": self.capacity_utilization,
            "decayType": self.decay_type,
            "computationMethod": self.computation_method,
            "domainRankAnalysis": self.domain_rank_analysis,
            "spectrumLength": self.spectrum_length,
        }


def classify_weight(layer_name: str, layer_idx: int) -> tuple[str, str]:
    """Classify a weight matrix by layer type and weight type."""
    layer_type = "attention" if layer_idx in ATTENTION_INDICES else "shortconv"

    # Extract weight type from full name
    # e.g. "model.layers.7.feed_forward.w1.weight" -> "feed_forward.w1"
    parts = layer_name.split(".")
    # Find the part after "layers.N."
    try:
        layers_idx = parts.index("layers")
        after = parts[layers_idx + 2 :]  # skip "layers" and the index
        # Remove trailing "weight" if present
        if after and after[-1] == "weight":
            after = after[:-1]
        weight_type = ".".join(after)
    except (ValueError, IndexError):
        weight_type = layer_name

    return layer_type, weight_type


def run_capacity_analysis(model_path: str, model_tag: str) -> dict | None:
    """Run Phase 1+2: capacity analysis + domain rank analysis for one model."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s (volume not mounted?)", model_path)
        return None

    logger.info("Analyzing %s at %s", model_tag, model_path)

    # Import ModelCypher service
    try:
        from modelcypher.cli.composition import get_capacity_analysis_service
    except ImportError:
        logger.error("ModelCypher not installed. Run: poetry install")
        return None

    service = get_capacity_analysis_service()

    # Build target module filters for layers 7 and 8
    target_modules = [f"layers.{i}" for i in TARGET_LAYERS]

    report = service.analyze(model_path, target_modules=target_modules)

    logger.info(
        "  %d layers analyzed, %d failed",
        report.analyzed_layers,
        len(report.failed_layers),
    )

    # Phase 2: Analyze each layer's spectrum at domain rank positions
    layer_analyses: list[LayerAnalysis] = []

    for lr in report.layer_reports:
        # Determine layer index from name
        layer_idx = _extract_layer_index(lr.layer_name)
        if layer_idx is None:
            continue

        layer_type, weight_type = classify_weight(lr.layer_name, layer_idx)

        # Domain rank analysis on the full spectrum
        dra = analyze_domain_ranks(lr.singular_values)

        analysis = LayerAnalysis(
            layer_name=lr.layer_name,
            weight_shape=lr.weight_shape,
            layer_index=layer_idx,
            layer_type=layer_type,
            weight_type=weight_type,
            effective_rank=lr.effective_rank,
            stable_rank=lr.stable_rank,
            shannon_effective_rank=lr.shannon_effective_rank,
            numerical_rank_f32=lr.numerical_rank_f32,
            null_space_fraction=lr.null_space_fraction,
            recommended_rank=lr.recommended_rank,
            spectral_gap_at_recommended=lr.spectral_gap_at_rank,
            capacity_utilization=lr.capacity_utilization,
            decay_type=lr.decay_type.value if lr.decay_type else None,
            computation_method=lr.computation_method,
            domain_rank_analysis=dra,
            spectrum_length=len(lr.singular_values),
        )
        layer_analyses.append(analysis)

    # Group by layer index
    by_layer: dict[int, list[dict]] = {}
    for la in layer_analyses:
        by_layer.setdefault(la.layer_index, []).append(la.to_dict())

    result = {
        "model": model_tag,
        "modelPath": model_path,
        "modelName": Path(model_path).name,
        "summary": {
            "analyzedLayers": report.analyzed_layers,
            "meanEffectiveRank": report.mean_effective_rank,
            "meanCapacityUtilization": report.mean_capacity_utilization,
            "failedLayers": dict(report.failed_layers),
        },
        "domainRankConstants": {str(k): v for k, v in DOMAIN_RANKS.items()},
        "layers": {str(k): v for k, v in by_layer.items()},
    }

    return result


def _extract_layer_index(layer_name: str) -> int | None:
    """Extract layer index from weight name like 'model.layers.7.feed_forward.w1.weight'."""
    parts = layer_name.split(".")
    try:
        idx = parts.index("layers")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Phase 3: Cross-scale comparison
# ---------------------------------------------------------------------------


def build_crossscale(per_model: dict[str, dict]) -> dict:
    """Build cross-scale comparison from per-model results."""
    crossscale: dict = {
        "domainRanks": list(DOMAIN_RANKS.keys()),
        "domainLabels": {str(k): v for k, v in DOMAIN_RANKS.items()},
        "models": list(per_model.keys()),
        "perDomainRank": {},
        "effectiveRankScaling": {},
        "nullSpaceScaling": {},
        "recommendedRankComparison": {},
    }

    # For each domain rank, compare gap/energy across models and layers
    for rank in DOMAIN_RANKS:
        rank_key = str(rank)
        rank_data: dict = {}
        for layer_idx in TARGET_LAYERS:
            layer_key = str(layer_idx)
            layer_data: dict = {}
            for model_tag, model_result in per_model.items():
                layers = model_result.get("layers", {}).get(layer_key, [])
                # Collect domain rank analysis from all weight matrices at this layer
                weight_gaps = []
                for weight_entry in layers:
                    dra = weight_entry.get("domainRankAnalysis", {}).get(rank_key, {})
                    if dra.get("rankWithinSpectrum"):
                        weight_gaps.append(
                            {
                                "weightType": weight_entry["weightType"],
                                "weightShape": weight_entry["weightShape"],
                                "spectralGapRatio": dra.get("spectralGapRatio"),
                                "energyFractionAtRank": dra.get(
                                    "energyFractionAtRank"
                                ),
                                "sigmaAtRank": dra.get("sigmaAtRank"),
                            }
                        )
                layer_data[model_tag] = weight_gaps
            rank_data[layer_key] = layer_data
        crossscale["perDomainRank"][rank_key] = rank_data

    # Effective rank scaling
    for layer_idx in TARGET_LAYERS:
        layer_key = str(layer_idx)
        scaling: dict = {}
        for model_tag, model_result in per_model.items():
            layers = model_result.get("layers", {}).get(layer_key, [])
            eff_ranks = {
                w["weightType"]: w["effectiveRank"] for w in layers
            }
            scaling[model_tag] = eff_ranks
        crossscale["effectiveRankScaling"][layer_key] = scaling

    # Null space scaling
    for layer_idx in TARGET_LAYERS:
        layer_key = str(layer_idx)
        scaling = {}
        for model_tag, model_result in per_model.items():
            layers = model_result.get("layers", {}).get(layer_key, [])
            null_fracs = {
                w["weightType"]: w["nullSpaceFraction"] for w in layers
            }
            scaling[model_tag] = null_fracs
        crossscale["nullSpaceScaling"][layer_key] = scaling

    # Recommended rank comparison
    for layer_idx in TARGET_LAYERS:
        layer_key = str(layer_idx)
        rr: dict = {}
        for model_tag, model_result in per_model.items():
            layers = model_result.get("layers", {}).get(layer_key, [])
            rec_ranks = {
                w["weightType"]: {
                    "recommendedRank": w["recommendedRank"],
                    "spectralGapAtRecommendedRank": w[
                        "spectralGapAtRecommendedRank"
                    ],
                }
                for w in layers
            }
            rr[model_tag] = rec_ranks
        crossscale["recommendedRankComparison"][layer_key] = rr

    return crossscale


# ---------------------------------------------------------------------------
# Phase 4: Console output
# ---------------------------------------------------------------------------


def print_summary(per_model: dict[str, dict], crossscale: dict) -> None:
    """Print human-readable summary table."""
    print()
    print("=" * 80)
    print("SPECTRAL CAPACITY vs DOMAIN RANK SIGNATURES")
    print("=" * 80)

    for model_tag, model_result in per_model.items():
        print(f"\n{'─' * 80}")
        print(f"Model: {model_tag} ({model_result['modelName']})")
        summary = model_result["summary"]
        print(
            f"  Mean effective rank: {summary['meanEffectiveRank']:.1f}"
            f"  |  Mean capacity utilization: {summary['meanCapacityUtilization']:.3f}"
        )

        for layer_idx in TARGET_LAYERS:
            layer_key = str(layer_idx)
            layers = model_result.get("layers", {}).get(layer_key, [])
            if not layers:
                continue

            ltype = layers[0]["layerType"]
            print(f"\n  Layer {layer_idx} ({ltype}):")
            print(
                f"    {'Weight':<28} {'Shape':>12} {'Eff.Rank':>9} "
                f"{'Rec.Rank':>9} {'Null%':>7} {'Decay':>14}"
            )
            print(f"    {'─' * 28} {'─' * 12} {'─' * 9} {'─' * 9} {'─' * 7} {'─' * 14}")

            for w in layers:
                shape_str = f"{w['weightShape'][0]}x{w['weightShape'][1]}"
                null_pct = w["nullSpaceFraction"] * 100
                decay = w["decayType"] or "?"
                print(
                    f"    {w['weightType']:<28} {shape_str:>12} "
                    f"{w['effectiveRank']:>9.1f} {w['recommendedRank']:>9d} "
                    f"{null_pct:>6.1f}% {decay:>14}"
                )

            # Domain rank gaps for this layer
            print("\n    Domain rank gaps:")
            print(
                f"    {'Rank':<6} {'Domain':<28} | ", end=""
            )
            for w in layers:
                print(f"{w['weightType'][:16]:>16} ", end="")
            print()
            print(f"    {'─' * 6} {'─' * 28}─┤─", end="")
            for _ in layers:
                print(f"{'─' * 16}─", end="")
            print()

            for rank in DOMAIN_RANKS:
                rank_key = str(rank)
                domain = DOMAIN_RANKS[rank]
                print(f"    {rank:<6} {domain:<28} | ", end="")
                for w in layers:
                    dra = w.get("domainRankAnalysis", {}).get(rank_key, {})
                    gap = dra.get("spectralGapRatio")
                    energy = dra.get("energyFractionAtRank")
                    if gap is not None and energy is not None:
                        if gap == float("inf"):
                            print(f"{'inf':>7} {energy:>7.3f} ", end="")
                        else:
                            print(f"{gap:>7.3f} {energy:>7.3f} ", end="")
                    else:
                        print(f"{'N/A':>7} {'N/A':>7} ", end="")
                print()
            print("    (format: gap_ratio energy_fraction)")

    # Cross-scale summary
    print(f"\n{'=' * 80}")
    print("CROSS-SCALE COMPARISON")
    print(f"{'=' * 80}")

    for rank in DOMAIN_RANKS:
        rank_key = str(rank)
        domain = DOMAIN_RANKS[rank]
        print(f"\n  Domain rank {rank} ({domain}):")
        rank_data = crossscale.get("perDomainRank", {}).get(rank_key, {})
        for layer_idx in TARGET_LAYERS:
            layer_key = str(layer_idx)
            layer_data = rank_data.get(layer_key, {})
            print(f"    Layer {layer_idx}:")
            # Collect all weight types across models
            all_types: set[str] = set()
            for wlist in layer_data.values():
                for w in wlist:
                    all_types.add(w["weightType"])
            for wtype in sorted(all_types):
                print(f"      {wtype}:", end="")
                for model_tag in per_model:
                    entries = layer_data.get(model_tag, [])
                    match = [e for e in entries if e["weightType"] == wtype]
                    if match:
                        g = match[0]["spectralGapRatio"]
                        e = match[0]["energyFractionAtRank"]
                        if g is not None and g != float("inf"):
                            print(f"  {model_tag} gap={g:.3f} E={e:.3f}", end="")
                        elif g == float("inf"):
                            print(f"  {model_tag} gap=inf E={e:.3f}", end="")
                        else:
                            print(f"  {model_tag} N/A", end="")
                    else:
                        print(f"  {model_tag} (no data)", end="")
                print()

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectral capacity vs domain rank investigation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiments",
        help="Output directory for JSON files (default: data/experiments)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        choices=list(MODELS.keys()),
        default=None,
        help="Which models to analyze (default: all available)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tags = args.models or list(MODELS.keys())

    # Phase 1+2: Capacity analysis + domain rank analysis per model
    per_model: dict[str, dict] = {}
    for tag in model_tags:
        path = MODELS[tag]
        result = run_capacity_analysis(path, tag)
        if result is not None:
            per_model[tag] = result

            # Save per-model JSON
            tag_lower = tag.lower().replace(".", "p")
            out_path = output_dir / f"spectral_capacity_domain_rank_{tag_lower}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("  Saved: %s", out_path)

    if not per_model:
        logger.error("No models analyzed. Is the volume mounted?")
        sys.exit(1)

    # Phase 3: Cross-scale comparison
    crossscale = build_crossscale(per_model)
    crossscale_path = output_dir / "spectral_capacity_domain_rank_crossscale.json"
    with open(crossscale_path, "w") as f:
        json.dump(crossscale, f, indent=2)
    logger.info("Saved cross-scale: %s", crossscale_path)

    # Phase 4: Console output
    print_summary(per_model, crossscale)


if __name__ == "__main__":
    main()
